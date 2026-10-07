"""Optional independent lyric text; provider timestamps never become onsets."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlencode

import httpx
import requests
from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

from .anchors_from_asr import text_units
from .job_schema import provider_error, retry_after_seconds, retry_fields
from .song_sources import (SongText, extract_song_metadata, find_verified_lyrics,
                           transcript_lines, video_song_info)
from .source_captions import read_document


class LyricRequestError(httpx.HTTPError):
    def __init__(self, report):
        super().__init__(report.get('error', 'Lyric provider is unavailable'))
        self.report = report


def lyric_request_error(error, minimum_delay=30):
    if isinstance(error, LyricRequestError):
        return error.report
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    transient = isinstance(error, (httpx.TransportError, requests.ConnectionError, requests.Timeout)) or status == 429 or (status or 0) >= 500
    return dict(status='unavailable', error=type(error).__name__, http_status=status,
                retry_at=time.time()+max(minimum_delay, retry_after_seconds(error) or 0) if transient else None)


def lyric_retry_due(report, provider, cache):
    if report.get('retry_at') is not None and report['retry_at'] <= time.time():
        return True
    if report.get('http_status') == 401:
        try:
            hold = json.loads((cache.root.parent/'lyric-provider-backoff'/f'{provider}.json').read_text())
        except (OSError, ValueError):
            return True
        return not hold.get('requires_access')
    return False


def cached_lyric_request(provider, identity, cache, request, minimum_delay=30):
    """Cache provider responses independently of duration and lyric selection."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = cache.root/'lyric-responses'/f'{provider}-{key}.json'
    stage = f'lyric_response_{provider}_{key}'
    try:
        previous = json.loads(path.read_text()).get('error') or {}
        if lyric_retry_due(previous, provider, cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__).with_name('source_captions.py')], outputs=[path],
                     config=dict(provider=provider, request=identity, schema_version=1)) as reused:
        if not reused:
            hold = cache.root.parent/'lyric-provider-backoff'/f'{provider}.json'
            try:
                previous = json.loads(hold.read_text())
            except (OSError, ValueError, TypeError):
                previous = {}
            if previous.get('requires_access') or (previous.get('retry_at') or 0) > time.time():
                # A provider-wide hold carries no identity or text from its song.
                raise LyricRequestError({k: previous[k] for k in
                    ('status', 'error', 'http_status', 'retry_at', 'requires_access') if k in previous})
            try:
                packet = dict(data=request())
            except Exception as error:
                report = lyric_request_error(error, minimum_delay)
                packet = dict(error=report)
                if report['http_status'] in (401, 429):
                    import fcntl
                    hold.parent.mkdir(parents=True, exist_ok=True)
                    with hold.with_suffix('.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        try:
                            previous = json.loads(hold.read_text())
                        except (OSError, ValueError):
                            previous = {}
                        pending = dict(report, requires_access=bool(previous.get('requires_access')) or report['http_status'] == 401,
                            retry_at=max(previous.get('retry_at') or 0, report.get('retry_at') or time.time()+30))
                        temporary = hold.with_suffix(f'.{os.getpid()}.{time.time_ns()}.tmp')
                        temporary.write_text(json.dumps(pending)+'\n')
                        temporary.replace(hold)
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    packet = json.loads(path.read_text())
    if 'error' in packet:
        raise LyricRequestError(packet['error'])
    return packet['data']


def names(value, *, title=False):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').strip()
    result = [value] if value else []
    if title:
        clean = re.sub(r'\s*[\[(](?:official\s+(?:music\s+video|video|mv|audio|visualizer)|'
                       r'(?:official\s+)?lyrics?\s+video)(?:\s+(?:hd|4k))?[\])]\s*$', '', value, flags=re.I)
        if clean and clean != value:
            return result + names(clean, title=True)
    # Music metadata often appends an English translation to a Chinese name.
    # Version labels are not translations: never strip "Live", "Remix", etc.
    native = r'[\u3040-\u30ff\u3400-\u9fff]+(?:[ ·][\u3040-\u30ff\u3400-\u9fff]+)*'
    latin = r'[A-Za-z][\x20-\x7e]*?'
    match = re.fullmatch(rf'({native})(?:\s+(?:[-–—]\s+)?({latin})|\s*\(({latin})\))', value)
    reverse = re.fullmatch(rf'({latin})\s+({native})', value)
    if match or reverse:
        local, english = (match[1], match[2] or match[3]) if match else (reverse[2], reverse[1])
        if not re.search(r'\b(live|remix|acoustic|instrumental|karaoke|remaster\w*)\b', english, re.I):
            result.extend((english, local))
    return result


def identity(value, *, title=False):
    converter = OpenCC('tw2s')
    aliases = [converter.convert(name).casefold() for name in names(value, title=title)]
    if not title:
        if aliases and re.search(r'[&,/×+]|\b(?:feat\.?|featuring|ft\.?|with)\s', aliases[0]):
            aliases = aliases[:1]  # A collaborator is not an alias for the whole credit.
        joined = re.fullmatch(r"([a-z][a-z.'-]*)([\u3040-\u30ff\u3400-\u9fff]{2,})(?:\s+[a-z][a-z .'-]*)?",
                              aliases[0] if aliases else '')
        if joined and not version_tags(joined[1]):
            aliases.append(joined[2])
    if title:
        for name in list(aliases):
            # A soundtrack credit describes its use, not another performance.
            # Only strip this explicit grammar; keep live/remix/version labels.
            match = re.fullmatch(r'(.+?)\s*\((?:大型)?(?:纪录片|电影|电视剧|动画片|网络剧|网剧|剧集)?'
                                 r'\s*《[^《》()]+》\s*(?:主题[歌曲]|片[头尾]曲|插曲)\)', name)
            if match:
                aliases.append(match[1])
    return {''.join(c for c in name if c.isalnum()) for name in aliases} - {''}


def live_performance(value):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').casefold()
    match = re.search(r'(?:^|[\(\[]|\s[-–—]\s)\s*live\s+(?:from|at|in)\s+([^()\[\]|\n]+)', value)
    if match:
        venue = re.sub(r'^the\s+', '', match[1].strip())
        return ''.join(c for c in venue if c.isalnum()) or None
    return None


def fetch_lyrics(song, metadata_path, cache):
    titles, artists = names(song.get('title'), title=True), names(song.get('artist'))
    if not titles or not artists or not metadata_path.is_file():
        return dict(status='missing_identity')
    queries = [dict(track_name=titles[0], artist_name=artists[0])]
    native = dict(track_name=titles[-1], artist_name=artists[-1])
    if native != queries[0]:
        queries.append(native)
    simplified = {key: OpenCC('tw2s').convert(value) for key, value in native.items()}
    if simplified not in queries:
        queries.append(simplified)
    path, stage = cache.root/'lyric-lookup.json', 'fetch_lyric_text'
    try:
        old = json.loads(path.read_text())
        if lyric_retry_due(old, 'lrclib', cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[metadata_path], outputs=[path], config=queries,
                     code=[Path(__file__), Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            packet = dict(status='ok', queries=queries, results=[])
            try:
                for query in queries:
                    if query == simplified and simplified != native and packet['results']:
                        break
                    url = 'https://lrclib.net/api/search?'+urlencode(query)
                    response = cached_lyric_request('lrclib', url, cache, lambda: read_document(url), 300)
                    if not isinstance(response, list):
                        raise ValueError('Invalid lyric search response')
                    packet['results'].extend({key: row.get(key) for key in (
                        'id', 'trackName', 'artistName', 'albumName', 'duration', 'instrumental', 'plainLyrics')}
                        for row in response if isinstance(row, dict))
            except (httpx.HTTPError, ValueError) as error:
                packet.update(lyric_request_error(error, 300))
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())


def unpadded_duration(audio_path, duration):
    """Measure exact-zero padding in the original PCM, without trimming audio."""
    import numpy as np
    import soundfile as sf
    from .stage_cache import digest
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return None
    first, last, offset = None, 0, 0
    with sf.SoundFile(audio_path) as audio:
        rate, frames = audio.samplerate, audio.frames
        if abs(frames/rate-duration) > 1.:
            return None
        for block in audio.blocks(blocksize=rate*5, dtype='float64', always_2d=True):
            if not np.isfinite(block).all():
                return None
            nonzero = np.flatnonzero(np.any(block != 0, axis=1))
            if nonzero.size:
                if first is None:
                    first = offset+int(nonzero[0])
                last = offset+int(nonzero[-1])+1
            offset += len(block)
    if first is None:
        return None
    return dict(kind='nonzero_pcm', audio_sha256=digest(audio_path),
                recording_duration=frames/rate, nonzero_duration=(last-first)/rate,
                leading_zero_seconds=first/rate, trailing_zero_seconds=(frames-last)/rate)


def text_language(text):
    if re.search(r'[\u3040-\u30ff]', text):
        return 'Japanese'
    if re.search(r'[\u3400-\u9fff]', text):
        return 'Chinese'
    if any(c.isalpha() for c in text) and all(not c.isalpha() or 'LATIN' in unicodedata.name(c, '') for c in text):
        return 'English'
    return None


def published_lyric_lines(text, title, artists=()):
    """Remove explicit credit headers/footers before wrapping published lyrics."""
    converter = OpenCC('t2s')
    credit = re.compile(r'^(?:歌名|歌曲|词|曲|词曲|作词|作曲|编曲|演唱|专辑|发行日|制作人|监制|'
        r'弦乐(?:编写|编曲)?|合声|和声(?:编写|编曲)?|钢琴|吉他|贝斯|鼓|'
        r'(?:第[一二三四\d]+)?(?:小提琴|中提琴|大提琴)|录音(?:师|助理)?|混音(?:师)?|母带(?:工程师)?|音频(?:助理|工程师)|'
        r'title|artist|album|composer|lyricist|producer|lyrics|music|written by|produced by|'
        r'guitar|bass|drums|piano|strings|recorded by|mixed by|mastered by|audio engineer|assistant engineer)\s*[：:]', re.I)
    rows, metadata, timestamps = [], [], {}
    for i, line in enumerate(text.splitlines(), 1):
        line = html.unescape(line).strip()
        if re.fullmatch(r'\[(?:ar|ti|al|by|offset|re|ve):[^\]]*\]', line, re.I):
            metadata.append(i)
            continue
        # Provider timestamps describe text order/repetitions, never our onsets.
        # Defer repeated LRC rows until they can be ordered with the other rows.
        prefix = re.match(r'^(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]\s*)+', line)
        if prefix:
            stamps = re.findall(r'\[(\d{1,3}):(\d{2}(?:[.:]\d{1,3})?)\]', prefix[0])
            timestamps[i] = [60*int(m)+float(s.replace(':', '.')) for m,s in stamps]
            line = line[prefix.end():]
        if re.fullmatch(r'\[(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|instrumental|interlude|refrain)(?:\s+\d+)?\]', line, re.I):
            metadata.append(i)
            continue
        if line:
            rows.append((i, line))
    is_credit = lambda line: bool(credit.match(converter.convert(unicodedata.normalize('NFKC', line))))
    artist_ids = set().union(*(identity(artist) for artist in artists))
    def title_artist(line):
        return any(identity(candidate_title, title=True) & identity(title, title=True) and
                   (identity(candidate_artist) | identity(re.split(r'[（(]', candidate_artist)[0])) & artist_ids
                   for match in re.finditer(r'[-–—]', line)
                   for candidate_title, candidate_artist in (
                       (line[:match.start()], line[match.end():]), (line[match.end():], line[:match.start()])))
    first = 0
    while first < len(rows):
        line = rows[first][1]
        if is_credit(line) or title_artist(line) or (identity(line) & identity(title) and
                first+1 < len(rows) and is_credit(rows[first+1][1])):
            first += 1
        else:
            break
    stop = next((i for i in range(first, len(rows)) if is_credit(rows[i][1])), len(rows))
    body = rows[first:stop]
    marker = re.sub(r'[^\w]', '', converter.convert(''.join(line for _, line in body))).casefold()
    if marker in {'纯音乐请欣赏', '纯音乐请您欣赏', '此歌曲为没有填词的纯音乐请您欣赏',
                  '暂无歌词', '暂无该歌曲歌词', '歌词暂未提供', 'nolyricsavailable',
                  'lyricsnotavailable', 'instrumental'}:
        return [], sorted(metadata+[i for i, _ in rows])
    if body and all(i in timestamps for i, _ in body):
        body = [(i, line) for _, i, line in sorted((at, i, line) for i, line in body for at in timestamps[i])]
    elif any(len(timestamps.get(i, [])) > 1 for i, _ in body):
        raise ValueError('Repeated LRC lines need a fully timed body to preserve lyric order')
    return [line for _, line in body], sorted(metadata+[i for i, _ in rows[:first]+rows[stop:]])


def version_tags(text):
    text = OpenCC('tw2s').convert(text or '').casefold()
    return {kind for kind, pattern in (
        ('live', r'\blive\b|演唱会|现场'), ('remix', r'\bremix\b|混音'),
        ('acoustic', r'\bacoustic\b'), ('instrumental', r'\binstrumental\b|纯音乐'),
        ('karaoke', r'\bkaraoke\b|伴奏版')) if re.search(pattern, text)}


def recording_mismatches(song, source_title, title, album):
    expected = version_tags(source_title) | version_tags(song['title'])
    actual = version_tags((title or '')+' '+(album or ''))
    # Live/acoustic releases may supply text for alignment to our own audio.
    # Instrumentals, karaoke edits and remixes are not interchangeable lyrics.
    if (expected ^ actual) - {'live', 'acoustic'}:
        return None
    mismatch = ['version'] if expected != actual else []
    performance = live_performance(source_title)
    if performance and performance not in (live_performance(title), live_performance(album)):
        mismatch.append('live_performance')
    return mismatch


def short_lyric_text(lines, duration):
    """Reject obvious scraps, not certify completeness or acoustic coverage."""
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return None
    text = unicodedata.normalize('NFKC', ' '.join(lines)).casefold()
    units = len(re.findall(r"[a-z\u00c0-\u024f]+(?:['’][a-z\u00c0-\u024f]+)*|\d+|[^\W_]", text))
    # Deliberately sparse: one Latin word / other character per ten seconds.
    # Count repetitions and ignore line wrapping; neither is evidence of junk.
    minimum = math.ceil(duration / 10)
    if units < minimum:
        return dict(reason='lyrics_too_short', units=units, minimum_units=minimum, duration=duration)
    return None


def published_candidate(rows, song, source_title, *, read_lyrics=None, allow_recording_mismatch=False,
                        duration=None, reference_lines=()):
    """Select published text; recording onsets still come entirely from our audio."""
    reference = text_units(''.join(reference_lines), text_language(''.join(reference_lines)) or 'English')
    usable_reference = len(reference) >= 40 and len(set(reference)) >= 10
    candidates, identity_candidates = [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get('trackTitle')
        if not isinstance(title, str) or not isinstance(row.get('artists'), list) or not isinstance(row.get('album'), dict):
            continue
        artists = [a.get('artistName') for a in row['artists'] if isinstance(a, dict)]
        album = row['album'].get('albumTitle', '')
        if not isinstance(album, str):
            continue
        if not any(identity(song['artist']) & identity(a) for a in artists):
            continue
        title_match = bool(identity(song['title'], title=True) & identity(title, title=True))
        if not title_match:
            identity_candidates.append(dict(title=title, track_id=str(row.get('trackId', ''))))
            if not usable_reference:
                continue
        mismatch = recording_mismatches(song, source_title, title, album)
        if mismatch is None:
            continue
        seconds = row.get('duration_seconds')
        seconds = seconds if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0 else None
        difference = abs(seconds-duration) if seconds is not None and duration is not None else math.inf
        if math.isfinite(difference) and difference > max(3., min(10., duration*.02)):
            mismatch = [*mismatch, 'duration']
        if mismatch and not allow_recording_mismatch:
            continue
        album_match = bool(identity(song.get('album')) & identity(album))
        rank = (not title_match, len(mismatch), not album_match, difference)
        candidates.append((rank, mismatch, row, title, artists, album, seconds, title_match))
    # A title can have different release edits. Use known recording metadata
    # before provider order, while retaining complete text from other releases.
    rejected, identity_attempts = [], 0
    for _, mismatch, row, title, artists, album, seconds, title_match in sorted(candidates, key=lambda c: c[0]):
        if not title_match:
            if identity_attempts >= 5:
                continue
            identity_attempts += 1
        try:
            text = row.get('lyrics')
            if text is None and read_lyrics is not None:
                text = read_lyrics(row)
            if not isinstance(text, str) or not text.strip() or len(text) > 30000:
                continue
            lines, excluded = published_lyric_lines(text, title, artists)
        except Exception as error:
            failure = lyric_request_error(error)
            if failure.get('requires_access') or failure.get('http_status') in (401, 403, 429):
                raise
            rejected.append(dict(reason='candidate_unavailable', track_id=str(row.get('trackId', '')),
                                 **failure))
            continue
        if not lines:
            rejected.append(dict(reason='empty_lyric_body', track_id=str(row.get('trackId', '')),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded_credit_lines=excluded))
            continue
        language = song.get('language') or text_language('\n'.join(lines))
        if language is None:
            continue
        identity_support = None
        if not title_match:
            left = text_units(''.join(lines), language)
            right = text_units(''.join(reference_lines), language)
            matched = sum(o.src_end-o.src_start for o in Levenshtein.opcodes(left, right) if o.tag == 'equal')
            if min(len(left), len(right)) < 40 or min(len(set(left)), len(set(right))) < 10 or (
                    matched < .9*min(len(left), len(right)) or matched < .65*max(len(left), len(right))):
                rejected.append(dict(reason='unconfirmed_title_alias', track_id=str(row.get('trackId', ''))))
                continue
            identity_support = dict(matched_units=matched, candidate_units=len(left), reference_units=len(right),
                reference_sha256=hashlib.sha256('\n'.join(reference_lines).encode()).hexdigest())
        lines = transcript_lines(dict(text='\n'.join(lines)), language)
        track_id = str(row.get('trackId', ''))
        if not track_id.isdigit() or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            continue
        if issue := short_lyric_text(lines, duration):
            rejected.append(dict(issue, track_id=track_id, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            continue
        return lines, dict(status='candidate', provider='line_music', language=language,
            url='https://music-tw.line.me/track/'+track_id, track=title, artists=artists, album=album,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded_credit_lines=excluded,
            recording_mismatches=mismatch, duration_seconds=seconds,
            title_match='title' if title_match else 'corroborated_lyric_text', identity_support=identity_support,
            rejected_candidates=rejected,
            text_transform='html_entities_decoded_credit_header_footer_removed_then_whitespace_only',
            provider_timestamps_used=False)
    return None, dict(status='no_matching_published_text', provider='line_music', rejected_candidates=rejected,
                      identity_candidates=identity_candidates)


def fetch_published_lyrics(song, source_title, cache, duration=None, *, reference_lines=()):
    if not song.get('title') or not song.get('artist'):
        return None, dict(status='missing_identity', provider='line_music')
    path, stage = cache.root/'published-lyrics.json', 'fetch_published_lyrics'
    title = names(song['title'], title=True)[-1]
    queries = list(dict.fromkeys([' '.join((title, names(song['artist'])[-1])), title]))
    try:
        old = json.loads(path.read_text())
        if lyric_retry_due(dict(old.get('report') or {}, retry_at=old.get('retry_at')), 'line_music', cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__)], outputs=[path],
                     config=dict(song=song, source_title=source_title, duration=duration, reference_lines=reference_lines),
                     code=[Path(__file__), Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            try:
                attempted, rejected = [], []
                for query in queries:
                    url = 'https://music-tw.line.me/api/search/v4/track.json?'+urlencode(
                        dict(query=query, start=1, display=30, sort='RELEVANCE'))
                    response = cached_lyric_request('line_music', url, cache,
                        lambda: read_document(url, headers={'CHANNEL-ID': '1581655766'}))
                    rows = response['response']['result']['tracks']
                    if not isinstance(rows, list):
                        raise ValueError('Invalid published lyric response')
                    attempted.append(dict(query=query, results=rows))
                    lines, report = published_candidate(rows, song, source_title,
                        allow_recording_mismatch=True, duration=duration, reference_lines=reference_lines)
                    rejected.extend(report.get('rejected_candidates', []))
                    if lines:
                        break
                report['rejected_candidates'] = rejected
                packet = dict(lines=lines, report=report, query_results=attempted)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                report = lyric_request_error(error)
                packet = dict(lines=None, report=dict(report, provider='line_music', rejected_candidates=rejected),
                    query_results=attempted, retry_at=report.get('retry_at'))
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    packet = json.loads(path.read_text())
    return packet['lines'], dict(packet['report'], **({'retry_at':packet['retry_at']} if packet.get('retry_at') else {}))


def prepare_online_lyrics(request, metadata, duration, cache, model, catalog_root):
    """Find supplied text before ASR; this is a candidate, not independent audio validation."""
    song = video_song_info(metadata)
    # Upload/default-track tags are hints, not the language of published text.
    # Keep explicit user choices; otherwise choose from the actual lyric body.
    metadata_language = song.pop('language', None)
    song.update({key: request[key] for key in ('title', 'artist', 'language') if request.get(key)})
    lines, provenance = request.get('lyrics'), dict(kind='user_supplied')
    metadata_error = None
    if not lines:
        catalog = find_verified_lyrics(request['url'], duration, catalog_root)
        if catalog:
            return dict(catalog, id='song', source_url=request['url'], provenance=dict(
                catalog['provenance'], route='lyrics_first', needs_review=True, independent_asr_coverage=None))
        description = (metadata.get('description') or '')[:30000]
        description_lines = description.splitlines()
        evidence = {key: metadata.get(key) for key in ('title', 'track', 'artist', 'artists', 'uploader')}
        evidence['description_lines'] = [dict(number=i+1, text=line) for i, line in enumerate(description_lines)]
        from openai import OpenAIError
        preserve_title = bool(request.get('title') or metadata.get('track'))
        metadata_read, parsed, metadata_failure = False, None, None
        metadata_retries = []
        def read_description(resolve_identity=(), *, retry=False):
            nonlocal metadata_error, metadata_read, parsed, metadata_failure
            if not resolve_identity and not description.strip():
                return None
            if retry and metadata_failure and not metadata_retries:
                error, failed_at = metadata_failure
                fields = retry_fields(error, '', 1, failed_at=failed_at)
                deadline = fields.get('next_retry_at')
                delay = max(0, deadline-time.time()) if deadline else None
                metadata_retries.append(dict(error=error, next_retry_at=deadline,
                                             attempted=delay is not None and delay <= 60))
                if metadata_retries[-1]['attempted']:
                    time.sleep(delay)
                    metadata_read = False
            try:
                if not metadata_read:
                    metadata_read = True
                    packet = extract_song_metadata(evidence, model, cache)
                    parsed = SongText.model_validate(packet['parsed']) if packet['parsed'] is not None else None
            except (OpenAIError, ValueError) as error:
                metadata_error = type(error).__name__
                classified = provider_error(error)
                metadata_failure = (classified, time.time()) if classified else None
                parsed = None
            if parsed is None:
                return None
            if 'title' in resolve_identity:
                song['title'] = parsed.title or song['title']
            if 'artist' in resolve_identity:
                song['artist'] = parsed.artist or song['artist']
            selected = parsed.lyric_line_numbers
            if not selected or selected != sorted(set(selected)) or not all(1 <= i <= len(description_lines) for i in selected):
                return None
            text = [description_lines[i-1].strip() for i in selected if description_lines[i-1].strip()]
            if not text:
                return None
            return text, dict(kind='video_description', provider='video_description', publisher='youtube',
                description_sha256=hashlib.sha256(description.encode()).hexdigest(), lyric_line_numbers=selected,
                language=song.get('language') or text_language(''.join(text)), url=request['url'],
                text_sha256=hashlib.sha256('\n'.join(text).encode()).hexdigest())
        from .lyric_consensus import fetch_lyric_consensus
        lines, report = fetch_lyric_consensus(song, metadata, duration, cache, read_description, preserve_title)
        if not lines or (parsed and report.get('crosscheck', {}).get('status') == 'insufficient_sources'):
            retained_lines, retained_report, retained_song = lines, report, dict(song)
            original = {key: song.get(key) for key in ('title', 'artist')}
            previous, attempts, accepted = report, [], False
            # Retain a platform-confirmed performer while resolving an upload
            # title. A description can also credit producers or session players.
            for key in ('title', 'artist'):
                if request.get(key):
                    continue
                before = dict(song)
                description_text = read_description([key], retry=True)
                if parsed is None and metadata_failure and not metadata_retries:
                    description_text = read_description([key], retry=True)
                if song == before:
                    continue
                blocked = {r['provider']: (None, r) for r in report.get('crosscheck', {}).get('unavailable_sources', [])
                           if r.get('error') and r.get('provider')}
                lines, report = fetch_lyric_consensus(song, metadata, duration, cache, description_text, True,
                                                      prefetched=blocked)
                attempts.append(dict(identity={key: song.get(key) for key in original}, lookup=report))
                # A source can attach another song's text to valid metadata.
                # Reuse already-read native credits to seek corroboration;
                # a second isolated guess cannot replace the first candidate.
                review = report.get('crosscheck', {})
                accepted = bool(lines) and (not retained_lines or review.get('status') == 'corroborated' or
                                           review.get('selection') == 'similarity_support')
                if accepted:
                    break
            if retained_lines and not accepted:
                lines, report = retained_lines, retained_report
                song.clear()
                song.update(retained_song)
            if attempts:
                report = dict(report, identity_recovery=dict(original=original,
                    resolved={key: song.get(key) for key in original}, recovered=accepted,
                    attempts=attempts, previous_lookup=previous))
        if metadata_language and report.get('language') and metadata_language != report['language']:
            report = dict(report, metadata_language_corrected=dict(original=metadata_language, resolved=report['language']))
        if metadata_error:
            report = dict(report, metadata_error=metadata_error)
        if metadata_retries:
            report = dict(report, metadata_retries=metadata_retries)
        (cache.root/'lyrics-first-lookup.json').write_text(json.dumps(report, ensure_ascii=False)+'\n')
        if not lines:
            return None
        song['language'] = report['language']
        provenance = dict(kind='online_lyric_text', lyric_lookup=report,
                          text_transform=report.get('text_transform', 'whitespace_only'))
    language = song.get('language') or text_language(''.join(lines))
    if not language or len(lines) > 1000 or any(not line.strip() or len(line) > 200 for line in lines):
        return None
    return dict(song, id='song', source_url=request['url'], language=language, lines=lines,
                provenance=dict(provenance, needs_review=True, route='lyrics_first', independent_asr_coverage=None,
                                **({'metadata_error': metadata_error} if metadata_error else {})))


def prepared_lyrics_identity(url, metadata):
    """Bind split-host lookup to the same media metadata and lookup implementation."""
    root = Path(__file__).parent
    return dict(url=url, metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                code_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('lyric_lookup.py', 'lyric_consensus.py', 'song_sources.py', 'source_captions.py')})


def read_prepared_lyrics(path, url, metadata):
    packet = json.loads(Path(path).read_text())
    if packet['identity'] != prepared_lyrics_identity(url, metadata):
        raise RuntimeError('Prepared lyric lookup does not match this source or implementation')
    lyrics = packet['lyrics']
    if lyrics is not None and (lyrics['source_url'] != url or lyrics['provenance'].get('route') != 'lyrics_first'):
        raise RuntimeError('Prepared lyrics have the wrong source or route')
    return lyrics


def lyric_candidate(packet, song, transcription, duration, *, audio_path=None, source_title=None,
                    allow_recording_mismatch=False):
    if allow_recording_mismatch and transcription is not None:
        raise ValueError('Text-only candidates cannot validate independent ASR coverage')
    report = {key: value for key, value in packet.items() if key != 'results'}
    if packet.get('status') != 'ok':
        return None, report
    performance = live_performance(source_title)
    if performance:
        report['required_live_performance'] = performance
    candidates, padding, padding_checked = [], None, False
    report['rejected_candidates'] = []
    for row in packet['results']:
        # A text search can return covers, live cuts and unrelated recordings.
        # Identity and duration must agree before using any audio-text score.
        if (not identity(song['title'], title=True) & identity(row.get('trackName'), title=True) or
                not identity(song['artist']) & identity(row.get('artistName')) or row.get('instrumental') is not False):
            continue
        mismatch = (recording_mismatches(song, source_title, row.get('trackName'), row.get('albumName'))
                    if allow_recording_mismatch else ['live_performance'] if performance and performance not in
                    {live_performance(row.get(field)) for field in ('trackName', 'albumName')} else [])
        if mismatch is None or mismatch and not allow_recording_mismatch:
            continue
        try:
            reference_duration, basis = duration, None
            difference = abs(float(row['duration'])-reference_duration)
            if difference > max(3., min(10., duration*.02)) and audio_path is not None:
                if not padding_checked:
                    padding = unpadded_duration(audio_path, duration)
                    padding_checked = True
                if padding is not None:
                    reference_duration, basis = padding['nonzero_duration'], padding
                    difference = abs(float(row['duration'])-reference_duration)
            if not math.isfinite(difference):
                continue
            if difference > max(3., min(10., reference_duration*.02)):
                if not allow_recording_mismatch:
                    continue
                mismatch.append('duration')
            raw_text = row['plainLyrics']
            if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > 30000:
                continue
            clean_lines, excluded = published_lyric_lines(raw_text, row['trackName'], [row['artistName']])
            text = '\n'.join(clean_lines)
            if not text:
                report['rejected_candidates'].append(dict(reason='empty_lyric_body', id=row.get('id'),
                    text_sha256=hashlib.sha256(raw_text.encode()).hexdigest(), excluded_credit_lines=excluded))
                continue
            language = song.get('language') or text_language(text)
            if language is None:
                continue
            lines = transcript_lines(dict(text=text), language)
            if transcription is None and (issue := short_lyric_text(lines, duration)):
                report['rejected_candidates'].append(dict(issue, id=row.get('id'),
                    text_sha256=hashlib.sha256(raw_text.encode()).hexdigest()))
                continue
            observed = [u for w in (transcription or {}).get('words', []) for u in text_units(w['word'], language)]
            expected = text_units(''.join(lines), language)
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
            support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
            if (not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines) or
                    (transcription is not None and min(support, coverage) < .55)):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        candidates.append((not mismatch, min(support, coverage), -difference, lines, dict(
            id=row['id'], url=f"https://lrclib.net/api/get/{row['id']}",
            track=row['trackName'], artist=row['artistName'], album=row.get('albumName'),
            duration=row['duration'], language=language,
            text_support=support if transcription is not None else None,
            audio_word_coverage=coverage if transcription is not None else None,
            duration_basis=basis or dict(kind='recording', recording_duration=duration),
            recording_mismatches=mismatch,
            excluded_credit_lines=excluded, provider_timestamps_used=False,
            text_sha256=hashlib.sha256(raw_text.encode()).hexdigest())))
    if not candidates:
        return None, dict(report, status='no_supported_recording')
    best = max(candidates, key=lambda c: c[:3])
    return best[3], dict(report, status='accepted' if transcription is not None else 'candidate', provider='lrclib', **best[4])
