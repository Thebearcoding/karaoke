"""Optional independent lyric text; provider timestamps never become onsets."""
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlencode

import httpx
from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

from .anchors_from_asr import text_units
from .job_schema import retry_after_seconds
from .song_sources import (SongText, extract_song_metadata, find_verified_lyrics,
                           transcript_lines, video_song_info)
from .source_captions import read_document


def names(value):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').strip()
    result = [value] if value else []
    # Music metadata often appends an English translation to a Chinese name.
    # Version labels are not translations: never strip "Live", "Remix", etc.
    match = re.fullmatch(r'([\u3400-\u9fff]+)\s+([A-Za-z][\x20-\x7e]*)', value)
    if match and not re.search(r'\b(live|remix|acoustic|instrumental|karaoke|remaster\w*)\b',
                               match[2], re.I):
        result.extend((match[2], match[1]))
    return result


def identity(value, *, title=False):
    converter = OpenCC('tw2s')
    aliases = [converter.convert(name).casefold() for name in names(value)]
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
    titles, artists = names(song.get('title')), names(song.get('artist'))
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
        if old.get('retry_at') is not None and old['retry_at'] <= time.time():
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
                    response = read_document('https://lrclib.net/api/search?'+urlencode(query))
                    if not isinstance(response, list) or len(response) > 100:
                        raise ValueError('Invalid lyric search response')
                    packet['results'].extend({key: row.get(key) for key in (
                        'id', 'trackName', 'artistName', 'albumName', 'duration', 'instrumental', 'plainLyrics')}
                        for row in response if isinstance(row, dict))
            except (httpx.HTTPError, ValueError) as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                transient = isinstance(error, httpx.TransportError) or status == 429 or (status or 0) >= 500
                packet.update(status='unavailable', error=type(error).__name__, http_status=status,
                              retry_at=time.time()+max(300, retry_after_seconds(error) or 0) if transient else None)
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


def published_candidate(rows, song, source_title):
    """Select published text; recording onsets still come entirely from our audio."""
    def versions(text):
        text = OpenCC('tw2s').convert(text or '').casefold()
        return {kind for kind, pattern in (
            ('live', r'\blive\b|演唱会|现场'), ('remix', r'\bremix\b|混音'),
            ('acoustic', r'\bacoustic\b'), ('instrumental', r'\binstrumental\b|纯音乐'),
            ('karaoke', r'\bkaraoke\b|伴奏版')) if re.search(pattern, text)}

    expected_versions = versions(source_title) | versions(song['title'])
    for row in rows:
        title = row.get('trackTitle')
        if not isinstance(title, str) or not isinstance(row.get('artists'), list) or not isinstance(row.get('album'), dict):
            continue
        artists = [a.get('artistName') for a in row['artists'] if isinstance(a, dict)]
        album = row['album'].get('albumTitle', '')
        if not isinstance(album, str):
            continue
        if (not identity(song['title'], title=True) & identity(title, title=True) or
                not any(identity(song['artist']) & identity(a) for a in artists) or
                versions((title or '')+' '+album) != expected_versions):
            continue
        performance = live_performance(source_title)
        if performance and performance not in (live_performance(title), live_performance(album)):
            continue
        text = row.get('lyrics')
        if not isinstance(text, str) or not text.strip() or len(text) > 30000:
            continue
        lines, excluded = [], []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            if not lines and re.match(
                    r'^\s*(?:歌名|詞曲|词曲|作詞|作词|作曲|編曲|编曲|演唱|專輯|专辑|發行日|发行日|'
                    r'title|artist|album|composer|lyricist)\s*[：:]', line, re.I):
                excluded.append(number)
            else:
                lines.append(line.strip())
        language = song.get('language') or text_language('\n'.join(lines))
        if not lines or language is None or any('[' in line or ']' in line for line in lines):
            continue
        lines = transcript_lines(dict(text='\n'.join(lines)), language)
        track_id = str(row.get('trackId', ''))
        if not track_id.isdigit() or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            continue
        return lines, dict(status='candidate', provider='line_music', language=language,
            url='https://music-tw.line.me/track/'+track_id, track=title, artists=artists, album=album,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(), excluded_credit_lines=excluded,
            text_transform='leading_metadata_removed_then_whitespace_only', provider_timestamps_used=False)
    return None, dict(status='no_matching_published_text', provider='line_music')


def fetch_published_lyrics(song, source_title, cache):
    if not song.get('title') or not song.get('artist'):
        return None, dict(status='missing_identity', provider='line_music')
    path, stage = cache.root/'published-lyrics.json', 'fetch_published_lyrics'
    query = ' '.join((names(song['title'])[-1], names(song['artist'])[-1]))
    try:
        old = json.loads(path.read_text())
        if old.get('retry_at') is not None and old['retry_at'] <= time.time():
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__)], outputs=[path], config=dict(song=song, source_title=source_title),
                     code=[Path(__file__), Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            try:
                response = read_document('https://music-tw.line.me/api/search/v4/track.json?'+urlencode(
                    dict(query=query, start=1, display=30, sort='RELEVANCE')),
                    headers={'CHANNEL-ID': '1581655766'})
                rows = response['response']['result']['tracks']
                if not isinstance(rows, list) or len(rows) > 30 or any(not isinstance(r, dict) for r in rows):
                    raise ValueError('Invalid published lyric response')
                lines, report = published_candidate(rows, song, source_title)
                packet = dict(lines=lines, report=report)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                transient = isinstance(error, httpx.TransportError) or status == 429 or (status or 0) >= 500
                packet = dict(lines=None, report=dict(status='unavailable', provider='line_music',
                    error=type(error).__name__, http_status=status),
                    retry_at=time.time()+max(30, retry_after_seconds(error) or 0) if transient else None)
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    packet = json.loads(path.read_text())
    return packet['lines'], packet['report']


def prepare_online_lyrics(request, metadata, duration, cache, model, catalog_root):
    """Find supplied text before ASR; this is a candidate, not independent audio validation."""
    song = video_song_info(metadata)
    song.update({key: request[key] for key in ('title', 'artist', 'language') if request.get(key)})
    lines, provenance = request.get('lyrics'), dict(kind='user_supplied')
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
        try:
            packet = (dict(parsed=None) if song['artist'] and (metadata.get('track') or
                      song['title'] != metadata.get('title') or request.get('title') and request.get('artist')) else
                      extract_song_metadata(evidence, model, cache))
            parsed = SongText.model_validate(packet['parsed']) if packet['parsed'] is not None else None
        except (OpenAIError, ValueError):
            parsed = None
        if parsed is not None:
            song.update(title=request.get('title') or metadata.get('track') or parsed.title or song['title'],
                        artist=song['artist'] or parsed.artist)
            selected = parsed.lyric_line_numbers
            if selected and selected == sorted(set(selected)) and all(1 <= i <= len(description_lines) for i in selected):
                lines = [description_lines[i-1].strip() for i in selected if description_lines[i-1].strip()]
                provenance = dict(kind='video_description', description_sha256=hashlib.sha256(description.encode()).hexdigest(),
                                  lyric_line_numbers=selected)
        if not lines:
            lines, report = fetch_published_lyrics(song, metadata.get('title'), cache)
            if not lines:
                packet = fetch_lyrics(song, cache.root/'download/source.info.json', cache)
                lines, fallback_report = lyric_candidate(packet, song, None, duration, source_title=metadata.get('title'))
                report = dict(fallback_report, published_lookup=report)
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
                provenance=dict(provenance, needs_review=True, route='lyrics_first', independent_asr_coverage=None))


def lyric_candidate(packet, song, transcription, duration, *, audio_path=None, source_title=None):
    report = {key: value for key, value in packet.items() if key != 'results'}
    if packet.get('status') != 'ok':
        return None, report
    performance = live_performance(source_title)
    if performance:
        report['required_live_performance'] = performance
    candidates, padding, padding_checked = [], None, False
    for row in packet['results']:
        # A text search can return covers, live cuts and unrelated recordings.
        # Identity and duration must agree before using any audio-text score.
        if (not identity(song['title'], title=True) & identity(row.get('trackName'), title=True) or
                not identity(song['artist']) & identity(row.get('artistName')) or row.get('instrumental') is not False):
            continue
        if performance and performance not in {live_performance(row.get(field)) for field in ('trackName', 'albumName')}:
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
            if not math.isfinite(difference) or difference > max(3., min(10., reference_duration*.02)):
                continue
            text = row['plainLyrics']
            if not isinstance(text, str) or not text.strip() or len(text) > 30000 or re.search(r'\[[^\]\n]+\]', text):
                continue
            language = song.get('language') or text_language(text)
            if language is None:
                continue
            lines = transcript_lines(dict(text=text), language)
            observed = [u for w in (transcription or {}).get('words', []) for u in text_units(w['word'], language)]
            expected = text_units(''.join(lines), language)
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
            support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
            if (not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines) or
                    (transcription is not None and min(support, coverage) < .55)):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        candidates.append((min(support, coverage), -difference, lines, dict(
            id=row['id'], track=row['trackName'], artist=row['artistName'], album=row.get('albumName'),
            duration=row['duration'], language=language,
            text_support=support if transcription is not None else None,
            audio_word_coverage=coverage if transcription is not None else None,
            duration_basis=basis or dict(kind='recording', recording_duration=duration),
            text_sha256=hashlib.sha256(text.encode()).hexdigest())))
    if not candidates:
        return None, dict(report, status='no_supported_recording')
    best = max(candidates, key=lambda c: c[:2])
    return best[2], dict(report, status='accepted' if transcription is not None else 'candidate', provider='lrclib', **best[3])
