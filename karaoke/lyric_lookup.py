"""Optional independent lyric text; provider timestamps never become onsets."""
import hashlib
import html
from html.parser import HTMLParser
from difflib import SequenceMatcher
from functools import lru_cache
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
from .job_schema import provider_error, retry_after_seconds, retry_fields, song_language
from .song_sources import (SongText, extract_song_metadata, find_verified_lyrics,
                           join_punctuation_lines, video_song_info)
from .source_captions import read_document


_traditional, _simplified = OpenCC('s2t'), OpenCC('tw2s')
_versioned_title = re.compile(r'(.+?)\s*\(([^()]*\b(?:version|ver\.?))\)\s*$', re.I)
_soundtrack_credit = re.compile(r'(?:大型)?(?:纪录片|电影|电视剧|动画片|网络剧|网剧|剧集)?'
    r'\s*《[^《》]+》\s*(?:人物)?(?:主题[歌曲]|片[头尾]曲|插曲)|from\s+["“][^"“”]+["”]', re.I)
_credit_separators = r'\s*[&,/×+、;]\s*|\s+·\s+|\s+\b(?:feat\.?|featuring|ft\.?|with)\s+'


def canonical_script(text):
    """Compare mixed Han scripts consistently; never rewrite the published body."""
    return _simplified.convert(_traditional.convert(unicodedata.normalize('NFKC', text)))


def comparison_key(text):
    return ''.join(c for c in canonical_script(text).casefold() if c.isalnum())


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
    # Split explicit bilingual names before parsing either side's annotations.
    pair = re.fullmatch(r'(.+?)\s+[-–—]\s+(.+)', value)
    if pair:
        native_side = lambda text: any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text)
        if native_side(pair[1]) != native_side(pair[2]) and version_tags(pair[1]) == version_tags(pair[2]):
            for part in ((pair[2], pair[1]) if native_side(pair[1]) else (pair[1], pair[2])):
                result.extend(names(part, title=title))
            return list(dict.fromkeys(result))
    if title:
        edition = _versioned_title.fullmatch(value)
        if edition:
            return result + names(edition[1], title=True)
        annotation = re.fullmatch(r'(.+?)\s*\(([^()]*)\)', value)
        if annotation:
            label = canonical_script(annotation[2]).casefold()
            release = re.fullmatch(r'(?:live|现场)(?:版本?)?(?:[ |/].*)?|cover(?: version)?|'
                r'(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?', label)
            if release or _soundtrack_credit.fullmatch(label):
                return result + names(annotation[1].strip(), title=True)
    # Platforms publish original names together with Latin-script aliases.
    # Version labels are not translations: never strip "Live", "Remix", etc.
    native = r'[^\W\d_\x00-\x7f]+(?:[ ,·、&/]+[^\W\d_\x00-\x7f]+)*'
    latin = r'[A-Za-z][\x20-\x7e]*?'
    match = re.fullmatch(rf'({native})(?:\s+(?:[-–—]\s+)?({latin})|\s*\(({latin})\))', value)
    reverse = re.fullmatch(rf'({latin})(?:\s+({native})|\s*\(({native})\))', value)
    if match or reverse:
        local, english = (match[1], match[2] or match[3]) if match else (reverse[2] or reverse[3], reverse[1])
        if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in local) and not re.search(r'\b(live|remix|acoustic|instrumental|karaoke|remaster\w*)\b', english, re.I) and not re.search(
                r'라이브|리믹스|어쿠스틱|인스트루멘탈|버전', local):
            result.extend((english, local))
    return result


def song_names(song, key):
    """Keep source-confirmed spellings available after choosing a display name."""
    return list(dict.fromkeys(name for value in [song.get(key), *song.get(key+'_aliases', [])]
                              for name in names(value, title=key == 'title')))


def identity(value, *, title=False):
    aliases = [canonical_script(name).casefold() for name in names(value, title=title)]
    if not title:
        if aliases and re.search(r'[&,/×+]|\b(?:feat\.?|featuring|ft\.?|with)\s', aliases[0]):
            aliases = aliases[:1]  # A collaborator is not an alias for the whole credit.
        joined = re.fullmatch(r"([a-z][a-z.'-]*)([\u3040-\u30ff\u3400-\u9fff]{2,})(?:\s+[a-z][a-z .'-]*)?",
                              aliases[0] if aliases else '')
        if joined and not version_tags(joined[1]):
            aliases.append(joined[2])
    return {''.join(c for c in name if c.isalnum()) for name in aliases} - {''}


def artist_credit_matches(credit, artists, *, partial=False):
    """Match whole credited names, including reordered multi-artist credits."""
    expected = identity(credit)
    if not expected:
        return False
    if any(expected & identity(artist) for artist in artists):
        return True
    members = [identity(part) for artist in artists if isinstance(artist, str)
               for part in re.split(_credit_separators, artist, flags=re.I) if part.strip()]
    if partial:
        return any(identity(part) & member for part in re.split(_credit_separators, credit, flags=re.I) for member in members)
    if len(members) < 2 or not all(members):
        return False
    expected |= identity(''.join(re.split(_credit_separators, credit, flags=re.I)))
    expected.update(name for alias in names(credit) for name in identity(alias))

    @lru_cache(None)
    def consume(text, remaining):
        if not remaining:
            return not text
        return any(text.startswith(name) and consume(text[len(name):], remaining[:i]+remaining[i+1:])
                   for i, member in enumerate(remaining) for name in members[member])

    # Use provider-supplied name boundaries: splitting on spaces would break
    # multi-word English names. Every member must account for the whole credit.
    return any(consume(text, tuple(range(len(members)))) for text in expected)


def live_performance(value):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').casefold()
    match = re.search(r'(?:^|[\(\[]|\s[-–—]\s)\s*live\s+(?:from|at|in)\s+([^()\[\]|\n]+)', value)
    if match:
        venue = re.sub(r'^the\s+', '', match[1].strip())
        return ''.join(c for c in venue if c.isalnum()) or None
    return None


def fetch_lyrics(song, metadata_path, cache):
    titles, artists = song_names(song, 'title'), song_names(song, 'artist')
    if not titles or not artists or not metadata_path.is_file():
        return dict(status='missing_identity')
    queries = [dict(track_name=titles[0], artist_name=artists[0])]
    native = dict(track_name=titles[-1], artist_name=artists[-1])
    if native != queries[0]:
        queries.append(native)
    simplified = {key: canonical_script(value) for key, value in native.items()}
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


@lru_cache(maxsize=128)
def text_language(text):
    if re.search(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]', text):
        return 'Korean'
    if re.search(r'[\u3040-\u30ff]', text):
        return 'Japanese'
    if re.search(r'[\u3400-\u9fff]', text):
        return 'Chinese'
    if any(c.isalpha() for c in text):
        import langid
        return song_language(langid.classify(text)[0])
    return None


def published_lyric_lines(text, title, artists=()):
    """Keep published phrases intact; remove only explicit credit metadata."""
    instrument = (r'(?:钢琴|键盘|(?:低音)?吉他|贝斯|鼓|打击乐|二胡|'
        r'(?:第[一二三四\d]+)?(?:小提琴|中提琴|大提琴))(?:手|独奏|\d+)?')
    production = (r'(?:音乐|音响|灯光|舞台秀?|乐队|声乐|演唱|歌曲)'
        r'(?:总监|设计|指导|统筹|队长|混音|编辑|发行|营销|策划|导演团队)')
    role = (rf'{instrument}|{production}|歌名|歌曲|(?:中文|英文)?词|曲|词曲|作词|作曲|(?:改编)?编曲(?:协力)?|原唱|演唱|专辑|发行(?:日|公司)|出品人?|音乐制作(?:人|助理)?|(?:定位)?制作人|制作助理|监制|'
        r'弦乐(?:编写|编曲|监制|录音)?|(?:人声|配唱|伴唱|主唱|合唱|童声|合声|合音|和声|和音)(?:编写|编曲|编配|制作|录音)?(?:人|师|指导)?|器乐|'
        r'录音(?:师|助理|(?:工作)?室)?|(?:前置)?混音(?:师|(?:工作)?室)?|母带(?:后期(?:处理|制作))?(?:人|工程师|录音室)?|音频(?:助理|工程师|编辑)|'
        r'(?:song )?title|song name|artist|album|composer|lyricist|producer|lyrics|music(?: production)?|written by|produced by|'
        r'guitar|bass|drums|piano|strings|instrumental|recorded by|mixed by|mastered by|audio engineer|assistant engineer|'
        r'乐团|乐队|orchestra|band|arrange(?:ment|r)|(?:backing |lead )?vocals?(?: arrangement)?|(?:recording|mixing|mastering)(?: engineer| studio)?(?: \(vocals\))?|'
        r'production assistant|program(?:ming)?|PGM|[OS]P|O[AC](?:\s*/\s*O[AC])?|publisher|copyright')
    # A translated role is still metadata: "编曲 Arrangement: ...".
    # Require complete recognized labels before the colon, not arbitrary prose.
    credit = re.compile(rf'^(?:{role})(?:(?:\s+|\s*[/／]\s*)(?:{role}))*\s*[：:]', re.I)
    people = [identity(part) for artist in artists if isinstance(artist, str)
              for part in re.split(_credit_separators, artist, flags=re.I) if part.strip()]
    rows, metadata, timestamps = [], [], {}
    for i, line in enumerate(text.splitlines(), 1):
        line = html.unescape(line).strip()
        if re.fullmatch(r'\[(?:ar|ti|al|by|offset|re|ve):[^\]]*\]', line, re.I):
            metadata.append(i)
            continue
        # Provider timestamps describe text order/repetitions, never our onsets.
        # Defer repeated LRC rows until they can be ordered with the other rows.
        prefix = re.match(r'^(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?(?:-\d+)?\]\s*)+', line)
        if prefix:
            stamps = re.findall(r'\[(\d{1,3}):(\d{2}(?:[.:]\d{1,3})?)(?:-\d+)?\]', prefix[0])
            timestamps[i] = [60*int(m)+float(s.replace(':', '.')) for m,s in stamps]
            line = line[prefix.end():]
        # An empty attribution row is a cue to the credited performers, not a
        # sung name. Resolve abbreviations only when one credited name fits.
        speaker = re.fullmatch(r'([^:：]+)[:：]\s*', line)
        if speaker:
            labels = identity(speaker[1])
            matches = [person for person in people if any(name.startswith(label) for name in person for label in labels)]
            if (any(labels & person for person in people) or len(matches) == 1 or
                    len(people) > 1 and labels & {'合', '合唱', 'all'}):
                metadata.append(i)
                continue
        if re.fullmatch(r'\[\s*(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|instrumental|interlude|refrain|solo)'
                        r'(?:\s+\d+)?(?:\s*[:：][^\]\n]+)?\s*\]', line, re.I):
            metadata.append(i)
            continue
        if line:
            rows.append((i, line))
    is_credit = lambda line: bool(credit.match(canonical_script(line.replace('︰', ':'))) or
                                 re.fullmatch(r'credits?(?:\s+list)?\s*[:：]?', line, re.I))
    # Copied page headers can put the title and its byline on separate rows.
    # Both must match known metadata; a sung title or a name alone is not a header.
    if len(rows) > 1 and identity(rows[0][1], title=True) & identity(title, title=True):
        byline = re.fullmatch(r'by\s+(.+)', rows[1][1], re.I)
        if byline and any(artist_credit_matches(name, artists) for name in
                (byline[1], re.sub(r'\s*Play$', '', byline[1], flags=re.I))):
            metadata.extend(i for i, _ in rows[:2])
            rows = rows[2:]
    def title_artist(line):
        return any(identity(candidate_title, title=True) & identity(title, title=True) and
                   any(artist_credit_matches(name, artists) for name in
                       (candidate_artist, re.split(r'[（(]', candidate_artist)[0]))
                   for match in re.finditer(r'[-–—]', line)
                   for candidate_title, candidate_artist in (
                       (line[:match.start()], line[match.end():]), (line[match.end():], line[:match.start()])))
    first = 0
    while first < len(rows):
        line = rows[first][1]
        note = canonical_script(line).strip('()').strip()
        heading_note = first and (_soundtrack_credit.fullmatch(note) or re.fullmatch(
            r'(?:又名|别名|also known as|a\.?k\.?a\.?)\s*[:：].+', note, re.I))
        if is_credit(line) or title_artist(line) or heading_note or (identity(line) & identity(title) and
                first+1 < len(rows) and is_credit(rows[first+1][1])):
            first += 1
        else:
            break
    stop = next((i for i in range(first, len(rows)) if is_credit(rows[i][1])), len(rows))
    body = rows[first:stop]
    marker = re.sub(r'[^\w]', '', canonical_script(''.join(line for _, line in body))).casefold()
    if marker in {'纯音乐请欣赏', '纯音乐请您欣赏', '此歌曲为没有填词的纯音乐请您欣赏',
                  '暂无歌词', '暂无该歌曲歌词', '歌词暂未提供', 'nolyricsavailable',
                  'lyricsnotavailable', 'instrumental'}:
        return [], sorted(metadata+[i for i, _ in rows])
    if body and all(i in timestamps for i, _ in body):
        body = [(i, line) for _, i, line in sorted((at, i, line) for i, line in body for at in timestamps[i])]
    elif any(len(timestamps.get(i, [])) > 1 for i, _ in body):
        raise ValueError('Repeated LRC lines need a fully timed body to preserve lyric order')
    return join_punctuation_lines([line for _, line in body]), sorted(metadata+[i for i, _ in rows[:first]+rows[stop:]])


def description_lyrics(metadata, reference_lines=()):
    """Select an explicitly labelled or corroborated block, never generate text."""
    from rapidfuzz.fuzz import partial_ratio
    description = (metadata.get('description') or '')[:30000]
    rows = list(enumerate(description.splitlines(), 1))
    normalize = lambda text: ''.join(c for c in canonical_script(text).casefold() if c.isalnum())
    reference = normalize(''.join(reference_lines))
    song = video_song_info(metadata)
    sections, section, labelled = [], [], False
    for number, line in [*rows, (len(rows)+1, '#end')]:
        if re.fullmatch(r'[\[【<＜#\s]*(?:lyrics?|歌詞|歌词|가사)[\]】>＞\s:：]*', line, re.I):
            section, labelled = [], True
        elif re.search(r'https?://|www\.|^#|^[-_=━─]{2,}$', line.strip()):
            if section:
                sections.append((section, labelled))
            section, labelled = [], False
        else:
            section.append((number, line.strip()))
    candidates = []
    for section, labelled in sections:
        if not labelled:
            if len(reference) < 40:
                continue
            matching = [i for i, (_, line) in enumerate(section) if len(normalize(line)) >= 4 and
                        partial_ratio(normalize(line), reference) >= 85]
            if len(matching) < 3:
                continue
            first, last = matching[0], matching[-1]
            # Retain whole opening/closing stanzas, including orthographic
            # variants that did not match a particular reference spelling.
            while first and section[first-1][1]: first -= 1
            while last+1 < len(section) and section[last+1][1]: last += 1
            section = section[first:last+1]
        selected = [(n, line) for n, line in section if line]
        text, excluded = published_lyric_lines('\n'.join(line for _, line in selected),
            song['title'], [song['artist']])
        body = normalize(''.join(text))
        overlap = sum(b.size for b in SequenceMatcher(None, body, reference, autojunk=False).get_matching_blocks())
        if not body or (not labelled and overlap < max(len(reference), len(body))*.7):
            continue
        candidates.append((labelled, overlap, text, [n for i, (n, _) in enumerate(selected, 1) if i not in excluded]))
    if not candidates:
        return None
    labelled, _, text, numbers = max(candidates, key=lambda item: item[:2])
    return text, dict(provider='video_description', publisher='youtube', kind='video_description',
        language=text_language(''.join(text)), url='https://www.youtube.com/watch?v='+metadata.get('id', ''),
        description_sha256=hashlib.sha256(description.encode()).hexdigest(), lyric_line_numbers=numbers,
        text_sha256=hashlib.sha256('\n'.join(text).encode()).hexdigest(),
        extraction='explicit_lyric_heading' if labelled else 'published_text_overlap',
        provider_timestamps_used=False)


def version_tags(text):
    text = canonical_script(text or '').casefold()
    return {kind for kind, pattern in (
        ('live', r'\blive(?:版本?)?\b|演唱会|现场'), ('remix', r'\bremix\b|混音'),
        ('acoustic', r'\bacoustic\b'), ('instrumental', r'\binstrumental\b|纯音乐'),
        ('karaoke', r'\bkaraoke\b|伴奏版')) if re.search(pattern, text)}


def recording_mismatches(song, source_title, title, album):
    expected = version_tags(source_title) | version_tags(song['title'])
    actual = version_tags((title or '')+' '+(album or ''))
    # Live/acoustic releases may supply text for alignment to our own audio.
    # Instrumentals, karaoke edits and remixes are not interchangeable lyrics.
    if (expected ^ actual) - {'live', 'acoustic'}:
        return None
    editions = lambda values: {canonical_script(m[2]).casefold() for value in values
        if (m := _versioned_title.fullmatch(unicodedata.normalize('NFKC', value or '')))}
    mismatch = ['version'] if expected != actual or editions((source_title, song['title'])) != editions((title, album)) else []
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
                        duration=None, reference_lines=(), source_url=None):
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
        artist_aliases = [name for a in row['artists'] if isinstance(a, dict)
                          for name in a.get('artistAliases') or [] if isinstance(name, str)]
        album = row['album'].get('albumTitle', '')
        if not isinstance(album, str):
            continue
        full_credit = any(artist_credit_matches(credit, artists) or artist_credit_matches(credit, artist_aliases)
                          for credit in song_names(song, 'artist'))
        partial_credit = allow_recording_mismatch and any(artist_credit_matches(credit, artists+artist_aliases, partial=True)
                                                          for credit in song_names(song, 'artist'))
        if not (full_credit or partial_credit):
            continue
        title_aliases = [title, *(name for name in row.get('trackAliases') or [] if isinstance(name, str))]
        title_match = any(identity(name, title=True) & identity(alias, title=True)
                          for name in song_names(song, 'title') for alias in title_aliases)
        if not title_match:
            identity_candidates.append(dict(title=title, track_id=str(row.get('trackId', ''))))
            if not usable_reference:
                continue
        mismatch = recording_mismatches(song, source_title, title, album)
        if mismatch is None:
            continue
        if not full_credit:
            mismatch.append('performers')
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
        track_id = str(row.get('trackId', ''))
        url = source_url(track_id) if source_url else ('https://music-tw.line.me/track/'+track_id if track_id.isdigit() else None)
        if not url or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            continue
        if issue := short_lyric_text(lines, duration):
            rejected.append(dict(issue, track_id=track_id, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            continue
        return lines, dict(status='candidate', provider='line_music', language=language,
            url=url, track=title, artists=artists, album=album,
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
    title = song_names(song, 'title')[-1]
    queries = list(dict.fromkeys([' '.join((title, song_names(song, 'artist')[-1])), title]))
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


def catalogue_lyric_identity(song, duration, cache):
    """Recover an uncredited work by title and duration, never by upload channel."""
    query = names(song.get('title'), title=True)
    if not query or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return None, dict(status='missing_title_or_duration')
    url = 'https://music.163.com/api/cloudsearch/pc?'+urlencode(dict(s=query[-1], type=1, limit=30, offset=0))
    try:
        packet = cached_lyric_request('netease', url, cache, lambda: read_document(url, timeout=10))
        if packet.get('code') != 200:
            raise ValueError('NetEase catalogue search rejected the request')
        rows = packet.get('result', {}).get('songs', [])
        if not isinstance(rows, list):
            raise ValueError('Invalid catalogue search response')
    except Exception as error:
        return None, lyric_request_error(error)
    candidates = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('ar'), list):
            continue
        if not identity(song['title'], title=True) & identity(row.get('name'), title=True):
            continue
        artists = [a['name'] for a in row.get('ar', []) if isinstance(a, dict) and isinstance(a.get('name'), str) and a['name'].strip()]
        milliseconds = row.get('dt')
        if not artists or not isinstance(milliseconds, (int, float)) or not math.isfinite(milliseconds):
            continue
        delta = abs(milliseconds/1000-duration)
        if delta > max(3., min(10., duration*.02)):
            continue
        key = tuple(sorted(tuple(sorted(identity(a))) for a in artists))
        candidate = dict(title=row['name'], artist=' · '.join(artists), duration_seconds=milliseconds/1000,
                         duration_difference=delta, url='https://music.163.com/#/song?id='+str(row['id']))
        if key not in candidates or delta < candidates[key]['duration_difference']:
            candidates[key] = candidate
    ordered = sorted(candidates.values(), key=lambda c:c['duration_difference'])
    report = dict(status='no_match', provider='netease', query=query[-1], candidates=ordered,
                  basis='Matching title and recording duration; candidate text still requires acoustic alignment')
    if ordered:
        # Sub-second differences between different credits are within metadata
        # rounding/encoder padding uncertainty, not evidence for one singer.
        if len(ordered) > 1 and ordered[1]['duration_difference'] < ordered[0]['duration_difference']+1:
            return None, dict(report, status='ambiguous')
        return ordered[0], dict(report, status='candidate')
    return None, report


def fetch_reading_lyrics(song, cache, duration=None, reference_lines=(), source_title=None):
    """Reuse the published ruby parser for original Japanese text as well."""
    from .lyric_readings import ReadingPage, read_page
    title, artist = song_names(song, 'title')[-1], song_names(song, 'artist')[-1]
    fetch = lambda url: cached_lyric_request('utaten_readings', url, cache, lambda: read_page(url))
    attempts, seen = [], set()
    reference = ''.join(reference_lines)
    for query in (dict(title=title, artist_name=artist), dict(title=title)):
        url = 'https://utaten.com/search?'+urlencode(query)
        listing = ReadingPage(fetch(url))
        for link in listing.links[:5]:
            if link in seen:
                continue
            seen.add(link)
            url = 'https://utaten.com'+link
            page = ReadingPage(fetch(url))
            actual = page.identity
            title_match = any(identity(name, title=True) & identity(actual.get('title'), title=True)
                              for name in song_names(song, 'title'))
            artist_match = any(artist_credit_matches(name, [actual.get('artist')]) for name in song_names(song, 'artist'))
            partial_credit = any(artist_credit_matches(name, [actual.get('artist')], partial=True) for name in song_names(song, 'artist'))
            mismatch = recording_mismatches(song, source_title, actual.get('title'), None)
            text, excluded = published_lyric_lines(page.text, actual.get('title') or '', [actual.get('artist')])
            left, right = text_units(''.join(text), 'Japanese'), text_units(reference, 'Japanese')
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(left, right) if b.tag == 'equal')
            corroborated = (min(len(left), len(right)) >= 40 and min(len(set(left)), len(set(right))) >= 10 and
                            matched >= min(len(left), len(right))*.9 and matched >= max(len(left), len(right))*.65)
            attempts.append(dict(url=url, identity=actual, title_match=title_match, artist_match=artist_match,
                                 corroborated=corroborated))
            if text and mismatch is not None and (title_match and (artist_match or partial_credit) or corroborated) and not short_lyric_text(text, duration):
                if partial_credit and not artist_match:
                    mismatch.append('performers')
                readings = [dict(after=len(comparison_key(page.text[:span['stop']])),
                                 text=comparison_key(span['reading'])) for span in page.spans
                            if comparison_key(span['reading'])] if comparison_key(page.text) == comparison_key(''.join(text)) else []
                return text, dict(status='candidate', provider='utaten', publisher='utaten', language=text_language('\n'.join(text)),
                    url=url, track=actual.get('title'), artist=actual.get('artist'), attempts=attempts,
                    text_sha256=hashlib.sha256(page.text.encode()).hexdigest(), excluded_credit_lines=excluded,
                    text_transform='published_ruby_body_without_readings', recording_mismatches=mismatch,
                    provider_timestamps_used=False, reading_annotations=readings)
    return None, dict(status='no_matching_published_text', provider='utaten', attempts=attempts)


def structured_recordings(document):
    """Read schema.org recording data, never the surrounding editorial text."""
    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.recordings, self.script = [], None
        def handle_starttag(self, tag, attrs):
            if tag == 'script' and dict(attrs).get('type') == 'application/ld+json':
                self.script = ''
        def handle_data(self, data):
            if self.script is not None:
                self.script += data
        def handle_endtag(self, tag):
            if tag != 'script' or self.script is None:
                return
            try:
                data = json.loads(self.script)
                records = data if isinstance(data, list) else data.get('@graph', [data])
                self.recordings.extend(r for r in records if isinstance(r, dict) and r.get('@type') == 'MusicRecording')
            except (ValueError, AttributeError, TypeError):
                pass  # Another script may contain the recording's valid data.
            self.script = None
    page = Page()
    page.feed(document)
    return page.recordings


def read_lyric_catalogue(url):
    """Discard repeated biographies/editorial data before caching search metadata."""
    with httpx.stream('GET', url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 16*1024*1024:
                raise ValueError('Lyric catalogue response exceeds 16 MiB')
    data = json.loads(body).get('data') or {}
    rows = []
    for row in (data.get('global_song_matches') or []) + (data.get('artist_songs') or []):
        if not isinstance(row, dict):
            continue
        credit = row.get('artist')
        artists = [credit] if isinstance(credit, dict) else credit if isinstance(credit, list) else []
        rows.append(dict(id=row.get('_id'), title=row.get('song_name'), path=row.get('public_song_url'),
                         artists=[a.get('artist_name') for a in artists if isinstance(a, dict)]))
    return dict(rows=rows, response_sha256=hashlib.sha256(body).hexdigest(), response_bytes=len(body))


def fetch_structured_lyrics(song, cache, duration=None, source_title=None):
    """Read a publisher's explicit lyric field with the existing identity matcher."""
    from .lyric_readings import read_page
    if not song.get('title') or not song.get('artist'):
        return None, dict(status='missing_identity', provider='lyricsweb')
    # This endpoint searches song names and artist names separately. Combining
    # them returns an artist catalogue, which can omit the requested song.
    # The provider accepts 2–120 characters. Keep short/long work names intact
    # for matching; query a declared alias or the artist's catalogue instead.
    query = next((name for key in ('title', 'artist') for name in reversed(song_names(song, key))
                  if 2 <= len(name) <= 120), None)
    if query is None:
        return None, dict(status='unsupported_query', provider='lyricsweb')
    url = 'https://lyricsweb.com/api/v1/user/search-result?'+urlencode(dict(query=query))
    packet = cached_lyric_request('lyricsweb', dict(url=url, representation='candidate_metadata_v1'), cache,
                                  lambda: read_lyric_catalogue(url))
    rows, links, publishers = [], {}, {}
    for row in packet['rows']:
        if not row.get('id'):
            continue
        path = row.get('path') or ''
        if not isinstance(path, str) or not re.fullmatch(r'/song/[\w%/-]+', path):
            continue
        key = str(row['id'])
        links[key] = 'https://lyricsweb.com'+path
        rows.append(dict(trackId=key, trackTitle=row.get('title'),
            artists=[dict(artistName=a) for a in row['artists']], album={}))
    def lyrics(row):
        url = links[row['trackId']]
        document = cached_lyric_request('lyricsweb', url, cache, lambda: read_page(url))
        for recording in structured_recordings(document):
            credit = recording.get('byArtist') or {}
            credits = [credit] if isinstance(credit, dict) else credit if isinstance(credit, list) else []
            artists = [a.get('name') for a in credits if isinstance(a, dict)]
            if (not identity(recording.get('name'), title=True) & identity(row['trackTitle'], title=True) or
                    not any(artist_credit_matches(a['artistName'], artists, partial=True) for a in row['artists'])):
                continue
            composition = recording.get('recordingOf')
            body = composition.get('lyrics') if isinstance(composition, dict) else None
            if not isinstance(body, dict):
                continue
            text = body.get('text')
            if not isinstance(text, str) or len(text) > 30000:
                continue
            provider = body.get('provider')
            publisher = provider.get('name') if isinstance(provider, dict) else None
            publishers[url] = publisher.casefold() if isinstance(publisher, str) and publisher else 'lyricsweb'
            # This publisher serializes line breaks inside the JSON text field.
            return text.replace('\\n', '\n')
        return None
    lines, report = published_candidate(rows, song, source_title, read_lyrics=lyrics,
        allow_recording_mismatch=True, duration=duration, source_url=links.get)
    if lines:
        report.update(publisher=publishers[report['url']],
                      text_transform='schema.org.lyrics_text_credit_metadata_removed')
    return lines, dict(report, provider='lyricsweb')


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
                recording = next((r.get('recording_identity') or {} for r in
                    report.get('crosscheck', {}).get('unavailable_sources', [])
                    if r.get('provider') == 'youtube_music' and
                    (r.get('recording_identity') or {}).get('video_id') == metadata.get('id')), {})
                confirmed_alias = recording.get(key)
                if confirmed_alias and not identity(confirmed_alias, title=key == 'title') & identity(song.get(key), title=key == 'title'):
                    song[key] = confirmed_alias
                    description_text = None
                else:
                    description_text = read_description([key], retry=True)
                    if parsed is None and metadata_failure and not metadata_retries:
                        description_text = read_description([key], retry=True)
                if song == before:
                    continue
                blocked = {r['provider']: (None, r) for r in report.get('crosscheck', {}).get('unavailable_sources', [])
                           if r.get('provider') and (r.get('error') or r['provider'] == 'youtube_music')}
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
        # Watch-page "artists" can merely repeat a video's upload channel.
        # Only recover missing credits; never replace an explicit performer.
        if (not lines and not request.get('artist') and not video_song_info(metadata).get('artist')
                and not (parsed and parsed.artist) and (not song.get('artist') or
                artist_credit_matches(song['artist'], [metadata.get('uploader'), metadata.get('channel')]))):
            candidate, recovery = catalogue_lyric_identity(song, duration, cache)
            original_report, original_song = report, dict(song)
            if candidate:
                song.update({key:candidate[key] for key in ('title','artist')})
                reused = {r['provider']:(None,r) for r in original_report.get('crosscheck', {}).get('unavailable_sources', [])
                          if r.get('provider') and (r.get('error') or r['provider'] == 'youtube_music')}
                lines, report = fetch_lyric_consensus(song, metadata, duration, cache, None, True, prefetched=reused)
                if not lines:
                    recovery = dict(recovery, rejected_lookup=report)
                    lines, report = None, original_report
                    song.clear()
                    song.update(original_song)
            report = dict(report, identity_recovery=dict(method='catalogue_title_and_duration',
                          catalogue=recovery, recovered=bool(lines), previous_lookup=original_report,
                          original={k:original_song.get(k) for k in ('title','artist')},
                          resolved={k:song.get(k) for k in ('title','artist')}))
        if report.get('language') == 'Chinese' and metadata_language == 'Cantonese' and not request.get('language'):
            report = dict(report, language='Cantonese')
        if metadata_language and report.get('language') and metadata_language != report['language']:
            report = dict(report, metadata_language_corrected=dict(original=metadata_language, resolved=report['language']))
        if metadata_error:
            report = dict(report, metadata_error=metadata_error)
        if metadata_retries:
            report = dict(report, metadata_retries=metadata_retries)
        (cache.root/'lyrics-first-lookup.json').write_text(json.dumps(report, ensure_ascii=False)+'\n')
        if not lines:
            return None
        song['language'] = report.get('language')
        provenance = dict(kind='online_lyric_text', lyric_lookup=report,
                          text_transform=report.get('text_transform', 'whitespace_only'))
    language = song.get('language') or text_language(''.join(lines))
    # Han script does not distinguish Mandarin from Cantonese. Retain a
    # specific recording tag when it is compatible with the published text.
    if language == 'Chinese' and metadata_language == 'Cantonese' and not request.get('language'):
        language = 'Cantonese'
    if len(lines) > 1000 or any(not line.strip() or len(line) > 200 for line in lines):
        return None
    return dict(song, id='song', source_url=request['url'], language=language, lines=lines,
                provenance=dict(provenance, needs_review=True, route='lyrics_first', independent_asr_coverage=None,
                                **({'metadata_error': metadata_error} if metadata_error else {})))


def prepared_lyrics_identity(url, metadata):
    """Bind split-host lookup to the same media metadata and lookup implementation."""
    root = Path(__file__).parent
    return dict(url=url, metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                code_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('lyric_lookup.py', 'lyric_consensus.py', 'song_sources.py', 'source_captions.py',
                                          'lyric_readings.py')})


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
        full_credit = any(artist_credit_matches(name, [row.get('artistName')]) for name in song_names(song, 'artist'))
        partial_credit = allow_recording_mismatch and any(artist_credit_matches(name, [row.get('artistName')], partial=True)
                                                          for name in song_names(song, 'artist'))
        if (not any(identity(name, title=True) & identity(row.get('trackName'), title=True) for name in song_names(song, 'title')) or
                not (full_credit or partial_credit) or row.get('instrumental') is not False):
            continue
        if not allow_recording_mismatch and version_tags(song['title']) != version_tags(row.get('trackName')):
            continue
        mismatch = (recording_mismatches(song, source_title, row.get('trackName'), row.get('albumName'))
                    if allow_recording_mismatch else ['live_performance'] if performance and performance not in
                    {live_performance(row.get(field)) for field in ('trackName', 'albumName')} else [])
        if mismatch is None or mismatch and not allow_recording_mismatch:
            continue
        if not full_credit:
            mismatch.append('performers')
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
            lines = clean_lines
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
