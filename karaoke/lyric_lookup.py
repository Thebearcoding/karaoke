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
from .song_sources import transcript_lines
from .source_captions import read_document


def names(value):
    value = unicodedata.normalize('NFKC', value if isinstance(value, str) else '').strip()
    result = [value] if value else []
    # Music metadata often appends an English translation to a Chinese name.
    # Version labels are not translations: never strip "Live", "Remix", etc.
    match = re.fullmatch(r'([\u3400-\u9fff]+)\s+([A-Za-z][\x20-\x7e]*)', value)
    if match and not re.search(r'\b(live|remix|acoustic|instrumental|karaoke|remaster\w*)\b',
                               match[2], re.I):
        result.append(match[1])
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


def lyric_candidate(packet, song, transcription, duration, *, audio_path=None, source_title=None):
    report = {key: value for key, value in packet.items() if key != 'results'}
    if packet.get('status') != 'ok':
        return None, report
    performance = live_performance(source_title)
    if performance:
        report['required_live_performance'] = performance
    language = song['language']
    observed = [u for w in transcription.get('words', []) for u in text_units(w['word'], language)]
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
            lines = transcript_lines(dict(text=text), language)
            expected = text_units(''.join(lines), language)
            matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
            support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
            if not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines) or min(support, coverage) < .55:
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        candidates.append((min(support, coverage), -difference, lines, dict(
            id=row['id'], track=row['trackName'], artist=row['artistName'], album=row.get('albumName'),
            duration=row['duration'], text_support=support, audio_word_coverage=coverage,
            duration_basis=basis or dict(kind='recording', recording_duration=duration),
            text_sha256=hashlib.sha256(text.encode()).hexdigest())))
    if not candidates:
        return None, dict(report, status='no_supported_recording')
    best = max(candidates, key=lambda c: c[:2])
    return best[2], dict(report, status='accepted', provider='lrclib', **best[3])
