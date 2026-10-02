"""Use publisher-supplied subtitle text from the same recording as lyric evidence."""
import hashlib
import inspect
import json
import math
import re
import time
from urllib.parse import parse_qs, urlsplit

import httpx

from .anchors_from_asr import text_units
from .job_schema import retry_after_seconds, song_language


def manual_track(metadata, language):
    if language is None:
        return None
    for tag, tracks in sorted((metadata.get('subtitles') or {}).items()):
        if song_language(tag) != language:
            continue
        for track in tracks:
            url = track.get('url', '')
            parsed = urlsplit(url)
            query = parse_qs(parsed.query)
            if (track.get('ext') == 'json3' and parsed.scheme == 'https' and
                    parsed.netloc in ('www.youtube.com', 'youtube.com') and
                    parsed.path == '/api/timedtext' and query.get('v') == [metadata.get('id')] and
                    query.get('fmt') == ['json3'] and song_language((query.get('lang') or [None])[0]) == language and
                    not query.get('kind') and not query.get('tlang')):
                return dict(language=tag, url=url)
    return None


def read_document(url):
    with httpx.stream('GET', url, timeout=20, follow_redirects=False,
                      headers={'User-Agent': 'Mozilla/5.0'}) as response:
        response.raise_for_status()
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > 2*1024*1024:
                raise ValueError('Subtitle response exceeds 2 MiB')
    return json.loads(body)


def fetch_manual_captions(metadata_path, language, cache):
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    track = manual_track(metadata, language)
    if track is None:
        return None
    path = cache.root/'source-captions.json'
    name = 'fetch_source_captions'
    try:
        old = json.loads(path.read_text())
        if old.get('error') and old.get('retry_at') is not None and old['retry_at'] <= time.time():
            (cache.directory/(name+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(name, inputs=[metadata_path], outputs=[path], config=dict(
            track=track, implementation=inspect.getsource(read_document))) as reused:
        if not reused:
            packet = dict(video_id=metadata['id'], language=track['language'])
            try:
                packet['response'] = read_document(track['url'])
            except (httpx.HTTPError, ValueError) as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                transient = isinstance(error, httpx.TransportError) or status == 429 or (status or 0) >= 500
                packet.update(error=dict(type=type(error).__name__, status=status),
                    retry_at=time.time()+max(300, retry_after_seconds(error) or 0) if transient else None)
            path.write_text(json.dumps(packet, ensure_ascii=False)+'\n')
    return path


def caption_candidate(packet, transcription, language, duration):
    """Check source text against audio; caption display times are never word times."""
    from rapidfuzz.distance import Levenshtein
    from .song_sources import transcript_lines
    if packet.get('error'):
        return None, dict(status='unavailable', error=packet['error'], retry_at=packet.get('retry_at'))
    if song_language(packet.get('language')) != language:
        return None, dict(status='wrong_caption_language')
    texts = []
    try:
        for event in packet['response']['events']:
            text = ''.join(s.get('utf8', '') for s in event.get('segs', []))
            if not text.strip():
                continue
            start = event['tStartMs']/1000
            end = start+event['dDurationMs']/1000
            if (event.get('aAppend') or not all(math.isfinite(t) for t in (start, end)) or
                    not 0 <= start < end <= duration+1):
                raise ValueError('Unsupported or out-of-range caption event')
            if re.search(r'\[[^\]\n]+\]', text):
                # Closed captions may label speakers or describe sounds. Do not
                # sing those labels or guess which bracketed words to delete;
                # retain the audio transcript instead of this optional source.
                return None, dict(status='unsupported_caption_annotations')
            texts.append(dict(text=text))
        # Caption display events can contain only musical notes or punctuation.
        # Wrap the complete text so those symbols join an adjacent lyric line.
        lines = transcript_lines(dict(segments=texts), language)
        if not lines or len(lines) > 1000 or any(len(line) > 200 for line in lines):
            raise ValueError('Invalid lyric length')
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None, dict(status='invalid_caption_document')
    expected = text_units(''.join(lines), language)
    observed = [unit for word in transcription.get('words', []) for unit in text_units(word['word'], language)]
    matched = sum(b.src_end-b.src_start for b in Levenshtein.opcodes(expected, observed) if b.tag == 'equal')
    support, coverage = matched/max(1, len(expected)), matched/max(1, len(observed))
    accepted = bool(lines) and min(support, coverage) >= .55
    report = dict(status='accepted' if accepted else 'insufficient_audio_support',
        text_support=support, audio_word_coverage=coverage, language=packet['language'],
        video_id=packet['video_id'], response_sha256=hashlib.sha256(
            json.dumps(packet['response'], sort_keys=True).encode()).hexdigest())
    return lines if accepted else None, report
