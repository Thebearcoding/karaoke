"""Obtain an ASR draft and coarse word times from an existing vocal preview."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from openai import OpenAI
from job_schema import song_language
from stage_progress import progress_items, report_progress


def repetition_loop(data):
    """Detect decoder loops, not ordinary repeated lyric lines or short vocalizations."""
    segments = data.get('segments') or []
    texts = [''.join(c for c in s['text'].casefold() if c.isalnum()) for s in segments]
    # A decoder can emit the same phrase in many tiny segments. Require collapsed
    # timing as corroboration so a normally timed repeated chorus stays valid.
    collapsed = sum('start' in s and 'end' in s and s['end']-s['start'] <= .025 for s in segments)
    candidates = texts + ([''.join(texts)] if collapsed > len(segments)/2 else [])
    for text in candidates:
        if len(text) >= 40 and re.search(r'(.{1,32}?)\1{7,}', text):
            return True
    return False


def needs_recovery(data):
    text = ''.join(s.get('text', '') for s in data.get('segments') or []) or data.get('text', '')
    return not any(c.isalnum() for c in text) or repetition_loop(data)


def transcribe(client, audio, language, hint, output):
    options = {} if language == 'auto' else {'language': language}

    def recognize(path, model='whisper-1', instruction=None):
        with path.open('rb') as stream:
            return client.audio.transcriptions.create(file=stream, model=model, **options,
                prompt=instruction if instruction is not None else hint,
                temperature=0, response_format='verbose_json' if model == 'whisper-1' else 'json',
                **({'timestamp_granularities': ['word', 'segment']} if model == 'whisper-1' else {})).model_dump()

    data = recognize(audio)
    recovery = dict(triggered=False, chunks=[])
    if not needs_recovery(data):
        return data, recovery
    report_progress(detail='识别结果异常，分段重试')
    (output/'initial-transcription.json').write_text(json.dumps(data, ensure_ascii=False)+'\n')
    # An unusable transcript's language detection is also untrusted. Preserve an
    # explicit user language, but let recovery re-detect automatic requests.
    duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=nw=1:nk=1', str(audio)], text=True))
    fallback_model = os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe')
    # Keep full recording context for text: independent chunks can omit a split
    # syllable or duplicate the opening of the next lyric. Chunks only locate words.
    for instruction in (None,
            'Transcribe only intelligible sung or spoken words. Omit wordless humming, '
            'sustained nonverbal vowels, and instrumental sounds. Do not expand held '
            'notes into repeated syllables. If there are no intelligible words, return an empty transcript.'):
        text = recognize(audio, fallback_model, instruction)['text'].strip()
        segments = [dict(start=0., end=duration, text=text, timing_source='recording_bounds')]
        if not needs_recovery(dict(segments=segments)):
            break
    if needs_recovery(dict(segments=segments)):
        raise ValueError('歌词文本重试仍有重复异常或为空，无法生成可靠歌词')
    (output/'recovered-text.json').write_text(json.dumps(segments, ensure_ascii=False)+'\n')

    def chunk(start):
        length = min(30., duration-start)
        path = output/f'chunk-{start:05d}.mp3'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(start), '-i', str(audio),
            '-t', str(length), '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', str(path)], check=True)
        try:
            part = recognize(path)
            diagnostic = dict(start=start, end=start+length, whisper=part)
            fallback = needs_recovery(part)
            if fallback:
                (output/f'chunk-{start:05d}.json').write_text(json.dumps(diagnostic, ensure_ascii=False)+'\n')
            # Do not give the text-only transcript invented word times. Unresolved
            # lyrics use neighboring anchors in the later acoustic forced alignment.
            words = [] if fallback else [dict(w, start=w['start']+start, end=min(length, w['end'])+start)
                for w in part.get('words', []) if 0 <= w['start'] < length and w['end'] > w['start']]
            return words, dict(start=start, end=start+length, word_times_discarded=fallback,
                               language=song_language(part.get('language')))
        finally:
            path.unlink(missing_ok=True)

    starts = list(range(0, math.ceil(duration), 30))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(chunk, start) for start in starts]
        parts = [future.result() for future in progress_items(futures, 'chunks')]
    languages = Counter()
    for words, record in parts:
        if record['language']:
            languages[record['language']] += sum(c.isalnum() for w in words for c in w['word'])
    detected = languages.most_common(1)[0][0] if languages.total() else None
    data['language'] = song_language(language) if language != 'auto' else detected
    data.update(duration=duration, text=text, segments=segments, words=[w for words, _ in parts for w in words])
    return data, dict(triggered=True, text_model=fallback_model, language_evidence=dict(languages),
                      chunks=[record for _, record in parts])


def filter_silent_transcription(data, audio, sample_rate):
    """Reject ASR hallucinations in digital silence, preserving even quiet speech."""
    filtered, rejected = dict(data), []
    for kind in ('segments', 'words'):
        filtered[kind] = []
        for item in data.get(kind) or []:
            start, end = item['start'], item['end']
            if not all(math.isfinite(t) for t in (start, end)) or end < start:
                reason = 'invalid_time_range'
            else:
                # Include nearby audio for imprecise or zero-duration ASR times.
                left = max(0, math.floor((start-.25)*sample_rate))
                right = min(len(audio), math.ceil((end+.25)*sample_rate))
                reason = ('outside_audio' if right <= left else
                          'digital_silence' if not audio[left:right].any() else None)
            if reason:
                rejected.append(dict(kind=kind, reason=reason, item=item))
            else:
                filtered[kind].append(item)
    filtered['text'] = ''.join(item.get('text', '') for item in filtered['segments'])
    return filtered, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--language', default='auto')
    parser.add_argument('--hint', default='')
    parser.add_argument('--source-offset', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    client = OpenAI(timeout=240., max_retries=1)
    language = {'Chinese': 'zh', 'Japanese': 'ja', 'English': 'en'}.get(args.language, args.language)
    args.output.mkdir(parents=True)
    data, recovery = transcribe(client, args.audio, language, args.hint, args.output)
    for key in ['words', 'segments']:
        for item in data.get(key, []):
            item['start'] += args.source_offset
            item['end'] += args.source_offset
    (args.output/'transcription.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    metadata = {'service': 'OpenAI Audio Transcriptions API', 'model': 'whisper-1',
        'uploaded_audio': str(args.audio.resolve()),
        'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        'language': args.language, 'hint': args.hint, 'seconds': time.perf_counter()-started,
        'source_offset_seconds': args.source_offset,
        'recovery': recovery,
        'status': 'ASR draft; requires lyric verification and timing evaluation'}
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
