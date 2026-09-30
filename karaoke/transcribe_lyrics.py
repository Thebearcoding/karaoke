"""Obtain an ASR draft and coarse word times from an existing vocal preview."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from openai import OpenAI


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
    language_options = {} if language == 'auto' else {'language': language}
    with args.audio.open('rb') as stream:
        result = client.audio.transcriptions.create(file=stream, model='whisper-1',
            **language_options, prompt=args.hint, temperature=0,
            response_format='verbose_json', timestamp_granularities=['word', 'segment'])
    args.output.mkdir(parents=True)
    data = result.model_dump()
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
        'status': 'ASR draft; requires lyric verification and timing evaluation'}
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
