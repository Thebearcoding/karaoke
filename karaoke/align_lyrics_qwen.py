"""Align supplied lyrics to an existing separated vocal recording on this Mac."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'research/subtitles-2026-09-18/python-deps'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from qwen_asr import Qwen3ForcedAligner

from subtitles import lyric_cues, write_captions


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lyrics', required=True, type=Path)
    parser.add_argument('--vocals', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--anchors', type=Path)
    parser.add_argument('--max-span', type=float, default=28.)
    parser.add_argument('--context', type=float, default=3.)
    parser.add_argument('--context-lines', type=int, default=2)
    parser.add_argument('--trim-edges', action='store_true',
                        help='Trim leading/trailing music when anchors already locate the vocals')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.backends.mps.is_available(), 'Run with access to the local Metal GPU'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    lyrics = json.loads(args.lyrics.read_text())
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(args.vocals),
        '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'])
    audio = np.frombuffer(raw, dtype='<f4').copy()
    model_path = ROOT/'models/alignment/qwen3-forced-aligner-0.6b'
    start = time.perf_counter()
    model = Qwen3ForcedAligner.from_pretrained(str(model_path), dtype=torch.float32,
        device_map='mps', attn_implementation='sdpa', local_files_only=True)
    torch.mps.synchronize()
    loading = time.perf_counter()-start
    raw_timestamps = []
    active_block = {}
    original_parse = model.aligner_processor.parse_timestamp
    def capture_parse(words, timestamps):
        raw_timestamps.append({**active_block, 'words': words,
            'milliseconds_before_monotonic_repair': timestamps.tolist()})
        return original_parse(words, timestamps)
    model.aligner_processor.parse_timestamp = capture_parse
    duration = len(audio)/16000
    count = len(lyrics['lines'])
    groups = [(0, count, 0, count, 0., duration)]
    if args.anchors:
        anchors = json.loads(args.anchors.read_text())
        assert [c['text'] for c in anchors] == lyrics['lines']
        groups = []
        left = 0
        for right in range(1, len(anchors)+1):
            if right == len(anchors) or anchors[right]['end']-anchors[left]['start'] > args.max_span:
                context_first = max(0, left-args.context_lines)
                context_stop = min(count, right+args.context_lines)
                begin = max(0., anchors[context_first]['start']-args.context) if context_first or args.trim_edges else 0.
                end = min(duration, anchors[context_stop-1]['end']+args.context) if context_stop < count or args.trim_edges else duration
                groups.append((left, right, context_first, context_stop, begin, end))
                left = right
    start = time.perf_counter()
    units = []
    cues = []
    for first, stop, context_first, context_stop, begin, end in groups:
        active_block.clear()
        active_block.update({'first_line': first, 'stop_line': stop,
            'context_first_line': context_first, 'context_stop_line': context_stop,
            'source_begin': begin, 'source_end': end})
        segment = audio[round(begin*16000):round(end*16000)]
        result = model.align(audio=(segment, 16000), text='\n'.join(lyrics['lines'][context_first:context_stop]),
                             language=lyrics['language'])[0]
        local_units = [{'text': item.text, 'start': round(item.start_time+begin, 3),
                        'end': round(item.end_time+begin, 3)} for item in result]
        units.extend(local_units)
        local_cues = lyric_cues(lyrics['lines'][context_first:context_stop], local_units, duration)
        cues.extend(local_cues[first-context_first:stop-context_first])
        print(json.dumps({'completed_lines': [first+1, stop], 'source_range': [begin, end]}), flush=True)
    torch.mps.synchronize()
    elapsed = time.perf_counter()-start
    assert [c['text'] for c in cues] == lyrics['lines']
    summary = write_captions(cues, args.output)
    (args.output/'aligned-units.json').write_text(json.dumps(units, ensure_ascii=False, indent=2)+'\n')
    (args.output/'raw-timestamps.json').write_text(json.dumps(raw_timestamps, ensure_ascii=False, indent=2)+'\n')
    metadata = {'model': 'Qwen/Qwen3-ForcedAligner-0.6B', 'model_path': str(model_path),
        'model_download_manifest_sha256': sha(model_path/'download-manifest.json'),
        'device': 'mps', 'dtype': 'float32', 'attention': 'sdpa',
        'lyrics': str(args.lyrics.resolve()), 'lyrics_sha256': sha(args.lyrics),
        'vocals': str(args.vocals.resolve()), 'vocals_sha256': sha(args.vocals),
        'duration_seconds': duration, 'load_seconds': loading, 'alignment_seconds': elapsed,
        'anchors': str(args.anchors) if args.anchors else None,
        'anchors_sha256': sha(args.anchors) if args.anchors else None,
        'groups': groups,
        'trim_edges': args.trim_edges,
        'runner_sha256': sha(__file__), 'summary': summary,
        'status': 'Automatic candidate; no human timestamp acceptance claimed'}
    (args.output/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
