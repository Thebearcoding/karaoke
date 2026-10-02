"""Isolated optional Qwen alignment evidence, with timestamp repair disabled."""
import argparse
import json
from pathlib import Path
import sys

from subtitles import highlight_spans, highlight_coverage_errors

ROOT = Path(__file__).resolve().parent


def raw_units(cue, prediction, begin):
    spans = list(highlight_spans(cue['text'], 'Chinese'))
    if len(prediction) != len(spans) or any(p.text != cue['text'][a:b] for (a, b), p in zip(spans, prediction)):
        raise ValueError('Independent aligner changed the displayed units')
    # Preserve invalid/collapsed intervals so the caller can reject them.
    return [dict(first=a, stop=b, text=p.text, start=p.start_time+begin, end=p.end_time+begin)
            for (a, b), p in zip(spans, prediction)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cues', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    # Keep the optional SDK's Transformers version outside the main worker.
    sys.path.insert(0, str(ROOT/'research/subtitles-2026-09-18/python-deps'))
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from qwen_asr import Qwen3ForcedAligner
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model = Qwen3ForcedAligner.from_pretrained(str(ROOT/'models/alignment/qwen3-forced-aligner-0.6b'),
        dtype=torch.float32, device_map=args.device, attn_implementation='sdpa', local_files_only=True)
    model.aligner_processor.fix_timestamp = lambda times: times.tolist()

    def encode(text, language):
        words = [text[a:b] for a, b in highlight_spans(text, 'Chinese')]
        return words, '<|audio_start|><|audio_pad|><|audio_end|>'+'<timestamp><timestamp>'.join(words)+'<timestamp><timestamp>'

    model.aligner_processor.encode_timestamp = encode
    cues = json.loads(args.cues.read_text())
    affected = sorted({e['line']-1 for e in highlight_coverage_errors(cues, 'Chinese')})
    result = {i: [] for i in affected}
    for source in (args.audio, args.original):
        wave, rate = sf.read(source, dtype='float32', always_2d=True)
        divisor = np.gcd(rate, 16000)
        wave = resample_poly(wave.mean(axis=1), 16000//divisor, rate//divisor).astype('float32')
        for index in affected:
            cue = cues[index]
            begin, end = max(0., cue['aligned_start']-.6), min(len(wave)/16000, cue['aligned_end']+.6)
            if end-begin > 30:
                continue
            prediction = list(model.align(audio=(wave[round(begin*16000):round(end*16000)], 16000),
                text=cue['text'], language='Chinese')[0])
            result[index].append(raw_units(cue, prediction, begin))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
