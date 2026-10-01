"""Force-align lyrics to vocal audio without video or external subtitle times."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_closing
import soundfile as sf
import stable_whisper
from stable_whisper import timing as whisper_timing
import torch
from whisper.timing import dtw_cpu

from subtitles import lyric_cues, normalized, write_captions
from stage_progress import progress_items, report_progress


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_number(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def alignment_blocks(anchors, audio):
    """Keep ASR-supported verses apart; unresolved runs share adjacent context."""
    count = len(anchors)
    valid = lambda i: anchors[i]['status'] == 'candidate'
    first = 0
    blocks = []
    while first < count:
        stop = first+1
        if not valid(first):
            while stop < count and not valid(stop):
                stop += 1
        else:
            while stop < count and valid(stop):
                if anchors[stop]['end']-anchors[first]['start'] > 18 or anchors[stop]['start']-anchors[stop-1]['end'] > 4:
                    break
                stop += 1
        context_first = first-1 if first and valid(first-1) else first
        context_stop = stop+1 if stop < count and valid(stop) else stop
        if valid(first) and context_first < first and anchors[first]['start']-anchors[context_first]['end'] > 4:
            context_first = first
        if valid(stop-1) and context_stop > stop and anchors[stop]['start']-anchors[stop-1]['end'] > 4:
            context_stop = stop
        begin = max(0., anchors[context_first]['start']-.6) if valid(context_first) else 0.
        # ASR often stops before a held final syllable; retain its acoustic tail.
        tail_padding = 3.0 if context_stop == count else .6
        end = min(len(audio)/16000, anchors[context_stop-1]['end']+tail_padding) if valid(context_stop-1) else len(audio)/16000
        # A global loudness threshold can miss quiet opening/closing lyrics.
        # Keep the anchor context; never trim it to estimated voice bounds.
        if not np.isfinite([begin, end]).all() or round(begin*16000) >= round(end*16000):
            raise ValueError(f'Empty alignment audio for lyric lines {first+1}-{stop}: {begin}-{end}s')
        blocks.append((first, stop, context_first, context_stop, begin, end))
        first = stop
    return blocks


def refine_chinese_onsets(cues, raw_blocks, audio, language):
    """Remove a preceding held note absorbed by a single initial syllable.

    Only a long, single-character Chinese word with an internal quiet gap is
    eligible. Multi-syllable Japanese/English tokens need a phonetic model.
    """
    if language not in ('zh', 'Chinese'):
        return []
    frames = audio[:len(audio)//320*320].reshape(-1, 320)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = max(.008, np.quantile(rms, .9)*.12)
    active = binary_closing(rms > threshold, structure=np.ones(5))
    changes = []
    for block in raw_blocks:
        offset = block['source_range'][0]
        context = block['context'][0]
        segments = block['result']['segments']
        assert len(segments) == block['context'][1]-context
        for i in range(*block['range']):
            segment = segments[i-context]
            assert normalized(segment['text']) == normalized(cues[i]['text'])
            word = next(w for w in segment['words'] if w['word'].strip())
            char = word['word'].strip()
            if len(char) != 1 or not '\u4e00' <= char <= '\u9fff' or word['end']-word['start'] < 1.15:
                continue
            begin = round((offset+word['start'])/.02)
            end = round((offset+word['end'])/.02)
            edges = np.diff(np.r_[False, ~active[begin:end], False].astype(int))
            gaps = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if b-a >= 8 and b < end-begin-4]
            if not gaps:
                continue
            a, b = gaps[-1]
            onset = round((begin+b)*.02-.04, 3)
            if onset-cues[i]['start'] <= .3:
                continue
            changes.append({'line': i+1, 'forced_start': cues[i]['start'], 'refined_start': onset,
                            'quiet_gap': [(begin+a)*.02, (begin+b)*.02], 'rms_threshold': float(threshold)})
            cues[i]['forced_start'] = cues[i]['start']
            cues[i]['start'] = onset
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--model', default='large-v3-turbo')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--dynamic-heads', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Whisper's CPU DTW promotes to float64; transfer before promotion on MPS.
    if args.device == 'mps':
        whisper_timing.dtw = lambda matrix: dtw_cpu(matrix.detach().cpu().double().numpy())
    jobs = json.loads(args.jobs.read_text())
    model_root = Path(__file__).resolve().parent/'models/alignment/whisper'
    loading_start = time.perf_counter()
    report_progress(detail='加载模型')
    model = stable_whisper.load_model(args.model, device='cpu', download_root=str(model_root))
    # The sparse head-selection buffer is indexed on the host; MPS cannot move it.
    heads = model._buffers.pop('alignment_heads')
    model.to(args.device)
    model.register_buffer('alignment_heads', heads, persistent=False)
    loading = time.perf_counter()-loading_start
    print(json.dumps({'model_loaded_seconds': loading, 'model': args.model}), flush=True)
    for job in jobs:
        output = Path(job['output'])
        if output.exists():
            raise FileExistsError(output)
        lyrics = json.loads(Path(job['lyrics']).read_text())
        audio_info = sf.info(job['audio'])
        duration = audio_info.frames/audio_info.samplerate
        assert audio_info.samplerate == 16000 and audio_info.channels == 1
        audio, _ = sf.read(job['audio'], dtype='float32')
        language = {'Japanese': 'ja', 'Chinese': 'zh', 'English': 'en'}[lyrics['language']]
        blocks = [(0, len(lyrics['lines']), 0, len(lyrics['lines']), 0., duration)]
        if job.get('anchors'):
            anchors = json.loads(Path(job['anchors']).read_text())
            assert [cue['text'] for cue in anchors] == lyrics['lines']
            blocks = alignment_blocks(anchors, audio)
        started = time.perf_counter()
        cues, raw_blocks, units = [], [], []
        for first, stop, context_first, context_stop, begin, end in progress_items(blocks, 'blocks'):
            print(json.dumps({'song': Path(job['lyrics']).stem, 'lines': [first+1, stop],
                              'window': [begin, end]}), flush=True)
            result = model.align(audio[round(begin*16000):round(end*16000)],
                '\n'.join(lyrics['lines'][context_first:context_stop]), language=language,
                original_split=True, regroup=False, token_step=80, max_word_dur=4.,
                word_dur_factor=3., nonspeech_skip=5., suppress_silence=True,
                dynamic_heads=args.dynamic_heads, verbose=None)
            if result is None:
                raise ValueError(f'Alignment returned no result for {job["lyrics"]}')
            local_units = [{'text': word.word, 'start': round(word.start+begin, 3),
                            'end': round(word.end+begin, 3), 'probability': word.probability}
                           for word in result.all_words()]
            local_cues = lyric_cues(lyrics['lines'][context_first:context_stop], local_units, duration)
            cues.extend(local_cues[first-context_first:stop-context_first])
            units.extend(local_units)
            raw_blocks.append({'range': [first, stop], 'context': [context_first, context_stop],
                               'source_range': [begin, end], 'result': result.to_dict()})
        if args.device == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter()-started
        refinements = refine_chinese_onsets(cues, raw_blocks, audio, language)
        for cue in cues:
            cue['source'] = 'stable-ts forced alignment of supplied lyrics to vocal audio'
        summary = write_captions(cues, output)
        summary['zero_duration_units'] = sum(unit['end'] <= unit['start'] for unit in units)
        summary['unit_count'] = len(units)
        (output/'raw.json').write_text(json.dumps(raw_blocks, ensure_ascii=False, indent=2, default=json_number)+'\n')
        (output/'units.json').write_text(json.dumps(units, ensure_ascii=False, indent=2, default=json_number)+'\n')
        report = {'model': args.model, 'device': args.device, 'dtype': 'float32',
            'model_loading_seconds': loading, 'alignment_seconds': elapsed,
            'audio': str(Path(job['audio']).resolve()), 'audio_sha256': sha(job['audio']),
            'lyrics': str(Path(job['lyrics']).resolve()), 'lyrics_sha256': sha(job['lyrics']),
            'model_sha256': sha(model_root/f'{args.model}.pt'), 'runner_sha256': sha(__file__),
            'dynamic_heads': args.dynamic_heads,
            'onset_refinements': refinements,
            'anchors': job.get('anchors'), 'anchors_sha256': sha(job['anchors']) if job.get('anchors') else None,
            'packages': {name: version(name) for name in ['stable-ts', 'openai-whisper', 'torch']},
            'summary': summary, 'status': 'Audio-only candidate; requires timing review'}
        (output/'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_number)+'\n')
        print(json.dumps({'lyrics': job['lyrics'], 'seconds': elapsed, **summary}, default=json_number), flush=True)


if __name__ == '__main__':
    main()
