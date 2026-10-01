"""Force-align lyrics to vocal audio without video or external subtitle times."""

import argparse
from copy import deepcopy
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

from lyric_progress import map_words
from subtitles import lyric_cues, normalized, presentation_cues, write_captions
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


def recover_changed_text(result, lines, duration, realign):
    """Retry a malformed alignment, never edit its words to fit the lyrics."""
    expected = [normalized(line) for line in lines]

    def matches(candidate):
        return candidate is not None and [normalized(''.join(w['word'] for w in s['words']))
                for s in candidate['segments']] == expected

    if matches(result):
        return result, None
    candidate = realign()
    report = dict(original_result=result, retry_result=candidate, accepted=False,
                  method='dynamic attention heads without inter-line gap padding')
    if matches(candidate):
        units = [dict(text=w['word'], start=w['start'], end=w['end'])
                 for s in candidate['segments'] for w in s['words'] if normalized(w['word'])]
        valid = all(np.isfinite([u['start'], u['end']]).all() and
                    0 <= u['start'] < u['end'] <= duration+.001 for u in units)
        valid = valid and all(a['start'] < b['start'] for a, b in zip(units, units[1:]))
        if valid:
            try:
                presentation_cues(lyric_cues(lines, units, duration), duration, lead=0, tail=0, bridge_gap=0)
            except ValueError:
                pass
            else:
                report['accepted'] = True
                return candidate, report
    return result, report


def recover_skipped_repeats(result, duration, align):
    """Check a skipped repetition using both its gap and neighbouring phrases.

    Attention can skip an interior refrain and leave the last copy untimed.
    A gap is only a search window: accepted times come from acoustic alignment.
    """
    result = deepcopy(result)
    segments, reports = result['segments'], []
    first = 0
    while first < len(segments):
        stop = first+1
        while stop < len(segments) and normalized(segments[stop]['text']) == normalized(segments[first]['text']):
            stop += 1
        run = segments[first:stop]
        first = stop
        # Keep this bounded to one missing copy in an otherwise timed run.
        if len(run) < 3 or run[-1]['start'] != run[-1]['end'] or any(
                s['start'] >= s['end'] for s in run[:-1]):
            continue
        index = max(range(len(run)-2), key=lambda i: run[i+1]['start']-run[i]['end'])
        left, right = run[index:index+2]
        if left['end'] >= right['start'] or right['end']-left['start']+1.2 > 45:
            continue
        report = dict(lines=[stop-len(run)+1, stop], gap=[left['end'], right['start']], accepted=False)
        reports.append(report)
        predictions = []
        for name, begin, end, count in (
                ('context', max(0., left['start']-.6), min(duration, right['end']+.6), 3),
                ('gap', max(0., left['end']-.3), min(duration, right['start']+.3), 1)):
            prediction = align(begin, end, '\n'.join([left['text'].strip()]*count))
            report[name] = dict(window=[begin, end], result=prediction)
            if prediction is None or len(prediction['segments']) != count or not prediction['segments'][count//2]['words']:
                break
            segment = deepcopy(prediction['segments'][count//2])
            for word in segment['words']:
                word['start'], word['end'] = round(word['start']+begin, 3), round(word['end']+begin, 3)
            segment['start'], segment['end'] = segment['words'][0]['start'], segment['words'][-1]['end']
            predictions.append(segment)
        if len(predictions) != 2:
            continue
        candidate, isolated = predictions
        words, evidence = candidate['words'], isolated['words']
        if (not words or len(words) != len(evidence) or
                normalized(''.join(w['word'] for w in words)) != normalized(left['text']) or
                [normalized(w['word']) for w in words] != [normalized(w['word']) for w in evidence] or
                any(not left['end'] < w['start'] < w['end'] < right['start'] for w in words) or
                any(b['start']-a['start'] < .015 for a, b in zip(words, words[1:]))):
            continue
        begin, end = report['gap']['window']
        agreement = [begin+.015 < b['start'] < b['end'] < end-.015 and
                     abs(a['start']-b['start']) <= .2 for a, b in zip(words, evidence)]
        # A boundary-clipped word cannot corroborate its own onset. Require
        # adjacent interior words and at least half of the phrase to agree.
        if sum(agreement) < max(2, (len(words)+1)//2) or not any(
                a and b for a, b in zip(agreement, agreement[1:])):
            continue
        report.update(accepted=True, corroborated_words=sum(agreement), original_segments=run)
        run = run[:-1]
        run.insert(index+1, candidate)
        segments[stop-len(run):stop] = run
    return result, reports


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


def stitch_blocks(lines, raw_blocks, duration, language):
    """Resolve crossings using existing predictions, without moving word times."""
    predictions, owners = [], [None]*len(lines)
    for block_index, block in enumerate(raw_blocks):
        first, stop = block['context']
        offset = block['source_range'][0]
        segments = block['result']['segments']
        units = [dict(text=w['word'], start=round(offset+w['start'], 3),
                      end=round(offset+w['end'], 3)) for s in segments for w in s['words']]
        cues = lyric_cues(lines[first:stop], units, duration)
        prediction = {}
        for index, (cue, segment) in enumerate(zip(cues, segments), first):
            try:
                words = map_words(cue, segment['words'], offset, language) if cue['status'] == 'candidate' else []
            except ValueError:
                words = []  # Retain the failure for the normal word-mapping gate.
            prediction[index] = (cue, words)
        predictions.append(prediction)
        for index in range(*block['range']):
            if owners[index] is not None:
                raise ValueError(f'Duplicate core lyric line: {index}')
            owners[index] = block_index
    if None in owners:
        raise ValueError('Raw alignment does not cover exactly the displayed lyrics')

    def conflicts(selection):
        selected = [predictions[owner][i] for i, owner in enumerate(selection)]
        bad = set()
        for i, ((left, left_words), (right, right_words)) in enumerate(zip(selected, selected[1:]), 1):
            if left_words and right_words and right_words[0]['start']-left_words[-1]['start'] < .015-1e-9:
                bad.add(i)
            try:
                # Share the downstream gate: repairing word starts must not
                # introduce invalid line ends or excessive overlap elsewhere.
                presentation_cues([left, right], duration, lead=0, tail=0, bridge_gap=0)
            except ValueError:
                bad.add(i)
        return bad

    repairs = []
    while bad := conflicts(owners):
        repaired = False
        for index in sorted(bad):
            # Prefer a joint prediction. If that introduces another crossing,
            # an adjoining window may already have a valid single-line result.
            for first, stop in ((index-1, index+1), (index-1, index), (index, index+1)):
                for owner in dict.fromkeys(owners[index-1:index+1]+list(range(len(predictions)))):
                    prediction = predictions[owner]
                    if not all(i in prediction and prediction[i][1] for i in range(first, stop)):
                        continue
                    words = [w for i in range(first, stop) for w in prediction[i][1]]
                    if any(b['start']-a['start'] < .015-1e-9 for a, b in zip(words, words[1:])):
                        continue
                    candidate = owners.copy()
                    candidate[first:stop] = [owner]*(stop-first)
                    if not conflicts(candidate) < bad:
                        continue
                    repairs.append(dict(lines=[first+1, stop],
                        previous_blocks=owners[first:stop], selected_block=owner,
                        context=raw_blocks[owner]['context'], source_range=raw_blocks[owner]['source_range']))
                    owners, repaired = candidate, True
                    break
                if repaired:
                    break
            if repaired:
                break
        if not repaired:
            break  # Never invent times or relax the downstream quality gate.

    # Preserve the selected raw prediction as well as the line timing: word and
    # phoneme stages must not accidentally read the old window for this line.
    stitched = []
    for index, owner in enumerate(owners):
        if index and owners[index-1] == owner:
            stitched[-1]['range'][1] = index+1
        else:
            stitched.append({**raw_blocks[owner], 'range': [index, index+1]})
    return [predictions[owner][i][0] for i, owner in enumerate(owners)], stitched, repairs


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
        raw_blocks, units = [], []
        for first, stop, context_first, context_stop, begin, end in progress_items(blocks, 'blocks'):
            print(json.dumps({'song': Path(job['lyrics']).stem, 'lines': [first+1, stop],
                              'window': [begin, end]}), flush=True)
            piece = audio[round(begin*16000):round(end*16000)]

            def align_piece(start, stop, text, **options):
                aligned = model.align(piece[round(start*16000):round(stop*16000)], text, language=language,
                    original_split=True, regroup=False, token_step=80, max_word_dur=4.,
                    word_dur_factor=3., nonspeech_skip=5., suppress_silence=True,
                    dynamic_heads=options.pop('dynamic_heads', args.dynamic_heads), verbose=None, **options)
                return aligned.to_dict() if aligned is not None else None

            lines = lyrics['lines'][context_first:context_stop]
            result = align_piece(0., len(piece)/16000, '\n'.join(lines))
            if result is None:
                raise ValueError(f'Alignment returned no result for {job["lyrics"]}')
            result, text_recovery = recover_changed_text(result, lines, len(piece)/16000,
                lambda: align_piece(0., len(piece)/16000, '\n'.join(lines), gap_padding=None, dynamic_heads=True))
            result, repeat_recoveries = recover_skipped_repeats(result, len(piece)/16000, align_piece)
            local_units = [{'text': word['word'], 'start': round(word['start']+begin, 3),
                            'end': round(word['end']+begin, 3), 'probability': word['probability']}
                           for segment in result['segments'] for word in segment['words']]
            units.extend(local_units)
            raw_blocks.append({'range': [first, stop], 'context': [context_first, context_stop],
                               'source_range': [begin, end], 'result': result,
                               'repeat_recoveries': repeat_recoveries, 'text_recovery': text_recovery})
        # Keep the acoustic evidence even when stitching rejects text or times.
        output.mkdir(parents=True)
        (output/'raw.json').write_text(json.dumps(raw_blocks, ensure_ascii=False, indent=2, default=json_number)+'\n')
        (output/'units.json').write_text(json.dumps(units, ensure_ascii=False, indent=2, default=json_number)+'\n')
        cues, raw_blocks, boundary_repairs = stitch_blocks(lyrics['lines'], raw_blocks, duration, lyrics['language'])
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
        report = {'model': args.model, 'device': args.device, 'dtype': 'float32',
            'model_loading_seconds': loading, 'alignment_seconds': elapsed,
            'audio': str(Path(job['audio']).resolve()), 'audio_sha256': sha(job['audio']),
            'lyrics': str(Path(job['lyrics']).resolve()), 'lyrics_sha256': sha(job['lyrics']),
            'model_sha256': sha(model_root/f'{args.model}.pt'), 'runner_sha256': sha(__file__),
            'dynamic_heads': args.dynamic_heads,
            'boundary_reconciliations': boundary_repairs,
            'onset_refinements': refinements,
            'anchors': job.get('anchors'), 'anchors_sha256': sha(job['anchors']) if job.get('anchors') else None,
            'packages': {name: version(name) for name in ['stable-ts', 'openai-whisper', 'torch']},
            'summary': summary, 'status': 'Audio-only candidate; requires timing review'}
        (output/'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_number)+'\n')
        print(json.dumps({'lyrics': job['lyrics'], 'seconds': elapsed, **summary}, default=json_number), flush=True)


if __name__ == '__main__':
    main()
