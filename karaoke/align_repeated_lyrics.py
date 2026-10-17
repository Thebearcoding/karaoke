"""Recover uncertain lyric onsets from another audible performance in the song.

This is a candidate rescue pass, not an accuracy certificate. It requires an
exact repeated lyric, a stable reference, and agreement between two audio
feature alignments. It never copies a fixed time offset between choruses.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from librosa.util.exceptions import ParameterError
from scipy.signal import resample_poly

from align_lyric_phonemes import SingingAligner
from lyric_progress import canonical_positions
from phoneme_progress import spoken_units
from stage_progress import progress_items


def fallback_fraction(cue):
    units = cue['units']
    return sum(len(u['text']) for u in units if u.get('timing_source') == 'prior_word_fallback') / max(1, sum(len(u['text']) for u in units))


def reference_pairs(cues, language):
    groups = defaultdict(list)
    for i, cue in enumerate(cues):
        groups[canonical_positions(cue['text'], language)[0]].append(i)
    return [(target, source) for text, indices in groups.items() if len(text) >= 5
            for target in indices if fallback_fraction(cues[target]) >= .6
            for source in indices if source != target and fallback_fraction(cues[source]) == 0]


def audible(piece, reference_rms):
    return len(piece) > 0 and float(np.sqrt(np.mean(piece**2))) > max(1e-5, reference_rms * .03)


def agree_onset(estimates, source_spread, start, end):
    """Reject missing evidence, unstable reference timing, and divergent paths."""
    values = np.asarray(estimates, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)) or source_spread > .12:
        return None
    if np.ptp(values) > .4 or not np.all((values >= start - .4) & (values < end)):
        return None
    return round(float(np.median(values)), 3)


def audio_features(audio, rate):
    import librosa
    divisor = np.gcd(rate, 16000)
    samples = resample_poly(audio, 16000 // divisor, rate // divisor)
    chroma = librosa.feature.chroma_stft(y=samples, sr=16000, n_fft=2048, hop_length=160, norm=2)
    mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_fft=1024, hop_length=160, n_mfcc=13)[1:]
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / np.maximum(mfcc.std(axis=1, keepdims=True), 1e-4)
    return {'pitch': chroma, 'spectrum': mfcc}


def audio_path(features, source, target, *, chroma_shift=0):
    import librosa
    windows = [(max(0, round((c['aligned_start']-1)*100)),
                min(features.shape[1], round((c['aligned_end']+1)*100))) for c in (source, target)]
    (a, b), (c, d) = windows
    target_features = np.roll(features[:, c:d], chroma_shift, axis=0) if chroma_shift else features[:, c:d]
    distance, path = librosa.sequence.dtw(X=features[:, a:b], Y=target_features, metric='cosine',
        step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]), weights_add=np.array([0., .05, .05]),
        global_constraints=True, band_rad=.2)
    path = path[::-1]
    return (path[:, 0]+a)/100, (path[:, 1]+c)/100, float(distance[-1, -1]/len(path))


def repeated_paths(features, source, target):
    """Estimate a single key change from timbral alignment, before pitch DTW."""
    spectrum = audio_path(features['spectrum'], source, target)
    x, y, _ = spectrum
    chroma = features['pitch']
    ix = np.clip(np.rint(x*100).astype(int), 0, chroma.shape[1]-1)
    iy = np.clip(np.rint(y*100).astype(int), 0, chroma.shape[1]-1)
    correlations = [np.mean(np.sum(chroma[:, ix]*np.roll(chroma[:, iy], shift, axis=0), axis=0))
                    for shift in range(12)]
    if not np.isfinite(correlations).all():
        raise ValueError('Invalid repeated pitch features')
    shift = int(np.argmax(correlations))
    pitch = audio_path(chroma, source, target, chroma_shift=shift) if shift else audio_path(chroma, source, target)
    return [('pitch', pitch), ('spectrum', spectrum)], shift


def reference_units(aligner, cues, index, samples, rate, language):
    cue = cues[index]
    units = spoken_units(cue['text'], language)
    predictions = []
    for padding in (.35, .6):
        begin, end = max(0, cue['aligned_start']-padding), cue['aligned_end']+padding
        chunk = samples[round(begin*rate):round(end*rate)]
        divisor = np.gcd(rate, 44100)
        chunk = resample_poly(chunk, 44100//divisor, rate//divisor)
        predictions.append(aligner.infer(chunk, begin, units, cue)['units'])
    # A cropped final syllable can align to reverberation. Include the actual
    # neighbouring lyrics, then require two windows to agree on each onset.
    first, stop = max(0, index-1), min(len(cues), index+2)
    text, anchors, offset = '', [], 0
    for i in range(first, stop):
        if i == index:
            offset = len(text)
        anchors.extend(dict(u, first=u['first']+len(text), stop=u['stop']+len(text)) for u in cues[i]['units'])
        text += cues[i]['text']+' '
    context: dict[str, Any] = dict(text=text.strip(), units=anchors,
                   aligned_start=cues[first]['aligned_start'], aligned_end=cues[stop-1]['aligned_end'])
    begin, end = max(0, context['aligned_start']-.6), context['aligned_end']+.6
    divisor = np.gcd(rate, 44100)
    chunk = resample_poly(samples[round(begin*rate):round(end*rate)], 44100//divisor, rate//divisor)
    extra = aligner.infer(chunk, begin, spoken_units(context['text'], language), context)['units']
    contextual = {u['first']-offset: u for u in extra if offset <= u['first'] < offset+len(cue['text'])}
    reference_rms = float(np.sqrt(np.mean(samples**2)))
    result = {}
    for first, second in zip(*predictions):
        key = len(canonical_positions(cue['text'][:first['first']], language)[0])
        choices = [first, second]
        if first['first'] in contextual:
            choices.append(contextual[first['first']])
        choices.sort(key=lambda u: u['start'])
        pair = min(zip(choices, choices[1:]), key=lambda pair: pair[1]['start']-pair[0]['start'])
        times = [u['start'] for u in pair]
        stable = times[1]-times[0]
        part = samples[round(min(times)*rate):round(min(u['end'] for u in pair)*rate)]
        if min(u['end']-u['start'] for u in pair) < .035 or not audible(part, reference_rms):
            stable = float('inf')
        result[key] = {'start': float(np.mean(times)), 'spread': stable, 'text': first['text']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    models = {}
    for job in json.loads(args.jobs.read_text()):
        output = Path(job['output'])
        if output.exists():
            raise FileExistsError(output)
        cues = json.loads(Path(job['cues']).read_text())
        coarse = json.loads(Path(job.get('coarse_cues', job['cues'])).read_text())
        if [c['text'] for c in coarse] != [c['text'] for c in cues]:
            raise ValueError('Coarse and refined lyrics must describe the same recording')
        updated, reports, best = deepcopy(cues), [], {}
        pairs = reference_pairs(cues, job['language'])
        started = time.perf_counter()
        if pairs:
            audio, rate = sf.read(job['audio'], dtype='float32', always_2d=True)
            mono = audio.mean(axis=1)
            features = audio_features(mono, rate)
            reference, reference_rate = sf.read(job.get('reference_audio', job['audio']), dtype='float32', always_2d=True)
            reference = reference.mean(axis=1)
            language = job['language']
            if language not in models:
                models[language] = SingingAligner(language, args.device)
            reference_cache = {}
            for target_id, source_id in progress_items(pairs, 'blocks'):
                source, target = cues[source_id], cues[target_id]
                try:
                    if source_id not in reference_cache:
                        reference_cache[source_id] = reference_units(models[language], coarse, source_id, reference, reference_rate, language)
                    paths = {name: audio_path(data, source, target) for name, data in features.items()}
                except (ValueError, ParameterError) as error:
                    reports.append({'target_line': target_id+1, 'reference_line': source_id+1,
                                    'ordered': False, 'accepted_units': 0, 'error': str(error)})
                    continue
                source_units = reference_cache[source_id]
                report = {'target_line': target_id+1, 'reference_line': source_id+1,
                          'path_costs': {name: curve[2] for name, curve in paths.items()}, 'units': []}
                candidate = deepcopy(target)
                for unit in candidate['units']:
                    key = len(canonical_positions(target['text'][:unit['first']], language)[0])
                    anchor = source_units.get(key)
                    if anchor is None:
                        continue
                    estimates = {name: float(np.interp(anchor['start'], x, y)) for name, (x, y, _) in paths.items()}
                    onset = agree_onset(list(estimates.values()), anchor['spread'], target['aligned_start'], target['aligned_end'])
                    if paths['pitch'][2] > .35 or paths['spectrum'][2] > .85:
                        onset = None
                    report['units'].append({'text': unit['text'], 'old_start': unit['start'], 'proposed_start': onset,
                        'estimates': estimates, 'reference_start': anchor['start'],
                        'reference_spread': anchor['spread'] if np.isfinite(anchor['spread']) else None})
                    if onset is not None:
                        unit.update(start=onset, timing_source='repeated_lyric_audio', reference_line=source_id+1,
                                    feature_disagreement=round(float(np.ptp(list(estimates.values()))), 3))
                ordered = all(b['start']-a['start'] >= .015 for a, b in zip(candidate['units'], candidate['units'][1:]))
                report['ordered'] = ordered
                accepted = [u for u in candidate['units'] if u.get('timing_source') == 'repeated_lyric_audio']
                rank = (len(accepted), -sum(u['feature_disagreement'] for u in accepted))
                report['accepted_units'] = len(accepted)
                if ordered and accepted and rank > best.get(target_id, (0, 0)):
                    for i, unit in enumerate(candidate['units']):
                        unit['end'] = candidate['units'][i+1]['start'] if i+1 < len(candidate['units']) else candidate['aligned_end']
                    candidate['start'] = candidate['aligned_start'] = candidate['units'][0]['start']
                    candidate['end'] = candidate['aligned_end']
                    candidate['timing_status'] = 'repeat_rescue_candidate'
                    updated[target_id] = candidate
                    best[target_id] = rank
                reports.append(report)
        # Keep acoustic candidates for the shared line-order repair. An
        # unrelated upstream crossing must not discard recovered repetitions.
        output.mkdir(parents=True)
        report = {'input': job, 'seconds': time.perf_counter()-started, 'attempts': reports,
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'status': 'Candidate timing from audio repetition; human onset accuracy unmeasured'}
        (output/'cues.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2)+'\n')
        (output/'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(job['song'], [(r['target_line'], r['reference_line'], r['ordered']) for r in reports], flush=True)


if __name__ == '__main__':
    main()
