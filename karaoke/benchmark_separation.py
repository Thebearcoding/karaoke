"""Evaluate fixed-mask candidates against real MUSDB stems without NNs."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from dsp import RATE, apply_masks, istft, masks_for, repeating_background, stft

ROOT = Path(__file__).resolve().parent


def decode(path, stream):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
        '-map', f'0:a:{stream}', '-ac', '2', '-ar', str(RATE), '-f', 'f32le', 'pipe:1'])
    return np.frombuffer(raw, dtype='<f4').reshape(-1, 2).copy()


def ratio(a, b):
    return float(10*np.log10((np.sum(a.astype(float)**2)+1e-20)/(np.sum(b.astype(float)**2)+1e-20)))


def load(path):
    # Published MUSDB mapping: 1 drums, 2 bass, 3 other, 4 vocals.
    stems = [decode(path, i) for i in [1, 2, 3, 4]]
    length = min(map(len, stems))
    stems = [x[:length] for x in stems]
    backing, vocals = sum(stems[:3]), stems[3]
    # Use exactly additive decoded stems, not separately AAC-encoded mix 0.
    mix = backing+vocals
    return backing, vocals, mix


def metrics(backing, vocals, zmix, zvocals, zbacking, masks):
    y = istft(apply_masks(zmix, masks), len(backing))
    v = istft(apply_masks(zvocals, masks), len(backing))
    b = istft(apply_masks(zbacking, masks), len(backing))
    assert np.max(np.abs(y-b-v)) < 2e-5
    windows = []
    energies = []
    for start in range(0, len(vocals)-RATE+1, RATE):
        end = start+RATE
        energies.append(float(np.mean(vocals[start:end]**2)))
        windows.append(ratio(vocals[start:end], v[start:end]))
    active = [s for s, e in zip(windows, energies) if e > max(energies, default=0)*1e-4 and e > 1e-10]
    reconstruction = ratio(backing, y-backing)
    improvement = reconstruction-ratio(backing, vocals)
    attenuation = ratio(vocals, v)
    floor = float(np.percentile(active, 5)) if active else None
    passed = bool(reconstruction >= 12 and improvement >= 6 and attenuation >= 15 and floor is not None and floor >= 10)
    return {'reconstruction_db': reconstruction, 'improvement_db': improvement,
        'vocal_attenuation_db': attenuation, 'vocal_attenuation_p05_1s_db': floor,
        'backing_preservation_db': ratio(backing, b-backing), 'passes_objective_gates': passed}


def select(paths, count, seed):
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    selected, artists = [], set()
    for path in paths:
        artist = path.name.split(' - ')[0]
        if artist in artists:
            continue
        artists.add(artist)
        selected.append(path)
        if len(selected) == count:
            break
    return selected


def candidates():
    result = [{'name': 'unchanged', 'method': 'identity'}, {'name': 'center-cancel', 'method': 'center-cancel'},
        {'name':'spatial-v1', 'method':'spatial', 'strength':0.96, 'protection':0},
        {'name':'spatial-v2', 'method':'spatial', 'strength':0.99, 'protection':0.5}]
    for weight in [1, 2, 4, 8]:
        for side in [0, 0.5, 1]:
            result.append({'name':f'repeat-w{weight}-s{side}', 'method':'repetition',
                           'residual_weight':weight, 'side_keep':side})
    for weight in [2, 4]:
        for side in [0, 0.5, 1]:
            result.append({'name':f'hybrid-w{weight}-s{side}', 'method':'hybrid',
                'residual_weight':weight, 'side_keep':side, 'strength':0.99})
    return result


def evaluate(paths, configs, report_path):
    records = []
    for path in paths:
        backing, vocals, mix = load(path)
        if np.mean(vocals**2) < 1e-10:
            records.append({'track':path.stem, 'excluded':'silent reference vocal'})
            continue
        start = time.perf_counter()
        zmix = stft(mix)
        stft_time = time.perf_counter()-start
        zvocals, zbacking = stft(vocals), stft(backing)
        start = time.perf_counter()
        background = repeating_background(zmix)
        repeat_time = time.perf_counter()-start
        results = {}
        for config in configs:
            start = time.perf_counter()
            masks = masks_for(zmix, config, background)
            output = istft(apply_masks(zmix, masks), len(mix))
            seconds = stft_time+time.perf_counter()-start
            if config['method'] in ['repetition','hybrid']:
                seconds += repeat_time
            assert np.isfinite(output).all()
            scores = metrics(backing, vocals, zmix, zvocals, zbacking, masks)
            scores['processing_seconds'] = seconds
            scores['rtf'] = seconds/(len(mix)/RATE)
            results[config['name']] = scores
        records.append({'track':path.stem,'source':str(path.relative_to(ROOT)),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'duration_seconds':len(mix)/RATE,'results':results})
        report_path.write_text(json.dumps(records, indent=2)+'\n')
        print(path.stem, 'done', flush=True)
    return records


def summarize(records, configs):
    summaries = {}
    for c in configs:
        values = [r['results'][c['name']] for r in records if 'results' in r]
        metrics_keys = ['reconstruction_db','improvement_db','vocal_attenuation_db','backing_preservation_db','rtf']
        s = {key:float(np.median([v[key] for v in values])) for key in metrics_keys}
        s['objective_passes'] = sum(v['passes_objective_gates'] for v in values)
        s['tracks'] = len(values)
        s['mean_target_deficit'] = float(np.mean([max(0,12-v['reconstruction_db'])+
            max(0,6-v['improvement_db'])+max(0,15-v['vocal_attenuation_db'])+
            max(0,10-(v['vocal_attenuation_p05_1s_db'] or 0)) for v in values]))
        summaries[c['name']] = s
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    corpus = ROOT/'data'/'musdb-sample'
    calibration = select((corpus/'train').glob('*.mp4'), args.count, 20260918)
    holdout = select((corpus/'test').glob('*.mp4'), args.count, 20260919)
    (out/'split.json').write_text(json.dumps({'calibration':[str(p.relative_to(ROOT)) for p in calibration],
        'heldout':[str(p.relative_to(ROOT)) for p in holdout]},indent=2)+'\n')
    configs = candidates()
    records = evaluate(calibration, configs, out/'calibration.json')
    summary = summarize(records, configs)
    # Infeasible results are still failures. Pick the smallest target deficit
    # only as the next research candidate, never mark it accepted by ranking.
    eligible = [c for c in configs if c['method'] not in ['identity','center-cancel']]
    chosen = min(eligible, key=lambda c:summary[c['name']]['mean_target_deficit'])
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    selected = [configs[0],configs[1],configs[2],configs[3]]
    if chosen not in selected:
        selected.append(chosen)
    # This is the first and only holdout evaluation for this experiment.
    heldout = evaluate(holdout, selected, out/'heldout.json')
    report = {'source':'https://github.com/sigsep/sigsep-mus-db/releases/tag/v0.4.0',
        'scope':'Real ~7 s excerpts, additive decoded real stems; not full-song quality or listening acceptance.',
        'sample_rate':RATE,'calibration':summary,'chosen':chosen,
        'heldout':summarize(heldout,selected),'candidate_configs':configs}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'chosen':chosen,'heldout':report['heldout']},indent=2),flush=True)


if __name__ == '__main__':
    main()
