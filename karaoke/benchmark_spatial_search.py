"""Diagnose stronger versions of low-damage A; keep its default unchanged."""

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_separation import ROOT, evaluate, select, summarize
from experiment_output import output_directory

BASELINE = {'name': 'spatial-v2', 'method': 'spatial', 'strength': .99, 'protection': .5}


def paired_summary(records, config):
    result = {}
    for key in ['vocal_attenuation_db', 'backing_preservation_db', 'reconstruction_db']:
        deltas = [r['results'][config['name']][key]-r['results'][BASELINE['name']][key]
                  for r in records if 'results' in r]
        result[key] = {'median': float(np.median(deltas)), 'min': float(np.min(deltas)),
                       'max': float(np.max(deltas)), 'improved': sum(d > 0 for d in deltas),
                       'tracks': len(deltas)}
    return result


def main():
    out = output_directory('separation-v006')
    previous = [ROOT/'experiments'/f'separation-v{i:03d}' for i in range(2, 6)]
    seen = set()
    for folder in previous:
        split = json.loads((folder/'split.json').read_text())
        for partition in ['calibration', 'heldout']:
            seen.update(Path(p).name.split(' - ')[0] for p in split[partition])
    calibration = [ROOT/p for p in json.loads((previous[0]/'split.json').read_text())['calibration']]
    # Official test artists have all been viewed. This is explicitly a new
    # internal holdout from unused training artists, not a fresh official test.
    heldout = select([p for p in (ROOT/'data'/'musdb-sample'/'train').glob('*.mp4')
                      if p.name.split(' - ')[0] not in seen], 12, 20260923)
    if len(heldout) != 12:
        raise RuntimeError('Need 12 previously unused artists for this protocol')
    configs = [dict(BASELINE, name=f'spatial-c{c}-p{p}', coherence_power=c, protection=p)
               for c in [1., 1.5, 2.] for p in [.2, .35, .5] if (c, p) != (2., .5)]
    protocol = {
        'feedback': 'experiments/listening-v002/feedback.json',
        'feedback_interpretation': 'A is preferred for low damage, not vocal removal. A/C both retain the original singer. A 1 dB search threshold is diagnostic, not sufficient audible improvement or acceptance.',
        'scope': '12 calibration clips; 12 unseen internal holdout clips from MUSDB train, artist-disjoint from every previously viewed clip. Short excerpts only.',
        'selection': 'Among variants with median paired vocal attenuation gain >=1 dB, median backing preservation change >=-1 dB, worst backing change >=-2 dB and median reconstruction change >=0 dB, choose minimum original mean target deficit. If none qualify, keep A and do not consume the holdout.',
        'acceptance': 'Research selection only. Original ACCEPTANCE.md quality gates are unchanged.',
        'candidate_configs': configs,
        'baseline': BASELINE,
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (out/'split.json').write_text(json.dumps({
        'calibration': [str(p.relative_to(ROOT)) for p in calibration],
        'heldout': [str(p.relative_to(ROOT)) for p in heldout],
        'heldout_origin': 'Previously unused MUSDB train artists; internal research holdout',
    }, indent=2)+'\n')
    code = out/'code'; code.mkdir()
    manifest = []
    for name in ['dsp.py', 'benchmark_spatial_search.py', 'benchmark_separation.py', 'experiment_output.py']:
        data = (ROOT/name).read_bytes()
        (code/name).write_bytes(data)
        manifest.append({'file': name, 'sha256': hashlib.sha256(data).hexdigest()})
    (out/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    records = evaluate(calibration, [BASELINE]+configs, out/'calibration.json')
    summary = summarize(records, [BASELINE]+configs)
    paired = {c['name']: paired_summary(records, c) for c in configs}
    eligible = [c for c in configs if
                paired[c['name']]['vocal_attenuation_db']['median'] >= 1 and
                paired[c['name']]['backing_preservation_db']['median'] >= -1 and
                paired[c['name']]['backing_preservation_db']['min'] >= -2 and
                paired[c['name']]['reconstruction_db']['median'] >= 0]
    chosen = min(eligible, key=lambda c: summary[c['name']]['mean_target_deficit']) if eligible else None
    (out/'frozen-config.json').write_text(json.dumps(chosen, indent=2)+'\n')
    report = {'protocol': protocol, 'calibration': summary, 'calibration_paired_vs_A': paired,
              'eligible': [c['name'] for c in eligible], 'chosen': chosen,
              'heldout': None, 'heldout_paired_vs_A': None, 'acceptance': 'not accepted'}
    if chosen:
        test = evaluate(heldout, [BASELINE, chosen], out/'heldout.json')
        report['heldout'] = summarize(test, [BASELINE, chosen])
        report['heldout_paired_vs_A'] = paired_summary(test, chosen)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['eligible', 'chosen', 'heldout', 'heldout_paired_vs_A']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
