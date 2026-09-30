"""Check whether following the center mask removes side vocal residue."""

import hashlib
import json
from pathlib import Path

from benchmark_separation import ROOT, evaluate, select, summarize
from benchmark_spatial_search import BASELINE, paired_summary
from experiment_output import output_directory


def main():
    out = output_directory('separation-v007')
    previous = [ROOT/'experiments'/f'separation-v{i:03d}' for i in range(2, 7)]
    seen = set()
    for folder in previous:
        split = json.loads((folder/'split.json').read_text())
        seen.update(Path(p).name.split(' - ')[0] for p in split['calibration'])
        # Reserved, unexamined holdouts from rejected searches remain usable.
        if (folder/'heldout.json').exists():
            seen.update(Path(p).name.split(' - ')[0] for p in split['heldout'])
    split = json.loads((previous[0]/'split.json').read_text())
    calibration = [ROOT/p for p in split['calibration']]
    heldout = select([p for p in (ROOT/'data'/'musdb-sample'/'train').glob('*.mp4')
                      if p.name.split(' - ')[0] not in seen], 12, 20260923)
    if len(heldout) != 12:
        raise RuntimeError('Need 12 previously unused artists')
    configs = [dict(BASELINE, name=f'coupled-p{p}-g{g}-s{s}', protection=p,
                    coherence_power=1.5, gain_power=g, side_strength=s)
               for p in [.2, .5] for g in [1, 2] for s in [.5, 1.]]
    protocol = {
        'scope': 'Side coupling after the A/C vocal-residue listening report. Same calibration as prior rounds; new artist-disjoint internal holdout from unused MUSDB train clips.',
        'selection': 'Vocal-first diagnostic: among configurations with calibration median vocal attenuation >=15 dB, choose highest median backing preservation. If none, choose highest median attenuation and explicitly report target failure. Only this frozen candidate is evaluated on heldout data.',
        'acceptance': 'No quality gate changed. Higher attenuation with damaged backing is not a successful separator.',
        'candidate_configs': configs, 'baseline': BASELINE,
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (out/'split.json').write_text(json.dumps({
        'calibration': [str(p.relative_to(ROOT)) for p in calibration],
        'heldout': [str(p.relative_to(ROOT)) for p in heldout],
        'heldout_origin': 'Previously unused MUSDB train artists; internal research holdout',
    }, indent=2)+'\n')
    code = out/'code'; code.mkdir()
    source_manifest = []
    for name in ['dsp.py', 'benchmark_spatial_coupled.py', 'benchmark_spatial_search.py',
                 'benchmark_separation.py', 'experiment_output.py']:
        data = (ROOT/name).read_bytes(); (code/name).write_bytes(data)
        source_manifest.append({'file': name, 'sha256': hashlib.sha256(data).hexdigest()})
    (out/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
    records = evaluate(calibration, [BASELINE]+configs, out/'calibration.json')
    summary = summarize(records, [BASELINE]+configs)
    eligible = [c for c in configs if summary[c['name']]['vocal_attenuation_db'] >= 15]
    chosen = (max(eligible, key=lambda c: summary[c['name']]['backing_preservation_db']) if eligible
              else max(configs, key=lambda c: summary[c['name']]['vocal_attenuation_db']))
    (out/'frozen-config.json').write_text(json.dumps(chosen, indent=2)+'\n')
    test = evaluate(heldout, [BASELINE, chosen], out/'heldout.json')
    report = {'protocol': protocol, 'calibration': summary,
              'calibration_paired_vs_A': {c['name']: paired_summary(records, c) for c in configs},
              'calibration_median_attenuation_target_met': bool(eligible),
              'chosen': chosen, 'heldout': summarize(test, [BASELINE, chosen]),
              'heldout_paired_vs_A': paired_summary(test, chosen),
              'acceptance': 'not accepted'}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['calibration_median_attenuation_target_met',
          'chosen', 'heldout', 'heldout_paired_vs_A']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
