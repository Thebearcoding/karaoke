"""Third separation family, with another never-inspected holdout artist set."""

import json
from pathlib import Path

from benchmark_separation import ROOT, evaluate, select, summarize
from experiment_output import output_directory


def main():
    out=output_directory('separation-v004')
    old=ROOT/'experiments'/'separation-v002'
    split=json.loads((old/'split.json').read_text())
    seen=set()
    for version in ['separation-v002','separation-v003']:
        previous=json.loads((ROOT/'experiments'/version/'split.json').read_text())
        seen.update(Path(s).name.split(' - ')[0] for s in previous['heldout'])
    calibration=[ROOT/p for p in split['calibration']]
    holdout=select([p for p in (ROOT/'data'/'musdb-sample'/'test').glob('*.mp4')
                   if p.name.split(' - ')[0] not in seen],12,20260921)
    (out/'split.json').write_text(json.dumps({'calibration':[str(p.relative_to(ROOT)) for p in calibration],
        'heldout':[str(p.relative_to(ROOT)) for p in holdout]},indent=2)+'\n')
    configs=[{'name':f'harmonic-w{w}-s{s}','method':'harmonic','width':w,'side_keep':s}
             for w in [18,35,70] for s in [0,0.5,1]]
    baselines=[{'name':'center-cancel','method':'center-cancel'},
        {'name':'spatial-v2','method':'spatial','strength':0.99,'protection':0.5},
        json.loads((old/'frozen-config.json').read_text())]
    records=evaluate(calibration,configs+baselines,out/'calibration.json')
    summary=summarize(records,configs+baselines)
    chosen=min(configs,key=lambda c:summary[c['name']]['mean_target_deficit'])
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    test=evaluate(holdout,[chosen]+baselines,out/'heldout.json')
    report={'scope':'Harmonic tracking candidate; new heldout artists not inspected in v002/v003.',
        'chosen':chosen,'calibration':summary,'heldout':summarize(test,[chosen]+baselines),'candidate_configs':configs+baselines}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['heldout'],indent=2),flush=True)


if __name__=='__main__':main()
