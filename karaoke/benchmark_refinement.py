"""Second real-stem experiment; reserve a new, untouched holdout selection."""

import hashlib
import json
import time
from pathlib import Path

from benchmark_separation import ROOT, load, metrics, select, summarize
from dsp import RATE, apply_masks, istft, lowrank_background, masks_for, repeating_background, stft
from experiment_output import output_directory


def main():
    out=output_directory('separation-v003')
    old=ROOT/'experiments'/'separation-v002'
    split=json.loads((old/'split.json').read_text())
    seen={Path(s).name.split(' - ')[0] for s in split['heldout']}
    calibration=[ROOT/p for p in split['calibration']]
    holdout=select([p for p in (ROOT/'data'/'musdb-sample'/'test').glob('*.mp4')
                   if p.name.split(' - ')[0] not in seen],12,20260920)
    (out/'split.json').write_text(json.dumps({'calibration':[str(p.relative_to(ROOT)) for p in calibration],
        'heldout':[str(p.relative_to(ROOT)) for p in holdout]},indent=2)+'\n')
    baselines=[{'name':'center-cancel','method':'center-cancel'},
        {'name':'spatial-v2','method':'spatial','strength':0.99,'protection':0.5},
        json.loads((old/'frozen-config.json').read_text())]
    configs=[]
    for sparsity in [0.5,0.8,1.1]:
        for weight in [1,2,4]:
            configs.append({'name':f'rpca-l{sparsity}-w{weight}','method':'rpca',
                'sparsity':sparsity,'residual_weight':weight,'side_keep':0})
    def run(paths,methods,name):
        records=[]
        for path in paths:
            backing,vocals,mix=load(path)
            start=time.perf_counter();z=stft(mix);stft_seconds=time.perf_counter()-start
            zv,zb=stft(vocals),stft(backing)
            results={};backgrounds={};costs={}
            for config in methods:
                key=('rpca',config['sparsity']) if config['method']=='rpca' else ('repeat',0)
                needs=config['method'] in ['rpca','hybrid','repetition']
                if needs and key not in backgrounds:
                    start=time.perf_counter()
                    backgrounds[key]=lowrank_background(z,key[1]) if key[0]=='rpca' else repeating_background(z)
                    costs[key]=time.perf_counter()-start
                start=time.perf_counter();m=masks_for(z,config,backgrounds.get(key));istft(apply_masks(z,m),len(mix))
                seconds=stft_seconds+time.perf_counter()-start+(costs[key] if needs else 0)
                r=metrics(backing,vocals,z,zv,zb,m);r.update(processing_seconds=seconds,rtf=seconds/(len(mix)/RATE))
                results[config['name']]=r
            records.append({'track':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'duration_seconds':len(mix)/RATE,'results':results})
            (out/f'{name}.json').write_text(json.dumps(records,indent=2)+'\n')
            print(name,path.stem,'done',flush=True)
        return records
    train=run(calibration,configs+baselines,'calibration')
    summary=summarize(train,configs+baselines)
    chosen=min(configs,key=lambda c:summary[c['name']]['mean_target_deficit'])
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    test=run(holdout,[chosen]+baselines,'heldout')
    report={'scope':'Pooled robust PCA candidate; new artist-disjoint 12-track holdout excluding v002 holdout artists.',
        'chosen':chosen,'calibration':summary,'heldout':summarize(test,[chosen]+baselines),'candidate_configs':configs+baselines}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['heldout'],indent=2),flush=True)


if __name__=='__main__':main()
