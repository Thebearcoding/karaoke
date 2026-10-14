import csv
import json
import time

import numpy as np

from benchmark_alignment import DATA, ROOT, measure
from benchmark_alignment_joint import build
from gaussian_align import make_model, phones, train
from phonetic_align import align_phonetic
from experiment_output import output_directory


def evaluate(song,acoustic,prior_weight):
    model=make_model(song['lines'],acoustic)
    start=time.perf_counter()
    predicted=align_phonetic(*song['features'],model,song['lines'],activity_weight=1.2,
        duration_weight=.2,line_prior=song['prior'],prior_weight=prior_weight)
    seconds=time.perf_counter()-start+song['preparation_seconds']
    onset=np.array([abs(p['start']-float(r['start_time'])) for p,r in zip(predicted,song['reference'])])
    offset=np.array([abs(p['end']-float(r['end_time'])) for p,r in zip(predicted,song['reference'])])
    summarize=lambda x:{'median_seconds':float(np.median(x)),'p95_seconds':float(np.percentile(x,95)),'max_seconds':float(max(x))}
    a,b=summarize(onset),summarize(offset)
    return {'id':song['id'],'onset':a,'offset':b,'lines':len(predicted),'predictions':predicted,
        'unknown_words':model['unknown_words'],'fraction_onsets_within_250ms':float(np.mean(onset<=.25)),
        'timing_gate_passed':bool(a['median_seconds']<=.1 and a['p95_seconds']<=.25 and a['max_seconds']<=.5 and b['p95_seconds']<=.3),
        'semantic_match_verified':False,'processing_seconds':seconds}


def main():
    out=output_directory('alignment-v004')
    selection=json.loads((DATA/'selection-v004.json').read_text())['songs']
    songs=[];groups=[]
    for row in selection:
        if row['split']!='calibration':continue
        song=build(row);songs.append(song)
        words=' '.join(song['lines']).split()
        ref=list(csv.DictReader((DATA/'annotations'/'words'/f"{song['id']}.csv").open()))
        assert len(words)==len(ref),(song['id'],len(words),len(ref))
        times,features,_=song['features']
        for word,label in zip(words,ref):
            frames=features[(times>=float(label['word_start']))&(times<float(label['word_end']))]
            groups.append((phones(word),frames))
    acoustic=train(groups)
    np.savez(out/'gaussian-parameters.npz',names=acoustic['names'],means=acoustic['means'],
             variance=acoustic['variance'],seconds=acoustic['seconds'])
    calibration=[]
    for weight in [0,.5,2]:
        rows=[evaluate(song,acoustic,weight) for song in songs]
        loss=float(np.mean([r['onset']['p95_seconds']+r['offset']['p95_seconds'] for r in rows]))
        calibration.append({'prior_weight':weight,'loss':loss,'songs':rows})
        print('calibration',weight,loss,flush=True)
    chosen=min(calibration,key=lambda r:r['loss'])['prior_weight']
    (out/'frozen-config.json').write_text(json.dumps({'prior_weight':chosen},indent=2)+'\n')
    previous=json.loads((ROOT/'experiments'/'alignment-v001'/'frozen-config.json').read_text())
    heldout=[]
    for row in selection:
        if row['split']!='heldout':continue
        song=build(row)
        result={'id':song['id'],'source_sha256':song['source_sha256'],
            'duration_baseline':measure(song,previous),'gaussian':evaluate(song,acoustic,chosen)}
        heldout.append(result);print('heldout',song['id'],result['gaussian']['onset'],flush=True)
    report={'scope':'Non-neural Gaussian phoneme prototypes fitted only on calibration-word annotations, with 3 fresh heldout songs.',
        'training_words_used':acoustic['training_words_used'],'training_frames':acoustic['training_frames'],
        'training_history':acoustic['history'],'chosen_prior_weight':chosen,
        'calibration':calibration,'heldout':heldout,'reference_times_used_at_inference':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
