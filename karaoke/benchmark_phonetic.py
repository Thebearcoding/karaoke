"""Evaluate actual text/acoustic matching on a fresh alignment holdout."""

import json
import subprocess
import time

import numpy as np

from benchmark_alignment import DATA, ROOT, measure, prepare
from experiment_output import output_directory
from phonetic_align import align_phonetic, alignment_features, templates


def source(row):
    raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(DATA/'mp3'/row['Filepath']),
        '-ac','2','-ar','16000','-f','f32le','pipe:1'])
    return np.frombuffer(raw,dtype='<f4').reshape(-1,2)


def build(row):
    song=prepare(row)
    audio=source(row)
    start=time.perf_counter()
    song['phonetic_model']=templates(song['lines'])
    song['acoustic']=alignment_features(audio,16000)
    song['preparation_seconds']=time.perf_counter()-start
    return song


def evaluate(song,config):
    start=time.perf_counter()
    predicted=align_phonetic(*song['acoustic'],song['phonetic_model'],song['lines'],**config)
    seconds=time.perf_counter()-start+song['preparation_seconds']
    errors=[abs(p['start']-float(r['start_time'])) for p,r in zip(predicted,song['reference'])]
    end_errors=[abs(p['end']-float(r['end_time'])) for p,r in zip(predicted,song['reference'])]
    onset={'median_seconds':float(np.median(errors)),'p95_seconds':float(np.percentile(errors,95)),
           'max_seconds':float(max(errors))}
    offset={'median_seconds':float(np.median(end_errors)),'p95_seconds':float(np.percentile(end_errors,95)),
           'max_seconds':float(max(end_errors))}
    return {'id':song['id'],'onset':onset,'offset':offset,'lines':len(errors),
        'fraction_onsets_within_250ms':float(np.mean(np.array(errors)<=.25)),
        'timing_gate_passed':bool(onset['median_seconds']<=.1 and onset['p95_seconds']<=.25
            and onset['max_seconds']<=.5 and offset['p95_seconds']<=.3),
        'semantic_match_verified':False,'processing_seconds':seconds,'predictions':predicted}


def main():
    out=output_directory('alignment-v002')
    selection=json.loads((DATA/'selection-v002.json').read_text())['songs']
    train=[]
    for row in selection:
        if row['split']=='calibration':
            train.append(build(row));print('prepared',row['Filepath'],flush=True)
    calibration=[]
    for aw in [0.4,1.2]:
        for dw in [.1,.4]:
            config={'activity_weight':aw,'duration_weight':dw}
            results=[evaluate(song,config) for song in train]
            loss=float(np.mean([r['onset']['p95_seconds']+r['offset']['p95_seconds'] for r in results]))
            calibration.append({'config':config,'loss':loss,'songs':results})
            print('calibration',config,loss,flush=True)
            (out/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    chosen=min(calibration,key=lambda r:r['loss'])['config']
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    previous=json.loads((ROOT/'experiments'/'alignment-v001'/'frozen-config.json').read_text())
    heldout=[]
    for row in selection:
        if row['split']!='heldout':continue
        song=build(row)
        result={'id':song['id'],'source_sha256':song['source_sha256'],
            'uniform_baseline':measure(song,{'mode':'uniform','threshold':.3,'boundary_weight':0}),
            'duration_baseline':measure(song,previous),'phonetic':evaluate(song,chosen)}
        heldout.append(result)
        print('heldout',song['id'],result['phonetic']['onset'],flush=True)
        (out/'heldout.json').write_text(json.dumps(heldout,indent=2)+'\n')
    report={'scope':'3 calibration + 3 fresh English holdout songs; references never fed into alignment.',
        'method':'Rule/formant eSpeak phoneme templates, MFCC, duration-regularized monotonic alignment; no neural weights.',
        'limitations':'Synthetic speech and mixed singing differ; all predictions require review. No CJK quality claim.',
        'chosen':chosen,'calibration':calibration,'heldout':heldout}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
