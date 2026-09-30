"""Test whether non-neural vocal extraction and weak timing priors aid alignment."""

import json
import subprocess
import time

import numpy as np
from scipy import signal

from align import align_lines
from benchmark_alignment import DATA, ROOT, measure, prepare
from dsp import separate
from phonetic_align import alignment_features, align_phonetic, templates
from experiment_output import output_directory


def build(row):
    song=prepare(row)
    raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(DATA/'mp3'/row['Filepath']),
        '-ac','2','-ar','44100','-f','f32le','pipe:1'])
    audio=np.frombuffer(raw,dtype='<f4').reshape(-1,2)
    start=time.perf_counter()
    _,vocals=separate(audio,{'method':'spatial','strength':.99,'protection':.5})
    vocals=signal.resample_poly(vocals,160,441,axis=0)
    song['features']=alignment_features(vocals,16000)
    song['model']=templates(song['lines'])
    t,_,cue=song['features']
    song['prior']=align_lines(t,cue,song['lines'],mode='dp',threshold=.2,boundary_weight=.25)
    song['preparation_seconds']=time.perf_counter()-start
    return song


def evaluate(song,config):
    start=time.perf_counter()
    result=align_phonetic(*song['features'],song['model'],song['lines'],
        activity_weight=1.2,duration_weight=.1,line_prior=song['prior'],**config)
    elapsed=time.perf_counter()-start+song['preparation_seconds']
    errors=np.array([[abs(p['start']-float(r['start_time'])),abs(p['end']-float(r['end_time']))]
        for p,r in zip(result,song['reference'])])
    onset={'median_seconds':float(np.median(errors[:,0])),'p95_seconds':float(np.percentile(errors[:,0],95)),
        'max_seconds':float(max(errors[:,0]))}
    offset={'median_seconds':float(np.median(errors[:,1])),'p95_seconds':float(np.percentile(errors[:,1],95)),
        'max_seconds':float(max(errors[:,1]))}
    return {'id':song['id'],'onset':onset,'offset':offset,'lines':len(result),
        'fraction_onsets_within_250ms':float(np.mean(errors[:,0]<=.25)),
        'timing_gate_passed':bool(onset['median_seconds']<=.1 and onset['p95_seconds']<=.25
            and onset['max_seconds']<=.5 and offset['p95_seconds']<=.3),
        'semantic_match_verified':False,'processing_seconds':elapsed,'predictions':result}


def main():
    out=output_directory('alignment-v003')
    selection=json.loads((DATA/'selection-v003.json').read_text())['songs']
    train=[]
    for row in selection:
        if row['split']=='calibration':
            train.append(build(row));print('prepared',row['Filepath'],flush=True)
    calibration=[]
    for weight in [0,.1,.5,2]:
        config={'prior_weight':weight}
        rows=[evaluate(song,config) for song in train]
        loss=float(np.mean([r['onset']['p95_seconds']+r['offset']['p95_seconds'] for r in rows]))
        calibration.append({'config':config,'loss':loss,'songs':rows})
        print('calibration',config,loss,flush=True)
    chosen=min(calibration,key=lambda x:x['loss'])['config']
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    heldout=[]
    previous=json.loads((ROOT/'experiments'/'alignment-v001'/'frozen-config.json').read_text())
    for row in selection:
        if row['split']!='heldout':continue
        song=build(row)
        r={'id':song['id'],'source_sha256':song['source_sha256'],
            'duration_baseline':measure(song,previous),'joint':evaluate(song,chosen),
            'vocal_only':evaluate(song,{'prior_weight':0})}
        heldout.append(r);print('heldout',song['id'],r['joint']['onset'],flush=True)
    report={'scope':'Rule-based vocal estimate + formant templates and coarse activity prior; 3 new English holdout songs.',
        'no_neural_inference':True,'reference_times_used_as_input':False,
        'chosen':chosen,'calibration':calibration,'heldout':heldout}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
