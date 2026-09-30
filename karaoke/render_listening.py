"""Render shared-level local-song comparisons; no quality approval implied."""

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp import RATE, separate
from experiment_output import output_directory


ROOT=Path(__file__).resolve().parent


def main():
    out=output_directory('listening-v002')
    manifest=json.loads((ROOT/'samples'/'manifest.json').read_text())
    configs={
        'spatial':{'method':'spatial','strength':.99,'protection':.5},
        'repeat':json.loads((ROOT/'experiments'/'separation-v002'/'frozen-config.json').read_text()),
        'harmonic':json.loads((ROOT/'experiments'/'separation-v004'/'frozen-config.json').read_text()),
    }
    # One stable mapping across all songs; store separately from listener page.
    names=list(configs);random.Random(73421).shuffle(names)
    labels=dict(zip('ABC',names))
    (out/'answer-key.json').write_text(json.dumps({'labels':labels,'configs':configs},indent=2)+'\n')
    reports=[]
    for song in manifest['songs']:
        folder=out/song['id'];folder.mkdir()
        raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(ROOT/'samples'/song['file']),
            '-ac','2','-ar',str(RATE),'-f','f32le','pipe:1'])
        audio=np.frombuffer(raw,dtype='<f4').reshape(-1,2)
        count=len(audio);timings={};versions={'R':audio}
        for label,name in labels.items():
            start=time.perf_counter();backing,_=separate(audio,configs[name])
            timings[label]=time.perf_counter()-start
            assert backing.shape==audio.shape and np.isfinite(backing).all()
            versions[label]=backing
            print(song['id'],label,'processed',round(timings[label],2),'s',flush=True)
        peak=max(float(np.max(np.abs(x))) for x in versions.values())
        gain=min(1,.95/max(peak,1e-8))
        files={}
        for label,version in versions.items():
            filename=f'{label}.flac'
            sf.write(folder/filename,version*gain,RATE,subtype='PCM_24')
            assert sf.info(folder/filename).frames==count
            files[label]={'file':filename,'sha256':hashlib.sha256((folder/filename).read_bytes()).hexdigest()}
        # Same fixed location used in v001: source 30% + 8 seconds, 12 s per arm.
        begin=round((song['duration_seconds']*.3+8)*RATE)
        pieces=[]
        for label in ['R','A','B','C']:
            clip=(versions[label][begin:begin+12*RATE]*gain).copy()
            ramp=np.linspace(0,1,441)[:,None]
            clip[:441]*=ramp;clip[-441:]*=ramp[::-1]
            pieces.extend([clip,np.zeros((RATE//2,2))])
        sf.write(folder/'comparison.flac',np.concatenate(pieces),RATE,subtype='PCM_24')
        reports.append({'id':song['id'],'source_sha256':song['sha256'],'duration_seconds':count/RATE,
            'shared_gain':gain,'processing_seconds':timings,'files':files,
            'comparison_source_start_seconds':begin/RATE,'comparison_order':['R','A','B','C'],
            'comparison_clip_seconds':12,'separator_seconds':.5,'quality_status':'unrated candidates; no acceptance claimed'})
        (out/'manifest.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
