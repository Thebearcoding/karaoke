"""Frozen six-song line-alignment experiment, using human word annotations."""

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from align import activity, align_lines
from experiment_output import output_directory

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'/'jamendo'


def prepare(row):
    stem = Path(row['Filepath']).stem
    # The algorithm receives normalized lyrics only; reference times are not
    # used for trimming, front-padding, activity extraction, or alignment.
    text = (DATA/'lyrics'/f'{stem}.txt').read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reference = list(csv.DictReader((DATA/'annotations'/'lines'/f'{stem}.csv').open()))
    # Check dataset integrity before treating annotations as ground truth.
    ref_text = [r['lyrics_line'].strip() for r in reference]
    assert lines == ref_text, (stem,len(lines),len(ref_text),'lyrics/annotation text mismatch')
    media = DATA/'mp3'/row['Filepath']
    raw = subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(media),
        '-ac','2','-ar','16000','-f','f32le','pipe:1'])
    audio = np.frombuffer(raw,dtype='<f4').reshape(-1,2)
    start = time.perf_counter()
    times,cue = activity(audio,16000)
    return {'id':stem,'lines':lines,'times':times,'cue':cue,
        'reference':reference,'split':row['split'],'seconds':len(audio)/16000,
        'feature_seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(media.read_bytes()).hexdigest()}


def measure(song, config):
    start = time.perf_counter()
    result = align_lines(song['times'],song['cue'],song['lines'],**config)
    elapsed = time.perf_counter()-start+song['feature_seconds']
    errors_start,errors_end = [],[]
    for predicted,truth in zip(result,song['reference']):
        if predicted['start'] is not None:
            errors_start.append(abs(predicted['start']-float(truth['start_time'])))
            errors_end.append(abs(predicted['end']-float(truth['end_time'])))
    resolved = len(errors_start)==len(song['reference'])
    def summary(x):
        return {'median_seconds':float(np.median(x)), 'p95_seconds':float(np.percentile(x,95)),
            'max_seconds':float(np.max(x))} if len(x) else None
    onset,offset = summary(errors_start),summary(errors_end)
    timing_passes = bool(resolved and onset['median_seconds']<=0.1 and onset['p95_seconds']<=0.25
        and onset['max_seconds']<=0.5 and offset['p95_seconds']<=0.3)
    return {'id':song['id'],'lines':len(result),'onset':onset,'offset':offset,
        'fraction_onsets_within_250ms':float(np.mean(np.array(errors_start)<=0.25)) if errors_start else 0,
        'timing_gate_passed':timing_passes,'semantic_match_verified':False,
        'processing_seconds':elapsed,'predictions':result}


def main():
    out=output_directory('alignment-v001')
    selection=json.loads((DATA/'selection.json').read_text())['songs']
    # Read the heldout audio/labels only after freezing parameters.
    train=[prepare(r) for r in selection if r['split']=='calibration']
    configs=[{'mode':'uniform','threshold':0.3,'boundary_weight':0}]
    for weight in [0.25,1,3]:
        for threshold in [0.2,0.4,0.6]:
            configs.append({'mode':'dp','boundary_weight':weight,'threshold':threshold})
    calibration: list[dict[str, Any]] =[]
    for config in configs:
        rows=[measure(song,config) for song in train]
        loss=float(np.mean([r['onset']['p95_seconds']+r['offset']['p95_seconds'] if r['onset'] else 1e6 for r in rows]))
        calibration.append({'config':config,'loss':loss,'songs':rows})
        print('calibration',config,round(loss,3),flush=True)
    candidates=[r for r in calibration if r['config']['mode']=='dp']
    chosen=min(candidates,key=lambda r:r['loss'])['config']
    (out/'frozen-config.json').write_text(json.dumps(chosen,indent=2)+'\n')
    heldout=[]
    for row in selection:
        if row['split']!='heldout':continue
        song=prepare(row)
        result={'id':song['id'],'source_sha256':song['source_sha256'],
            'baseline':measure(song,configs[0]),'candidate':measure(song,chosen)}
        heldout.append(result)
        print('heldout',song['id'],result['candidate']['onset'],flush=True)
    report={'scope':'Legacy Jamendo human-annotated English songs: 3 calibration, 3 heldout. Not CJK validation.',
        'method_limit':'Handcrafted activity + syllable count DP, no phonetic recognition. Every output requires review.',
        'references_used_as_input':False,'chosen':chosen,'calibration':calibration,'heldout':heldout}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
