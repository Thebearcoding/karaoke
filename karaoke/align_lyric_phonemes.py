"""Refine lyric progress with singing phonemes inside existing lyric windows."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from lyric_progress import canonical_positions
from phoneme_progress import ROOT, phoneme_sequence, spoken_units

def overlapping(unit, anchors):
    return [i for i,a in enumerate(anchors) if a['first']<unit['stop'] and a['stop']>unit['first']]


def guided_scores(ids, log_probs, owners, units, anchors, times):
    """Give repeated phonemes separate time priors without changing silence."""
    expanded=np.column_stack([log_probs[:,0],log_probs[:,ids]])
    state_ids=np.arange(len(ids))+1
    state_ids[ids==0]=0
    for state,owner in enumerate(owners):
        if owner<0:
            continue
        parents=[anchors[i] for i in overlapping(units[owner],anchors)]
        a,b=min(p['start'] for p in parents)-.3,max(p['end'] for p in parents)+.3
        distance=np.maximum(np.maximum(a-times,times-b),0)
        expanded[:,state+1]-=20*distance**2
    return state_ids,expanded.astype('float32')


def protect_units(cue, candidate, language):
    """Keep prior timing for collapsed spans; never interpolate missing words."""
    anchors=cue['units']
    bad=set()
    reasons=[]
    for unit in candidate:
        parents=overlapping(unit,anchors)
        same_start=next((a for a in anchors if a['first']==unit['first']),None)
        reason=None
        if unit['end']-unit['start']<.035 and unit.get('phones')!=['cl']:
            reason='collapsed phonetic span'
        elif unit['start']>=cue['aligned_end']-.025:
            reason='onset beyond line end'
        elif unit['start']<cue['aligned_start']-.4:
            reason='onset before lyric window'
        elif same_start and abs(unit['start']-same_start['start'])>.7:
            reason='large onset change requires independent verification'
        if reason:
            bad.update(parents)
            reasons.append({'text':unit['text'],'reason':reason})
    # A compound overlapping a fallback span must be replaced as a whole.
    changed=True
    while changed:
        before=len(bad)
        for unit in candidate:
            parents=set(overlapping(unit,anchors))
            if parents & bad:
                bad.update(parents)
        changed=len(bad)>before
    result=[dict(u,timing_source='singing_phonemes') for u in candidate
            if not set(overlapping(u,anchors)) & bad]
    result.extend(dict(anchors[i],timing_source='prior_word_fallback') for i in sorted(bad))
    result.sort(key=lambda u:u['first'])
    valid_text=canonical_positions(''.join(u['text'] for u in result),language)[0]==canonical_positions(cue['text'],language)[0]
    if not valid_text or any(b['start']-a['start']<.015 for a,b in zip(result,result[1:])):
        return [dict(u,timing_source='prior_word_fallback') for u in anchors],reasons+[{'reason':'non-monotonic or incomplete candidate'}]
    for i,unit in enumerate(result):
        limit=result[i+1]['start'] if i+1<len(result) else cue['aligned_end']
        unit['end']=round(min(limit,max(unit['end'],unit['start']+.015)),3)
        unit['start']=round(unit['start'],3)
    return result,reasons


def protect_line_order(cues, prior):
    """Revert conflicting refinements to measured coarse timing, never shift words."""
    if [c['text'] for c in cues] != [c['text'] for c in prior]:
        raise ValueError('Coarse and refined lyric identities differ')

    def conflicts(lines):
        if any(not c.get('units') for c in lines):
            raise ValueError('A lyric line has no timing units')
        units = [(i, u) for i, c in enumerate(lines) for u in c['units']]
        return {i for (a, left), (b, right) in zip(units, units[1:])
                if right['start']-left['start'] < .015-1e-9 for i in (a, b)}

    if conflicts(prior):
        raise ValueError('Coarse lyric onsets are not ordered; the recording/lyrics need review')
    result, reverted = deepcopy(cues), set()
    while bad := conflicts(result):
        fresh = bad-reverted
        if not fresh:
            raise ValueError('Cannot restore lyric order from the coarse alignment')
        for index in fresh:
            result[index] = deepcopy(prior[index])
            result[index]['units'] = [dict(u, timing_source='prior_word_fallback') for u in prior[index]['units']]
            result[index]['timing_status'] = 'line_order_fallback'
        reverted.update(fresh)
    return result, [i+1 for i in sorted(reverted)]


class SingingAligner:
    def __init__(self,language,device):
        sys.path.insert(0,str(ROOT/'research/subtitles-2026-09-18/sofa'))
        from modules.task.forced_alignment import LitForcedAlignmentTask
        from modules.rmvpe.spec import MelSpectrogram
        folder=ROOT/'models/alignment/sofa'/('japanese' if language=='Japanese' else 'mandarin')
        self.checkpoint=next(folder.rglob('*.ckpt'))
        weights=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        self.model=LitForcedAlignmentTask(**weights['hyper_parameters'])
        self.model.load_state_dict(weights['state_dict'])
        self.model.eval().to(device)
        self.device=device
        self.config=self.model.melspec_config
        self.decode=self.model._decode
        c=self.config
        self.extract=MelSpectrogram(n_mel_channels=c['n_mels'],sampling_rate=c['sample_rate'],
            win_length=c['win_length'],hop_length=c['hop_length'],n_fft=c['n_fft'],
            mel_fmin=c['fmin'],mel_fmax=c['fmax'],clamp=c['clamp'])

    def infer(self,audio,begin,units,cue):
        phones,words,owners=phoneme_sequence(units)
        missing=set(phones)-self.model.vocab.keys()
        if missing:
            raise ValueError(f'Phonemes absent from model: {missing}')
        audio=audio.copy()
        audio*=.97/max(.97,float(np.max(np.abs(audio))))
        mel=self.extract(torch.from_numpy(audio).unsqueeze(0))
        mel=(mel-mel.mean())/mel.std().clamp_min(1e-6)
        mel=mel.repeat_interleave(self.config['scale_factor'],dim=-1).to(self.device)
        def decode(ids,log_probs,edges):
            times=np.arange(len(log_probs))*self.config['hop_length']/(44100*self.config['scale_factor'])+begin
            state_ids,scores=guided_scores(ids,log_probs,owners,units,cue['units'],times)
            return self.decode(state_ids,scores,edges)
        self.model._decode=decode
        ph,intervals,word_seq,word_times,confidence,_,_=self.model._infer_once(mel,len(audio)/44100,phones,words,owners)
        if list(word_seq)!=words:
            raise ValueError('Phonetic decoder omitted a displayed unit')
        result=[dict(u,start=round(float(t[0])+begin,3),end=round(float(t[1])+begin,3))
                for u,t in zip(units,word_times)]
        return dict(units=result,confidence=float(confidence),phones=ph.tolist(),phone_intervals=(intervals+begin).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    models={}
    for job in json.loads(args.jobs.read_text()):
        output=Path(job['output'])
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        language=job['language']
        if language not in models:
            models[language]=SingingAligner(language,args.device)
        aligner=models[language]
        cues=json.loads(Path(job['cues']).read_text())
        audio,rate=sf.read(job['audio'],dtype='float32',always_2d=True)
        divisor=np.gcd(rate,44100)
        audio=resample_poly(audio.mean(axis=1),44100//divisor,rate//divisor).astype('float32')
        started=time.perf_counter()
        raw=[]
        for index,cue in enumerate(cues):
            if job.get('lines') is not None and index+1 not in job['lines']:
                continue
            begin=max(0,cue['aligned_start']-.35)
            end=min(len(audio)/44100,cue['aligned_end']+.35)
            chunk=audio[round(begin*44100):round(end*44100)]
            attempts=[]
            seen=set()
            for isolated in ([False,True] if language=='Japanese' else [False]):
                try:
                    units=spoken_units(cue['text'],language,isolated)
                    signature=tuple(tuple(u['phones']) for u in units)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    prediction=aligner.infer(chunk,begin,units,cue)
                    protected,reasons=protect_units(cue,prediction['units'],language)
                    prediction.update(isolated_reading=isolated,protected=protected,fallbacks=reasons)
                    attempts.append(prediction)
                except ValueError as error:
                    attempts.append({'error':str(error),'isolated_reading':isolated})
            usable=[p for p in attempts if 'protected' in p]
            if usable:
                chosen=min(usable,key=lambda p:(sum(u['timing_source']=='prior_word_fallback' for u in p['protected']),-p['confidence']))
                cue['units']=chosen['protected']
                cue['timing_status']='singing_phoneme_refinement'
            else:
                cue['units']=[dict(u,timing_source='prior_word_fallback') for u in cue['units']]
                cue['timing_status']='prior_word_fallback'
            raw.append({'line':index+1,'text':cue['text'],'source_range':[begin,end],'attempts':attempts})
            if (index+1)%10==0:
                print(job['song'],index+1,len(cues),flush=True)
        elapsed=time.perf_counter()-started
        report={'song':job['song'],'seconds':elapsed,'input':job,'device':args.device,
            'checkpoint':str(aligner.checkpoint),'lines':len(cues),
            'fallback_units':sum(u.get('timing_source')=='prior_word_fallback' for c in cues for u in c['units']),
            'total_units':sum(len(c['units']) for c in cues),
            'fully_refined_lines':sum(all(u.get('timing_source')=='singing_phonemes' for u in c['units']) for c in cues),
            'packages':{name:version(name) for name in ['torch','lightning','pyopenjtalk','pykakasi','librosa','numba']},
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'status':'Audio-based candidate; no human-labelled timing accuracy claimed'}
        (output/'cues.json').write_text(json.dumps(cues,ensure_ascii=False,indent=2)+'\n')
        (output/'raw.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
        (output/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
