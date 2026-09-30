"""Experimental formant-synthesis templates + monotonic phoneme alignment.

eSpeak NG is a rule/formant synthesizer, not a neural TTS model. Synthetic
templates are only alignment features; they never replace the original music.
"""

import ctypes as C

import espeakng_loader
import numpy as np
from scipy import fft, signal

from align import activity


class EventId(C.Union):
    _fields_=[('number',C.c_int),('name',C.c_char_p),('string',C.c_char*8)]


class Event(C.Structure):
    _fields_=[('type',C.c_int),('uid',C.c_uint),('position',C.c_int),('length',C.c_int),
        ('audio_position',C.c_int),('sample',C.c_int),('user_data',C.c_void_p),('id',EventId)]


class FormantSynth:
    def __init__(self, voice='en-us'):
        self.lib=C.CDLL(espeakng_loader.get_library_path())
        self.lib.espeak_Initialize.argtypes=[C.c_int,C.c_int,C.c_char_p,C.c_int]
        self.rate=self.lib.espeak_Initialize(2,0,espeakng_loader.get_data_path().encode(),0x8001)
        if self.rate <= 0 or self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError('eSpeak voice unavailable: '+voice)
        self.lib.espeak_Synth.argtypes=[C.c_void_p,C.c_size_t,C.c_uint,C.c_int,C.c_uint,C.c_uint,
            C.POINTER(C.c_uint),C.c_void_p]
        self.lib.espeak_SetParameter(1,150,0)
        self.chunks=[];self.events=[]
        callback_type=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_short),C.c_int,C.POINTER(Event))
        self.callback=callback_type(self.receive)
        self.lib.espeak_SetSynthCallback(self.callback)

    def receive(self, wav, count, events):
        if count and wav:
            self.chunks.append(np.ctypeslib.as_array(wav,shape=(count,)).copy())
        i=0
        while events and events[i].type:
            event=events[i]
            if event.type==7:
                self.events.append((event.audio_position/1000,event.id.string.decode(errors='replace')))
            i+=1
        return 0

    def speak(self,text):
        self.chunks=[];self.events=[]
        encoded=C.create_string_buffer(text.encode())
        uid=C.c_uint()
        result=self.lib.espeak_Synth(encoded,len(encoded),0,1,0,1,C.byref(uid),None)
        if result != 0 or not self.chunks:
            raise RuntimeError('formant synthesis failed')
        return np.concatenate(self.chunks).astype(np.float32)/32768, list(self.events)

    def close(self):
        self.lib.espeak_Terminate()


STEP=0.04


def cepstra(audio, rate):
    if rate != 16000:
        divisor=np.gcd(rate,16000)
        audio=signal.resample_poly(audio,16000//divisor,rate//divisor)
    f,t,z=signal.stft(audio,fs=16000,nperseg=512,noverlap=192)
    # 20 ms analysis, pooled to a 40 ms alignment lattice.
    mel=lambda hz:2595*np.log10(1+hz/700)
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(mel(80),mel(7600),42))
    filters=np.array([np.maximum(0,np.minimum((f-a)/(b-a),(c-f)/(c-b)))
        for a,b,c in zip(edges[:-2],edges[1:-1],edges[2:])])
    log_energy=np.log(filters@(np.abs(z)**2)+1e-10)
    coeff=fft.dct(log_energy,axis=0,norm='ortho')[1:14]
    return t[::2],coeff[:,::2].T.astype(np.float32)


def normalize(features):
    normalized=(features-np.mean(features,axis=0))/(np.std(features,axis=0)+0.1)
    return normalized/(np.linalg.norm(normalized,axis=1,keepdims=True)+1e-8)


def templates(lines,voice='en-us'):
    synth=FormantSynth(voice)
    features=[];line_ids=[-1];durations=[1.0];names=['silence']
    try:
        for line_id,text in enumerate(lines):
            audio,events=synth.speak(text)
            times,coeff=cepstra(audio,synth.rate)
            kept=0
            for i,(start,name) in enumerate(events):
                end=events[i+1][0] if i+1<len(events) else len(audio)/synth.rate
                if not name or name.startswith('_') or end-start < 0.012:
                    continue
                frames=(times>=start)&(times<end)
                feature=np.mean(coeff[frames],axis=0) if frames.any() else coeff[np.argmin(abs(times-start))]
                features.append(feature);line_ids.append(line_id);durations.append(end-start);names.append(name);kept+=1
            if not kept:
                raise ValueError('no phonemes for line '+str(line_id))
            line_ids.append(-1);durations.append(1.0);names.append('silence')
    finally:
        synth.close()
    speech_features=normalize(np.array(features))
    feature_matrix=np.zeros((len(line_ids),speech_features.shape[1]),dtype=np.float32)
    feature_matrix[np.array(line_ids)>=0]=speech_features
    return {'features':feature_matrix,'line_ids':np.array(line_ids),'durations':np.array(durations),
        'phonemes':names,'voice':voice}


def alignment_features(audio,rate):
    times,features=cepstra(np.mean(audio,axis=1),rate)
    cue_times,cue=activity(audio,rate)
    return times,normalize(features),np.interp(times,cue_times,cue)


def align_phonetic(times,features,cue,model,lines,activity_weight=0.5,duration_weight=0.2,
                   line_prior=None,prior_weight=0):
    if not lines:
        return []
    if not len(cue) or float(np.max(cue)) < 1e-5:
        return [{'text':line,'start':None,'end':None,'status':'unresolved_silence'} for line in lines]
    states=len(model['line_ids']);frames=len(times)
    silent=model['line_ids']<0
    if frames < states:
        raise ValueError('too few frames to visit all phonemes')
    if 'variance' in model:
        means=model['features'];variance=model['variance']
        costs=.5*(features**2@(1/variance).T-2*features@(means/variance).T+
                  np.sum(means**2/variance+np.log(variance),axis=1)[None,:])/features.shape[1]
        costs-=np.min(costs[:,~silent],axis=1,keepdims=True)
        costs=np.clip(costs,0,8).astype(np.float32)
    else:
        costs=(1-features@model['features'].T).astype(np.float32)
    costs[:,~silent]+=activity_weight*(1-cue[:,None])
    costs[:,silent]=activity_weight*cue[:,None]
    if line_prior is not None and prior_weight:
        for line_id, prior in enumerate(line_prior):
            if prior['start'] is None:
                continue
            ids=np.flatnonzero(model['line_ids']==line_id)
            durations=model['durations'][ids]
            fraction=(np.cumsum(durations)-durations*.5)/sum(durations)
            expected=prior['start']+(prior['end']-prior['start'])*fraction
            scale=max(4.0,prior['end']-prior['start'])
            costs[:,ids]+=prior_weight*((times[:,None]-expected[None,:])/scale)**2
    seconds_per_phone=float(np.sum(cue)*STEP/max(1,np.sum(~silent)))
    scale=seconds_per_phone/np.mean(model['durations'][~silent])
    expected=model['durations']*np.clip(scale,0.5,5)
    expected[silent]=1.0
    p=np.clip(np.exp(-STEP/np.maximum(expected,STEP)),0.01,0.99)
    stay=-duration_weight*np.log(p)
    advance=-duration_weight*np.log(1-p)
    prev=np.full(states,np.inf,dtype=np.float32);prev[0]=0
    parents=np.zeros((frames,states),dtype=np.uint8)
    skip=np.zeros(states,dtype=bool)
    skip[2:]=silent[1:-1] & ~silent[2:]
    for t in range(frames):
        c0=prev+stay
        c1=np.r_[np.inf,prev[:-1]+advance[:-1]]
        c2=np.r_[np.inf,np.inf,prev[:-2]+advance[:-2]]
        c2[~skip]=np.inf
        options=np.stack([c0,c1,c2])
        best=np.argmin(options,axis=0)
        prev=options[best,np.arange(states)]+costs[t]
        parents[t]=best
    state=states-1
    if not np.isfinite(prev[state]):
        raise ValueError('no complete phonetic path')
    path=np.empty(frames,dtype=np.int32)
    for t in reversed(range(frames)):
        path[t]=state
        state-=int(parents[t,state])
    mapped=model['line_ids'][path]
    predictions=[]
    for line_id,text in enumerate(lines):
        positions=np.flatnonzero(mapped==line_id)
        predictions.append({'text':text,'start':float(times[positions[0]]) if len(positions) else None,
            'end':float(times[positions[-1]]+STEP) if len(positions) else None,'status':'candidate_requires_review'})
    return predictions
