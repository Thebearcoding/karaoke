"""Small classical Gaussian/HMM acoustic experiment, never a neural network.

Only calibration-song manual WORD times supervise parameter estimation. Test
and product inference receive audio and untimed text, never reference times.
"""

import re

import cmudict
import numpy as np


DICTIONARY=cmudict.dict()
VOWELS={'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW'}


def phones(word):
    key=word.lower().strip('.,!?;:"()')
    options=DICTIONARY.get(key)
    return [re.sub('[0-9]','',p) for p in options[0]] if options else ['AH']


def train(word_groups, iterations=4):
    names=sorted({p for seq,frames in word_groups for p in seq})
    index={name:i for i,name in enumerate(names)}
    groups=[]
    for seq,frames in word_groups:
        if len(frames)<len(seq):continue
        prior=np.array([1 if p in VOWELS else .6 for p in seq])
        bounds=np.rint(np.r_[0,np.cumsum(prior)/sum(prior)]*len(frames)).astype(int)
        labels=np.concatenate([np.repeat(i,b-a) for i,(a,b) in enumerate(zip(bounds[:-1],bounds[1:]))])
        groups.append((np.array([index[p] for p in seq]),frames,labels))
    if not groups:raise ValueError('No complete training words')
    all_frames=np.concatenate([g[1] for g in groups])
    global_mean=np.mean(all_frames,axis=0);global_var=np.var(all_frames,axis=0)+1e-3
    history=[]
    for iteration in range(iterations+1):
        buckets=[[] for _ in names];durations=[[] for _ in names]
        for ids,frames,labels in groups:
            for i,pid in enumerate(ids):
                rows=frames[labels==i]
                if len(rows):buckets[pid].append(rows);durations[pid].append(len(rows)*.04)
        means=[];variance=[];seconds=[]
        for rows,lens in zip(buckets,durations):
            data=np.concatenate(rows) if rows else global_mean[None,:]
            confidence=len(data)/(len(data)+20)
            means.append(confidence*np.mean(data,axis=0)+(1-confidence)*global_mean)
            variance.append(.75*np.var(data,axis=0)+.25*global_var+1e-3)
            seconds.append(float(np.median(lens)) if lens else .1)
        means=np.array(means);variance=np.array(variance)
        if iteration==iterations:break
        updated=[];loss=0
        for ids,frames,old_labels in groups:
            m,v=means[ids],variance[ids]
            cost=.5*(frames**2@(1/v).T-2*frames@(m/v).T+np.sum(m*m/v+np.log(v),axis=1)[None,:])
            prev=np.full(len(ids),np.inf);prev[0]=cost[0,0]
            parents=np.zeros(cost.shape,dtype=bool)
            for t in range(1,len(frames)):
                advance=np.r_[np.inf,prev[:-1]]
                parents[t]=advance<prev
                prev=np.minimum(prev,advance)+cost[t]
            j=len(ids)-1;labels=np.zeros(len(frames),dtype=int)
            for t in reversed(range(len(frames))):
                labels[t]=j
                if t and parents[t,j]:j-=1
            loss+=float(prev[-1]);updated.append((ids,frames,labels))
        groups=updated;history.append({'iteration':iteration+1,'training_nll':loss})
    return {'names':names,'means':means,'variance':variance,'seconds':np.array(seconds),
        'history':history,'training_words_used':len(groups),'training_frames':len(all_frames)}


def make_model(lines,acoustic):
    index={p:i for i,p in enumerate(acoustic['names'])}
    ids=[];line_ids=[];durations=[];names=[];unknown=[]
    def silence():
        ids.append(-1);line_ids.append(-1);durations.append(1);names.append('silence')
    silence()
    for line_id,line in enumerate(lines):
        for word in line.split():
            if word.lower().strip('.,!?;:"()') not in DICTIONARY:unknown.append(word)
            for phone in phones(word):
                pid=index.get(phone,index.get('AH',0))
                ids.append(pid);line_ids.append(line_id);durations.append(acoustic['seconds'][pid]);names.append(phone)
        silence()
    features=np.zeros((len(ids),acoustic['means'].shape[1]),dtype=np.float32)
    variance=np.ones_like(features)
    for i,pid in enumerate(ids):
        if pid>=0:features[i]=acoustic['means'][pid];variance[i]=acoustic['variance'][pid]
    return {'features':features,'variance':variance,'line_ids':np.array(line_ids),
        'durations':np.array(durations),'phonemes':names,'unknown_words':unknown}
