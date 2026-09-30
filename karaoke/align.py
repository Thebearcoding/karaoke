"""Experimental non-neural line alignment; outputs require review.

This first candidate combines text-derived syllable counts, center-channel
activity, and dynamic programming over phrase boundaries. It does not yet
recognize phonetic content and cannot establish that a particular word is sung.
"""

import re

import numpy as np
from scipy import ndimage, signal


def syllables(text):
    cjk = re.findall(r'[\u3040-\u30ff\u3400-\u9fff]', text)
    latin = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    count = len(cjk)
    for word in latin:
        count += max(1, len(re.findall(r'[aeiouy]+', re.sub(r'e$', '', word))))
    return max(1, count)


def activity(audio, rate):
    """Handcrafted activity cue, deliberately not called a voice detector."""
    f, t, z = signal.stft(audio.T, fs=rate, nperseg=1024, noverlap=768)
    mid = np.abs((z[0]+z[1])*0.5)
    side = np.abs((z[0]-z[1])*0.5)
    band = (f >= 180) & (f <= 4000)
    mag = mid[band]
    spatial = mag/(mag+side[band]+1e-8)
    broadband = ndimage.median_filter(mag, size=(15, 1), mode='nearest')
    harmonic = np.maximum(mag-0.6*broadband, 0)
    energy = np.sqrt(np.mean(harmonic**2*spatial, axis=0))
    low, high = np.percentile(energy, [15, 85])
    normalized = np.clip((energy-low)/(high-low+1e-9), 0, 1)
    normalized = ndimage.gaussian_filter1d(normalized, 0.04/(t[1]-t[0]))
    grid = np.arange(0, len(audio)/rate, 0.05)
    return grid, np.interp(grid, t, normalized)


def align_lines(times, cue, lines, boundary_weight=1, threshold=0.3, mode='dp'):
    if not lines:
        return []
    if len(times) < 2 or np.max(cue) < 1e-5:
        return [{'text':x, 'start':None, 'end':None, 'status':'unresolved_silence'} for x in lines]
    step = float(times[1]-times[0])
    weights = np.array([syllables(line) for line in lines], dtype=float)
    if mode == 'uniform':
        bounds = np.rint(np.r_[0, np.cumsum(weights)/weights.sum()]*(len(times)-1)).astype(int)
    else:
        voiced = np.maximum(cue-threshold, 0)
        if voiced.sum() < 1e-6:
            return [{'text':x, 'start':None, 'end':None, 'status':'unresolved_activity'} for x in lines]
        mass = np.r_[0, np.cumsum(voiced[:-1])]
        total = mass[-1]+1e-8
        target = total*weights/weights.sum()
        cumulative = np.r_[0, np.cumsum(weights)/weights.sum()]
        tolerance = max(0.07, 2/len(lines))
        candidates = []
        for fraction in cumulative:
            ix = np.flatnonzero(np.abs(mass/total-fraction) <= tolerance)
            if not len(ix):
                ix = np.array([int(np.argmin(np.abs(mass/total-fraction)))])
            # 100 ms grid plus local activity minima: bounded DP memory.
            minima = signal.find_peaks(-cue[ix])[0]
            ix = np.unique(np.r_[ix[::2], ix[minima], ix[-1]])
            if len(ix) > 500:
                ix = ix[np.linspace(0,len(ix)-1,500).astype(int)]
            candidates.append(ix)
        scores = 5*(mass[candidates[0]]/max(target[0],1))**2+boundary_weight*cue[candidates[0]]
        parents = []
        for i in range(len(lines)):
            left, right = candidates[i], candidates[i+1]
            observed = mass[right][None,:]-mass[left][:,None]
            duration = (right[None,:]-left[:,None])*step
            cost = ((observed-target[i])/max(target[i],1))**2
            cost += 0.03*((duration-max(weights[i]*0.25,0.5))/max(weights[i]*0.25,0.5))**2
            cost += boundary_weight*cue[right][None,:]
            cost[duration < 0.25] = np.inf
            combined = scores[:,None]+cost
            best = np.argmin(combined,axis=0)
            scores = combined[best,np.arange(len(right))]
            parents.append(best)
        scores += 5*((total-mass[candidates[-1]])/max(target[-1],1))**2
        if not np.isfinite(scores).any():
            return [{'text':x,'start':None,'end':None,'status':'unresolved_path'} for x in lines]
        j = int(np.argmin(scores))
        bounds = [int(candidates[-1][j])]
        for i in reversed(range(len(lines))):
            j = int(parents[i][j])
            bounds.append(int(candidates[i][j]))
        bounds.reverse()
    result = []
    for text, begin, end in zip(lines,bounds[:-1],bounds[1:]):
        active = np.flatnonzero(cue[begin:end+1] > threshold)
        if len(active) and mode != 'uniform':
            start_time = times[min(begin+int(active[0]),len(times)-1)]
            end_time = times[min(begin+int(active[-1]),len(times)-1)]+step
        else:
            start_time,end_time = times[begin],times[end]
        result.append({'text':text,'start':float(start_time),'end':float(end_time),
            'status':'candidate_requires_review'})
    return result
