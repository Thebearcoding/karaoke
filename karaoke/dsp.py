"""Inspectible, non-neural batch audio separation candidates.

Repetition candidates use nearby musical context to estimate a repeating
background. No target stems, lyrics, song IDs or learned weights enter masks.
"""

import numpy as np
from scipy import linalg, ndimage, signal

RATE = 44100
FFT = 2048
HOP = 512


def stft(audio):
    return signal.stft(np.asarray(audio, dtype=np.float32).T, fs=RATE,
                       nperseg=FFT, noverlap=FFT-HOP, boundary="zeros")[2]


def istft(z, length):
    return signal.istft(z, fs=RATE, nperseg=FFT, noverlap=FFT-HOP,
                        boundary=True)[1].T[:length]


def apply_masks(z, masks):
    mid = (z[0] + z[1]) * (0.5 * masks[0])
    side = (z[0] - z[1]) * (0.5 * masks[1])
    if len(masks)>2:
        mid += (z[0]-z[1])*(0.5*masks[2])
    return np.stack([mid + side, mid - side])


def spatial_mask(z, strength=0.99, protection=0.5, coherence_power=2.0,
                 gain_power=1.0, side_strength=0.0):
    """Suppress correlated center energy, optionally applying its mask to side.

    Side coupling can remove spatial vocal residue but also affects instruments;
    it is an experimental tradeoff, not evidence of source identity.
    """
    mid, side = (z[0]+z[1])*0.5, (z[0]-z[1])*0.5
    def past(x):
        return signal.lfilter([0.25], [1, -0.75], x, axis=-1)
    mp, sp = np.abs(mid)**2, np.abs(side)**2
    dominance = past(mp)/(past(mp+sp)+1e-12)
    coherence = np.clip(past(np.real(z[0]*np.conj(z[1]))) / np.sqrt(
        past(np.abs(z[0])**2)*past(np.abs(z[1])**2)+1e-24), 0, 1)
    freq = np.fft.rfftfreq(FFT, 1/RATE)
    band = np.clip((freq-100)/120, 0, 1)*np.clip((10000-freq)/2500, 0, 1)
    suppression = strength*dominance*coherence**coherence_power*band[:, None]
    if protection:
        mag = np.abs(mid)
        broad = ndimage.median_filter(mag, size=(17, 1), mode="nearest")
        suppression *= 1-protection*np.clip(broad/(past(mag)+1e-8), 0, 1)**2
    gain = np.asarray(1-suppression, dtype=np.float32)
    if gain_power != 1:
        gain = gain**gain_power
    side_gain = 1-side_strength*(1-gain) if side_strength else 1.0
    return gain, side_gain


def repeating_background(z, neighbors=20, exclusion_seconds=0.5):
    """Estimate magnitude background from similar non-adjacent frames.

    Nearest-neighbor search uses pooled frequencies and at most ~22 frames/s.
    Chunking bounds similarity memory, including when processing full songs.
    """
    mag = np.sqrt(np.mean(np.abs(z)**2, axis=0))
    stride = 4
    coarse = mag[:, ::stride]
    edges = np.unique(np.geomspace(1, len(mag), 65).astype(int)-1)
    edges = np.r_[edges, len(mag)]
    features = np.stack([np.mean(coarse[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:])])
    features = np.sqrt(features)
    features /= np.maximum(np.linalg.norm(features, axis=0, keepdims=True), 1e-8)
    count = coarse.shape[1]
    exclusion = max(1, round(exclusion_seconds*RATE/HOP/stride))
    if count <= exclusion*2+2:
        return ndimage.median_filter(mag, size=(1, max(1, mag.shape[1]//2*2-1)))
    k = min(neighbors, max(1, count-exclusion*2-1))
    background = np.empty_like(coarse)
    for begin in range(0, count, 128):
        end = min(count, begin+128)
        similarity = features[:, begin:end].T @ features
        distance = np.abs(np.arange(begin, end)[:, None]-np.arange(count)[None, :])
        similarity[distance <= exclusion] = -np.inf
        index = np.argpartition(similarity, -k, axis=1)[:, -k:]
        background[:, begin:end] = np.median(coarse[:, index], axis=-1)
    full = np.empty_like(mag)
    for i in range(len(mag)):
        full[i] = np.interp(np.arange(mag.shape[1]), np.arange(count)*stride, background[i])
    return np.minimum(full, mag)


def repetition_masks(z, background, residual_weight=2, side_keep=0):
    mag = np.sqrt(np.mean(np.abs(z)**2, axis=0))
    residual = np.maximum(mag-background, 0)
    gain = background**2/(background**2+(residual_weight*residual)**2+1e-16)
    gain = np.clip(gain, 0, 1).astype(np.float32)
    return gain, side_keep+(1-side_keep)*gain


def lowrank_background(z, sparsity=1.0, iterations=20):
    """Principal-component pursuit on a pooled magnitude spectrogram.

    The low-rank term is a candidate background; sparse energy is not assumed
    to be guaranteed vocals. IALM shrinkage runs on CPU without learned weights.
    """
    mag = np.sqrt(np.mean(np.abs(z)**2, axis=0))
    small = ndimage.uniform_filter(mag, size=(2, 2), mode='nearest')[::2, ::2]
    # Whitening stops low-frequency instruments dominating the matrix fit.
    scale = np.maximum(np.mean(small, axis=1, keepdims=True), 1e-5)
    matrix = (small/np.sqrt(scale)).astype(np.float32)
    norm = float(np.linalg.norm(matrix))
    if norm < 1e-8:
        return np.zeros_like(mag)
    lam = sparsity/np.sqrt(max(matrix.shape))
    spectral_norm = float(linalg.svdvals(matrix)[0])
    dual_norm = max(spectral_norm, float(np.max(np.abs(matrix)))/lam)
    dual = matrix/dual_norm
    sparse = np.zeros_like(matrix)
    mu = 1.25/max(spectral_norm,1e-8)
    for _ in range(iterations):
        u, s, vh = linalg.svd(matrix-sparse+dual/mu, full_matrices=False, check_finite=False)
        keep = s > 1/mu
        low = (u[:,keep]*(s[keep]-1/mu)) @ vh[keep]
        residual = matrix-low+dual/mu
        sparse = np.sign(residual)*np.maximum(np.abs(residual)-lam/mu,0)
        error = matrix-low-sparse
        dual += mu*error
        if np.linalg.norm(error)/norm < 1e-5:
            break
        mu *= 1.5
    small_background = np.maximum(low,0)*np.sqrt(scale)
    row = np.minimum(np.arange(mag.shape[0])/2,small_background.shape[0]-1)
    col = np.minimum(np.arange(mag.shape[1])/2,small_background.shape[1]-1)
    coords = np.meshgrid(row,col,indexing='ij')
    background = ndimage.map_coordinates(small_background,coords,order=1,mode='nearest')
    return np.minimum(background,mag)


def harmonic_masks(z, width=35, side_keep=0.5):
    """Track a prominent centered harmonic source, then suppress its partials.

    Pitch is not identity: a solo instrument can also win this tracker. This
    ambiguity must be measured on real stems, especially instrumental passages.
    """
    mid=np.abs((z[0]+z[1])*0.5)
    side=np.abs((z[0]-z[1])*0.5)
    white=mid/(ndimage.median_filter(mid,size=(31,1),mode='nearest')+1e-5)
    white=np.log1p(np.clip(white,0,100))
    white=ndimage.maximum_filter1d(white,size=3,axis=0)
    midi=np.arange(40,85,1/3)
    pitches=440*2**((midi-69)/12)
    score=np.zeros((len(pitches),mid.shape[1]),dtype=np.float32)
    for partial in range(1,9):
        bins=np.rint(pitches*partial*FFT/RATE).astype(int)
        score+=white[np.minimum(bins,len(mid)-1)]/np.sqrt(partial)
    coarse=score[:,::4]
    top=np.argpartition(coarse,-6,axis=0)[-6:]
    values=np.take_along_axis(coarse,top,axis=0)
    values=(values-np.mean(values,axis=0))/(np.std(values,axis=0)+1e-3)
    previous=-values[:,0]
    parents=np.zeros_like(top,dtype=np.uint8)
    for t in range(1,coarse.shape[1]):
        jumps=np.abs(midi[top[:,t-1]][:,None]-midi[top[:,t]][None,:])
        costs=previous[:,None]+0.18*jumps
        parents[:,t]=np.argmin(costs,axis=0)
        previous=costs[parents[:,t],np.arange(6)]-values[:,t]
    state=int(np.argmin(previous));path=np.empty(coarse.shape[1])
    for t in reversed(range(len(path))):
        path[t]=pitches[top[state,t]]
        state=int(parents[state,t])
    f0=np.interp(np.arange(mid.shape[1]),np.arange(len(path))*4,path)
    frequencies=np.fft.rfftfreq(FFT,1/RATE)[:,None]
    nearest=np.rint(frequencies/f0[None,:])
    distance=np.abs(frequencies-nearest*f0[None,:])
    harmonics=np.exp(-.5*(distance/(width+.012*frequencies))**2)
    harmonics *= (nearest>=1)&(nearest<=24)&(frequencies>=110)&(frequencies<=10000)
    centered=np.clip(mid/(mid+side+1e-8),0,1)
    suppression=.99*harmonics*centered
    return (1-suppression).astype(np.float32),(1-(1-side_keep)*suppression).astype(np.float32)


def regression_masks(z, seconds=.8, bass_hz=120, floor=.06):
    """Predict correlated center accompaniment from the stereo difference.

    Locally regress mid on side, avoiding deletion of all centered audio. Purely
    centered instruments have no side evidence and remain a difficult case.
    """
    mid=(z[0]+z[1])*.5;side=(z[0]-z[1])*.5
    window=max(3,int(seconds*RATE/HOP)//2*2+1)
    cross=mid*np.conj(side)
    cross=ndimage.uniform_filter1d(cross.real,window,axis=-1)+1j*ndimage.uniform_filter1d(cross.imag,window,axis=-1)
    power=ndimage.uniform_filter1d(np.abs(side)**2,window,axis=-1)
    reference=ndimage.uniform_filter1d(np.abs(mid)**2,window,axis=-1)
    transfer=cross/(power+1e-10)
    transfer*=np.minimum(1,4/(np.abs(transfer)+1e-8))
    reliability=np.clip(np.abs(cross)**2/(power*reference+1e-20),0,1)
    transfer*=np.sqrt(reliability)
    frequencies=np.fft.rfftfreq(FFT,1/RATE)
    keep=floor+(1-floor)*np.clip((bass_hz+80-frequencies)/80,0,1)
    return keep[:,None].astype(np.float32),1.0,((1-keep[:,None])*transfer).astype(np.complex64)


def masks_for(z, config, background=None):
    method = config['method']
    if method == 'identity':
        return 1.0, 1.0
    if method == 'center-cancel':
        return 0.0, 1.0
    if method == 'spatial':
        return spatial_mask(z, config['strength'], config['protection'],
                            config.get('coherence_power', 2.0),
                            config.get('gain_power', 1.0), config.get('side_strength', 0.0))
    if method == 'harmonic':
        return harmonic_masks(z,config['width'],config['side_keep'])
    if method == 'regression':
        return regression_masks(z,config['seconds'],config['bass_hz'],config['floor'])
    if background is None:
        background = lowrank_background(z, config.get('sparsity',1)) if method == 'rpca' else repeating_background(z)
    masks = repetition_masks(z, background, config['residual_weight'], config['side_keep'])
    if method == 'hybrid':
        spatial, _ = spatial_mask(z, config['strength'], 0.25)
        return np.sqrt(masks[0]*spatial), masks[1]
    return masks


def separate(audio, config):
    z = stft(audio)
    masks = masks_for(z, config)
    backing = istft(apply_masks(z, masks), len(audio))
    return backing, np.asarray(audio)-backing
