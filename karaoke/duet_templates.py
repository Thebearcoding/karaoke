"""Use solo passages in the same recording as spectral dictionaries for a duet.

This is non-neural NMF. Lyric singer labels select enrollment passages and route
solo sections; only simultaneous singing uses the learned spectral masks.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import stft, istft
import soundfile as sf


def learn_dictionary(values, rank, rng, iterations=70):
    indices = rng.choice(values.shape[1], min(1800, values.shape[1]), replace=False)
    values = np.ascontiguousarray(values[:, indices])
    basis = values[:, rng.choice(values.shape[1], rank, replace=False)].copy()+1e-6
    basis /= basis.sum(axis=0)
    activation = np.maximum(basis.T@values, 1e-6)
    for _ in range(iterations):
        ratio = values/(basis@activation+1e-8)
        activation *= (basis.T@ratio)/(basis.sum(axis=0)[:, None]+1e-8)
        ratio = values/(basis@activation+1e-8)
        basis *= (ratio@activation.T)/(activation.sum(axis=1)[None, :]+1e-8)
        scale = basis.sum(axis=0)
        basis /= scale
        activation *= scale[:, None]
    return basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--lyrics', type=Path, required=True)
    parser.add_argument('--cues', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir()
    started = time.perf_counter()
    audio, rate = sf.read(args.audio, dtype='float32', always_2d=True)
    lyrics = json.loads(args.lyrics.read_text())
    cues = json.loads(args.cues.read_text())
    assert len(cues) == len(lyrics['line_singers'])
    assert [c['text'] for c in cues] == lyrics['lines']
    _, times, spectrum = stft(audio.T, rate, nperseg=2048, noverlap=1536)
    magnitude = np.sqrt(np.mean(np.abs(spectrum)**2, axis=0)).astype('float32')
    weight = 1/np.maximum(gaussian_filter1d(magnitude.mean(axis=1), 8), 1e-5)**.6
    values = np.ascontiguousarray(magnitude*weight[:, None])
    rng = np.random.default_rng(42)
    bases, enrollment, holdout = [], [], []
    for singer in ['孙楠', '韩红']:
        lines = [i for i, label in enumerate(lyrics['line_singers']) if label == singer]
        training_lines, validation_lines = lines[::2], lines[1::2]
        selected = np.zeros(len(times), dtype=bool)
        for i in training_lines:
            selected |= (times >= cues[i]['start']+.3)&(times <= cues[i]['end']-.3)
        selected &= magnitude.sum(axis=0) > np.quantile(magnitude.sum(axis=0), .3)
        bases.append(learn_dictionary(values[:, selected], args.rank, rng))
        enrollment.append({'singer': singer, 'lines': [i+1 for i in training_lines],
                           'frames': int(selected.sum())})
        holdout.extend((i, singer) for i in validation_lines)
        print(json.dumps({'learned': singer, 'frames': int(selected.sum())}, ensure_ascii=False), flush=True)
    basis = np.concatenate(bases, axis=1)
    estimate = np.zeros((2, *values.shape), dtype='float32')
    for start in range(0, len(times), 2048):
        stop = start+2048
        target = values[:, start:stop]
        activation = np.maximum(basis.T@target, 1e-6)
        for _ in range(45):
            activation *= (basis.T@(target/(basis@activation+1e-8)))/(basis.sum(axis=0)[:, None]+1e-8)
        for k in range(2):
            estimate[k, :, start:stop] = bases[k]@activation[k*args.rank:(k+1)*args.rank]
    powers = estimate**2
    raw_mask = powers[1]/(powers.sum(axis=0)+1e-10)
    raw_mask = np.clip(gaussian_filter(raw_mask, (.65, .8)), 0, 1)
    diagnostics = []
    energy = magnitude**2
    for i, singer in holdout:
        selected = (times >= cues[i]['start']+.3)&(times <= cues[i]['end']-.3)
        fraction = np.sum(energy[:, selected]*raw_mask[:, selected]**2)/(np.sum(energy[:, selected])+1e-12)
        diagnostics.append({'line': i+1, 'singer': singer, 'fraction_of_vocal_energy_assigned_to_female': float(fraction)})
    # Keep known solos out of the underdetermined simultaneous-voice problem.
    feminine = np.zeros(len(times), dtype='float32')
    duet = np.ones(len(times), dtype='float32')
    # Keep the preceding singer's sustained tail until near the next onset.
    boundaries = [0.]+[max(0., c['start']-.35) for c in cues[1:]]+[len(audio)/rate]
    for i, singer in enumerate(lyrics['line_singers']):
        selected = (times >= boundaries[i])&(times < boundaries[i+1])
        feminine[selected] = singer == '韩红'
        duet[selected] = singer == '合唱'
    feminine = gaussian_filter1d(feminine, 2)
    duet = gaussian_filter1d(duet, 2)
    mask = np.clip(feminine[None, :]+duet[None, :]*raw_mask, 0, 1)
    _, removed = istft(spectrum*mask[None, :, :], rate, nperseg=2048, noverlap=1536)
    removed = removed.T[:len(audio)].astype('float32')
    sf.write(args.output/'female.wav', removed, rate, subtype='FLOAT')
    sf.write(args.output/'remaining-vocals.wav', audio-removed, rate, subtype='FLOAT')
    np.savez_compressed(args.output/'dictionary.npz', male=bases[0], female=bases[1], weight=weight)
    report = {'method': 'same-song NMF spectral dictionaries; no pretrained speaker model',
        'rank_per_singer': args.rank, 'seconds': time.perf_counter()-started, 'enrollment': enrollment,
        'role_switch_preroll_seconds': .35, 'audio_transition_sigma_frames': 2,
        'heldout_solo_diagnostics': diagnostics,
        'limitations': 'Singer annotations route solos. Heldout solo discrimination is not a ground-truth metric for overlap separation.'}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
