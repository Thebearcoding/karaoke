"""Render the frozen stronger candidate against the unchanged A reference."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp import RATE, separate
from experiment_output import output_directory


ROOT = Path(__file__).resolve().parent


def main():
    out = output_directory('listening-v003')
    old = ROOT/'experiments'/'listening-v002'
    previous = {r['id']: r for r in json.loads((old/'manifest.json').read_text())}
    songs = json.loads((ROOT/'samples'/'manifest.json').read_text())['songs']
    config = json.loads((ROOT/'experiments'/'separation-v007'/'frozen-config.json').read_text())
    (out/'answer-key.json').write_text(json.dumps({
        'A': json.loads((old/'answer-key.json').read_text())['configs']['spatial'],
        'D': config, 'scope': 'Nonblind diagnostic comparison, not quality approval',
    }, indent=2)+'\n')
    records = []
    for song in songs:
        folder = out/song['id']; folder.mkdir()
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i',
            str(ROOT/'samples'/song['file']), '-ac', '2', '-ar', str(RATE), '-f', 'f32le', 'pipe:1'])
        audio = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
        before = previous[song['id']]
        reference, rate = sf.read(old/song['id']/'A.flac', dtype='float32', always_2d=True)
        assert rate == RATE and reference.shape == audio.shape
        reference /= before['shared_gain']
        start = time.perf_counter()
        candidate, _ = separate(audio, config)
        seconds = time.perf_counter()-start
        assert candidate.shape == audio.shape and np.isfinite(candidate).all()
        peak = max(float(np.max(np.abs(x))) for x in [audio, reference, candidate])
        gain = min(before['shared_gain'], .95/max(peak, 1e-8))
        sf.write(folder/'D.flac', candidate*gain, RATE, subtype='PCM_24')
        begin = round(before['comparison_source_start_seconds']*RATE)
        pieces = []
        for version in [reference, candidate]:
            clip = (version[begin:begin+12*RATE]*gain).copy()
            ramp = np.linspace(0, 1, 441)[:, None]
            clip[:441] *= ramp; clip[-441:] *= ramp[::-1]
            pieces.extend([clip, np.zeros((RATE//2, 2))])
        sf.write(folder/'comparison.flac', np.concatenate(pieces), RATE, subtype='PCM_24')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(folder/'comparison.flac'),
            '-codec:a', 'libmp3lame', '-q:a', '2', str(folder/'comparison.mp3')], check=True)
        files = {}
        for name in ['D.flac', 'comparison.flac', 'comparison.mp3']:
            path = folder/name
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
                            '-f', 'null', '-'], check=True)
            files[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'decode_check': 'passed'}
        assert sf.info(folder/'D.flac').frames == len(audio)
        records.append({'id': song['id'], 'source_sha256': song['sha256'],
            'source_start_seconds': begin/RATE, 'clip_seconds': 12, 'gap_seconds': .5,
            'order': ['A', 'D'], 'shared_gain': gain, 'original_A_file': str(old/song['id']/'A.flac'),
            'D_processing_seconds': seconds, 'duration_seconds': len(audio)/RATE,
            'quality_status': 'Stronger candidate failed reference-stem quality gates; listening diagnostic only',
            'files': files})
        (out/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        print(song['id'], round(seconds, 2), 's; 3 files decoded', flush=True)


if __name__ == '__main__':
    main()
