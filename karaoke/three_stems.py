"""Split an accepted vocal master into lead/backing vocals, preserving music."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf

from exact_stems import (decode_source, encode_previews, complementary_split, native_vocals,
                         read_masters, reconstruction_report, sha, source_samples)


def check_three(source, lead, backing_vocals, instrumental):
    """Record the vocal-first mix's numerical error without a bitwise gate."""
    return {
        'vocal_first': reconstruction_report(source, lead + backing_vocals, instrumental),
    }


def build(existing, estimate, output, video=None, *, source_pcm=None, karaoke_preview=True):
    existing, estimate, output = (Path(p).resolve() for p in (existing, estimate, output))
    if output.exists():
        raise FileExistsError(f'Preserve existing output: {output}')
    started = time.perf_counter()
    old = json.loads((existing/'manifest.json').read_text())
    spec = old['source']
    if sha(spec['path']) != spec['container_sha256']:
        raise ValueError('Original source changed')
    for master in old['masters'].values():
        if sha(existing/master['file']) != master['sha256']:
            raise ValueError(f"Existing master changed: {master['file']}")
    source, rate, _ = source_samples(spec['path'], source_pcm)
    vocals, instrumental = read_masters(existing, rate, source.shape)
    reconstruction_report(source, vocals, instrumental)
    raw_lead, resampling = native_vocals(estimate, rate, source.shape)
    lead, backing_vocals = complementary_split(vocals, raw_lead)
    output.mkdir(parents=True)
    for name, audio in [('lead', lead), ('backing-vocals', backing_vocals)]:
        sf.write(output/f'{name}.wav', audio, rate, subtype='DOUBLE')
    shutil.copy2(existing/'backing.wav', output/'instrumental.wav')
    # Read actual serialized masters, including the copied instrumental.
    tracks = {}
    for name in ('lead', 'backing-vocals', 'instrumental'):
        path = output/f'{name}.wav'
        audio, actual_rate = sf.read(path, dtype='float64', always_2d=True)
        if actual_rate != rate or audio.shape != source.shape or sf.info(path).subtype != 'DOUBLE':
            raise ValueError(f'Incorrect master format: {path}')
        tracks[name] = audio
    checks = check_three(source, tracks['lead'], tracks['backing-vocals'], tracks['instrumental'])
    if sha(output/'instrumental.wav') != old['masters']['backing']['sha256']:
        raise ArithmeticError('Instrumental must be an unchanged copy')
    karaoke = tracks['instrumental'] + tracks['backing-vocals']
    peak = max(float(np.max(np.abs(a))) for a in (*tracks.values(), source, karaoke))
    gain = min(1., .84/max(peak, 1e-12))
    temporary_mix = output/'.karaoke-mix.wav'
    if karaoke_preview or video:
        sf.write(temporary_mix, karaoke, rate, subtype='FLOAT')
    preview_inputs = {name: output/f'{name}.wav' for name in tracks}
    if karaoke_preview:
        preview_inputs['karaoke'] = temporary_mix
    gain, previews = encode_previews(preview_inputs, output, gain, len(source), source.shape[1])
    video_record = None
    if video:
        video = Path(video).resolve()
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video),
            '-i', str(temporary_mix), '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-af', f'volume={gain:.17g}:precision=double',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(output/'karaoke.mp4')]
        subprocess.run(command, check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(output/'karaoke.mp4'),
                        '-f', 'null', '-'], check=True)
        video_record = {'file': 'karaoke.mp4', 'source_video': str(video),
            'source_sha256': sha(video), 'sha256': sha(output/'karaoke.mp4'),
            'command': command, 'full_decode_passed': True,
            'video': 'Stream copy of the existing hard-cut, aligned-subtitle render'}
    temporary_mix.unlink(missing_ok=True)
    manifest = {'schema_version': 1, 'source': spec,
        'existing_package': {'path': str(existing), 'manifest_sha256': sha(existing/'manifest.json')},
        'lead_estimate': {'path': str(estimate), 'sha256': sha(estimate), **resampling},
        'masters': {name: {'file': f'{name}.wav', 'sha256': sha(output/f'{name}.wav'),
                          'format': 'WAV DOUBLE; unscaled'} for name in tracks},
        'semantics': {'lead': 'Karaoke model lead-vocal estimate; original reverb retained',
            'backing-vocals': 'Existing full vocal master minus the lead estimate',
            'instrumental': 'Unchanged existing instrumental master',
            'karaoke': 'Instrumental plus backing vocals'},
        'reconstruction_order': '(lead + backing-vocals) + instrumental, float64',
        'reconstruction': checks,
        'preview_shared_gain': gain, 'previews': previews, 'video': video_record,
        'purity': 'Not established by reconstruction. All three tracks require listening.',
        'packaging_seconds': time.perf_counter()-started, 'code_sha256': sha(__file__)}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def verify(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    spec = manifest['source']
    if sha(spec['path']) != spec['container_sha256']:
        raise ValueError('Original source changed')
    source, rate, _ = decode_source(spec['path'])
    tracks = {}
    for name, record in manifest['masters'].items():
        path = output/record['file']
        if sha(path) != record['sha256']:
            raise ValueError(f'Master changed: {path}')
        tracks[name], actual_rate = sf.read(path, dtype='float64', always_2d=True)
        if rate != actual_rate or tracks[name].shape != source.shape or sf.info(path).subtype != 'DOUBLE':
            raise ValueError(f'Incorrect master format: {path}')
    checks = check_three(source, tracks['lead'], tracks['backing-vocals'], tracks['instrumental'])
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('--existing', type=Path, required=True)
    create.add_argument('--lead-estimate', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--video', type=Path)
    create.add_argument('--source-pcm', type=Path)
    create.add_argument('--no-karaoke-preview', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (build(args.existing, args.lead_estimate, args.output, args.video, source_pcm=args.source_pcm,
                    karaoke_preview=not args.no_karaoke_preview)
              if args.command == 'build' else verify(args.output))
    print(json.dumps(result, indent=2))
