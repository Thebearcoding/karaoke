"""Package unmodified vocal estimates and their complement as float64 WAVs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pcm_sha(audio):
    data = np.ascontiguousarray(audio, dtype='<f8')
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()


def reconstruction_report(source, vocals, backing):
    """Record numerical error for diagnosis; it is not an acceptance gate."""
    source = np.asarray(source, dtype='float64')
    vocals = np.asarray(vocals, dtype='float64')
    backing = np.asarray(backing, dtype='float64')
    if source.shape != vocals.shape or source.shape != backing.shape:
        raise ValueError('Source and both stems must have identical shapes')
    if not all(np.isfinite(a).all() for a in (source, vocals, backing)):
        raise ValueError('Non-finite audio')
    combined = vocals+backing
    if not np.isfinite(combined).all():
        raise ValueError('Non-finite audio')
    return {'sample_values': int(source.size),
            'max_absolute_error': float(np.max(np.abs(combined-source))),
            'source_pcm_sha256': pcm_sha(source)}


def complementary_split(source, estimate):
    source = np.asarray(source, dtype='float64')
    vocals = np.array(estimate, dtype='float64', copy=True)
    if source.ndim != 2 or not source.size or source.shape != vocals.shape:
        raise ValueError('Expected equally sized, nonempty frames-by-channels audio')
    if not np.isfinite(source).all() or not np.isfinite(vocals).all():
        raise ValueError('Non-finite audio')
    backing = source-vocals
    if not np.isfinite(backing).all():
        raise ValueError('Non-finite audio')
    return vocals, backing


def decode_source(path):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,sample_fmt,codec_name', '-of', 'json', str(path)]))
    stream = probe['streams'][0]
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
               '-c:a', 'pcm_f64le', '-f', 'f64le', 'pipe:1']
    raw = subprocess.check_output(command)
    audio = np.frombuffer(raw, dtype='<f8').reshape(-1, stream['channels'])
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Source is empty or contains non-finite audio')
    return audio, int(stream['sample_rate']), {'stream': stream, 'command': command}


def cache_source(path, pcm):
    """Keep the exact native decode for both packaging stages in this attempt."""
    pcm = Path(pcm)
    audio, rate, decoder = decode_source(path)
    sf.write(pcm, audio, rate, subtype='DOUBLE')
    record = dict(source_sha256=sha(path), sha256=sha(pcm), sample_rate=rate,
                  frames=len(audio), channels=audio.shape[1], decoder=decoder)
    pcm.with_suffix('.json').write_text(json.dumps(record)+'\n')
    return record


def source_samples(path, pcm=None):
    if pcm is None:
        return decode_source(path)
    pcm = Path(pcm)
    record = json.loads(pcm.with_suffix('.json').read_text())
    if record['source_sha256'] != sha(path) or record['sha256'] != sha(pcm):
        raise ValueError('Cached native audio or its source changed')
    audio, rate = sf.read(pcm, dtype='float64', always_2d=True)
    if (rate != record['sample_rate'] or audio.shape != (record['frames'], record['channels'])
            or sf.info(pcm).subtype != 'DOUBLE' or not audio.size or not np.isfinite(audio).all()):
        raise ValueError('Invalid cached native audio')
    return audio, rate, record['decoder']


def native_vocals(path, rate, shape):
    vocals, estimate_rate = sf.read(path, dtype='float32', always_2d=True)
    if vocals.shape[1] != shape[1]:
        raise ValueError('Vocal estimate channel count differs from source')
    if abs(len(vocals)/estimate_rate-shape[0]/rate) > 2/min(rate, estimate_rate):
        raise ValueError('Vocal estimate does not cover the complete source duration')
    divisor = int(np.gcd(rate, estimate_rate))
    up, down = rate//divisor, estimate_rate//divisor
    if up != down:
        vocals = resample_poly(vocals, up, down, axis=0)
    delta = len(vocals)-shape[0]
    if delta < 0:
        vocals = np.pad(vocals, ((0, -delta), (0, 0)))
    return vocals[:shape[0]], {'sample_rate': estimate_rate, 'resampling': 'scipy.signal.resample_poly',
        'up': up, 'down': down, 'window': ['kaiser', 5.0], 'padtype': 'constant',
        'length_difference_before_endpoint_fit': delta}


def read_masters(output, rate, shape):
    waves = []
    for name in ('vocals', 'backing'):
        path = output/f'{name}.wav'
        info = sf.info(path)
        audio, actual_rate = sf.read(path, dtype='float64', always_2d=True)
        if info.subtype != 'DOUBLE' or actual_rate != rate or audio.shape != shape:
            raise ValueError(f'Incorrect master format or time axis: {path}')
        waves.append(audio)
    return waves


class PreviewClippingError(ValueError):
    def __init__(self, path, peak):
        self.peak = peak
        super().__init__(f'Clipped listening preview ({peak:g}): {path}')


def validate_preview(path, frames, channels, sample_rate):
    """Accept codec tail padding, while rejecting missing, shifted-format or clipped audio."""
    decoded, rate, _ = decode_source(path)
    delta = len(decoded)-frames
    peak = float(np.max(np.abs(decoded)))
    if rate != sample_rate or decoded.shape[1] != channels or not 0 <= delta <= 1152:
        raise ValueError(f'Invalid length or format of listening preview: {path}')
    padding_peak = float(np.max(np.abs(decoded[frames:]), initial=0.))
    if padding_peak > 1e-7:
        raise ValueError(f'Extra preview samples are not silent codec padding: {path}')
    if peak >= 1:
        raise PreviewClippingError(path, peak)
    return dict(file=Path(path).name, sha256=sha(path), sample_rate=rate, frames=frames,
                decoded_frames=len(decoded), decoded_length_delta_samples=delta,
                decoded_peak=peak, padding_peak=padding_peak)


def encode_preview(source, output, gain, frames, channels):
    info = sf.info(source)
    if (info.frames, info.channels) != (frames, channels):
        raise ValueError('Preview input must preserve the complete master time axis')
    # The LAME frontend records the full end padding, including padding that
    # spans two MP3 frames. FFmpeg's wrapper can truncate that count.
    with tempfile.TemporaryDirectory(prefix='.preview-', dir=Path(output).parent) as directory:
        scaled = Path(directory)/'scaled.wav'
        # libsndfile-enabled LAME frontends peak-normalize floating WAV input.
        # Integer PCM preserves the shared gain on both Linux and macOS.
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source),
            '-af', f'volume={gain:.17g}:precision=double', '-c:a', 'pcm_s24le', str(scaled)], check=True)
        subprocess.run(['lame', '--silent', '--noreplaygain', '-V', '2', str(scaled), str(output)], check=True)
    return validate_preview(output, frames, channels, info.samplerate)


def encode_previews(sources, output, gain, frames, channels):
    """Re-encode the whole group at one gain if lossy reconstruction overshoots."""
    for _ in range(3):
        previews, clipped = {}, []
        with ThreadPoolExecutor(max_workers=min(2, len(sources))) as pool:
            pending = {name: pool.submit(encode_preview, source, Path(output)/f'{name}.mp3', gain, frames, channels)
                       for name, source in sources.items()}
            for name, future in pending.items():
                try:
                    previews[name] = future.result()
                except PreviewClippingError as error:
                    clipped.append(error)
        if not clipped:
            return gain, previews
        worst = max(clipped, key=lambda error: error.peak)
        gain *= .9/worst.peak
    raise worst


def build(source_path, estimate_path, output, *, previews=True, source_pcm=None):
    source_path, estimate_path, output = (Path(p).resolve() for p in (source_path, estimate_path, output))
    if output.exists():
        raise FileExistsError(f'Preserve existing stem package: {output}')
    started = time.perf_counter()
    source, rate, decoder = source_samples(source_path, source_pcm)
    estimate, resampling = native_vocals(estimate_path, rate, source.shape)
    vocals, backing = complementary_split(source, estimate)
    output.mkdir(parents=True)
    for name, audio in (('vocals', vocals), ('backing', backing)):
        sf.write(output/f'{name}.wav', audio, rate, subtype='DOUBLE')
    # Validate the actual serialized files, not only arrays before export.
    written = read_masters(output, rate, source.shape)
    check = reconstruction_report(source, *written)
    gain, preview_records = None, {}
    if previews:
        peak = max(float(np.max(np.abs(a))) for a in (source, *written))
        gain = min(1., .9/max(peak, 1e-12))
        gain, preview_records = encode_previews({name: output/f'{name}.wav' for name in ('vocals', 'backing')},
                                               output, gain, len(source), source.shape[1])
    manifest = {'schema_version': 1, 'source': {'path': str(source_path),
        'container_sha256': sha(source_path), 'decoder': decoder,
        'decoder_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'pcm_format': 'native-rate, interleaved little-endian float64; no gain or resampling',
        'sample_rate': rate, 'channels': source.shape[1], 'frames': len(source)},
        'estimate': {'path': str(estimate_path), 'sha256': sha(estimate_path), **resampling},
        'masters': {name: {'file': f'{name}.wav', 'sha256': sha(output/f'{name}.wav'),
                          'format': 'WAV DOUBLE; unscaled'} for name in ('vocals', 'backing')},
        'reconstruction': check,
        'preview_shared_gain': gain, 'previews': preview_records,
        'purity': 'Not established by reconstruction; check vocals and backing separately.',
        'packaging_seconds': time.perf_counter()-started, 'code_sha256': sha(__file__)}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def verify(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    spec = manifest['source']
    if sha(spec['path']) != spec['container_sha256']:
        raise ValueError('Source file has changed')
    for master in manifest['masters'].values():
        if sha(output/master['file']) != master['sha256']:
            raise ValueError(f"Master file has changed: {master['file']}")
    source, rate, _ = decode_source(spec['path'])
    if rate != spec['sample_rate'] or source.shape != (spec['frames'], spec['channels']):
        raise ValueError('Source time axis or channels have changed')
    check = reconstruction_report(source, *read_masters(output, rate, source.shape))
    if check['source_pcm_sha256'] != manifest['reconstruction']['source_pcm_sha256']:
        raise ValueError('Decoded source differs from the recorded reference')
    return check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    create.add_argument('--source', required=True, type=Path)
    create.add_argument('--vocals-estimate', required=True, type=Path)
    create.add_argument('--output', required=True, type=Path)
    create.add_argument('--no-previews', action='store_true', help='Build and verify only the lossless masters')
    create.add_argument('--source-pcm', type=Path)
    decode = commands.add_parser('decode')
    decode.add_argument('--source', required=True, type=Path)
    decode.add_argument('--output', required=True, type=Path)
    check = commands.add_parser('verify')
    check.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = (build(args.source, args.vocals_estimate, args.output, previews=not args.no_previews, source_pcm=args.source_pcm)
              if args.command == 'build' else cache_source(args.source, args.output) if args.command == 'decode'
              else verify(args.output))
    print(json.dumps(result, indent=2))
