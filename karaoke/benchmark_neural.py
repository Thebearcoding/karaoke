"""Run explicitly authorized pretrained comparisons at fixed original gain.

These outputs are candidates, never reference stems. Use the separate existing
audio-separator environment; no NN dependency is added to the heuristic runtime.
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import soundfile as sf
from stage_progress import progress_items, report_progress


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT/'experiments'/'neural-v001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snr(reference, error):
    return float(10*np.log10((np.sum(reference.astype(float)**2)+1e-20)/
                            (np.sum(error.astype(float)**2)+1e-20)))


def demix_batched(instance, mix, batch_size):
    """Batch single-target RoFormer windows without changing overlap or precision."""
    import torch
    from scipy.signal.windows import hamming
    cfg = instance.model_data_cfgdict
    if not instance.is_roformer or not cfg.training.target_instrument or instance.pitch_shift:
        raise ValueError('Batched inference requires a single-target RoFormer without pitch shift')
    chunk = int(cfg.model.stft_hop_length) * (int(cfg.inference.dim_t)-1)
    step = min(int(instance.overlap * cfg.audio.sample_rate), chunk)
    if batch_size < 1 or step < 1 or mix.shape[-1] < chunk:
        raise ValueError('Invalid batch, overlap, or unpadded audio')
    audio = torch.as_tensor(mix, dtype=torch.float32)
    result, counter = torch.zeros_like(audio), torch.zeros_like(audio)
    window = torch.tensor(hamming(chunk), dtype=torch.float32)
    # Preserve the pinned runner's final repeated window and addition order.
    starts = [min(i, audio.shape[-1]-chunk) for i in range(0, audio.shape[-1], step)]
    device = next(instance.model_run.parameters()).device
    for first in progress_items(range(0, len(starts), batch_size)):
        positions = starts[first:first+batch_size]
        inputs = torch.stack([audio[:, i:i+chunk] for i in positions]).to(device)
        with torch.inference_mode():
            outputs = instance.model_run(inputs).cpu()
        if outputs.shape != inputs.shape:
            raise ValueError('Unexpected single-target RoFormer output shape')
        for i, output in zip(positions, outputs):
            result[:, i:i+chunk] += output * window
            counter[:, i:i+chunk] += window
    primary = (result / counter.clamp(min=1e-10)).numpy()
    return {cfg.training.target_instrument: primary, instance.secondary_stem_name: mix-primary}


def run_cases(infer, output, metadata, cases_path=EXPERIMENT/'inputs.json'):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite experiment output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    metadata = dict(metadata, inputs_manifest=str(cases_path), inputs_manifest_sha256=sha(cases_path))
    (output/'model.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for case in json.loads(cases_path.read_text()):
        x, rate = sf.read(case['input'], dtype='float32', always_2d=True)
        if rate != 44100 or sha(case['input']) != case['input_sha256']:
            raise ValueError('Separation input rate or source hash differs from its manifest')
        started = time.perf_counter()
        separated = infer(x)
        backing, vocals = separated[:2]
        extra_stems = separated[2] if len(separated) == 3 else {}
        elapsed = time.perf_counter()-started
        if backing.shape != x.shape or vocals.shape != x.shape:
            raise ValueError('Separation outputs must preserve the input shape')
        if not np.isfinite(backing).all() or not np.isfinite(vocals).all():
            raise ValueError('Separation outputs contain non-finite samples')
        folder = output/case['id']; folder.mkdir()
        for label, wave in [('backing', backing), ('vocals', vocals)]:
            sf.write(folder/f'{label}.wav', wave, rate, subtype='FLOAT')
        for label, wave in extra_stems.items():
            if wave.shape != x.shape or not np.isfinite(wave).all():
                raise ValueError(f'Invalid additional model output: {label}')
            sf.write(folder/f'model-{label}.wav', wave, rate, subtype='FLOAT')
        row = {'id': case['id'], 'seconds': elapsed, 'rtf': elapsed/(len(x)/rate),
               'sum_reconstruction_snr_db': snr(x, x-backing-vocals),
               'backing_sha256': sha(folder/'backing.wav'), 'vocals_sha256': sha(folder/'vocals.wav')}
        if extra_stems:
            row['model_stems'] = {label: {'file': f'model-{label}.wav',
                'sha256': sha(folder/f'model-{label}.wav')} for label in extra_stems}
        if 'reference_backing' in case:
            truth, truth_rate = sf.read(case['reference_backing'], dtype='float32', always_2d=True)
            assert truth_rate == rate and truth.shape == x.shape
            row['reconstruction_db'] = snr(truth, backing-truth)
            row['input_reconstruction_db'] = snr(truth, x-truth)
            row['improvement_db'] = row['reconstruction_db']-row['input_reconstruction_db']
        records.append(row)
        (output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', type=Path,
                        help='Explicit local RoFormer YAML for a checkpoint outside the model registry')
    parser.add_argument('--target-stem', default='vocals', help='Selected model output, e.g. vocals or female')
    parser.add_argument('--export-stem', action='append', default=[],
                        help='Also save this raw model output, separately from the target complement')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=EXPERIMENT/'inputs.json')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--batch-size', type=int, default=1, help='Audio windows per forward pass')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.cache/numba'))
    import torch
    from audio_separator.separator import Separator
    from audio_separator.separator.architectures import mdxc_separator
    if version('audio-separator') != '0.44.5':
        raise ValueError('Inspected audio-separator 0.44.5 implementation required')
    if os.getenv('KARAOKE_EVENTS_STDOUT') == '1':
        # This pinned implementation uses tqdm only around its inference loop.
        mdxc_separator.tqdm = progress_items
    report_progress(detail='加载模型')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.perf_counter()
    separator = Separator(log_level=logging.WARNING, model_file_dir=str(ROOT/'models/pretrained'),
        output_dir=str(args.output), output_format='WAV', use_soundfile=True,
        normalization_threshold=1.0, amplification_threshold=0.0, use_autocast=False,
        mdxc_params={'segment_size': 256, 'override_model_segment_size': False,
                     'batch_size': args.batch_size, 'overlap': 2, 'pitch_shift': 0})
    separator.torch_device = torch.device(device)
    separator.torch_device_mps = torch.device('mps') if device == 'mps' else None
    original_loader = separator.load_model_data_from_yaml
    def typed_yaml(filename):
        config = original_loader(filename)
        config['is_roformer'] = True
        config['model_type'] = ('mel_band_roformer' if 'num_bands' in config['model'] else 'bs_roformer')
        return config
    separator.load_model_data_from_yaml = typed_yaml
    if args.config:
        checkpoint = (ROOT/'models/pretrained'/args.model).resolve(strict=True)
        config_path = args.config.resolve(strict=True)
        def local_model(filename):
            if filename != args.model:
                raise ValueError(f'Unexpected model requested: {filename}')
            return filename, 'MDXC', checkpoint.stem, str(checkpoint), str(config_path)
        separator.download_model_files = local_model
    separator.load_model(model_filename=args.model)
    instance = separator.model_instance
    if not instance.is_roformer or separator.torch_device.type != device:
        raise ValueError('Loaded separator model or device differs from the requested configuration')
    cfg = instance.model_data_cfgdict
    chunk = int(cfg.model.stft_hop_length)*(int(cfg.inference.dim_t)-1)
    # The inspected local 0.44.5 implementation treats `overlap` as HOP seconds,
    # not a fraction or overlap count. Explicitly request 50% window overlap.
    instance.overlap = (chunk//2)/44100
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    def infer(x):
        gain = min(1., .7/max(float(np.max(np.abs(x))), 1e-8))
        pad = chunk//2
        # External reflection avoids this implementation's negative indices
        # for inputs shorter than one window and protects the scored boundaries.
        padded = np.pad((x*gain).T, ((0, 0), (pad, pad)), mode='reflect')
        with torch.inference_mode():
            samples = np.ascontiguousarray(padded)
            stems = (instance.demix(samples) if args.batch_size == 1 else
                     demix_batched(instance, samples, args.batch_size))
        report_progress(detail='整理音轨')
        vocals = np.asarray(stems[args.target_stem], dtype='float32').T[pad:pad+len(x)]/gain
        if args.export_stem:
            extra = {name: np.asarray(stems[name], dtype='float32').T[pad:pad+len(x)]/gain
                     for name in args.export_stem}
            return x-vocals, vocals, extra
        return x-vocals, vocals
    metadata = {'model': args.model, 'target_stem': args.target_stem, 'exported_model_stems': args.export_stem,
        'output_semantics': {'vocals.wav': 'selected target stem', 'backing.wav': 'input minus selected target stem'},
        'model_sha256': sha(ROOT/'models/pretrained'/args.model),
        'audio_separator_version': version('audio-separator'), 'torch_version': torch.__version__,
        'device': device, 'batch_size': args.batch_size, 'load_seconds_excluding_imports': load_seconds,
        'chunk_samples': chunk, 'hop_samples': chunk//2, 'reflection_pad_samples_each_side': chunk//2,
        'input_gain': 'min(1, 0.7 / peak); restored for every stem',
        'export': 'Float WAV, no independent normalization or time alignment',
        'inference_file': inspect.getfile(mdxc_separator), 'inference_sha256': sha(inspect.getfile(mdxc_separator)),
        'architecture_file': inspect.getfile(instance.model_run.__class__),
        'architecture_sha256': sha(inspect.getfile(instance.model_run.__class__)),
        'runner_sha256': sha(__file__), 'acceptance': 'comparison only; not accepted'}
    if args.config:
        metadata['config'] = {'path': str(config_path), 'sha256': sha(config_path)}
    run_cases(infer, args.output, metadata, args.inputs)


if __name__ == '__main__':
    main()
