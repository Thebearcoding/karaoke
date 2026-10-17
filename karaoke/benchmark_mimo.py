"""Run Sony's released deterministic MIMO BS-RoFormer on the fixed inputs.

Chunking/crossfade follows SonyResearch/mimo-audio-separation src/evaluate.py
at 9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5 (MIT, Copyright 2025 Yukara
Ikemiya). The full license is retained in ../licenses/mimo-MIT.txt.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from benchmark_neural import EXPERIMENT, ROOT, run_cases, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    research = ROOT/'research/2026-09-18'
    vendor = research/'vendor/mimo-audio-separation-9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5'
    sys.path[:0] = [str(research/'python-deps'), str(vendor/'src')]
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import hydra
    import torch
    from omegaconf import OmegaConf
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config_path = next((ROOT/'models/pretrained/mimo-large').rglob('config.yaml'))
    config = OmegaConf.load(config_path)
    # These unweighted training-only modules are not part of model.inference().
    config.model.discriminator = None
    config.model.loss_modules = {}
    config.model.loss_lambdas = {}
    started = time.perf_counter()
    model = hydra.utils.instantiate(config.model)
    weights = config_path.parent/'backbone_model.pth'
    model.backbone_model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    assert model.max_iter == 3 and model.mixture_consistency and not model.noise_addition
    # Validate using the actual eight-second inference window. Keep a short
    # probe too: relative error alone can be misleading on near-silent stems.
    x, rate = sf.read(EXPERIMENT/'inputs/lemon.wav', dtype='float32', always_2d=True)
    assert rate == 44100
    probe_specs = [('low-vocal-2s', 8, 2), ('low-vocal-8s', 8, 8), ('active-vocal-8s', 22, 8)]
    probes = [torch.from_numpy(x[start*rate:(start+seconds)*rate].T.copy()).unsqueeze(0)
              for _, start, seconds in probe_specs]
    with torch.inference_mode():
        cpu_outputs = [model.inference(probe, iters=[3])[0].numpy() for probe in probes]
        model.to('mps')
        mps_outputs = [model.inference(probe.to('mps'), iters=[3])[0].cpu().numpy() for probe in probes]
    diagnostics: list[dict[str, Any]] = []
    for (name, start, seconds), probe, cpu, mps in zip(probe_specs, probes, cpu_outputs, mps_outputs):
        errors = [float(np.linalg.norm(cpu[:, s]-mps[:, s])/max(np.linalg.norm(cpu[:, s]), 1e-8)) for s in range(2)]
        diagnostics.append({'probe': name, 'start_within_input_seconds': start,
            'probe_seconds': seconds, 'relative_l2_per_stem': errors,
            'reference_rms_per_stem': [float(np.sqrt(np.mean(cpu[:, s]**2))) for s in range(2)],
            'max_absolute_error': float(np.max(np.abs(cpu-mps))),
            'error_l2_relative_to_mixture': float(np.linalg.norm(cpu-mps)/np.linalg.norm(probe.numpy()))})
    (research/'mimo-port-numerics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    errors = diagnostics[-1]['relative_l2_per_stem']
    assert max(d['max_absolute_error'] for d in diagnostics) < 1e-5, diagnostics
    assert max(errors) < 1e-3, f'CPU/MPS discrepancy at actual window length: {diagnostics}'
    load_and_port_check_seconds = time.perf_counter()-started
    sample_size = 352800
    overlap = sample_size//2
    step = sample_size-overlap
    window = torch.bartlett_window(overlap*2)
    def infer(audio):
        length = len(audio)
        count = math.ceil(max(length-overlap, 0)/step)+1
        full_length = (count-1)*step+sample_size
        padded = torch.nn.functional.pad(torch.from_numpy(audio.T.copy()), (0, full_length-length))
        result = torch.zeros(2, 2, full_length)
        with torch.inference_mode():
            for i in range(count):
                offset = i*step
                batch = padded[:, offset:offset+sample_size].unsqueeze(0).to('mps')
                pred = model.inference(batch, iters=[3])[0][0].cpu()
                if i:
                    pred[..., :overlap] *= window[:overlap]
                if i != count-1:
                    pred[..., -overlap:] *= window[overlap:]
                result[..., offset:offset+sample_size] += pred
        result = result[..., :length].numpy()
        return result[1].T, result[0].T
    metadata = {'model': 'Sony MIMO BS-RoFormer large', 'iterations': 3,
        'weights_sha256': sha(weights), 'config_sha256': sha(config_path),
        'source_commit': '9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5',
        'source_port': str(research/'mimo-mps.patch'),
        'torch_version': torch.__version__, 'device': 'mps', 'dtype': 'float32',
        'chunk_samples': sample_size, 'hop_samples': step, 'padding': 'upstream end-zero padding',
        'training_only_modules_omitted': ['discriminator', 'loss_modules', 'loss_lambdas'],
        'cpu_mps_relative_l2_error_per_stem': errors,
        'load_and_port_check_seconds': load_and_port_check_seconds,
        'runner_sha256': sha(__file__), 'acceptance': 'comparison only; not accepted'}
    print(json.dumps({'cpu_mps_relative_l2_error_per_stem': errors}), flush=True)
    run_cases(infer, args.output, metadata)


if __name__ == '__main__':
    main()
