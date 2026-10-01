"""Reference-based separation scores with fixed roles, gain and sample alignment.

These require independently recorded stems. Model outputs are never references.
SI scores remove each channel's DC offset and flatten channels; fixed-gain
reconstruction and output levels use the original PCM. Ratios are capped at
±300 dB only to keep exact silence/perfect matches JSON-serializable.
"""
import numpy as np


def db_ratio(numerator, denominator):
    if numerator <= 0:
        return -300. if denominator > 0 else None
    if denominator <= 0:
        return 300.
    return float(np.clip(10*np.log10(numerator/denominator), -300., 300.))


def _window_scores(reference, estimates):
    names = list(reference)
    sources = [reference[n]-reference[n].mean(axis=0) for n in names]
    gram = np.array([[np.sum(a*b) for b in sources] for a in sources])
    active = np.diag(gram) > 0
    active_gram = gram[np.ix_(active, active)]
    scales = np.sqrt(np.diag(active_gram))
    identifiable = bool(len(scales) and np.linalg.cond(
        active_gram/np.outer(scales, scales)) < 1e8)
    rows = {}
    for i, name in enumerate(names):
        source, output = reference[name], estimates[name]
        ref_energy = float(np.sum(source**2))
        output_energy = float(np.sum(output**2))
        row = dict(reference_silent=ref_energy == 0,
                   reference_dbfs=db_ratio(ref_energy, source.size),
                   output_dbfs=db_ratio(output_energy, output.size),
                   fixed_gain_db=db_ratio(ref_energy, float(np.sum((output-source)**2)))
                   if ref_energy else None,
                   si_sdr_db=None, si_sir_db=None, si_sar_db=None, target_gain=None)
        y = output-output.mean(axis=0)
        energy = float(np.sum(y**2))
        if active[i]:
            products = np.array([np.sum(s*y) for s in sources])
            target = float(products[i]**2/gram[i, i])
            row.update(target_gain=float(products[i]/gram[i, i]),
                       si_sdr_db=db_ratio(target, max(energy-target, 0.)) if energy else -300.)
            if identifiable:
                projection = float(products[active] @ np.linalg.solve(active_gram, products[active]))
                row.update(si_sir_db=db_ratio(target, max(projection-target, 0.)),
                           si_sar_db=db_ratio(projection, max(energy-projection, 0.)))
        rows[name] = row
    return dict(roles=rows, interference_identifiable=identifiable)


def score_stems(reference, estimates, sample_rate, window_seconds=1.):
    """Score all reference-active windows, including a quiet or missing target.

    There is no permutation, fitted offset or per-track gain correction.
    Silent-reference windows report false output separately. Highly correlated
    reference roles cannot identify interference and receive no SIR/SAR score.
    This measures quality; it does not set uncalibrated listening thresholds.
    """
    if not reference or reference.keys() != estimates.keys():
        raise ValueError('Reference and output role names must match exactly')
    reference = {n: np.asarray(x, dtype=np.float64) for n, x in reference.items()}
    estimates = {n: np.asarray(x, dtype=np.float64) for n, x in estimates.items()}
    shape = next(iter(reference.values())).shape
    if len(shape) != 2 or not all(shape):
        raise ValueError('Audio must have non-empty (frames, channels) shape')
    if any(x.shape != shape or not np.isfinite(x).all() for x in [*reference.values(), *estimates.values()]):
        raise ValueError('All stems must have identical finite PCM shapes')
    if not np.isfinite([sample_rate, window_seconds]).all() or sample_rate <= 0 or window_seconds <= 0:
        raise ValueError('Sample rate and window duration must be positive and finite')
    window = max(1, round(sample_rate*window_seconds))
    windows = []
    for start in range(0, shape[0], window):
        stop = min(start+window, shape[0])
        windows.append(dict(start=start/sample_rate, end=stop/sample_rate,
                            **_window_scores({n: x[start:stop] for n, x in reference.items()},
                                             {n: x[start:stop] for n, x in estimates.items()})))
    summary = _window_scores(reference, estimates)
    for name, row in summary['roles'].items():
        active = [w['roles'][name] for w in windows if not w['roles'][name]['reference_silent']]
        silent = [w['roles'][name] for w in windows if w['roles'][name]['reference_silent']]
        row['active_windows'] = len(active)
        row['silent_windows'] = len(silent)
        row['active_window_p10'] = {
            key: float(np.percentile(values, 10)) if values else None
            for key in ('fixed_gain_db', 'si_sdr_db', 'si_sir_db', 'si_sar_db')
            for values in [[a[key] for a in active if a[key] is not None]]}
        row['silent_window_peak_output_dbfs'] = max((s['output_dbfs'] for s in silent), default=None)
    return dict(sample_rate=sample_rate, frames=shape[0], channels=shape[1],
                summary=summary, windows=windows)
