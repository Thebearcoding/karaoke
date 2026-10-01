"""Score fixed lyric-word onset events against independent reference times."""
import math
import unicodedata

import numpy as np

if __package__:
    from .subtitles import highlight_spans
else:
    from subtitles import highlight_spans


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def word_events(cues):
    """Keep every expected word, including words lacking one complete highlight."""
    result = []
    for line, cue in enumerate(cues):
        groups = cue.get('units', [])
        words = list(highlight_spans(cue['text'], 'English'))
        for first, stop in words:
            matches = [g for g in groups if g['first'] <= first and g['stop'] >= stop]
            group = matches[0] if len(matches) == 1 else None
            merged = bool(group and sum(group['first'] <= a and group['stop'] >= b
                                        for a, b in words) > 1)
            result.append(dict(id=len(result), line=line+1, text=cue['text'][first:stop],
                               start=group['start'] if group else None,
                               merged=merged, incomplete_highlight=group is None))
    return result


def score_onsets(reference, predictions, fps=None):
    """PCO denominator is all reference units; missing units are always failures.

    fps projects timestamps onto a render grid for diagnosis only. It does not
    measure the displayed video frames or correct an audio/video clock offset.
    """
    if not reference or len(reference) != len(predictions):
        raise ValueError('Predictions must retain every reference unit, including missing ones')
    if fps is not None and fps <= 0:
        raise ValueError('fps must be positive')
    rows = []
    for ref, pred in zip(reference, predictions):
        if ref['id'] != pred['id'] or canonical(ref['text']) != canonical(pred['text']):
            raise ValueError('Reference and prediction identity/text mismatch')
        if not math.isfinite(ref['start']):
            raise ValueError('Reference onset is not finite')
        value = pred.get('start')
        if value is not None and not math.isfinite(value):
            value = None
        if value is not None and fps:
            value = math.ceil(value*fps-1e-9)/fps
        error = value-ref['start'] if value is not None else None
        rows.append(dict(id=ref['id'], text=ref['text'], reference=ref['start'],
                         predicted=value, error_ms=1000*error if error is not None else None))
    errors = np.array([r['error_ms'] for r in rows if r['error_ms'] is not None])
    absolute = np.abs(errors)
    total = len(rows)
    missing = total-len(errors)
    hits = {str(t):int(np.sum(absolute <= t+1e-7)) for t in [50, 100, 250]}
    summary = dict(units=total, missing=missing, coverage=(total-missing)/total,
                   hits=hits, pco={t:n/total for t,n in hits.items()},
                   catastrophic_or_missing=(int(np.sum(absolute > 250+1e-7))+missing)/total,
                   early_over_100ms=int(np.sum(errors < -100-1e-7))/total,
                   late_over_100ms=int(np.sum(errors > 100+1e-7))/total,
                   median_signed_error_ms=float(np.median(errors)) if len(errors) else None,
                   mean_absolute_error_ms=float(np.mean(absolute)) if len(errors) else None,
                   absolute_error_ms={str(q):float(np.percentile(absolute, q)) if len(errors) else None
                                      for q in [50, 95, 99]},
                   projected_fps=fps,
                   merged_units=sum(bool(p.get('merged')) for p in predictions))
    summary['passed'] = bool(summary['pco']['50'] >= .9 and summary['pco']['100'] >= .98
                             and summary['catastrophic_or_missing'] <= .01 and not missing)
    return dict(summary=summary, errors=rows)
