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


def word_events(cues, language='English'):
    """Keep every expected word, including words lacking one complete highlight."""
    if language not in ('English', 'Chinese'):
        raise ValueError('Japanese evaluation needs independently fixed reading-group events')
    result = []
    for line, cue in enumerate(cues):
        groups = cue.get('units', [])
        words = list(highlight_spans(cue['text'], language))
        for first, stop in words:
            matches = [g for g in groups if g['first'] <= first and g['stop'] >= stop]
            group = matches[0] if len(matches) == 1 else None
            merged = bool(group and sum(group['first'] <= a and group['stop'] >= b
                                        for a, b in words) > 1)
            result.append(dict(id=len(result), line=line+1, text=cue['text'][first:stop],
                               start=group['start'] if group else None,
                               merged=merged, incomplete_highlight=group is None))
    return result


def onset_risks(predictions, verifier, tolerance=.1):
    """Flag disagreements, not correctness. No reference times enter this check.

    Even independent models can agree on the same wrong onset. Calibrate this
    detector with score_risks before interpreting an unflagged word as reliable.
    Missing evidence is a review flag, never permission to remove a lyric.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and non-negative')
    if not predictions or len(predictions) != len(verifier) or any(
            p['id'] != v['id'] or canonical(p['text']) != canonical(v['text'])
            for p, v in zip(predictions, verifier)):
        raise ValueError('Verifier must retain every prediction unit and its identity')
    rows = []
    for p, v in zip(predictions, verifier):
        times = [t if t is not None and math.isfinite(t) else None
                 for t in (p.get('start'), v.get('start'))]
        valid = all(t is not None for t in times)
        delta = times[1]-times[0] if valid else None
        reason = ('missing_acoustic_evidence' if not valid else
                  'incomplete_or_merged_highlight' if any(
                      x.get('merged') or x.get('incomplete_highlight') for x in (p, v)) else
                  'onset_disagreement' if abs(delta) > tolerance+1e-9 else None)
        rows.append(dict(id=p['id'], text=p['text'], predicted=times[0],
                         verifier=times[1], difference_ms=None if delta is None else 1000*delta,
                         review=reason is not None, reason=reason))
    return rows


def score_risks(onset_score, risks, tolerance_ms=100):
    """Measure error-detector precision/recall against independent annotations."""
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError('Tolerance must be finite and non-negative')
    errors = onset_score['errors']
    if len(errors) != len(risks) or any(e['id'] != r['id'] or canonical(e['text']) != canonical(r['text'])
                                      for e, r in zip(errors, risks)):
        raise ValueError('Risk flags must retain every annotated unit')
    counts = dict(true_positive=0, false_positive=0, true_negative=0, false_negative=0)
    for error, risk in zip(errors, risks):
        bad = error['error_ms'] is None or abs(error['error_ms']) > tolerance_ms+1e-7
        counts[('true_' if bad == risk['review'] else 'false_') +
               ('positive' if risk['review'] else 'negative')] += 1
    tp, fp, tn, fn = (counts[k] for k in ('true_positive', 'false_positive', 'true_negative', 'false_negative'))
    return dict(**counts, tolerance_ms=tolerance_ms, units=len(errors),
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                unflagged_error_rate=fn/(tn+fn) if tn+fn else None)


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
