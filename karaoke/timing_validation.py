"""One acoustic timing contract for normal alignment and playback fallbacks.

A disagreement is a screening signal. Only corroborated acoustic predictions
invalidate an old onset; unavailable evidence never turns a quiet voice into
silence. Invalidations outlive individual fallback candidates.
"""
from copy import deepcopy
from itertools import groupby

import numpy as np

if __package__:
    from .anchors_from_asr import phrase_conflict
    from .subtitles import acoustic_end, collapsed_phrase, normalized, highlight_coverage_errors, latin_word_text
else:
    from anchors_from_asr import phrase_conflict
    from subtitles import acoustic_end, collapsed_phrase, normalized, highlight_coverage_errors, latin_word_text


def timing_violations(cues, constraints):
    errors = []
    for proof in constraints:
        index = proof['line']-1
        if index >= len(cues) or cues[index]['text'] != proof['text']:
            errors.append(dict(proof=proof, reason='lyric identity changed'))
            continue
        units = [u for u in cues[index].get('units', [])
                 if u['first'] <= proof['first'] < u['stop']]
        if (len(units) != 1 or units[0]['first'] != proof['first']
                or not proof['start_min'] <= units[0]['start'] <= proof['start_max']):
            errors.append(dict(proof=proof, reason='contradicted onset reused'))
    return errors


def localized_constraints(report, blocks):
    """Retire a wrong-phrase proof only after relocation and new acoustic proof.

    A phonetic aligner assumes its supplied words occur in the crop. A later
    independently recognized phrase can disprove that premise. Keep both
    proofs and the localization evidence instead of making a bad crop eternal.
    """
    active, superseded = [], []
    # The selected candidate can itself belong to the wrong repetition. Use
    # independent phrase identity to choose the applicable proof, then let
    # final validation reject/reselect the candidate under that proof.
    # Established constraints are same-source/model evidence too; a later
    # review need not measure their already settled words again.
    proposals = [*report['constraints'],
                 *(p for row in report['lines'] for p in row.get('proposed_constraints', []))]
    observations = [o for block in blocks for o in block.get('asr_localizations', [])]
    for proof in report['constraints']:
        replacement = None
        for observed in observations:
            anchors = observed.get('anchors', [])
            if not anchors:
                continue  # Retry summaries contain no localization evidence.
            index = proof['line']-1-observed['context'][0]
            if not 0 <= index < len(anchors):
                continue
            anchor = anchors[index]
            if anchor['status']!='candidate' or anchor['text']!=proof['text']:
                continue
            # Alignment acceptance and independent text localization are
            # different evidence. A complete heard clause can contradict an
            # interior word even when the enclosing line did not move.
            if proof['line'] not in observed.get('accepted_lines', []) and (
                    anchor.get('match_basis') not in ('complete_phrase', 'complete_clause')
                    or anchor.get('ambiguous_occurrences') or not any(
                    e['first'] <= proof['first'] < e['stop'] for e in anchor.get('observed_edges', []))):
                continue
            anchor = dict(anchor, match_basis=anchor.get('match_basis', 'complete_phrase'))
            def contradicts(p):
                unit = dict(first=p['first'], stop=p['stop'], text=p['text'][p['first']:p['stop']],
                            start=p['onset'], end=p['onset'])
                return phrase_conflict(dict(text=p['text'], start=p['onset'], end=p['onset'], units=[unit]),
                                       anchor, fragments=False)
            if not contradicts(proof):
                continue
            revised = next((p for p in proposals if all(p[k]==proof[k] for k in ('line','text','first','stop'))
                and not contradicts(p)), None)
            if revised is not None:
                replacement = dict(proof=proof, replacement=revised, anchor=anchor,
                    source_range=observed['source_range'], reason='independently_relocalized_phrase')
                break
        if replacement is None:
            if proof not in active:
                active.append(proof)
        else:
            superseded.append(replacement)
            if replacement['replacement'] not in active:
                active.append(replacement['replacement'])
    return active, superseded


def retain_consistent_input(original, revised, constraints, duration, language):
    """Reject a contradicted revision when its already reviewed input still fits."""
    failed = {v['proof']['line']-1 for v in timing_violations(revised, constraints)}
    retained = {i for i in failed if not timing_violations(original,
                [p for p in constraints if p['line']==i+1])}
    if not retained:
        return revised, []
    candidate = [original[i] if i in retained else cue for i,cue in enumerate(revised)]
    try:
        # Retaining input must not break neighbouring revisions or the text.
        # Other unresolved constraints still go through the final hard gate.
        validate_timeline(candidate, [c['text'] for c in original], duration, language=language)
    except ValueError:
        return revised, []
    return candidate, [i+1 for i in sorted(retained)]


def reuse_expanded_timeline(previous, cues, report, duration, language):
    """Keep reviewed phrases only where every ordered expansion maps them alike."""
    old, new = [c['text'] for c in previous], [c['text'] for c in cues]
    if len(old) >= len(new) or report.get('unresolved') or 'pending_review_lines' not in report:
        return cues, [], []
    try:
        validate_timeline(previous, old, duration, report['constraints'], language)
    except ValueError:
        return cues, [], []
    def locate(lines, positions):
        return [next((j for j in positions if new[j] == text), None) for text in lines]
    left = locate(old, iter(range(len(new))))
    right = locate(reversed(old), iter(range(len(new)-1, -1, -1)))[::-1]
    if None in left or None in right:
        return cues, [], []
    pending = set(report['pending_review_lines'])
    matches = {a: i for i,(a,b) in enumerate(zip(left,right)) if a == b and i+1 not in pending}
    result, retained = list(cues), []
    # Adjacent retained phrases move together; inserted neighbours remain fixed.
    for _, items in groupby(enumerate(matches), lambda item:item[1]-item[0]):
        indexes = [j for _,j in items]
        first, last = indexes[0], indexes[-1]
        block = [previous[matches[j]] for j in indexes]
        if ((first and result[first-1]['units'][-1]['start'] >= block[0]['units'][0]['start']) or
                (last+1 < len(cues) and block[-1]['units'][-1]['start'] >= cues[last+1]['units'][0]['start'])):
            continue
        result[first:last+1] = block
        retained.extend((matches[j]+1, j+1) for j in indexes)
    mapping = dict(retained)
    constraints = [dict(p, line=mapping[p['line']]) for p in report['constraints'] if p['line'] in mapping]
    validate_timeline(result, new, duration, constraints, language)
    return result, constraints, retained


def screened_units(cue, wave, protocol, *, floor=None):
    """Cheap hints select acoustic work; no hint invalidates a timestamp."""
    units = cue['units']
    gaps = np.diff([u['start'] for u in units])
    typical = float(np.median(gaps)) if len(gaps) else 0.
    if floor is None:
        floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    selected = []
    for i, unit in enumerate(units):
        piece = wave[round(unit['start']*44100):round((unit['start']+.035)*44100)]
        coarse = unit.get('timing_source') in ('prior_word_fallback', 'linear_fallback')
        gap = gaps[i] if i < len(gaps) else 0.
        if (coarse or gap >= 2 or (gap >= .8 and typical and gap >= 2.5*typical)
                or not len(piece) or float(np.sqrt(np.mean(piece**2))) <= floor):
            selected.append(i)
    return selected


def closure_pronunciation(unit, predictions):
    """A closure is a stop gesture, not a voiced sound to find with RMS."""
    matches = [[u for u in view if (u['first'], u['stop'], u['text']) ==
                (unit['first'], unit['stop'], unit['text'])] for view in predictions]
    return len(matches) == 3 and all(len(items) == 1 and items[0].get('phones') == ['cl']
                                     for items in matches)


def matching_onsets(cue, groups):
    """Match lexical starts even when old tokens include leading punctuation."""
    return {g['first']: u for g in groups for u in cue['units']
            if u['first'] <= g['first'] < u['stop']
            and not normalized(cue['text'][u['first']:g['first']])}


def displaced_units(cue, views, tolerance):
    """Coarse agreement selects another observation, never assigns a time."""
    if len(views) != 2 or not views[0] or len(views[0]) != len(views[1]):
        return []
    old = matching_onsets(cue, views[0])
    return [a['first'] for a, b in zip(*views) if a['first'] in old
            and (a['first'], a['stop'], a['text']) == (b['first'], b['stop'], b['text'])
            # A coarse disagreement is uncertainty, not permission to keep a
            # far-away old onset. Reobserve when both views exclude it by
            # more than their own disagreement; adoption remains unchanged.
            and (old[a['first']]['start'] < min(a['start'], b['start'])-max(tolerance,abs(a['start']-b['start']))
                 or old[a['first']]['start'] > max(a['start'], b['start'])+max(tolerance,abs(a['start']-b['start'])))]


def pending_review_lines(cues, report, protocol):
    settled = {(p['line'], p['first']) for p in report['constraints']
               if not timing_violations(cues, [p])}
    pending = {row['line']-1 for row in report['lines'] if row['status'] != 'unverified'
               for first in displaced_units(cues[row['line']-1], row.get('witnesses', []),
                                             protocol['maximum_change_seconds'])
               if (row['line'], first) not in settled}
    return pending | {v['proof']['line']-1 for v in timing_violations(cues, report['constraints'])}


def reconcile_views(cues, observations, protocol, language, constraints=(), *, playback=False, previous=(), contexts=None):
    """Spend additional acoustic work only on unresolved local disagreements.

    Each observation uses the same acceptance contract and retains all previous
    onset constraints. The iterator may load a different audio view lazily.
    """
    from cjk_recovery import audio_floor
    result, report, pending = deepcopy(cues) if contexts else cues, dict(lines=[], constraints=list(constraints)), None
    requested = set(contexts or {})
    for index, (begin, end) in (contexts or {}).items():
        result[index].update(aligned_start=begin, aligned_end=end)
    history, floors = [], {}
    for name, wave, witness, predict in observations:
        result, reviewed = reconcile_timing(result, wave, witness, predict, protocol, language,
            report['constraints'], playback=playback, _repeat_context=not requested,
            _review_lines=(set(pending or ()) | requested) if pending is not None or requested else None)
        report['lines'].extend(dict(row, observation=name) for row in reviewed['lines'])
        report.update(constraints=reviewed['constraints'], unresolved=reviewed['unresolved'], accuracy=None)
        unresolved_lines = pending_review_lines(result, report, protocol)
        if unresolved_lines:
            wider = deepcopy(result)
            for i in unresolved_lines:
                bounds = [result[i-1]['units'][-1]['start']+.015 if i else 0.,
                          result[i+1]['units'][0]['start'] if i+1<len(result) else len(wave)/44100]
                # An uncertain old cue can cut off half of a repeated phrase.
                # Reobserve the available neighbouring context before treating
                # the old boundary as a limit. Only measured times are adopted.
                begin, end = (bounds if bounds[1]-bounds[0] <= 30 else
                              [max(bounds[0], result[i]['start']-1.5),
                               min(bounds[1], result[i]['end']+1.5)])
                wider[i].update(aligned_start=begin, aligned_end=end, audio_bounds=bounds)
            wider, retried = reconcile_timing(wider, wave, witness, predict, protocol, language,
                report['constraints'], playback=playback, _retry_context=False,
                _repeat_context=False, _review_lines=unresolved_lines)
            for i in unresolved_lines:
                if wider[i]['units'] != result[i]['units']:
                    result[i] = wider[i]
            report['lines'].extend(dict(row, observation=name, expanded_context=True) for row in retried['lines'])
            report.update(constraints=retried['constraints'],
                          unresolved=timing_violations(result, retried['constraints']))
        history.extend((wave, row) for row in [*previous, *report['lines']]
                       if row.get('observation') == name and row.get('predictions'))
        # An accepted neighbour can make an earlier measured onset fit.
        # Replay the actual measurements, not another model request or the
        # constraint's tolerance boundary. Only accept a strict reduction in
        # violations, preserving every already established onset constraint.
        remaining = len(timing_violations(result, report['constraints']))
        while remaining:
            for recorded_wave, row in history:
                line = row['line']
                if not any(v['proof']['line']==line for v in timing_violations(result, report['constraints'])):
                    continue
                candidate, replay = reconcile_timing(result, recorded_wave,
                    lambda *args, **kwargs: row['witnesses'], lambda *args: row['predictions'],
                    protocol, language, report['constraints'], playback=playback,
                    _retry_context=False, _repeat_context=False, _review_lines={line-1})
                if len(replay['unresolved']) < len(timing_violations(result, report['constraints'])):
                    result = candidate
                    report['lines'].extend(dict(r, observation=row['observation'], reused_measurement=True)
                                           for r in replay['lines'])
                    report.update(constraints=replay['constraints'], unresolved=replay['unresolved'])
            unresolved = len(timing_violations(result, report['constraints']))
            if unresolved >= remaining:
                break
            remaining = unresolved
        if playback:
            for recorded_wave, row in history:
                index = row['line']-1
                known = [u for r in [*previous, *report['lines']] if r['line']==index+1
                         for u in r.get('corroborated_units', [])]
                if id(recorded_wave) not in floors:
                    floors[id(recorded_wave)] = audio_floor(recorded_wave, protocol)
                candidate = compose_timing_bounds(result, index, row, known, report['constraints'],
                                                   recorded_wave, protocol, language, floor=floors[id(recorded_wave)])
                if candidate is not None:
                    result[index] = candidate
                    report['lines'].append(dict(line=index+1, observation=row['observation'], status='reviewed',
                        reused_measurement=True, repairs=[dict(accepted=True, mode='composed_acoustic_bounds',
                        reason='local sweep between independently measured onsets')]))
            report['unresolved'] = timing_violations(result, report['constraints'])
        pending = pending_review_lines(result, report, protocol)
        if not pending and not requested:
            break
    report['pending_review_lines'] = [i+1 for i in sorted(pending or [])]
    return result, report


def compose_timing_bounds(cues, index, row, measured, constraints, wave, protocol, language, *, floor=None):
    """Compose local playback groups from independently measured onsets.

    A previous observation can locate one word and a later observation its
    neighbour. Requiring both to succeed in the same model call discards that
    evidence. Interior word times stay unspecified; the two acoustic bounds
    and every established onset remain explicit. Proposed onsets may be tested
    here but become durable only after the complete candidate is accepted.
    """
    cue = cues[index]
    proofs = [p for p in constraints if p['line']==index+1]
    failed = timing_violations(cues, proofs)
    if len(row.get('witnesses', [])) != 2:
        return None
    if any(normalized(''.join(u['text'] for u in view)) != normalized(cue['text']) or
           any(u['text'] != cue['text'][u['first']:u['stop']] for u in view) for view in row['witnesses']):
        return None
    views = [{u['first']:u for u in view} for view in row['witnesses']]
    if not failed:
        # A displaced coarse interior can use the same playback sweep without
        # claiming a new word onset. Its already confirmed bounding onsets stay
        # exactly where they are; fresh label events need not repeat them.
        units = cue['units']
        bounds = [i for i,u in enumerate(units)
                  if u.get('timing_source') not in ('prior_word_fallback', 'linear_fallback') and
                  any((p['first'],p['stop'],p['onset']) == (u['first'],u['stop'],u['start']) for p in proofs)]
        displaced = set(displaced_units(cue,row['witnesses'],protocol['maximum_change_seconds']))
        if not displaced or len(bounds) < 2:
            return None
        if floor is None:
            floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
        for lo,hi in zip(bounds,bounds[1:]):
            left,right = units[lo],units[hi]
            interior = units[lo+1:hi]
            if (not any(u['first'] in displaced for u in interior) or
                    any(u.get('timing_source') != 'prior_word_fallback' for u in interior)):
                continue
            observed = [[u for u in view.values() if left['stop'] <= u['first'] < u['stop'] <= right['first']]
                        for view in views]
            if any(normalized(''.join(u['text'] for u in view)) != normalized(cue['text'][left['stop']:right['first']]) or
                   any(not left['start']-.1 <= u['start'] <= right['start']+.1 for u in view) for view in observed):
                continue
            if any(not len(piece:=wave[round(t*44100):round((t+.035)*44100)]) or
                   not np.isfinite(piece).all() or float(np.sqrt(np.mean(piece**2)))<=floor
                   for t in (left['start'],right['start'])):
                continue
            group = dict(first=left['first'],stop=right['first'],text=cue['text'][left['first']:right['first']],
                         start=left['start'],end=right['start'],acoustic_end=None,
                         timing_source='linear_fallback',bounds_source='cross_model_onsets')
            candidate = dict(cue,units=[*units[:lo],group,*units[hi:]])
            try:
                validate_timeline([candidate],[cue['text']],len(wave)/44100,
                                  [dict(p,line=1) for p in proofs],language)
            except ValueError:
                continue
            return candidate
        return None
    points = {}
    def supports(first, time):
        from cjk_recovery import label_events_support
        if any(first not in view for view in views):
            return False
        secondary = [view[first] for view in views]
        if all(abs(u['start']-time)<=.1 for u in secondary):
            return True
        return any(u['first']==first and u['start']==time and
            u.get('independent_evidence')=='ctc_label_localization' and
            u.get('timing_evidence') in ('ctc_event','phonetic_span','attention_span') and
            label_events_support(time, u.get('first_event_end', u.get('emission_end')
                if u['timing_evidence']=='ctc_event' else u.get('acoustic_end',u['end'])),
                secondary, protocol['maximum_change_seconds']) for u in measured)
    for unit in measured:
        first, time = unit['first'], unit['start']
        if supports(first, time):
            points.setdefault(first, time)
    for proof in proofs:
        first, time = proof['first'], proof['onset']
        if (not supports(first, time)
                or any(p['first']==first and not p['start_min']<=time<=p['start_max'] for p in proofs)):
            return None
        points[first] = time
    positions = {u['first']:i for i,u in enumerate(cue['units'])}
    first, last = min(v['proof']['first'] for v in failed), max(v['proof']['first'] for v in failed)
    before = cues[index-1]['units'][-1]['start'] if index else -.015
    after = cues[index+1]['units'][0]['start'] if index+1<len(cues) else len(wave)/44100
    if floor is None:
        floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    for left in sorted((p for p in points if p<=first and p in positions), reverse=True):
        lo = positions[left]
        if points[left] <= (cue['units'][lo-1]['start'] if lo else before)+.015:
            continue
        right_bounds = sorted(p for p in points if p>last and p in positions)
        # A corrected terminal onset can itself close the preceding sweep;
        # retain its old display end instead of inventing a later word.
        if last>left and last in points and last in positions:
            right_bounds.append(last)
        for right in right_bounds:
            hi = positions[right]
            guard = cue['units'][hi]
            limit = cue['units'][hi+1]['start'] if hi+1<len(cue['units']) else after
            if not points[right] < min(guard['end'], limit):
                continue
            marks = sorted((p,t) for p,t in points.items() if left<=p<=right)
            if any(b[1]-a[1]<.015 for a,b in zip(marks,marks[1:])):
                continue
            if any(not len(piece:=wave[round(t*44100):round((t+.035)*44100)]) or
                   not np.isfinite(piece).all() or float(np.sqrt(np.mean(piece**2)))<=floor for _,t in marks):
                continue
            if any(not start-.1 <= u['start'] <= end+.1
                   for (a,start),(b,end) in zip(marks,marks[1:])
                   for view in views for p,u in view.items() if a<p<b):
                continue
            groups = [dict(first=a,stop=b,text=cue['text'][a:b],start=start,end=end,acoustic_end=None,
                           timing_source='linear_fallback',bounds_source='cross_model_onsets')
                      for (a,start),(b,end) in zip(marks,marks[1:])]
            candidate = dict(cue,units=[*cue['units'][:lo],*groups,
                dict(guard,start=points[right],acoustic_end=None,timing_source='linear_fallback',
                     bounds_source='cross_model_onset_prior_acoustic_end'),*cue['units'][hi+1:]])
            candidate.update(start=candidate['units'][0]['start'],aligned_start=candidate['units'][0]['start'])
            try:
                validate_timeline([candidate],[cue['text']],len(wave)/44100,
                                  [dict(p,line=1) for p in proofs],language)
            except ValueError:
                continue
            return candidate
    return None


def reconcile_timing(cues, wave, witness, predict, protocol, language, constraints=(), *,
                     playback=False, _retry_context=True, _repeat_context=True, _review_lines=None):
    """Corroborate displaced onsets, then replace only a compatible local span.

    witness returns original/vocal word views. predict returns three independent
    context predictions without old word priors. Both are cacheable adapters.
    """
    from cjk_recovery import corroborated_group, same_acoustic_event
    result, reports, ledger = deepcopy(cues), [], deepcopy(list(constraints))
    context_hints = {}
    repeated = {}
    for _, group in groupby(range(len(result)), key=lambda i: normalized(result[i]['text']).casefold()):
        indices = list(group)
        if len(indices) < 2:
            continue
        text = '\n'.join(result[i]['text'] for i in indices)
        offset = 0
        for i in indices:
            repeated[i] = (indices[0], indices[-1]+1, text, offset)
            offset += len(result[i]['text'])+1
    for cue in result:
        cue.setdefault('aligned_start', cue['start'])
        cue.setdefault('aligned_end', cue['end'])
        for unit in cue['units']:
            unit['acoustic_end'] = acoustic_end(unit)
    floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    for index, cue in enumerate(result):
        if _review_lines is not None and index not in _review_lines:
            continue
        selected = screened_units(cue, wave, protocol, floor=floor)
        if not selected and _review_lines is None and not timing_violations(result, [p for p in ledger if p['line']==index+1]):
            continue
        report = dict(line=index+1, screened_units=selected, status='unverified', repairs=[])
        reports.append(report)
        try:
            # A stale first word can stretch the coarse search across an
            # interlude. Locate the phrase before applying the short phonetic
            # window budget; predict() still requires three contexts <= 32 s.
            if cue['aligned_end']-cue['aligned_start'] > 60:
                raise ValueError('Acoustic localization context exceeds 60 seconds')
            letters = [c for c in latin_word_text(cue['text']) if c.isalpha()]
            review_language = 'English' if language in ('Chinese', 'Japanese') and letters and all(c.isascii() for c in letters) else language
            observation, offset = cue, None
            # Joint localization initializes occurrence identity. Later passes
            # already selected a new crop or recorded observations: rebuilding
            # the joint context would repeat the very observation that failed.
            if _repeat_context and index in repeated:
                first, stop, text, start_char = repeated[index]
                begin = result[first-1]['units'][-1]['start']+.015 if first else 0.
                end = result[stop]['units'][0]['start'] if stop<len(result) else len(wave)/44100
                joint_budget = 32-2*max(protocol.get('context_padding_seconds', [0.]))
                if 0 < end-begin <= joint_budget:
                    # Identical neighbouring phrases need occurrence identity:
                    # align their complete text once in the same audio context.
                    observation = dict(text=text, aligned_start=begin, aligned_end=end,
                                       audio_bounds=[begin,end])
                    offset = start_char
                    report['occurrence_context'] = dict(lines=[first+1,stop], source_range=[begin,end])
            views = witness(observation, wave, language=review_language)
            if offset is not None:
                joint = predict(observation, views, [begin,end], review_language)
                report['occurrence_context'].update(witnesses=views, predictions=joint)
                if any(normalized(''.join(u['text'] for u in view)) != normalized(observation['text']) or
                       any(a['start']>=b['start'] for a,b in zip(view,view[1:])) for view in [*views,*joint]):
                    raise ValueError('Incomplete repeated phrase context')
                locations = [[dict(u, first=u['first']-offset, stop=u['stop']-offset)
                              for u in view if offset<=u['first']<u['stop']<=offset+len(cue['text'])]
                             for view in joint]
                if len(locations)!=3 or not all(locations):
                    raise ValueError('Incomplete repeated phrase predictions')
                # The joint pass proposes occurrence ranges, not final word
                # times. Reobserve each range, then refine in short contexts.
                observation = dict(cue, aligned_start=min(v[0]['start'] for v in locations),
                    aligned_end=max(v[-1]['end'] for v in locations), audio_bounds=[begin,end])
                views = witness(observation, wave, language=review_language)
            report['witnesses'] = views
            signature = lambda v: [(u['first'], u['stop'], u['text']) for u in v]
            if (len(views)!=2 or not views[0] or signature(views[0]) != signature(views[1])
                    or normalized(''.join(u['text'] for u in views[0])) != normalized(cue['text'])
                    or any(u['text'] != cue['text'][u['first']:u['stop']] for u in views[0])):
                raise ValueError('Incomplete acoustic text evidence')
            bounds = [result[index-1]['units'][-1]['start']+.015 if index else 0.,
                      result[index+1]['units'][0]['start'] if index+1<len(result) else len(wave)/44100]
            # Observation context may overlap an uncertain neighbour. Only
            # the proposed local replacement must fit the lyric boundaries.
            if any(not np.isfinite([u['start'],u['end']]).all()
                   or not 0 <= u['start'] < u['end'] <= len(wave)/44100 for v in views for u in v) or any(
                       a['start'] >= b['start'] for v in views for a,b in zip(v,v[1:])):
                raise ValueError('Invalid acoustic localization intervals')
            old = matching_onsets(cue, views[0])
            # Coarse localization need not meet the final onset tolerance to
            # justify another observation. Two nearby estimates can both
            # exclude an old time while differing by more than 100 ms.
            if _retry_context and displaced_units(cue, views, protocol['maximum_change_seconds']):
                begin, end = min(v[0]['start'] for v in views), max(v[-1]['end'] for v in views)
                if (0 < end-begin <= 30 and
                        max(abs(begin-cue['aligned_start']),abs(end-cue['aligned_end'])) > protocol['maximum_change_seconds']):
                    context_hints[index] = [begin,end]
                    report['context_hint'] = [begin,end]
            challenged = {v['proof']['first'] for v in timing_violations(result, ledger)
                          if v['proof']['line']==index+1}
            # A coarse fallback has not earned the fine result's displacement
            # threshold. Review it even when the label is less than 0.4 s away;
            # adoption still requires the same independent acoustic evidence.
            suspects = [a for a,b in zip(*views) if a['first'] in old
                        and same_acoustic_event(a, b, protocol['maximum_change_seconds'])
                        and (a['first'] in challenged or
                             old[a['first']].get('timing_source') in ('prior_word_fallback', 'linear_fallback') or
                             abs(old[a['first']]['start']-a['start']) > protocol['maximum_change_seconds'])]
            if not suspects:
                report['status'] = 'no_corroborated_displacement'
                continue
            predictions = predict(cue, views, bounds, review_language)
            report['predictions'] = predictions
            confirmed, proposals = {}, {}
            for group in views[0]:
                choose = lambda vs: [[u for u in v if (u['first'],u['stop'])==(group['first'],group['stop'])] for v in vs]
                replacement, reason = corroborated_group(group, choose(predictions), choose(views), wave,
                    bounds, protocol, spans=[(group['first'],group['stop'])], floor=floor)
                if replacement is None:
                    if group in suspects:
                        report.setdefault('unconfirmed', []).append(dict(first=group['first'],reason=reason))
                    continue
                replacement = dict(replacement[0], timing_source={
                    'ctc_event': 'cross_model_ctc', 'attention_span': 'cross_model_attention'}.get(
                    replacement[0].get('timing_evidence'), 'cross_model_phonemes'),
                    acoustic_end=acoustic_end(replacement[0]))
                confirmed[group['first']] = replacement
                if group not in suspects:
                    continue
                start = replacement['start']
                # The broad interval rejects grossly displaced older candidates;
                # it is a localization constraint, not a word-accuracy score.
                proof = dict(line=index+1,text=cue['text'],first=group['first'],stop=group['stop'],
                    start_min=max(0.,start-protocol['maximum_change_seconds']),
                    start_max=start+protocol['maximum_change_seconds'],onset=start)
                proposals[group['first']] = proof
                report.setdefault('proposed_constraints', []).append(proof)
            report['corroborated_units'] = list(confirmed.values())
            for first, anchor in sorted(confirmed.items()):
                # A merged old token can contain several measured words. Use
                # all corroborated words inside its replacement, but only a
                # displaced old onset initiates an edit or a new constraint.
                if not any(g['first']==first for g in suspects):
                    continue
                units = cue['units']
                hits = [i for i,u in enumerate(units) if u['first']<=first<u['stop']]
                if not hits:continue
                lo,hi=hits[0],hits[0]+1
                if units[lo]['first']==first and units[lo]['start']==anchor['start']:
                    continue
                while True:
                    members = [dict(a,timing_source='prior_word_fallback',
                                    localization_source='mms_original_and_vocals',acoustic_end=None)
                               for a in views[0] if a['first']<units[hi-1]['stop'] and a['stop']>units[lo]['first']]
                    # Close the edit over both tokenizations before checking
                    # time neighbours. A measured word can cover several old
                    # characters, and an old group can straddle measured words.
                    if members and lo and members[0]['first']<units[lo]['first']:
                        lo-=1;continue
                    if members and hi<len(units) and members[-1]['stop']>units[hi-1]['stop']:
                        hi+=1;continue
                    if not members or normalized(''.join(u['text'] for u in members)) != normalized(cue['text'][units[lo]['first']:units[hi-1]['stop']]):
                        members=[];break
                    members = [dict(confirmed.get(u['first'],u)) for u in members]
                    if lo and members[0]['start'] <= units[lo-1]['start']+.015:
                        lo-=1;continue
                    if hi<len(units) and members[-1]['start'] >= units[hi]['start']-.015:
                        hi+=1;continue
                    break
                reason=None
                localized=False
                if not members:reason='incomplete local replacement'
                else:
                    original={u['first']:u for u in views[0]}
                    other={u['first']:u for u in views[1]}
                    localized=all(abs(original[u['first']]['start']-other[u['first']]['start'])<=protocol['maximum_change_seconds'] for u in members)
                    if any(not (u.get('independent_evidence') == 'ctc_label_localization' and
                                same_acoustic_event(original[u['first']], other[u['first']], protocol['maximum_change_seconds']))
                           and abs(original[u['first']]['start']-other[u['first']]['start'])>.1 for u in members):
                        reason='original and vocal localization disagree'
                    limit=units[hi]['start'] if hi<len(units) else bounds[1]
                    for member_index,(unit,following) in enumerate(zip(members,[*members[1:],dict(start=limit)])):
                        unit['end']=min(unit['end'],following['start'])
                        # Recheck the clipped interval using the same support
                        # contract as corroborated_group. A phonetic word can
                        # begin with a quiet closure before its voiced part.
                        support_end=min(unit['end'],unit['start']+.035)
                        if unit['first'] in confirmed and unit.get('timing_evidence')!='ctc_event':
                            support_end=unit['end']
                        piece=wave[round(unit['start']*44100):round(support_end*44100)]
                        if unit['end']<=unit['start'] or following['start']-unit['start']<.015:
                            reason='invalid local word order'
                            localized=False
                        valid_audio = len(piece) and np.isfinite(piece).all()
                        quiet_closure = (0 < member_index < len(members)-1 and valid_audio and
                                         closure_pronunciation(unit, predictions) and
                                         not closure_pronunciation(members[member_index-1], predictions) and
                                         not closure_pronunciation(following, predictions))
                        if quiet_closure:
                            unit['support'] = 'phonetic_closure_between_audible_units'
                            report.setdefault('closure_spans', []).append(dict(
                                first=unit['first'], stop=unit['stop'], start=unit['start'], end=unit['end']))
                        if not valid_audio or (float(np.sqrt(np.mean(piece**2)))<=floor and not quiet_closure):
                            reason='replacement lacks audible vocal support'
                            report.setdefault('unsupported_spans', []).append(dict(
                                first=unit['first'], stop=unit['stop'], start=unit['start'], end=unit['end']))
                    if any(not bounds[0] <= u['start'] < bounds[1] for u in members):
                        reason='local replacement crosses neighbouring lyrics'
                        localized=False
                attempt=dict(first=units[lo]['first'],stop=units[hi-1]['stop'],accepted=reason is None,reason=reason)
                report['repairs'].append(attempt)
                # An isolated matching syllable is not a durable constraint
                # when the surrounding replacement maps to different words
                # in the two recordings or crosses the lyric occurrence.
                # Keep that proposal for a bounded group retry; only a
                # coherent local span establishes it.
                if localized and proposals[first] not in ledger:
                    ledger.append(proposals[first])
                if reason is not None:continue
                candidate=dict(cue,units=units[:lo]+members+units[hi:])
                if highlight_coverage_errors([candidate],language) and not highlight_coverage_errors([cue],language):
                    attempt.update(accepted=False,reason='replacement changed lyric coverage');continue
                cue['units']=candidate['units']
                cue.update(start=cue['units'][0]['start'],aligned_start=cue['units'][0]['start'],
                           end=cue['units'][-1]['end'],aligned_end=cue['units'][-1]['end'])
            report['status']='reviewed'
        except (ValueError,RuntimeError,OSError) as error:
            if 'out of memory' in str(error).lower():
                raise
            report['reason']=str(error)
    if _retry_context:
        retry = deepcopy(result)
        latest = {r['line']-1: r for r in reports}
        windows = {i: window for i, window in context_hints.items()
                   if displaced_units(result[i], latest[i]['witnesses'], protocol['maximum_change_seconds'])}
        for violation in timing_violations(result, ledger):
            proof = violation['proof']
            index = proof['line']-1
            if not 0 <= index < len(result) or violation['reason'] == 'lyric identity changed':
                continue
            cue = retry[index]
            # A disproved line boundary should not remain the only crop used
            # to resolve its neighbours. Move the observation window, never
            # the word times, and retain every prior acoustic constraint.
            if (proof['first'] == cue['units'][0]['first'] and
                    abs(proof['onset']-cue['aligned_start']) > protocol['maximum_change_seconds'] and
                    0 < cue['aligned_end']-proof['onset'] <= 30):
                windows[index] = [proof['onset'], cue['aligned_end']]
                cue['aligned_start'] = proof['onset']
            elif proof['first'] != cue['units'][0]['first']:
                row = next((r for r in reversed(reports) if r['line']==index+1 and
                            r.get('status')=='reviewed' and r.get('predictions')), None)
                if row is None:
                    continue
                views = row['witnesses']
                begin, end = min(v[0]['start'] for v in views), max(v[-1]['end'] for v in views)
                # An internal merged word can also inherit an obsolete long
                # crop. Reobserve the whole measured phrase once; its boundary
                # envelope is only context, never an assigned word time.
                if (0 < end-begin <= 30 and
                        max(abs(begin-cue['aligned_start']), abs(end-cue['aligned_end'])) > protocol['maximum_change_seconds']):
                    windows[index] = [begin, end]
                    cue.update(aligned_start=begin, aligned_end=end)
        if windows:
            for index, (begin,end) in windows.items():
                retry[index].update(aligned_start=begin,aligned_end=end)
            retry, reviewed = reconcile_timing(retry, wave, witness, predict, protocol, language, ledger,
                                               _retry_context=False, _repeat_context=False, _review_lines=windows)
            for index in windows:
                cue = retry[index]
                if cue['units'] != result[index]['units']:
                    cue['aligned_start'] = cue['units'][0]['start']
                    result[index] = cue
            reports.extend(dict(r, context_retry=windows[r['line']-1]) for r in reviewed['lines'])
            ledger = reviewed['constraints']
    if playback:
        # A short phoneme can have a corroborated onset without enough room
        # for an independent highlight. Verify a minimal playback group
        # instead of restoring a grossly displaced earlier word time.
        for row in reports:
            if not row.get('predictions'):
                continue
            index = row['line']-1
            bounds = [result[index-1]['units'][-1]['start']+.015 if index else 0.,
                      result[index+1]['units'][0]['start'] if index+1<len(result) else len(wave)/44100]
            candidates = {p['first'] for p in row.get('proposed_constraints', [])}
            candidates.update(p['first'] for p in row.get('unconfirmed', []) if
                p['reason'] in ('insufficient time between onset events', 'collapsed phonetic span'))
            for first in sorted(candidates):
                old = next((u for u in result[index]['units'] if u['first']==first), None)
                if old is None:
                    continue
                if any(p['first']==first and p['start_min']<=old['start']<=p['start_max']
                       for p in row.get('proposed_constraints', [])):
                    continue
                proof = dict(line=index+1,text=result[index]['text'],first=first)
                candidate, repair = grouped_timing_fallback(result[index], row, proof, ledger,
                    wave, bounds, protocol, language, floor=floor)
                promoted = []
                if candidate is None:
                    proposal = next((p for p in row.get('proposed_constraints', []) if p['first']==first), None)
                    if proposal is not None:
                        # A coarse interior can collide with a measured word
                        # while its surrounding onsets are independently sound.
                        candidate = compose_timing_bounds(result, index, row, row.get('corroborated_units', []),
                            [*ledger, proposal], wave, protocol, language, floor=floor)
                    if candidate is not None:
                        promoted = [p for p in row['proposed_constraints'] if any(
                            (u['first'],u['start'])==(p['first'],p['onset']) for u in candidate['units'])]
                        repair = dict(first=first, accepted=True, mode='composed_acoustic_bounds',
                                      reason='local sweep between independently measured onsets')
                if candidate is None:
                    continue
                anchor = next((u for u in candidate['units'] if u['first']==first), None)
                if anchor is None or abs(anchor['start']-old['start']) <= protocol['maximum_change_seconds']:
                    continue
                proof.update(stop=anchor['stop'],onset=anchor['start'],
                             start_min=max(0.,anchor['start']-protocol['maximum_change_seconds']),
                             start_max=anchor['start']+protocol['maximum_change_seconds'])
                ledger.extend(p for p in [proof, *promoted] if p not in ledger)
                result[index] = candidate
                row['repairs'].append(repair)
        # A repaired neighbour can make an earlier rejected local edit fit.
        # Reuse the same evidence until no more constraints are resolved;
        # never rerun models or discard a constraint to make progress.
        remaining = len(timing_violations(result, ledger))
        while remaining:
            for row in reports:
                index = row['line']-1
                proofs = [v['proof'] for v in timing_violations(result, ledger) if v['proof']['line']==index+1]
                if not proofs or 'predictions' not in row:
                    continue
                bounds = [result[index-1]['units'][-1]['start']+.015 if index else 0.,
                          result[index+1]['units'][0]['start'] if index+1<len(result) else len(wave)/44100]
                for proof in proofs:
                    if not timing_violations(result, [proof]):
                        continue
                    candidate, repair = grouped_timing_fallback(result[index], row, proof, ledger,
                        wave, bounds, protocol, language, floor=floor)
                    if candidate is not None:
                        result[index] = candidate
                        row['repairs'].append(repair)
            unresolved = len(timing_violations(result, ledger))
            if unresolved >= remaining:
                break
            remaining = unresolved
    return result,dict(lines=reports,constraints=ledger,unresolved=timing_violations(result,ledger),
                       accuracy=None)


def grouped_timing_fallback(cue, report, proof, constraints, wave, bounds, protocol, language, *, floor=None):
    """Sweep an uncertain interior only within corroborated acoustic bounds.

    This is a playback group, never a replacement word-accuracy measurement.
    The first corroborated following source unit closes the smallest group.
    A source unit may already contain several words from an earlier fallback.
    """
    from cjk_recovery import audio_floor, corroborated_group
    if floor is None:
        floor = audio_floor(wave, protocol)
    units, views = cue['units'], report['witnesses']
    positions = {u['first']:i for i,u in enumerate(units)}
    first = proof['first']
    if first not in positions:
        return None, None
    def corroborate(group, *, boundary=False):
        select = lambda vs: [[u for u in v if group['first'] <= u['first'] < u['stop'] <= group['stop']] for v in vs]
        selected, primary = select(views), select(report['predictions'])
        if not selected[0] or normalized(''.join(u['text'] for u in selected[0])) != normalized(group['text']):
            return None
        if boundary:
            # Keep the measured outside boundaries; the interior is explicitly
            # a playback sweep, not fabricated per-character timestamps.
            if any(not v or normalized(''.join(u['text'] for u in v))!=normalized(group['text'])
                   or any(a['stop']>b['first'] or a['start']>=b['start'] for a,b in zip(v,v[1:]))
                   for v in [*primary,*selected]):
                return None
            def joined(v):
                return [dict(v[0],first=group['first'],stop=group['stop'],text=group['text'],
                             end=v[-1]['end'],emission_end=v[-1].get('emission_end'),acoustic_end=None,
                             first_event_end=v[0].get('emission_end')
                                 if v[0].get('timing_evidence')=='ctc_event' else v[0]['end'])]
            primary, selected = [joined(v) for v in primary], [joined(v) for v in selected]
        found, _ = corroborated_group(group, primary, selected,
            wave, bounds, protocol, spans=[(u['first'],u['stop']) for u in selected[0]], floor=floor)
        if found and boundary:
            start = found[0]['start']
            piece = wave[round(start*44100):round((start+.035)*44100)]
            if not len(piece) or not np.isfinite(piece).all() or float(np.sqrt(np.mean(piece**2)))<=floor:
                return None
        return found

    word = next((u for u in views[0] if u['first']==first), None)
    measured = corroborate(word) if word else None
    if word and not measured:
        for last in views[0]:
            if last['first'] <= first:
                continue
            group = dict(first=first,stop=last['stop'],text=cue['text'][first:last['stop']])
            measured = corroborate(group,boundary=True)
            if measured:
                break
    if not measured:
        return None, None
    anchor, lo = measured[0], positions[first]
    if lo and units[lo-1]['start']+.015 >= anchor['start']:
        # Correcting a late onset can cross earlier, uncertain word times.
        # Start the smallest group at a corroborated preceding source unit;
        # keep the displaced onset as its independently measured right edge.
        for previous in reversed(range(lo)):
            prefix = corroborate(units[previous])
            if (prefix and prefix[0]['start'] < anchor['start'] and
                    (not previous or units[previous-1]['start']+.015 < prefix[0]['start'])):
                first, lo, anchor = units[previous]['first'], previous, prefix[0]
                break
        else:
            return None, None
    guard = None
    for hi in range(lo+1, len(units)):
        if units[hi]['first'] < anchor['stop']:
            continue
        guard = corroborate(units[hi])
        if not guard:
            # An existing group need not be split to correct its known onset.
            # Preserve its acoustic display end; do not manufacture interior
            # word times or claim that the retained end was remeasured.
            head = next((u for u in views[0] if u['first']==units[hi]['first'] and
                         u['stop']<units[hi]['stop']), None)
            known = corroborate(head) if head else None
            if known and known[0]['start'] < units[hi]['end']:
                guard = [dict(first=units[hi]['first'],stop=units[hi]['stop'],text=units[hi]['text'],
                    start=known[0]['start'],end=units[hi]['end'],acoustic_end=None,
                    timing_source='linear_fallback',bounds_source='cross_model_onset_prior_acoustic_end')]
        if not guard:
            # The right boundary can itself begin with an inseparable short
            # phoneme. Apply the same minimal-group rule on both sides, and
            # retain the existing display end rather than inventing an offset.
            for last in range(hi+1, len(units)):
                group = dict(first=units[hi]['first'], stop=units[last]['stop'],
                             text=cue['text'][units[hi]['first']:units[last]['stop']])
                joined = corroborate(group, boundary=True)
                if joined and joined[0]['start'] < units[last]['end']:
                    guard = [dict(group, start=joined[0]['start'], end=units[last]['end'],
                        acoustic_end=None, timing_source='linear_fallback',
                        bounds_source='cross_model_onset_prior_acoustic_end')]
                    hi = last
                    break
        if guard:
            break
    if guard is not None:
        stop, end = guard[0]['first'], guard[0]['start']
        limit = units[hi+1]['start'] if hi+1<len(units) else bounds[1]
        if any(u['start']+.015 >= limit or
               u.get('timing_evidence')=='ctc_event' and u['emission_end']>limit for u in guard):
            return None, None
        guard = [dict(u, end=min(u['end'],limit), acoustic_end=acoustic_end(u),
                      timing_source='linear_fallback' if u.get('bounds_source')=='cross_model_onset_prior_acoustic_end' else
                      {'ctc_event':'cross_model_ctc','attention_span':'cross_model_attention'}.get(
                          u.get('timing_evidence'),'cross_model_phonemes'))
                 for u in guard]
        suffix, source = guard+units[hi+1:], 'cross_model_onsets'
    else:
        # The last word has no following onset. Phoneme and grapheme CTC
        # events need not end together: require a stable terminal event and
        # an overlapping independent word event, not equal token durations.
        # The conservative envelope is only a playback boundary, never a
        # claimed singing offset for countdowns.
        tails = [v[-1] for v in [*report['predictions'], *views]]
        if any((u['first'],u['stop'],u['text']) != (tails[0]['first'],tails[0]['stop'],tails[0]['text'])
               or normalized(cue['text'][u['stop']:]) for u in tails):
            return None, None
        values = [u.get('emission_end') if u.get('timing_evidence')=='ctc_event' else u['end'] for u in tails]
        if any(not isinstance(e,(int,float)) or not np.isfinite(e) or e<=u['start']
               for e,u in zip(values,tails)):
            return None, None
        ends = [round(e,3) for e in values]
        center = float(np.median(ends[:3]))
        starts = [round(u['start'],3) for u in tails[:3]]
        event_start = float(np.median(starts))
        if (not np.isfinite(ends).all() or any(not anchor['start']<e<=bounds[1] for e in ends)
                or round(max(ends[:3])-min(ends[:3]),3)>protocol['maximum_spread_seconds']
                or round(max(starts)-min(starts),3)>protocol['maximum_spread_seconds']
                or any(round(e-center,3)>.1 for e in ends[3:])
                or not any(u['start']<=center+.1 and e>=event_start-.1 for u,e in zip(tails[3:],ends[3:]))):
            return None, None
        hi, stop, end = len(units), len(cue['text']), max(ends)
        suffix, source = [], 'cross_model_span_end'
    if hi <= lo:
        return None, None
    # Never hide another established onset inside a coarse group, or span
    # words that the acoustic witnesses place outside these two boundaries.
    changed_stop = guard[-1]['stop'] if guard is not None else stop
    covered = [p for p in constraints if p['line']==proof['line'] and first<=p['first']<changed_stop]
    if any(first<p['first']<stop for p in covered) or any(
            not anchor['start']<u['start']<end
            for view in views for u in view if first<u['first']<stop):
        return None, None
    group = dict(first=first, stop=stop, text=cue['text'][first:stop],
                 start=anchor['start'], end=end, acoustic_end=None,
                 timing_source='linear_fallback', bounds_source=source)
    candidate = dict(cue, units=units[:lo]+[group]+suffix)
    candidate.update(start=candidate['units'][0]['start'], aligned_start=candidate['units'][0]['start'],
                     end=candidate['units'][-1]['end'], aligned_end=candidate['units'][-1]['end'])
    try:
        validate_timeline([candidate], [cue['text']], len(wave)/44100,
                          [dict(p,line=1) for p in covered], language)
    except ValueError:
        return None, None
    return candidate, dict(first=first, stop=stop, accepted=True,
        reason='uncertain interior swept within corroborated acoustic bounds', mode='grouped_fallback')


def validate_timeline(cues, lines, duration, constraints=(), language='English'):
    """Publication contract shared by fine and fallback timing."""
    if not cues or [c['text'] for c in cues] != lines:
        raise ValueError('Final timing omits supplied lyric lines')
    units=[u for c in cues for u in c.get('units',[])]
    if not units or any(not 0<=u['start']<u['end']<=duration+.05 for u in units):
        raise ValueError('Final timing contains invalid word intervals')
    if any(a['start']>=b['start'] for a,b in zip(units,units[1:])):
        raise ValueError('Final timing contains unordered word onsets')
    for cue in cues:
        if normalized(''.join(u['text'] for u in cue['units']))!=normalized(cue['text']):
            raise ValueError('Final timing omits supplied lyric text')
        if collapsed_phrase(cue,language):
            raise ValueError('Final timing contains a collapsed phrase')
    if timing_violations(cues,constraints):
        raise ValueError('Final timing reuses contradicted acoustic onsets')


def acoustic_model_files(root, language=None):
    files=[root/'models/alignment/mms/model.pt',
           root/'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py']
    files.extend(p for p in (root/'models/alignment/sofa').rglob('*')
                 if p.suffix in ('.ckpt','.ptsd','.yaml') or p.name=='dict.txt')
    if language not in ('Chinese','Japanese','English'):
        files.append(root/'models/alignment/whisper/large-v3-turbo.pt')
    return sorted(p for p in files if p.is_file())


class AcousticReview:
    """Lazily load models and persist bounded acoustic requests, not decisions."""
    def __init__(self, original, device, cache, identity, protocol):
        from pathlib import Path
        from cjk_recovery import AcousticWitness
        self.device,self.cache,self.identity,self.protocol=device,Path(cache),identity,protocol
        self.witness=AcousticWitness(original,device)
        self.models={}
        self.cache.mkdir(parents=True,exist_ok=True)

    def cached(self, kind, request, compute):
        import hashlib,json
        key=hashlib.sha256(json.dumps([self.identity,kind,request],sort_keys=True).encode()).hexdigest()
        path=self.cache/(key+'.json')
        if path.is_file():return json.loads(path.read_text())
        value=compute()
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
        temporary.replace(path)
        return value

    def observe(self, cue, wave, *, language):
        request={k:cue.get(k) for k in ('text','aligned_start','aligned_end','audio_bounds')}
        return self.cached('witness',[language,request],lambda:self.witness(cue,wave,language=language))

    def predict(self, wave, *, decoder='ctc'):
        def run(cue, views, bounds, language):
            from phoneme_progress import spoken_units
            from subtitles import highlight_spans, latin_word_spans
            if language == 'Chinese':
                # Keep source offsets while giving each phonetic model only
                # its language. The caller still corroborates every onset.
                parts = []
                for unit in views[0]:
                    letters = [c for c in latin_word_text(unit['text']) if c.isalpha()]
                    local_language = 'English' if letters and all(c.isascii() for c in letters) else language
                    if parts and parts[-1][0] == local_language:
                        parts[-1][2] = unit['stop']
                    else:
                        parts.append([local_language, unit['first'], unit['stop']])
                if any(part[0] == 'English' for part in parts):
                    result = [[], [], []]
                    for local_language, first, stop in parts:
                        local_views = [[dict(u, first=u['first']-first, stop=u['stop']-first)
                                        for u in view if first <= u['first'] and u['stop'] <= stop]
                                       for view in views]
                        local_cue = dict(cue, text=cue['text'][first:stop], units=[])
                        for combined, predicted in zip(result, run(local_cue, local_views, bounds, local_language)):
                            combined.extend(dict(u, first=u['first']+first, stop=u['stop']+first) for u in predicted)
                    return result
            start=min(v[0]['start'] for v in views)
            end=max(v[-1]['end'] for v in views)
            windows=[(max(bounds[0],start-pad),min(bounds[1],end+pad))
                     for pad in self.protocol['context_padding_seconds']]
            # If neighbouring onsets collapse distinct crops, allow acoustic
            # context outside them. Accepted onsets still obey those bounds.
            if len(set(windows)) != 3:
                windows=[(max(0.,start-pad),min(len(wave)/44100,end+pad))
                         for pad in self.protocol['context_padding_seconds']]
            if len(set(windows))!=3 or any(b<=a or b-a>32 for a,b in windows):
                raise ValueError('Acoustic review needs three bounded distinct contexts')
            def compute():
                if language not in self.models:
                    if language=='English':
                        from align_english_sofa import EnglishSofaAligner
                        self.models[language]=EnglishSofaAligner(self.device)
                    elif language not in ('Chinese','Japanese'):
                        from align_lyrics_whisper import load_aligner
                        self.models[language]=load_aligner('large-v3-turbo',self.device)
                    else:
                        from align_lyric_phonemes import SingingAligner
                        self.models[language]=SingingAligner(language,self.device)
                model=self.models[language]
                result=[]
                for a,b in windows:
                    audio=wave[round(a*44100):round(b*44100)]
                    if language=='English':
                        spans=list(latin_word_spans(cue['text']))
                        words=[cue['text'][s:e] for s,e in spans]
                        predicted,_,_=model.align(audio,words)
                        if [u['text'] for u in predicted]!=words:
                            raise ValueError('Acoustic review changed lyric text')
                        result.append([dict(first=s,stop=e,text=u['text'],start=a+u['start'],end=a+u['end'],
                                            timing_evidence='phonetic_span')
                                       for (s,e),u in zip(spans,predicted)])
                    elif language not in ('Chinese','Japanese'):
                        from scipy.signal import resample_poly
                        from lyric_progress import map_words
                        from local_asr import alignment_language
                        clip=resample_poly(audio,160,441).astype('float32')
                        selected_language, probes=alignment_language(model,clip,language)
                        aligned=model.align(clip,
                            cue['text'],language=selected_language,original_split=True,regroup=False,token_step=80,
                            max_word_dur=None,word_dur_factor=None,gap_padding=None,
                            dynamic_heads=True,verbose=None)
                        if aligned is None:
                            raise ValueError('Multilingual acoustic review returned no word times')
                        words=[w for s in aligned.to_dict()['segments'] for w in s['words']]
                        units=map_words(dict(cue,start=a,end=b,aligned_start=a,aligned_end=b),words,a,language)
                        spans=list(highlight_spans(cue['text'],language))
                        if len(units)!=len(spans) or any(normalized(u['text'])!=normalized(cue['text'][s:e])
                                for u,(s,e) in zip(units,spans)):
                            raise ValueError('Multilingual acoustic review omitted or merged words')
                        result.append([dict(u,first=s,stop=e,text=cue['text'][s:e],timing_evidence='attention_span',
                                            alignment_language=selected_language, language_detection=probes)
                                       for u,(s,e) in zip(units,spans)])
                    else:
                        parsed=spoken_units(cue['text'],language)
                        result.append(model.infer(audio,a,parsed,
                            dict(cue,aligned_start=start,aligned_end=end),decoder=decoder,
                            use_word_priors=False)['units'])
                return result
            return self.cached('prediction',[language,cue['text'],windows,
                'phonemes' if language=='English' else decoder],compute)
        return run


def main():
    import argparse,hashlib,json,time
    from pathlib import Path
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from phoneme_progress import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--lyrics',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--lead',type=Path)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],required=True)
    parser.add_argument('--playback-fallback',action='store_true')
    parser.add_argument('--previous-alignment',type=Path,
                        help='Same-run reviewed candidate before a lyric-source retry')
    parser.add_argument('--context-windows',type=Path,
                        help='Observed phrase ranges for one bounded acoustic repair, never word timestamps')
    parser.add_argument('--defer-constraints',action='store_true',
                        help='Save preliminary evidence; the pipeline must enforce constraints before publication')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    def write(p,value):
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    cues,lyrics=read(args.input),read(args.lyrics)
    wave,rate=sf.read(args.audio,dtype='float32',always_2d=True)
    frames=len(wave)
    duration=frames/rate
    validate_timeline(cues,lyrics['lines'],duration,language=lyrics['language'])
    contexts = {}
    context_request = read(args.context_windows) if args.context_windows else {}
    for row in context_request.get('windows', []):
        index, (begin, end) = row['line']-1, row['source_range']
        begin, end = max(0.,begin), min(duration,end)
        if (not 0 <= index < len(cues) or row['text'] != cues[index]['text'] or
                not 0 <= begin < end <= duration or end-begin > 30):
            raise ValueError('Acoustic repair context does not match the current lyric or audio')
        contexts[index] = [begin, end]
    divisor=np.gcd(rate,44100)
    wave=resample_poly(wave.mean(axis=1),44100//divisor,rate//divisor).astype('float32')
    code_root=Path(__file__).resolve().parent
    protocol=read(code_root/'experiments/cjk-onsets-v003/protocol.json')
    source=dict(original=sha(args.original),vocals=sha(args.audio),lyrics=lyrics['lines'])
    if context_request and context_request['report']['source'] != source:
        raise ValueError('Acoustic repair observations refer to a different recording or lyric')
    ledger_file=out.parent/'timing-constraints.json'
    ledger=read(ledger_file) if ledger_file.is_file() else {}
    paths=[code_root/name for name in ('timing_validation.py','cjk_recovery.py','align_lyric_phonemes.py',
        'phoneme_progress.py','align_english_sofa.py','align_english_onsets.py','subtitles.py','anchors_from_asr.py')]
    if lyrics['language'] not in ('Chinese','Japanese','English'):
        paths.extend(code_root/name for name in ('align_lyrics_whisper.py','lyric_progress.py',
                                                'local_asr.py','job_schema.py','requirements-subtitles.txt'))
    if lyrics['language'] == 'Cantonese':
        paths.extend(code_root/name for name in ('subtitle_style.py', 'subtitle-styles.json'))
    models=acoustic_model_files(ROOT, lyrics['language'])
    identity=dict(source=source,protocol=protocol,code={str(p):sha(p) for p in paths},
                  models={str(p):sha(p) for p in models if p.is_file()})
    lead_sha256 = sha(args.lead) if args.lead is not None else None
    same_evidence = (ledger.get('identity') == identity and
                     (not ledger.get('lead_sha256') or ledger['lead_sha256'] == lead_sha256))
    constraints = ledger.get('constraints', []) if same_evidence else []
    if ledger and not same_evidence:
        history = out.parent/'timing-history'
        history.mkdir(exist_ok=True)
        write(history/(sha(ledger_file)+'.json'), ledger)
    reused_source = None
    if args.previous_alignment is not None:
        prior = args.previous_alignment
        proof = read(prior/'timing-constraints.json')
        prior_identity = deepcopy(proof.get('identity', {}))
        if 'source' in prior_identity:
            prior_identity['source']['lyrics'] = lyrics['lines']
        lead_file = out.parent/'stems/lead.wav'
        lead_matches = (not proof.get('lead_sha256') or
                        (lead_file.is_file() and proof['lead_sha256'] == sha(lead_file)))
        if (prior_identity == identity and lead_matches and
                read(prior/'lyrics.json')['language'] == lyrics['language']):
            old = read(prior/'timing-checked/cues.json')
            old_report = read(prior/'timing-checked/report.json')
            # The pipeline writes cues only after the full timing contract.
            # A later text-coverage failure does not invalidate those timings.
            if (old == read(prior/'cues.json') and old_report.get('source') == proof.get('source') and
                    [c['text'] for c in old] == proof['source']['lyrics']):
                cues, carried, retained = reuse_expanded_timeline(old,cues,old_report,duration,lyrics['language'])
                constraints.extend(p for p in carried if p not in constraints)
                reused_source = dict(directory=str(prior), cues_sha256=sha(prior/'cues.json'), retained_lines=retained)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    review=AcousticReview(args.original,args.device,out.parent/'timing-cache',identity,protocol)
    started=time.perf_counter()
    def observations():
        yield 'vocals_ctc', wave, review.observe, review.predict(wave)
        if lyrics['language'] in ('Chinese','Japanese'):
            yield 'vocals_phonemes', wave, review.observe, review.predict(wave,decoder='phonemes')
        if args.lead is not None:
            lead, lead_rate=sf.read(args.lead,dtype='float64',always_2d=True)
            # Resampling rounds the same endpoint on two sample grids. Use
            # integer frame counts and one frame of the coarser grid.
            if abs(len(lead)*rate-frames*lead_rate) > max(rate,lead_rate):
                raise ValueError('Alternate acoustic view does not match the source duration')
            divisor=np.gcd(lead_rate,44100)
            lead=resample_poly(lead.mean(axis=1),44100//divisor,lead_rate//divisor).astype('float32')
            lead_review=AcousticReview(args.original,args.device,out.parent/'timing-cache',
                dict(identity,lead_sha256=sha(args.lead)),protocol)
            yield 'lead_ctc', lead, lead_review.observe, lead_review.predict(lead)
            if lyrics['language'] in ('Chinese','Japanese'):
                yield 'lead_phonemes', lead, lead_review.observe, lead_review.predict(lead,decoder='phonemes')
    previous = ledger.get('observations', []) if same_evidence else []
    if args.lead is None or ledger.get('lead_sha256') != lead_sha256:
        previous = [row for row in previous if not row.get('observation', '').startswith('lead_')]
    repaired,report=reconcile_views(cues,observations(),protocol,lyrics['language'],constraints,
                                   playback=args.playback_fallback,previous=previous,contexts=contexts)
    if context_request:
        report['lines'] = [*context_request['report']['lines'], *report['lines']]
    localization=out.parent/'alignment/raw.json'
    active,superseded=localized_constraints(report,read(localization) if localization.is_file() else [])
    superseded=(ledger.get('superseded',[]) if same_evidence else [])+superseded
    revised=repaired
    repaired,retained=retain_consistent_input(cues,revised,active,duration,lyrics['language'])
    report['retained_input_lines']=retained
    report['discarded_revisions']=[dict(line=i,cue=revised[i-1]) for i in retained]
    report.update(constraints=active,unresolved=timing_violations(repaired,active),superseded_constraints=superseded)
    report['pending_review_lines']=[i+1 for i in sorted(pending_review_lines(repaired,report,protocol))]
    if args.lead is not None:
        report['lead_sha256']=lead_sha256
    report.update(seconds=time.perf_counter()-started,source=source,input_sha256=sha(args.input))
    if contexts:
        report['context_windows'] = contexts
    if reused_source is not None:
        report['reused_source_alignment'] = reused_source
    measurements = list({json.dumps(row, sort_keys=True): row for row in [*previous, *report['lines']]
                         if row.get('proposed_constraints') and not row.get('reused_measurement')}.values())
    write(ledger_file,dict(source=source,constraints=report['constraints'],superseded=superseded,
        identity=identity, observations=measurements, lead_sha256=report.get('lead_sha256')))
    write(out/'cues.json',repaired);write(out/'report.json',report)
    validate_timeline(repaired,lyrics['lines'],duration,
                      () if args.defer_constraints else report['constraints'],lyrics['language'])
    print(json.dumps(dict(lines=len(cues),reviewed_lines=len({r['line'] for r in report['lines']}),
                         review_attempts=len(report['lines']),
                         constraints=len(report['constraints']),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    main()
