"""One acoustic timing contract for normal alignment and playback fallbacks.

A disagreement is a screening signal. Only corroborated acoustic predictions
invalidate an old onset; unavailable evidence never turns a quiet voice into
silence. Invalidations outlive individual fallback candidates.
"""
from copy import deepcopy

import numpy as np

if __package__:
    from .subtitles import acoustic_end, collapsed_phrase, normalized, highlight_coverage_errors
else:
    from subtitles import acoustic_end, collapsed_phrase, normalized, highlight_coverage_errors


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


def localized_constraints(cues, report, blocks):
    """Retire a wrong-phrase proof only after relocation and new acoustic proof.

    A phonetic aligner assumes its supplied words occur in the crop. A later
    independently recognized phrase can disprove that premise. Keep both
    proofs and the localization evidence instead of making a bad crop eternal.
    """
    active, superseded = [], []
    proposals = [p for row in report['lines'] for p in row.get('proposed_constraints', [])
                 if not timing_violations(cues, [p])]
    observations = [o for block in blocks for o in block.get('asr_localizations', [])]
    for proof in report['constraints']:
        replacement = None
        if timing_violations(cues, [proof]):
            for observed in observations:
                # A stretched old phrase can overlap the right passage while
                # one of its word proofs points elsewhere. Compare the proof
                # itself with the newly located phrase, not the old envelope.
                if proof['line'] not in observed.get('accepted_lines', []):
                    continue
                index = proof['line']-1-observed['context'][0]
                anchors = observed.get('anchors', [])
                if not 0 <= index < len(anchors):
                    continue
                anchor = anchors[index]
                if anchor['status']!='candidate' or anchor['text']!=proof['text']:
                    continue
                if anchor['start']-2 <= proof['onset'] <= anchor['end']+2:
                    continue
                revised = next((p for p in proposals if all(p[k]==proof[k] for k in ('line','text','first','stop'))
                    and anchor['start']-2 <= p['onset'] <= anchor['end']+2), None)
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


def displaced_units(cue, views, tolerance):
    """Coarse agreement selects another observation, never assigns a time."""
    old = {u['first']: u for u in cue['units']}
    if len(views) != 2 or not views[0] or len(views[0]) != len(views[1]):
        return []
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


def reconcile_views(cues, observations, protocol, language, constraints=(), *, playback=False, previous=(), resume=None):
    """Spend additional acoustic work only on unresolved local disagreements.

    Each observation uses the same acceptance contract and retains all previous
    onset constraints. The iterator may load a different audio view lazily.
    """
    result = cues
    report = deepcopy(resume) if resume is not None else dict(lines=[])
    report['constraints'] = list(constraints)
    pending = pending_review_lines(cues, report, protocol) if resume is not None else None
    if resume is not None and not pending:
        return result, report
    history = []
    for name, wave, witness, predict in observations:
        result, reviewed = reconcile_timing(result, wave, witness, predict, protocol, language,
            report['constraints'], playback=playback, _review_lines=pending)
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
                report['constraints'], playback=playback, _retry_context=False, _review_lines=unresolved_lines)
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
                    _retry_context=False, _review_lines={line-1})
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
                candidate = compose_timing_bounds(result, index, row, known, report['constraints'],
                                                   recorded_wave, protocol, language)
                if candidate is not None:
                    result[index] = candidate
                    report['lines'].append(dict(line=index+1, observation=row['observation'], status='reviewed',
                        reused_measurement=True, repairs=[dict(accepted=True, mode='composed_acoustic_bounds',
                        reason='local sweep between independently measured onsets')]))
            report['unresolved'] = timing_violations(result, report['constraints'])
        pending = pending_review_lines(result, report, protocol)
        if not pending:
            break
    report['pending_review_lines'] = [i+1 for i in sorted(pending or [])]
    return result, report


def compose_timing_bounds(cues, index, row, measured, constraints, wave, protocol, language):
    """Compose local playback groups from durable, independently measured onsets.

    A previous observation can locate one word and a later observation its
    neighbour. Requiring both to succeed in the same model call discards that
    evidence. Interior word times stay unspecified; the two acoustic bounds
    and every established onset remain explicit.
    """
    cue = cues[index]
    proofs = [p for p in constraints if p['line']==index+1]
    failed = timing_violations(cues, proofs)
    if not failed or len(row.get('witnesses', [])) != 2:
        return None
    if any(normalized(''.join(u['text'] for u in view)) != normalized(cue['text']) or
           any(u['text'] != cue['text'][u['first']:u['stop']] for u in view) for view in row['witnesses']):
        return None
    views = [{u['first']:u for u in view} for view in row['witnesses']]
    points = {}
    for unit in measured:
        first, time = unit['first'], unit['start']
        if all(first in view and abs(view[first]['start']-time)<=.1 for view in views):
            points.setdefault(first, time)
    for proof in proofs:
        first, time = proof['first'], proof['onset']
        if (not all(first in view and abs(view[first]['start']-time)<=.1 for view in views)
                or any(p['first']==first and not p['start_min']<=time<=p['start_max'] for p in proofs)):
            return None
        points[first] = time
    positions = {u['first']:i for i,u in enumerate(cue['units'])}
    first, last = min(v['proof']['first'] for v in failed), max(v['proof']['first'] for v in failed)
    before = cues[index-1]['units'][-1]['start'] if index else -.015
    after = cues[index+1]['units'][0]['start'] if index+1<len(cues) else len(wave)/44100
    floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    for left in sorted((p for p in points if p<=first and p in positions), reverse=True):
        lo = positions[left]
        if points[left] <= (cue['units'][lo-1]['start'] if lo else before)+.015:
            continue
        for right in sorted(p for p in points if p>last and p in positions):
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
                     playback=False, _retry_context=True, _review_lines=None):
    """Corroborate displaced onsets, then replace only a compatible local span.

    witness returns original/vocal word views. predict returns three independent
    context predictions without old word priors. Both are cacheable adapters.
    """
    from cjk_recovery import corroborated_group, same_acoustic_event
    result, reports, ledger = deepcopy(cues), [], deepcopy(list(constraints))
    context_hints = {}
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
        if not selected and not timing_violations(result, [p for p in ledger if p['line']==index+1]):
            continue
        report = dict(line=index+1, screened_units=selected, status='unverified', repairs=[])
        reports.append(report)
        try:
            # A stale first word can stretch the coarse search across an
            # interlude. Locate the phrase before applying the short phonetic
            # window budget; predict() still requires three contexts <= 32 s.
            if cue['aligned_end']-cue['aligned_start'] > 60:
                raise ValueError('Acoustic localization context exceeds 60 seconds')
            letters = [c for c in cue['text'] if c.isalpha()]
            review_language = 'English' if letters and all(c.isascii() for c in letters) else language
            views = witness(cue, wave, language=review_language)
            report['witnesses'] = views
            signature = lambda v: [(u['first'], u['stop'], u['text']) for u in v]
            if (len(views)!=2 or not views[0] or signature(views[0]) != signature(views[1])
                    or normalized(''.join(u['text'] for u in views[0])) != normalized(cue['text'])
                    or any(u['text'] != cue['text'][u['first']:u['stop']] for u in views[0])):
                raise ValueError('Incomplete acoustic text evidence')
            bounds = [result[index-1]['units'][-1]['start']+.015 if index else 0.,
                      result[index+1]['units'][0]['start'] if index+1<len(result) else len(wave)/44100]
            if any(not np.isfinite([u['start'],u['end']]).all()
                   or not bounds[0] <= u['start'] < bounds[1]
                   or not u['start'] < u['end'] <= len(wave)/44100 for v in views for u in v) or any(
                       a['start'] >= b['start'] for v in views for a,b in zip(v,v[1:])):
                raise ValueError('Acoustic localization crosses neighbouring lyrics')
            old = {u['first']:u for u in cue['units']}
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
            suspects = [a for a,b in zip(*views) if a['first'] in old
                        and same_acoustic_event(a, b, protocol['maximum_change_seconds'])
                        and (a['first'] in challenged or
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
                replacement = dict(replacement[0], timing_source='cross_model_ctc' if
                    replacement[0].get('timing_evidence')=='ctc_event' else 'cross_model_phonemes',
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
                attempt=dict(first=units[lo]['first'],stop=units[hi-1]['stop'],accepted=reason is None,reason=reason)
                report['repairs'].append(attempt)
                # An isolated matching syllable is not a durable constraint
                # when the surrounding replacement maps to different words
                # in the two recordings. Keep that proposal for a bounded
                # group retry; only a coherent local span establishes it.
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
                                               _retry_context=False, _review_lines=windows)
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
                    wave, bounds, protocol, language)
                if candidate is None:
                    continue
                anchor = next((u for u in candidate['units'] if u['first']==first), None)
                if anchor is None or abs(anchor['start']-old['start']) <= protocol['maximum_change_seconds']:
                    continue
                proof.update(stop=anchor['stop'],onset=anchor['start'],
                             start_min=max(0.,anchor['start']-protocol['maximum_change_seconds']),
                             start_max=anchor['start']+protocol['maximum_change_seconds'])
                ledger.append(proof)
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
                        wave, bounds, protocol, language)
                    if candidate is not None:
                        result[index] = candidate
                        row['repairs'].append(repair)
            unresolved = len(timing_violations(result, ledger))
            if unresolved >= remaining:
                break
            remaining = unresolved
    return result,dict(lines=reports,constraints=ledger,unresolved=timing_violations(result,ledger),
                       accuracy=None)


def grouped_timing_fallback(cue, report, proof, constraints, wave, bounds, protocol, language):
    """Sweep an uncertain interior only within corroborated acoustic bounds.

    This is a playback group, never a replacement word-accuracy measurement.
    The first corroborated following source unit closes the smallest group.
    A source unit may already contain several words from an earlier fallback.
    """
    from cjk_recovery import corroborated_group
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
            wave, bounds, protocol, spans=[(u['first'],u['stop']) for u in selected[0]])
        if found and boundary:
            start = found[0]['start']
            piece = wave[round(start*44100):round((start+.035)*44100)]
            floor = max(1e-5,protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
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
                      'cross_model_ctc' if u.get('timing_evidence')=='ctc_event' else 'cross_model_phonemes')
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


def acoustic_model_files(root):
    files=[root/'models/alignment/mms/model.pt',
           root/'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py']
    files.extend(p for p in (root/'models/alignment/sofa').rglob('*')
                 if p.suffix in ('.ckpt','.ptsd','.yaml') or p.name=='dict.txt')
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
            from subtitles import highlight_spans
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
                    else:
                        from align_lyric_phonemes import SingingAligner
                        self.models[language]=SingingAligner(language,self.device)
                model=self.models[language]
                result=[]
                for a,b in windows:
                    audio=wave[round(a*44100):round(b*44100)]
                    if language=='English':
                        spans=list(highlight_spans(cue['text'],language))
                        words=[cue['text'][s:e] for s,e in spans]
                        predicted,_,_=model.align(audio,words)
                        if [u['text'] for u in predicted]!=words:
                            raise ValueError('Acoustic review changed lyric text')
                        result.append([dict(first=s,stop=e,text=u['text'],start=a+u['start'],end=a+u['end'])
                                       for (s,e),u in zip(spans,predicted)])
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
    parser.add_argument('--defer-constraints',action='store_true',
                        help='Save preliminary evidence; the pipeline must enforce constraints before publication')
    parser.add_argument('--resume',type=Path,help='Continue this preliminary review on the lead master')
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
    divisor=np.gcd(rate,44100)
    wave=resample_poly(wave.mean(axis=1),44100//divisor,rate//divisor).astype('float32')
    code_root=Path(__file__).resolve().parent
    protocol=read(code_root/'experiments/cjk-onsets-v003/protocol.json')
    source=dict(original=sha(args.original),vocals=sha(args.audio),lyrics=lyrics['lines'])
    ledger_file=out.parent/'timing-constraints.json'
    ledger=read(ledger_file) if ledger_file.is_file() else {}
    paths=[code_root/name for name in ('timing_validation.py','cjk_recovery.py','align_lyric_phonemes.py',
        'phoneme_progress.py','align_english_sofa.py','align_english_onsets.py','subtitles.py')]
    models=acoustic_model_files(ROOT)
    identity=dict(source=source,protocol=protocol,code={str(p):sha(p) for p in paths},
                  models={str(p):sha(p) for p in models if p.is_file()})
    resume_key=hashlib.sha256(json.dumps([identity,sha(args.input),sha(args.lyrics),args.device,
                                        args.playback_fallback],sort_keys=True).encode()).hexdigest()
    resume=None
    if args.resume and args.lead is not None:
        try:
            saved=read(args.resume/'report.json')
            if (isinstance(saved,dict) and saved.get('resume_key')==resume_key and not saved.get('lead_sha256') and
                    saved.get('output_sha256')==sha(args.resume/'cues.json')):
                cues,resume=read(args.resume/'cues.json'),saved
        except (OSError,ValueError):
            pass
    lead_sha256 = sha(args.lead) if args.lead is not None else None
    same_evidence = (ledger.get('identity') == identity and
                     (not ledger.get('lead_sha256') or ledger['lead_sha256'] == lead_sha256))
    constraints = ledger.get('constraints', []) if same_evidence else []
    if ledger and not same_evidence:
        history = out.parent/'timing-history'
        history.mkdir(exist_ok=True)
        write(history/(sha(ledger_file)+'.json'), ledger)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    started=time.perf_counter()
    def observations():
        if resume is None:
            review=AcousticReview(args.original,args.device,out.parent/'timing-cache',identity,protocol)
            yield 'vocals_ctc', wave, review.observe, review.predict(wave)
            if lyrics['language'] != 'English':
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
            if lyrics['language'] != 'English':
                yield 'lead_phonemes', lead, lead_review.observe, lead_review.predict(lead,decoder='phonemes')
    previous = ledger.get('observations', []) if same_evidence else []
    if args.lead is None or ledger.get('lead_sha256') != lead_sha256:
        previous = [row for row in previous if not row.get('observation', '').startswith('lead_')]
    repaired,report=reconcile_views(cues,observations(),protocol,lyrics['language'],
                                   resume['constraints'] if resume is not None else constraints,
                                   playback=args.playback_fallback,previous=previous,resume=resume)
    localization=out.parent/'alignment/raw.json'
    active,superseded=localized_constraints(repaired,report,read(localization) if localization.is_file() else [])
    superseded=(ledger.get('superseded',[]) if same_evidence else [])+superseded
    report.update(constraints=active,unresolved=timing_violations(repaired,active),superseded_constraints=superseded)
    report['pending_review_lines']=[i+1 for i in sorted(pending_review_lines(repaired,report,protocol))]
    if args.lead is not None:
        report['lead_sha256']=lead_sha256
    report.update(seconds=time.perf_counter()-started,source=source,input_sha256=sha(args.input))
    measurements = list({json.dumps(row, sort_keys=True): row for row in [*previous, *report['lines']]
                         if row.get('proposed_constraints') and not row.get('reused_measurement')}.values())
    write(ledger_file,dict(source=source,constraints=report['constraints'],superseded=superseded,
        identity=identity, observations=measurements, lead_sha256=report.get('lead_sha256')))
    write(out/'cues.json',repaired)
    report.update(resume_key=resume_key,output_sha256=sha(out/'cues.json'),continued_from_preliminary=resume is not None)
    write(out/'report.json',report)
    validate_timeline(repaired,lyrics['lines'],duration,
                      () if args.defer_constraints else report['constraints'],lyrics['language'])
    print(json.dumps(dict(lines=len(cues),reviewed_lines=len({r['line'] for r in report['lines']}),
                         review_attempts=len(report['lines']),
                         constraints=len(report['constraints']),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    main()
