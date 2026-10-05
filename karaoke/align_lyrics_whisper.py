"""Force-align lyrics to vocal audio without video or external subtitle times."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_closing
import soundfile as sf
import stable_whisper
from stable_whisper import timing as whisper_timing
import torch
from whisper.timing import dtw_cpu

from lyric_progress import map_words
from subtitles import lyric_cues, normalized, presentation_cues, write_captions
from stage_progress import progress_items, report_progress


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_number(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def align_candidate(align, *args, **kwargs):
    """Treat stable-ts's empty attention window as an unusable candidate."""
    try:
        return align(*args, **kwargs)
    except RuntimeError as error:
        if str(error) != 'selected index k out of range':
            raise  # Resource failures and unexpected model errors stay fatal.
        raise ValueError(f'Unusable alignment window: {error}') from error


def valid_recovery(candidate, lines, duration):
    if candidate is None or [normalized(''.join(w['word'] for w in s['words']))
            for s in candidate['segments']] != [normalized(line) for line in lines]:
        return False
    units = [dict(text=w['word'], start=w['start'], end=w['end'])
             for s in candidate['segments'] for w in s['words'] if normalized(w['word'])]
    if not units or any(not np.isfinite([u['start'], u['end']]).all() or
                       not 0 <= u['start'] < u['end'] <= duration+.001 for u in units):
        return False
    if any(a['start'] >= b['start'] for a, b in zip(units, units[1:])):
        return False
    try:
        presentation_cues(lyric_cues(lines, units, duration), duration, lead=0, tail=0, bridge_gap=0)
    except ValueError:
        return False
    return True


def recover_changed_text(result, lines, duration, realign):
    """Retry a malformed alignment, never edit its words to fit the lyrics."""
    expected = [normalized(line) for line in lines]

    def matches(candidate):
        return candidate is not None and [normalized(''.join(w['word'] for w in s['words']))
                for s in candidate['segments']] == expected

    if matches(result):
        return result, None
    candidate = realign()
    report = dict(original_result=result, retry_result=candidate, accepted=False,
                  method='dynamic attention heads without inter-line gap padding')
    if valid_recovery(candidate, lines, duration):
        report['accepted'] = True
        return candidate, report
    return result, report


def recover_unresolved_block(block, lines, anchors, duration, align):
    """Retry wholly untimed lines with another neighbour and no speech-duration cap."""
    if block['result']['segments'] and all(any(w['end'] > w['start'] for w in s['words'] if normalized(w['word']))
           for s in block['result']['segments']):
        return block
    left, right = block['context']
    begin, end = block['source_range']
    if anchors:
        if left and anchors[left-1]['status'] == 'candidate':
            left -= 1
            begin = min(begin, max(0., anchors[left]['start']-.6))
        if right < len(lines) and anchors[right]['status'] == 'candidate':
            end = max(end, min(duration, anchors[right]['end']+.6))
            right += 1
    candidate = align(begin, end, '\n'.join(lines[left:right]),
                      max_word_dur=None, word_dur_factor=None, gap_padding=None)
    accepted = valid_recovery(candidate, lines[left:right], end-begin)
    report = dict(original_context=block['context'], original_source_range=block['source_range'],
                  original_result=block['result'], context=[left, right], source_range=[begin, end],
                  retry_result=candidate, accepted=accepted)
    block = dict(block, unresolved_recovery=report)
    if accepted:
        block.update(context=[left, right], source_range=[begin, end], result=candidate)
    return block


def recover_unresolved_lines(block, lines, duration, language, align, max_attempts=8):
    """Re-align only broken runs inside their measured neighbouring phrases."""
    from lyric_progress import partial_alignment_cues
    first, stop = block['context']
    if len(block['result']['segments']) != stop-first:
        return dict(block, line_recoveries=[dict(accepted=False, reason='segment_count_mismatch')])
    result = deepcopy(block)
    cues = partial_alignment_cues(lines, [result], duration, language)
    bad = [i for i in range(*block['range']) if cues[i]['status'] != 'candidate']
    reports, attempts = [], 0
    while bad:
        begin_line = bad.pop(0)
        end_line = begin_line+1
        while bad and bad[0] == end_line:
            bad.pop(0)
            end_line += 1
        left = begin_line-1 if begin_line > first and cues[begin_line-1]['status'] == 'candidate' else begin_line
        right = end_line+1 if end_line < stop and cues[end_line]['status'] == 'candidate' else end_line
        begin = max(block['source_range'][0], cues[left]['start']-.6) if left < begin_line else block['source_range'][0]
        end = min(block['source_range'][1], cues[right-1]['end']+.6) if right > end_line else block['source_range'][1]
        report = dict(lines=[begin_line+1, end_line], context=[left, right],
                      source_range=[begin, end], accepted=False)
        reports.append(report)
        if left == begin_line and right == end_line:
            report['reason'] = 'no_measured_neighbour'
            continue
        if not 0 <= begin < end <= duration+.001 or end-begin > 45:
            report['reason'] = 'requires_coarse_localization'
            continue
        if attempts >= max_attempts:
            report['reason'] = 'local_retry_budget_exhausted'
            continue
        attempts += 1
        try:
            candidate = align_candidate(align, begin, end, '\n'.join(lines[left:right]),
                              gap_padding=None, dynamic_heads=True,
                              max_word_dur=None, word_dur_factor=None)
        except ValueError as error:
            report.update(reason='local_alignment_failed', error=str(error))
            continue
        report['result'] = candidate
        if not valid_recovery(candidate, lines[left:right], end-begin):
            report['reason'] = 'invalid_acoustic_result'
            continue
        replacement = deepcopy(candidate['segments'][begin_line-left:end_line-left])
        for segment in replacement:
            for word in segment['words']:
                for key in ('start', 'end'):
                    word[key] = round(word[key]+begin-block['source_range'][0], 3)
            segment.update(start=segment['words'][0]['start'], end=segment['words'][-1]['end'])
        updated = deepcopy(result)
        updated['result']['segments'][begin_line-first:end_line-first] = replacement
        proposed = partial_alignment_cues(lines, [updated], duration, language)
        try:
            # Context is read-only. A retry must fit the original neighbouring
            # timings rather than moving an already usable phrase out of its way.
            presentation_cues(proposed[left:right], duration, lead=0, tail=0, bridge_gap=0)
            words = [w for cue in proposed[left:right] for w in cue['units']]
            if any(a['start'] >= b['start'] for a, b in zip(words, words[1:])):
                raise ValueError('word_order_conflict')
        except (KeyError, ValueError):
            report['reason'] = 'conflicts_with_preserved_neighbours'
            continue
        report.update(accepted=True, original_segments=result['result']['segments'][begin_line-first:end_line-first])
        result, cues = updated, proposed
    if reports:
        result['line_recoveries'] = reports
    return result


def recover_with_local_asr(block, lines, audio, language, recognize, align, *,
                          force_global=False, max_alignment_attempts=24):
    """Use recognition only to locate failed lyrics, never as replacement text."""
    from anchors_from_asr import anchors_from_words
    from lyric_progress import partial_alignment_cues
    duration = len(audio)/16000
    cues = partial_alignment_cues(lines, [block], duration, language)
    bad = {i for i in range(*block['range']) if cues[i]['status'] != 'candidate'}
    first, stop = block['context']
    if len(block['result']['segments']) != stop-first:
        return block
    for i in range(*block['range']):
        words = [w for w in block['result']['segments'][i-first]['words'] if normalized(w['word'])]
        if len(words) >= 4 and sum(w['end']-w['start'] < .015 for w in words) > len(words)/2:
            bad.add(i)  # A tiny surviving token must not hide a collapsed phrase.
    if not bad:
        return block
    # Widespread collapse makes the old neighbouring times unsafe as anchors.
    # Otherwise recognize just the failed run plus its two neighbouring lines.
    windows, remaining = [], sorted(bad)
    while remaining:
        left = remaining.pop(0)
        right = left+1
        while remaining and remaining[0] == right:
            remaining.pop(0)
            right += 1
        left = left-1 if left > first else left
        right = right+1 if right < stop else right
        begin = max(0., cues[left]['start']-.6) if cues[left]['status'] == 'candidate' else 0.
        end = min(duration, cues[right-1]['end']+.6) if cues[right-1]['status'] == 'candidate' else duration
        windows.append((left, right, begin, end))
    global_recovery = force_global or len(bad) >= max(4, (stop-first+4)//5) or any(e-b > 45 for _, _, b, e in windows)
    if global_recovery:
        windows = [(first, stop, block['source_range'][0], block['source_range'][1])]
    result, reports, budget = deepcopy(block), [], max_alignment_attempts
    for left, right, begin, end in windows[:8]:
        report = dict(context=[left, right], source_range=[begin, end], global_localization=global_recovery,
                      accepted_lines=[], attempts=[])
        reports.append(report)
        if not 0 <= begin < end <= duration or end-begin > 600:
            report['reason'] = 'localization_budget_exceeded'
            continue
        try:
            response = recognize(begin, end)
        except ValueError as error:
            report.update(reason='recognition_failed', error=str(error))
            continue
        report['recognition'] = response
        words = [dict(w, start=w['start']+begin, end=w['end']+begin)
                 for s in response.get('segments', [])
                 if s.get('no_speech_prob', 0) < .6 and s.get('compression_ratio', 0) < 2.4
                 for w in s.get('words') or [] if np.isfinite([w['start'], w['end']]).all()
                 and 0 <= w['start'] < w['end'] <= end-begin+.001]
        anchors = anchors_from_words(lines[left:right], words, language, allow_missing=True)
        if global_recovery:
            report['recognition_anchors'] = anchors
            anchors = recovery_anchors(anchors, cues[left:right], {i-left for i in bad})
        report['anchors'] = anchors
        if not any(a['status'] == 'candidate' for a in anchors):
            report['reason'] = 'insufficient_recognition_support'
            continue
        targets = bad.intersection(range(left, right))
        mislocalized = set()
        if global_recovery:
            # A several-second contradiction from independently recognized
            # words is evidence that a superficially valid old line also drifted.
            mislocalized.update(i+left for i, a in enumerate(anchors) if a['status'] == 'candidate'
                and cues[i+left]['status'] == 'candidate' and (
                    abs(cues[i+left]['start']-a['start']) > 4 or
                    cues[i+left]['end'] < a['start'] or cues[i+left]['start'] > a['end']))
            for i, anchor in enumerate(anchors):
                if anchor['status'] == 'candidate' or cues[i+left]['status'] != 'candidate':
                    continue
                before = [a for a in anchors[:i] if a['status'] == 'candidate']
                after = [a for a in anchors[i+1:] if a['status'] == 'candidate']
                if before and after and (cues[i+left]['start'] < before[-1]['start']-2 or
                                         cues[i+left]['end'] > after[0]['end']+2):
                    mislocalized.add(i+left)
            targets.update(mislocalized)
        report['contradicted_lines'] = sorted(i+1 for i in mislocalized)
        proposed, replaced, contexts = deepcopy(result), [], []
        try:
            blocks = alignment_blocks(anchors, audio)
        except ValueError:
            report['reason'] = 'invalid_recognition_anchors'
            continue
        for start, stop_line, cf, cs, ab, ae in blocks:
            selected = targets.intersection(range(start+left, stop_line+left))
            if ae-ab > 45 and all(anchors[i]['status'] == 'candidate' for i in (start, stop_line-1)):
                # A long neighbouring intro must not suppress an otherwise
                # well-localized verse. Its core still needs acoustic alignment.
                cf, cs = start, stop_line
                ab, ae = max(0., anchors[cf]['start']-.6), min(duration, anchors[cs-1]['end']+.6)
            if ae-ab > 45 and anchors[start]['status'] != 'candidate' and anchors[stop_line-1]['status'] != 'candidate':
                # Search an untimed opening/ending near its one measured
                # neighbour; the crop is a search bound, never a lyric time.
                if cf == start and cs > stop_line:
                    ab = max(ab, ae-45)
                elif cf < start and cs == stop_line:
                    ae = min(ae, ab+45)
            # One recognized line can straddle an MV's long spoken interlude.
            # Give that isolated line one bounded acoustic search, not a skip.
            limit = 90 if stop_line-start == 1 and cf == start and cs == stop_line and anchors[start]['status'] == 'candidate' else 45
            if not selected or not 0 < ae-ab <= limit or budget <= 0:
                continue
            budget -= 1
            attempt = dict(lines=[start+left+1, stop_line+left], source_range=[ab, ae])
            report['attempts'].append(attempt)
            try:
                candidate = align_candidate(align, ab, ae, '\n'.join(lines[cf+left:cs+left]), gap_padding=None,
                                  dynamic_heads=True, max_word_dur=None, word_dur_factor=None)
            except ValueError as error:
                attempt['error'] = str(error)
                continue
            attempt['result'] = candidate
            if candidate is None:
                continue
            if any(not np.isfinite([w['start'], w['end']]).all() or
                   not 0 <= w['start'] <= w['end'] <= ae-ab+.001
                   for s in candidate['segments'] for w in s['words']):
                continue
            part = dict(range=[start+left, stop_line+left], context=[cf+left, cs+left],
                        source_range=[ab, ae], result=candidate)
            if valid_recovery(candidate, lines[cf+left:cs+left], ae-ab):
                # Keep the joint context prediction for later boundary repair;
                # another window's core can disagree with this valid overlap.
                contexts.append(dict(part, range=[start+left, start+left]))
            fresh = partial_alignment_cues(lines, [part], duration, language)
            for i in sorted(selected):
                cue, anchor = fresh[i], anchors[i-left]
                if cue['status'] != 'candidate':
                    continue
                if anchor['status'] == 'candidate':
                    if cue['end'] < anchor['start']-2 or cue['start'] > anchor['end']+2:
                        continue
                else:
                    before = [a for a in anchors[cf:i-left] if a['status'] == 'candidate']
                    after = [a for a in anchors[i-left+1:cs] if a['status'] == 'candidate']
                    low, high = before[-1]['start'] if before else ab, after[0]['end'] if after else ae
                    if not (before or after) or not low <= cue['start'] < cue['end'] <= high:
                        continue
                segment = deepcopy(candidate['segments'][i-cf-left])
                for word in segment['words']:
                    for key in ('start', 'end'):
                        word[key] = round(word[key]+ab-block['source_range'][0], 3)
                segment.update(start=segment['words'][0]['start'], end=segment['words'][-1]['end'])
                proposed['result']['segments'][i-first] = segment
                replaced.append(i)
        def ordered(candidate):
            fresh = partial_alignment_cues(lines, [candidate], duration, language)
            usable = [c for c in fresh if c['status'] == 'candidate']
            try:
                presentation_cues(usable, duration, lead=0, tail=0, bridge_gap=0)
                units = [w for c in usable for w in c['units']]
                if any(a['start'] >= b['start'] for a, b in zip(units, units[1:])):
                    return None
            except ValueError:
                return None
            return fresh

        fresh = ordered(proposed)
        if fresh is None:
            def reconcile(begin, end, text, **options):
                nonlocal budget
                if budget <= 0:
                    return None
                budget -= 1
                return align(begin, end, text, **options)
            try:
                _, selected, repairs = stitch_blocks(lines, [proposed, *contexts], duration, language, reconcile)
                report['boundary_reconciliations'] = repairs
                for selected_block in selected:
                    for i in range(*selected_block['range']):
                        segment = deepcopy(selected_block['result']['segments'][i-selected_block['context'][0]])
                        if not segment['words']:
                            continue
                        offset = selected_block['source_range'][0]-block['source_range'][0]
                        for word in segment['words']:
                            for key in ('start', 'end'):
                                word[key] = round(word[key]+offset, 3)
                        segment.update(start=segment['words'][0]['start'], end=segment['words'][-1]['end'])
                        if segment['words'] != proposed['result']['segments'][i-first]['words']:
                            proposed['result']['segments'][i-first] = segment
                            replaced.append(i)
                fresh = ordered(proposed)
            except ValueError as error:
                report['boundary_error'] = str(error)
        if fresh is None:
            # Preserve recoveries elsewhere when a boundary cannot be repaired.
            # Conflicting candidates stay in diagnostics, not in playable cues.
            rejected = {}
            while fresh is None:
                candidates = [(i, c) for i, c in enumerate(partial_alignment_cues(lines, [proposed], duration, language))
                              if c['status'] == 'candidate']
                conflicts = {i for i, c in candidates if any(a['start'] >= b['start']
                             for a, b in zip(c['units'], c['units'][1:]))}
                for (i, a), (j, b) in zip(candidates, candidates[1:]):
                    try:
                        presentation_cues([a, b], duration, lead=0, tail=0, bridge_gap=0)
                        if a['units'][-1]['start'] >= b['units'][0]['start']:
                            raise ValueError('word_order_conflict')
                    except ValueError:
                        conflicts.update({i, j}-set(replaced) or {i, j})
                if not conflicts:
                    break
                for i in conflicts:
                    segment = proposed['result']['segments'][i-first]
                    rejected[i+1] = deepcopy(segment)
                    segment['words'] = []
                replaced = [i for i in replaced if i not in conflicts]
                bad.update(conflicts)
                fresh = ordered(proposed)
            report.update(reason='partial_recovery_with_unresolved_conflicts', unresolved_segments=rejected)
            if fresh is None:
                continue
        report.update(accepted_lines=sorted({i+1 for i in replaced}),
                      original_segments={i+1: result['result']['segments'][i-first] for i in replaced})
        result, cues = proposed, fresh
        bad.difference_update(replaced)
    if bad and not global_recovery and budget > 0:
        # Apparently valid neighbours can squeeze missing lyrics into a window
        # that never contained them. Broaden once, sharing the remaining budget.
        result = recover_with_local_asr(result, lines, audio, language, recognize, align,
                                       force_global=True, max_alignment_attempts=budget)
        reports.extend(result.get('asr_localizations', []))
    result['asr_localizations'] = reports
    return result


def recovery_anchors(recognized, previous, invalid):
    """An omitted refrain makes its text match ambiguous, not its old time wrong."""
    anchors = deepcopy(recognized)
    repeats = {}
    for i, anchor in enumerate(anchors):
        repeats.setdefault(normalized(anchor['text']).casefold(), []).append(i)
    for indexes in repeats.values():
        if 0 < sum(anchors[i]['status'] == 'candidate' for i in indexes) < len(indexes):
            for i in indexes:
                anchors[i] = dict(anchors[i], status='ambiguous_repeat', start=None, end=None)
                anchors[i].pop('search_start', None)
    # Only independent recognized neighbours bound this search. Previously
    # aligned lines must not bootstrap one another into trusted evidence.
    recognized = deepcopy(anchors)
    for i, anchor in enumerate(anchors):
        cue = previous[i]
        if anchor['status'] == 'candidate' or i in invalid or cue['status'] != 'candidate':
            continue
        before = [a for a in recognized[:i] if a['status'] == 'candidate']
        after = [a for a in recognized[i+1:] if a['status'] == 'candidate']
        if before and after and before[-1]['end'] <= cue['start'] < cue['end'] <= after[0]['start']:
            anchors[i] = dict(anchor, start=cue['start'], end=cue['end'], status='candidate',
                              source='preserved_acoustic_search_bound')
    return anchors


def recover_skipped_repeats(result, duration, align):
    """Check a skipped repetition using both its gap and neighbouring phrases.

    Attention can skip an interior refrain and leave the last copy untimed.
    A gap is only a search window: accepted times come from acoustic alignment.
    """
    result = deepcopy(result)
    segments, reports = result['segments'], []
    first = 0
    while first < len(segments):
        stop = first+1
        while stop < len(segments) and normalized(segments[stop]['text']) == normalized(segments[first]['text']):
            stop += 1
        run = segments[first:stop]
        first = stop
        # Keep this bounded to one missing copy in an otherwise timed run.
        if len(run) < 3 or run[-1]['start'] != run[-1]['end'] or any(
                s['start'] >= s['end'] for s in run[:-1]):
            continue
        index = max(range(len(run)-2), key=lambda i: run[i+1]['start']-run[i]['end'])
        left, right = run[index:index+2]
        if left['end'] >= right['start'] or right['end']-left['start']+1.2 > 45:
            continue
        report = dict(lines=[stop-len(run)+1, stop], gap=[left['end'], right['start']], accepted=False)
        reports.append(report)
        predictions = []
        for name, begin, end, count in (
                ('context', max(0., left['start']-.6), min(duration, right['end']+.6), 3),
                ('gap', max(0., left['end']-.3), min(duration, right['start']+.3), 1)):
            prediction = align(begin, end, '\n'.join([left['text'].strip()]*count))
            report[name] = dict(window=[begin, end], result=prediction)
            if prediction is None or len(prediction['segments']) != count or not prediction['segments'][count//2]['words']:
                break
            segment = deepcopy(prediction['segments'][count//2])
            for word in segment['words']:
                word['start'], word['end'] = round(word['start']+begin, 3), round(word['end']+begin, 3)
            segment['start'], segment['end'] = segment['words'][0]['start'], segment['words'][-1]['end']
            predictions.append(segment)
        if len(predictions) != 2:
            continue
        candidate, isolated = predictions
        words, evidence = candidate['words'], isolated['words']
        if (not words or len(words) != len(evidence) or
                normalized(''.join(w['word'] for w in words)) != normalized(left['text']) or
                [normalized(w['word']) for w in words] != [normalized(w['word']) for w in evidence] or
                any(not left['end'] < w['start'] < w['end'] < right['start'] for w in words) or
                any(b['start']-a['start'] < .015 for a, b in zip(words, words[1:]))):
            continue
        begin, end = report['gap']['window']
        agreement = [begin+.015 < b['start'] < b['end'] < end-.015 and
                     abs(a['start']-b['start']) <= .2 for a, b in zip(words, evidence)]
        # A boundary-clipped word cannot corroborate its own onset. Require
        # adjacent interior words and at least half of the phrase to agree.
        if sum(agreement) < max(2, (len(words)+1)//2) or not any(
                a and b for a, b in zip(agreement, agreement[1:])):
            continue
        report.update(accepted=True, corroborated_words=sum(agreement), original_segments=run)
        run = run[:-1]
        run.insert(index+1, candidate)
        segments[stop-len(run):stop] = run
    return result, reports


def alignment_blocks(anchors, audio):
    """Keep ASR-supported verses apart; unresolved runs share adjacent context."""
    count = len(anchors)
    valid = lambda i: anchors[i]['status'] == 'candidate'
    first = 0
    blocks = []
    while first < count:
        stop = first+1
        if not valid(first):
            while stop < count and not valid(stop):
                stop += 1
        else:
            while stop < count and valid(stop):
                if anchors[stop]['end']-anchors[first]['start'] > 18 or anchors[stop]['start']-anchors[stop-1]['end'] > 4:
                    break
                stop += 1
        context_first = first-1 if first and valid(first-1) else first
        context_stop = stop+1 if stop < count and valid(stop) else stop
        if valid(first) and context_first < first and anchors[first]['start']-anchors[context_first]['end'] > 4:
            context_first = first
        if valid(stop-1) and context_stop > stop and anchors[stop]['start']-anchors[stop-1]['end'] > 4:
            context_stop = stop
        begin = max(0., anchors[context_first]['start']-.6) if valid(context_first) else 0.
        hint = anchors[first].get('search_start')
        if (not valid(first) and context_first < first and stop < count and valid(stop)
                and hint is not None and anchors[context_first]['end']+4 < hint < anchors[stop]['start']):
            # Do not drag a partially recognized phrase back across a long
            # instrumental gap just to retain the previous line as context.
            context_first, begin = first, max(0., hint-.6)
        # ASR often stops before a held final syllable; retain its acoustic tail.
        tail_padding = 3.0 if context_stop == count else .6
        end = min(len(audio)/16000, anchors[context_stop-1]['end']+tail_padding) if valid(context_stop-1) else len(audio)/16000
        # A global loudness threshold can miss quiet opening/closing lyrics.
        # Keep the anchor context; never trim it to estimated voice bounds.
        if not np.isfinite([begin, end]).all() or round(begin*16000) >= round(end*16000):
            raise ValueError(f'Empty alignment audio for lyric lines {first+1}-{stop}: {begin}-{end}s')
        blocks.append((first, stop, context_first, context_stop, begin, end))
        first = stop
    return blocks


def stitch_blocks(lines, raw_blocks, duration, language, align=None):
    """Use existing predictions, then bounded joint inference; never move times."""
    if any(not any(c.isalnum() for c in line) for line in lines):
        raise ValueError('Lyric lines must contain alignable text')
    raw_blocks = list(raw_blocks)

    def prediction_for(block):
        first, stop = block['context']
        offset = block['source_range'][0]
        segments = block['result']['segments']
        units = [dict(text=w['word'], start=round(offset+w['start'], 3),
                      end=round(offset+w['end'], 3)) for s in segments for w in s['words']]
        cues = lyric_cues(lines[first:stop], units, duration)
        prediction = {}
        for index, (cue, segment) in enumerate(zip(cues, segments), first):
            try:
                words = map_words(cue, segment['words'], offset, language) if cue['status'] == 'candidate' else []
            except ValueError:
                words = []  # Retain the failure for the normal word-mapping gate.
            prediction[index] = (cue, words)
        return prediction

    predictions, owners = [], [None]*len(lines)
    for block_index, block in enumerate(raw_blocks):
        predictions.append(prediction_for(block))
        for index in range(*block['range']):
            if owners[index] is not None:
                raise ValueError(f'Duplicate core lyric line: {index}')
            owners[index] = block_index
    if None in owners:
        raise ValueError('Raw alignment does not cover exactly the displayed lyrics')

    def conflicts(selection):
        selected = [predictions[owner][i] for i, owner in enumerate(selection)]
        bad = set()
        for i, ((left, left_words), (right, right_words)) in enumerate(zip(selected, selected[1:]), 1):
            if left_words and right_words and right_words[0]['start']-left_words[-1]['start'] < .015-1e-9:
                bad.add(i)
            try:
                # Share the downstream gate: repairing word starts must not
                # introduce invalid line ends or excessive overlap elsewhere.
                presentation_cues([left, right], duration, lead=0, tail=0, bridge_gap=0)
            except ValueError:
                bad.add(i)
        return bad

    repairs, attempted = [], set()
    while bad := conflicts(owners):
        repaired = False
        for index in sorted(bad):
            # Prefer a joint prediction. If that introduces another crossing,
            # an adjoining window may already have a valid single-line result.
            for first, stop in ((index-1, index+1), (index-1, index), (index, index+1)):
                for owner in dict.fromkeys(owners[index-1:index+1]+list(range(len(predictions)))):
                    prediction = predictions[owner]
                    if not all(i in prediction and prediction[i][1] for i in range(first, stop)):
                        continue
                    words = [w for i in range(first, stop) for w in prediction[i][1]]
                    if any(b['start']-a['start'] < .015-1e-9 for a, b in zip(words, words[1:])):
                        continue
                    candidate = owners.copy()
                    candidate[first:stop] = [owner]*(stop-first)
                    if not conflicts(candidate) < bad:
                        continue
                    repairs.append(dict(lines=[first+1, stop],
                        previous_blocks=owners[first:stop], selected_block=owner,
                        context=raw_blocks[owner]['context'], source_range=raw_blocks[owner]['source_range']))
                    owners, repaired = candidate, True
                    break
                if repaired:
                    break
            if repaired:
                break
        if not repaired and align is not None:
            for index in sorted(bad-attempted):
                attempted.add(index)
                # Neighbouring phrases establish the boundary; only the two
                # conflicting core lines may change. Replacing context lines
                # can merely move the crossing to the edge of the new crop.
                first, stop = max(0, index-3), min(len(lines), index+3)
                context = [predictions[owners[i]][i][0] for i in range(first, stop)]
                if any(c['status'] != 'candidate' for c in context):
                    continue
                begin = max(0., min(c['start'] for c in context)-.6)
                end = min(duration, max(c['end'] for c in context)+.6)
                if not 0 < end-begin <= 45:
                    continue
                try:
                    result = align_candidate(align, begin, end, '\n'.join(lines[first:stop]),
                        gap_padding=None, dynamic_heads=True, max_word_dur=None, word_dur_factor=None)
                except ValueError as error:
                    repairs.append(dict(method='joint_context_retry', accepted=False,
                        range=[index-1, index+1], context=[first, stop],
                        source_range=[begin, end], error=str(error)))
                    continue
                block = dict(range=[index-1, index+1], context=[first, stop],
                             source_range=[begin, end], result=result)
                report = dict(method='joint_context_retry', accepted=False, **block)
                repairs.append(report)
                if result is None or [normalized(''.join(w['word'] for w in s['words']))
                        for s in result['segments']] != [normalized(line) for line in lines[first:stop]]:
                    continue
                if any(not np.isfinite([w['start'], w['end']]).all() or
                       not 0 <= w['start'] <= w['end'] <= end-begin+.001
                       for s in result['segments'] for w in s['words']):
                    continue
                prediction = prediction_for(block)
                if any(not prediction[i][1] for i in range(index-1, index+1)):
                    continue
                words = [w for i in range(index-1, index+1) for w in prediction[i][1]]
                if any(b['start']-a['start'] < .015-1e-9 for a, b in zip(words, words[1:])):
                    continue
                owner = len(predictions)
                predictions.append(prediction)
                candidate = owners.copy()
                candidate[index-1:index+1] = [owner, owner]
                if not conflicts(candidate) < bad:
                    predictions.pop()
                    continue
                report['accepted'] = True
                raw_blocks.append(block)
                owners, repaired = candidate, True
                break
        if not repaired:
            break  # Never invent times or relax the downstream quality gate.

    # Preserve the selected raw prediction as well as the line timing: word and
    # phoneme stages must not accidentally read the old window for this line.
    stitched = []
    for index, owner in enumerate(owners):
        if index and owners[index-1] == owner:
            stitched[-1]['range'][1] = index+1
        else:
            stitched.append({**raw_blocks[owner], 'range': [index, index+1]})
    return [predictions[owner][i][0] for i, owner in enumerate(owners)], stitched, repairs


def refine_chinese_onsets(cues, raw_blocks, audio, language):
    """Remove a preceding held note absorbed by a single initial syllable.

    Only a long, single-character Chinese word with an internal quiet gap is
    eligible. Multi-syllable Japanese/English tokens need a phonetic model.
    """
    if language not in ('zh', 'Chinese'):
        return []
    frames = audio[:len(audio)//320*320].reshape(-1, 320)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = max(.008, np.quantile(rms, .9)*.12)
    active = binary_closing(rms > threshold, structure=np.ones(5))
    changes = []
    for block in raw_blocks:
        offset = block['source_range'][0]
        context = block['context'][0]
        segments = block['result']['segments']
        assert len(segments) == block['context'][1]-context
        for i in range(*block['range']):
            segment = segments[i-context]
            assert normalized(segment['text']) == normalized(cues[i]['text'])
            word = next(w for w in segment['words'] if w['word'].strip())
            char = word['word'].strip()
            if len(char) != 1 or not '\u4e00' <= char <= '\u9fff' or word['end']-word['start'] < 1.15:
                continue
            begin = round((offset+word['start'])/.02)
            end = round((offset+word['end'])/.02)
            edges = np.diff(np.r_[False, ~active[begin:end], False].astype(int))
            gaps = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if b-a >= 8 and b < end-begin-4]
            if not gaps:
                continue
            a, b = gaps[-1]
            onset = round((begin+b)*.02-.04, 3)
            if onset-cues[i]['start'] <= .3:
                continue
            changes.append({'line': i+1, 'forced_start': cues[i]['start'], 'refined_start': onset,
                            'quiet_gap': [(begin+a)*.02, (begin+b)*.02], 'rms_threshold': float(threshold)})
            cues[i]['forced_start'] = cues[i]['start']
            cues[i]['start'] = onset
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--model', default='large-v3-turbo')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--dynamic-heads', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Whisper's CPU DTW promotes to float64; transfer before promotion on MPS.
    if args.device == 'mps':
        whisper_timing.dtw = lambda matrix: dtw_cpu(matrix.detach().cpu().double().numpy())
    jobs = json.loads(args.jobs.read_text())
    model_root = Path(__file__).resolve().parent/'models/alignment/whisper'
    loading_start = time.perf_counter()
    report_progress(detail='加载模型')
    model = stable_whisper.load_model(args.model, device='cpu', download_root=str(model_root))
    # The sparse head-selection buffer is indexed on the host; MPS cannot move it.
    heads = model._buffers.pop('alignment_heads')
    model.to(args.device)
    model.register_buffer('alignment_heads', heads, persistent=False)
    loading = time.perf_counter()-loading_start
    print(json.dumps({'model_loaded_seconds': loading, 'model': args.model}), flush=True)
    for job in jobs:
        output = Path(job['output'])
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        lyrics = json.loads(Path(job['lyrics']).read_text())
        audio_info = sf.info(job['audio'])
        duration = audio_info.frames/audio_info.samplerate
        assert audio_info.samplerate == 16000 and audio_info.channels == 1
        audio, _ = sf.read(job['audio'], dtype='float32')
        language = {'Japanese': 'ja', 'Chinese': 'zh', 'English': 'en'}[lyrics['language']]
        blocks = [(0, len(lyrics['lines']), 0, len(lyrics['lines']), 0., duration)]
        anchors = None
        if job.get('anchors'):
            anchors = json.loads(Path(job['anchors']).read_text())
        if anchors:
            assert [cue['text'] for cue in anchors] == lyrics['lines']
            blocks = alignment_blocks(anchors, audio)
        started = time.perf_counter()
        raw_blocks, units = [], []
        for first, stop, context_first, context_stop, begin, end in progress_items(blocks, 'blocks'):
            print(json.dumps({'song': Path(job['lyrics']).stem, 'lines': [first+1, stop],
                              'window': [begin, end]}), flush=True)
            piece = audio[round(begin*16000):round(end*16000)]

            def align_audio(start, stop, text, **options):
                aligned = model.align(audio[round(start*16000):round(stop*16000)], text, language=language,
                    original_split=True, regroup=False, token_step=80, max_word_dur=options.pop('max_word_dur', 4.),
                    word_dur_factor=options.pop('word_dur_factor', 3.), nonspeech_skip=5., suppress_silence=True,
                    dynamic_heads=options.pop('dynamic_heads', args.dynamic_heads), verbose=None, **options)
                return aligned.to_dict() if aligned is not None else None

            def align_piece(start, stop, text, **options):
                return align_audio(begin+start, begin+stop, text, **options)

            def recognize_audio(start, stop):
                from stage_cache import StageCache
                import anchors_from_asr
                config = dict(start=start, stop=stop, language=language, model=args.model,
                              model_sha256=sha(model_root/f'{args.model}.pt'),
                              temperature=0, condition_on_previous_text=False)
                key = 'alignment_localization_'+hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
                path = output.parent/'asr-requests'/(key+'.json')
                path.parent.mkdir(exist_ok=True)
                with StageCache(output.parent).stage(key, inputs=[job['audio']], outputs=[path], config=config,
                        code=[Path(__file__), Path(anchors_from_asr.__file__)]) as reused:
                    if not reused:
                        report_progress(detail='定位未对齐的唱段')
                        recognized = model.transcribe(audio[round(start*16000):round(stop*16000)],
                            language=language, temperature=0, condition_on_previous_text=False,
                            word_timestamps=True, regroup=False, verbose=None,
                            fp16=args.device == 'cuda')
                        path.write_text(json.dumps(recognized.to_dict(), ensure_ascii=False, default=json_number)+'\n')
                return dict(json.loads(path.read_text()), localization_cache=dict(file=path.name, reused=reused))

            lines = lyrics['lines'][context_first:context_stop]
            result = align_piece(0., len(piece)/16000, '\n'.join(lines))
            if result is None:
                raise ValueError(f'Alignment returned no result for {job["lyrics"]}')
            result, text_recovery = recover_changed_text(result, lines, len(piece)/16000,
                lambda: align_piece(0., len(piece)/16000, '\n'.join(lines), gap_padding=None, dynamic_heads=True))
            result, repeat_recoveries = recover_skipped_repeats(result, len(piece)/16000, align_piece)
            block = dict(range=[first, stop], context=[context_first, context_stop],
                source_range=[begin, end], result=result, repeat_recoveries=repeat_recoveries,
                text_recovery=text_recovery)
            block = (recover_unresolved_block(block, lyrics['lines'], anchors, duration, align_audio)
                     if anchors else recover_unresolved_lines(block, lyrics['lines'], duration,
                                                              lyrics['language'], align_audio))
            if not anchors:
                (output/'before-localization.json').write_text(json.dumps(block, ensure_ascii=False, default=json_number)+'\n')
                block = recover_with_local_asr(block, lyrics['lines'], audio, lyrics['language'],
                                               recognize_audio, align_audio)
            result, begin = block['result'], block['source_range'][0]
            local_units = [{'text': word['word'], 'start': round(word['start']+begin, 3),
                            'end': round(word['end']+begin, 3), 'probability': word['probability']}
                           for segment in result['segments'] for word in segment['words']]
            units.extend(local_units)
            raw_blocks.append(block)
        # Keep the acoustic evidence even when stitching rejects text or times.
        (output/'raw.json').write_text(json.dumps(raw_blocks, ensure_ascii=False, indent=2, default=json_number)+'\n')
        (output/'units.json').write_text(json.dumps(units, ensure_ascii=False, indent=2, default=json_number)+'\n')
        cues, raw_blocks, boundary_repairs = stitch_blocks(
            lyrics['lines'], raw_blocks, duration, lyrics['language'], align_audio)
        if args.device == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter()-started
        refinements = refine_chinese_onsets(cues, raw_blocks, audio, language)
        for cue in cues:
            cue['source'] = 'stable-ts forced alignment of supplied lyrics to vocal audio'
        summary = write_captions(cues, output)
        summary['zero_duration_units'] = sum(unit['end'] <= unit['start'] for unit in units)
        summary['unit_count'] = len(units)
        (output/'raw.json').write_text(json.dumps(raw_blocks, ensure_ascii=False, indent=2, default=json_number)+'\n')
        report = {'model': args.model, 'device': args.device, 'dtype': 'float32',
            'model_loading_seconds': loading, 'alignment_seconds': elapsed,
            'audio': str(Path(job['audio']).resolve()), 'audio_sha256': sha(job['audio']),
            'lyrics': str(Path(job['lyrics']).resolve()), 'lyrics_sha256': sha(job['lyrics']),
            'model_sha256': sha(model_root/f'{args.model}.pt'), 'runner_sha256': sha(__file__),
            'dynamic_heads': args.dynamic_heads,
            'boundary_reconciliations': boundary_repairs,
            'line_recoveries': [r for block in raw_blocks for r in block.get('line_recoveries', [])],
            'asr_localizations': [r for block in raw_blocks for r in block.get('asr_localizations', [])],
            'onset_refinements': refinements,
            'anchors': job.get('anchors'), 'anchors_sha256': sha(job['anchors']) if job.get('anchors') else None,
            'packages': {name: version(name) for name in ['stable-ts', 'openai-whisper', 'torch']},
            'summary': summary, 'status': 'Audio-only candidate; requires timing review'}
        (output/'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_number)+'\n')
        print(json.dumps({'lyrics': job['lyrics'], 'seconds': elapsed, **summary}, default=json_number), flush=True)


if __name__ == '__main__':
    main()
