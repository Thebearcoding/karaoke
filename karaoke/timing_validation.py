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


def reconcile_timing(cues, wave, witness, predict, protocol, language, constraints=(), *,
                     playback=False, _retry_context=True, _review_lines=None):
    """Corroborate displaced onsets, then replace only a compatible local span.

    witness returns original/vocal word views. predict returns three independent
    context predictions without old word priors. Both are cacheable adapters.
    """
    from cjk_recovery import corroborated_group
    result, reports, ledger = deepcopy(cues), [], deepcopy(list(constraints))
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
            if cue['aligned_end']-cue['aligned_start'] > 30:
                raise ValueError('Acoustic review context exceeds 30 seconds')
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
            suspects = [a for a,b in zip(*views) if a['first'] in old
                        and abs(a['start']-b['start']) <= .1
                        and abs(old[a['first']]['start']-a['start']) > protocol['maximum_change_seconds']]
            if not suspects:
                report['status'] = 'no_corroborated_displacement'
                continue
            predictions = predict(cue, views, bounds, review_language)
            report['predictions'] = predictions
            confirmed = {}
            for group in views[0]:
                choose = lambda vs: [[u for u in v if (u['first'],u['stop'])==(group['first'],group['stop'])] for v in vs]
                replacement, reason = corroborated_group(group, choose(predictions), choose(views), wave,
                    bounds, protocol, spans=[(group['first'],group['stop'])])
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
                if proof not in ledger:ledger.append(proof)
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
                if not members:reason='incomplete local replacement'
                else:
                    other={u['first']:u for u in views[1]}
                    if any(abs(u['start']-other[u['first']]['start'])>.1 for u in members):
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
        windows = {}
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
        for row in reports:
            index = row['line']-1
            proofs = [v['proof'] for v in timing_violations(result, ledger) if v['proof']['line']==index+1]
            if not proofs or 'predictions' not in row:
                continue
            bounds = [result[index-1]['units'][-1]['start']+.015 if index else 0.,
                      result[index+1]['units'][0]['start'] if index+1<len(result) else len(wave)/44100]
            for proof in proofs:
                candidate, repair = grouped_timing_fallback(result[index], row, proof, ledger,
                    wave, bounds, protocol, language)
                if candidate is not None:
                    result[index] = candidate
                    row['repairs'].append(repair)
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
    def corroborate(group):
        select = lambda vs: [[u for u in v if group['first'] <= u['first'] < u['stop'] <= group['stop']] for v in vs]
        selected = select(views)
        if not selected[0] or normalized(''.join(u['text'] for u in selected[0])) != normalized(group['text']):
            return None
        found, _ = corroborated_group(group, select(report['predictions']), selected,
            wave, bounds, protocol, spans=[(u['first'],u['stop']) for u in selected[0]])
        return found

    word = next((u for u in views[0] if u['first']==first), None)
    measured = corroborate(word) if word else None
    if not measured:
        return None, None
    anchor, lo = measured[0], positions[first]
    guard = None
    for hi in range(lo+1, len(units)):
        guard = corroborate(units[hi])
        if guard:
            break
    if guard is not None:
        stop, end = guard[0]['first'], guard[0]['start']
        limit = units[hi+1]['start'] if hi+1<len(units) else bounds[1]
        if any(u['start']+.015 >= limit or
               u.get('timing_evidence')=='ctc_event' and u['emission_end']>limit for u in guard):
            return None, None
        guard = [dict(u, end=min(u['end'],limit), acoustic_end=acoustic_end(u),
                      timing_source='cross_model_ctc' if u.get('timing_evidence')=='ctc_event' else 'cross_model_phonemes')
                 for u in guard]
        suffix, source = guard+units[hi+1:], 'cross_model_onsets'
    else:
        # The last word has no following onset. Compare actual alignment
        # ends, never a padded display hold, at the output's millisecond
        # resolution. The conservative envelope is a text-span boundary;
        # it must not become a claimed singing offset for countdowns.
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
        if (not np.isfinite(ends).all() or any(not anchor['start']<e<=bounds[1] for e in ends)
                or round(max(ends[:3])-min(ends[:3]),3)>protocol['maximum_spread_seconds']
                or round(abs(ends[3]-ends[4]),3)>.1
                or any(round(abs(e-center),3)>.1 for e in ends[3:])):
            return None, None
        hi, stop, end = len(units), len(cue['text']), max(ends)
        suffix, source = [], 'cross_model_span_end'
    if hi <= lo:
        return None, None
    # Never hide another established onset inside a coarse group, or span
    # words that the acoustic witnesses place outside these two boundaries.
    covered = [p for p in constraints if p['line']==proof['line'] and first<=p['first']<=stop]
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

    def predict(self, wave):
        def run(cue, views, bounds, language):
            from phoneme_progress import spoken_units
            from subtitles import highlight_spans
            start=min(v[0]['start'] for v in views)
            end=max(v[-1]['end'] for v in views)
            windows=[(max(bounds[0],start-pad),min(bounds[1],end+pad))
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
                            dict(cue,aligned_start=start,aligned_end=end),decoder='ctc')['units'])
                return result
            return self.cached('prediction',[language,cue['text'],windows],compute)
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
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],required=True)
    parser.add_argument('--playback-fallback',action='store_true')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    def write(p,value):
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    cues,lyrics=read(args.input),read(args.lyrics)
    wave,rate=sf.read(args.audio,dtype='float32',always_2d=True)
    duration=len(wave)/rate
    validate_timeline(cues,lyrics['lines'],duration,language=lyrics['language'])
    divisor=np.gcd(rate,44100)
    wave=resample_poly(wave.mean(axis=1),44100//divisor,rate//divisor).astype('float32')
    code_root=Path(__file__).resolve().parent
    protocol=read(code_root/'experiments/cjk-onsets-v003/protocol.json')
    source=dict(original=sha(args.original),vocals=sha(args.audio),lyrics=lyrics['lines'])
    ledger_file=out.parent/'timing-constraints.json'
    ledger=read(ledger_file) if ledger_file.is_file() else {}
    constraints=ledger.get('constraints',[]) if ledger.get('source')==source else []
    paths=[code_root/name for name in ('timing_validation.py','cjk_recovery.py','align_lyric_phonemes.py',
        'phoneme_progress.py','align_english_sofa.py','align_english_onsets.py','subtitles.py')]
    models=acoustic_model_files(ROOT)
    identity=dict(source=source,protocol=protocol,code={str(p):sha(p) for p in paths},
                  models={str(p):sha(p) for p in models if p.is_file()})
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    review=AcousticReview(args.original,args.device,out.parent/'timing-cache',identity,protocol)
    started=time.perf_counter()
    repaired,report=reconcile_timing(cues,wave,review.observe,review.predict(wave),protocol,
                                     lyrics['language'],constraints,playback=args.playback_fallback)
    report.update(seconds=time.perf_counter()-started,source=source,input_sha256=sha(args.input))
    write(ledger_file,dict(source=source,constraints=report['constraints']))
    write(out/'cues.json',repaired);write(out/'report.json',report)
    validate_timeline(repaired,lyrics['lines'],duration,report['constraints'],lyrics['language'])
    print(json.dumps(dict(lines=len(cues),reviewed_lines=len({r['line'] for r in report['lines']}),
                         review_attempts=len(report['lines']),
                         constraints=len(report['constraints']),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    main()
