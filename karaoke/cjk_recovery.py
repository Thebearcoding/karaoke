"""Recover merged Chinese highlights with independent acoustic corroboration."""
from copy import deepcopy
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from align_lyric_phonemes import protect_line_order
from phoneme_progress import ROOT, chinese_pronunciation_text, spoken_units
from subtitles import highlight_spans


class AcousticWitness:
    """A different aligner on both the mixture and the separated voice."""
    def __init__(self, original_audio, device):
        self.original_audio, self.device = original_audio, device
        self.bundle = self.model = None

    def __call__(self, cue, wave):
        import torch
        import torchaudio
        from pypinyin import lazy_pinyin
        weights = ROOT/'models/alignment/mms/model.pt'
        if not weights.is_file():
            raise ValueError('Independent MMS alignment weights are unavailable')
        if self.model is None:
            self.bundle = torchaudio.pipelines.MMS_FA
            self.model = self.bundle.get_model(dl_kwargs={'model_dir': str(weights.parent),
                                                          'progress': False}).to(self.device).eval()
        spans = list(highlight_spans(cue['text'], 'Chinese'))
        syllables = lazy_pinyin(chinese_pronunciation_text(cue['text']), errors=lambda value: list(value))
        if len(syllables) != len(cue['text']):
            raise ValueError('Independent pronunciation changes character positions')
        tokens = [re.sub("[^a-z']", '', ''.join(syllables[a:b]).lower()) for a, b in spans]
        if not all(tokens):
            raise ValueError('Unsupported independent pronunciation')
        begin, end = max(0., cue['aligned_start']-.6), min(len(wave)/44100, cue['aligned_end']+.6)
        original = sf.info(self.original_audio)
        mixture, rate = sf.read(self.original_audio, start=round(begin*original.samplerate),
                                stop=round(end*original.samplerate), dtype='float32', always_2d=True)
        views = [(wave[round(begin*44100):round(end*44100)], 44100), (mixture.mean(axis=1), rate)]
        result = []
        for audio, rate in views:
            divisor = np.gcd(rate, 16000)
            audio = resample_poly(audio, 16000//divisor, rate//divisor).astype('float32')
            with torch.inference_mode():
                emissions, _ = self.model(torch.from_numpy(audio).unsqueeze(0).to(self.device))
            emissions = emissions.cpu()
            aligned = self.bundle.get_aligner()(emissions[0], self.bundle.get_tokenizer()(['*', *tokens, '*']))[1:-1]
            if len(aligned) != len(spans) or any(not times for times in aligned):
                raise ValueError('Independent aligner omitted characters')
            step = len(audio)/16000/emissions.shape[1]
            result.append([dict(first=a, stop=b, text=cue['text'][a:b],
                                start=begin+times[0].start*step, end=begin+times[-1].end*step)
                           for (a, b), times in zip(spans, aligned)])
        return result


def corroborated_group(group, predictions, witnesses, wave, bounds, protocol):
    """Return measured boundaries only when all views independently agree."""
    expected = [(group['first']+a, group['first']+b) for a, b in highlight_spans(group['text'], 'Chinese')]
    signature = lambda units: [(u['first'], u['stop']) for u in units]
    if (len(predictions) != 3 or len(witnesses) != 2 or not expected
            or any(signature(v) != expected for v in [*predictions, *witnesses])
            or any(u['text'] != group['text'][u['first']-group['first']:u['stop']-group['first']]
                   for v in [*predictions, *witnesses] for u in v)):
        return None, 'missing or mismatched character evidence'
    floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    result = []
    for index in range(len(expected)):
        primary = [v[index] for v in predictions]
        secondary = [v[index] for v in witnesses]
        events = [u.get('timing_evidence') == 'ctc_event' for u in primary]
        if any(events) and not all(events):
            return None, 'inconsistent onset evidence types'
        if all(events) and any(not np.isfinite(u.get('emission_end', np.nan))
                               or not u['start'] < u['emission_end'] <= u['end'] for u in primary):
            return None, 'invalid CTC onset event'
        if any(not np.isfinite([u['start'], u['end']]).all() or u['end'] <= u['start']
               for u in [*primary, *secondary]):
            return None, 'invalid acoustic interval'
        if any(not bounds[0] <= u['start'] < bounds[1] or u['end'] > len(wave)/44100
               for u in [*primary, *secondary]):
            return None, 'acoustic evidence crosses adjacent words'
        if any(u['end']-u['start'] < .035 for u in primary):
            return None, 'insufficient time between onset events' if all(events) else 'collapsed phonetic span'
        starts = [u['start'] for u in primary]
        start, end = float(np.median(starts)), min(bounds[1], float(np.median([u['end'] for u in primary])))
        if max(starts)-min(starts) > protocol['maximum_spread_seconds']:
            return None, 'context disagreement'
        if (any(abs(u['start']-start) > .1 for u in secondary)
                or abs(secondary[0]['start']-secondary[1]['start']) > .1):
            return None, 'independent acoustic disagreement'
        if not bounds[0] <= start < end <= bounds[1] or (result and start-result[-1]['start'] < .015):
            return None, 'conflict with adjacent words'
        if all(events) and (end-start < .035 or any(u['emission_end'] > bounds[1] for u in primary)):
            return None, 'CTC event lacks room before the following word'
        # CTC emissions can be one frame wide. Verify actual vocal support at
        # the onset; a longer display hold alone cannot supply that evidence.
        support_end = min(end, start+.035) if all(events) else end
        piece = wave[round(start*44100):round(support_end*44100)]
        if not len(piece) or not np.isfinite(piece).all() or float(np.sqrt(np.mean(piece**2))) <= floor:
            return None, 'insufficient audible vocal support'
        result.append(dict(primary[0], start=round(start, 3), end=round(end, 3),
                           timing_source='cross_model_phonemes'))
    for left, right in zip(result, result[1:]):
        left['end'] = min(left['end'], right['start'])
    return result, 'agreement across phonetic contexts, original and separated audio'


def neighboring_context(cues, line, group, radius=1):
    """Keep complete Chinese context while excluding adjacent foreign vocals."""
    text, anchors, offset = '', [], 0
    for index in range(max(0, line-radius), min(len(cues), line+radius+1)):
        if index == line:
            offset = len(text)
        anchors.extend(dict(u, first=u['first']+len(text), stop=u['stop']+len(text))
                       for u in cues[index]['units'])
        text += cues[index]['text']+' '
    foreign = [i for i, c in enumerate(text) if c.isalnum() and not '\u3400' <= c <= '\u9fff']
    first = max((i+1 for i in foreign if i < offset+group['first']), default=0)
    stop = min((i for i in foreign if i >= offset+group['stop']), default=len(text))
    left = max((u['end'] for u in anchors if u['stop'] <= first), default=0.)
    right = min((u['start'] for u in anchors if u['first'] >= stop), default=float('inf'))
    anchors = [dict(u, first=u['first']-first, stop=u['stop']-first)
               for u in anchors if first <= u['first'] and u['stop'] <= stop]
    if not anchors or anchors[-1]['end']-anchors[0]['start'] > 30:
        raise ValueError('No bounded Chinese lyric context')
    return (dict(text=text[first:stop], units=anchors, aligned_start=anchors[0]['start'],
                 aligned_end=anchors[-1]['end']), offset-first, left, right)


def recover_context_groups(cues, wave, model, witness, protocol):
    """Recheck neighbors too when a stale word boundary blocks a real onset."""
    result, reports = deepcopy(cues), []
    for line, cue in enumerate(cues):
        for group in cue['units']:
            if (len(list(highlight_spans(group['text'], 'Chinese'))) < 2 or
                    any(c.isalnum() and not '\u3400' <= c <= '\u9fff' for c in group['text'])):
                continue
            current = result[line]['units']
            index = next((i for i, u in enumerate(current)
                          if (u['first'], u['stop']) == (group['first'], group['stop'])), None)
            if index is None:  # An earlier joint repair already covered this group.
                continue
            report = dict(line=line+1, first=group['first'], stop=group['stop'],
                          phase='neighbor_context', attempts=[], accepted=False)
            reports.append(report)
            seen_contexts = set()
            for radius in (1, 0):
                try:
                    context, offset, left, right = neighboring_context(cues, line, group, radius)
                    signature = (context['text'].strip(), context['aligned_start'], context['aligned_end'])
                    if signature in seen_contexts:
                        continue
                    seen_contexts.add(signature)
                    parsed = spoken_units(context['text'], 'Chinese')
                    independent = witness(context, wave)
                    decoders = [('phonemes', (1., 1.5, 2.))]
                    if 'context_padding_seconds' in protocol:
                        decoders.append(('ctc', protocol['context_padding_seconds']))
                    # Prefer a complete corroborated run, so neighboring words
                    # do not retain stale boundaries around the repaired group.
                    ranges = sorted(((lo, hi) for lo in range(index+1)
                                     for hi in range(index+1, len(current)+1)), key=lambda r: r[1]-r[0], reverse=True)
                    for decoder, pads in decoders:
                        windows = [(max(left, context['aligned_start']-pad),
                                    min(right, len(wave)/44100, context['aligned_end']+pad)) for pad in pads]
                        if len(set(windows)) != 3 or any(b <= a for a, b in windows):
                            continue
                        options = {'decoder': 'ctc'} if decoder == 'ctc' else {}
                        predictions = [model.infer(wave[round(a*44100):round(b*44100)], a, parsed,
                                                   context, **options)['units'] for a, b in windows]
                        for lo, hi in ranges:
                            first, stop = current[lo]['first'], current[hi-1]['stop']
                            text = cue['text'][first:stop]
                            if any(c.isalnum() and not '\u3400' <= c <= '\u9fff' for c in text):
                                continue
                            previous = current[lo-1] if lo else (result[line-1]['units'][-1] if line else None)
                            following = current[hi] if hi < len(current) else (
                                result[line+1]['units'][0] if line+1 < len(result) else None)
                            bounds = [previous['end'] if previous else 0.,
                                      following['start'] if following else len(wave)/44100]

                            def select(views):
                                return [[dict(u, first=u['first']-offset, stop=u['stop']-offset) for u in view
                                         if first+offset <= u['first'] and u['stop'] <= stop+offset] for view in views]

                            primary, secondary = select(predictions), select(independent)
                            replacement, reason = corroborated_group(dict(first=first, stop=stop, text=text),
                                primary, secondary, wave, bounds, protocol)
                            report['attempts'].append(dict(decoder=decoder, context_radius=radius, first=first, stop=stop, reason=reason,
                                predictions=primary, witnesses=secondary, bounds=bounds, source_windows=windows))
                            if replacement is not None:
                                if decoder == 'ctc':
                                    for unit in replacement:
                                        unit['timing_source'] = 'cross_model_ctc'
                                current[lo:hi] = replacement
                                report.update(accepted=True, decoder=decoder)
                                break
                        if report['accepted']:
                            break
                except (ValueError, RuntimeError, OSError) as error:
                    report['reason'] = str(error)
                if report['accepted']:
                    break
    result, reverted = protect_line_order(result, cues)
    for report in reports:
        if report['line'] in reverted:
            report.update(accepted=False, reason='reverted to preserve lyric order')
    return result, reports


def recover_repeated_groups(cues, wave, witness, protocol):
    """Transfer verified onsets through four audio paths, then check the target."""
    from align_repeated_lyrics import agree_onset, audio_features, repeated_paths, audible
    from librosa.util.exceptions import ParameterError
    from lyric_progress import canonical_positions

    result, reports, features = deepcopy(cues), [], None
    reference_rms = float(np.sqrt(np.mean(wave**2)))

    def verified_reference(unit):
        return (unit is not None and unit.get('timing_source') in ('cross_model_phonemes', 'cross_model_ctc')
                and len(list(highlight_spans(unit['text'], 'Chinese'))) == 1)

    keys = [canonical_positions(c['text'], 'Chinese')[0] for c in cues]
    for line, cue in enumerate(cues):
        if len(keys[line]) < 5:
            continue
        observed = None
        for group in cue['units']:
            group_spans = [(group['first']+a, group['first']+b) for a, b in highlight_spans(group['text'], 'Chinese')]
            if len(group_spans) < 2:
                continue
            current = result[line]['units']
            index = next((i for i, u in enumerate(current)
                          if (u['first'], u['stop']) == (group['first'], group['stop'])), None)
            if index is None:
                continue
            for source_line, source in enumerate(cues):
                if source_line == line or keys[source_line] != keys[line]:
                    continue
                reference = {len(canonical_positions(source['text'][:u['first']], 'Chinese')[0]): u
                             for u in source['units']}
                report = dict(line=line+1, first=group['first'], stop=group['stop'],
                              reference_line=source_line+1, phase='verified_repeat', accepted=False, attempts=[])
                reports.append(report)
                try:
                    units = [reference.get(len(canonical_positions(cue['text'][:a], 'Chinese')[0])) for a, _ in group_spans]
                    if not all(verified_reference(u) for u in units):
                        raise ValueError('Repeated phrase lacks a verified reference')
                    if features is None:
                        original, rate = sf.read(witness.original_audio, dtype='float32', always_2d=True)
                        features = [audio_features(wave, 44100), audio_features(original.mean(axis=1), rate)]
                    views = [repeated_paths(data, source, cue) for data in features]
                    report['chroma_shifts'] = [shift for _, shift in views]
                    if len(set(report['chroma_shifts'])) != 1:
                        raise ValueError('Repeated key change differs between original and vocals')
                    paths = [path for view, _ in views for path in view]
                    report['path_costs'] = [path[2] for _, path in paths]
                    if any(path[2] > (.35 if name == 'pitch' else .85) for name, path in paths):
                        raise ValueError('Repeated audio is not sufficiently similar')
                    if observed is None:
                        observed = witness(cue, wave)
                    ranges = sorted(((lo, hi) for lo in range(index+1)
                                     for hi in range(index+1, len(current)+1)), key=lambda r: r[1]-r[0])
                    for lo, hi in ranges:
                        first, stop = current[lo]['first'], current[hi-1]['stop']
                        attempt = dict(first=first, stop=stop, estimates=[])
                        report['attempts'].append(attempt)
                        try:
                            spans = [(first+a, first+b) for a, b in highlight_spans(cue['text'][first:stop], 'Chinese')]
                            units = [reference.get(len(canonical_positions(cue['text'][:a], 'Chinese')[0])) for a, _ in spans]
                            if not all(verified_reference(u) for u in units):
                                raise ValueError('Repeated phrase lacks a verified reference')
                            previous = current[lo-1] if lo else (result[line-1]['units'][-1] if line else None)
                            following = current[hi] if hi < len(current) else (
                                result[line+1]['units'][0] if line+1 < len(cues) else None)
                            lower, upper = previous['end'] if previous else 0., following['start'] if following else len(wave)/44100
                            if hi < len(current):
                                boundary = [next((u for u in view if u['first'] == following['first']), None) for view in observed]
                                if (len(boundary) == 2 and all(u is not None and np.isfinite(u['start'])
                                        and abs(u['start']-upper) > .1 for u in boundary)
                                        and abs(boundary[0]['start']-boundary[1]['start']) <= .1):
                                    raise ValueError('Following boundary disagrees with target audio')
                            replacement = []
                            for (a, b), unit in zip(spans, units):
                                values = [float(np.interp(unit['start'], x, y)) for _, (x, y, _) in paths]
                                attempt['estimates'].append(values)
                                onset = agree_onset(values, protocol['maximum_spread_seconds'], cue['aligned_start'], cue['aligned_end'])
                                matches = [next((u for u in view if (u['first'], u['stop']) == (a, b)), None) for view in observed]
                                if (onset is None or not lower <= onset < upper or len(matches) != 2 or
                                        any(u is None or u['text'] != cue['text'][a:b] or
                                            not np.isfinite([u['start'], u['end']]).all() or u['end'] <= u['start'] or
                                            abs(u['start']-onset) > .1 for u in matches) or
                                        abs(matches[0]['start']-matches[1]['start']) > .1):
                                    raise ValueError('Repeated onset lacks independent target support')
                                replacement.append(dict(first=a, stop=b, text=cue['text'][a:b], start=onset,
                                    timing_source='repeated_lyric_audio', reference_line=source_line+1))
                            for i, unit in enumerate(replacement):
                                unit['end'] = replacement[i+1]['start'] if i+1 < len(replacement) else min(current[hi-1]['end'], upper)
                                if unit['end']-unit['start'] < .035-1e-9 or not audible(
                                        wave[round(unit['start']*44100):round((unit['start']+.035)*44100)], reference_rms):
                                    raise ValueError('Repeated onset has no audible interval')
                            current[lo:hi] = replacement
                            report.update(accepted=True, estimates=attempt['estimates'], replacement=replacement)
                            break
                        except ValueError as error:
                            attempt['reason'] = str(error)
                    if report['accepted']:
                        break
                except (ValueError, RuntimeError, OSError, ParameterError) as error:
                    report['reason'] = str(error)
    result, reverted = protect_line_order(result, cues)
    for report in reports:
        if report['line'] in reverted:
            report.update(accepted=False, reason='reverted to preserve lyric order')
    return result, reports


def recover_groups(cues, wave, model, witness, protocol):
    """Keep foreign words intact while retrying adjacent Chinese compounds."""
    result, reports = deepcopy(cues), []
    for line, cue in enumerate(cues):
        independent = None
        for index, group in enumerate(cue['units']):
            if (len(list(highlight_spans(group['text'], 'Chinese'))) < 2
                    or any(c.isalnum() and not '\u3400' <= c <= '\u9fff' for c in group['text'])
                    or group.get('timing_source') == 'repeated_lyric_audio'):
                continue
            report = dict(line=line+1, first=group['first'], stop=group['stop'], accepted=False)
            reports.append(report)
            try:
                foreign = [i for i, c in enumerate(cue['text']) if c.isalnum() and not '\u3400' <= c <= '\u9fff']
                first = max((i+1 for i in foreign if i < group['first']), default=0)
                stop = min((i for i in foreign if i >= group['stop']), default=len(cue['text']))
                parsed = [dict(u, first=u['first']+first, stop=u['stop']+first)
                          for u in spoken_units(cue['text'][first:stop], 'Chinese')]
                if cue['aligned_end']-cue['aligned_start'] > 30:
                    raise ValueError('Lyric recovery context exceeds 30 seconds')
                previous = cue['units'][index-1] if index else (cues[line-1]['units'][-1] if line else None)
                following = cue['units'][index+1] if index+1 < len(cue['units']) else (cues[line+1]['units'][0] if line+1 < len(cues) else None)
                bounds = [previous['end'] if previous else 0., following['start'] if following else len(wave)/44100]
                decoders = [('phonemes', (1., 1.5, 2.))]
                # CTC gives sparse onset events. Only an internal group has a
                # following measured onset to bound every display interval.
                if any(u['first'] >= group['stop'] for u in parsed):
                    decoders.append(('ctc', protocol['context_padding_seconds']))
                report['attempts'] = []
                for decoder, paddings in decoders:
                    predictions, windows = [], []
                    for padding in paddings:
                        begin, end = max(0., cue['aligned_start']-padding), min(len(wave)/44100, cue['aligned_end']+padding)
                        options = {'decoder': 'ctc'} if decoder == 'ctc' else {}
                        prediction = model.infer(wave[round(begin*44100):round(end*44100)], begin, parsed, cue, **options)
                        predictions.append([u for u in prediction['units'] if group['first'] <= u['first'] < group['stop']])
                        windows.append([begin, end])
                    if len({tuple(w) for w in windows}) != 3:
                        raise ValueError('Recovery contexts are not distinct')
                    if independent is None:
                        independent = witness(cue, wave)
                    observed = [[u for u in view if group['first'] <= u['first'] < group['stop']] for view in independent]
                    replacement, reason = corroborated_group(group, predictions, observed, wave, bounds, protocol)
                    attempt = dict(decoder=decoder, reason=reason, predictions=predictions,
                                   witnesses=observed, source_windows=windows, bounds=bounds)
                    report['attempts'].append(attempt)
                    report.update(attempt)
                    if replacement is not None:
                        if decoder == 'ctc':
                            for unit in replacement:
                                unit['timing_source'] = 'cross_model_ctc'
                        units = result[line]['units']
                        where = next(i for i, u in enumerate(units) if (u['first'], u['stop']) == (group['first'], group['stop']))
                        units[where:where+1] = replacement
                        report['accepted'] = True
                        break
            except (ValueError, RuntimeError, OSError) as error:
                report['reason'] = str(error)
    result, reverted = protect_line_order(result, cues)
    for report in reports:
        if report['line'] in reverted:
            report.update(accepted=False, reason='reverted to preserve lyric order')
    return result, reports
