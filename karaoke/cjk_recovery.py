"""Recover merged Chinese highlights with independent acoustic corroboration."""
from copy import deepcopy
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from align_lyric_phonemes import protect_line_order
from phoneme_progress import ROOT, SPOKEN_CHINESE, spoken_units
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
        syllables = lazy_pinyin(SPOKEN_CHINESE.convert(cue['text']), errors=lambda value: list(value))
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
        if any(not np.isfinite([u['start'], u['end']]).all() or u['end'] <= u['start']
               for u in [*primary, *secondary]):
            return None, 'invalid acoustic interval'
        if any(not bounds[0] <= u['start'] < bounds[1] or u['end'] > len(wave)/44100
               for u in [*primary, *secondary]):
            return None, 'acoustic evidence crosses adjacent words'
        if any(u['end']-u['start'] < .035 for u in primary):
            return None, 'collapsed phonetic span'
        starts = [u['start'] for u in primary]
        start, end = float(np.median(starts)), min(bounds[1], float(np.median([u['end'] for u in primary])))
        if max(starts)-min(starts) > protocol['maximum_spread_seconds']:
            return None, 'context disagreement'
        if (any(abs(u['start']-start) > .1 for u in secondary)
                or abs(secondary[0]['start']-secondary[1]['start']) > .1):
            return None, 'independent acoustic disagreement'
        if not bounds[0] <= start < end <= bounds[1] or (result and start-result[-1]['start'] < .015):
            return None, 'conflict with adjacent words'
        piece = wave[round(start*44100):round(end*44100)]
        if not len(piece) or not np.isfinite(piece).all() or float(np.sqrt(np.mean(piece**2))) <= floor:
            return None, 'insufficient audible vocal support'
        result.append(dict(primary[0], start=round(start, 3), end=round(end, 3),
                           timing_source='cross_model_phonemes'))
    for left, right in zip(result, result[1:]):
        left['end'] = min(left['end'], right['start'])
    return result, 'agreement across phonetic contexts, original and separated audio'


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
                predictions, windows = [], []
                for padding in (1., 1.5, 2.):
                    begin, end = max(0., cue['aligned_start']-padding), min(len(wave)/44100, cue['aligned_end']+padding)
                    prediction = model.infer(wave[round(begin*44100):round(end*44100)], begin, parsed, cue)
                    predictions.append([u for u in prediction['units'] if group['first'] <= u['first'] < group['stop']])
                    windows.append([begin, end])
                if len({tuple(w) for w in windows}) != 3:
                    raise ValueError('Recovery contexts are not distinct')
                if independent is None:
                    independent = witness(cue, wave)
                observed = [[u for u in view if group['first'] <= u['first'] < group['stop']] for view in independent]
                previous = cue['units'][index-1] if index else (cues[line-1]['units'][-1] if line else None)
                following = cue['units'][index+1] if index+1 < len(cue['units']) else (cues[line+1]['units'][0] if line+1 < len(cues) else None)
                bounds = [previous['end'] if previous else 0., following['start'] if following else len(wave)/44100]
                replacement, reason = corroborated_group(group, predictions, observed, wave, bounds, protocol)
                report.update(reason=reason, predictions=predictions, witnesses=observed, source_windows=windows, bounds=bounds)
                if replacement is not None:
                    units = result[line]['units']
                    where = next(i for i, u in enumerate(units) if (u['first'], u['stop']) == (group['first'], group['stop']))
                    units[where:where+1] = replacement
                    report['accepted'] = True
            except (ValueError, RuntimeError, OSError) as error:
                report['reason'] = str(error)
    result, reverted = protect_line_order(result, cues)
    for report in reports:
        if report['line'] in reverted:
            report.update(accepted=False, reason='reverted to preserve lyric order')
    return result, reports
