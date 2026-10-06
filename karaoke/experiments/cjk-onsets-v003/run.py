"""Compare fixed CJK display units across three acoustic contexts."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from align_lyric_phonemes import SingingAligner, protect_units, protect_line_order
from lyric_progress import canonical_positions
from phoneme_progress import spoken_units
from subtitles import presentation_cues, highlight_spans, highlight_coverage_errors
from stage_progress import progress_items


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def fixed_units(cue, language):
    """Keep the previous display segmentation, including ambiguous compounds."""
    parsed = spoken_units(cue['text'], language)
    result = []
    for old in cue['units']:
        unit = {k: old[k] for k in ('first', 'stop', 'text')}
        if old.get('phones'):
            unit['phones'] = old['phones']
        else:
            parts = [p for p in parsed if old['first'] <= p['first'] and p['stop'] <= old['stop']]
            normalize = lambda text: canonical_positions(text, language)[0]
            if normalize(''.join(p['text'] for p in parts)) != normalize(old['text']):
                raise ValueError('Cannot map pronunciation without changing display units')
            unit['phones'] = [phone for part in parts for phone in part['phones']]
        result.append(unit)
    return result


def repair_character_groups(cues, baseline, wave, model, protocol):
    """Retry merged Chinese characters with their actual neighbouring lyrics."""
    reports = []
    floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    for index, cue in enumerate(cues):
        groups = [u for u in cue['units'] if len(list(highlight_spans(u['text'], 'Chinese'))) > 1
                  and u.get('timing_source') != 'repeated_lyric_audio']
        if not groups:
            continue
        report = dict(line=index+1, attempts=[], groups=[])
        reports.append(report)
        first, stop = max(0, index-1), min(len(baseline), index+2)
        text, anchors, offset = '', [], 0
        for line in range(first, stop):
            if line == index:
                offset = len(text)
            anchors.extend(dict(u, first=u['first']+len(text), stop=u['stop']+len(text))
                           for u in baseline[line]['units'])
            text += baseline[line]['text']+' '
        context = dict(text=text.strip(), units=anchors,
                       aligned_start=baseline[first]['aligned_start'],
                       aligned_end=baseline[stop-1]['aligned_end'])
        try:
            parsed = spoken_units(context['text'], 'Chinese')
            if context['aligned_end']-context['aligned_start'] > 30:
                raise ValueError('Neighbouring lyric context exceeds 30 seconds')
            for pad in protocol['context_padding_seconds']:
                begin = max(0., context['aligned_start']-pad)
                end = min(len(wave)/44100, context['aligned_end']+pad)
                prediction = model.infer(wave[round(begin*44100):round(end*44100)], begin, parsed, context)
                units = [dict(u, first=u['first']-offset, stop=u['stop']-offset)
                         for u in prediction['units'] if offset <= u['first'] < offset+len(cue['text'])]
                guarded, reasons = protect_units(baseline[index], units, 'Chinese')
                report['attempts'].append(dict(padding=pad, source_range=[begin, end],
                                               guarded=guarded, reasons=reasons))
        except (ValueError, RuntimeError) as error:
            report['error'] = str(error)
            continue
        replacements = {}
        for group in groups:
            variants = [[u for u in a['guarded'] if group['first'] <= u['first'] and u['stop'] <= group['stop']]
                        for a in report['attempts']]
            decision = dict(first=group['first'], stop=group['stop'], accepted=False)
            report['groups'].append(decision)
            signatures = [[(u['first'], u['stop']) for u in variant] for variant in variants]
            if (not variants or not variants[0] or any(s != signatures[0] for s in signatures)
                    or any(canonical_positions(''.join(u['text'] for u in v), 'Chinese')[0]
                           != canonical_positions(group['text'], 'Chinese')[0] for v in variants)
                    or any(u['timing_source'] != 'singing_phonemes'
                           or len(list(highlight_spans(u['text'], 'Chinese'))) != 1 for v in variants for u in v)):
                decision['reason'] = 'not all contexts produced separate usable characters'
                continue
            candidate, reasons = [], []
            for unit_index, choices in enumerate(zip(*variants)):
                starts = [u['start'] for u in choices]
                start, end = float(np.median(starts)), float(np.median([u['end'] for u in choices]))
                piece = wave[round(start*44100):round(end*44100)]
                if max(starts)-min(starts) > protocol['maximum_spread_seconds']:
                    reasons.append('context disagreement')
                if unit_index == 0 and abs(start-group['start']) > protocol['maximum_change_seconds']:
                    reasons.append('large change needs independent verification')
                if not len(piece) or float(np.sqrt(np.mean(piece**2))) <= floor:
                    reasons.append('insufficient audible vocal support')
                candidate.append(dict(choices[0], start=round(start, 3), end=round(end, 3),
                                      timing_source='singing_context_characters'))
            if reasons:
                decision['reason'] = '; '.join(sorted(set(reasons)))
                continue
            replacements[(group['first'], group['stop'])] = candidate
            decision.update(accepted=True, units=candidate)
        cue['units'] = [part for u in cue['units'] for part in replacements.get((u['first'], u['stop']), [u])]
    result, reverted = protect_line_order(cues, baseline)
    for report in reports:
        if report['line'] in reverted:
            for group in report['groups']:
                group.update(accepted=False, reason='reverted to preserve lyric order')
    return result, reports


def process(job, model, protocol):
    output = Path(job['output']) if job.get('output') else HERE / job['song']
    output.mkdir(parents=True)
    baseline = json.loads(Path(job['cues']).read_text())
    cues = copy.deepcopy(baseline)
    wave, rate = sf.read(job['audio'], dtype='float32', always_2d=True)
    divisor = np.gcd(rate, 44100)
    wave = resample_poly(wave.mean(axis=1), 44100//divisor, rate//divisor).astype('float32')
    floor = max(1e-5, protocol['minimum_relative_rms']*float(np.sqrt(np.mean(wave**2))))
    raw, decisions = [], []
    started = time.perf_counter()
    for line, (cue, old) in enumerate(zip(progress_items(cues, 'lines'), baseline), 1):
        attempts = []
        try:
            units = fixed_units(cue, job['language'])
            for pad in protocol['context_padding_seconds']:
                begin = max(0., cue['aligned_start']-pad)
                end = min(len(wave)/44100, cue['aligned_end']+pad)
                prediction = model.infer(wave[round(begin*44100):round(end*44100)], begin, units, cue)
                guarded, reasons = protect_units(cue, prediction['units'], job['language'])
                attempts.append(dict(padding=pad, guarded=guarded, reasons=reasons, **prediction))
        except (ValueError, RuntimeError) as error:
            attempts.append(dict(error=str(error)))
        raw.append(dict(line=line, attempts=attempts))
        for index, (unit, previous) in enumerate(zip(cue['units'], old['units'])):
            decision = dict(line=line, unit=index+1, text=unit['text'], previous=previous['start'])
            valid = [a['guarded'][index] for a in attempts if 'guarded' in a
                     and a['guarded'][index]['timing_source'] == 'singing_phonemes']
            if previous.get('timing_source') == 'repeated_lyric_audio':
                decision['reason'] = 'preserve existing independent repeated-audio recovery'
            elif len(valid) != len(protocol['context_padding_seconds']):
                decision['reason'] = 'not all windows produced a usable boundary'
            else:
                starts = [v['start'] for v in valid]
                start, end = float(np.median(starts)), float(np.median([v['end'] for v in valid]))
                piece = wave[round(start*44100):round(end*44100)]
                rms = float(np.sqrt(np.mean(piece**2))) if len(piece) else 0.
                decision.update(proposals=starts, spread=max(starts)-min(starts), rms=rms)
                if decision['spread'] > protocol['maximum_spread_seconds']:
                    decision['reason'] = 'context disagreement'
                elif abs(start-previous['start']) > protocol['maximum_change_seconds']:
                    decision['reason'] = 'large change needs independent verification'
                elif rms <= floor:
                    decision['reason'] = 'insufficient audible vocal support'
                else:
                    unit.update(start=round(start, 3), end=round(end, 3),
                                timing_source='singing_window_consensus')
                    decision['reason'] = 'accepted three-window median'
            decision['proposed'] = unit['start']
            decisions.append(decision)
        if line % 15 == 0:
            print(job['song'], line, len(cues), flush=True)
    old_units = [u for c in baseline for u in c['units']]
    all_units = [u for c in cues for u in c['units']]
    while True:
        bad = {j for i in range(len(all_units)-1)
               if all_units[i+1]['start']-all_units[i]['start'] < .015-1e-9
               for j in (i, i+1)}
        changed = [i for i in bad if all_units[i]['start'] != old_units[i]['start']]
        if not changed:
            if bad:
                raise ValueError('Baseline contains unordered onsets')
            break
        for i in changed:
            all_units[i].update(old_units[i])
            decisions[i]['reason'] = 'reverted to preserve lyric order'
    for i, unit in enumerate(all_units):
        if i+1 < len(all_units):
            unit['end'] = min(unit['end'], all_units[i+1]['start'])
        if unit['end'] <= unit['start']:
            raise ValueError('Non-positive voiced interval')
        decisions[i]['final'] = unit['start']
    character_repairs = []
    verified_repairs = []
    witness = None
    if job['language'] == 'Chinese':
        cues, character_repairs = repair_character_groups(cues, baseline, wave, model, protocol)
        if job.get('original_audio'):
            from cjk_recovery import AcousticWitness, recover_groups, recover_context_groups, recover_repeated_groups
            witness = AcousticWitness(job['original_audio'], model.device)
            cues, verified_repairs = recover_groups(cues, wave, model,
                witness, protocol)
            cues, contextual_repairs = recover_context_groups(cues, wave, model, witness, protocol)
            verified_repairs.extend(contextual_repairs)
            cues, repeated_repairs = recover_repeated_groups(cues, wave, witness, protocol)
            verified_repairs.extend(repeated_repairs)
            if highlight_coverage_errors(cues, 'Chinese') and (ROOT/'models/alignment/qwen3-forced-aligner-0.6b/model.safetensors').is_file():
                write(output/'qwen-input.json', cues)
                try:
                    with (output/'qwen.log').open('w') as log:
                        subprocess.run([os.getenv('KARAOKE_QWEN_PYTHON', sys.executable), str(ROOT/'qwen_witness.py'),
                            '--cues', str(output/'qwen-input.json'), '--audio', job['audio'],
                            '--original', job['original_audio'], '--output', str(output/'qwen-witnesses.json'),
                            '--device', str(model.device)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                    views = json.loads((output/'qwen-witnesses.json').read_text())
                    indices = {id(c): str(i) for i, c in enumerate(cues)}
                    cues, alternate = recover_groups(cues, wave, model,
                        lambda cue, _: views.get(indices[id(cue)], []), protocol)
                    verified_repairs.extend(dict(r, verification_model='qwen3_forced_aligner') for r in alternate)
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    verified_repairs.append(dict(verification_model='qwen3_forced_aligner', accepted=False,
                                                 reason=type(error).__name__))
        all_units = [u for c in cues for u in c['units']]
        for left, right in zip(all_units, all_units[1:]):
            left['end'] = min(left['end'], right['start'])
        if any(u['end'] <= u['start'] for u in all_units):
            raise ValueError('Non-positive character interval')
    latin_repairs = []
    if job.get('original_audio'):
        from cjk_recovery import AcousticWitness, recover_latin_groups
        witness = witness or AcousticWitness(job['original_audio'], model.device)
        cues, latin_repairs = recover_latin_groups(cues, wave, witness, protocol)
        all_units = [u for c in cues for u in c['units']]
    first_word_repairs = []
    coarse_relocations = []
    if witness is not None:
        from cjk_recovery import recover_first_words, relocalize_fallback_lines
        cues, coarse_relocations = relocalize_fallback_lines(cues, wave, model, witness, protocol, job['language'])
        cues, first_word_repairs = recover_first_words(cues, wave, model, witness, protocol,
                                                       job['language'], raw)
        all_units = [u for c in cues for u in c['units']]
    for cue in cues:
        cue.update(start=cue['units'][0]['start'], end=cue['units'][-1]['end'])
    cues, presentation = presentation_cues(cues, job['duration'], lead=0., tail=0., bridge_gap=0.)
    changes = [d for d in decisions if abs(d['final']-d['previous']) >= .005]
    report = dict(song=job['song'], seconds=time.perf_counter()-started, lines=len(cues),
        units=len(all_units), changed_at_least_5ms=len(changes),
        changed_line_starts=sum(d['unit'] == 1 for d in changes),
        changed_at_least_50ms=sum(abs(d['final']-d['previous']) >= .05 for d in changes),
        preserved_repeat_units=sum(u.get('timing_source') == 'repeated_lyric_audio' for u in all_units),
        accepted_window_units=sum(u.get('timing_source') == 'singing_window_consensus' for u in all_units),
        repaired_first_words=sum(r['accepted'] for r in first_word_repairs),
        relocalized_coarse_lines=sum(r['accepted'] for r in coarse_relocations),
        repaired_character_units=sum(u.get('timing_source') == 'singing_context_characters' for u in all_units),
        corroborated_character_units=sum(u.get('timing_source') in ('cross_model_phonemes', 'cross_model_ctc')
            and any('\u3400' <= c <= '\u9fff' for c in u['text']) for u in all_units),
        corroborated_latin_units=sum(u.get('timing_source') == 'cross_model_phonemes'
            and any(c.isascii() and c.isalpha() for c in u['text']) for u in all_units),
        maximum_window_change_ms=max((abs(d['final']-d['previous'])*1000 for d in decisions), default=0),
        checkpoint=str(model.checkpoint), presentation=presentation,
        accuracy=None, accuracy_note='No independent manual onset reference supplied; consistency is not accuracy')
    write(output/'cues.json', cues)
    write(output/'raw.json', raw)
    write(output/'decisions.json', decisions)
    write(output/'character-repairs.json', character_repairs)
    write(output/'verified-repairs.json', verified_repairs)
    write(output/'latin-repairs.json', latin_repairs)
    write(output/'first-word-repairs.json', first_word_repairs)
    write(output/'coarse-relocations.json', coarse_relocations)
    write(output/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, default=HERE/'jobs.json')
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    models, reports = {}, []
    for job in json.loads(args.jobs.read_text()):
        if job['language'] not in models:
            models[job['language']] = SingingAligner(job['language'], args.device)
        reports.append(process(job, models[job['language']], protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output/'snapshot'
    snapshot.mkdir()
    sources = []
    for name in ('align_lyric_phonemes.py', 'phoneme_progress.py', 'subtitles.py', 'lyric_progress.py'):
        shutil.copy2(ROOT/name, snapshot/name)
        sources.append(dict(file=name, sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()))
    write(args.output/'report.json', dict(songs=reports, source_files=sources))
