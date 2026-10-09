"""Align known English lyrics with a separate frame-level acoustic model."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from align_lyrics_whisper import alignment_blocks, sha
from stage_progress import progress_items
from word_onset_metrics import canonical
from onset_consensus import context_scope_violations
from subtitles import highlight_spans, latin_word_text

ROOT = Path(__file__).resolve().parent
SINGING_ROOT = ROOT / 'research/word-onset-2026-09-18/lyrics-alignment-multilingual'


def refine_onset_edges(log_probs, targets, word_spans, token_spans, step, ratio, max_lead=.12):
    """Use the connected leading probability edge, guarded by the prior word."""
    result = []
    previous_end = 0
    for first, stop in word_spans:
        span = token_spans[first]
        start = span.start
        probabilities = np.exp(log_probs[:, targets[first]])
        peak = float(np.max(probabilities[span.start:span.end]))
        if peak >= .05:
            limit = max(previous_end, start-int(max_lead/step))
            threshold = peak*ratio
            while start > limit and probabilities[start-1] >= threshold:
                start -= 1
        result.append(start*step)
        previous_end = token_spans[stop-1].end
    return result


def context_blocks(anchors, audio):
    """Retain ASR phrase grouping but do not trim the song to energy activity."""
    blocks = alignment_blocks(anchors, audio)
    result = []
    for first, stop, left, right, begin, end in blocks:
        if left == 0:
            begin = max(0., anchors[left]['start']-.6) if anchors[left]['status']=='candidate' else 0.
        if right == len(anchors):
            end = min(len(audio)/16000, anchors[right-1]['end']+3.) if anchors[right-1]['status']=='candidate' else len(audio)/16000
        result.append((first, stop, left, right, begin, end))
    return result


class EnglishOnsetAligner:
    def __init__(self, method, device, onset_ratio=None):
        self.method, self.device = method, device
        self.onset_ratio = onset_ratio
        if method == 'wav2vec2':
            torch.hub.set_dir(str(ROOT / 'models/alignment/wav2vec2'))
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self.model = bundle.get_model()
            self.vocab = {c:i for i,c in enumerate(bundle.get_labels())}
            self.blank = 0
            self.space = self.vocab['|']
            self.model_identity = json.loads((ROOT / 'models/alignment/wav2vec2/manifest.json').read_text())
        else:
            sys.path.insert(0, str(SINGING_ROOT))
            from model import AcousticModel, train_audio_transforms
            from utils import phone2int
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            from phonemizer.separator import Separator
            os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            self.phonemizer = EspeakBackend('en-us')
            self.separator = Separator(phone=';', word=None)
            self.model = AcousticModel(1, 256, 72, 32, 1, .1)
            checkpoint = torch.load(SINGING_ROOT / 'checkpoints/checkpoint_Baseline',
                                    map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.transform = train_audio_transforms
            self.vocab, self.blank, self.space = phone2int, 71, 69
            self.model_identity = json.loads((SINGING_ROOT / 'source-manifest.json').read_text())
        self.model.eval().to(device)

    def tokenize(self, words):
        words = [latin_word_text(word) for word in words]
        if self.method == 'wav2vec2':
            sequences = []
            for word in words:
                text = ''.join(c for c in word.upper().replace('’', "'") if c.isalnum() or c=="'")
                if not text or any(c not in self.vocab for c in text):
                    raise ValueError(f'Unsupported word for English CTC model: {word}')
                sequences.append([self.vocab[c] for c in text])
        else:
            texts = [re.sub(r'[^\w\s\'’-]', '', word).lower() for word in words]
            phonemes = self.phonemizer.phonemize(texts, separator=self.separator, strip=True)
            # A sustained hum is not a sequence of spoken letter names.
            phonemes = ['m' if re.fullmatch(r'm{2,}', text) else value for text, value in zip(texts, phonemes)]
            sequences = [[self.vocab.get(p, 70) for p in value.split(';') if p] for value in phonemes]
        targets, spans = [], []
        for word, seq in zip(words, sequences):
            if not seq:
                raise ValueError(f'No pronunciation for a required word: {word!r}')
            if targets:
                targets.append(self.space)
            start = len(targets)
            targets.extend(seq)
            spans.append((start, len(targets)))
        return targets, spans

    @torch.inference_mode()
    def align(self, audio, words):
        targets, words_to_tokens = self.tokenize(words)
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
        if self.method == 'wav2vec2':
            # The fixed bundle has stride 320 and receptive field 400 samples.
            # Acoustic chunks retain two seconds of context at either side;
            # the CTC path is still solved over the complete requested audio.
            frame_count = (waveform.shape[1]-400)//320+1
            chunks = []
            for first in range(0, frame_count, 1000):
                stop = min(frame_count, first+1000)
                left, right = max(0, first-100), min(frame_count, stop+100)
                samples = waveform[:, left*320:(right-1)*320+400].to(self.device)
                output, _ = self.model(samples)
                if output.shape[1] != right-left:
                    raise ValueError('Unexpected acoustic frame stride')
                chunks.append(output[:, first-left:stop-left].cpu())
            if not chunks:
                raise ValueError('Audio shorter than one acoustic frame')
            logits = torch.cat(chunks, dim=1)
            step = .02
        else:
            waveform = torchaudio.functional.resample(waveform, 16000, 22050)
            spec = self.transform(waveform).unsqueeze(1)
            logits = self.model(spec.to(self.device))
            step = 256/22050*3
            logits = logits[:, :int(len(audio)/16000/step)]
        emissions = logits.log_softmax(-1).cpu()
        if emissions.shape[1] < len(targets)+sum(a==b for a,b in zip(targets,targets[1:])):
            raise ValueError('Insufficient acoustic frames for required phonemes')
        path, scores = torchaudio.functional.forced_align(
            emissions, torch.tensor([targets], dtype=torch.int64), blank=self.blank)
        spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp(), blank=self.blank)
        if [int(s.token) for s in spans] != targets:
            raise ValueError('CTC path does not cover the required token sequence')
        result = [dict(text=word, start=spans[a].start*step, end=spans[b-1].end*step,
                       confidence=float(np.mean([s.score for s in spans[a:b]])))
                  for word, (a,b) in zip(words, words_to_tokens)]
        if self.onset_ratio is not None:
            refined = refine_onset_edges(emissions[0].numpy(), targets, words_to_tokens, spans,
                                         step, self.onset_ratio)
            for row, start in zip(result, refined):
                row['start'] = start
        return result, emissions[0].numpy(), step


def align_job(aligner, job, full_song=False):
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lyrics = json.loads(Path(job['lyrics']).read_text())
    if lyrics['language'] != 'English':
        raise ValueError('This adapter is validated only for English')
    audio, rate = sf.read(job['audio'], dtype='float32')
    expected_rate = getattr(aligner, 'sample_rate', 16000)
    if aligner.method == 'sofa' and audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != expected_rate or audio.ndim != 1:
        raise ValueError(f'Expected {expected_rate} Hz audio with a valid mono mix')
    lines = lyrics['lines']
    anchors = json.loads(Path(job['anchors']).read_text()) if job.get('anchors') else None
    matches = [list(highlight_spans(line, 'English')) for line in lines]
    unsupported = {i for i, row in enumerate(matches) if not row or
                   canonical(''.join(lines[i][a:b] for a, b in row)) != canonical(lines[i])}
    # An unsupported line is a boundary for this model, not a reason to lose
    # every other line's fine timing. Reuse the normal anchored block builder:
    # neighbouring supported phrases cannot take this text as their context.
    context_anchors = [dict(a, status='unsupported_text') if i in unsupported else a
                       for i, a in enumerate(anchors)] if anchors else None
    if aligner.method == 'sofa':
        if anchors is None or len(anchors) != len(lines):
            raise ValueError('SOFA requires one coarse audio-aligned window per lyric line')
        blocks = [(i, i+1, i, i+1, max(0., c['start']-.8), min(len(audio)/rate, c['end']+.8))
                  for i,c in enumerate(anchors)]
    else:
        blocks = context_blocks(context_anchors, audio) if anchors and (not full_song or unsupported) else [
            (0, len(lines), 0, len(lines), 0., len(audio)/rate)]
    cues = [dict(text=s, start=0., end=0., units=[]) for s in lines]
    failures = []
    started = time.perf_counter()

    def run_blocks(blocks, start_id=0):
        for block_id, (first, stop, left, right, begin, end) in enumerate(progress_items(blocks, 'context' if start_id else 'blocks'), start_id):
            block_matches = matches[left:right]
            words = [s[a:b] for s, row in zip(lines[left:right], block_matches) for a, b in row]
            try:
                if any(not row for row in block_matches):
                    raise ValueError('Lyric lines must contain alignable text')
                if canonical(''.join(words)) != canonical(''.join(lines[left:right])):
                    raise ValueError('Unsupported text in English lyric alignment')
                aligned, emissions, step = aligner.align(audio[round(begin*rate):round(end*rate)], words)
            except (ValueError, RuntimeError) as error:
                failures.append(dict(lines=[first, stop], error=str(error)))
                continue
            if emissions is not None:
                np.savez_compressed(output / f'emissions-{block_id:03d}.npz', values=emissions, step=step,
                                    begin=begin, core=[first, stop], context=[left, right])
            cursor = 0
            for line in range(left, right):
                units = []
                for begin_char, end_char in matches[line]:
                    word = aligned[cursor]
                    units.append(dict(text=lines[line][begin_char:end_char], first=begin_char, stop=end_char,
                                      start=word['start']+begin, end=word['end']+begin,
                                      confidence=word['confidence'], timing_source=aligner.method))
                    cursor += 1
                # Punctuation has no separate phoneme or onset, but still belongs
                # to the displayed line, including leading and trailing marks.
                units[0]['first'] = 0
                for index, unit in enumerate(units):
                    unit['stop'] = units[index+1]['first'] if index+1 < len(units) else len(lines[line])
                    unit['text'] = lines[line][unit['first']:unit['stop']]
                if first <= line < stop:
                    cues[line] = dict(text=lines[line], start=units[0]['start'], end=units[-1]['end'],
                                      aligned_start=units[0]['start'], aligned_end=units[-1]['end'], units=units)
            print(json.dumps(dict(song=lyrics.get('id'), method=aligner.method, lines=[first+1, stop])), flush=True)

    run_blocks(blocks)
    scope_blocks = context_blocks(context_anchors, audio) if full_song and anchors and aligner.method != 'sofa' else None
    context_retries = []
    if scope_blocks and not failures:
        retries = []
        for original in context_scope_violations(cues, scope_blocks):
            first, stop, left, right, begin, end = original
            # A coarse boundary can cut off a sustained syllable. Include one
            # more phrase at either edge before retrying, so similar vowels
            # retain their context instead of being forced into a short crop.
            left, right = max(0, left-1), min(len(lines), right+1)
            if anchors[left]['status'] == 'candidate':
                begin = min(begin, max(0., anchors[left]['start']-.6))
            if anchors[right-1]['status'] == 'candidate':
                end = max(end, min(len(audio)/rate, anchors[right-1]['end']+(3. if right == len(lines) else .6)))
            expanded = (first, stop, left, right, begin, end)
            retries.append(expanded)
            scope_blocks[scope_blocks.index(original)] = expanded
            context_retries.append(dict(core=[first, stop], context=[left, right], source_range=[begin, end],
                                        original_scope=original,
                                        original_cues=cues[first:stop]))
            cues[first:stop] = [dict(text=lines[i], start=0., end=0., units=[]) for i in range(first, stop)]
        run_blocks(retries, start_id=len(blocks))
        remaining = context_scope_violations(cues, scope_blocks)
        for block, report in zip(retries, context_retries):
            report.update(accepted=block not in remaining, candidate_cues=cues[block[0]:block[1]])
        if remaining:
            failures.append(dict(lines=[b[:2] for b in remaining], error='Word timing falls outside acoustic phrase context'))

    def order_conflicts():
        units = [(i, u) for i, cue in enumerate(cues) for u in cue['units']]
        return {(i, j) for (i, a), (j, b) in zip(units, units[1:]) if a['start'] >= b['start']}

    boundary_retries, phrase_retries = [], []

    def repair_boundaries():
        if not scope_blocks or failures:
            return
        for initial in sorted(order_conflicts()):
            if initial not in order_conflicts():
                continue
            first, last = initial
            before = order_conflicts()
            for _ in range(12):
                left, right = max(0, first-1), min(len(lines), last+2)
                if any(anchors[i]['status'] != 'candidate' for i in (left, right-1)):
                    break
                begin, end = max(0., anchors[left]['start']-.6), min(len(audio)/rate, anchors[right-1]['end']+.6)
                if not 0 < end-begin <= 60:
                    break
                previous = cues[first:last+1]
                run_blocks([(first, last+1, left, right, begin, end)],
                           start_id=len(blocks)+len(context_retries)+len(boundary_retries)+len(phrase_retries))
                after = order_conflicts()
                outside = context_scope_violations(cues, scope_blocks) if not failures else []
                accepted = not failures and after < before and not outside
                boundary_retries.append(dict(core=[first, last+1], context=[left, right],
                    source_range=[begin, end], accepted=accepted, original_cues=previous,
                    candidate_cues=cues[first:last+1], conflicts=sorted(after), outside_scope=outside))
                if accepted:
                    break
                cues[first:last+1] = previous
                # A local repair may move the crossing to its edge. Include
                # that neighbor in the next joint decode; never commit the
                # partial repair or enlarge the original acceptance bounds.
                moved = [(a, b) for a, b in after-before if a < first <= b or a <= last < b]
                if failures or outside or not moved:
                    break
                first = min(first, min(a for a, _ in moved))
                last = max(last, max(b for _, b in moved))

    repair_boundaries()
    # A prediction can stay inside a multi-line context yet move a complete
    # phrase into the following dialogue. Revisit only disjoint phrases, with
    # neighbouring text for context; never clamp words to the rough anchor.
    def displaced(i):
        a, c = anchors[i], cues[i]
        return a['status'] == 'candidate' and (
            c['end']+2 < a['start'] or c['start']-2 > a['end'] or
            # Complete audio-aligned words also locate the whole phrase.
            # A partial ASR clause does not bound its unheard remainder.
            len(c['units']) > 1 and (a.get('units') or a.get('match_basis') == 'complete_phrase') and (
                c['start']+2 < a['start'] or c['units'][-1]['start'] > a['end']+2))
    def needs_phrase_retry(i):
        a, c = anchors[i], cues[i]
        # Mere overlap can hide a compressed prefix or suffix. The coarse
        # envelope requests a fresh decode; it never supplies word times.
        # A one-word held note may have a long coarse tail but one short CTC
        # event; that alone is not compressed word timing.
        return displaced(i) or len(c['units']) > 1 and a['status'] == 'candidate' and (
            c['start'] > a['start']+2 or c['end']+2 < a['end'])
    if scope_blocks and not failures and not order_conflicts():
        for i in range(len(cues)):
            if not needs_phrase_retry(i):
                continue
            left, right = max(0, i-1), min(len(cues), i+2)
            if any(anchors[j]['status'] != 'candidate' for j in (left, right-1)):
                continue
            begin, end = max(0., anchors[left]['start']-.6), min(len(audio)/rate, anchors[right-1]['end']+.6)
            if not 0 < end-begin <= 60:
                continue
            previous = cues[i]
            run_blocks([(i, i+1, left, right, begin, end)],
                       start_id=len(blocks)+len(context_retries)+len(boundary_retries)+len(phrase_retries))
            accepted = not needs_phrase_retry(i) and not context_scope_violations(cues, scope_blocks)
            phrase_retries.append(dict(line=i+1, context=[left,right], source_range=[begin,end],
                original_cue=previous, candidate_cue=cues[i], accepted=accepted))
            if not accepted:
                cues[i] = previous
        outside = context_scope_violations(cues, scope_blocks)
        if outside:
            failures.append(dict(lines=[b[:2] for b in outside], error='Phrase retry left its acoustic context'))

    # A corrected phrase can cross a neighbor that was previously ordered.
    # Resolve the new boundary jointly before accepting or rejecting the song.
    if phrase_retries:
        repair_boundaries()
    if order_conflicts():
        failures.append(dict(lines=sorted(order_conflicts()), error='Word onsets remain out of order'))
    if scope_blocks:
        unresolved = [i for i in range(len(cues)) if displaced(i)]
        if unresolved:
            failures.append(dict(lines=unresolved, error='Phrase timing contradicts its acoustic location'))
        unsafe = set(unresolved)
        unsafe.update(i for a,b in order_conflicts() for i in (a,b))
        unsafe.update(i for b in context_scope_violations(cues, scope_blocks) for i in range(b[0],b[1]))
        for i,cue in enumerate(cues):
            cue['status'] = 'unresolved_timing' if i in unsafe else 'candidate'
    if aligner.device == 'mps':
        torch.mps.synchronize()
    report = dict(method=aligner.method, device=aligner.device, seconds=time.perf_counter()-started,
                  full_song=full_song, failures=failures, scope_blocks=scope_blocks,
                  context_retries=context_retries, boundary_retries=boundary_retries, audio_sha256=sha(job['audio']),
                  phrase_retries=phrase_retries,
                  onset_ratio=aligner.onset_ratio,
                  anchors_sha256=sha(job['anchors']) if job.get('anchors') else None,
                  lyrics_sha256=sha(job['lyrics']), runner_sha256=sha(__file__),
                  model_identity=aligner.model_identity, torch_version=torch.__version__,
                  torchaudio_version=torchaudio.__version__,
                  decoder=('SOFA dense phoneme and boundary dynamic programming' if aligner.method=='sofa'
                           else 'torchaudio CTC forced_align; unaligned words are missing, no interpolation'),
                  acoustic_chunking=({'sofa':'Individual coarse lyric lines with 0.8s padding',
                                      'wav2vec2':'20s core + 2s context each side; 20ms frame stride',
                                      'singing':'Complete requested audio; 34.83ms frame stride'}[aligner.method]),
                  boundary_resolution='model frame grid, no fitted offset')
    (output / 'cues.json').write_text(json.dumps(cues, ensure_ascii=False, indent=2)+'\n')
    (output / 'run.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['wav2vec2', 'singing'], required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--full-song', action='store_true')
    parser.add_argument('--onset-ratio', type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.onset_ratio is not None and not 0 < args.onset_ratio <= 1:
        parser.error('--onset-ratio must be in (0, 1]')
    aligner = EnglishOnsetAligner(args.method, args.device, args.onset_ratio)
    for job in json.loads(args.jobs.read_text()):
        print(json.dumps(align_job(aligner, job, args.full_song)), flush=True)
