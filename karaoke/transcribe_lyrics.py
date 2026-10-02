"""Obtain an ASR draft and coarse word times from an existing vocal preview."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from openai import OpenAI
from job_schema import song_language
from stage_progress import progress_items, report_progress
from stage_cache import StageCache, digest


def cached_recognition(client, audio, request, output):
    """Keep every API response, including rejected drafts, across stage retries."""
    config = dict(request=request, implementation=inspect.getsource(cached_recognition))
    key = hashlib.sha256((audio.name+digest(audio)+json.dumps(config, sort_keys=True)).encode()).hexdigest()[:16]
    folder = output.parent/'asr-requests'
    folder.mkdir(exist_ok=True)
    path = folder/(key+'.json')
    with StageCache(output.parent).stage('asr_request_'+key, inputs=[audio], outputs=[path],
                                        config=config) as reused:
        if not reused:
            with audio.open('rb') as stream:
                data = client.audio.transcriptions.create(file=stream, **request).model_dump()
            path.write_text(json.dumps(data, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())


def cached_local_recognition(audio, language, output, *, weights=None):
    """Use the existing alignment model to check a short disputed ending."""
    from importlib.metadata import version
    weights = Path(weights) if weights is not None else Path(__file__).resolve().parent/'models/alignment/whisper/large-v3-turbo.pt'
    if not weights.is_file():
        raise ValueError('Local ASR repair requires the installed large-v3-turbo alignment model')
    config = dict(language=language, implementation=inspect.getsource(cached_local_recognition),
                  packages={name: version(name) for name in ('openai-whisper', 'torch')})
    key = hashlib.sha256((digest(audio)+json.dumps(config, sort_keys=True)).encode()).hexdigest()[:16]
    path = output.parent/'asr-requests'/('local-'+key+'.json')
    path.parent.mkdir(exist_ok=True)
    with StageCache(output.parent).stage('local_asr_'+key, inputs=[audio, weights],
                                        outputs=[path], config=config) as reused:
        if not reused:
            import torch
            import whisper
            from whisper import timing
            torch.set_num_threads(4)
            torch.set_grad_enabled(False)
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = whisper.load_model(str(weights), device='cpu')
            heads = model._buffers.pop('alignment_heads')
            model.to(device)
            model.register_buffer('alignment_heads', heads, persistent=False)
            if device == 'mps':
                timing.dtw = lambda matrix: timing.dtw_cpu(matrix.detach().cpu().double().numpy())
            code = {'Chinese': 'zh', 'Japanese': 'ja', 'English': 'en'}[song_language(language)]
            data = whisper.transcribe(model, str(audio), language=code, fp16=False,
                temperature=0, condition_on_previous_text=False, word_timestamps=True, verbose=False)
            data['words'] = [word for segment in data['segments'] for word in segment['words']]
            path.write_text(json.dumps(data, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())


def repetition_loop(data):
    """Detect decoder loops, not ordinary repeated lyric lines or short vocalizations."""
    segments = data.get('segments') or []
    texts = [''.join(c for c in s['text'].casefold() if c.isalnum()) for s in segments]
    # A decoder can emit the same phrase in many tiny segments. Require collapsed
    # timing as corroboration so a normally timed repeated chorus stays valid.
    collapsed = sum('start' in s and 'end' in s and s['end']-s['start'] <= .025 for s in segments)
    candidates = texts + ([''.join(texts)] if collapsed > len(segments)/2 else [])
    for text in candidates:
        if len(text) >= 40 and re.search(r'(.{1,32}?)\1{7,}', text):
            return True
    # A local burst can be much shorter than the rest of the song. Eight
    # repetitions in contiguous collapsed intervals are already unusable.
    burst, previous_end = '', None
    for segment, text in zip(segments, texts):
        start, end = segment.get('start'), segment.get('end')
        if start is None or end is None or end-start > .025:
            burst, previous_end = '', None
            continue
        burst = (burst if previous_end is not None and 0 <= start-previous_end <= .025 else '') + text
        previous_end = end
        if re.search(r'(.{1,32}?)\1{7,}', burst):
            return True
    return False


def repeated_tail(data):
    """Locate a repeated ending with both silence warnings and collapsed words."""
    segments = data.get('segments') or []
    texts = [''.join(c for c in s.get('text', '').casefold() if c.isalnum()) for s in segments]
    if not texts or not texts[-1]:
        return None
    first = len(texts)-1
    while first and texts[first-1] == texts[-1]:
        first -= 1
    tail = segments[first:]
    if len(tail) < 8 or len(texts[-1])*len(tail) < 40:
        return None
    words = [w for w in data.get('words') or [] if w['start'] >= tail[0]['start']]
    if (any(s.get('no_speech_prob', 0) >= .6 for s in tail) and
            sum(w['end']-w['start'] <= .001 for w in words) >= 3):
        return first
    return None


def needs_recovery(data):
    # A fluent phrase can still be hallucinated over an intro or outro. Require
    # collapsed timing to corroborate no-speech warnings (common in real singing).
    # Re-recognize these words; do not silently delete them or score their onset.
    for segment in data.get('segments') or []:
        if segment.get('no_speech_prob', 0) < .6:
            continue
        start, end = segment.get('start', 0), segment.get('end', 0)
        words = [w for w in data.get('words') or [] if start <= w['start'] < end]
        collapsed = sum(w['end']-w['start'] <= .001 for w in words)
        if len(words) >= 3 and (end-start <= .2 or collapsed >= len(words)/2):
            return True
    # A wrong language/tokenization can collapse several lines into one "word"
    # and push the first segment past later segments. Such timing cannot guide
    # the alignment windows; retry it rather than sorting or clamping timestamps.
    for kind in ('segments', 'words'):
        timed = [item for item in data.get(kind) or [] if 'start' in item and 'end' in item]
        if any(not math.isfinite(item['start']) or not math.isfinite(item['end']) or
               not 0 <= item['start'] <= item['end'] for item in timed):
            return True
        if any(right['start'] < left['start'] for left, right in zip(timed, timed[1:])):
            return True
        if kind == 'words':
            # A few zero-length subwords are normal; a whole phrase at one
            # instant is not usable timing evidence. Re-recognize, never delete it.
            collapsed = Counter(item['start'] for item in timed
                if item['start'] == item['end'] and any(c.isalnum() for c in item.get('word', '')))
            if max(collapsed.values(), default=0) >= 8:
                return True
    text = ''.join(s.get('text', '') for s in data.get('segments') or []) or data.get('text', '')
    return not any(c.isalnum() for c in text) or repetition_loop(data) or repeated_tail(data) is not None


def select_recovery_text(text, initial_text, words, language):
    """Keep full-recording text; detect a response containing only an intro."""
    from anchors_from_asr import text_units
    from rapidfuzz.distance import Levenshtein
    observed = [unit for word in words for unit in text_units(word['word'], language)]
    recovered = text_units(text, language)
    report = dict(recovered_units=len(recovered), timed_units=len(observed), selected='recovered_text')
    if len(recovered) >= len(observed)/2:
        return text, report
    initial = text_units(initial_text, language)
    matched = sum(block.src_end-block.src_start for block in Levenshtein.opcodes(initial, observed)
                  if block.tag == 'equal')
    report.update(initial_units=len(initial), initial_support=matched/max(1, len(initial)),
                  timed_word_coverage=matched/max(1, len(observed)), selected=None)
    # Invalid initial timestamps do not invalidate its text. Reuse it only with
    # independent word evidence in both directions, never its broken timestamps.
    if (not needs_recovery(dict(segments=[dict(text=initial_text)])) and
            min(report['initial_support'], report['timed_word_coverage']) >= .55):
        report['selected'] = 'initial_transcription'
        return initial_text, report
    return None, report


def splice_loop_window(original, replacement, left, right, begin, end=None):
    """Replace a broken passage only between independently matched sung anchors."""
    from rapidfuzz.distance import Levenshtein
    from subtitles import normalized
    segments = original['segments']
    parts = replacement.get('segments') or []
    if song_language(replacement.get('language')) != song_language(original.get('language')):
        return None

    def boundary(anchor, side):
        expected = normalized(anchor['text']).casefold()
        candidates = []
        # An independent decode may split the same lyric into several segments.
        for first, part in enumerate(parts):
            if abs(part['start']+begin-anchor['start']) > 2:
                continue
            for stop in range(first+1, min(len(parts), first+3)+1):
                if abs(parts[stop-1]['end']+begin-anchor['end']) > 2:
                    continue
                observed = normalized(''.join(p['text'] for p in parts[first:stop])).casefold()
                # Padding can contain the tail/head of a neighboring lyric.
                # Only ignore text on the outside of the eventual splice.
                observed = observed[-len(expected):] if side == 'left' else observed[:len(expected)]
                score = Levenshtein.normalized_similarity(expected, observed)
                if score >= .8:
                    candidates.append((score, first, stop))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        best = candidates[0]
        # Two disjoint, nearly identical matches do not identify a safe cut.
        if any(score >= best[0]-.05 and (stop <= best[1] or first >= best[2])
               for score, first, stop in candidates[1:]):
            return None
        return best[1:]

    before = boundary(segments[left], 'left')
    after = boundary(segments[right], 'right') if right is not None else (len(parts), len(parts))
    if before is None or after is None or before[1] >= after[0] or (right is None and end is None):
        return None
    first, stop = before[1], after[0]
    # Word times are taken directly from this decode, never interpolated at cuts.
    lo = parts[first]['start']
    hi = parts[stop]['start'] if right is not None else end-begin
    limit = segments[right]['start'] if right is not None else end
    body = [dict(s, start=s['start']+begin, end=s['end']+begin) for s in parts[first:stop]]
    words = [dict(w, start=w['start']+begin, end=w['end']+begin)
             for w in replacement.get('words') or [] if lo <= w['start'] < hi]
    if (not words or body[0]['start'] < segments[left]['end'] or
            body[-1]['end'] > limit or
            any(w['start'] < segments[left]['end'] or w['end'] > limit for w in words)):
        return None
    joined = segments[:left+1] + body + (segments[right:] if right is not None else [])
    result = dict(original, segments=joined, text=' '.join(s['text'] for s in joined),
        words=[w for w in original.get('words') or [] if w['start'] < segments[left]['end']] + words +
              ([w for w in original.get('words') or [] if w['start'] >= segments[right]['start']]
               if right is not None else []))
    # Cropped audio can itself loop at its final cut. Apply the quality gate to
    # the actual joined transcript, including a recovered end-of-file passage.
    return None if needs_recovery(result) else result


def repair_loop_window(data, audio, duration, recognize, output, local_recognize=None):
    """Try one bounded decode around a localized loop before retrying the song."""
    from subtitles import normalized
    segments = data.get('segments') or []
    bad = [i for i, s in enumerate(segments) if repetition_loop(dict(segments=[s]))]
    tail = repeated_tail(data)
    if tail is not None:
        bad.extend(range(tail, len(segments)))
        bad = sorted(set(bad))
    if not bad or not data.get('words') or song_language(data.get('language')) is None:
        return None

    def usable_anchor(index):
        text = normalized(segments[index]['text']).casefold()
        # Short repeated syllables cannot identify where one decode joins another.
        return len(text) >= 6 and len(set(text)) >= 4

    lefts = [i for i in range(bad[0]-1, max(-1, bad[0]-4), -1) if usable_anchor(i)]
    left = lefts[0] if lefts else None
    right = next((i for i in range(bad[-1]+1, min(len(segments), bad[-1]+4)) if usable_anchor(i)), None)
    if left is None or (right is None and (tail is None or bad[-1] != len(segments)-1)):
        return None
    # The nearest phrase can itself contain an ASR error. Include up to three
    # preceding anchors in one tail decode, then require one to match.
    begin = max(0., segments[lefts[-1] if right is None else left]['start']-.5)
    end = min(duration, segments[right]['end']+.5) if right is not None else duration
    if not 0 < end-begin <= 90:
        return None
    clip = output/'loop-repair.mp3'
    report = dict(begin=begin, end=end, left=left, right=right, accepted=False)
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(begin), '-i', str(audio),
            '-t', str(end-begin), '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', str(clip)], check=True)
        part = local_recognize(clip) if right is None and local_recognize else recognize(clip)
        result = None
        for candidate in lefts if right is None else [left]:
            result = splice_loop_window(data, part, candidate, right, begin, end)
            if result is not None:
                report['left'] = candidate
                break
        report['accepted'] = result is not None
        (output/'loop-repair.json').write_text(json.dumps(dict(report, response=part), ensure_ascii=False)+'\n')
        return (result, report) if result is not None else None
    finally:
        clip.unlink(missing_ok=True)


def transcribe(client, audio, language, hint, output, *, allow_unresolved_text=False):
    options = {} if language == 'auto' else {'language': language}

    def unusable(prediction):
        return needs_recovery(prediction) or (language == 'auto' and
            song_language(prediction.get('language')) is None)

    def recognize(path, model='whisper-1', instruction=None):
        request = dict(model=model, **options, prompt=instruction if instruction is not None else hint,
            temperature=0, response_format='verbose_json' if model == 'whisper-1' else 'json',
            **({'timestamp_granularities': ['word', 'segment']} if model == 'whisper-1' else {}))
        return cached_recognition(client, path, request, output)

    data = recognize(audio)
    recovery = dict(triggered=False, chunks=[])
    if not unusable(data):
        return data, recovery
    (output/'initial-transcription.json').write_text(json.dumps(data, ensure_ascii=False)+'\n')
    # Verify against the uploaded recording, before an empty tail can trigger
    # another whole-song request. Do not use a loudness threshold: weak words
    # remain evidence. Invalid timestamps still need recovery, not deletion.
    from io import BytesIO
    import soundfile as sf
    pcm, rate = sf.read(BytesIO(subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin',
        '-i', str(audio), '-c:a', 'pcm_f32le', '-f', 'wav', '-'])), dtype='float32')
    filtered, rejected = filter_silent_transcription(data, pcm, rate)
    applied = bool(rejected) and all(r['reason'] == 'digital_silence' for r in rejected)
    recovery['silence_filter'] = dict(applied=applied, rejected=len(rejected))
    (output/'silence-filter.json').write_text(json.dumps(dict(
        audio_sha256=digest(audio), sample_rate=rate, applied=applied, rejected=rejected),
        ensure_ascii=False)+'\n')
    if applied:
        data = filtered
        if not unusable(data):
            return data, recovery
    report_progress(detail='识别结果异常，分段重试')
    # An unusable transcript's language detection is also untrusted. Preserve an
    # explicit user language, but let recovery re-detect automatic requests.
    duration = len(pcm)/rate
    repaired = repair_loop_window(data, audio, duration, recognize, output,
        local_recognize=lambda path: cached_local_recognition(path, data['language'], output))
    if repaired is not None:
        data, report = repaired
        return data, dict(recovery, triggered=True,
            text_model='local-large-v3-turbo' if report['right'] is None else 'whisper-1', loop_repair=report)
    if repeated_tail(data) is not None and song_language(data.get('language')) is not None:
        raise ValueError('重复的结尾转录未通过局部核验，已保留诊断')
    fallback_model = os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe')
    # Keep full recording context for text: independent chunks can omit a split
    # syllable or duplicate the opening of the next lyric. Chunks only locate words.
    for instruction in (None,
            'Transcribe only intelligible sung or spoken words. Omit wordless humming, '
            'sustained nonverbal vowels, and instrumental sounds. Do not expand held '
            'notes into repeated syllables. If there are no intelligible words, return an empty transcript.'):
        text = recognize(audio, fallback_model, instruction)['text'].strip()
        segments = [dict(start=0., end=duration, text=text, timing_source='recording_bounds')]
        if not needs_recovery(dict(segments=segments)):
            break
    if needs_recovery(dict(segments=segments)):
        raise ValueError('歌词文本重试仍有重复异常或为空，无法生成可靠歌词')
    (output/'recovered-text.json').write_text(json.dumps(segments, ensure_ascii=False)+'\n')

    def chunk(start, window=30):
        length = min(window, duration-start)
        suffix = '' if window == 30 else f'-{window}s'
        path = output/f'chunk-{start:05d}{suffix}.mp3'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(start), '-i', str(audio),
            '-t', str(length), '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', str(path)], check=True)
        try:
            part = recognize(path)
            diagnostic = dict(start=start, end=start+length, whisper=part)
            fallback = unusable(part)
            if fallback:
                path.with_suffix('.json').write_text(json.dumps(diagnostic, ensure_ascii=False)+'\n')
            # Do not give the text-only transcript invented word times. Unresolved
            # lyrics use neighboring anchors in the later acoustic forced alignment.
            words = [] if fallback else [dict(w, start=w['start']+start, end=min(length, w['end'])+start)
                for w in part.get('words', []) if 0 <= w['start'] < length and w['end'] > w['start']]
            return words, dict(start=start, end=start+length, word_times_discarded=fallback,
                               language=song_language(part.get('language')), window_seconds=window)
        finally:
            path.unlink(missing_ok=True)

    starts = list(range(0, math.ceil(duration), 30))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(chunk, start) for start in starts]
        parts = [future.result() for future in progress_items(futures, 'chunks')]
        # Retry only failed windows, once, at half length. Keep good word times
        # and the full-recording text; never splice independently decoded text.
        retry_windows = []
        for _, record in parts:
            if record['word_times_discarded'] and record['end']-record['start'] > 15:
                # Balance the final partial window too, avoiding a tiny tail request.
                window = math.ceil((record['end']-record['start'])/2)
                retry_windows.extend((start, window)
                    for start in range(record['start'], math.ceil(record['end']), window))
        if retry_windows:
            report_progress(detail='缩短异常片段，重试词时间')
            futures = [pool.submit(chunk, start, window) for start, window in retry_windows]
            parts.extend(future.result() for future in progress_items(futures, 'chunks'))
    parts.sort(key=lambda part: part[1]['start'])
    languages = Counter()
    for words, record in parts:
        if record['language']:
            languages[record['language']] += sum(c.isalnum() for w in words for c in w['word'])
    detected = languages.most_common(1)[0][0] if languages.total() else None
    data['language'] = song_language(language) if language != 'auto' else detected
    words = [w for words, _ in parts for w in words]
    initial_text = data.get('text') or ' '.join(s.get('text', '') for s in data.get('segments') or [])
    text, selection = select_recovery_text(text, initial_text, words, data['language'])
    (output/'text-selection.json').write_text(json.dumps(selection, ensure_ascii=False)+'\n')
    if text is None:
        (output/'recovery-words.json').write_text(json.dumps(words, ensure_ascii=False)+'\n')
        if not allow_unresolved_text:
            raise ValueError('歌词文本遗漏了分段识别中的大量内容，无法生成完整歌词')
        # Keep independently decoded word evidence for supplied/source lyrics.
        # Neither rejected draft can become an ASR lyric fallback downstream.
        text = ''
    segments = [dict(start=0., end=duration, text=text, timing_source='recording_bounds')] if text else []
    data.update(duration=duration, text=text, segments=segments, words=words)
    return data, dict(recovery, triggered=True, text_model=fallback_model, language_evidence=dict(languages),
                      text_selection=selection, chunks=[record for _, record in parts])


def filter_silent_transcription(data, audio, sample_rate):
    """Reject ASR hallucinations in digital silence, preserving even quiet speech."""
    filtered, rejected = dict(data), []
    for kind in ('segments', 'words'):
        filtered[kind] = []
        for item in data.get(kind) or []:
            start, end = item['start'], item['end']
            if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start <= end:
                reason = 'invalid_time_range'
            else:
                # Include nearby audio for imprecise or zero-duration ASR times.
                left = max(0, math.floor((start-.25)*sample_rate))
                right = min(len(audio), math.ceil((end+.25)*sample_rate))
                reason = ('outside_audio' if right <= left else
                          'digital_silence' if not audio[left:right].any() else None)
            if reason:
                rejected.append(dict(kind=kind, reason=reason, item=item))
            else:
                filtered[kind].append(item)
    filtered['text'] = ''.join(item.get('text', '') for item in filtered['segments'])
    return filtered, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--language', default='auto')
    parser.add_argument('--hint', default='')
    parser.add_argument('--allow-unresolved-text', action='store_true',
        help='Retain word evidence for independent lyrics when both full-text drafts are rejected')
    parser.add_argument('--source-offset', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    client = OpenAI(timeout=240., max_retries=1)
    language = {'Chinese': 'zh', 'Japanese': 'ja', 'English': 'en'}.get(args.language, args.language)
    args.output.mkdir(parents=True)
    try:
        data, recovery = transcribe(client, args.audio, language, args.hint, args.output,
            allow_unresolved_text=args.allow_unresolved_text)
    except Exception as error:
        from job_schema import provider_error
        details = provider_error(error)
        if details:
            (args.output/'error.json').write_text(json.dumps(details)+'\n')
        raise
    for key in ['words', 'segments']:
        for item in data.get(key, []):
            item['start'] += args.source_offset
            item['end'] += args.source_offset
    (args.output/'transcription.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    metadata = {'service': 'OpenAI Audio Transcriptions API', 'model': 'whisper-1',
        'uploaded_audio': str(args.audio.resolve()),
        'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        'language': args.language, 'hint': args.hint, 'seconds': time.perf_counter()-started,
        'source_offset_seconds': args.source_offset,
        'recovery': recovery,
        'status': 'ASR draft; requires lyric verification and timing evaluation'}
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
