"""Recover missing coarse lyric anchors with bounded, cached ASR windows."""
import hashlib
import json
from pathlib import Path
import subprocess

from openai import OpenAI

from job_schema import song_language
from transcribe_lyrics import needs_recovery


def recover_anchor_words(client, audio, lines, words, language, output, cache, minimum_coverage, duration):
    """Re-recognize bounded gaps with neighboring sung context, retaining word times."""
    from anchors_from_asr import anchors_from_words

    def assess(candidate):
        cues = anchors_from_words(lines, candidate, language, allow_missing=True)
        score = sum(c['matched_fraction']*len(c['text']) for c in cues)/max(1, sum(map(len, lines)))
        return cues, score

    cues, score = assess(words)
    report = dict(initial_coverage=score, attempts=[])
    attempted = set()
    for _ in range(3):
        if score >= minimum_coverage:
            break
        windows, first = [], 0
        while first < len(cues):
            if cues[first]['status'] == 'candidate':
                first += 1
                continue
            stop = first+1
            while stop < len(cues) and cues[stop]['status'] != 'candidate':
                stop += 1
            before = [c for c in cues[:first] if c['status'] == 'candidate'][-2:]
            after = [c for c in cues[stop:] if c['status'] == 'candidate'][:2]
            if before and after:
                begin, end = max(0., before[0]['start']-.6), min(duration, after[-1]['end']+.6)
                window = (round(begin, 3), round(end, 3))
                if 0 < end-begin <= 45 and window not in attempted:
                    windows.append((sum(len(c['text']) for c in cues[first:stop]), window))
            first = stop
        if not windows:
            break
        _, (begin, end) = max(windows)
        attempted.add((begin, end))
        config = dict(begin=begin, end=end, language=language, model='whisper-1')
        key = 'anchor_asr_'+hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        response_path, clip = output.parent/(key+'.json'), output.parent/(key+'.mp3')
        attempt = dict(**config, response=response_path.name, accepted=False)
        report['attempts'].append(attempt)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        with cache.stage(key, inputs=[audio], outputs=[response_path], config=config,
                         code=[Path(__file__)]) as reused:
            if not reused:
                try:
                    if client is None:
                        client = OpenAI(timeout=60, max_retries=0)
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(begin),
                        '-i', str(audio), '-t', str(end-begin), '-ar', '16000', '-ac', '1',
                        '-c:a', 'libmp3lame', '-b:a', '64k', str(clip)], check=True)
                    with clip.open('rb') as stream:
                        data = client.audio.transcriptions.create(file=stream, model='whisper-1',
                            language={'Chinese':'zh', 'Japanese':'ja', 'English':'en'}[language],
                            temperature=0, response_format='verbose_json',
                            timestamp_granularities=['word', 'segment']).model_dump()
                    response_path.write_text(json.dumps(data, ensure_ascii=False)+'\n')
                finally:
                    clip.unlink(missing_ok=True)
        data = json.loads(response_path.read_text())
        accepted = False
        candidate_score = None
        if song_language(data.get('language')) == language and not needs_recovery(data):
            replacement = [dict(w, start=w['start']+begin, end=w['end']+begin)
                for w in data.get('words', []) if 0 <= w['start'] < w['end'] <= end-begin]
            candidate = [w for w in words if w['start'] < begin] + replacement + [w for w in words if w['start'] >= end]
            candidate_cues, candidate_score = assess(candidate)
            if replacement and candidate_score > score and all(
                    a['start'] <= b['start'] for a, b in zip(candidate, candidate[1:])):
                words, cues, score = candidate, candidate_cues, candidate_score
                accepted = True
        attempt.update(accepted=accepted, coverage=candidate_score, reused=reused)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    report['final_coverage'] = score
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return words, cues, score
