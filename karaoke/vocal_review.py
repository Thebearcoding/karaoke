"""Review uncovered speech without treating ASR words as proof of singing."""
import base64
import io
import json
import hashlib
from pathlib import Path

from .stage_cache import StageCache, digest

MODEL = 'gpt-audio-1.5'
PROMPT = ('Classify the audible human vocal sounds in this audio clip. Singing includes sustained melodic vowels, '
          'humming, backing vocals, and rap; non-lexical singing still counts. Spoken dialogue or narration with '
          'background music is speech, not singing. Do not infer any lyrics from a song you recognize. Return only '
          'a JSON object with singing_present (boolean), speech_present (boolean), uncertain_vocals (boolean), '
          'and explanation (short string). Do not provide timestamps. Set uncertain_vocals true if the distinction '
          'or audibility cannot be established.')
SOURCES = ('source-44100.wav', 'music-separation/song/vocals.wav')


def decision(response):
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
        raise ValueError('Audio review did not provide a complete decision')
    text = choice['message']['content'].strip().removeprefix('```json').removesuffix('```').strip()
    result = json.loads(text)
    if not isinstance(result, dict) or any(type(result.get(k)) is not bool
            for k in ('singing_present', 'speech_present', 'uncertain_vocals')):
        raise ValueError('Audio review has no valid vocal classification')
    return result


def review_clip(path, start, end):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(path)
    if not 0 <= start < end <= info.duration+.05:
        raise ValueError('Audio review is outside the source recording')
    wave, rate = sf.read(path, start=round(start*info.samplerate), stop=round(end*info.samplerate),
                         dtype='float32', always_2d=True)
    divisor = np.gcd(rate, 16000)
    wave = resample_poly(wave.mean(axis=1), 16000//divisor, rate//divisor)
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='PCM_16')
    return buffer.getvalue()


def review_gaps(folder, gaps, words, *, client=None):
    """Cache both views; only review the actual uncovered words, at most 90 s."""
    from openai import OpenAI
    folder = Path(folder)
    cache, reviews = StageCache(folder), []
    if not all((folder/source).is_file() for source in SOURCES):
        return reviews  # No audio evidence means no exemption from coverage.
    for gap in gaps:
        uncovered = [w for w in words if gap['start'] <= w['start'] < w['end'] <= gap['end']]
        if not uncovered:
            continue
        start = max(gap['start'], min(w['start'] for w in uncovered)-.5)
        end = min(gap['end'], max(w['end'] for w in uncovered)+.5)
        if end-start > 90:
            continue  # A long unverified passage still fails the original gate.
        views = []
        for source in SOURCES:
            path = folder/source
            config = dict(model=MODEL, prompt=PROMPT, start=start, end=end)
            key = hashlib.sha256((source+json.dumps(config, sort_keys=True)).encode()).hexdigest()[:20]
            output = folder/'vocal-reviews'/(key+'.json')
            output.parent.mkdir(exist_ok=True)
            with cache.stage('vocal_review_'+key, inputs=[path], outputs=[output], config=config,
                             code=[Path(__file__)]) as reused:
                if not reused:
                    client = client or OpenAI(timeout=90, max_retries=0)
                    audio = review_clip(path, start, end)
                    response = client.chat.completions.create(model=MODEL, modalities=['text'], temperature=0,
                        max_completion_tokens=500, messages=[dict(role='user', content=[dict(type='text', text=PROMPT),
                        dict(type='input_audio', input_audio=dict(data=base64.b64encode(audio).decode(), format='wav'))])])
                    record = dict(config, source=source, source_sha256=digest(path),
                                  input_sha256=hashlib.sha256(audio).hexdigest(), response=response.model_dump(mode='json'))
                    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
            views.append(json.loads(output.read_text()))
        reviews.append(dict(start=start, end=end, views=views))
    return reviews


def speech_intervals(folder, reviews):
    """Recheck decisions and their exact audio inputs before excluding any words."""
    folder, intervals, hashes = Path(folder), [], {}
    for review in reviews:
        start, end = review['start'], review['end']
        views = review['views']
        if [v['source'] for v in views] != list(SOURCES) or not 0 <= start < end or end-start > 90:
            raise ValueError('Audio review is missing a source view or has invalid bounds')
        accepted = True
        for view in views:
            path = folder/view['source']
            if path not in hashes:
                hashes[path] = digest(path)
            if (view['source_sha256'] != hashes[path] or view['model'] != MODEL or view['prompt'] != PROMPT
                    or (view['start'], view['end']) != (start, end)
                    or view['input_sha256'] != hashlib.sha256(review_clip(path, start, end)).hexdigest()):
                raise ValueError('Audio review does not match its current input')
            try:
                result = decision(view['response'])
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                accepted = False
                continue
            accepted &= result['speech_present'] and not result['singing_present'] and not result['uncertain_vocals']
        if accepted:
            intervals.append((start, end))
    return intervals
