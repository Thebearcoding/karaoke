"""Offline Whisper inference with explicit, fully covered audio batches."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

if __package__:
    from .job_schema import language_code, song_language
    from .stage_cache import StageCache, digest
else:
    from job_schema import language_code, song_language
    from stage_cache import StageCache, digest


def whisper_language(value):
    """Resolve against the installed model's vocabulary, never default to English."""
    if value is None or value == 'auto':
        return None
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
    for name in (language_code(value), value, song_language(value)):
        key = str(name or '').lower()
        code = key if key in LANGUAGES else TO_LANGUAGE_CODE.get(key)
        if code in LANGUAGES:
            return code
    raise ValueError(f'当前 Whisper 模型不支持语言：{value}')


def offline():
    backend = os.getenv('KARAOKE_ASR_BACKEND', 'api')
    if backend not in ('api', 'local'):
        raise ValueError('KARAOKE_ASR_BACKEND must be api or local')
    return backend == 'local'


def configuration():
    if not offline():
        return dict(backend='api')
    model = Path(os.getenv('KARAOKE_WHISPER_MODEL',
        str(Path(__file__).resolve().parent/'models/alignment/faster-whisper-large-v3')))
    batch = int(os.getenv('KARAOKE_WHISPER_BATCH_SIZE', '8'))
    if not 1 <= batch <= 32:
        raise ValueError('KARAOKE_WHISPER_BATCH_SIZE must be between 1 and 32')
    files = [model/name for name in ('config.json', 'model.bin', 'tokenizer.json', 'preprocessor_config.json')]
    return dict(backend='local', model=str(model), batch_size=batch, compute_type='float32',
        model_files={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files},
        packages={name: version(name) for name in ('faster-whisper', 'ctranslate2')})


def offset_segments(segments, offset):
    """Translate decoder-local times without clipping words at chunk edges."""
    if not offset:
        return segments
    from copy import deepcopy
    result = deepcopy(segments)
    for segment in result:
        for unit in (segment, *(segment.get('words') or [])):
            for key in ('start', 'end'):
                if key in unit:
                    unit[key] += offset
        if 'seek' in segment:
            segment['seek'] += round(offset*100)
    return result


def transcribe(model, samples, language, batch_size):
    from faster_whisper import BatchedInferencePipeline
    if not len(samples):
        raise ValueError('Local Whisper input is empty')
    duration, window = len(samples)/16000, 30*16000
    detection = []
    if language is None or language == 'auto':
        language, detection = detect_audio_language(samples,
            lambda clip: model.detect_language(audio=clip, vad_filter=False)[:2])
    else:
        language = whisper_language(language)
    # Do not let a speech VAD remove quiet singing. Every sample belongs to a
    # window, including the final partial window. These are ASR bounds, not cues.
    windows = [dict(start=start/16000, end=min(start+window, len(samples))/16000)
               for start in range(0, len(samples), window)]
    # Disagreement is evidence of a possible language switch, not empty audio.
    # Decode the existing windows independently when one language cannot
    # describe them all. A single-language recording keeps its batched path.
    mixed = len({d['language'] for d in detection}) > 1
    requests = windows if mixed or language is None else [dict(start=0., end=duration)]
    segments, decoded_windows = [], []
    engine = BatchedInferencePipeline(model)
    for request in requests:
        start, end = request['start'], request['end']
        clip = samples if len(requests) == 1 else samples[round(start*16000):round(end*16000)]
        result, info = engine.transcribe(clip,
            language=None if mixed else language,
            batch_size=batch_size, beam_size=5, temperature=0, word_timestamps=True,
            without_timestamps=False, condition_on_previous_text=False,
            vad_filter=False, clip_timestamps=(windows if len(requests) == 1 else [dict(start=0., end=end-start)]))
        segments.extend(offset_segments([asdict(segment) for segment in result], start))
        decoded_windows.append(dict(start=start, end=end, language=getattr(info, 'language', language)))
    if mixed:
        language = None
    return dict(language=language, duration=duration, text=''.join(s['text'] for s in segments),
        segments=segments, words=[w for s in segments for w in s['words'] or []],
        local_inference=dict(language_detection=detection, windows=windows,
                             decoded_windows=decoded_windows, batch_size=batch_size, vad_filter=False))


def detect_audio_language(samples, detect):
    """Use interior audio samples so an intro cannot pick the whole song's language."""
    if not len(samples):
        raise ValueError('Language detection input is empty')
    window, detection = 30*16000, []
    starts = [0] if len(samples) <= window else sorted({
        round(max(0, min(len(samples)-window, len(samples)*f-window/2))) for f in (.25, .5, .75)})
    for start in starts:
        code, probability = detect(samples[start:start+window])
        detection.append(dict(start=start/16000, end=min(start+window,len(samples))/16000,
                              language=code, probability=probability))
    code, count = Counter(d['language'] for d in detection).most_common(1)[0]
    return (code if count > len(detection)/2 else None), detection


def detect_whisper_language(model, samples):
    import whisper
    def detect(clip):
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(clip), n_mels=model.dims.n_mels).to(model.device)
        _, probabilities = model.detect_language(mel)
        code = max(probabilities, key=probabilities.get)
        return code, probabilities[code]
    return detect_audio_language(samples, detect)


def alignment_language(model, samples, hint=None, *, detect=False):
    """Give the aligner's required tokenizer a hint from this audio crop."""
    try:
        code = whisper_language(hint)
    except ValueError:
        code = None  # A metadata label is not proof that the audio is unsupported.
    if code is not None and not detect:
        return code, []
    selected, probes = detect_whisper_language(model, samples)
    # A multilingual crop can lack a majority. The tokenizer still requires
    # one code; retain all observations and use its strongest audio hint.
    # This never removes text or establishes word-time accuracy.
    return selected or max(probes, key=lambda p:p['probability'])['language'], probes


def detect_singing_language(samples, rate, detect):
    """Keep sung-language evidence even when the event classifier says music."""
    if not len(samples):
        raise ValueError('Language detection input is empty')
    window = 20*rate
    starts = sorted({round(max(0, min(len(samples)-window, len(samples)*i/10-window/2)))
                     for i in range(1, 10)})
    probes = []
    for start in starts:
        result = detect(samples[start:start+window])
        code = result['lang'].removeprefix('<|').removesuffix('|>')
        probes.append(dict(start=start/rate, end=min(start+window, len(samples))/rate,
                           language=code, event=result['event']))
    votes = Counter(p['language'] for p in probes if p['language'] != 'nospeech')
    # Singing often gets generic zh even within a Cantonese recording. Keep
    # an explicit yue observation instead of letting that broad label erase it.
    if 'yue' in votes:
        votes['yue'] += votes.pop('zh', 0)
    selected = None
    if votes:
        code, count = votes.most_common(1)[0]
        if count > sum(votes.values())/2:
            selected = song_language(code)
    return selected, probes


def detect_recording_language(audio, output):
    """Use SenseVoice language labels only; never replace the supplied lyrics."""
    import time

    import sherpa_onnx
    import soundfile as sf
    started = time.perf_counter()
    weights = Path(__file__).resolve().parent/'models/alignment/sensevoice'
    model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(weights/'model.int8.onnx'), tokens=str(weights/'tokens.txt'),
        num_threads=4, language='auto', use_itn=False)
    samples, rate = sf.read(audio, dtype='float32', always_2d=True)
    def detect(clip):
        stream = model.create_stream()
        stream.accept_waveform(rate, clip)
        model.decode_stream(stream)
        return json.loads(str(stream.result))
    language, probes = detect_singing_language(samples.mean(axis=1), rate, detect)
    Path(output).write_text(json.dumps(dict(language=language, probes=probes,
        model='SenseVoiceSmall-int8', audio_sha256=digest(audio),
        seconds=time.perf_counter()-started), ensure_ascii=False)+'\n')


@contextmanager
def isolated_recognizer(config, device, scratch):
    """Keep CTranslate2's CUDA libraries out of the PyTorch alignment process."""
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='recognizer-', dir=scratch) as folder:
        audio = Path(folder)/'audio.npy'
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            '--worker', device, json.dumps(config)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1)
        assert process.stdin is not None and process.stdout is not None
        writer, reader = process.stdin, process.stdout
        try:
            def recognize(samples):
                np.save(audio, samples, allow_pickle=False)
                writer.write(str(audio)+'\n')
                writer.flush()
                response = reader.readline()
                if not response:
                    raise RuntimeError(f'Local recognizer exited ({process.wait()}); see stage log')
                data = json.loads(response)
                if 'error' in data:
                    raise ValueError(data['error'])
                return data
            yield recognize
        finally:
            writer.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.close()


def recognition_worker(device, config):
    import numpy as np
    from faster_whisper import WhisperModel
    model = WhisperModel(config['model'], device=device, compute_type=config['compute_type'],
                         cpu_threads=4, local_files_only=True)
    for path in sys.stdin:
        try:
            result = transcribe(model, np.load(path.strip(), allow_pickle=False), 'auto', config['batch_size'])
        except ValueError as error:
            result = dict(error=str(error))
        print(json.dumps(result, ensure_ascii=False), flush=True)


def cached_recognition(audio, language, output):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    config = dict(configuration(), language=language)
    model_path = Path(config['model'])
    key = hashlib.sha256((digest(audio)+json.dumps(config, sort_keys=True)).encode()).hexdigest()[:16]
    path = output.parent/'asr-requests'/('batch-'+key+'.json')
    path.parent.mkdir(exist_ok=True)
    with StageCache(output.parent).stage('batch_asr_'+key,
            inputs=[audio, *[model_path/name for name in config['model_files']]],
            outputs=[path], config=config, code=[Path(__file__)]) as reused:
        if not reused:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = WhisperModel(str(model_path), device=device, compute_type=config['compute_type'],
                                 cpu_threads=4, local_files_only=True)
            data = transcribe(model, decode_audio(str(audio), sampling_rate=16000), language, config['batch_size'])
            data['local_inference'].update(config, device=device)
            path.write_text(json.dumps(data, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--detect-language':
        detect_recording_language(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == '--worker':
        recognition_worker(sys.argv[2], json.loads(sys.argv[3]))
    else:
        raise SystemExit('Use --worker DEVICE CONFIG for isolated recognition')
