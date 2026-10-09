"""Offline Whisper inference with explicit, fully covered audio batches."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from .stage_cache import StageCache, digest
    from .job_schema import song_language
else:
    from stage_cache import StageCache, digest
    from job_schema import song_language


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


def transcribe(model, samples, language, batch_size):
    from faster_whisper import BatchedInferencePipeline
    if not len(samples):
        raise ValueError('Local Whisper input is empty')
    duration, window = len(samples)/16000, 30*16000
    detection = []
    if language == 'auto':
        language, detection = detect_audio_language(samples,
            lambda clip: model.detect_language(audio=clip, vad_filter=False)[:2])
    else:
        language = {'Chinese': 'zh', 'Japanese': 'ja', 'English': 'en'}[song_language(language)]
    # Do not let a speech VAD remove quiet singing. Every sample belongs to a
    # window, including the final partial window. These are ASR bounds, not cues.
    windows = [dict(start=start/16000, end=min(start+window, len(samples))/16000)
               for start in range(0, len(samples), window)]
    segments = []
    if language is not None:
        result, _ = BatchedInferencePipeline(model).transcribe(samples, language=language,
            batch_size=batch_size, beam_size=5, temperature=0, word_timestamps=True,
            without_timestamps=False, condition_on_previous_text=False,
            vad_filter=False, clip_timestamps=windows)
        segments = [asdict(segment) for segment in result]
    return dict(language=language, duration=duration, text=''.join(s['text'] for s in segments),
        segments=segments, words=[w for s in segments for w in s['words'] or []],
        local_inference=dict(language_detection=detection, windows=windows,
                             batch_size=batch_size, vad_filter=False))


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


@contextmanager
def isolated_recognizer(config, device, scratch):
    """Keep CTranslate2's CUDA libraries out of the PyTorch alignment process."""
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='recognizer-', dir=scratch) as folder:
        audio = Path(folder)/'audio.npy'
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            '--worker', device, json.dumps(config)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            def recognize(samples):
                np.save(audio, samples, allow_pickle=False)
                process.stdin.write(str(audio)+'\n')
                process.stdin.flush()
                response = process.stdout.readline()
                if not response:
                    raise RuntimeError(f'Local recognizer exited ({process.wait()}); see stage log')
                data = json.loads(response)
                if 'error' in data:
                    raise ValueError(data['error'])
                return data
            yield recognize
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


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
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit('Use --worker DEVICE CONFIG for isolated recognition')
    recognition_worker(sys.argv[2], json.loads(sys.argv[3]))
