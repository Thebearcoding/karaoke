"""Offline Whisper inference with explicit, fully covered audio batches."""
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path

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


def transcribe(model, samples, language, batch_size, *, contextual=False):
    from faster_whisper import BatchedInferencePipeline
    if not len(samples):
        raise ValueError('Local Whisper input is empty')
    duration, window = len(samples)/16000, 30*16000
    detection = []
    if language == 'auto':
        starts = [0] if len(samples) <= window else sorted({
            round(max(0, min(len(samples)-window, len(samples)*f-window/2))) for f in (.25, .5, .75)})
        for start in starts:
            code, probability, _ = model.detect_language(audio=samples[start:start+window], vad_filter=False)
            detection.append(dict(start=start/16000, language=code, probability=probability))
        code, count = Counter(d['language'] for d in detection).most_common(1)[0]
        language = code if count > len(detection)/2 else None
    else:
        language = {'Chinese': 'zh', 'Japanese': 'ja', 'English': 'en'}[song_language(language)]
    # Do not let a speech VAD remove quiet singing. Every sample belongs to a
    # window, including the final partial window. These are ASR bounds, not cues.
    windows = [dict(start=start/16000, end=min(start+window, len(samples))/16000)
               for start in range(0, len(samples), window)]
    if contextual:
        windows = [dict(start=0., end=duration)]
    segments = []
    if language is not None:
        decoder = model if contextual else BatchedInferencePipeline(model)
        options = {} if contextual else dict(batch_size=batch_size, clip_timestamps=windows)
        result, _ = decoder.transcribe(samples, language=language,
            beam_size=5, temperature=0, word_timestamps=True,
            without_timestamps=False, condition_on_previous_text=contextual,
            vad_filter=False, **options)
        segments = [asdict(segment) for segment in result]
    return dict(language=language, duration=duration, text=''.join(s['text'] for s in segments),
        segments=segments, words=[w for s in segments for w in s['words'] or []],
        local_inference=dict(language_detection=detection, windows=windows,
                             batch_size=1 if contextual else batch_size, vad_filter=False,
                             contextual=contextual))


def cached_recognition(audio, language, output, *, contextual=False):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    config = dict(configuration(), language=language)
    if contextual:
        config.update(batch_size=1, contextual=True)
    model_path = Path(config['model'])
    key = hashlib.sha256((digest(audio)+json.dumps(config, sort_keys=True)).encode()).hexdigest()[:16]
    mode = 'context' if contextual else 'batch'
    path = output.parent/'asr-requests'/(mode+'-'+key+'.json')
    path.parent.mkdir(exist_ok=True)
    with StageCache(output.parent).stage(mode+'_asr_'+key,
            inputs=[audio, *[model_path/name for name in config['model_files']]],
            outputs=[path], config=config, code=[Path(__file__)]) as reused:
        if not reused:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = WhisperModel(str(model_path), device=device, compute_type=config['compute_type'],
                                 cpu_threads=4, local_files_only=True)
            data = transcribe(model, decode_audio(str(audio), sampling_rate=16000), language,
                              config['batch_size'], contextual=contextual)
            data['local_inference'].update(config, device=device)
            path.write_text(json.dumps(data, ensure_ascii=False)+'\n')
    return json.loads(path.read_text())
