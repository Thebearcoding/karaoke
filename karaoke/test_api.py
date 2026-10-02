import hashlib
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urljoin
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from karaoke.api import Store, create_app
from karaoke.exact_stems import validate_preview
from karaoke.job_schema import JobRequest, video_url
from karaoke.song_sources import SongText, find_verified_lyrics, prepare_song_text, video_song_info
from karaoke.align_lyric_phonemes import protect_line_order
from karaoke.cache import CacheFull, GIB, can_restore_video
from karaoke.pipeline import Pipeline, StageFailure, create_poster, file_hash, processing_error
from karaoke.stage_progress import PREFIX, parse_progress, progress_items
from karaoke.transcribe_lyrics import repetition_loop, transcribe


URL = 'https://www.youtube.com/watch?v=n5sjpCQhTDk'


def fake_worker(folder):
    code = '''import hashlib,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1])
(p/'progress.json').write_text(json.dumps({'active_stages':['render'],'timings':{}}))
time.sleep(.1)
artifacts={}
for name,filename in [('mv','mv.mp4'),('instrumental','instrumental.wav'),('backing_vocals','backing.wav'),('lead_vocals','lead.wav')]:
    data=b'0123456789';(p/filename).write_bytes(data)
    artifacts[name]={'path':filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
(p/'result.json').write_text(json.dumps({'artifacts':artifacts}))
'''
    return [sys.executable, '-c', code, str(folder)]


def finished(client, job_id):
    deadline = time.monotonic()+8
    while time.monotonic() < deadline:
        job = client.get(f'/v1/jobs/{job_id}').json()
        if job['status'] in ('completed', 'failed'):
            return job
        time.sleep(.03)
    raise AssertionError('Worker did not finish')


@pytest.mark.parametrize('value,expected', [
    ('https://youtu.be/n5sjpCQhTDk?t=30', URL),
    ('https://www.youtube.com/watch?v=n5sjpCQhTDk&list=anything', URL),
    ('https://m.youtube.com/shorts/n5sjpCQhTDk', URL),
    ('https://www.bilibili.com/video/BV1bK411W797/?p=2&share_source=copy',
     'https://www.bilibili.com/video/BV1bK411W797?p=2'),
    ('http://www.bilibili.com/video/av1074402/', 'https://www.bilibili.com/video/av1074402'),
    ('https://b23.tv/abcd123', 'https://b23.tv/abcd123'),
])
def test_video_identity(value, expected):
    assert video_url(value) == expected


@pytest.mark.parametrize('value', [
    'file:///etc/passwd', 'http://127.0.0.1/private', 'https://youtube.com.attacker.test/watch?v=n5sjpCQhTDk',
    'https://user:secret@youtube.com/watch?v=n5sjpCQhTDk',
    'https://youtube.com:9000/watch?v=n5sjpCQhTDk', 'https://youtube.com/playlist?list=foo',
    'https://bilibili.com/video/BV1bK411W797?p=../../secret',
    'https://bilibili.com/video/../../../secret',
])
def test_reject_non_video_or_local_targets(value):
    with pytest.raises(ValueError):
        video_url(value)


def test_submission_outputs_cache_and_range_requests(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        response = client.post('/v1/jobs', json={'url': URL})
        assert response.status_code == 202
        job_id = response.json()['id']
        duplicate = client.post('/v1/jobs', json={'url': 'https://youtu.be/n5sjpCQhTDk'})
        assert duplicate.json()['id'] == job_id
        assert duplicate.json()['cache_hit'] is True
        assert client.get(f'/v1/jobs/{job_id}/artifacts/mv').status_code == 409
        job = finished(client, job_id)
        assert job['status'] == 'completed'
        artifact = job['result']['artifacts']['mv']
        assert 'path' not in artifact and artifact['url'].endswith('/artifacts/mv')
        response = client.get(artifact['url'], headers={'Range': 'bytes=2-5'})
        assert response.status_code == 206 and response.content == b'2345'
        assert client.get(f'/v1/jobs/{job_id}/artifacts/request.json').status_code == 404
        cached = client.post('/v1/jobs', json={'url': URL})
        assert cached.status_code == 200 and cached.json()['id'] == job_id
        (tmp_path/job_id/'mv.mp4').unlink()
        repaired = client.post('/v1/jobs', json={'url': URL})
        assert repaired.status_code == 202 and repaired.json()['id'] != job_id
        assert finished(client, repaired.json()['id'])['status'] == 'completed'


def test_failures_are_visible_and_retry_creates_new_job(tmp_path):
    def failed_worker(folder):
        return [sys.executable, '-c',
            "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);(p/'error.json').write_text(json.dumps({'code':'download_failed','message':'unavailable'}));sys.exit(1)", str(folder)]
    with TestClient(create_app(tmp_path, worker_command=failed_worker, version='test')) as client:
        first = client.post('/v1/jobs', json={'url': URL}).json()
        job = finished(client, first['id'])
        assert job['status'] == 'failed' and job['error']['code'] == 'download_failed'
        assert client.get(f"/v1/jobs/{first['id']}/artifacts/mv").status_code == 409
        second = client.post('/v1/jobs', json={'url': URL}).json()
        assert second['id'] != first['id']
        finished(client, second['id'])


@pytest.mark.parametrize('rejected', [False, True])
def test_external_failures_retain_safe_causes_and_do_not_retry_as_software_bugs(tmp_path, rejected):
    import httpx
    from openai import APITimeoutError, ContentFilterFinishReasonError
    from karaoke.tools.soak import eligible, record_failure

    error = (ContentFilterFinishReasonError() if rejected else
             APITimeoutError(request=httpx.Request('POST', 'https://api.openai.com/')))
    direct = processing_error(error, tmp_path)
    assert direct['provider'] == 'openai' and direct['retryable'] is not rejected
    log = f'openai.{type(error).__name__}: private response and credentials\n'
    (tmp_path/'transcribe.log').write_text(log)
    wrapped = processing_error(StageFailure('transcribe', 'stage failed'), tmp_path)
    assert wrapped == direct and 'private' not in str(wrapped)
    assert processing_error(RuntimeError('Request timed out'), tmp_path)['code'] == 'processing_failed'

    attempt = dict(pipeline_version='v1', external_attempt=1)
    legacy = dict(id=tmp_path.name, error=dict(code='transcribe_failed', message='Generic stage failure'))
    record_failure(attempt, legacy, tmp_path)
    assert attempt['error'] == direct and attempt['retry_provider'] == 'openai'
    assert not attempt['provider_blocked']
    case = {'attempts': [attempt]}
    assert not eligible(case, 'v2', time.time())
    if rejected:
        assert attempt['status'] == 'blocked' and attempt['next_retry_at'] is None
        assert not eligible(case, 'v3', time.time()+86400)
    else:
        assert attempt['status'] == 'external' and time.time()+20 < attempt['next_retry_at'] <= time.time()+30
        assert eligible(case, 'v2', attempt['next_retry_at']+1)
        attempt['external_attempt'] = 3
        assert not eligible(case, 'v3', attempt['next_retry_at']+1)


@pytest.mark.parametrize('message,headers,code,delay', [
    ('Rate limit reached for gpt-5-mini in project private. Please try again in 6ms.', {}, 'rate_limit', .006),
    ('Rate limit reached for whisper-1 in project private.', {'retry-after':'120'}, 'rate_limit', 120),
    ('Rate limit reached for whisper-1 in project private.', {'retry-after-ms':'7500'}, 'rate_limit', 7.5),
    ('Request too large for gpt-5-mini in project private.', {}, 'request_too_large', None),
    ('insufficient_quota for gpt-5-mini in project private.', {}, 'quota', None),
])
def test_rate_limit_uses_server_hint_and_distinguishes_request_errors(message, headers, code, delay):
    import httpx
    from openai import RateLimitError
    from karaoke.job_schema import provider_error
    from karaoke.tools.soak import retry_fields, retry_deadline
    response=httpx.Response(429, headers=headers, request=httpx.Request('POST','https://api.openai.com/'))
    error=provider_error(RateLimitError(message,response=response,body={'message':message}))
    assert error['code']=='openai_'+code and error.get('retry_after_seconds')==delay
    assert 'private' not in str(error)
    policy=retry_fields(error,'',1,failed_at=1000)
    if code=='rate_limit':
        assert policy['next_retry_at']==1000+max(30,delay)
        state={'provider_backoff':{policy['retry_provider']:policy['next_retry_at']}}
        assert retry_deadline(state,{'provider':'youtube'})==(policy['next_retry_at'] if error['model']=='whisper-1' else 0)
    else:
        assert policy['status']=='blocked' and policy['next_retry_at'] is None


@pytest.mark.parametrize('diagnostic,status,global_limit', [
    ('HTTP Error 403: Forbidden','external',False),
    ('File "worker.py", line 429\nHTTP Error 403: Forbidden','external',False),
    ('HTTP Error 429: Too Many Requests','external',True),
    ('Failed to resolve www.youtube.com','external',False),
    ('Sign in to confirm you are not a bot','blocked',False),
    ('HTTP Error 429: Too Many Requests\nTry again in 120 seconds\nSign in to confirm you are not a bot','blocked',True),
])
def test_video_retry_does_not_apply_one_resource_error_to_every_song(diagnostic,status,global_limit):
    from karaoke.tools.soak import retry_fields
    policy=retry_fields({'code':'download_failed'},diagnostic,1,failed_at=1000)
    assert policy['status']==status and policy['provider_blocked']==global_limit
    assert policy['next_retry_at'] is None if status=='blocked' and not global_limit else 1000<policy['next_retry_at']<=1300
    if 'Try again in 120 seconds' in diagnostic:
        assert policy['next_retry_at']==1120


def test_asr_child_preserves_server_retry_header_for_parent(tmp_path, monkeypatch):
    import httpx
    from openai import RateLimitError
    from karaoke import transcribe_lyrics
    response=httpx.Response(429,headers={'Retry-After':'95'},request=httpx.Request('POST','https://api.openai.com/'))
    error=RateLimitError('Rate limit reached',response=response,body=None)
    monkeypatch.setattr(sys,'argv',['transcribe_lyrics','--audio',str(tmp_path/'audio.mp3'),'--output',str(tmp_path/'asr')])
    with patch.object(transcribe_lyrics,'OpenAI'), patch.object(transcribe_lyrics,'transcribe',side_effect=error):
        with pytest.raises(RateLimitError):
            transcribe_lyrics.main()
    result=processing_error(StageFailure('transcribe','stage failed'),tmp_path)
    assert result['retry_after_seconds']==95 and result['code']=='openai_rate_limit'


def test_soak_preserves_failed_stage_diagnostics_after_cache_eviction(tmp_path):
    import gzip
    import shutil
    from karaoke.tools.soak import record_failure

    folder = tmp_path/'job'
    folder.mkdir()
    (folder/'worker.log').write_text('StageFailure: align_repeats failed\n')
    traceback = 'Traceback:\nParameterError: No valid sub-sequence warping path\n'
    (folder/'align_repeats.log').write_text('old progress\n'*4000+traceback)
    (folder/'asr').mkdir()
    (folder/'asr/transcription.json').write_text('{"language": "chinese"}')
    (folder/'asr-requests').mkdir()
    rejected = {'language': 'zh', 'words': [{'word': 'bad draft', 'start': 1, 'end': 1}]}
    (folder/'asr-requests/local-rejected.json').write_text(json.dumps(rejected))
    (folder/'repeated-lyrics').mkdir()
    (folder/'repeated-lyrics/run.json').write_text('{"attempts": []}')
    final_validation = {
        'cues.json': [dict(text='final cue', start=2., end=3.)],
        'audio-result.json': dict(duration_seconds=4., artifacts={'subtitles': {'path': 'cues.json'}}),
        'result.json': dict(duration_seconds=4., artifacts={'mv': {'path': 'mv.mp4'}}),
        'lyric-coverage-validation.json': dict(status='incomplete', uncovered_passages=[dict(start=0., end=2.)]),
    }
    for name, value in final_validation.items():
        (folder/name).write_text(json.dumps(value))
    (folder/'vocals.mp3').write_bytes(b'private audio')
    attempt = dict(external_attempt=1)
    error = dict(code='align_repeats_failed', message='Generic stage failure')
    record_failure(attempt, dict(id='job', error=error), folder)
    (folder/'repeated-lyrics/run.json').write_text('{"attempts": [{"reason": "changed alignment"}]}')
    retry = dict(external_attempt=1)
    record_failure(retry, dict(id='job', error=error), folder)
    shutil.rmtree(folder)

    assert attempt['error_fingerprint'] == retry['error_fingerprint']
    assert attempt['reproduction_file'] != retry['reproduction_file']
    with gzip.open(retry['reproduction_file'], 'rt') as stream:
        assert json.load(stream)['repeated-lyrics/run.json'] == {'attempts': [{'reason': 'changed alignment'}]}
    with gzip.open(attempt['reproduction_file'], 'rt') as stream:
        saved = json.load(stream)
    assert saved['align_repeats.log'].endswith(traceback)
    assert len(saved['align_repeats.log'].encode()) <= 16384
    assert saved['asr/transcription.json'] == {'language': 'chinese'}
    assert saved['asr-requests/local-rejected.json'] == rejected
    assert saved['repeated-lyrics/run.json'] == {'attempts': []}
    assert {name: saved[name] for name in final_validation} == final_validation
    assert 'vocals.mp3' not in saved
    assert traceback in attempt['diagnostic_tail']
    assert attempt['status'] == 'failed' and attempt['error'] == error


@pytest.mark.parametrize('code,diagnostic,external', [
    ('interrupted', '', True),
    ('align_phonemes_failed', 'ImportError: dlopen(/tmp/module.so): system is shutting down', True),
    ('align_phonemes_failed', 'ImportError: dlopen(/tmp/module.so): invalid code signature', False),
    ('processing_failed', 'Lyric: system is shutting down', False),
])
def test_soak_host_shutdown_waits_before_retry_without_blocking_provider(tmp_path, code, diagnostic, external):
    from karaoke.tools.soak import eligible, record_failure
    (tmp_path/'worker.log').write_text(diagnostic)
    attempt = dict(pipeline_version='v1', external_attempt=1)
    error = dict(code=code, message='Original failure')
    record_failure(attempt, dict(id=tmp_path.name, error=error), tmp_path)
    assert attempt['error'] == error
    assert attempt['status'] == ('external' if external else 'failed')
    assert not eligible({'attempts': [attempt]}, 'v1', time.time())
    if external:
        assert attempt['retry_provider'] == 'local' and not attempt['provider_blocked']
        assert time.time()+50 < attempt['next_retry_at'] < time.time()+70
        assert eligible({'attempts': [attempt]}, 'v1', attempt['next_retry_at']+1)


def test_retry_preserves_completed_stages_and_only_reruns_affected_work(tmp_path):
    from karaoke.stage_cache import StageCache
    store = Store(tmp_path)
    job, _ = store.create(JobRequest(url=URL), 'v1')
    folder = store.root/job['id']
    source, separated, aligned = [folder/name for name in ('source.wav', 'separated.wav', 'cues.json')]
    source.write_bytes(b'source')
    cache = StageCache(folder)
    with cache.stage('decode', inputs=[folder/'request.json'], outputs=[source]) as hit:
        assert not hit
        source.write_bytes(b'source')
    with cache.stage('separate', inputs=[source], outputs=[separated], config={'model': 'v1'}) as hit:
        assert not hit
        separated.write_bytes(b'expensive output')
    with pytest.raises(ValueError, match='broken alignment'):
        with cache.stage('align', inputs=[separated], outputs=[aligned]):
            aligned.write_text('partial')
            raise ValueError('broken alignment')
    store.cache.compact(store.update(job['id'], status='failed'))
    assert separated.read_bytes() == b'expensive output'
    assert not (folder/'checkpoints/align.json').exists()
    retried, cached = store.create(JobRequest(url=URL), 'alignment-fix')
    assert retried['id'] == job['id'] and not cached
    assert retried['pipeline_version'] == 'alignment-fix'
    resumed = StageCache(folder)
    with resumed.stage('separate', inputs=[source], outputs=[separated], config={'model': 'v1'}) as hit:
        assert hit  # A downstream algorithm update did not discard source/separation.
    with resumed.stage('align', inputs=[separated], outputs=[aligned], config={'algorithm': 'v2'}) as hit:
        assert not hit and not aligned.exists()
        aligned.write_text('complete')
    with resumed.stage('align', inputs=[separated], outputs=[aligned], config={'algorithm': 'v2'}) as hit:
        assert hit
    separated.write_bytes(b'corrupt')
    with resumed.stage('separate', inputs=[source], outputs=[separated], config={'model': 'v1'}) as hit:
        assert not hit and not separated.exists()
        separated.write_bytes(b'repaired output')
    with resumed.stage('align', inputs=[separated], outputs=[aligned], config={'algorithm': 'v2'}) as hit:
        assert not hit  # Changed upstream content invalidates its dependents.
        aligned.write_text('realigned')
    forced, _ = store.create(JobRequest(url=URL, force=True), 'alignment-fix')
    assert forced['id'] != job['id']


def test_checkpoint_directory_can_gain_downstream_diagnostics_and_respects_budget(tmp_path):
    from karaoke.stage_cache import StageCache
    store = Store(tmp_path/'cache')
    job, _ = store.create(JobRequest(url=URL), 'v1')
    folder = store.root/job['id']
    output = folder/'asr'
    with StageCache(folder).stage('transcribe', inputs=[folder/'request.json'], outputs=[output]):
        output.mkdir()
        (output/'transcription.json').write_text('{"text":"hello"}')
    (output/'filtered.json').write_text('{"text":"hello"}')
    with StageCache(folder).stage('transcribe', inputs=[folder/'request.json'], outputs=[output]) as hit:
        assert hit
    store.cache.compact(store.update(job['id'], status='failed'))
    store.cache.enforce()
    assert store.cache.stats['used_bytes'] >= (output/'transcription.json').stat().st_size
    store.cache.set_limit(1)
    assert not folder.exists()  # Checkpoints never escape the shared LRU budget.


def test_checkpoint_invalidates_changed_algorithm_without_discarding_inputs(tmp_path):
    from karaoke.stage_cache import StageCache
    source, output, code = [tmp_path/name for name in ('source', 'output', 'algorithm.py')]
    source.write_bytes(b'original')
    code.write_text('version = 1')
    cache = StageCache(tmp_path)
    for expected in (False, True, False):
        with cache.stage('test', inputs=[source], outputs=[output], code=[code]) as hit:
            assert hit is expected
            if not hit:
                assert not output.exists() and source.read_bytes() == b'original'
                output.write_bytes(b'result')
        if expected:
            code.write_text('version = 2')


def test_pipeline_reuses_decoded_audio_and_rebuilds_corrupted_stage(tmp_path):
    import soundfile as sf
    pipeline = Pipeline(dict(url=URL), tmp_path)
    pipeline.source.parent.mkdir()
    sf.write(pipeline.source, np.zeros((800, 2)), 8000, format='WAV', subtype='PCM_16')
    output = tmp_path/'source-44100.wav'
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', pipeline.source,
               '-ar', '44100', '-ac', '2', '-c:a', 'pcm_f32le', output]
    pipeline.run('decode_audio', command)
    expected = output.read_bytes()
    with patch.object(pipeline, 'run_process', side_effect=AssertionError('Repeated expensive stage')):
        pipeline.run('decode_audio', command)
    assert 'decode_audio' in pipeline.checkpoints.reused
    output.write_bytes(b'truncated')
    pipeline.run('decode_audio', command)
    assert output.read_bytes() == expected


def test_alignment_cache_ignores_unrelated_model_changes(tmp_path, monkeypatch):
    from karaoke import pipeline as module
    code, folder = tmp_path/'code', tmp_path/'job'
    code.mkdir()
    monkeypatch.setattr(module, 'ROOT', code)
    pipeline = Pipeline(dict(url=URL), folder)
    (folder/'lyrics.json').write_text(json.dumps(dict(language='Chinese')))
    stages = {
        'align_words': [sys.executable, code/'align_lyrics_whisper.py'],
        'align_phonemes': [sys.executable, code/'align_lyric_phonemes.py'],
        'transcribe': [sys.executable, code/'transcribe_lyrics.py',
                      '--lead-audio', folder/'lead-separation/song/vocals.wav'],
        'align_context': [sys.executable, code/'experiments/cjk-onsets-v003/run.py'],
    }
    models = [code/'models/alignment'/name for name in (
        'whisper/large-v3-turbo.pt', 'sofa/mandarin/model.ckpt', 'sofa/english/model.ckpt',
        'mms/model.pt', 'qwen3-forced-aligner-0.6b/model.safetensors')]
    for model in models:
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b'original model')
    for stage, command in stages.items():
        spec = pipeline.stage_spec(stage, command)
        for path in spec['inputs'] + spec['code']:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'input')
    calls = []
    def run(stage, command, timeout):
        calls.append(stage)
        output = pipeline.stage_spec(stage, command)['outputs'][0]
        output.mkdir()
        (output/'result.json').write_text('{}')
    monkeypatch.setattr(pipeline, 'run_process', run)
    def prepare():
        for stage, command in stages.items():
            pipeline.run(stage, command)
    prepare()
    (code/'subtitle-styles.json').write_text('{}')
    fingerprint = module.pipeline_fingerprint()
    unrelated = code/'models/alignment/experimental/model.pt'
    unrelated.parent.mkdir()
    unrelated.write_bytes(b'new experimental model')
    assert module.pipeline_fingerprint() == fingerprint
    models[2].write_bytes(b'updated English model')
    assert module.pipeline_fingerprint() != fingerprint
    prepare()
    assert calls == list(stages)
    calls.clear()
    fingerprint = module.pipeline_fingerprint()
    models[3].write_bytes(b'updated independent alignment model')
    assert module.pipeline_fingerprint() != fingerprint
    prepare()
    assert calls == ['align_context']
    calls.clear()
    fingerprint = module.pipeline_fingerprint()
    models[4].write_bytes(b'updated optional verification model')
    assert module.pipeline_fingerprint() != fingerprint
    prepare()
    assert calls == ['align_context']
    calls.clear()
    monkeypatch.setenv('KARAOKE_QWEN_PYTHON', '/different/isolated/python')
    prepare()
    assert calls == ['align_context']
    calls.clear()
    models[1].write_bytes(b'updated Mandarin model')
    prepare()
    assert calls == ['align_phonemes', 'align_context']
    calls.clear()
    models[0].write_bytes(b'updated Whisper model')
    prepare()
    assert calls == ['align_words', 'transcribe']
    calls.clear()
    (folder/'source-44100.wav').write_bytes(b'updated original audio')
    prepare()
    assert calls == ['transcribe', 'align_context']
    calls.clear()
    (folder/'lead-separation/song/vocals.wav').write_bytes(b'updated lead audio')
    prepare()
    assert calls == ['transcribe']
    # An English phrase makes the English pronunciation assets relevant only
    # to the final cross-model repair, not to prior separation or recognition.
    (folder/'lyrics.json').write_text(json.dumps(dict(language='Japanese', lines=['go home now'])))
    dictionary = code/'models/alignment/sofa/english/dict.txt'
    dictionary.write_text('go G OW\n')
    g2p = code/'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py'
    g2p.parent.mkdir(parents=True)
    g2p.write_text('pronunciation = 1\n')
    (code/'align_english_sofa.py').write_text('aligner = 1\n')
    prepare()
    calls.clear()
    for asset in (dictionary, g2p, models[3]):
        fingerprint = module.pipeline_fingerprint()
        asset.write_bytes(asset.read_bytes()+b'updated')
        assert module.pipeline_fingerprint() != fingerprint
        prepare()
        assert calls == ['align_context']
        calls.clear()
    fingerprint = module.pipeline_fingerprint()
    (dictionary.parent/'README.txt').write_text('Documentation only')
    assert module.pipeline_fingerprint() == fingerprint
    prepare()
    assert calls == []


@pytest.mark.parametrize('formatting', ['lyrics', 'symbols', 'annotations', 'unavailable'])
def test_manual_source_captions_are_cached_and_replace_an_asr_draft(tmp_path, monkeypatch, formatting):
    import httpx
    import soundfile as sf
    from karaoke import source_captions, pipeline as module
    pipeline = Pipeline(dict(url=URL), tmp_path)
    metadata = dict(id='n5sjpCQhTDk', subtitles={'zh-Hant': [dict(ext='json3', url=
        'https://www.youtube.com/api/timedtext?v=n5sjpCQhTDk&lang=zh-Hant&fmt=json3')]})
    info = tmp_path/'info.json'
    info.write_text(json.dumps(metadata))
    lines = ['微風裡慢慢歌唱', '星光下靜靜傾聽', '微風裡慢慢歌唱']
    displayed = ['♪', lines[0], '♪♪♪', lines[1], lines[2], '…'] if formatting == 'symbols' else list(lines)
    expected = ['♪'+lines[0]+'♪♪♪', lines[1], lines[2]+'…'] if formatting == 'symbols' else lines
    if formatting == 'annotations':
        displayed[1] = '[Singer] '+displayed[1]
    document = dict(events=[dict(tStartMs=1000*i, dDurationMs=900, segs=[dict(utf8=t)])
                            for i,t in enumerate(displayed)])
    calls = []
    def fetch(url):
        calls.append(url)
        if formatting == 'unavailable':
            httpx.Response(404, request=httpx.Request('GET', url)).raise_for_status()
        return document
    monkeypatch.setattr(source_captions, 'read_document', fetch)
    path = source_captions.fetch_manual_captions(info, 'Chinese', pipeline.checkpoints)
    assert source_captions.fetch_manual_captions(info, 'Chinese', pipeline.checkpoints) == path
    assert len(calls) == 1
    # Captions are displayed much earlier than these measured word times.
    # Their display timestamps must not become the alignment evidence.
    words = [dict(word=t, start=10+i, end=10+i+.9) for i,t in enumerate(lines)]
    transcription = dict(language='chinese', segments=[dict(text=w['word'], start=w['start'], end=w['end'])
                         for w in words], words=words)
    (tmp_path/'asr').mkdir()
    (tmp_path/'asr/transcription.json').write_text(json.dumps(transcription))
    sf.write(tmp_path/'vocals-16k.wav', np.ones(16000*14)*.1, 16000)
    draft = dict(title='Song',artist='Singer',language='Chinese',lines=lines,
                 provenance=dict(kind='asr_draft'),source_url=URL)
    monkeypatch.setattr(module, 'prepare_song_text', lambda *a,**k: json.loads(json.dumps(draft)))
    pipeline.prepare_lyrics(metadata, 14)
    result = json.loads((tmp_path/'lyrics.json').read_text())
    anchors = json.loads((tmp_path/'anchors.json').read_text())
    assert result['provenance']['kind'] == ('asr_draft' if formatting in ('annotations', 'unavailable')
                                           else 'source_manual_subtitles')
    assert result['lines'] == expected
    if formatting == 'annotations':
        assert result['provenance']['source_captions']['status'] == 'unsupported_caption_annotations'
    elif formatting == 'unavailable':
        assert result['provenance']['source_captions'] == dict(status='unavailable',
            error=dict(type='HTTPStatusError', status=404), retry_at=None)
    else:
        assert ''.join(expected) == ''.join(displayed)
    assert all(any(c.isalnum() for c in line) for line in result['lines'])
    assert [a['start'] for a in anchors] == [10,11,12]
    unrelated = dict(transcription, words=[dict(word='完全無關的內容',start=1,end=2)])
    candidate, report = source_captions.caption_candidate(json.loads(path.read_text()), unrelated, 'Chinese', 14)
    assert candidate is None and report['status'] == {
        'annotations': 'unsupported_caption_annotations', 'unavailable': 'unavailable'
    }.get(formatting, 'insufficient_audio_support')
    # Never override explicit lyrics, even with a matching publisher track.
    draft['provenance']['kind'] = 'user_supplied'
    pipeline.prepare_lyrics(metadata, 14)
    assert json.loads((tmp_path/'lyrics.json').read_text())['provenance']['kind'] == 'user_supplied'


@pytest.mark.parametrize('url', [
    'https://www.youtube.com/api/timedtext?v=other&lang=zh&fmt=json3',
    'https://www.youtube.com/api/timedtext?v=n5sjpCQhTDk&lang=zh&fmt=json3&kind=asr',
    'https://www.youtube.com/api/timedtext?v=n5sjpCQhTDk&lang=en&fmt=json3&tlang=zh',
    'https://youtube.com.attacker.test/api/timedtext?v=n5sjpCQhTDk&lang=zh&fmt=json3',
    'http://127.0.0.1/api/timedtext?v=n5sjpCQhTDk&lang=zh&fmt=json3',
])
def test_only_same_source_manual_language_captions_are_eligible(url):
    from karaoke.source_captions import manual_track
    track = {'zh': [dict(ext='json3',url=url)]}
    assert manual_track(dict(id='n5sjpCQhTDk',subtitles=track), 'Chinese') is None
    assert manual_track(dict(id='n5sjpCQhTDk',automatic_captions=track), 'Chinese') is None


def test_manual_caption_failure_obeys_backoff_and_keeps_cached_diagnostics(tmp_path, monkeypatch):
    import httpx
    from karaoke import source_captions
    from karaoke.stage_cache import StageCache, retained_files
    url = 'https://www.youtube.com/api/timedtext?v=example&lang=en&fmt=json3'
    info = tmp_path/'info.json'
    info.write_text(json.dumps(dict(id='example',subtitles={'en':[dict(ext='json3',url=url)]})))
    clock, calls = [1000.], []
    monkeypatch.setattr(source_captions.time, 'time', lambda:clock[0])
    def fetch(url):
        calls.append(url)
        response=httpx.Response(429,headers={'Retry-After':'900'},request=httpx.Request('GET',url))
        raise httpx.HTTPStatusError('rate limited',request=response.request,response=response)
    monkeypatch.setattr(source_captions, 'read_document', fetch)
    cache = StageCache(tmp_path)
    path = source_captions.fetch_manual_captions(info, 'English', cache)
    assert json.loads(path.read_text())['retry_at'] == 1900.
    candidate, report = source_captions.caption_candidate(json.loads(path.read_text()), {}, 'English', 10)
    assert candidate is None and report == dict(status='unavailable',
        error=dict(type='HTTPStatusError', status=429), retry_at=1900.)
    assert path.name in retained_files(tmp_path)
    source_captions.fetch_manual_captions(info, 'English', cache)
    assert len(calls) == 1
    clock[0] = 1901.
    source_captions.fetch_manual_captions(info, 'English', cache)
    assert len(calls) == 2


@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('soundtrack_credit', [False, True])
@pytest.mark.parametrize('performance', [None, 'same', 'different', 'unspecified'])
def test_online_lyrics_correct_asr_text_without_using_provider_times(tmp_path, monkeypatch, padded, soundtrack_credit, performance):
    import soundfile as sf
    from urllib.parse import parse_qs, urlsplit
    from karaoke import lyric_lookup, pipeline as module
    pipeline = Pipeline(dict(url=URL), tmp_path)
    (tmp_path/'download').mkdir()
    info = tmp_path/'download/source.info.json'
    info.write_text(json.dumps(dict(id='example')))
    lines = ['微風裡慢慢歌唱', '星光下靜靜傾聽', '微風裡慢慢歌唱']
    row = dict(id=1, trackName='晚風（大型紀錄片《遠山》主題歌）' if soundtrack_credit else '晚風',
               artistName='林林', duration=15., instrumental=False,
               plainLyrics='\n'.join(lines), syncedLyrics='[00:00.00]wrong times must be ignored')
    if performance in ('same', 'different'):
        row['albumName'] = 'Live at Elm Hall' if performance == 'same' else 'Live from Cedar Hall'
    metadata = dict(title='林林 - 晚風 (Live from The Elm Hall) [Official Performance Video]' if performance else
                         '林林 - 晚風 (We Live in Elm Hall) [Official Music Video]')
    calls = []
    def fetch(url):
        query = parse_qs(urlsplit(url).query)
        calls.append(query)
        return [row] if query['track_name'] == ['晚風'] else []
    monkeypatch.setattr(lyric_lookup, 'read_document', fetch)
    offset, duration = (2, 20) if padded else (0, 15)
    words = [dict(word=text, start=offset+10+i, end=offset+10.9+i) for i,text in enumerate(lines)]
    transcription = dict(language='Chinese', text='\n'.join(lines), words=words)
    (tmp_path/'asr').mkdir()
    (tmp_path/'asr/transcription.json').write_text(json.dumps(transcription))
    audio = np.zeros(16000*duration)
    audio[16000*offset:16000*(offset+15)] = .1
    sf.write(tmp_path/'vocals-16k.wav', audio, 16000)
    source = tmp_path/'source-44100.wav'
    sf.write(source, audio, 16000, subtype='DOUBLE')
    original_hash = file_hash(source)
    draft = dict(title='晚風 Evening Wind', artist='林林 Lin Lin', language='Chinese',
                 lines=['微風你慢慢歌場', '星光下盡盡請停', '微風你慢慢歌場'], provenance=dict(kind='asr_draft'))
    monkeypatch.setattr(module, 'prepare_song_text', lambda *a, **k: json.loads(json.dumps(draft)))
    pipeline.prepare_lyrics(metadata, duration)
    result = json.loads((tmp_path/'lyrics.json').read_text())
    if performance in ('different', 'unspecified'):
        assert result['provenance']['kind'] == 'asr_draft' and result['lines'] == draft['lines']
        return
    assert result['lines'] == lines and result['provenance']['kind'] == 'online_lyric_text'
    assert [a['start'] for a in json.loads((tmp_path/'anchors.json').read_text())] == [offset+i for i in [10,11,12]]
    assert file_hash(source) == original_hash
    assert len(calls) == 2
    pipeline.prepare_lyrics(metadata, duration)
    assert len(calls) == 2  # A retry keeps the fetched text, just like downloaded audio.
    for kind in ('user_supplied', 'verified_catalog', 'video_description'):
        draft.update(lines=lines, provenance=dict(kind=kind))
        pipeline.prepare_lyrics(metadata, duration)
        assert json.loads((tmp_path/'lyrics.json').read_text())['provenance']['kind'] == kind
    assert len(calls) == 2


@pytest.mark.parametrize('change', [
    dict(artistName='另一個歌手'), dict(trackName='別的歌'), dict(trackName='晚風 Live'),
    dict(trackName='晚風 (Live)（電影《遠山》主題曲）'),
    dict(trackName='晚風（電影《遠山》主題曲）Live'),
    dict(trackName='晚風（現場演唱《遠山》主題曲）'),
    dict(trackName='晚風（電影《遠山》主題曲混音版）'),
    dict(trackName='晚風（鋼琴版）'), dict(trackName='別的歌（電影《遠山》主題曲）'),
    dict(artistName='林林（電影《遠山》主題曲）'),
    dict(duration=309), dict(duration=float('nan')), dict(instrumental=True),
    dict(plainLyrics='毫不相干的文字'), dict(plainLyrics='[Music] 微風裡慢慢歌唱'),
    dict(trackName=['malformed']), dict(plainLyrics=None),
])
def test_online_lyrics_reject_other_versions_and_unsupported_text(change):
    from karaoke.lyric_lookup import lyric_candidate
    song = dict(title='晚風 Evening Wind', artist='林林 Lin Lin', language='Chinese')
    row = dict(id=1, trackName='晚风', artistName='林林', duration=266., instrumental=False,
               plainLyrics='微風裡慢慢歌唱')
    transcription = dict(words=[dict(word=row['plainLyrics'])])
    candidate, report = lyric_candidate(dict(status='ok', results=[row]), song, transcription, 265.)
    assert candidate and report['status'] == 'accepted'
    candidate, report = lyric_candidate(dict(status='ok', results=[dict(row, **change)]), song, transcription, 265.)
    assert candidate is None and report['status'] == 'no_supported_recording'


@pytest.mark.parametrize('found', [True, False])
@pytest.mark.parametrize('audio_only', [True, False])
def test_online_lyric_candidate_skips_asr_only_for_interactive_playback(tmp_path, monkeypatch, found, audio_only):
    from karaoke import lyric_lookup
    from concurrent.futures import Future
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=audio_only)
    lyrics = dict(title='Evening', artist='Singer', language='English', lines=['Stars above the quiet shore'],
                  provenance=dict(kind='online_lyric_text', route='lyrics_first', needs_review=True))
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'api')
    monkeypatch.setattr(lyric_lookup, 'prepare_online_lyrics', lambda *a: lyrics if found else None)
    stages = []
    def run(name, *args, **kwargs):
        stages.append(name)
        if name == 'transcribe':
            raise StageFailure('transcribe', 'ASR reached')
    monkeypatch.setattr(pipeline, 'run', run)
    lead = Future()
    lead.set_result(None)
    if found and not audio_only:
        assert pipeline.prepare_text({}, 30, lead) == lyrics
        assert 'transcribe' not in stages
        assert json.loads((tmp_path/'anchors.json').read_text()) == []
        assert not (tmp_path/'asr').exists()
    else:
        with pytest.raises(StageFailure, match='ASR reached'):
            pipeline.prepare_text({}, 30, lead)


def test_lyrics_first_search_uses_metadata_duration_and_retains_review_status(tmp_path, monkeypatch):
    from karaoke import lyric_lookup
    from karaoke.stage_cache import StageCache
    (tmp_path/'download').mkdir()
    (tmp_path/'download/source.info.json').write_text('{"id":"fixture"}')
    row = dict(id=1, trackName='晚风', artistName='林林', duration=30., instrumental=False,
               plainLyrics='微風裡慢慢歌唱\n星光下靜靜傾聽', syncedLyrics='[99:99]irrelevant')
    queries = []
    def fetch(url):
        queries.append(url)
        return [] if len(queries) == 1 else [dict(row, artistName='Wrong singer'), dict(row, duration=80), row]
    monkeypatch.setattr(lyric_lookup, 'read_document', fetch)
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', lambda *a: pytest.fail('Tagged recording needs no model'))
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL), dict(track='晚風', artist='林林'),
                                               30, StageCache(tmp_path), 'unused', None)
    assert len(queries) == 2 and result['lines'] == row['plainLyrics'].splitlines()
    assert result['language'] == 'Chinese'
    provenance = result['provenance']
    assert provenance['needs_review'] and provenance['route'] == 'lyrics_first'
    assert provenance['independent_asr_coverage'] is None
    assert provenance['lyric_lookup']['status'] == 'candidate'
    assert provenance['lyric_lookup']['audio_word_coverage'] is None
    assert provenance['lyric_lookup']['text_sha256'] == hashlib.sha256(row['plainLyrics'].encode()).hexdigest()


@pytest.mark.parametrize('padding', ['zero', 'quiet', 'other_channel', 'all_silent', 'nonfinite', 'wrong_duration'])
def test_lyric_duration_ignores_only_digital_padding(tmp_path, padding):
    import soundfile as sf
    from karaoke.lyric_lookup import lyric_candidate
    source = tmp_path/'source.wav'
    audio = np.zeros((20000, 2))
    if padding != 'all_silent':
        audio[2000:17000] = .1
    if padding == 'quiet':
        audio[[0, -1], :] = 1e-12
    if padding == 'other_channel':
        audio[:, 1] = 1e-12
    if padding == 'nonfinite':
        audio[3000, 0] = np.nan
    sf.write(source, audio, 1000, subtype='DOUBLE')
    song = dict(title='晚風', artist='林林', language='Chinese')
    row = dict(id=1, trackName='晚風', artistName='林林', duration=15., instrumental=False,
               plainLyrics='微風裡慢慢歌唱')
    candidate, report = lyric_candidate(dict(status='ok', results=[row]), song,
        dict(words=[dict(word=row['plainLyrics'])]), 40. if padding == 'wrong_duration' else 20., audio_path=source)
    assert bool(candidate) is (padding == 'zero')
    if candidate:
        assert report['duration_basis']['nonzero_duration'] == 15.
        assert report['duration_basis']['leading_zero_seconds'] == 2.
        assert report['duration_basis']['trailing_zero_seconds'] == 3.


def test_online_lyric_fetch_retries_after_backoff_and_invalidates_changed_source(tmp_path, monkeypatch):
    import httpx
    from karaoke import lyric_lookup
    from karaoke.stage_cache import StageCache, retained_files
    info = tmp_path/'source.info.json'
    info.write_text('{"id":"one"}')
    song = dict(title='A Song', artist='A Singer')
    clock, calls = [1000.], []
    monkeypatch.setattr(lyric_lookup.time, 'time', lambda:clock[0])
    def unavailable(url):
        calls.append(url)
        response = httpx.Response(429, headers={'Retry-After':'900'}, request=httpx.Request('GET', url))
        raise httpx.HTTPStatusError('rate limited', request=response.request, response=response)
    monkeypatch.setattr(lyric_lookup, 'read_document', unavailable)
    cache = StageCache(tmp_path)
    packet = lyric_lookup.fetch_lyrics(song, info, cache)
    assert packet['status'] == 'unavailable' and packet['retry_at'] == 1900.
    assert 'lyric-lookup.json' in retained_files(tmp_path)
    assert lyric_lookup.fetch_lyrics(song, info, cache) == packet and len(calls) == 1
    clock[0] = 1901.
    lyric_lookup.fetch_lyrics(song, info, cache)
    assert len(calls) == 2
    info.write_text('{"id":"two"}')
    lyric_lookup.fetch_lyrics(song, info, cache)
    assert len(calls) == 3


def test_soak_retries_changed_code_not_unchanged_failures_or_finished_cases():
    from karaoke.tools.soak import eligible
    assert eligible({'attempts': []}, 'v1', 100)
    for status in ('passed', 'failed'):
        case = {'attempts': [{'status': status, 'pipeline_version': 'v1'}]}
        assert not eligible(case, 'v1', 100)
        assert eligible(case, 'v2', 100)
    case = {'attempts': [{'status': 'external', 'pipeline_version': 'v1', 'next_retry_at': 200}]}
    assert not eligible(case, 'v2', 100)
    assert eligible(case, 'v2', 201)


def test_soak_selection_is_fixed_distinct_and_does_not_fill_missing_quota():
    from karaoke.tools.soak import select_cases
    pool = [dict(url=f'https://youtu.be/{i:011d}', title=f'Song {i}', language='English') for i in range(12)]
    pool.append(pool[0].copy())
    one = select_cases(pool, {'English': 8}, 42)
    assert one == select_cases(list(reversed(pool)), {'English': 8}, 42)
    assert len({c['url'] for c in one}) == 8
    with pytest.raises(ValueError, match='distinct candidates'):
        select_cases(pool, {'English': 20}, 42)


def test_soak_observes_failed_job_when_library_no_longer_exposes_job_id():
    import httpx
    from karaoke.tools.soak import wait_job
    calls = []
    def respond(request):
        calls.append(request.url.path)
        if request.url.path.endswith('/events'):
            return httpx.Response(200, text='data: '+json.dumps({'library': [
                {'job_id': None, 'url': URL, 'status': 'failed'}]})+'\n\n')
        return httpx.Response(200, json={'status': 'running' if len(calls)==1 else 'failed', 'input': {'url': URL}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert wait_job(client, 'http://test', 'job')['status'] == 'failed'
    assert calls == ['/v1/jobs/job', '/v1/room/events', '/v1/jobs/job']


def test_soak_collects_retries_in_submission_order(tmp_path, monkeypatch):
    import hashlib
    import httpx
    from karaoke.tools import soak
    cases = [dict(url=f'https://youtu.be/{i:011d}', title=str(i), language='English',
        provider='youtube', attempts=[dict(job_id=str(i), status='submitted',
            pipeline_version='v1', submitted_at=f'2026-10-01T00:00:0{1-i}Z', entry_id=str(i))])
        for i in range(2)]
    selection = [dict(url=c['url'], language=c['language']) for c in cases]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=cases, provider_backoff={}, cache_limit_bytes=GIB,
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())))
    observed = []

    def respond(request):
        path = request.url.path
        if path == '/health':
            data = {'pipeline_version': 'v2'}
        elif path == '/v1/room':
            data = {'entries': [], 'current_id': None}
        elif path == '/v1/storage':
            data = {'limit_bytes': GIB, 'can_prepare': True}
        else:
            job_id = path.removeprefix('/v1/jobs/')
            observed.append(job_id)
            data = dict(id=job_id, status='failed', pipeline_version='v2',
                        error={'code': 'processing_failed', 'message': 'test'})
        return httpx.Response(200, json=data)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kwargs: client)
    soak.run(manifest, tmp_path, 'http://test')
    assert observed == ['1', '0']
    saved = json.loads(manifest.read_text())['cases']
    assert [c['url'] for c in saved] == [c['url'] for c in cases]
    assert all(c['attempts'][-1]['status'] == 'failed' for c in saved)
    assert all(c['attempts'][-1]['pipeline_version'] == 'v2' for c in saved)


@pytest.mark.parametrize('provider,login', [('youtube', False), ('youtube', True), ('openai', False)])
@pytest.mark.parametrize('previous_delay', [3600, 14400])
def test_soak_new_failure_never_shortens_provider_backoff(tmp_path, monkeypatch, provider, login, previous_delay):
    import hashlib
    import httpx
    from karaoke.tools import soak
    previous_retry = time.time()+previous_delay
    case = dict(url=URL, title='Song', language='English', provider='youtube', attempts=[
        dict(job_id='job', status='submitted', pipeline_version='v1', submitted_at='1',
             entry_id='entry', external_attempt=1)])
    selection = [dict(url=URL, language='English')]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=[case], cache_limit_bytes=GIB,
        provider_backoff={provider: previous_retry},
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())))
    folder = tmp_path/'job'
    folder.mkdir()
    (folder/'download.log').write_text('HTTP Error 429: Too Many Requests'+(
        '\nSign in to confirm you are not a bot' if login else ''))
    error = (dict(code='openai_rate_limit', provider='openai', retryable=True) if provider == 'openai'
             else dict(code='download_failed'))
    responses = {'/health': {'pipeline_version': 'v1'},
        '/v1/room': {'entries': [], 'current_id': None},
        '/v1/storage': {'limit_bytes': GIB, 'can_prepare': True},
        '/v1/jobs/job': dict(id='job', status='failed', error=error)}
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=responses[request.url.path])))
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kwargs: client)
    soak.run(manifest, tmp_path, 'http://test')
    saved = json.loads(manifest.read_text())
    attempt = saved['cases'][0]['attempts'][-1]
    assert attempt['status'] == ('blocked' if login else 'external')
    assert saved['provider_backoff'][provider] >= attempt['next_retry_at']
    assert saved['provider_backoff'][provider] == max(previous_retry, attempt['next_retry_at'])


def test_soak_checks_actual_stem_samples_not_only_pipeline_claim(tmp_path):
    import soundfile as sf
    from karaoke.exact_stems import pcm_sha
    from karaoke.tools.soak import verify_stems
    original = np.array([[.125, -.25], [.5, -.5]], dtype=np.float64)
    paths = {name: tmp_path/(name+'.wav') for name in ('lead_vocals','backing_vocals','instrumental')}
    for (name, path), gain in zip(paths.items(), (.25, .25, .5)):
        sf.write(path, original*gain, 8000, subtype='DOUBLE')
    assert verify_stems(paths, pcm_sha(original)) == pcm_sha(original)
    sf.write(paths['lead_vocals'], original*.5, 8000, subtype='DOUBLE')
    with pytest.raises(ValueError, match='bitwise'):
        verify_stems(paths, pcm_sha(original))


@pytest.mark.parametrize('diagnostic,reason', [
    ("Unable to connect to proxy: Tunnel connection failed: 403 Forbidden", '网络代理'),
    ("Sign in to confirm you're not a bot", '登录验证'),
    ('ERROR: Video unavailable', '无法访问'),
    ('ERROR: connection timed out', '超时'),
    ("Failed to resolve 'www.youtube.com' ([Errno 8] nodename nor servname provided, or not known)", 'DNS'),
    ('Temporary failure in name resolution', 'DNS'),
    ('Name or service not known', 'DNS'),
    ('Unknown upstream error', '下载失败'),
])
def test_download_failure_has_actionable_reason_without_raw_diagnostics(tmp_path, diagnostic, reason):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    # Exercise the failed subprocess path; no network or model is involved.
    with pytest.raises(StageFailure) as caught:
        pipeline.run('download', [sys.executable, '-c',
            'import sys;print(sys.argv[1]);sys.exit(1)', diagnostic+' https://user:secret@example.invalid/'])
    assert caught.value.stage == 'download' and reason in str(caught.value)
    assert 'secret' not in str(caught.value) and 'example.invalid' not in str(caught.value)
    assert (tmp_path/'download.log').exists()
    if reason == 'DNS':
        from karaoke.tools.soak import eligible, record_failure
        # Also classify retained generic errors from workers predating the fix.
        attempt = dict(pipeline_version='v1', external_attempt=1)
        record_failure(attempt, dict(id=tmp_path.name,
            error=dict(code='download_failed', message='Generic download failure')), tmp_path)
        assert attempt['status'] == 'external' and not attempt['provider_blocked']
        assert not eligible({'attempts': [attempt]}, 'v2', time.time())
        assert eligible({'attempts': [attempt]}, 'v1', attempt['next_retry_at']+1)


def test_stage_counters_advance_after_work_and_do_not_finish_on_failure(monkeypatch, capsys):
    monkeypatch.setenv('KARAOKE_EVENTS_STDOUT', '1')
    monkeypatch.setattr('karaoke.stage_progress.time.monotonic', iter(range(100)).__next__)
    items = progress_items(['first', 'second', 'third'])
    assert next(items) == 'first'
    assert parse_progress(capsys.readouterr().out.strip())['completed'] == 0
    assert next(items) == 'second'
    assert parse_progress(capsys.readouterr().out.strip())['completed'] == 1
    items.close()  # The second item failed; it and the final item are not done.
    assert not capsys.readouterr().out


@pytest.mark.parametrize('payload', [
    dict(completed=500, total=None, unit='bytes'),
    dict(completed=500, total='NA', unit='bytes'),
    dict(completed=500, total=100, unit='bytes'),
    dict(completed=500, total=float('nan'), unit='bytes'),
])
def test_download_without_a_reliable_total_keeps_bytes_without_a_percentage(payload):
    result = parse_progress(PREFIX+json.dumps(payload))
    assert result['completed'] == 500 and result['total'] is None


def test_subprocess_progress_arrives_before_exit_and_keeps_parallel_stages(tmp_path, monkeypatch):
    monkeypatch.setenv('KARAOKE_EVENTS_STDOUT', '1')
    pipeline = Pipeline(dict(url=URL), tmp_path)
    pipeline.phase('prepare_lyrics', True, song_info=dict(title='A song'))
    release = tmp_path/'release'
    code = '''import pathlib,sys,time
print('KARAOKE_STAGE_PROGRESS not-json', flush=True)
print('KARAOKE_STAGE_PROGRESS []', flush=True)
sys.stdout.write('KARAOKE_STAGE_PROGRESS {"completed":1,');sys.stdout.flush()
time.sleep(.05)
print('"total":4,"unit":"chunks"}', flush=True)
while not pathlib.Path(sys.argv[1]).exists(): time.sleep(.01)
'''
    errors = []
    def run():
        try:
            pipeline.run_process('separate_lead', [sys.executable, '-c', code, str(release)], timeout=5)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            progress = json.loads((tmp_path/'progress.json').read_text())
            if progress['stage_progress'].get('separate_lead', {}).get('completed') == 1:
                break
            time.sleep(.01)
        assert thread.is_alive()
        assert progress['stage_progress']['separate_lead']['total'] == 4
        assert progress['active_stages'] == ['prepare_lyrics', 'separate_lead']
        assert progress['song_info'] == {'title': 'A song'}
    finally:
        release.touch()
        thread.join(6)
    assert not errors and not thread.is_alive()
    progress = json.loads((tmp_path/'progress.json').read_text())
    assert progress['active_stages'] == ['prepare_lyrics'] and not progress['stage_progress']
    assert 'not-json' in (tmp_path/'separate_lead.log').read_text()


def test_stage_pipeline_overlaps_different_work_but_serializes_separation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv('KARAOKE_STAGE_LOCK_DIR', str(tmp_path/'slots'))
    first, second = [Pipeline(dict(url=URL), tmp_path/name) for name in ('first', 'second')]
    code = '''import pathlib,sys,time
pathlib.Path(sys.argv[1]).touch()
while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)
'''
    def stage(pipeline, name, label):
        pipeline.run_process(name, [sys.executable, '-c', code,
            str(tmp_path/(label+'.started')), str(tmp_path/(label+'.release'))], timeout=10)
    def wait(check):
        deadline = time.monotonic()+8
        while not check() and time.monotonic() < deadline:
            time.sleep(.01)
        assert check()
    def first_song():
        stage(first, 'separate_music', 'first-separation')
        stage(first, 'align_words', 'first-alignment')
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first_song)
        two = None
        try:
            wait(lambda: (tmp_path/'first-separation.started').exists())
            two = pool.submit(stage, second, 'separate_lead', 'second-separation')
            wait(lambda: second.stage_progress.get('separate_lead', {}).get('waiting'))
            assert not (tmp_path/'second-separation.started').exists()
            (tmp_path/'first-separation.release').touch()
            wait(lambda: (tmp_path/'first-alignment.started').exists() and
                         (tmp_path/'second-separation.started').exists())
            assert not one.done() and not two.done()
        finally:
            for label in ('first-separation', 'first-alignment', 'second-separation'):
                (tmp_path/(label+'.release')).touch()
        one.result()
        if two is not None:
            two.result()
    assert not first.active and not second.active


def test_silent_subprocess_still_obeys_timeout(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    with pytest.raises(StageFailure, match='超过允许的处理时间'):
        pipeline.run('render', [sys.executable, '-c', 'import time;time.sleep(30)'], timeout=.1)
    assert not pipeline.active and pipeline.timings['render'] < 3


def test_frontend_changes_invalidate_browser_cache_including_imports(tmp_path, monkeypatch):
    web = tmp_path/'web'
    web.mkdir()
    (web/'index.html').write_text('<script type="module" src="/assets/app.js"></script>')
    (web/'app.js').write_text('import "./player.js";')
    (web/'player.js').write_text('export const version=1;')
    (web/'style.css').write_text('body{color:white}')
    monkeypatch.setattr('karaoke.api.ROOT', tmp_path)
    with TestClient(create_app(tmp_path/'jobs', version='test')) as client:
        def script_url():
            home = client.get('/')
            assert home.headers['cache-control'] == 'no-cache'
            return re.search(r'src="([^"]+)"', home.text)[1]
        first = script_url()
        assert client.get(first).text == (web/'app.js').read_text()
        (web/'player.js').write_text('export const version=2;')
        second = script_url()
        assert first != second
        assert client.get(urljoin(second, './player.js')).text == 'export const version=2;'
        (web/'style.css').write_text('body{color:gold}')
        assert script_url() != second
        assert client.get('/assets/player.js').status_code == 200
        assert client.get(first.rsplit('/', 1)[0]+'/%2e%2e/test.py').status_code == 404


def test_empty_alignment_error_identifies_the_failed_stage(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    with pytest.raises(StageFailure, match='歌词对齐失败.*没有有效音频') as caught:
        pipeline.run_process('align_words', [sys.executable, '-c',
            'import sys;print(\'Empty alignment audio for lyric lines 1-1\');sys.exit(1)'])
    assert caught.value.stage == 'align_words'


def test_poster_uses_thumbnail_or_video_frame_and_failure_is_optional(tmp_path):
    from PIL import Image
    thumbnail = tmp_path/'thumbnail.jpg'
    video = tmp_path/'video.mp4'
    Image.new('RGB', (80, 40), 'red').save(thumbnail)
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=blue:s=80x40:d=1',
                    '-c:v', 'mpeg4', str(video)], check=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        poster = tmp_path/'poster.jpg'
        try:
            assert create_poster(tmp_path/'missing.mp4', poster, thumbnail=base+'/thumbnail.jpg')
            with Image.open(poster) as image:
                red, _, blue = image.getpixel((0, 0))
                assert red > 240 and blue < 10
            assert create_poster(video, poster, thumbnail=base+'/missing.jpg', seconds=0)
            with Image.open(poster) as image:
                red, _, blue = image.getpixel((0, 0))
                assert blue > 240 and red < 10
            assert not create_poster(tmp_path/'missing.mp4', tmp_path/'absent.jpg')
            assert not (tmp_path/'absent.jpg').exists()
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize('tags,expected', [
    ({'track': 'Tagged song', 'artist': 'Tagged singer'}, ('Tagged song', 'Tagged singer')),
    ({'artists': ['Singer A', 'Singer B']}, ('A new video', 'Singer A · Singer B')),
    ({}, ('A new video', '')),
])
def test_download_publishes_metadata_before_separation(tmp_path, monkeypatch, capsys, tags, expected):
    monkeypatch.setenv('KARAOKE_EVENTS_STDOUT', '1')
    pipeline = Pipeline(dict(url=URL), tmp_path)

    def download(*args, **kwargs):
        pipeline.source.write_bytes(b'video fixture')
        (tmp_path/'download/source.info.json').write_text(json.dumps(dict(
            title='A new video', uploader='Not necessarily the singer', duration=120, **tags)))

    probe = dict(streams=[dict(codec_type='video'), dict(codec_type='audio')], format=dict(duration='120'))
    with patch.object(pipeline, 'run', side_effect=download), patch(
            'karaoke.pipeline.subprocess.check_output', return_value=json.dumps(probe).encode()):
        pipeline.download()
    pipeline.phase('separate_music', True)
    events = [json.loads(line.removeprefix('KARAOKE_PROGRESS '))
              for line in capsys.readouterr().out.splitlines()]
    info = dict(title=expected[0], artist=expected[1], duration_seconds=120)
    assert events[0]['song_info'] == events[1]['song_info'] == info
    assert events[0]['active_stages'] == [] and events[1]['active_stages'] == ['separate_music']
    assert json.loads((tmp_path/'progress.json').read_text())['song_info'] == info
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('metadata,requested,display,expected', [
    (dict(title='English song (日本語字幕)', subtitles={'ja': []}), None, None, 'auto'),
    (dict(language='ja', acodec='opus'), None, 'Japanese', 'auto'),
    (dict(requested_formats=[dict(acodec='none', language='ja'),
                             dict(acodec='opus', language='en-US')]), None, 'English', 'auto'),
    (dict(language='fr'), None, None, 'auto'),
    (dict(language='ja'), 'Chinese', 'Japanese', 'Chinese'),
    (dict(language='en'), None, 'English', 'auto'),
])
def test_asr_detects_language_unless_requested(tmp_path, monkeypatch, metadata, requested, display, expected):
    assert video_song_info(metadata).get('language') == display
    pipeline = Pipeline(dict(url=URL, language=requested), tmp_path)
    calls = []
    class TranscriptionRequested(Exception):
        pass
    def capture(name, command, **kwargs):
        if name == 'transcribe':
            calls.append(command[command.index('--language')+1])
            raise TranscriptionRequested
    monkeypatch.setattr(pipeline, 'run', capture)
    with pytest.raises(TranscriptionRequested):
        pipeline.prepare_text(metadata, 10)
    assert calls == [expected]


@pytest.mark.parametrize('matching_video', [True, False])
@pytest.mark.parametrize('audio_codec', ['pcm_s16le', 'libopus'])
def test_audio_download_promotes_to_video_without_replacing_tested_audio(tmp_path, monkeypatch, matching_video, audio_codec):
    from karaoke.exact_stems import decode_source, pcm_sha
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=True)
    calls = []
    def download(name, command, **kwargs):
        calls.append(name)
        path = pipeline.source if name == 'download' else tmp_path/'video-download/source.mkv'
        args = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i']
        if name == 'download':
            assert command[command.index('-f')+1] == 'ba'
            args += ['sine=frequency=440:duration=2', '-c:a', audio_codec]
        else:
            args += ['color=blue:s=160x90:r=30:d=2', '-c:v', 'libx264']
        subprocess.run(args+[str(path)], check=True)
        path.with_suffix('.info.json').write_text(json.dumps(dict(id='same' if matching_video or name=='download' else 'wrong',
            extractor_key='Youtube', title='Example', duration=2, format_id='a1' if name=='download' else 'v1')))
    monkeypatch.setattr(pipeline, 'run', download)
    with patch('karaoke.pipeline.create_poster', side_effect=AssertionError('Audio test downloaded artwork')):
        pipeline.download()
        pipeline.download()
    before = pcm_sha(decode_source(pipeline.source)[0])
    assert calls == ['download'] and not (tmp_path/'result.json').exists()
    pipeline.audio_only = False
    if not matching_video:
        with pytest.raises(StageFailure, match='来源不同'):
            pipeline.download(artwork=False)
        assert json.loads((tmp_path/'source.json').read_text())['audio_only']
    else:
        metadata, _ = pipeline.download(artwork=False)
        assert metadata['format_id'] == 'v1+a1'
        pipeline.download(artwork=False)
        assert json.loads((tmp_path/'source.json').read_text())['audio_only']
        assert pcm_sha(decode_source(pipeline.video)[0]) == before
        assert pipeline.video != pipeline.source
    assert calls == ['download', 'download_video']
    assert pcm_sha(decode_source(pipeline.source)[0]) == before


@pytest.mark.parametrize('audio_only', [False, True])
@pytest.mark.parametrize('failure', ['transcribe', 'prepare_lyrics', 'align_words', 'align_phonemes', 'separate_lead', 'package_three_stems', 'overlap'])
def test_playback_survives_text_failure_but_audio_and_acceptance_remain_strict(tmp_path, monkeypatch, audio_only, failure):
    import soundfile as sf
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=audio_only, device='cpu')
    sf.write(tmp_path/'source-44100.wav', np.zeros(44100), 44100)
    pipeline.vocals.parent.mkdir(parents=True)
    sf.write(pipeline.vocals, np.zeros(44100), 44100)
    (tmp_path/'stems').mkdir()
    for name in ('instrumental', 'backing-vocals', 'lead'):
        for extension in ('wav', 'mp3'):
            (tmp_path/f'stems/{name}.{extension}').write_bytes(b'artifact fixture')
    (tmp_path/'stems/manifest.json').write_text(json.dumps(dict(
        preview_shared_gain=.8, reconstruction=dict(vocal_first={'verified': True}))))
    lyrics = dict(title='Example', artist='Singer', language='English', lines=['A little song'], provenance={})
    (tmp_path/'lyrics.json').write_text(json.dumps(lyrics))
    (tmp_path/'alignment').mkdir()
    coarse = [dict(text='A little song', start=.1, end=.9, status='candidate')]
    (tmp_path/'alignment/cues.json').write_text(json.dumps(coarse))
    monkeypatch.setenv('OPENAI_API_KEY', 'unit-test-placeholder')
    monkeypatch.setattr(pipeline, 'download', lambda: ({'format_id': 'fixture'}, 1))
    alignment_started = threading.Event()
    def fail(stage):
        if stage == 'separate_lead' and failure == 'overlap':
            assert alignment_started.wait(10), 'Alignment waited for independent lead separation'
        if stage == failure:
            raise StageFailure(stage, 'fixture failure')
    monkeypatch.setattr(pipeline, 'run', lambda stage, *a, **kw: fail(stage))
    def prepare(*args):
        fail('transcribe')
        fail('prepare_lyrics')
        if failure != 'overlap':
            args[-1].result()
        return lyrics
    monkeypatch.setattr(pipeline, 'prepare_text', prepare)
    def align(*args):
        alignment_started.set()
        (tmp_path/'quality.json').write_text('{}')
        (tmp_path/'cues.json').write_text(json.dumps(coarse))
        fail('align_words')
        fail('align_phonemes')
    monkeypatch.setattr(pipeline, 'align', align)
    monkeypatch.setattr(pipeline, 'render_video', lambda result: (tmp_path/'mv.mp4').write_bytes(b'video fixture'))
    if failure == 'overlap':
        result = pipeline.execute()
        assert result['validation_scope'] == ('audio_and_lyrics' if audio_only else 'full_mv')
    elif audio_only or failure in ('separate_lead', 'package_three_stems'):
        with pytest.raises(StageFailure, match='fixture failure'):
            pipeline.execute()
        assert not (tmp_path/'result.json').exists()
    else:
        result = pipeline.execute()
        assert result['validation_scope'] == 'playback_with_lyrics_review'
        assert result['quality']['lyrics_provenance']['needs_review']
        assert not result['quality']['playback_fallback']['word_alignment_passed']
        assert result['quality']['playback_fallback']['error']['code'] == failure+'_failed'
        assert json.loads((tmp_path/'cues.json').read_text()) == [dict(coarse[0], units=[])]
        assert 'mv' in result['artifacts'] and result['shared_preview_gain'] == .8
        assert not (tmp_path/'audio-result.json').exists()


@pytest.mark.parametrize('bad', ['text', 'overlap', 'missing', 'nan'])
def test_playback_fallback_never_invents_missing_phrase_times(tmp_path, bad):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(title='Example', artist='', language='English',
        lines=['First line', 'Next line'], provenance={})))
    (tmp_path/'alignment').mkdir()
    cues = [dict(text=text, start=start, end=end, status='candidate')
            for text, start, end in [('First line', 0, 1), ('Next line', 2, 3)]]
    if bad == 'text': cues[0]['text'] = 'Wrong lyric'
    if bad == 'overlap': cues[1]['start'] = .5
    if bad == 'missing': cues[0]['start'] = None
    if bad == 'nan': cues[0]['start'] = float('nan')
    (tmp_path/'alignment/cues.json').write_text(json.dumps(cues))
    pipeline.playback_fallback({}, 4, StageFailure('align_words', 'unresolved'))
    assert json.loads((tmp_path/'cues.json').read_text()) == []
    assert json.loads((tmp_path/'quality.json').read_text())['playback_fallback']['subtitle_mode'] == 'unavailable'


def test_audio_pipeline_runs_separation_and_alignment_but_cannot_claim_a_full_mv(tmp_path, monkeypatch):
    import soundfile as sf
    from karaoke.exact_stems import pcm_sha, encode_preview
    from karaoke.tools.soak import verify_outputs
    folder = tmp_path/'job'
    pipeline = Pipeline(dict(url=URL), folder, audio_only=True, device='cpu')
    sample_rate = 44100
    tone = (.1*np.sin(np.arange(sample_rate)*2*np.pi*440/sample_rate)).astype('float32')[:,None]
    artifacts = folder/'stems'
    artifacts.mkdir()
    for name, scale in [('lead', .25), ('backing-vocals', .25), ('instrumental', .5)]:
        sf.write(artifacts/f'{name}.wav', tone*scale, sample_rate, subtype='DOUBLE')
        encode_preview(artifacts/f'{name}.wav', artifacts/f'{name}.mp3', 1, len(tone), 1)
    (artifacts/'manifest.json').write_text(json.dumps(dict(preview_shared_gain=1,
        reconstruction=dict(vocal_first=dict(source_pcm_sha256=pcm_sha(tone.astype('float64')))))))
    sf.write(folder/'source-44100.wav', tone, sample_rate)
    pipeline.vocals.parent.mkdir(parents=True)
    sf.write(pipeline.vocals, tone, sample_rate)
    (folder/'cues.json').write_text(json.dumps([dict(text='Hello',start=.1,end=.8,status='candidate',
        units=[dict(text='Hello',first=0,stop=5,start=.1,end=.8)])]))
    (folder/'asr').mkdir()
    repeat_path = folder/'asr/lyric-repeats.json'
    repeat_path.write_text(json.dumps(dict(reference_lines=['Hello'],
        words=[dict(word='Hello', start=.1, end=.8)], status='complete', expansions=[])))
    (folder/'quality.json').write_text(json.dumps(dict(english_context_scope=[[0, 1, 0, 1, 0., 1.]])))
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-placeholder')
    monkeypatch.setenv('KARAOKE_SEPARATION_BATCH_SIZE', '4')
    monkeypatch.setattr(pipeline, 'download', lambda: ({'format_id':'a1'},1))
    stages=[]
    def run_stage(name, command, **kwargs):
        stages.append(name)
        if name.startswith('separate_'):
            assert command[command.index('--batch-size')+1] == '4'
    monkeypatch.setattr(pipeline, 'run', run_stage)
    monkeypatch.setattr(pipeline, 'prepare_text', lambda *args: dict(title='Example',artist='',language='English'))
    monkeypatch.setattr(pipeline, 'align', lambda *args: stages.append('align'))
    monkeypatch.setattr(pipeline, 'render_video', lambda *args: pytest.fail('Audio-only test rendered a video'))
    result=pipeline.execute()
    assert {'separate_music','separate_lead','package_three_stems','align'}<=set(stages)
    assert result['validation_scope']=='audio_and_lyrics' and 'mv' not in result['artifacts']
    assert not (folder/'result.json').exists()
    evidence=verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    assert evidence['validation_scope']=='audio_and_lyrics' and len(evidence['artifact_sha256'])==7
    # The long-lived controller never imports Pipeline; its clean interpreter
    # must verify results without the research scripts added to sys.path.
    import os
    from pathlib import Path
    import subprocess
    import sys
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-c',
        'from pathlib import Path; from karaoke.tools.soak import verify_outputs; '
        f'verify_outputs(None, "", Path({str(tmp_path)!r}), {{"id":"job"}}, audio_only=True)'],
        cwd=Path(__file__).resolve().parent.parent, env=env, check=True, capture_output=True)
    # A complete phrase collapsed into one positive interval also passed the
    # old text/monotonicity gates. Checksums must not make it a valid highlight.
    subtitle_path = folder/'cues.json'
    original = subtitle_path.read_text()
    subtitle_path.write_text(json.dumps([dict(text='Hello world', start=.1, end=.12,
        units=[dict(text='Hello world', first=0, stop=11, start=.1, end=.12)])]))
    result['artifacts']['subtitles'].update(bytes=subtitle_path.stat().st_size, sha256=file_hash(subtitle_path))
    (folder/'audio-result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='independent highlights'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    subtitle_path.write_text(original)
    result['artifacts']['subtitles'].update(bytes=subtitle_path.stat().st_size, sha256=file_hash(subtitle_path))
    (folder/'audio-result.json').write_text(json.dumps(result))
    repeated = json.loads(repeat_path.read_text())
    repeat_path.write_text(json.dumps(dict(repeated, words=[*repeated['words'],
        dict(word='An entire later verse was omitted from the lyrics', start=.85, end=.99)])))
    with pytest.raises(ValueError, match='lyric passages'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    coverage_path = folder/'lyric-coverage-validation.json'
    coverage = json.loads(coverage_path.read_text())
    assert coverage['status'] == 'incomplete'
    assert [(gap['start'], gap['end']) for gap in coverage['uncovered_passages']] == [(.8, 1.)]
    assert coverage['input_sha256'] == dict(subtitles=file_hash(subtitle_path), recognition=file_hash(repeat_path))
    from karaoke.vocal_review import review_gaps
    spoken = json.loads(repeat_path.read_text())
    with patch('openai.OpenAI') as client:
        client.return_value.chat.completions.create.return_value.model_dump.return_value = vocal_review_response()
        spoken['non_singing_reviews'] = review_gaps(folder, [dict(start=.8, end=1.)], spoken['words'])
    repeat_path.write_text(json.dumps(spoken))
    assert verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)['subtitle_units'] == 1
    coverage = json.loads(coverage_path.read_text())
    assert coverage['status'] == 'complete'
    assert coverage['input_sha256']['recognition'] == file_hash(repeat_path)
    spoken['non_singing_reviews'][0]['views'][0]['response'] = vocal_review_response(singing=True)
    repeat_path.write_text(json.dumps(spoken))
    with pytest.raises(ValueError, match='lyric passages'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    spoken['non_singing_reviews'][0]['views'][0]['input_sha256'] = 'changed'
    repeat_path.write_text(json.dumps(spoken))
    with pytest.raises(ValueError, match='current input'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    repeat_path.write_text(json.dumps(dict(status='failed')))
    with pytest.raises(ValueError, match='repetitions'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    # Valid hashes, monotonic subtitles and playable audio must not conceal a
    # lyric placed in unrelated speech outside its acoustic phrase context.
    for scope in (None, [], [[0, 1, 0, 1, .9, 1.]], [[0, 0, 0, 1, 0., 1.]]):
        result['quality']['english_context_scope'] = scope
        (folder/'audio-result.json').write_text(json.dumps(result))
        with pytest.raises(ValueError, match='context'):
            verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    with pytest.raises(FileNotFoundError):
        verify_outputs(None,'',tmp_path,{'id':'job'})


@pytest.mark.parametrize('oversized', [False, True])
def test_focused_probe_obeys_shared_budget_and_preserves_checkpoints(tmp_path, oversized):
    import httpx
    import os
    from karaoke.tools.soak import run_probe
    folder = tmp_path/'job'
    folder.mkdir()
    (folder/'checkpoint.wav').write_bytes(b'reusable')
    pidfile = folder/'pid'
    code = ('import os,pathlib,time;'
            f'pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));'
            f'time.sleep({30 if oversized else 0})')
    def respond(request):
        if request.url.path == '/health':
            data = dict(preparation_paused=True, running_job=None)
        elif request.url.path == '/v1/storage':
            data = dict(used_bytes=200 if oversized and pidfile.exists() else 20,
                        limit_bytes=100, can_prepare=True)
        else:
            data = dict(status='queued')
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if oversized:
            with pytest.raises(ValueError, match='cache or free-space limit'):
                run_probe(client, 'http://test', folder, [sys.executable, '-c', code])
        else:
            assert run_probe(client, 'http://test', folder, [sys.executable, '-c', code]) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert (folder/'checkpoint.wav').read_bytes() == b'reusable'


@pytest.mark.parametrize('evict', [False, True])
def test_audio_ready_job_releases_queue_protection_and_resumes_its_checkpoints(tmp_path,evict):
    from karaoke.stage_cache import StageCache
    hold=tmp_path/'preparation.pause'
    hold.write_text('test')
    calls=[]
    def command(folder):
        calls.append(folder.name)
        return fake_worker(folder)
    app=create_app(tmp_path,worker_command=command,version='test')
    with TestClient(app) as client:
        room=client.post('/v1/room/entries',json={'url':URL}).json()
        entry=room['entries'][0];folder=tmp_path/entry['job_id']; output=folder/'audio.wav'
        with StageCache(folder).stage('separate',inputs=[folder/'request.json'],outputs=[output]):
            output.write_bytes(b'computed audio')
        app.state.store.update(entry['job_id'],status='audio_ready')
        room=client.post('/v1/room/commands',json={'action':'remove','entry_id':entry['id']}).json()
        assert client.get('/v1/library').json()['songs'][0]['status']=='unprepared'
        assert client.get('/health').json()['queued']==0
        assert entry['job_id'] not in app.state.store.cache.protected(app.state.store.list(-1))
        app.state.store.cache.compact(app.state.store.read(entry['job_id']))
        assert output.read_bytes()==b'computed audio'
        if evict:
            app.state.store.cache.set_limit(1)
            assert not folder.exists()
            app.state.store.cache.set_limit(GIB)
        hold.unlink()
        deadline=time.monotonic()+2
        while client.get('/health').json()['queued'] and time.monotonic()<deadline:
            time.sleep(.02)
        assert calls==[]
        resumed=client.post('/v1/jobs',json={'url':URL}).json()
        assert (resumed['id']==entry['job_id']) is not evict
        assert finished(client,resumed['id'])['status']=='completed'
        assert calls==[resumed['id']]


def test_audio_regression_stops_on_software_failure_and_keeps_full_results_separate(tmp_path,monkeypatch):
    import hashlib
    import httpx
    from pathlib import Path
    from karaoke.room import AddSong,RoomCommand
    from karaoke.tools import soak
    data=tmp_path/'jobs';data.mkdir();(data/'preparation.pause').write_text('test')
    app=create_app(data,version='test');store=app.state.store;room=app.state.room
    cases=[dict(url=f'https://www.youtube.com/watch?v={i:011d}',title=str(i),language='English',provider='youtube',attempts=[]) for i in range(3)]
    selection=[dict(url=c['url'],language=c['language']) for c in cases]
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=cases,provider_backoff={},selection_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest())))
    version='test'
    def respond(request):
        path=request.url.path
        if path=='/health':result={'pipeline_version':version,'preparation_paused':True,'running_job':None}
        elif path=='/v1/room':result=room.snapshot()
        elif path=='/v1/room/entries':result=room.add(AddSong.model_validate_json(request.content))
        elif path=='/v1/room/commands':result=room.command(RoomCommand.model_validate_json(request.content))
        else:result=store.read(path.split('/')[-1])
        return httpx.Response(200,json=result)
    client_type=httpx.Client
    monkeypatch.setattr(soak.httpx,'Client',lambda **kwargs:client_type(transport=httpx.MockTransport(respond)))
    calls=[]
    def execute(client,base,folder,command):
        calls.append(folder.name)
        if len(calls) in (2,3):
            (folder/'error.json').write_text(json.dumps(dict(code='align_words_failed',message='real failed boundary')))
            return 1
        (folder/'audio-result.json').write_text(json.dumps(dict(title='Song',artist='',language='English',duration_seconds=100)))
        return 0
    monkeypatch.setattr(soak,'run_probe',execute)
    monkeypatch.setattr(soak,'verify_outputs',lambda *args,**kwargs:dict(validation_scope='audio_and_lyrics'))
    soak.run_audio(manifest,data,'http://test')
    saved=json.loads(manifest.read_text())['cases']
    assert saved[0]['audio_attempts'][-1]['status']=='passed_audio'
    assert saved[1]['audio_attempts'][-1]['status']=='failed'
    assert not saved[2].get('audio_attempts') and len(calls)==2
    assert all(c['attempts']==[] for c in saved)
    assert store.read(calls[0])['status']=='audio_ready'
    assert room.read()['entries']==[]
    soak.run_audio(manifest,data,'http://test')
    assert len(calls)==2
    version='after-fix'
    soak.run_audio(manifest,data,'http://test')
    assert len(calls)==3
    assert store.read(calls[-1])['input']['url']==cases[1]['url']
    # Focus another known failure without clearing the batch hold or quietly
    # resubmitting the unchanged failure in the previous case.
    pause = manifest.with_name('audio.pause')
    pause.write_text('hold batch')
    soak.run_audio(manifest,data,'http://test')
    assert len(calls)==3
    soak.run_audio(manifest,data,'http://test',case_url=cases[2]['url'])
    assert len(calls)==4 and pause.read_text()=='hold batch'
    assert store.read(calls[-1])['input']['url']==cases[2]['url']
    saved=json.loads(manifest.read_text())
    assert saved['audio_passed_current_version']==1
    assert saved['cases'][1]['audio_attempts'][-1]['status']=='failed'
    assert saved['cases'][2]['audio_attempts'][-1]['status']=='passed_audio'
    assert all(c['attempts']==[] for c in saved['cases'])
    assert room.read()['entries']==[]
    # Explicitly defer the unresolved case while revalidating every other
    # recording on the new version. Deferral never contributes a pass.
    soak.run_audio(manifest,data,'http://test',defer_urls=[cases[1]['url']])
    assert len(calls)==4 and pause.exists()
    pause.unlink()
    failure = saved['cases'][1]['audio_attempts'][:]
    soak.run_audio(manifest,data,'http://test',defer_urls=[cases[1]['url']])
    saved=json.loads(manifest.read_text())
    assert len(calls)==5 and saved['audio_passed_current_version']==2
    assert saved['audio_deferred_urls']==[cases[1]['url']]
    assert saved['cases'][1]['audio_attempts']==failure
    assert len(saved['cases'])==3 and room.read()['entries']==[]
    soak.run_audio(manifest,data,'http://test',defer_urls=[cases[1]['url']])
    assert len(calls)==5
    with pytest.raises(ValueError,match='fixed selection'):
        soak.run_audio(manifest,data,'http://test',defer_urls=[URL])
    with pytest.raises(ValueError,match='fixed selection'):
        soak.run_audio(manifest,data,'http://test',case_url=URL)

    # Survey mode retains the failure, verifies the other songs and does not
    # retry the same broken input or remove a person's queue entry.
    user_entry = room.add(AddSong(url=URL))['entries'][-1]['id']
    version = 'survey'
    def survey_execute(client, base, folder, command):
        calls.append(folder.name)
        if store.read(folder.name)['input']['url'] == cases[1]['url']:
            (folder/'error.json').write_text(json.dumps(dict(code='align_words_failed', message='missing onset')))
            (folder/'highlight-validation.json').write_text('{"errors":["missing onset"]}')
            return 1
        (folder/'audio-result.json').write_text(json.dumps(dict(title='Song', artist='', language='English', duration_seconds=100)))
        return 0
    monkeypatch.setattr(soak, 'run_probe', survey_execute)
    soak.run_audio(manifest, data, 'http://test', keep_going=True)
    saved = json.loads(manifest.read_text())
    assert len(calls) == 8 and saved['audio_passed_current_version'] == 2
    assert [c['audio_attempts'][-1]['status'] for c in saved['cases']] == ['passed_audio', 'failed', 'passed_audio']
    assert Path(saved['cases'][1]['audio_attempts'][-1]['reproduction_file']).is_file()
    assert all(c['attempts'] == [] for c in saved['cases'])
    assert [e['id'] for e in room.read()['entries']] == [user_entry]
    soak.run_audio(manifest, data, 'http://test', keep_going=True)
    assert len(calls) == 8
    version = 'interrupted-survey'
    def interrupted_probe(*args):
        calls.append('interrupted')
        raise ValueError('Focused probe reached the shared cache or free-space limit')
    monkeypatch.setattr(soak, 'run_probe', interrupted_probe)
    soak.run_audio(manifest, data, 'http://test', keep_going=True)
    assert len(calls) == 9
    assert [e['id'] for e in room.read()['entries']] == [user_entry]

    # A final login prompt must not hide the earlier provider-wide HTTP 429.
    # Retain this song's login hold and stop submitting other songs during backoff.
    version = 'rate-limited-survey'
    def rate_limited_probe(client, base, folder, command):
        calls.append(folder.name)
        (folder/'error.json').write_text(json.dumps(dict(code='download_failed', message='登录验证')))
        (folder/'download.log').write_text('HTTP Error 429: Too Many Requests\nTry again in 120 seconds\n'
                                          'Sign in to confirm you are not a bot')
        return 1
    monkeypatch.setattr(soak, 'run_probe', rate_limited_probe)
    before = time.time()
    soak.run_audio(manifest, data, 'http://test', keep_going=True)
    saved = json.loads(manifest.read_text())
    assert len(calls) == 10
    assert saved['provider_backoff']['youtube'] >= before+120
    attempted = [c['audio_attempts'][-1] for c in saved['cases'] if c['audio_attempts'][-1]['pipeline_version']==version]
    assert len(attempted) == 1 and attempted[0]['status'] == 'blocked'
    soak.run_audio(manifest, data, 'http://test', keep_going=True)
    assert len(calls) == 10
    assert [e['id'] for e in room.read()['entries']] == [user_entry]

    # Prefetch records only download evidence and leaves inference pending.
    # A downloaded source cannot count as audio/MV acceptance or trigger models.
    import soundfile as sf
    before_fetch = json.loads(manifest.read_text())
    before_fetch['provider_backoff'] = {'openai': time.time()+1000, 'openai:whisper-1': time.time()+1000}
    manifest.write_text(json.dumps(before_fetch))
    version = 'prefetch'
    fetched = []
    def download(client, base, folder, command):
        assert '--download-only' in command and '--audio-only' not in command
        fetched.append(folder.name)
        (folder/'download').mkdir(exist_ok=True)
        sf.write(folder/'download/source.mkv', np.zeros((800, 1)), 8000, format='WAV')
        source = dict(url=store.read(folder.name)['input']['url'], duration=.1,
                      sha256=file_hash(folder/'download/source.mkv'))
        (folder/'source.json').write_text(json.dumps(source))
        (folder/'download/source.info.json').write_text('{}')
        return 0
    monkeypatch.setattr(soak, 'run_probe', download)
    monkeypatch.setattr(soak, 'verify_outputs', lambda *a, **k: pytest.fail('Prefetch ran audio verification'))
    soak.run_audio(manifest, data, 'http://test', keep_going=True, download_only=True)
    after_fetch = json.loads(manifest.read_text())
    assert len(fetched) == 2  # The source's own backoff still applies.
    assert [c.get('audio_attempts') for c in after_fetch['cases']] == [c.get('audio_attempts') for c in before_fetch['cases']]
    assert all(c['attempts'] == [] for c in after_fetch['cases'])
    assert sum(bool(c.get('download_attempts')) and c['download_attempts'][-1]['status'] == 'downloaded_audio'
               for c in after_fetch['cases']) == 2
    assert all(store.read(job_id)['status'] == 'source_ready' for job_id in fetched)
    assert [e['id'] for e in room.read()['entries']] == [user_entry]
    soak.run_audio(manifest, data, 'http://test', keep_going=True, download_only=True)
    assert len(fetched) == 2
    first = fetched[0]
    assert first not in store.cache.protected(store.list(-1))
    assert room.describe(dict(id='test-entry', job_id=first))['status'] == 'unprepared'
    resubmitted, cached = store.create(JobRequest.model_validate(store.read(first)['input']), version)
    assert resubmitted['id'] == first and resubmitted['status'] == 'queued'
    store.update(first, status='source_ready')
    store.cache.limit = 1
    store.cache.enforce()
    assert not (data/first).exists()
    recovered, _ = store.create(JobRequest.model_validate(resubmitted['input']), version)
    assert recovered['id'] != first and recovered['status'] == 'queued'


@pytest.mark.parametrize('cache_state', ['valid', 'missing', 'corrupt', 'different_source'])
def test_offline_pipeline_requires_verified_source_without_downloading(tmp_path, monkeypatch, cache_state):
    monkeypatch.setenv('KARAOKE_REQUIRE_CACHED_SOURCE', '1')
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=True)
    monkeypatch.setattr(pipeline, 'run', lambda *a, **k: pytest.fail('Offline run contacted provider'))
    if cache_state != 'missing':
        (tmp_path/'download').mkdir()
        pipeline.source.write_bytes(b'cached source')
        (tmp_path/'download/source.info.json').write_text('{}')
        (tmp_path/'source.json').write_text(json.dumps(dict(
            url=URL if cache_state != 'different_source' else 'https://www.youtube.com/watch?v=00000000000',
            sha256=file_hash(pipeline.source) if cache_state != 'corrupt' else 'wrong',
            duration=10, audio_only=True)))
    if cache_state == 'valid':
        assert pipeline.download() == ({}, 10)
    else:
        with pytest.raises(StageFailure, match='离线处理'):
            pipeline.download()
    if cache_state != 'missing':
        assert pipeline.source.read_bytes() == b'cached source'


def test_roformer_batches_preserve_window_boundaries_and_partial_tail():
    import torch
    from karaoke.benchmark_neural import demix_batched
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.25))
            self.batches = []
        def forward(self, x):
            self.batches.append(len(x))
            # A context-dependent result exposes misplaced/omitted windows.
            return x*self.weight + x.mean(dim=-1, keepdim=True)
    cfg = SimpleNamespace(training=SimpleNamespace(target_instrument='vocals'),
        model=SimpleNamespace(stft_hop_length=2), inference=SimpleNamespace(dim_t=9),
        audio=SimpleNamespace(sample_rate=8))
    instance = SimpleNamespace(model_data_cfgdict=cfg, model_run=Model(),
        is_roformer=True, pitch_shift=0, overlap=1, secondary_stem_name='other')
    audio = np.random.default_rng(7).normal(size=(2, 61)).astype('float32')
    expected = demix_batched(instance, audio, 1)
    instance.model_run.batches.clear()
    actual = demix_batched(instance, audio, 3)
    assert instance.model_run.batches == [3, 3, 2]
    for stem in expected:
        np.testing.assert_array_equal(actual[stem], expected[stem])
    np.testing.assert_allclose(actual['vocals']+actual['other'], audio, atol=2e-7)


@pytest.mark.parametrize('retry_at,old_repaired,keep_going,expected_index', [
    (900, False, False, 1), (1100, False, False, 2),
    (900, True, False, 3), (900, True, True, 2), (1100, True, True, 2),
])
def test_audio_regression_prioritizes_due_failures_then_unfinished_cases(
        tmp_path, monkeypatch, retry_at, old_repaired, keep_going, expected_index):
    import httpx
    from karaoke.tools import soak
    cases = [dict(url=f'https://www.youtube.com/watch?v={i:011d}', title=str(i),
                  language='English', provider='youtube', attempts=[]) for i in range(4)]
    cases[0]['audio_attempts'] = [dict(status='passed_audio', pipeline_version='old')]
    cases[1]['attempts'] = [dict(status='external', pipeline_version='old', next_retry_at=retry_at,
                               finished_at='2026-10-01T01:00:00+00:00')]
    cases[2]['attempts'] = [dict(status='submitted', pipeline_version='old')]
    cases[3]['audio_attempts'] = [dict(status='external', pipeline_version='current', next_retry_at=retry_at,
                                     finished_at='2026-10-01T02:00:00+00:00')]
    if old_repaired:
        cases[1]['audio_attempts'] = [dict(status='passed_audio', pipeline_version='current')]
    selection = [dict(url=c['url'], language=c['language']) for c in cases]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=cases, provider_backoff={},
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())))
    data = tmp_path/'jobs'
    data.mkdir()
    submitted = []
    def respond(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'pipeline_version': 'current', 'preparation_paused': True, 'running_job': None})
        if request.url.path == '/v1/room':
            return httpx.Response(200, json={'entries': [], 'storage': {'limit_bytes': 1000, 'can_prepare': True}})
        submitted.append(json.loads(request.content)['url'])
        raise RuntimeError('observed first submission')
    client_type = httpx.Client
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kwargs: client_type(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(soak.time, 'time', lambda: 1000)
    with pytest.raises(RuntimeError, match='observed first submission'):
        soak.run_audio(manifest, data, 'http://test', keep_going=keep_going)
    assert submitted == [cases[expected_index]['url']]
    # Deferring a due retry must not discard it: run it once the other work is done.
    if keep_going and retry_at < 1000:
        saved = json.loads(manifest.read_text())
        for case in saved['cases'][:3]:
            case['audio_attempts'] = [dict(status='passed_audio', pipeline_version='current')]
        manifest.write_text(json.dumps(saved))
        submitted.clear()
        with pytest.raises(RuntimeError, match='observed first submission'):
            soak.run_audio(manifest, data, 'http://test', keep_going=True)
        assert submitted == [cases[3]['url']]


@pytest.mark.parametrize('kind,resolved', [('model', True), ('download', True), ('old_download', False), ('software', False), ('source_backoff', False)])
def test_audio_controller_retries_only_resolved_external_failures(tmp_path, monkeypatch, kind, resolved):
    import httpx
    from karaoke.tools import soak
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
    error = dict(provider='openai', code='openai_access_denied') if kind == 'model' else dict(
        code='alignment_failed' if kind == 'software' else 'download_failed')
    attempt = dict(status='blocked', pipeline_version='old', error=error, finished_at='2026-10-02T10:00:00+00:00')
    case = dict(url=URL, title='same source', language='English', provider='youtube', audio_attempts=[attempt],
        download_attempts=[dict(status='downloaded_audio', finished_at=(
            '2026-10-02T09:00:00+00:00' if kind == 'old_download' else '2026-10-02T11:00:00+00:00'))])
    selection = [dict(url=URL, language='English')]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=[case], provider_backoff={'openai': 2000, 'youtube': 2000 if kind == 'source_backoff' else 0},
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())))
    data = tmp_path/'jobs'; data.mkdir()
    calls = []
    def respond(request):
        if request.url.path == '/health':
            result = dict(pipeline_version='new', preparation_paused=True, running_job=None)
        elif request.url.path == '/v1/room':
            result = dict(entries=[], storage=dict(limit_bytes=1000, can_prepare=True))
        else:
            calls.append(json.loads(request.content)['url'])
            raise RuntimeError('observed submission')
        return httpx.Response(200, json=result)
    client_type = httpx.Client
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kw: client_type(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(soak.time, 'time', lambda: 1000)
    if resolved:
        with pytest.raises(RuntimeError, match='observed submission'):
            soak.run_audio(manifest, data, 'http://test', keep_going=True)
        assert calls == [URL]
    else:
        soak.run_audio(manifest, data, 'http://test', keep_going=True)
        assert calls == []
    assert json.loads(manifest.read_text())['cases'][0]['audio_attempts'] == [attempt]


@pytest.mark.parametrize('delayed_ledger', ['attempts', 'audio_attempts', 'provider'])
def test_focused_probe_respects_both_retry_ledgers(tmp_path, monkeypatch, delayed_ledger):
    import httpx
    from karaoke.tools import soak
    folder = tmp_path/'job'
    folder.mkdir()
    retained = folder/'separated.wav'
    retained.write_bytes(b'already computed')
    case = dict(url=URL, provider='youtube', attempts=[], audio_attempts=[])
    state = dict(cases=[case], provider_backoff={})
    if delayed_ledger == 'provider':
        state['provider_backoff']['youtube'] = 1030
    else:
        case[delayed_ledger].append(dict(status='external', next_retry_at=1030))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(state))
    job = dict(id='job', status='queued', input=dict(url=URL))
    def respond(request):
        result = {'pipeline_version': 'current'} if request.url.path == '/health' else job
        return httpx.Response(200, json=result)
    client_type = httpx.Client
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kwargs: client_type(transport=httpx.MockTransport(respond)))
    calls = []
    def execute(*args):
        calls.append(args[-1])
        return 0
    monkeypatch.setattr(soak, 'run_probe', execute)
    monkeypatch.setattr(soak, 'verify_outputs', lambda *args, **kwargs: dict(validation_scope='audio_and_lyrics'))
    monkeypatch.setattr(soak.time, 'time', lambda: 1000)
    with pytest.raises(ValueError, match='Provider backoff'):
        soak.probe(manifest, tmp_path, 'http://test', 'job')
    assert calls == [] and not (folder/'audio-verification.json').exists()
    assert retained.read_bytes() == b'already computed'
    monkeypatch.setattr(soak.time, 'time', lambda: 1031)
    soak.probe(manifest, tmp_path, 'http://test', 'job')
    assert len(calls) == 1 and '--audio-only' in calls[0]
    assert json.loads((folder/'audio-verification.json').read_text())['pipeline_version'] == 'current'
    assert retained.read_bytes() == b'already computed'


def test_partial_song_information_updates_preserve_language_and_title(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    pipeline.phase(song_info=dict(title='Video', language='Japanese'))
    pipeline.phase(song_info=dict(language='English'))
    pipeline.phase(song_info=dict(title='Song', artist='Singer'))
    info = json.loads((tmp_path/'progress.json').read_text())['song_info']
    assert info == dict(title='Song', artist='Singer', language='English')


def test_restart_recovers_pending_and_marks_interrupted_jobs(tmp_path):
    store = Store(tmp_path)
    interrupted, _ = store.create(JobRequest(url=URL), 'test')
    store.update(interrupted['id'], status='running')
    pending, _ = store.create(JobRequest(url=URL, force=True), 'test')
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        old = client.get(f"/v1/jobs/{interrupted['id']}").json()
        assert old['status'] == 'failed' and old['error']['code'] == 'interrupted'
        assert finished(client, pending['id'])['status'] == 'completed'


def test_queued_job_uses_executing_version_and_updates_cache_identity(tmp_path):
    store = Store(tmp_path)
    pending, _ = store.create(JobRequest(url=URL), 'before-update')
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='after-update')) as client:
        result = finished(client, pending['id'])
        assert result['pipeline_version'] == 'after-update'
        cached = client.post('/v1/jobs', json={'url': URL}).json()
        assert cached['id'] == pending['id'] and cached['cache_hit']


def test_pending_input_is_reused_after_version_change_without_merging_distinct_requests(tmp_path):
    store=Store(tmp_path)
    request=JobRequest(url=URL)
    original,_=store.create(request,'before-update')
    reused,cached=store.create(request,'after-update')
    assert cached and reused['id']==original['id'] and reused['pipeline_version']=='after-update'
    assert len(store.list(-1,'queued'))==1
    different,cached=store.create(JobRequest(url=URL,language='Japanese'),'after-update')
    assert not cached and different['id']!=original['id']


@pytest.mark.parametrize('concurrency', [1, 2])
def test_shutdown_stops_current_job_and_preserves_the_queue(tmp_path, concurrency):
    def slow_worker(folder):
        return [sys.executable, '-c', "import time;time.sleep(30)"]
    app = create_app(tmp_path, worker_command=slow_worker, version='test', worker_concurrency=concurrency)
    with TestClient(app) as client:
        current = [client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id']
                   for _ in range(concurrency)]
        queued = client.post('/v1/jobs', json={'url': 'https://youtu.be/2Vv-BfVoq4g'}).json()['id']
        deadline = time.monotonic()+5
        while (len(app.state.worker.processes) < concurrency or
               any(p is None for p in app.state.worker.processes.values())) and time.monotonic() < deadline:
            time.sleep(.02)
        processes = list(app.state.worker.processes.values())
        assert len(processes) == concurrency and all(p is not None for p in processes)
        assert client.get(f'/v1/jobs/{queued}').json()['status'] == 'queued'
    assert all(p.returncode is not None for p in processes)
    assert all(app.state.store.read(j)['error']['code'] == 'interrupted' for j in current)
    assert app.state.store.read(queued)['status'] == 'queued'
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        assert finished(client, queued)['status'] == 'completed'


def test_two_workers_keep_progress_results_and_failure_isolated_and_honor_pause(tmp_path):
    def controlled_worker(folder):
        code = '''import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1])
print('KARAOKE_PROGRESS '+json.dumps({'active_stages':['render'],'song_info':{'title':p.name}}),flush=True)
(p/'started').touch()
while not (p/'release').exists(): time.sleep(.02)
if (p/'fail').exists():
    (p/'error.json').write_text(json.dumps({'code':'fixture_failure','message':p.name}))
    sys.exit(1)
(p/'media').write_text(p.name)
artifacts={name:{'path':'media','bytes':len(p.name)} for name in ['mv','instrumental','backing_vocals','lead_vocals']}
(p/'result.json').write_text(json.dumps({'artifacts':artifacts}))
'''
        return [sys.executable, '-u', '-c', code, str(folder)]
    app = create_app(tmp_path, worker_command=controlled_worker, version='test', worker_concurrency=2)
    with TestClient(app) as client:
        ids = [client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id'] for _ in range(3)]
        deadline = time.monotonic()+5
        while not all((tmp_path/j/'started').exists() for j in ids[:2]) and time.monotonic() < deadline:
            time.sleep(.02)
        health = client.get('/health').json()
        assert health['running_jobs'] == ids[:2] and health['running_job'] == ids[0]
        assert health['preparation_concurrency'] == 2 and health['queued'] == 1
        assert client.get('/v1/room').json()['preparation_jobs'] == ids[:2]
        assert not (tmp_path/ids[2]/'started').exists()
        for j in ids[:2]:
            job = app.state.store.read(j)
            assert job['active_stages'] == ['render'] and job['song_info']['title'] == j
        hold = tmp_path/'preparation.pause'
        hold.touch()
        (tmp_path/ids[0]/'fail').touch()
        (tmp_path/ids[0]/'release').touch()
        failed = finished(client, ids[0])
        assert failed['error']['message'] == ids[0]
        assert client.get('/health').json()['running_jobs'] == ids[1:2]
        assert app.state.store.read(ids[1])['status'] == 'running'
        (tmp_path/ids[1]/'release').touch()
        assert finished(client, ids[1])['status'] == 'completed'
        assert app.state.store.read(ids[2])['status'] == 'queued'
        hold.unlink()
        (tmp_path/ids[2]/'release').touch()
        assert finished(client, ids[2])['status'] == 'completed'
        for j in ids[1:]:
            assert (tmp_path/j/'media').read_text() == j
        assert client.get('/health').json()['running_jobs'] == []


def test_parallel_workers_share_the_cache_limit(tmp_path):
    def writer(folder):
        code = '''import pathlib,sys,time
p=pathlib.Path(sys.argv[1]);(p/'started').touch()
while not (p.parent/'release').exists(): time.sleep(.02)
(p/'working.wav').write_bytes(b'x'*110000)
time.sleep(30)
'''
        return [sys.executable, '-c', code, str(folder)]
    app = create_app(tmp_path, worker_command=writer, version='test',
                     worker_concurrency=2, cache_limit_bytes=200000)
    with TestClient(app) as client:
        ids = [client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id'] for _ in range(2)]
        deadline = time.monotonic()+5
        while not all((tmp_path/j/'started').exists() for j in ids) and time.monotonic() < deadline:
            time.sleep(.02)
        assert len(app.state.worker.processes) == 2
        (tmp_path/'release').touch()
        deadline = time.monotonic()+5
        while not any(app.state.store.read(j)['status'] == 'failed' for j in ids) and time.monotonic() < deadline:
            time.sleep(.02)
        failed = [app.state.store.read(j) for j in ids if app.state.store.read(j)['status'] == 'failed']
        assert failed and all(j['error']['code'] == 'cache_full' for j in failed)
        assert app.state.store.cache.limit == 200000
    app.state.store.cache.enforce()
    assert app.state.store.cache.snapshot()['used_bytes'] < 200000


def test_preparation_hold_survives_restart_and_finishes_current_job(tmp_path):
    hold = tmp_path/'preparation.pause'
    hold.touch()
    ids = []
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        for _ in range(2):
            ids.append(client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id'])
        time.sleep(.3)
        assert client.get('/health').json()['queued'] == 2
        assert client.get('/health').json()['running_job'] is None
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        assert client.get('/health').json()['preparation_paused']
        hold.unlink()
        deadline = time.monotonic()+5
        while app.state.worker.current is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert app.state.worker.current == ids[0]
        hold.touch()
        assert finished(client, ids[0])['status'] == 'completed'
        time.sleep(.3)
        assert client.get(f'/v1/jobs/{ids[1]}').json()['status'] == 'queued'
        hold.unlink()
        assert finished(client, ids[1])['status'] == 'completed'


def test_successful_process_without_required_outputs_is_not_completed(tmp_path):
    def incomplete_worker(folder):
        return [sys.executable, '-c',
            "import pathlib,sys;pathlib.Path(sys.argv[1],'result.json').write_text('{\"artifacts\": {}}')", str(folder)]
    with TestClient(create_app(tmp_path, worker_command=incomplete_worker, version='test')) as client:
        job_id = client.post('/v1/jobs', json={'url': URL}).json()['id']
        assert finished(client, job_id)['status'] == 'failed'


def test_mixed_language_wrapping_does_not_split_a_word_at_punctuation():
    from karaoke.song_sources import transcript_lines
    text = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰 so-so-sorry,café don’t'
    for language in ('Chinese', 'Japanese'):
        lines = transcript_lines(dict(text=text), language)
        assert ''.join(''.join(lines).split()) == ''.join(text.split())
        for word in ('so-so-sorry', 'café', 'don’t'):
            assert sum(word in line for line in lines) == 1


@pytest.mark.parametrize('language,phrase,tail', [
    ('Japanese', '別に大したことないよ 話すだけで十分だよ', 'よ'),
    ('Chinese', 'You should know about it', 'it'),
    ('English', 'One two three four five six seven eight nine ten eleven', 'eleven'),
    ('Chinese', '甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申。', '申。'),
])
def test_asr_wrapping_keeps_a_singleton_tail_with_its_phrase(language, phrase, tail):
    from copy import deepcopy
    from karaoke.song_sources import transcript_lines
    transcription = dict(segments=[dict(text=phrase, start=3., end=9.),
                                   dict(text='短句', start=11., end=12.)])
    original = deepcopy(transcription)
    lines = transcript_lines(transcription, language)
    assert len(lines) == 3 and tail not in lines
    assert lines[-1] == '短句'  # Do not merge across independently observed phrases.
    assert ''.join(''.join(lines[:-1]).split()) == ''.join(phrase.split())
    assert transcription == original


def test_local_preview_can_submit_cross_origin(tmp_path):
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        headers = {'Origin': 'http://127.0.0.1:8745', 'Access-Control-Request-Method': 'POST',
                   'Access-Control-Request-Headers': 'content-type'}
        response = client.options('/v1/jobs', headers=headers)
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == headers['Origin']
        headers['Origin'] = 'https://unrelated.example'
        assert 'access-control-allow-origin' not in client.options('/v1/jobs', headers=headers).headers


def test_different_lyrics_mix_or_pipeline_version_have_distinct_cache_keys(tmp_path):
    store = Store(tmp_path)
    cases = [(JobRequest(url=URL), 'a'), (JobRequest(url=URL, mv_audio='instrumental'), 'a'),
             (JobRequest(url=URL, lyrics='第一句\n第二句', language='zh'), 'a'),
             (JobRequest(url=URL), 'b')]
    ids = []
    for request, version in cases:
        job, _ = store.create(request, version)
        ids.append(job['id'])
        store.update(job['id'], status='running')
    assert len(set(ids)) == 4
    assert cases[2][0].lyrics == ['第一句', '第二句']
    with pytest.raises(ValueError):
        JobRequest(url=URL, lyrics=[' '])


def test_catalog_requires_video_identity_and_matching_duration(tmp_path):
    (tmp_path/'lyrics').mkdir()
    (tmp_path/'samples/song').mkdir(parents=True)
    (tmp_path/'lyrics/song.json').write_text(json.dumps(dict(source_url=URL, title='Song',
        language='Chinese', lines=['测试歌词'])))
    (tmp_path/'samples/song/manifest.json').write_text(json.dumps(dict(duration_seconds=120, artists=['Artist'])))
    result = find_verified_lyrics(URL, 120.1, tmp_path)
    assert result['provenance']['kind'] == 'verified_catalog' and result['artist'] == 'Artist'
    assert find_verified_lyrics(URL, 140, tmp_path) is None
    assert find_verified_lyrics('https://www.youtube.com/watch?v=2Vv-BfVoq4g', 120, tmp_path) is None


def test_automatic_lyrics_are_marked_as_draft_and_user_overrides_survive():
    parsed = SongText(title='Detected', artist='Singer', language='English', lines=['Invented replacement'])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test-model', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        result = prepare_song_text(dict(url=URL, title='User title'), {},
            dict(language='english', segments=[dict(text='One short line', start=1, end=3),
                                               dict(text='One short line', start=4, end=6)]),
            120, catalog_root=None, model='test-model')
    assert result['title'] == 'User title' and result['lines'] == ['One short line', 'One short line']
    assert result['provenance']['needs_review'] is True
    payload = json.loads(client.return_value.beta.chat.completions.parse.call_args.kwargs['messages'][1]['content'])
    assert 'segments' not in payload and 'transcription' not in payload


def test_lyric_text_survives_metadata_refusal_without_resubmitting_content():
    from openai import ContentFilterFinishReasonError
    with patch('karaoke.song_sources.OpenAI') as client:
        parse = client.return_value.beta.chat.completions.parse
        parse.side_effect = ContentFilterFinishReasonError()
        result = prepare_song_text(dict(url=URL), dict(title='Source title'),
            dict(language='english', segments=[dict(text="We're here. We're here again!", start=1, end=8)]),
            120, catalog_root=None, model='test-model')
    assert result['lines'] == ["We're here.", "We're here again!"]
    assert result['title'] == 'Source title'
    assert result['provenance']['metadata_error']['code'] == 'openai_content_rejected'
    assert parse.call_count == 1


def test_song_metadata_survives_downstream_retries_without_freezing_transcription(tmp_path):
    import soundfile as sf
    pipeline = Pipeline(dict(url=URL), tmp_path, catalog_root=None)
    (tmp_path/'asr').mkdir()
    sf.write(tmp_path/'vocals-16k.wav', np.ones(5000)*.1, 1000)
    def prepare(text):
        (tmp_path/'asr/transcription.json').write_text(json.dumps(dict(language='English',
            segments=[dict(text=text, start=1., end=2.)], words=[dict(word=text, start=1., end=2.)])))
        pipeline.prepare_lyrics(dict(title='Source title'), 5.)
        return json.loads((tmp_path/'lyrics.json').read_text())
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        parsed=SongText(title='Song', artist='原歌手')))], model='test-model', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        parse = client.return_value.beta.chat.completions.parse
        parse.return_value = response
        first = prepare('First draft')
        # A downstream retry must not re-query a stochastic metadata service,
        # but new ASR text still flows through the lyric preparation step.
        parse.side_effect = AssertionError('metadata was requested again')
        second = prepare('Corrected draft')
    assert first['artist'] == second['artist'] == '原歌手'
    assert first['lines'] == ['First draft'] and second['lines'] == ['Corrected draft']
    assert parse.call_count == 1


@pytest.mark.parametrize('change', ['metadata', 'model', 'damaged_output'])
def test_song_metadata_checkpoint_invalidates_changed_inputs_or_output(tmp_path, change):
    from karaoke.stage_cache import StageCache
    cache = StageCache(tmp_path)
    metadata, model = dict(title='Source title'), 'test-model'
    transcription = dict(language='English', text='Test lyrics')
    def response(artist):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=SongText(title='Song', artist=artist)))], model=model, usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        parse = client.return_value.beta.chat.completions.parse
        parse.side_effect = [response('First singer'), response('Second singer')]
        prepare_song_text(dict(url=URL), metadata, transcription, 120,
                          catalog_root=None, model=model, cache=cache)
        if change == 'metadata':
            metadata['description'] = 'A different recording'
        elif change == 'model':
            model = 'updated-model'
        else:
            (tmp_path/'song-metadata.json').write_text('{}')
        result = prepare_song_text(dict(url=URL), metadata, transcription, 120,
                                   catalog_root=None, model=model, cache=cache)
    assert result['artist'] == 'Second singer' and parse.call_count == 2


def test_failed_metadata_request_does_not_poison_the_checkpoint(tmp_path):
    import httpx
    from openai import APITimeoutError
    from karaoke.stage_cache import StageCache
    cache = StageCache(tmp_path)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        parsed=SongText(title='Song', artist='Singer')))], model='test-model', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        parse = client.return_value.beta.chat.completions.parse
        parse.side_effect = [APITimeoutError(request=httpx.Request('POST', 'https://example.test')), response]
        first = prepare_song_text(dict(url=URL), {}, dict(language='English', text='Test lyrics'),
                                  120, catalog_root=None, model='test-model', cache=cache)
        second = prepare_song_text(dict(url=URL), {},
                                   dict(language='English', text='Test lyrics'),
                                   120, catalog_root=None, model='test-model', cache=cache)
    assert first['provenance']['metadata_error']['code'] == 'openai_timeout'
    assert second['artist'] == 'Singer' and parse.call_count == 2


@pytest.mark.parametrize('source', ['asr', 'description', 'user'])
@pytest.mark.parametrize('language,lines,expected', [
    ('English', ['“', 'Will you stay?”', 'I’ll wait here!'], ['“Will you stay?”', 'I’ll wait here!']),
    ('Japanese', ['「', 'また会おう！', '」', '待っている。'], ['「また会おう！」', '待っている。']),
])
def test_lyric_wrapping_keeps_standalone_punctuation_without_empty_alignment_rows(source, language, lines, expected):
    request = dict(url=URL, language=language)
    if source == 'user':
        request['lyrics'] = lines
    parsed = SongText(title='Song', artist='Singer',
                      lyric_line_numbers=list(range(1, len(lines)+1)) if source == 'description' else [])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        result = prepare_song_text(request, dict(description='\n'.join(lines)),
            dict(language=language, segments=[dict(text=line) for line in lines]),
            120, catalog_root=None, model='test')
    assert result['lines'] == expected
    assert ''.join(''.join(result['lines']).split()) == ''.join(''.join(lines).split())
    if source == 'user':
        with pytest.raises(ValueError, match='歌词为空或过长'):
            prepare_song_text(dict(request, lyrics=['…', '“”']), {}, {}, 120, catalog_root=None, model='test')


@pytest.mark.parametrize('language,lines', [
    ('Chinese', ['踏过山河去寻你', '晚风吹过小树林', '踏过山河去寻你']),
    ('Japanese', ['小さな猫が静かな庭の木陰で眠っている', '白い雲を見ながら静かな道をゆっくり歩いている',
                  '小さな猫が静かな庭の木陰で眠っている']),
    ('English', ['We walk along the quiet river while the moon shines above',
                 'The stars are bright', 'We walk along the quiet river while the moon shines above']),
])
@pytest.mark.parametrize('selected,accepted', [
    ([3, 4, 5], True), ([], False), ([3, 99], False), ([4, 3], False), ([3, 3], False),
])
def test_description_lyrics_select_existing_lines_without_rewriting(tmp_path, selected, accepted, language, lines):
    description = '\n'.join(['视频介绍', '作词：示例', *lines, '订阅频道'])
    parsed = SongText(title='Song', artist='Singer', lyric_line_numbers=selected)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        result = prepare_song_text(dict(url=URL), dict(description=description),
            dict(language=language, segments=[dict(text='错字转录')]),
            120, catalog_root=None, model='test')
    assert result['lines'] == (lines if accepted else ['错字转录'])
    assert result['provenance']['needs_review'] is True
    assert result['provenance']['kind'] == ('video_description' if selected == [3, 4, 5] else 'asr_draft')
    if selected == [3, 4, 5]:
        assert result['provenance']['lyric_line_numbers'] == selected
        assert result['provenance']['description_sha256'] == hashlib.sha256(description.encode()).hexdigest()
        # Published words are not automatically valid for this performance.
        # The real pipeline must still reject an unsupported description block.
        import soundfile as sf
        pipeline = Pipeline(dict(url=URL), tmp_path/'job', catalog_root=None)
        (pipeline.output/'asr').mkdir()
        (pipeline.output/'asr/transcription.json').write_text(json.dumps(dict(language=language,
            segments=[dict(text='完全不同', start=1, end=3)], words=[dict(word='完全不同', start=1, end=3)])))
        sf.write(pipeline.output/'vocals-16k.wav', np.ones(5000)*.1, 1000)
        with patch('karaoke.song_sources.OpenAI') as client:
            client.return_value.beta.chat.completions.parse.return_value = response
            with pytest.raises(StageFailure, match='匹配不足'):
                pipeline.prepare_lyrics(dict(description=description), 5)


@pytest.mark.parametrize('evidence', ['independent', 'same_request', 'mismatched', 'user_supplied'])
def test_bad_description_uses_only_independently_corroborated_transcript(tmp_path, evidence):
    import soundfile as sf
    prose = ['今天在海边旅行', '明年我们搬去另一个城市']
    lines = ['清风吹过树林', '晚霞染红天空']
    request = dict(url=URL)
    if evidence == 'user_supplied':
        request.update(lyrics=prose, language='zh')
    pipeline = Pipeline(request, tmp_path/'job', catalog_root=None)
    folder = pipeline.output/'asr'
    folder.mkdir()
    words = [dict(word=text, start=1+2*i, end=2+2*i) for i, text in enumerate(lines)]
    transcription = dict(language='Chinese', segments=[dict(text=' '.join(lines), start=0., end=5.)], words=words)
    if evidence == 'mismatched':
        transcription['segments'][0]['text'] = '另一段完全不同的转录'
    (folder/'transcription.json').write_text(json.dumps(transcription))
    (folder/'run.json').write_text(json.dumps(dict(recovery=dict(
        triggered=evidence != 'same_request', chunks=[dict(word_times_discarded=False)]))))
    sf.write(pipeline.output/'vocals-16k.wav', np.ones(5000)*.1, 1000)
    parsed = SongText(title='Song', artist='Singer', lyric_line_numbers=[1, 2])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        if evidence != 'independent':
            with pytest.raises(StageFailure, match='匹配不足'):
                pipeline.prepare_lyrics(dict(description='\n'.join(prose)), 5.)
            return
        pipeline.prepare_lyrics(dict(description='\n'.join(prose)), 5.)
    result = json.loads((pipeline.output/'lyrics.json').read_text())
    assert result['lines'] == [' '.join(lines)]
    assert result['title'] == 'Song' and result['artist'] == 'Singer'
    provenance = result['provenance']
    assert provenance['kind'] == 'asr_draft' and provenance['needs_review']
    assert provenance['independent_word_support'] == 1.
    assert provenance['rejected_description']['lines'] == prose
    assert provenance['rejected_description']['anchor_coverage'] < .55
    anchors = json.loads((pipeline.output/'anchors.json').read_text())
    assert [(c['start'], c['end']) for c in anchors] == [(1., 4.)]


@pytest.mark.parametrize('source', ['independent', 'initial_excerpt', 'same_request', 'user', 'complete_description'])
def test_matching_description_excerpt_cannot_replace_the_whole_song(tmp_path, source):
    import soundfile as sf
    lines = ['春风吹过青青山坡', '小河映着远处灯火', '旅人沿着石阶远行', '我们相约明天再见']
    supplied = lines if source == 'complete_description' else lines[-1:]
    request = dict(url=URL)
    if source == 'user':
        request.update(lyrics=supplied, language='zh')
    pipeline = Pipeline(request, tmp_path/'job', catalog_root=None)
    folder = pipeline.output/'asr'
    folder.mkdir()
    words = [dict(word=text, start=1+2*i, end=2+2*i) for i, text in enumerate(lines)]
    transcription = dict(language='Chinese', segments=[dict(text='\n'.join(lines), start=0., end=9.)], words=words)
    if source == 'initial_excerpt':
        (folder/'initial-transcription.json').write_text(json.dumps(dict(language='Chinese',
            segments=[dict(text=lines[-1], start=7., end=8.)], words=words[-1:])))
        words[-1] = dict(words[-1], word='他们相约明天再见')
    (folder/'transcription.json').write_text(json.dumps(transcription))
    (folder/'run.json').write_text(json.dumps(dict(recovery=dict(
        triggered=source != 'same_request', chunks=[dict(word_times_discarded=False)]))))
    sf.write(pipeline.output/'vocals-16k.wav', np.full(10000, .1), 1000)
    parsed = SongText(title='Song', artist='Singer', lyric_line_numbers=list(range(1, len(supplied)+1)))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        if source in ('same_request', 'user'):
            with pytest.raises(StageFailure, match='未覆盖完整演唱'):
                pipeline.prepare_lyrics(dict(description='\n'.join(supplied)), 10.)
            return
        pipeline.prepare_lyrics(dict(description='\n'.join(supplied)), 10.)
    result = json.loads((pipeline.output/'lyrics.json').read_text())
    assert result['lines'] == lines
    if source in ('independent', 'initial_excerpt'):
        rejected = result['provenance']['rejected_description']
        assert rejected['anchor_coverage'] == 1.
        assert rejected['lyric_coverage']['uncovered_passages']
        if source == 'independent':
            assert rejected['lyric_coverage']['recognized_coverage'] == .25
    else:
        assert result['provenance']['kind'] == 'video_description'


def test_lyric_coverage_detects_a_missing_verse_despite_high_overall_match():
    from karaoke.song_sources import lyric_coverage
    lines = ['A small bird flies above the river', 'The wind carries our voices toward the hill',
             'Morning sunshine lights the empty houses', 'We will find the garden in the valley']
    words = [dict(word=text, start=1+2*i, end=2+2*i) for i, text in enumerate(lines)]
    cues = [dict(text=w['word'], start=w['start'], end=w['end']) for w in words]
    assert lyric_coverage(cues, words, 'English', 10.)['status'] == 'complete'
    missing = lyric_coverage([cues[0], *cues[2:]], words, 'English', 10.)
    assert missing['recognized_coverage'] > .55
    assert missing['status'] == 'incomplete' and len(missing['uncovered_passages']) == 1
    adlib = [*words, dict(word='hey', start=9., end=9.5)]
    assert lyric_coverage(cues, adlib, 'English', 10.)['status'] == 'complete'
    # Even a speech review must not suppress text inside an actual lyric cue.
    reviewed = lyric_coverage(cues, words, 'English', 10., speech_intervals=[(0., 10.)])
    assert reviewed['excluded_speech_intervals'] == []
    assert reviewed['recognized_units'] == lyric_coverage(cues, words, 'English', 10.)['recognized_units']


def vocal_review_response(*, singing=False, speech=True, uncertain=False):
    return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(dict(
        singing_present=singing, speech_present=speech, uncertain_vocals=uncertain))))])


@pytest.mark.parametrize('verdict', ['speech', 'singing', 'mixed', 'uncertain', 'instrumental', 'invalid'])
def test_uncovered_dialogue_requires_two_complete_audio_reviews(tmp_path, monkeypatch, verdict):
    import soundfile as sf
    from karaoke.vocal_review import speech_intervals
    lines = ['清风吹过树林', '晚霞染红天空']
    speech = dict(word='Please come back tomorrow because the road is closed tonight', start=1., end=4.)
    words = [speech, *[dict(word=t, start=6.+3*i, end=8.+3*i) for i, t in enumerate(lines)]]
    pipeline = Pipeline(dict(url=URL), tmp_path/'job', catalog_root=None)
    folder = pipeline.output/'asr'
    folder.mkdir()
    (folder/'transcription.json').write_text(json.dumps(dict(language='Chinese',
        segments=[dict(text='\n'.join(lines), start=6., end=11.)], words=words)))
    (folder/'run.json').write_text(json.dumps(dict(recovery=dict(triggered=False))))
    for path in (pipeline.output/'vocals-16k.wav', pipeline.output/'source-44100.wav', pipeline.vocals):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.full(12*16000, .1), 16000)
    monkeypatch.setattr('karaoke.pipeline.prepare_song_text', lambda *a, **k:
        dict(title='Example', artist='', language='Chinese', lines=lines, provenance=dict(kind='video_description')))
    response = vocal_review_response(singing=verdict in ('singing', 'mixed'),
        speech=verdict not in ('singing', 'instrumental'), uncertain=verdict == 'uncertain')
    if verdict == 'invalid':
        response['choices'][0]['message']['content'] = 'I will listen to it.'
    with patch('openai.OpenAI') as client:
        client.return_value.chat.completions.create.side_effect = [
            SimpleNamespace(model_dump=lambda **k: vocal_review_response()),
            SimpleNamespace(model_dump=lambda **k: response)]
        if verdict == 'speech':
            pipeline.prepare_lyrics({}, 12.)
        else:
            with pytest.raises(StageFailure, match='未覆盖完整演唱'):
                pipeline.prepare_lyrics({}, 12.)
        assert client.return_value.chat.completions.create.call_count == 2
    result = json.loads((pipeline.output/'lyrics.json').read_text())
    report = json.loads((folder/'lyric-repeats.json').read_text())
    assert result['lines'] == lines and result['provenance']['kind'] == 'video_description'
    assert report['words'] == words  # Raw recognition survives the coverage review.
    intervals = speech_intervals(pipeline.output, report['non_singing_reviews'])
    assert intervals == ([(.5, 4.5)] if verdict == 'speech' else [])
    assert report['lyric_coverage']['status'] == ('complete' if verdict == 'speech' else 'incomplete')
    # A retry reuses both positive and negative decisions rather than fishing
    # for a different answer from the same model and audio.
    with patch('openai.OpenAI') as client:
        if verdict == 'speech':
            pipeline.prepare_lyrics({}, 12.)
        else:
            with pytest.raises(StageFailure, match='未覆盖完整演唱'):
                pipeline.prepare_lyrics({}, 12.)
        client.assert_not_called()


def test_audio_reviews_revalidate_sources_and_reuse_only_unchanged_inputs(tmp_path):
    import soundfile as sf
    from copy import deepcopy
    from karaoke.vocal_review import SOURCES, review_gaps, speech_intervals
    words = [dict(word='spoken dialogue', start=1., end=2.)]
    for source in SOURCES:
        path = tmp_path/source
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.full(3*16000, .1), 16000)
    with patch('openai.OpenAI') as client:
        client.return_value.chat.completions.create.return_value.model_dump.return_value = vocal_review_response()
        reviews = review_gaps(tmp_path, [dict(start=0., end=3.)], words)
        assert speech_intervals(tmp_path, reviews) == [(.5, 2.5)]
        for change in ('missing_view', 'input_hash', 'prompt', 'model', 'bounds'):
            altered = deepcopy(reviews)
            if change == 'missing_view':
                altered[0]['views'].pop()
            elif change == 'bounds':
                altered[0]['end'] = 2.6
            else:
                altered[0]['views'][0][{'input_hash':'input_sha256'}.get(change, change)] = 'changed'
            with pytest.raises(ValueError):
                speech_intervals(tmp_path, altered)
        sf.write(tmp_path/SOURCES[0], np.full(3*16000, .2), 16000)
        with pytest.raises(ValueError, match='current input'):
            speech_intervals(tmp_path, reviews)
        refreshed = review_gaps(tmp_path, [dict(start=0., end=3.)], words)
        assert client.return_value.chat.completions.create.call_count == 3
        assert speech_intervals(tmp_path, refreshed) == [(.5, 2.5)]
        (tmp_path/SOURCES[1]).unlink()
        assert review_gaps(tmp_path, [dict(start=0., end=3.)], words) == []


@pytest.mark.parametrize('supplied', [False, True])
def test_description_repeats_require_asr_support_and_preserve_supplied_lyrics(tmp_path, monkeypatch, supplied):
    import soundfile as sf
    from karaoke.song_sources import missing_repetitions, restore_repetitions
    from anchors_from_asr import anchors_from_words
    lines = ['An opening by the river', 'Golden lanterns in the window,',
             'Every sparrow flies back home!', 'Winter carries distant thunder', 'The closing of the curtain']
    performed = lines[:4]+lines[1:3]+lines[4:]
    words = [dict(word=text, start=1.+2*i, end=2.+2*i) for i, text in enumerate(performed)]
    pipeline = Pipeline(dict(url=URL), tmp_path/'job', catalog_root=None)
    (pipeline.output/'asr').mkdir()
    (pipeline.output/'asr/transcription.json').write_text(json.dumps(dict(language='English',
        segments=[dict(text=' '.join(performed), start=1., end=14.)], words=words)))
    sf.write(pipeline.output/'vocals-16k.wav', np.full(16000, .1), 1000)
    monkeypatch.setattr('karaoke.pipeline.prepare_song_text', lambda *args, **kwargs:
        dict(title='Example', artist='', language='English', lines=lines,
             provenance=dict(kind='user_supplied' if supplied else 'video_description')))
    if supplied:
        with pytest.raises(StageFailure, match='重复唱段'):
            pipeline.prepare_lyrics({}, 16.)
    else:
        pipeline.prepare_lyrics({}, 16.)
    lyrics = json.loads((pipeline.output/'lyrics.json').read_text())
    report = json.loads((pipeline.output/'asr/lyric-repeats.json').read_text())
    assert lyrics['lines'] == (lines if supplied else performed)
    assert report['reference_lines'] == lines and report['words'] == words
    assert report['status'] == ('failed' if supplied else 'complete')
    # The original ordered timeline looks valid while still missing a refrain.
    original = anchors_from_words(lines, words, 'English')
    assert missing_repetitions(lines, original, words, 'English', 16.)
    if not supplied:
        anchors = json.loads((pipeline.output/'anchors.json').read_text())
        assert [(c['start'], c['end']) for c in anchors] == [(w['start'], w['end']) for w in words]
        assert report['expansions'][0]['source_range'] == [1, 3]
        assert not missing_repetitions(lines, anchors, words, 'English', 16.)
        again, _, next_report = restore_repetitions(performed, anchors, words, 'English', 16., allow_expansion=True)
        assert again == performed and not next_report['expansions']


@pytest.mark.parametrize('invalid', ['unrelated', 'nonfinite', 'zero_duration', 'backward', 'moves_existing', 'outside_gap'])
def test_repetition_recovery_rejects_unsupported_evidence(invalid, monkeypatch):
    from karaoke.song_sources import restore_repetitions
    from anchors_from_asr import anchors_from_words
    lines = ['Opening by the river', 'Golden lanterns in the window',
             'Every sparrow flies back home', 'Winter brings distant thunder', 'Closing of the curtain']
    performed = lines[:4]+lines[1:3]+lines[4:]
    words = [dict(word=text, start=1.+2*i, end=2.+2*i) for i, text in enumerate(performed)]
    anchors = anchors_from_words(lines, words, 'English')
    if invalid == 'backward':
        with pytest.raises(ValueError, match='unordered'):
            restore_repetitions(lines, anchors, words[::-1], 'English', 16., allow_expansion=True)
        return
    if invalid in ('moves_existing', 'outside_gap'):
        def moved(texts, *args, **kwargs):
            result = anchors_from_words(texts, *args, **kwargs)
            if len(texts) == len(performed):
                index = 0 if invalid == 'moves_existing' else 4
                result[index].update(start=14., end=15.)
            return result
        monkeypatch.setattr('karaoke.anchors_from_asr.anchors_from_words', moved)
    else:
        for word in words[4:6]:
            if invalid == 'unrelated':
                word['word'] = 'Speech about cooking an entirely different meal'
            elif invalid == 'nonfinite':
                word['end'] = float('nan')
            else:
                word['end'] = word['start']
    result, _, report = restore_repetitions(lines, anchors, words, 'English', 16., allow_expansion=True)
    assert result == lines and not report['expansions']
    if invalid in ('moves_existing', 'outside_gap'):
        assert report['status'] == 'failed'


@pytest.mark.parametrize('same_text', [True, False])
def test_reused_asr_text_keeps_silence_boundaries_but_uses_recovered_word_times(tmp_path, same_text):
    import soundfile as sf
    pipeline = Pipeline(dict(url=URL), tmp_path/'job', catalog_root=None)
    folder = pipeline.output/'asr'
    folder.mkdir()
    segments = [dict(text='Hello there.', start=1., end=2.),
                dict(text='Quiet ending.', start=2.5, end=3.),
                dict(text='Invented credits.', start=5., end=5.5)]
    text = ' '.join(s['text'] for s in segments) if same_text else 'Different uncorroborated transcript.'
    words = [dict(word=text, start=a, end=b) for text,a,b in
             [('Hello',1.1,1.4), ('there.',1.4,1.8), ('Quiet',2.5,2.7), ('ending.',2.7,3.)]]
    (folder/'initial-transcription.json').write_text(json.dumps(dict(
        language='English', segments=segments, words=[dict(word='Broken times', start=0., end=0.)])))
    (folder/'transcription.json').write_text(json.dumps(dict(language='English', text=text,
        segments=[dict(text=text, start=0., end=6.)], words=words)))
    (folder/'run.json').write_text(json.dumps(dict(recovery=dict(triggered=True,
        text_selection=dict(selected='initial_transcription'), chunks=[dict(word_times_discarded=False)]))))
    audio = np.zeros(6000)
    audio[1000:2000] = .1
    audio[2500:3000] = 1e-7
    sf.write(pipeline.output/'vocals-16k.wav', audio, 1000, subtype='FLOAT')
    parsed = SongText(title='Song', artist='Singer')
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        if not same_text:
            with pytest.raises(StageFailure, match='匹配不足'):
                pipeline.prepare_lyrics({}, 6.)
            assert json.loads((pipeline.output/'lyrics.json').read_text())['lines'] == [text]
            return
        pipeline.prepare_lyrics({}, 6.)
    assert json.loads((pipeline.output/'lyrics.json').read_text())['lines'] == ['Hello there.', 'Quiet ending.']
    anchors = json.loads((pipeline.output/'anchors.json').read_text())
    assert [(c['start'],c['end']) for c in anchors] == [(1.1,1.8), (2.5,3.)]
    rejected = json.loads((folder/'filtered.json').read_text())['rejected']
    assert [(r['reason'],r['item']['text']) for r in rejected] == [('digital_silence','Invented credits.')]


@pytest.mark.parametrize('initial_state', ['valid', 'backward', 'wrong_language', 'out_of_bounds'])
def test_independent_lyrics_can_reuse_valid_original_asr_anchors(tmp_path, initial_state):
    import soundfile as sf
    lines = ['清风吹过树林', '晚霞染红天空']
    pipeline = Pipeline(dict(url=URL, lyrics=lines, language='zh'), tmp_path/'job', catalog_root=None)
    folder = pipeline.output/'asr'
    folder.mkdir()
    recovered = dict(language='Chinese', segments=[dict(text='完全不同', start=1, end=4)],
                     words=[dict(word='完全不同', start=1, end=4)])
    words = [dict(word=text, start=1+2*i, end=2+2*i) for i, text in enumerate(lines)]
    if initial_state == 'backward':
        words.reverse()
    if initial_state == 'out_of_bounds':
        for word in words:
            word['end'] = 10
    initial = dict(language='English' if initial_state == 'wrong_language' else 'Chinese',
                   segments=[], words=words)
    (folder/'transcription.json').write_text(json.dumps(recovered))
    (folder/'initial-transcription.json').write_text(json.dumps(initial))
    sf.write(pipeline.output/'vocals-16k.wav', np.ones(5000)*.1, 1000)
    if initial_state != 'valid':
        with pytest.raises(StageFailure, match='匹配不足'):
            pipeline.prepare_lyrics({}, 5)
        return
    pipeline.prepare_lyrics({}, 5)
    anchors = json.loads((pipeline.output/'anchors.json').read_text())
    assert [(c['text'],c['start'],c['end']) for c in anchors] == [(lines[0],1,2),(lines[1],3,4)]
    assert json.loads((folder/'filtered.json').read_text())['anchor_source'] == 'initial_transcription'
    assert json.loads((pipeline.output/'lyrics.json').read_text())['lines'] == lines


@pytest.mark.parametrize('bad_timing', [False, True])
@pytest.mark.parametrize('location', ['middle', 'opening', 'ending', 'unanchored'])
@pytest.mark.parametrize('backend', ['api', 'local'])
def test_missing_anchors_use_cached_context_without_lyric_prompt(tmp_path, monkeypatch, bad_timing, location, backend):
    import soundfile as sf
    from karaoke.anchor_recovery import recover_anchor_words
    from karaoke.stage_cache import StageCache
    lines = ['first line', 'the missing middle passage', 'another missing longer passage', 'last line']
    audio = tmp_path/'audio.wav'
    sf.write(audio, np.ones(12000)*.1, 1000)
    known = {'middle': [(lines[0],1.,2.), (lines[-1],8.,9.)],
             'opening': [(lines[-1],8.,9.)], 'ending': [(lines[0],1.,2.)],
             'unanchored': []}[location]
    words = [dict(word=text, start=a, end=b) for text,a,b in
             [('序',0.,.2), *known, ('終',11.,11.2)]]
    begin = 0. if location == 'opening' else .4
    recovered = [dict(word=text, start=a-begin, end=b-begin) for text,a,b in
                 [(lines[0],1.,2.), (lines[1],3.,4.), (lines[2],5.,6.), (lines[3],8.,9.)]]
    if bad_timing:
        recovered.reverse()
    calls = []
    def create(**kwargs):
        assert 'prompt' not in kwargs and kwargs['language'] == 'en'
        calls.append(kwargs['file'].name)
        return SimpleNamespace(model_dump=lambda:dict(language='english', words=recovered,
            segments=[dict(text=' '.join(lines), start=.6, end=8.6)]))
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    if backend == 'local':
        from karaoke import anchor_recovery
        monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
        monkeypatch.setattr(anchor_recovery.local_asr, 'configuration', lambda: dict(backend='local'))
        def local(clip, language, target):
            assert language == 'English'
            with clip.open('rb') as stream:
                return create(file=stream, language='en').model_dump()
        monkeypatch.setattr(anchor_recovery.local_asr, 'cached_recognition', local)
        client = None
    output = tmp_path/'recovery.json'
    args = (client, audio, lines, words, 'English', output, StageCache(tmp_path), .55, 12.)
    result, _, score = recover_anchor_words(*args)
    if bad_timing or location == 'unanchored':
        assert result == words and score < .55
    else:
        assert score == 1.
        expected = [dict(w, start=w['start']+begin, end=w['end']+begin) for w in recovered]
        if location != 'opening':
            expected.insert(0, words[0])
        if location != 'ending':
            expected.append(words[-1])
        assert result == expected
    count = int(location != 'unanchored')
    assert len(calls) == count and not list(tmp_path.glob('*.mp3'))
    assert recover_anchor_words(*args)[0] == result and len(calls) == count
    if count:
        assert json.loads(output.read_text())['attempts'][0]['reused'] is True


def test_long_anchor_gap_can_use_cached_mix_and_keep_usable_asr_segments(tmp_path):
    import soundfile as sf
    from karaoke.anchor_recovery import recover_anchor_words
    from karaoke.stage_cache import StageCache
    lines = ['opening words', 'missing middle phrase', 'another missing sequence', 'closing words']
    vocals, mixture = tmp_path/'vocals.wav', tmp_path/'mix.wav'
    for audio in (vocals, mixture):
        sf.write(audio, np.ones(82000)*.1, 1000)
    words = [dict(word=lines[0], start=1., end=2.), dict(word=lines[-1], start=78., end=79.)]
    begin = 79.6-45
    good = [dict(word=text, start=a-begin, end=b-begin) for text,a,b in
            [(lines[1],40.,45.), (lines[2],60.,65.), (lines[3],78.,79.)]]
    calls = []

    def create(**kwargs):
        assert 'prompt' not in kwargs
        calls.append(kwargs['file'].name)
        if len(calls) == 1:
            return SimpleNamespace(model_dump=lambda:dict(language='english', text='', segments=[], words=[]))
        collapsed = [dict(word='noise', start=.5, end=.5) for _ in range(8)]
        bad_segment = dict(text='noise '*8, start=0., end=1., no_speech_prob=.99)
        return SimpleNamespace(model_dump=lambda:dict(language='english',
            words=collapsed+good, segments=[bad_segment]+[
                dict(text=w['word'], start=w['start'], end=w['end']) for w in good]))

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    output = tmp_path/'recovery.json'
    args = (client, vocals, lines, words, 'English', output, StageCache(tmp_path), .55, 82.)
    result, cues, score = recover_anchor_words(*args, fallback_audio=mixture)
    assert score == 1. and [c['text'] for c in cues] == lines
    assert result[0] == words[0]
    np.testing.assert_allclose([(w['start'],w['end']) for w in result[1:]],
                               [(40.,45.), (60.,65.), (78.,79.)], rtol=0, atol=1e-12)
    report = json.loads(output.read_text())
    assert len(calls) == 2 and all(a['end']-a['begin'] <= 45 for a in report['attempts'])
    assert [a['audio'] for a in report['attempts']] == [str(vocals), str(mixture)]
    assert len(report['attempts'][-1]['rejected_segments']) == 1
    assert recover_anchor_words(*args, fallback_audio=mixture)[0] == result and len(calls) == 2
    # Changing only the fallback PCM invalidates its response, not the vocal response.
    sf.write(mixture, np.ones(82000)*.2, 1000)
    assert recover_anchor_words(*args, fallback_audio=mixture)[0] == result and len(calls) == 3


@pytest.mark.parametrize('language,text', [
    ('English', "One two three four five six seven eight nine ten eleven twelve. Don't lose déjà vu!"),
    ('Chinese', '甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸，今天hello world不要漏字。'),
    ('Japanese', 'あいうえおかきくけこさしすせそたちつてとなにぬねの。もう一度。'),
])
def test_automatic_line_breaking_only_adds_whitespace(language, text):
    parsed = SongText(title='Detected', artist='', language=language, lines=['Changed'])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        result = prepare_song_text(dict(url=URL), {}, dict(language=language, segments=[dict(text=text)]),
                                   120, catalog_root=None, model='test-model')
    assert ''.join(''.join(result['lines']).split()) == ''.join(text.split())
    assert len(result['lines']) > 1
    assert max(map(len, result['lines'])) < 80


@pytest.mark.parametrize('extra,tail,rate,valid', [(0,0.,48000,True), (37,0.,48000,True),
    (1153,0.,48000,False), (-1,0.,48000,False), (37,.02,48000,False), (0,0.,44100,False)])
def test_preview_padding_never_hides_missing_or_non_silent_audio(tmp_path, extra, tail, rate, valid):
    path = tmp_path/'preview.mp3'
    path.write_bytes(b'fixture')
    samples = np.zeros((10000+extra,2))
    samples[:min(10000,len(samples))] = .1
    if extra > 0:
        samples[10000:] = tail
    with patch('karaoke.exact_stems.decode_source', return_value=(samples, rate, {})):
        if valid:
            report = validate_preview(path, 10000, 2, 48000)
            assert report['decoded_length_delta_samples'] == extra
        else:
            with pytest.raises(ValueError):
                validate_preview(path, 10000, 2, 48000)


@pytest.mark.parametrize('rate,frames,channels', [(48000,57600,2), (48000,57601,2),
    (48000,57629,2), (48000,57646,2), (48000,57647,2), (44100,57629,1)])
def test_preview_encoder_preserves_partial_final_frames(tmp_path, rate, frames, channels):
    import soundfile as sf
    from karaoke.exact_stems import encode_preview

    tone = .1*np.sin(2*np.pi*440*np.arange(frames)/rate)
    source, output = tmp_path/'master.wav', tmp_path/'preview.mp3'
    sf.write(source, np.repeat(tone[:,None], channels, axis=1), rate, subtype='DOUBLE')
    report = encode_preview(source, output, .5, frames, channels)
    assert report['decoded_length_delta_samples'] == 0
    assert report['padding_peak'] == 0
    assert .04 < report['decoded_peak'] < .06
    quieter = encode_preview(source, tmp_path/'quieter.mp3', .25, frames, channels)
    assert quieter['decoded_peak'] == pytest.approx(report['decoded_peak']*.5, rel=.05)
    assert sf.info(source).frames == frames and sf.info(source).subtype == 'DOUBLE'


def test_preview_clipping_reports_measured_peak(tmp_path):
    from karaoke.exact_stems import PreviewClippingError

    with patch('karaoke.exact_stems.decode_source', return_value=(np.full((10000, 2), 1.02), 48000, {})):
        with pytest.raises(PreviewClippingError) as error:
            validate_preview(tmp_path/'clipped.mp3', 10000, 2, 48000)
    assert error.value.peak == 1.02


def test_preview_group_reencodes_every_track_with_the_same_gain(tmp_path):
    from karaoke.exact_stems import encode_previews, PreviewClippingError

    calls = []
    def encode(source, output, gain, frames, channels):
        peak = gain * (1.4 if source.name == 'backing.wav' else .5)
        calls.append((source.name, gain))
        if peak >= 1:
            raise PreviewClippingError(output, peak)
        return dict(decoded_peak=peak, encoded_gain=gain)

    sources = {name: tmp_path/f'{name}.wav' for name in ('vocals', 'backing')}
    with patch('karaoke.exact_stems.encode_preview', side_effect=encode):
        gain, previews = encode_previews(sources, tmp_path, .9, 10000, 2)
    assert gain < .9 and set(previews) == set(sources)
    assert all(p['encoded_gain'] == gain and p['decoded_peak'] < 1 for p in previews.values())
    assert {name for name, used_gain in calls if used_gain == gain} == {'vocals.wav', 'backing.wav'}
    assert ('vocals.wav', .9) in calls


def test_preview_group_keeps_format_failures_and_bounds_clipping_retries(tmp_path):
    from karaoke.exact_stems import encode_previews, PreviewClippingError

    sources = {'vocals': tmp_path/'vocals.wav'}
    with patch('karaoke.exact_stems.encode_preview', side_effect=ValueError('Missing audio')) as encode:
        with pytest.raises(ValueError, match='Missing audio'):
            encode_previews(sources, tmp_path, .9, 10000, 2)
        assert encode.call_count == 1
    with patch('karaoke.exact_stems.encode_preview', side_effect=PreviewClippingError('vocals.mp3', 1.02)) as encode:
        with pytest.raises(PreviewClippingError):
            encode_previews(sources, tmp_path, .9, 10000, 2)
        assert encode.call_count == 3


def test_cross_line_refinement_falls_back_without_inventing_times():
    prior = [dict(text=text, units=[dict(text=text, start=start, end=start+.3)])
             for text, start in [('first',1.), ('second',1.5), ('third',3.)]]
    refined = json.loads(json.dumps(prior))
    refined[1]['units'][0]['start'] = .98
    refined[2]['units'][0]['start'] = 2.95
    fixed, lines = protect_line_order(refined, prior)
    assert lines == [1,2]
    assert [c['units'][0]['start'] for c in fixed] == [1.,1.5,2.95]
    assert refined[1]['units'][0]['start'] == .98
    prior[1]['units'][0]['start'] = .98
    with pytest.raises(ValueError, match='Coarse lyric onsets'):
        protect_line_order(refined, prior)


def test_exact_fifteen_milliseconds_is_not_a_floating_point_conflict():
    cues = [dict(text=str(t), units=[dict(start=t,end=t+.01)]) for t in [1.,1.015]]
    assert protect_line_order(cues,cues)[1] == []


def cache_fixture(store, suffix, *, restorable=False):
    job, _ = store.create(JobRequest(url='https://youtu.be/0000000000'+suffix, title=suffix, force=True), 'test')
    folder = store.root/job['id']
    (folder/'mv.mp4').write_bytes(b'x'*2048)
    result = dict(title=suffix, artist='Singer', language='English', duration_seconds=10,
                  shared_preview_gain=.8, mv_audio='instrumental_backing',
                  reconstruction={'source_pcm_sha256':'fixture'},
                  artifacts={'mv':dict(path='mv.mp4',bytes=2048,sha256='fixture')})
    if restorable:
        for name in ('instrumental', 'backing_vocals', 'lead_vocals', 'instrumental_preview',
                     'backing_vocals_preview', 'lead_vocals_preview', 'subtitles', 'poster'):
            (folder/name).write_bytes(b'keep me')
            result['artifacts'][name] = dict(path=name, bytes=7, sha256=file_hash(folder/name))
        (folder/'result.json').write_text(json.dumps(result))
    return store.update(job['id'], status='completed', last_accessed_at='2020-01-0'+suffix,
                        result=result)


def test_cache_evicts_all_video_lru_before_any_computed_audio(tmp_path):
    store = Store(tmp_path)
    first, second = [cache_fixture(store, str(i), restorable=True) for i in (1, 2)]
    store.cache.enforce()
    used = store.cache.stats['used_bytes']
    store.cache.set_limit(used-1)
    assert store.read(first['id'])['status'] == 'video_evicted'
    assert store.read(second['id'])['status'] == 'completed'
    store.cache.set_limit(used-2049)
    assert store.read(second['id'])['status'] == 'video_evicted'
    for job in (first, second):
        assert not (tmp_path/job['id']/'mv.mp4').exists()
        assert can_restore_video(tmp_path/job['id'], job['result'])
    # Only when the entire video tier is exhausted may the older audio disappear.
    store.cache.set_limit(used-4097)
    assert store.read(first['id'])['status'] == 'evicted'
    assert not store.read(first['id'])['restore_video']
    assert not (tmp_path/first['id']).exists()
    assert can_restore_video(tmp_path/second['id'], second['result'])


def test_video_eviction_still_protects_queue_and_transfers(tmp_path):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    first, second = [cache_fixture(store, str(i), restorable=True) for i in (1, 2)]
    room = app.state.room.read()
    room['entries'] = [dict(id='waiting', job_id=first['id'])]
    app.state.room.save(room)
    store.cache.acquire(second['id'])
    store.cache.enforce()
    store.cache.limit = store.cache.stats['used_bytes']-1
    assert not store.cache.enforce()
    assert all((tmp_path/job['id']/'mv.mp4').exists() for job in (first, second))
    store.cache.release(second['id'])
    assert store.cache.enforce()
    assert (tmp_path/first['id']/'mv.mp4').exists()
    assert store.read(second['id'])['status'] == 'video_evicted'


def test_audio_only_failure_is_not_evicted_as_disposable_video(tmp_path):
    store = Store(tmp_path)
    audio, _ = store.create(JobRequest(url=URL), 'test')
    folder = tmp_path/audio['id']
    (folder/'download').mkdir()
    (folder/'download/source.mkv').write_bytes(b'audio'*200)
    (folder/'source.json').write_text(json.dumps(dict(audio_only=True)))
    store.update(audio['id'], status='failed', last_accessed_at='1')
    video = cache_fixture(store, '2', restorable=True)
    store.cache.enforce()
    store.cache.set_limit(store.cache.stats['used_bytes']-1)
    assert (folder/'download/source.mkv').exists()
    assert store.read(video['id'])['status'] == 'video_evicted'


@pytest.mark.parametrize('audio_only', [True, False])
@pytest.mark.parametrize('status', ['failed', 'audio_ready'])
def test_offline_cache_keeps_sources_when_reclaiming_computed_audio(tmp_path, monkeypatch, audio_only, status):
    monkeypatch.setenv('KARAOKE_REQUIRE_CACHED_SOURCE', '1')
    store = Store(tmp_path)
    job, _ = store.create(JobRequest(url=URL), 'test')
    folder = tmp_path/job['id']
    (folder/'download').mkdir()
    source = folder/'download/source.mkv'
    source.write_bytes(b'cached source'*100)
    source_sha = file_hash(source)
    (folder/'download/source.info.json').write_text('{}')
    (folder/'source.json').write_text(json.dumps(dict(
        url=URL, sha256=source_sha, audio_only=audio_only, duration=10)))
    (folder/'alignment.json').write_text('{"reason":"unresolved"}')
    (folder/'worker.log').write_bytes(b'x'*100000)
    (folder/'working.wav').write_bytes(b'x'*(2*1024**2))
    error = dict(code='align_words_failed') if status == 'failed' else None
    store.update(job['id'], status=status, error=error)
    store.cache.limit = 128*1024
    store.cache.acquire(job['id'])
    assert not store.cache.enforce()
    assert (folder/'working.wav').exists()  # Transfers keep the whole task protected.
    store.cache.release(job['id'])
    assert store.cache.enforce()
    stats = store.cache.snapshot()
    assert 0 < stats['used_bytes'] == stats['protected_bytes'] < store.cache.limit
    assert stats['evictable_bytes'] == 0 and stats['can_prepare']
    assert file_hash(source) == source_sha
    assert not (folder/'working.wav').exists()
    assert (folder/'alignment.json').read_text() == '{"reason":"unresolved"}'
    assert (folder/'worker.log').stat().st_size == 65536
    saved = store.read(job['id'])
    assert saved['status'] == 'source_ready' and saved['error'] == error
    with pytest.raises(CacheFull):
        store.cache.set_limit(stats['used_bytes']-1)
    store.cache.limit = stats['used_bytes']-1
    assert not store.cache.enforce() and not store.cache.snapshot()['can_prepare']
    assert file_hash(source) == source_sha  # Lack of space never deletes an offline input.
    store.cache.limit = 128*1024
    retried, _ = store.create(JobRequest(url=URL), 'new-version')
    assert retried['id'] == job['id'] and retried['status'] == 'queued'
    pipeline = Pipeline(dict(url=URL), folder, audio_only=True)
    monkeypatch.setattr(pipeline, 'run', lambda *a, **k: pytest.fail('Offline retry contacted provider'))
    assert pipeline.download() == ({}, 10)


def test_video_restore_reuses_job_and_recovers_from_failure(tmp_path):
    def failed_download(folder):
        assert app.state.store.read(folder.name)['restore_video']
        return [sys.executable, '-c', "import pathlib,sys; p=pathlib.Path(sys.argv[1]);"
                "(p/'mv.mp4').write_bytes(b'partial');sys.exit(1)", str(folder)]
    app = create_app(tmp_path, worker_command=failed_download, version='new-renderer')
    store = app.state.store
    job = cache_fixture(store, '1', restorable=True)
    folder = tmp_path/job['id']
    store.cache.enforce()
    store.cache.set_limit(store.cache.stats['used_bytes']-1)
    store.cache.set_limit(GIB)
    with TestClient(app) as client:
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['id'] == job['id'])
        assert song['job_id'] == job['id'] and song['status'] == 'video_evicted'
        assert client.get(song['poster']).content == b'keep me'
        assert client.get(f"/v1/jobs/{job['id']}/artifacts/mv").status_code == 410
        assert client.get(f"/v1/jobs/{job['id']}/artifacts/lead_vocals").content == b'keep me'
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert entry['job_id'] == job['id']
        failed = finished(client, job['id'])
        assert failed['status'] == 'failed' and failed['restore_video']
        assert not (folder/'mv.mp4').exists()
        assert can_restore_video(folder, store.read(job['id'])['result'])
        def restored_download(folder):
            return [sys.executable, '-c', '''import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);result=json.loads((p/'result.json').read_text())
(p/'mv.mp4').write_bytes(b'restored')
result['artifacts']['mv'].update(bytes=8,sha256=hashlib.sha256(b'restored').hexdigest())
(p/'result.json').write_text(json.dumps(result))
''', str(folder)]
        app.state.worker.command = restored_download
        retried = client.post('/v1/room/commands', json={'action':'retry','entry_id':entry['id']}).json()
        assert retried['entries'][0]['job_id'] == job['id']
        assert finished(client, job['id'])['status'] == 'completed'
        assert not store.read(job['id'])['restore_video']
        assert client.get(f"/v1/jobs/{job['id']}/artifacts/mv").content == b'restored'
        cached = client.post('/v1/jobs', json={**job['input'], 'force':False})
        assert cached.status_code == 200 and cached.json()['id'] == job['id']
        assert len(store.list()) == 1
        assert all((folder/a['path']).read_bytes() == b'keep me'
                   for name, a in job['result']['artifacts'].items() if name != 'mv')


def test_video_restore_url_deduplicates_across_renderer_versions(tmp_path):
    store = Store(tmp_path)
    original = cache_fixture(store, '1', restorable=True)
    store.cache.enforce()
    store.cache.set_limit(store.cache.stats['used_bytes']-1)
    request = JobRequest(**{**original['input'], 'force':False})
    restored, cached = store.create(request, 'new-renderer')
    duplicate, duplicate_cached = store.create(request, 'new-renderer')
    assert restored['id'] == duplicate['id'] == original['id']
    assert not cached and duplicate_cached and restored['restore_video']
    assert restored['result'] == original['result']
    changed, _ = store.create(request.model_copy(update={'mv_audio':'instrumental'}), 'new-renderer')
    assert changed['id'] != original['id']
    forced, _ = store.create(request.model_copy(update={'force':True}), 'new-renderer')
    assert forced['id'] != original['id']


@pytest.mark.parametrize('source_matches', [True, False])
def test_video_only_pipeline_uses_retained_audio_and_cues(tmp_path, monkeypatch, source_matches):
    import shutil
    import soundfile as sf
    from karaoke.exact_stems import decode_source, pcm_sha
    source = tmp_path/'original.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
        'color=c=black:s=320x180:r=30:d=2', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
        '-c:v', 'libx264', '-c:a', 'pcm_s16le', str(source)], check=True)
    audio, rate, _ = decode_source(source)
    store = Store(tmp_path/'cache')
    job = cache_fixture(store, '1', restorable=True)
    folder, result = store.root/job['id'], job['result']
    result.update(duration_seconds=2)
    result['reconstruction']['source_pcm_sha256'] = pcm_sha(audio) if source_matches else 'different-recording'
    for name, level in [('instrumental', .5), ('backing_vocals', .25), ('lead_vocals', .25)]:
        sf.write(folder/name, audio*level, rate, format='WAV', subtype='DOUBLE')
    (folder/'subtitles').write_text(json.dumps([dict(text='Hello',start=0,end=2,
        units=[dict(text='Hello',start=.5,end=1.5,first=0,stop=5)])]))
    before = {}
    for name, artifact in result['artifacts'].items():
        if name != 'mv':
            path = folder/artifact['path']
            artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))
            before[name] = path.read_bytes()
    (folder/'result.json').write_text(json.dumps(result))
    (folder/'mv.mp4').unlink()
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    pipeline = Pipeline(job['input'], folder, separator_python='/missing/python')
    def download(**kwargs):
        pipeline.source.parent.mkdir()
        shutil.copyfile(source, pipeline.source)
    monkeypatch.setattr(pipeline, 'download', download)
    if source_matches:
        restored = pipeline.restore_video()
        assert set(restored['video_restore']['timings']) == {'verify_video', 'render'}
        assert (folder/'mv.mp4').stat().st_size > 2048
        assert restored['artifacts']['mv']['sha256'] == file_hash(folder/'mv.mp4')
    else:
        with pytest.raises(StageFailure, match='音频与缓存版本不同'):
            pipeline.restore_video()
        assert not (folder/'mv.mp4').exists()
    assert all((folder/name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize('fallback_text', ['恢复的歌词', '哒'*100])
@pytest.mark.parametrize('initial_kind', ['loop', 'punctuation', 'fragmented_loop',
                                        'backward_segments', 'backward_words', 'invalid_range',
                                        'collapsed_words', 'unsupported_language',
                                        'short_window_recovery', 'short_tail_recovery', 'local_loop',
                                        'suspected_silence', 'compressed_silence'])
def test_invalid_asr_recovers_in_chunks_without_inventing_word_times(tmp_path, fallback_text, initial_kind):
    import soundfile as sf
    audio = tmp_path/'audio.wav'
    short_tail = initial_kind == 'short_tail_recovery'
    duration = 45.05 if short_tail else 65.
    sf.write(audio, np.zeros(round(duration*1000)), 1000)
    output = tmp_path/'asr'
    output.mkdir()
    calls = []
    def create(**kwargs):
        name = kwargs['file'].name
        calls.append((name, kwargs['model']))
        bad_chunk = name.endswith('chunk-00030.mp3') or (
            initial_kind not in ('short_window_recovery', 'short_tail_recovery') and name.endswith('-15s.mp3'))
        if kwargs['model'] != 'whisper-1':
            assert 'timestamp_granularities' not in kwargs
            data = dict(text=fallback_text)
        elif name == str(audio) and initial_kind == 'punctuation':
            data = dict(language='nynorsk', segments=[dict(start=1,end=2,text='...')], words=[])
        elif name == str(audio) and initial_kind == 'fragmented_loop':
            data = dict(language='english', segments=[dict(start=i,end=i,text='same empty phrase')
                                                     for i in range(24)], words=[])
        elif name == str(audio) and initial_kind == 'local_loop':
            # A short loop inside an otherwise normal transcript must not be
            # hidden by the proportion of correctly timed verses.
            data = dict(language='chinese', segments=[
                *[dict(start=i, end=i+.8, text='正常的一句') for i in range(30)],
                *[dict(start=30+i*.02, end=30+(i+1)*.02, text='耶') for i in range(21)],
                *[dict(start=31+i, end=31+i+.8, text='后面的歌词') for i in range(30)]], words=[])
        elif initial_kind in ('suspected_silence', 'compressed_silence') and (name == str(audio) or bad_chunk):
            end = 2 if initial_kind == 'suspected_silence' else 1.1
            times = [(1,1.8),(1.8,1.8),(1.8,1.8),(1.8,2)] if end == 2 else [
                (1,1.02),(1.02,1.04),(1.04,1.06),(1.06,1.1)]
            data = dict(language='chinese', segments=[dict(start=1, end=end,
                        text='A fluent invented phrase', no_speech_prob=.606)],
                        words=[dict(start=a, end=b, word=w) for (a,b),w in zip(times,
                               ['A','fluent','invented','phrase'])])
        elif initial_kind == 'collapsed_words' and (name == str(audio) or bad_chunk):
            data = dict(language='japanese', segments=[dict(start=1,end=5,text='一緒に歌うこの歌の続きを')],
                        words=[dict(start=1,end=1,word=c) for c in '一緒に歌うこの歌の続きを'])
        elif initial_kind == 'unsupported_language' and (name == str(audio) or bad_chunk):
            data = dict(language='russian', segments=[dict(start=1,end=2,text='пример слов')],
                        words=[dict(start=1,end=2,word='пример слов')])
        elif initial_kind in ('backward_segments', 'backward_words', 'invalid_range') and (
                name == str(audio) or bad_chunk):
            data = dict(language='english', segments=[dict(start=1,end=2,text='第一句'),
                                                      dict(start=3,end=4,text='第二句')],
                        words=[dict(start=1,end=2,word='第一句'), dict(start=3,end=4,word='第二句')])
            if initial_kind == 'invalid_range':
                data['words'][0]['end'] = .5
            else:
                data[initial_kind.removeprefix('backward_')][0].update(start=10, end=11)
        elif name == str(audio) or bad_chunk:
            data = dict(language='chinese', segments=[dict(start=0,end=1,text='呜'*100)], words=[])
        else:
            assert 'language' not in kwargs
            data = dict(language='chinese', segments=[dict(start=1,end=2,text='歌词')],
                        words=[dict(start=1,end=2,word='歌词')])
        return SimpleNamespace(model_dump=lambda: data)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    if repetition_loop(dict(segments=[dict(text=fallback_text)])):
        with pytest.raises(ValueError, match='文本重试仍有重复异常'):
            transcribe(client, audio, 'auto', '', output)
        # Retrying the failed stage retains both rejected fallback responses.
        from karaoke.stage_cache import StageCache
        previous_calls = list(calls)
        with pytest.raises(ValueError, match='文本重试仍有重复异常'):
            with StageCache(tmp_path).stage('transcribe', inputs=[audio], outputs=[output]):
                output.mkdir()
                transcribe(client, audio, 'auto', '', output)
        assert calls == previous_calls
        assert len(list((tmp_path/'asr-requests').glob('*.json'))) == 3
        assert not (tmp_path/'checkpoints/transcribe.json').exists()
        return
    data, recovery = transcribe(client, audio, 'auto', '', output)
    assert recovery['triggered'] and len(calls) == (6 if short_tail else 7)
    if short_tail:
        expected = [(1,2),(31,32),(39,40)]
    elif initial_kind == 'short_window_recovery':
        expected = [(1,2),(31,32),(46,47),(61,62)]
    else:
        expected = [(1,2),(61,62)]
    assert [(w['start'],w['end']) for w in data['words']] == expected
    assert data['segments'] == [dict(text='恢复的歌词', start=0., end=duration, timing_source='recording_bounds')]
    assert len(recovery['chunks']) == (4 if short_tail else 5) and recovery['chunks'][1]['word_times_discarded']
    assert recovery['text_model'] == 'gpt-4o-transcribe'
    assert data['language'] == 'Chinese'
    assert {key: count for key, count in recovery['language_evidence'].items() if count} == {'Chinese': 2*len(expected)}
    assert not list(output.glob('chunk-*.mp3'))
    assert (output/'initial-transcription.json').exists()


@pytest.mark.parametrize('language', ['auto', 'zh'])
def test_normal_choruses_do_not_trigger_asr_retry(tmp_path, language):
    data = dict(language='chinese', segments=[dict(start=i*4, end=i*4+3, text='唱到这一句，接着唱下一句', no_speech_prob=.8) for i in range(8)],
                words=[dict(start=1,end=1,word='一'), dict(start=1,end=1,word='起'),
                       dict(start=1,end=2,word='唱'), dict(start=1.9,end=2.5,word='歌'),
                       dict(start=2.5,end=2.8,word='吧')])
    assert not repetition_loop(data)
    assert not repetition_loop(dict(segments=[dict(start=i, end=i+.02, text='耶') for i in range(21)]))
    audio = tmp_path/'audio.mp3'
    audio.write_bytes(b'input')
    output = tmp_path/'asr'
    output.mkdir()
    with patch('karaoke.transcribe_lyrics.OpenAI') as client:
        client.audio.transcriptions.create.return_value.model_dump.return_value = data
        actual, recovery = transcribe(client, audio, language, '', output,
                                      original_audio=tmp_path/'unused-original.wav')
        assert actual == data and not recovery['triggered']
        client.audio.transcriptions.create.assert_called_once()


@pytest.mark.parametrize('collapsed', [False, True])
@pytest.mark.parametrize('nested_words', [False, True])
def test_repeated_positive_length_phrases_still_reject_collapsed_word_loops(collapsed, nested_words):
    from karaoke.transcribe_lyrics import needs_recovery
    data = dict(segments=[], words=[])
    for i in range(10):
        words = [dict(word=char, start=i+j*.2, end=i+j*.2 if collapsed and j<3 else i+(j+1)*.2)
                 for j, char in enumerate('甲乙丙丁戊')]
        segment = dict(start=i, end=i+1., text='甲乙丙丁戊', no_speech_prob=.1)
        if nested_words:
            segment['words'] = words
        data['segments'].append(segment)
        data['words'].extend(words)
    assert repetition_loop(data) == collapsed
    assert needs_recovery(data) == collapsed


@pytest.mark.parametrize('tail', ['silence', 'quiet', 'quiet_stereo', 'negative_time', 'outside_audio',
                                  'silent_parent', 'quiet_parent', 'mismatched_parent', 'shared_parent'])
def test_asr_checks_exact_silence_before_repeating_full_recording(tmp_path, tail):
    from copy import deepcopy
    import soundfile as sf
    from karaoke.transcribe_lyrics import needs_recovery

    audio, output = tmp_path/'audio.wav', tmp_path/'asr'
    output.mkdir()
    pcm = np.zeros(3*16000, dtype='float32')
    pcm[8000:24000] = 1e-7
    if tail.startswith('quiet'):
        pcm[40000:] = 1e-7
    if tail == 'quiet_stereo':
        pcm = np.column_stack((pcm, -pcm))  # Downmix cancellation is not silence.
    sf.write(audio, pcm, 16000, subtype='FLOAT')
    start = {'negative_time': -.1, 'outside_audio': 9.}.get(tail, 2.8)
    initial = dict(language='english', text='Soft opening invented ending',
        segments=[dict(start=.5, end=1.5, text='Soft opening'),
                  dict(start=start, end=start, text=' invented ending')],
        words=[dict(start=.5, end=1.5, word='opening'),
               *[dict(start=start, end=start, word='invented') for _ in range(8)]])
    if tail.endswith('_parent'):
        for word in initial['words'][1:]:
            word.update(start=3.5, end=3.5)
        initial['segments'][-1].update(start=2.8, end=3.5, words=deepcopy(initial['words'][1:]))
        if tail == 'mismatched_parent':
            for word in initial['segments'][-1]['words']:
                word['word'] = 'different'
        if tail == 'shared_parent':
            initial['segments'][0]['words'] = deepcopy(initial['words'][1:])
    original, calls = deepcopy(initial), []
    assert needs_recovery(initial)

    def create(**request):
        calls.append(request['model'])
        if request['model'] != 'whisper-1':
            raise RuntimeError('still requires recovery')
        return SimpleNamespace(model_dump=lambda: initial)

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    silent = tail in ('silence', 'silent_parent')
    if silent:
        result, recovery = transcribe(client, audio, 'auto', '', output)
        assert calls == ['whisper-1'] and not recovery['triggered']
        assert result['words'] == original['words'][:1]
        assert result['segments'] == original['segments'][:1]
    else:
        with pytest.raises(RuntimeError, match='still requires recovery'):
            transcribe(client, audio, 'auto', '', output)
        assert calls == ['whisper-1', 'gpt-4o-transcribe']
    assert initial == original
    assert json.loads((output/'initial-transcription.json').read_text()) == original
    report = json.loads((output/'silence-filter.json').read_text())
    assert report['applied'] == silent
    if silent:
        assert len(report['rejected']) == 9
        assert {r['reason'] for r in report['rejected']} == {'digital_silence'}
    if tail == 'silent_parent':
        assert all(r['silent_parent'] == dict(start=2.8, end=3.5)
                   for r in report['rejected'] if r['kind'] == 'words')
    if tail in ('quiet', 'quiet_stereo'):
        assert not report['rejected']


def test_asr_request_cache_checks_input_parameters_and_response(tmp_path):
    from karaoke.transcribe_lyrics import cached_recognition
    from karaoke.stage_cache import retained_files
    audio = tmp_path/'audio.mp3'
    audio.write_bytes(b'first audio')
    output = tmp_path/'asr'
    output.mkdir()
    calls = []
    def create(**kwargs):
        calls.append(kwargs['prompt'])
        data = dict(text='known response', call=len(calls))
        return SimpleNamespace(model_dump=lambda: data)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    request = dict(model='whisper-1', prompt='')
    first = cached_recognition(client, audio, request, output)
    assert cached_recognition(client, audio, request, output) == first
    path = next((tmp_path/'asr-requests').glob('*.json'))
    assert path.relative_to(tmp_path).as_posix() in retained_files(tmp_path)
    path.write_text('{"text":"corrupted response"}')
    assert cached_recognition(client, audio, request, output)['call'] == 2
    audio.write_bytes(b'different audio')
    assert cached_recognition(client, audio, request, output)['call'] == 3
    assert cached_recognition(client, audio, dict(request, prompt='new context'), output)['call'] == 4


@pytest.mark.parametrize('scenario', ['intro', 'disagreement', 'short', 'explicit'])
def test_local_asr_language_uses_the_recording_instead_of_only_the_intro(tmp_path, monkeypatch, scenario):
    import torch
    import whisper
    from karaoke.transcribe_lyrics import cached_local_recognition

    audio, weights, output = tmp_path/'audio.wav', tmp_path/'model.pt', tmp_path/'asr'
    audio.write_bytes(b'recording')
    weights.write_bytes(b'model')
    output.mkdir()
    duration = 10 if scenario == 'short' else 180
    samples = np.arange(duration*16000, dtype=np.float32)/16000
    windows, decodes = [], []
    def detect(mel):
        start = float(mel[0])
        windows.append(start)
        code = 'en' if start < 30 else ('zh' if scenario != 'disagreement' or start < 60
                                       else 'ja' if start < 100 else 'en')
        return None, {code: .99, 'fr': .01}
    def load(*args, **kwargs):
        return SimpleNamespace(_buffers={'alignment_heads': None}, device='cpu',
            dims=SimpleNamespace(n_mels=128), to=lambda *a: None,
            register_buffer=lambda *a, **k: None, detect_language=detect)
    def transcribe(model, path, *, language, **kwargs):
        if language is None:
            _, probabilities = model.detect_language(torch.tensor([0.]))
            language = max(probabilities, key=probabilities.get)
        decodes.append(language)
        return dict(language=language, segments=[dict(words=[dict(word='song', start=1., end=2.)])])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(whisper, 'load_model', load)
    monkeypatch.setattr(whisper, 'load_audio', lambda path: samples)
    monkeypatch.setattr(whisper, 'log_mel_spectrogram', lambda pcm, *a, **k: torch.tensor([float(pcm[0])]))
    monkeypatch.setattr(whisper, 'transcribe', transcribe)
    language = 'en' if scenario == 'explicit' else 'auto'
    if scenario == 'disagreement':
        result = cached_local_recognition(audio, language, output, weights=weights)
        assert result['language'] is None and not result['words'] and not decodes
        before = list(windows)
        assert cached_local_recognition(audio, language, output, weights=weights) == result
        assert windows == before
    else:
        result = cached_local_recognition(audio, language, output, weights=weights)
        assert result['language'] == ('zh' if scenario == 'intro' else 'en')
        if scenario == 'intro':
            assert len(windows) == 3 and min(windows) >= 30
        elif scenario == 'explicit':
            assert not windows
        else:
            assert windows == [0.]
        before = list(windows), list(decodes)
        assert cached_local_recognition(audio, language, output, weights=weights) == result
        assert (windows, decodes) == before


@pytest.mark.parametrize('batch_size', [1, 8])
@pytest.mark.parametrize('language', ['en', 'auto', 'disagreement'])
def test_local_whisper_batches_cover_the_recording_and_keep_word_times(monkeypatch, batch_size, language):
    from dataclasses import dataclass
    from karaoke import local_asr
    @dataclass
    class Segment:
        text: str
        start: float
        end: float
        words: list
    calls, detected = [], []
    samples = np.arange(61*16000+37, dtype=np.float32)/16000
    def detect(*, audio, vad_filter):
        detected.append(float(audio[0]))
        code = ('en', 'zh', 'ja')[len(detected)-1] if language == 'disagreement' else 'ja'
        return code, .99, []
    def run(audio, **options):
        assert audio is samples
        calls.append(options)
        return iter([Segment(' quiet song', 30.125, 32.75,
                             [dict(word='quiet', start=30.125, end=31.25),
                              dict(word='song', start=31.375, end=32.75)])]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(
        BatchedInferencePipeline=lambda model: SimpleNamespace(transcribe=run)))
    result = local_asr.transcribe(SimpleNamespace(detect_language=detect), samples,
                                 'auto' if language == 'disagreement' else language, batch_size)
    if language == 'disagreement':
        assert result['language'] is None and not calls and not result['words']
    else:
        assert result['words'][0]['start'] == 30.125 and result['words'][-1]['end'] == 32.75
        assert result['text'] == ' quiet song'
        assert calls[0]['language'] == ('en' if language == 'en' else 'ja')
        assert calls[0]['batch_size'] == batch_size and not calls[0]['vad_filter']
        assert calls[0]['clip_timestamps'] == [dict(start=0., end=30.), dict(start=30., end=60.),
                                               dict(start=60., end=len(samples)/16000)]
        assert bool(detected) == (language == 'auto')


@pytest.mark.parametrize('valid', [True, False])
def test_offline_transcription_never_falls_back_to_an_api(tmp_path, monkeypatch, valid):
    import soundfile as sf
    from karaoke import transcribe_lyrics
    audio, output = tmp_path/'audio.wav', tmp_path/'asr'
    output.mkdir()
    sf.write(audio, np.full(48000, .01), 16000)
    text = 'A quiet song' if valid else 'same '*100
    data = dict(language='en', text=text, segments=[dict(text=text, start=.5, end=2.5)],
                words=[dict(word=text, start=.5, end=2.5)])
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
    monkeypatch.setattr(transcribe_lyrics.local_asr, 'cached_recognition', lambda *a: data)
    with patch.object(transcribe_lyrics, 'cached_recognition', side_effect=AssertionError('API called')):
        if valid:
            result, recovery = transcribe(None, audio, 'en', '', output)
            assert result == data and not recovery['triggered']
        else:
            with pytest.raises(ValueError, match='本地 Whisper'):
                transcribe(None, audio, 'en', '', output)
            assert json.loads((output/'initial-transcription.json').read_text()) == data
            assert (output/'recovery.json').is_file()


@pytest.mark.parametrize('collapsed,expected', [(1, False), (7, True)])
def test_silence_warning_includes_collapsed_words_at_the_segment_end(collapsed, expected):
    from karaoke.transcribe_lyrics import needs_recovery
    words = [dict(word=f'word{i}', start=i, end=i+1) for i in range(5)]
    words += [dict(word=f'tail{i}', start=5., end=5.) for i in range(collapsed)]
    # The following phrase starts at the same boundary, but has real duration.
    words += [dict(word=f'next{i}', start=5., end=6.+i) for i in range(8)]
    data = dict(segments=[dict(text='A possible phrase', start=0., end=5., no_speech_prob=.7)], words=words)
    assert needs_recovery(data) is expected


def test_offline_metadata_and_vocal_review_keep_unverified_gaps(tmp_path, monkeypatch):
    from karaoke.song_sources import extract_song_metadata
    from karaoke.vocal_review import review_gaps, speech_intervals
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
    with patch('karaoke.song_sources.OpenAI', side_effect=AssertionError('API called')), \
         patch('openai.OpenAI', side_effect=AssertionError('API called')):
        assert extract_song_metadata({}, 'unused')['parsed'] is None
        reviews = review_gaps(tmp_path, [dict(start=0, end=3)], [dict(start=1, end=2, word='hello')])
        assert reviews == [] and speech_intervals(tmp_path, reviews) == []


def test_local_batch_cache_tracks_model_audio_and_batch_size(tmp_path, monkeypatch):
    from karaoke import local_asr
    audio, output, weights = tmp_path/'audio.wav', tmp_path/'asr', tmp_path/'weights'
    output.mkdir(); weights.mkdir(); audio.write_bytes(b'first source')
    for name in ('config.json', 'model.bin', 'tokenizer.json', 'preprocessor_config.json'):
        (weights/name).write_bytes(b'model')
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
    monkeypatch.setenv('KARAOKE_WHISPER_MODEL', str(weights))
    monkeypatch.setattr(local_asr, 'version', lambda name: 'test')
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, 'faster_whisper.audio', SimpleNamespace(decode_audio=lambda *a, **kw: [0.]))
    calls = []
    def recognize(*args):
        calls.append(args)
        return dict(text=f'transcript {len(calls)}', local_inference={})
    monkeypatch.setattr(local_asr, 'transcribe', recognize)
    first = local_asr.cached_recognition(audio, 'en', output)
    assert local_asr.cached_recognition(audio, 'en', output) == first and len(calls) == 1
    (weights/'model.bin').write_bytes(b'new model')
    local_asr.cached_recognition(audio, 'en', output)
    audio.write_bytes(b'new source')
    local_asr.cached_recognition(audio, 'en', output)
    monkeypatch.setenv('KARAOKE_WHISPER_BATCH_SIZE', '1')
    local_asr.cached_recognition(audio, 'en', output)
    assert len(calls) == 4


@pytest.mark.parametrize('problem', [None, 'loop', 'collapsed', 'outside_audio',
                                     'wrong_language', 'unsupported_language', 'duration', 'missing_model'])
@pytest.mark.parametrize('track', ['original', 'lead'])
def test_unusable_vocal_asr_rechecks_recording_tracks_before_whole_song_retry(tmp_path, monkeypatch, problem, track):
    from copy import deepcopy
    import soundfile as sf

    audio, original, output = tmp_path/'vocals.wav', tmp_path/'source.wav', tmp_path/'asr'
    output.mkdir()
    sf.write(audio, np.full(3000, .01), 1000)
    sf.write(original, np.full(2000 if problem == 'duration' else 3000, .02), 1000)
    initial = dict(language='nynorsk', text='1, '*100,
        segments=[dict(start=.5, end=2.5, text='1, '*100)],
        words=[dict(start=.5, end=2.5, word='1')])
    candidate = dict(language='en', text='A quiet opening',
        segments=[dict(start=.5, end=2.5, text='A quiet opening')],
        words=[dict(start=.5, end=1., word='A'), dict(start=1., end=2., word='quiet'),
               dict(start=2., end=2.5, word='opening')])
    if problem == 'loop':
        candidate['segments'][0]['text'] = 'repeated phrase '*10
    elif problem == 'collapsed':
        candidate['words'] = [dict(start=.5, end=.5, word='word') for _ in range(8)]
    elif problem == 'outside_audio':
        candidate['words'][-1]['end'] = 4.
    elif problem == 'unsupported_language':
        candidate['language'] = 'ru'
    preserved, calls = deepcopy(candidate), []
    language = 'zh' if problem == 'wrong_language' else 'auto'

    def local(path, detected, target):
        calls.append('local')
        # A broken transcript's detected language must not constrain this decode.
        assert (path, detected, target) == (original, language, output)
        if problem == 'missing_model':
            raise ValueError('model unavailable')
        return candidate

    def create(**request):
        calls.append(request['model'])
        if request['model'] != 'whisper-1':
            raise RuntimeError('provider recovery still needed')
        return SimpleNamespace(model_dump=lambda: initial)

    monkeypatch.setattr('karaoke.transcribe_lyrics.cached_local_recognition', local)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    options = {track+'_audio': original}
    if problem is None:
        data, recovery = transcribe(client, audio, language, '', output, **options)
        assert data == candidate and recovery[track+'_audio']['accepted']
        assert recovery['text_model'] == 'local-large-v3-turbo'
        assert calls == ['whisper-1', 'local']
    else:
        with pytest.raises(RuntimeError, match='provider recovery still needed'):
            transcribe(client, audio, language, '', output, **options)
        report = json.loads((output/f'{track}-recovery.json').read_text())
        assert not report['accepted'] and report['reason']
        assert calls == ['whisper-1'] + ([] if problem == 'duration' else ['local']) + ['gpt-4o-transcribe']
    assert candidate == preserved
    assert json.loads((output/'initial-transcription.json').read_text()) == initial


def test_failed_original_asr_uses_lead_before_another_remote_request(tmp_path, monkeypatch):
    import soundfile as sf

    audio, original, lead = [tmp_path/(name+'.wav') for name in ('vocals', 'original', 'lead')]
    for path in (audio, original, lead):
        sf.write(path, np.full(3000, .01), 1000)
    output = tmp_path/'asr'
    output.mkdir()
    broken = dict(language='en', text='word '*100,
        segments=[dict(start=.5, end=2.5, text='word '*100)], words=[])
    valid = dict(language='en', text='A quiet ending',
        segments=[dict(start=.5, end=2.5, text='A quiet ending')],
        words=[dict(word='A', start=.5, end=1), dict(word='quiet', start=1, end=2),
               dict(word='ending', start=2, end=2.5)])
    calls = []
    def recognize(path, language, target):
        calls.append(path)
        return broken if path == original else valid
    def remote(**request):
        assert request['model'] == 'whisper-1'
        return SimpleNamespace(model_dump=lambda: broken)
    monkeypatch.setattr('karaoke.transcribe_lyrics.cached_local_recognition', recognize)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=remote)))
    result, recovery = transcribe(client, audio, 'en', '', output, original_audio=original, lead_audio=lead)
    assert result == valid and calls == [original, lead]
    assert not recovery['original_audio']['accepted'] and recovery['lead_audio']['accepted']
    assert json.loads((output/'initial-transcription.json').read_text()) == broken


@pytest.mark.parametrize('problem', [None, 'padding', 'missing_boundary', 'wrong_language', 'overlap', 'loop'])
@pytest.mark.parametrize('initial_error', ['loop', 'silence_warning', 'collapsed_phrase'])
def test_local_asr_repair_keeps_surrounding_words_and_requires_matching_boundaries(tmp_path, problem, initial_error):
    from copy import deepcopy
    from karaoke.transcribe_lyrics import repair_loop_window, splice_loop_window
    import soundfile as sf
    original = dict(language='chinese', duration=60, segments=[
        dict(start=3., end=5., text='此刻一起唱着歌'),
        # The decoder starts at 6s but only emits loop words much later. A
        # repair must include the lost audio before those words, not just 30s+.
        dict(start=30., end=32., seek=600, text='哒'*100),
        dict(start=45., end=48., text='走向明天的远方')],
        words=[dict(start=3.,end=5.,word='此刻一起唱着歌'),
               dict(start=30.,end=32.,word='哒'*100),
               dict(start=45.,end=48.,word='走向明天的远方')])
    if initial_error != 'loop':
        text = '不应出现的虚构歌词内容'
        original['segments'][1].update(text=text,
            no_speech_prob=.9 if initial_error == 'silence_warning' else .1)
        words = [dict(start=32., end=32., word=c) for c in text]
        if initial_error == 'silence_warning':
            words[0]['start'] = 30.
        original['words'][1:2] = words
    original['segments'].append(dict(start=55., end=60., text=''))
    replacement = dict(language='chinese', segments=[
        dict(start=.5,end=2.5,text='此刻一起唱着歌'),
        dict(start=8.,end=10.,text='轻轻唱出这一句'),
        dict(start=20.,end=22.,text='然后听见你的回答'),
        dict(start=42.5,end=45.5,text='走向明天的远方')],
        words=[dict(start=.5,end=2.5,word='此刻一起唱着歌'),
               dict(start=8.,end=10.,word='轻轻唱出这一句'),
               dict(start=20.,end=22.,word='然后听见你的回答'),
               dict(start=42.5,end=45.5,word='走向明天的远方')])
    damaged = deepcopy(replacement)
    if problem == 'missing_boundary':
        damaged['segments'][-1]['text'] = '完全无关的一句话'
    elif problem == 'wrong_language':
        damaged['language'] = 'english'
    elif problem == 'overlap':
        damaged['words'][1]['end'] = 46.
    elif problem == 'loop':
        damaged['segments'][1]['text'] = '啦'*100
    elif problem == 'padding':
        damaged['segments'][0]['text'] = '我说' + damaged['segments'][0]['text']
        damaged['segments'].append(dict(start=45.5,end=45.5,text='啦'*100))
        damaged['words'].append(dict(start=45.5,end=45.5,word='啦'*100))
    result = splice_loop_window(original, damaged, 0, 2, 2.5)
    if problem not in (None, 'padding'):
        assert result is None
    else:
        assert result['segments'][0] == original['segments'][0]
        assert result['segments'][-1] == original['segments'][-1]
        assert result['words'] == [original['words'][0],
            dict(start=10.5,end=12.5,word='轻轻唱出这一句'),
            dict(start=22.5,end=24.5,word='然后听见你的回答'), original['words'][-1]]
    audio, output = tmp_path/'audio.wav', tmp_path/'asr'
    sf.write(audio, np.zeros(60000), 1000)
    output.mkdir()
    def recognize(path):
        assert path.is_file()
        return damaged
    preserved = deepcopy(original)
    actual = repair_loop_window(original, audio, 60, recognize, output)
    report = json.loads((output/'loop-repair.json').read_text())
    assert original == preserved
    if result is None:
        assert actual is None and not report['accepted']
    else:
        assert actual[0] == result and report['accepted']
    assert (report['begin'],report['end']) == (2.5,48.5)
    assert not (output/'loop-repair.mp3').exists()


@pytest.mark.parametrize('decoder_window', [600, None])
@pytest.mark.parametrize('matching_boundary', [True, False])
@pytest.mark.parametrize('distant_boundary', [True, False])
def test_local_asr_repair_anchors_outside_corrupted_decoder_window(tmp_path, decoder_window, matching_boundary, distant_boundary):
    from copy import deepcopy
    from karaoke.transcribe_lyrics import repair_loop_window
    import soundfile as sf

    segments = [dict(start=3., end=5., text='此刻一起唱着歌'),
        dict(start=6., end=8., text='貌似完整的错误句子', seek=decoder_window),
        dict(start=9., end=10., text='哒'*100, seek=decoder_window),
        dict(start=11., end=13., text='另一句错误识别结果', seek=decoder_window),
        dict(start=105. if distant_boundary else 45., end=108. if distant_boundary else 48., text='走向明天的远方')]
    original = dict(language='chinese', segments=segments,
        words=[dict(start=s['start'], end=s['end'], word=s['text']) for s in segments])
    preserved = deepcopy(original)
    parts = [dict(start=.5, end=2.5, text='此刻一起唱着歌'),
        dict(start=8., end=10., text='轻轻唱出这一句'),
        dict(start=20., end=22., text='然后听见你的回答'),
        dict(start=42.5, end=45.5, text='走向明天的远方' if matching_boundary else '完全不同的一句话')]
    replacement = dict(language='chinese', segments=parts,
        words=[dict(start=s['start'], end=s['end'], word=s['text']) for s in parts])
    audio, output = tmp_path/'audio.wav', tmp_path/'asr'
    sf.write(audio, np.zeros(60000), 1000)
    output.mkdir()
    with patch('karaoke.transcribe_lyrics.subprocess.run', wraps=subprocess.run) as extract:
        actual = repair_loop_window(original, audio, 120 if distant_boundary else 60, lambda _: replacement, output)
        if decoder_window is not None and distant_boundary:
            extract.assert_not_called()
    assert original == preserved
    if decoder_window is None or not matching_boundary or distant_boundary:
        assert actual is None
    else:
        result, report = actual
        assert (report['begin'], report['end']) == (2.5, 48.5)
        assert result['segments'] == [segments[0],
            dict(start=10.5, end=12.5, text='轻轻唱出这一句'),
            dict(start=22.5, end=24.5, text='然后听见你的回答'), segments[-1]]
        assert result['words'][0] == original['words'][0]
        assert result['words'][-1] == original['words'][-1]


@pytest.mark.parametrize('recoverable', [True, False])
@pytest.mark.parametrize('independent_source', [None, 'description', 'user', 'unrelated'])
def test_truncated_recovery_text_cannot_drop_later_sung_words(tmp_path, recoverable, independent_source):
    import soundfile as sf
    from karaoke.transcribe_lyrics import select_recovery_text
    greeting = "Hello, I'm here."
    lines = ['沿着山路走到远方', '听见晚风轻轻歌唱', '抬头又见漫天星光']
    complete = greeting + ''.join(lines)
    words = [dict(word=text, start=i*5+1, end=i*5+4)
             for i, text in enumerate([greeting, *lines])]
    initial = complete if recoverable else '呜'*100
    audio = tmp_path/'audio.wav'
    sf.write(audio, np.zeros(30000), 1000)
    output = tmp_path/'asr'
    output.mkdir()
    def create(**kwargs):
        if kwargs['model'] != 'whisper-1':
            data = dict(text=greeting)
        elif kwargs['file'].name == str(audio):
            data = dict(language='chinese', text=initial,
                        segments=[dict(start=2., end=1., text=initial)], words=[])
        else:
            data = dict(language='chinese', text=complete,
                        segments=[dict(start=1., end=19., text=complete)], words=words)
        return SimpleNamespace(model_dump=lambda: data)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    if recoverable:
        result, recovery = transcribe(client, audio, 'auto', '', output)
        assert result['text'] == complete and result['segments'][0]['text'] == complete
        assert recovery['text_selection']['selected'] == 'initial_transcription'
        assert result['words'] == words
    elif independent_source is None:
        with pytest.raises(ValueError, match='遗漏了分段识别'):
            transcribe(client, audio, 'auto', '', output)
        assert json.loads((output/'recovery-words.json').read_text()) == words
        assert json.loads((output/'text-selection.json').read_text())['selected'] is None
    else:
        result, recovery = transcribe(client, audio, 'auto', '', output, allow_unresolved_text=True)
        assert result['text'] == '' and result['segments'] == []
        assert result['words'] == words
        assert recovery['text_selection']['selected'] is None
        # Usable word evidence must survive a rejected text draft. Only an
        # independent source that passes the unchanged audio gate can use it.
        request = dict(url=URL)
        supplied = [greeting, *lines]
        if independent_source == 'user':
            request['lyrics'] = supplied
        if independent_source == 'unrelated':
            supplied = ['另外一个不相关的故事']
        pipeline = Pipeline(request, tmp_path, catalog_root=None)
        sf.write(tmp_path/'vocals-16k.wav', np.ones(30000)*.1, 1000)
        (output/'transcription.json').write_text(json.dumps(result))
        (output/'run.json').write_text(json.dumps(dict(recovery=recovery)))
        parsed = SongText(title='Song', artist='Singer', lyric_line_numbers=list(range(1,len(supplied)+1)))
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test', usage=None)
        with patch('karaoke.song_sources.OpenAI') as metadata_client:
            metadata_client.return_value.beta.chat.completions.parse.return_value = response
            if independent_source == 'unrelated':
                with pytest.raises(StageFailure, match='匹配不足'):
                    pipeline.prepare_lyrics(dict(description='\n'.join(supplied)), 30.)
            else:
                pipeline.prepare_lyrics(dict(description='\n'.join(supplied)), 30.)
                lyrics = json.loads((tmp_path/'lyrics.json').read_text())
                assert lyrics['lines'] == supplied
                assert lyrics['provenance']['kind'] == ('user_supplied' if independent_source == 'user' else 'video_description')
        with patch('karaoke.song_sources.OpenAI') as metadata_client:
            response.choices[0].message.parsed = SongText(title='Song', artist='Singer', lyric_line_numbers=[])
            metadata_client.return_value.beta.chat.completions.parse.return_value = response
            with pytest.raises(ValueError, match='歌词为空'):
                prepare_song_text(dict(url=URL), {}, result, 30., catalog_root=None, model='test')
    # A complete fallback remains preferred; words are evidence, not a source
    # of fabricated text or replacement timestamps.
    assert select_recovery_text(complete, initial, words, 'Chinese')[0] == complete
    assert [(w['start'], w['end']) for w in words] == [(1,4), (6,9), (11,14), (16,19)]


@pytest.mark.parametrize('ending', ['hallucinated', 'genuine', 'clean', 'earlier_anchor', 'wrong_anchor', 'outside_audio'])
def test_repeated_asr_ending_is_redecoded_locally_without_dropping_genuine_repeats(tmp_path, monkeypatch, ending):
    import soundfile as sf
    from copy import deepcopy
    from pathlib import Path
    audio, output = tmp_path/'audio.wav', tmp_path/'asr'
    sf.write(audio, np.full(18000, .01), 1000)
    output.mkdir()
    prefix = [dict(start=0., end=1., text='Before the ending'),
              dict(start=3., end=5., text='A familiar opening phrase')]
    initial = dict(language='english', segments=prefix+[
        dict(start=8.+i, end=8.8+i, text='Sing this phrase', no_speech_prob=.9 if i >= 6 else .1)
        for i in range(8)], words=[dict(start=s['start'], end=s['end'], word=s['text']) for s in prefix]+[
        dict(start=8.+i, end=8.+i+(.5 if ending == 'clean' else 0), word='Sing this phrase') for i in range(8)])
    untouched, calls = deepcopy(initial), []
    repeated = ending == 'genuine'
    parts = deepcopy(prefix)+[
        dict(start=8.+i, end=8.7+i, text='Sing this phrase') for i in range(8 if repeated else 1)]
    replacement = dict(language='english', segments=parts,
        words=[dict(start=s['start'], end=s['end'], word=s['text']) for s in parts])
    if ending == 'earlier_anchor':
        replacement['segments'][1]['text'] = 'A corrected nearby phrase'
        replacement['words'][1]['word'] = 'A corrected nearby phrase'
    if ending == 'wrong_anchor':
        for segment in replacement['segments'][:2]:
            segment['text'] = 'A different recording entirely'
    if ending == 'outside_audio':
        replacement['segments'][-1]['end'] = 20.
        replacement['words'][-1]['end'] = 20.

    def create(**request):
        calls.append((Path(request['file'].name).name, request['model']))
        if request['model'] != 'whisper-1':
            raise RuntimeError('unverified ending requires recovery')
        return SimpleNamespace(model_dump=lambda: initial if len(calls) == 1 else replacement)

    def local(path, language, target):
        calls.append((path.name, 'local-large-v3-turbo'))
        assert path.is_file() and target == output and language == 'english'
        return replacement

    monkeypatch.setattr('karaoke.transcribe_lyrics.cached_local_recognition', local)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    if ending in ('wrong_anchor', 'outside_audio'):
        with pytest.raises(ValueError, match='局部核验'):
            transcribe(client, audio, 'auto', '', output)
        assert not json.loads((output/'loop-repair.json').read_text())['accepted']
    else:
        result, recovery = transcribe(client, audio, 'auto', '', output)
        if ending == 'clean':
            assert not recovery['triggered'] and result == initial and len(calls) == 1
        else:
            assert recovery['triggered'] and recovery['loop_repair']['accepted']
            assert recovery['loop_repair']['right'] is None
            assert calls == [('audio.wav', 'whisper-1'), ('loop-repair.mp3', 'local-large-v3-turbo')]
            assert result['segments'][:1] == prefix[:1]
            assert result['segments'][1]['text'] == replacement['segments'][1]['text']
            assert len(result['segments']) == 2+(8 if repeated else 1)
            assert result['words'][:1] == initial['words'][:1]
            assert [(w['start'], w['end']) for w in result['words'][2:]] == [
                (w['start'], w['end']) for w in replacement['words'][2:]]
    assert initial == untouched
    assert not (output/'loop-repair.mp3').exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_failed_job_reuses_only_verified_downloads_on_retry(tmp_path, corrupt):
    store = Store(tmp_path)
    old, _ = store.create(JobRequest(url=URL), 'old')
    source = tmp_path/old['id']
    (source/'download').mkdir()
    video = source/'download/source.mkv'
    video.write_bytes(b'downloaded video')
    (source/'source.json').write_text(json.dumps(dict(url=URL, duration=10, sha256=file_hash(video))))
    (source/'download/source.info.json').write_text(json.dumps(dict(title='Cached recording')))
    store.cache.compact(store.update(old['id'], status='failed'))
    if corrupt:
        video.write_bytes(b'corrupt')
    retried, _ = store.create(JobRequest(url=URL), 'new')
    folder = tmp_path/retried['id']
    assert (folder/'download/source.mkv').exists() is not corrupt
    if not corrupt:
        pipeline = Pipeline(retried['input'], folder)
        with patch.object(pipeline, 'run', side_effect=AssertionError('Must not download again')):
            metadata, duration = pipeline.download(artwork=False)
        assert metadata['title'] == 'Cached recording' and duration == 10
        # Eviction of the old failed job's video must not affect the queued retry.
        store.cache.enforce()
        store.cache.set_limit(store.cache.stats['used_bytes']-1)
        assert not video.exists() and (folder/'download/source.mkv').read_bytes() == b'downloaded video'
    forced, _ = store.create(JobRequest(url=URL, force=True), 'new')
    assert not (tmp_path/forced['id']/'download').exists()


def test_lru_uses_real_access_persists_and_evicted_song_can_be_requested_again(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    store = app.state.store
    first, second, third = [cache_fixture(store, str(i)) for i in (1,2,3)]
    with TestClient(app) as client:
        # Reading a list is not a song access. Fetching media, including a range, is.
        response = client.get(f"/v1/jobs/{first['id']}/artifacts/mv", headers={'Range':'bytes=0-3'})
        assert response.status_code == 206 and response.content == b'xxxx'
        assert not store.cache.leases
        client.get('/v1/library')
        store.cache.enforce()
        store.cache.set_limit(store.cache.stats['used_bytes'] - store.cache.size(store.root/second['id']))
        assert store.read(first['id'])['status'] == store.read(third['id'])['status'] == 'completed'
        assert store.read(second['id'])['status'] == 'evicted'
        assert not (store.root/second['id']).exists()
        assert client.get(f"/v1/jobs/{second['id']}/artifacts/mv").status_code == 410
        library = client.get('/v1/library').json()['songs']
        assert next(s for s in library if s['id']==first['id'])['job_id']
        assert next(s for s in library if s['id']==second['id'])['job_id'] is None
        assert Store(tmp_path).read(first['id'])['last_accessed_at'] == store.read(first['id'])['last_accessed_at']
        # A stale library button can transparently enqueue the evicted song again.
        store.cache.set_limit(GIB)
        retried = client.post('/v1/room/entries', json={'job_id':second['id']}).json()['entries'][0]
        assert retried['job_id'] != second['id']
        assert finished(client,retried['job_id'])['status'] == 'completed'


def test_lru_protects_playlist_running_jobs_and_active_downloads(tmp_path):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    first, second, downloaded = [cache_fixture(store,str(i)) for i in (1,2,3)]
    active, _ = store.create(JobRequest(url=URL, force=True), 'test')
    store.update(active['id'], status='running')
    (store.root/active['id']/'working.wav').write_bytes(b'w'*2048)
    room = app.state.room.read()
    room.update(entries=[dict(id='playing',job_id=first['id']),dict(id='waiting',job_id=second['id'])],current_id='playing')
    app.state.room.save(room)
    store.cache.acquire(downloaded['id'])
    store.cache.enforce()
    used = store.cache.stats['used_bytes']
    with pytest.raises(CacheFull):
        store.cache.set_limit(used-1)
    store.cache.limit = used-1
    assert not store.cache.enforce()
    assert all((store.root/j['id']).exists() for j in (first,second,downloaded,active))
    store.cache.release(downloaded['id'])
    assert store.cache.enforce()
    assert store.read(downloaded['id'])['status'] == 'evicted'
    assert all((store.root/j['id']).exists() for j in (first,second,active))


def test_preview_and_fallback_keep_resumable_work_and_failure_diagnostics(tmp_path):
    from karaoke.stage_cache import StageCache
    store = Store(tmp_path/'cache')
    job = cache_fixture(store, '1')
    folder = store.root/job['id']
    working = folder/'retained-stem.wav'
    with StageCache(folder).stage('separation', inputs=[folder/'request.json'], outputs=[working]):
        working.write_bytes(b'computed audio')
    diagnostic = folder/'highlight-validation.json'
    diagnostic.write_text('{"errors":[{"line":1}]}')
    result = dict(job['result'], validation_scope='playback_audio_preview',
                  quality=dict(playback_fallback=dict(subtitle_mode='pending')))
    store.cache.compact(store.update(job['id'], result=result))
    assert working.exists() and diagnostic.exists()
    result['validation_scope'] = 'playback_with_lyrics_review'
    store.cache.compact(store.update(job['id'], result=result))
    assert diagnostic.exists()
    assert not working.exists()  # Published audio replaces temporary separation work.


def test_compaction_keeps_outputs_and_bounded_logs_without_following_symlinks(tmp_path):
    store = Store(tmp_path/'cache')
    job = cache_fixture(store,'1')
    folder = store.root/job['id']
    (folder/'raw').mkdir()
    (folder/'raw/input.wav').write_bytes(b'working')
    (folder/'worker.log').write_bytes(b'l'*100000)
    outside = tmp_path/'precious.wav'
    outside.write_bytes(b'keep me')
    (folder/'raw/link').symlink_to(outside)
    store.cache.compact(job)
    assert (folder/'mv.mp4').read_bytes() == b'x'*2048
    assert (folder/'request.json').exists() and (folder/'worker.log').stat().st_size == 65536
    assert not (folder/'raw').exists() and outside.read_bytes() == b'keep me'
    store.update(job['id'], status='failed', result=None, error=dict(code='cache_full'))
    (folder/'asr').mkdir()
    (folder/'asr/transcription.json').write_text('{"words": []}')
    (folder/'alignment').mkdir()
    (folder/'alignment/raw.json').write_text('{"segments": []}')
    store.cache.compact(store.read(job['id']))
    assert not (folder/'mv.mp4').exists()
    assert (folder/'asr/transcription.json').exists()
    assert (folder/'alignment/raw.json').exists()
    store.cache.set_limit(1)
    assert not folder.exists()
    assert store.read(job['id'])['status'] == 'failed'
    assert store.read(job['id'])['error']['code'] == 'cache_full'


def test_oversized_worker_is_stopped_cleaned_and_queue_can_continue(tmp_path):
    def writer(folder):
        return [sys.executable,'-c',
                'import pathlib,sys,time;pathlib.Path(sys.argv[1],"working.wav").write_bytes(b"x"*400000);time.sleep(30)',str(folder)]
    app = create_app(tmp_path, worker_command=writer, version='test', cache_limit_bytes=200000)
    with TestClient(app) as client:
        data = client.post('/v1/room/entries',json={'url':URL}).json()
        job = finished(client,data['entries'][0]['job_id'])
        assert job['status'] == 'failed' and job['error']['code'] == 'cache_full'
        assert app.state.worker.process is None
        assert not (tmp_path/job['id']/'working.wav').exists()
        assert client.get('/v1/storage').json()['used_bytes'] < 200000
        app.state.worker.command = fake_worker
        retry = client.post('/v1/room/commands',json={'action':'retry','entry_id':data['entries'][0]['id']}).json()
        assert finished(client,retry['entries'][0]['job_id'])['status'] == 'completed'


def test_cache_limit_setting_is_validated_and_survives_restart(tmp_path):
    with TestClient(create_app(tmp_path,version='test')) as client:
        assert client.post('/v1/storage',json={'limit_gib':0}).status_code == 422
        saved = client.post('/v1/storage',json={'limit_gib':2}).json()
        assert saved['storage']['limit_bytes'] == 2*GIB
    with TestClient(create_app(tmp_path,version='test')) as client:
        assert client.get('/v1/storage').json()['limit_bytes'] == 2*GIB


def test_low_free_disk_blocks_processing_without_evicting_the_playlist(tmp_path):
    def must_not_start(folder):
        raise AssertionError('The worker must not start without disk headroom')
    app = create_app(tmp_path, worker_command=must_not_start, version='test')
    pinned = cache_fixture(app.state.store,'1')
    with TestClient(app) as client:
        client.post('/v1/room/entries',json={'job_id':pinned['id']})
        with patch('karaoke.cache.shutil.disk_usage', return_value=SimpleNamespace(free=GIB//2)):
            response = client.post('/v1/room/entries',json={'url':URL}).json()
            job = finished(client,response['entries'][-1]['job_id'])
            assert job['status'] == 'failed' and job['error']['code'] == 'cache_full'
            assert client.get('/v1/storage').json()['can_prepare'] is False
        assert (tmp_path/pinned['id']/'mv.mp4').exists()


def test_video_search_normalizes_results_and_rejects_unplayable_links():
    from karaoke.video_search import search_results
    bili = search_results('bilibili', [dict(bvid='BV1bK411W797', title='<em>歌曲</em> &amp; 现场',
        duration='1:02:03', pic='//i0.hdslb.com/cover.jpg', author='Uploader',
        official_identity={'type':0, 'desc':'歌手'})])
    assert bili == [dict(url='https://www.bilibili.com/video/BV1bK411W797', title='歌曲 & 现场',
        duration=3723, thumbnail='https://i0.hdslb.com/cover.jpg', uploader='Uploader', provider='bilibili',
        official_evidence='bilibili_certified_artist')]
    item = dict(url=URL, title='Video title', duration=180, channel_is_official_artist=True,
                thumbnails=[{'url':'https://i.ytimg.com/cover.jpg'}])
    results = search_results('youtube', [item, item, dict(item, live_status='is_live'),
        dict(item, url='http://127.0.0.1/private'), dict(item, url='https://youtube.com.attacker.test/'),
        dict(item, url='https://www.bilibili.com/video/BV1bK411W797')])
    assert len(results) == 1 and results[0]['url'] == URL
    assert results[0]['thumbnail'] == 'https://i.ytimg.com/cover.jpg'
    assert search_results('youtube', [dict(item, thumbnail='https://localhost/cover')])[0]['thumbnail'] is None


def test_search_api_failure_is_distinct_from_empty_and_creates_no_jobs(tmp_path, monkeypatch):
    from karaoke import video_search
    async def fetch(provider, query):
        if provider == 'bilibili':
            raise RuntimeError('provider unavailable')
        return []
    monkeypatch.setattr(video_search, 'fetch_results', fetch)
    app = create_app(tmp_path, version='test')
    with TestClient(app) as client:
        found = client.get('/v1/search', params=dict(provider='youtube', q='  some song  '))
        assert found.status_code == 200 and found.json()['results'] == []
        assert found.json()['query'] == 'some song'
        assert client.get('/v1/search', params=dict(provider='bilibili', q='some song')).status_code == 502
        for params in [dict(provider='other', q='song'), dict(provider='youtube', q=' '), dict(provider='youtube', q='a'*161)]:
            assert client.get('/v1/search', params=params).status_code == 422
        assert client.get('/v1/room').json()['entries'] == []
        assert app.state.store.list() == []


def test_video_search_deduplicates_inflight_queries_and_bounds_load(monkeypatch):
    import asyncio
    from karaoke import video_search
    async def scenario():
        service = video_search.VideoSearch()
        release, started = asyncio.Event(), asyncio.Event()
        calls, active, peak = [], 0, 0
        async def fetch(provider, query):
            nonlocal active, peak
            calls.append((provider, query)); active += 1; peak = max(peak, active)
            started.set()
            try:
                await release.wait()
                return [query]
            finally:
                active -= 1
        monkeypatch.setattr(video_search, 'fetch_results', fetch)
        one = asyncio.create_task(service.search('youtube', 'same'))
        await started.wait()
        two = asyncio.create_task(service.search('youtube', 'same'))
        await asyncio.sleep(0)
        one.cancel()
        await asyncio.gather(one, return_exceptions=True)
        others = [asyncio.create_task(service.search('bilibili', str(i))) for i in range(7)]
        await asyncio.sleep(0)
        with pytest.raises(OverflowError):
            await service.search('youtube', 'overflow')
        release.set()
        assert await two == ['same']
        await asyncio.gather(*others)
        assert peak <= 2 and calls.count(('youtube', 'same')) == 1
        assert await service.search('youtube', 'same') == ['same']
        assert len(calls) == 8 and not service.pending
        # Expired entries refresh, and the metadata cache has a fixed size.
        service.cache[('youtube', 'same')] = (0, ['stale'])
        assert await service.search('youtube', 'same') == ['same']
        for i in range(40):
            await service.search('youtube', str(i))
        assert len(service.cache) == 32
        await service.close()
    asyncio.run(scenario())


def test_cancelling_video_search_terminates_its_subprocess(monkeypatch):
    import asyncio
    from karaoke import video_search
    async def scenario():
        original = asyncio.create_subprocess_exec
        started, children = asyncio.Event(), []
        async def spawn(*args, **kwargs):
            process = await original(sys.executable, '-c', 'import time; time.sleep(60)', **kwargs)
            children.append(process); started.set()
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
        task = asyncio.create_task(video_search.fetch_results('youtube', 'anything'))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert children[0].returncode is not None
    asyncio.run(scenario())


@pytest.mark.parametrize('provider,item,allowed', [
    ('youtube', {'title':'A song', 'channel_is_official_artist':True}, True),
    ('youtube', {'title':'A song (Official Music Video)', 'channel':'Example Records', 'channel_is_verified':True}, True),
    ('youtube', {'title':'A song (Official Music Video)', 'channel':'Example Records'}, False),
    ('youtube', {'title':'A song (Lyrics)', 'channel':'A verified creator', 'channel_is_verified':True}, False),
    ('youtube', {'title':'A song (Official Music Video)', 'channel':'Fan Music', 'channel_is_verified':True}, False),
    ('youtube', {'title':'A song Cover', 'channel_is_official_artist':True}, True),
    ('youtube', {'title':'《某首歌》某组合翻唱 / 某卫视官方HD', 'channel':'某卫视 Music Channel', 'channel_is_verified':True}, True),
    ('youtube', {'title':'A song (Unofficial)', 'channel':'Example Music', 'channel_is_verified':True}, False),
    ('bilibili', {'title':'官方 MV', 'author':'官方音乐'}, False),
    ('bilibili', {'title':'官方 MV', 'official_identity':{'type':0, 'desc':'bilibili 知名音乐UP主'}}, False),
    ('bilibili', {'title':'A song', 'official_identity':{'type':0, 'desc':'歌手、音乐制作人'}}, True),
    ('bilibili', {'title':'A song 翻唱', 'official_identity':{'type':0, 'desc':'歌手'}}, True),
    ('bilibili', {'title':'A song', 'official_identity':{'type':1, 'desc':'某唱片公司官方账号'}}, True),
    ('bilibili', {'title':'A song', 'official_identity':{'type':1, 'desc':'某科技公司官方账号'}}, False),
    ('bilibili', {'title':'官方 MV', 'official_identity':None}, False),
])
def test_official_video_filter_requires_source_evidence(provider, item, allowed):
    from karaoke.video_search import official_source
    assert bool(official_source(provider, item)) == allowed


def test_bilibili_verification_uses_certification_not_self_description(monkeypatch):
    import asyncio
    import httpx
    from karaoke import video_search
    calls = []
    def respond(request):
        calls.append(request)
        mid = request.url.params['mid']
        if mid == '3':
            return httpx.Response(503)
        return httpx.Response(200, json={'code':0, 'data':{'card':{
            'name':'Official Music', 'sign':'I am the official artist',
            'official_verify':{'type':1, 'desc':'Example 唱片公司官方账号'} if mid == '1' else {'type':-1, 'desc':''}
        }}})
    async def scenario():
        video_search._bilibili_identities.clear()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            slots = asyncio.Semaphore(4)
            verified = await video_search.bilibili_identity(client, '1', slots)
            assert await video_search.bilibili_identity(client, '1', slots) == verified
            unknown = await video_search.bilibili_identity(client, '2', slots)
            assert video_search.official_source('bilibili', {'official_identity':verified})
            assert not video_search.official_source('bilibili', {'official_identity':unknown})
            assert await video_search.bilibili_identity(client, '3', slots) is None
            assert len(calls) == 3
        video_search._bilibili_identities.clear()
    asyncio.run(scenario())


def test_official_search_keeps_only_a_few_releases_without_padding():
    from karaoke.video_search import search_results
    def item(index, **extra):
        return dict(url=f'https://www.youtube.com/watch?v={index:011d}', title=f'Cover {index}',
                    channel_is_official_artist=True, **extra)
    found = search_results('youtube', [item(i) for i in range(12)])
    assert [x['title'] for x in found] == ['Cover 0', 'Cover 1', 'Cover 2']
    assert len(search_results('youtube', [item(0), {'url':URL,'title':'A fan upload'}])) == 1


@pytest.mark.parametrize('query,title,channel,expected', [
    ('周杰伦 晴天','周杰倫【晴天 Sunny Day】','周杰倫 Jay Chou', True),
    ('周杰伦 晴天','周杰倫【星晴 Starry Mood】','周杰倫 Jay Chou', False),
    ('凤凰传奇海底','《海底》凤凰传奇 官方舞台','Music TV', True),
    ('海底凤凰传奇','凤凰传奇《海底》','Music TV', True),
    ('米津玄師Lemon','Kenshi Yonezu - Lemon','米津玄師', True),
    ('Ed Sheeran Perfect','Ed Sheeran - Thinking Out Loud','Ed Sheeran', False),
    ('Ed Sheeran Perfect','Ed Sheeran - Perfect Symphony','Ed Sheeran', True),
])
def test_search_matches_song_and_performer_without_related_song_padding(query, title, channel, expected):
    from karaoke.video_search import matches_query
    assert matches_query(query, dict(title=title,channel=channel)) == expected


def test_bilibili_identity_cools_down_after_rate_limit(monkeypatch):
    import asyncio
    import httpx
    from karaoke import video_search
    calls = []
    def limited(request):
        calls.append(request)
        return httpx.Response(200, json={'code':-352})
    async def scenario():
        monkeypatch.setattr(video_search, '_bilibili_retry_at', 0)
        monkeypatch.setattr(video_search, '_bilibili_identities', {})
        async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
            slots = asyncio.Semaphore(2)
            results = await asyncio.gather(*(video_search.bilibili_identity(client, str(i), slots) for i in range(4)))
            assert results == [None]*4
            assert len(calls) == 1
    asyncio.run(scenario())
