import hashlib
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urljoin
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


@pytest.mark.parametrize('bad_state', ['{', '[]', 'true', 'NaN', '-1'])
def test_corrupt_provider_backoff_does_not_kill_dispatch_or_bypass_hold(tmp_path, bad_state):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    (tmp_path/'preparation.pause').touch()
    folder = tmp_path/'stage-locks'; folder.mkdir()
    state = folder/'youtube.retry.json'; state.write_text(bad_state)
    with TestClient(app) as client:
        blocked = client.post('/v1/jobs', json={'url':URL}).json()['id']
        other = client.post('/v1/jobs', json={'url':'https://www.bilibili.com/video/BV1bK411W797'}).json()['id']
        (tmp_path/'preparation.pause').unlink()
        assert finished(client, other)['status'] == 'completed'
        assert client.get('/v1/jobs/'+blocked).json()['status'] == 'queued'
        health = client.get('/health').json()
        assert health['status'] == 'degraded' and blocked in health['dispatch_blocked_jobs']
        assert not app.state.worker.dispatcher.done() and state.read_text() == bad_state
        state.write_text(json.dumps(time.time()-.1))
        assert finished(client, blocked)['status'] == 'completed'
        assert client.get('/health').json()['status'] == 'ok'


def test_bad_progress_messages_preserve_the_worker_and_valid_result(tmp_path):
    def command(folder):
        worker = fake_worker(folder)
        messages = ['{', 'null', '[]', '{"song_info":[]}', '{"active_stages":null}',
                    json.dumps(dict(song_info=dict(title='Kept metadata')))]
        worker[2] = '\n'.join('print('+repr('KARAOKE_PROGRESS '+m)+', flush=True)' for m in messages)+'\n'+worker[2]
        return worker
    with TestClient(create_app(tmp_path, worker_command=command, version='test')) as client:
        job_id = client.post('/v1/jobs', json={'url':URL}).json()['id']
        job = finished(client, job_id)
        assert job['status'] == 'completed' and job['song_info']['title'] == 'Kept metadata'
        assert client.get(f'/v1/jobs/{job_id}/artifacts/mv').content == b'0123456789'


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


@pytest.mark.parametrize('missing', ['poster', 'mv'])
def test_optional_cover_failure_does_not_hide_valid_media(tmp_path, missing):
    def worker(folder):
        command = fake_worker(folder)
        command[2] += f"\nr=json.loads((p/'result.json').read_text());r['artifacts'][{missing!r}]={{'path':'missing.jpg','bytes':100,'sha256':'absent'}}\n(p/'result.json').write_text(json.dumps(r))\n"
        return command
    with TestClient(create_app(tmp_path, worker_command=worker, version='test')) as client:
        job_id = client.post('/v1/jobs', json={'url':URL}).json()['id']
        job = finished(client, job_id)
        assert job['status'] == ('completed' if missing == 'poster' else 'failed')
        if missing == 'poster':
            assert 'poster' not in job['result']['artifacts']
            assert client.get(f'/v1/jobs/{job_id}/artifacts/mv').content == b'0123456789'


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
        assert 'path' not in artifact and artifact['url'].endswith('/artifacts/mv?v=' + artifact['sha256'])
        response = client.get(artifact['url'], headers={'Range': 'bytes=2-5'})
        assert response.status_code == 206 and response.content == b'2345'
        assert client.get(f'/v1/jobs/{job_id}/artifacts/request.json').status_code == 404
        cached = client.post('/v1/jobs', json={'url': URL})
        assert cached.status_code == 200 and cached.json()['id'] == job_id
        (tmp_path/job_id/'mv.mp4').unlink()
        repaired = client.post('/v1/jobs', json={'url': URL})
        assert repaired.status_code == 202 and repaired.json()['id'] != job_id
        assert finished(client, repaired.json()['id'])['status'] == 'completed'


@pytest.mark.parametrize('rate,playback_rate', [(16000,48000), (44100,48000), (48000,44100)])
def test_playback_windows_preserve_channels_gain_and_boundaries_and_release_leases(tmp_path, rate, playback_rate):
    import soundfile as sf
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job, _ = store.create(JobRequest(url=URL), 'test')
    folder = tmp_path/job['id']
    samples = np.random.default_rng(7).uniform(-.7, .7, (rate*9, 2))
    artifacts = {}
    names = ('instrumental', 'backing_vocals', 'lead_vocals')
    for i, name in enumerate(names):
        path = folder/f'{name}.wav'
        sf.write(path, samples/(i+1), rate, subtype='DOUBLE')
        artifacts[name] = dict(path=path.name, bytes=path.stat().st_size)
    result = dict(artifacts=artifacts, shared_preview_gain=.6)
    store.update(job['id'], status='completed', result=result, last_accessed_at='2000-01-01T00:00:00Z')
    # Do not start the worker: only exercise the existing completed artifacts.
    client = TestClient(app)
    url = f"/v1/jobs/{job['id']}/playback"
    info = client.get(url, params={'sample_rate':playback_rate}).json()
    assert info['frames'] == rate*9 and info['channels'] == 2
    assert info['padding_frames']*playback_rate % rate == 0
    assert store.read(job['id'])['last_accessed_at'] == '2000-01-01T00:00:00Z'
    for i, name in enumerate(names):
        parts = []
        for index in (0, 1):
            response = client.get(f'{url}/{name}/{index}', params={'sample_rate':playback_rate})
            assert response.status_code == 200
            assert response.headers['content-type'] == 'audio/flac'
            audio, sample_rate = sf.read(io.BytesIO(response.content), always_2d=True)
            assert sample_rate == rate
            parts.append(audio[:rate*8] if index == 0 else audio[info['padding_frames']:])
        decoded = np.concatenate(parts)
        np.testing.assert_allclose(decoded, samples*.6/(i+1), rtol=0, atol=9e-8)
        assert not store.cache.leases
    assert store.read(job['id'])['last_accessed_at'] != '2000-01-01T00:00:00Z'
    assert client.get(f'{url}/instrumental/2').status_code == 409
    assert client.get(f'{url}/instrumental/-1').status_code == 404
    assert client.get(f'{url}/unknown/0').status_code == 404
    artifacts['lead_vocals']['path'] = '../outside.wav'
    store.update(job['id'], result=result)
    assert client.get(url).status_code == 409
    assert not store.cache.leases
    store.update(job['id'], status='evicted')
    assert client.get(url).status_code == 410
    store.update(job['id'], status='running')
    assert client.get(url).status_code == 409


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


@pytest.mark.parametrize('stage', ['download', 'download_video'])
def test_download_retry_survives_restart_without_blocking_other_songs(tmp_path, stage):
    from karaoke.stage_cache import StageCache
    calls = []

    def command(folder):
        calls.append(folder.name)
        if calls.count(folder.name) == 1 and len(calls) == 1:
            source = folder/'source.wav'
            output = folder/'separated.wav'
            source.write_bytes(b'source')
            with StageCache(folder).stage('separate', inputs=[source], outputs=[output]):
                output.write_bytes(b'expensive output')
            return [sys.executable, '-c',
                f"import pathlib,json,sys;p=pathlib.Path(sys.argv[1]);"
                f"(p/'{stage}.log').write_text('HTTP Error 403: Forbidden');"
                f"(p/'error.json').write_text(json.dumps({{'code':'{stage}_failed',"
                "'message':'视频站点拒绝下载（403）；请检查下载器与登录信息后重试'}));sys.exit(1)", str(folder)]
        return fake_worker(folder)

    app = create_app(tmp_path, worker_command=command, version='test')
    with TestClient(app) as client:
        first = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][0]
        deadline = time.monotonic()+5
        while not app.state.store.read(first['job_id']).get('download_retry') and time.monotonic()<deadline:
            time.sleep(.02)
        job = app.state.store.read(first['job_id'])
        retry = job['download_retry']
        assert job['status'] == 'queued' and retry['count'] == 1
        assert retry['next_retry_at']-retry['history'][-1]['at'] == 5
        assert (tmp_path/job['id']/'separated.wav').read_bytes() == b'expensive output'
        assert len(client.get('/v1/room').json()['entries']) == 1
        assert client.get('/v1/room').json()['entries'][0]['next_retry_at'] == retry['next_retry_at']
        assert client.post('/v1/jobs', json={'url':URL}).json()['id'] == job['id']
        other = client.post('/v1/jobs', json={'url':'https://youtu.be/C0ppulkmXPw'}).json()
        assert finished(client, other['id'])['status'] == 'completed'
        assert calls.count(job['id']) == 1
    restarted = create_app(tmp_path, worker_command=command, version='test')
    with TestClient(restarted) as client:
        assert restarted.state.store.read(job['id'])['download_retry'] == retry
        # Advance the persisted timer so restart recovery does not depend on wall time.
        restarted.state.store.update(job['id'], download_retry={**retry, 'next_retry_at':time.time()-.1})
        result = finished(client, job['id'])
        assert result['status'] == 'completed' and calls.count(job['id']) == 2
        assert result['error'] is None and result['download_retry']['next_retry_at'] is None
        assert len(result['download_retry']['history']) == 1
        assert client.get('/v1/room').json()['entries'][0]['job_id'] == job['id']


@pytest.mark.parametrize('last_error,expected_attempts', [('download_failed', 4), ('alignment_failed', 2)])
def test_automatic_download_retry_is_bounded_and_does_not_retry_other_failures(tmp_path, monkeypatch,
                                                                            last_error, expected_attempts):
    from karaoke import api
    original = api.retry_fields
    def short_retry(*args, **kwargs):
        result = original(*args, **kwargs)
        if result['next_retry_at']:
            result['next_retry_at'] = time.time()+.05
        return result
    monkeypatch.setattr(api, 'retry_fields', short_retry)
    calls = []
    def command(folder):
        calls.append(folder.name)
        error = dict(code='download_failed' if len(calls) == 1 else last_error, message='连接超时')
        return [sys.executable, '-c', "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);"
            "(p/'error.json').write_text(sys.argv[2]);sys.exit(1)", str(folder), json.dumps(error)]
    with TestClient(create_app(tmp_path, worker_command=command, version='test')) as client:
        job_id = client.post('/v1/jobs', json={'url':URL}).json()['id']
        result = finished(client, job_id)
        assert result['status'] == 'failed' and len(calls) == expected_attempts
        assert result['error']['code'] == last_error
        assert result['download_retry']['next_retry_at'] is None


def test_removing_last_room_entry_cancels_pending_download_retry(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    (tmp_path/'preparation.pause').touch()
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][0]
        duplicate = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][1]
        job = app.state.store.read(entry['job_id'])
        app.state.worker.download_failed(job, dict(code='download_failed',message='连接超时'), True)
        client.post('/v1/room/commands', json={'action':'remove','entry_id':entry['id']})
        assert app.state.store.read(job['id'])['status'] == 'queued'
        client.post('/v1/room/commands', json={'action':'remove','entry_id':duplicate['id']})
        result = app.state.store.read(job['id'])
        assert result['status'] == 'failed' and result['download_retry']['next_retry_at'] is None
        assert app.state.worker.queue.empty()
        # Removing while a download is running also prevents a later auto retry.
        app.state.worker.download_failed(result, dict(code='download_failed',message='连接超时'), True)
        assert app.state.store.read(job['id'])['status'] == 'failed'
        assert app.state.worker.queue.empty()


def test_provider_limit_and_login_hold_survive_restart_and_allow_other_providers(tmp_path):
    from karaoke.job_schema import download_backoff
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    job, _ = app.state.store.create(JobRequest(url=URL), 'test')
    folder = tmp_path/job['id']
    (folder/'download.log').write_text('HTTP Error 429: Too Many Requests\nRetry-After: 420\nSign in to confirm')
    app.state.worker.download_failed(job, dict(code='download_failed',message='登录验证'), False)
    result = app.state.store.read(job['id'])
    assert result['status'] == 'failed' and result['download_retry']['next_retry_at'] is None
    until = download_backoff(URL, tmp_path/'stage-locks')
    assert 410 < until-time.time() <= 420
    assert download_backoff(URL, tmp_path/'stage-locks', until=time.time()+10) == until
    restarted = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(restarted) as client:
        waiting = client.post('/v1/jobs', json={'url':'https://youtu.be/C0ppulkmXPw'}).json()
        other = client.post('/v1/jobs', json={'url':'https://www.bilibili.com/video/BV1bK411W797'}).json()
        assert finished(client, other['id'])['status'] == 'completed'
        assert client.get(f"/v1/jobs/{waiting['id']}").json()['status'] == 'queued'
        assert client.get(f"/v1/jobs/{job['id']}").json()['status'] == 'failed'


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
    ('HTTP Error 503: Service Unavailable','external',False),
    ('The read operation timed out','external',False),
    ('HTTP Error 404: Not Found','blocked',False),
    ('Video unavailable','blocked',False),
    ('Unable to connect to proxy','blocked',False),
])
def test_video_retry_does_not_apply_one_resource_error_to_every_song(diagnostic,status,global_limit):
    from karaoke.tools.soak import retry_fields
    policy=retry_fields({'code':'download_failed'},diagnostic,1,failed_at=1000)
    assert policy['status']==status and policy['provider_blocked']==global_limit
    assert policy['next_retry_at'] is None if status=='blocked' and not global_limit else 1000<policy['next_retry_at']<=1300
    if 'Try again in 120 seconds' in diagnostic:
        assert policy['next_retry_at']==1120


@pytest.mark.parametrize('code', ['download_failed', 'download_video_failed'])
def test_download_retry_distinguishes_login_advice_and_respects_long_server_delays(code):
    from karaoke.api import retry_fields
    error = dict(code=code, message='视频站点拒绝下载（403）；请检查下载器与登录信息后重试')
    for number, delay in enumerate((5, 20, 80), 1):
        policy = retry_fields(error, '', number, failed_at=1000)
        assert policy['status'] == 'external' and policy['next_retry_at'] == 1000+delay
        policy = retry_fields(error, 'HTTP Error 429: Too Many Requests', number, failed_at=1000)
        assert policy['provider_blocked'] and policy['next_retry_at'] == 1000+60*2**(number-1)
    policy = retry_fields(error, 'HTTP Error 403: Forbidden\nRetry-After: 120', 1, failed_at=1000)
    assert not policy['provider_blocked'] and policy['next_retry_at'] == 1120
    policy = retry_fields(error, 'HTTP Error 403: Forbidden\nSign in to confirm', 1, failed_at=1000)
    assert policy['status'] == 'blocked' and policy['next_retry_at'] is None
    for diagnostic in ('The read operation timed out', 'HTTP Error 503: Service Unavailable',
                       'Failed to resolve www.youtube.com'):
        policy = retry_fields(error, diagnostic, 1, failed_at=1000)
        assert policy['status'] == 'external' and policy['next_retry_at'] == 1005
    policy = retry_fields(error, 'HTTP Error 429: Too Many Requests\nRetry-After: 3600', 1, failed_at=1000)
    assert policy['provider_blocked'] and policy['next_retry_at'] == 4600
    policy = retry_fields(error, 'HTTP Error 429: Too Many Requests\nRetry-After: 3600\nLogin required', 1, failed_at=1000)
    assert policy['status'] == 'blocked' and policy['provider_blocked'] and policy['next_retry_at'] == 4600


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
    (folder/'alignment').mkdir()
    raw_alignment = dict(context=['A measured phrase']*130000, segments=[dict(text='A measured phrase', words=[])])
    (folder/'alignment/raw.json').write_text(json.dumps(raw_alignment))
    rejected_source = folder/'lyric-source-attempts/1'
    rejected_source.mkdir(parents=True)
    (rejected_source/'playback-warning.json').write_text('{"omitted_lines": [2]}')
    final_validation = {
        'cues.json': [dict(text='final cue', start=2., end=3.)],
        'audio-result.json': dict(duration_seconds=4., artifacts={'subtitles': {'path': 'cues.json'}}),
        'result.json': dict(duration_seconds=4., artifacts={'mv': {'path': 'mv.mp4'}}),
        'lyric-coverage-validation.json': dict(status='incomplete', uncovered_passages=[dict(start=0., end=2.)]),
        'localization-validation.json': [dict(context=[0,1],conflicts=[dict(line=1,observed_range=[10.,12.])])],
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
    assert saved['alignment/raw.json'] == raw_alignment
    assert saved['lyric-source-attempts/1/playback-warning.json'] == {'omitted_lines': [2]}
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


@pytest.mark.parametrize('lead_file', ['lead-separation/song/vocals.wav', 'stems/lead.wav'])
def test_timing_review_cache_tracks_its_additional_audio_view(tmp_path, monkeypatch, lead_file):
    from karaoke import pipeline as module
    code, folder = tmp_path/'code', tmp_path/'job'
    code.mkdir()
    monkeypatch.setattr(module, 'ROOT', code)
    pipeline = Pipeline(dict(url=URL), folder)
    (folder/'lyrics.json').write_text(json.dumps(dict(language='Chinese')))
    lead = folder/lead_file
    lead.parent.mkdir(parents=True)
    lead.write_bytes(b'lead version 1')
    base = [sys.executable, code/'timing_validation.py']
    command = base+['--lead', lead]
    spec = pipeline.stage_spec('validate_timing', command)
    for path in spec['inputs']+spec['code']:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
    calls = []
    def run(*args):
        calls.append(True)
        output = folder/'timing-checked'
        output.mkdir()
        (output/'cues.json').write_text('[]')
    monkeypatch.setattr(pipeline, 'run_process', run)
    pipeline.run('validate_timing', base)
    lead.write_bytes(b'lead version 2')
    pipeline.run('validate_timing', base)
    assert len(calls)==1
    pipeline.run('validate_timing', command)
    pipeline.run('validate_timing', command)
    assert len(calls)==2
    lead.write_bytes(b'lead version 3')
    pipeline.run('validate_timing', command)
    assert len(calls)==3
    localization = folder/'alignment/raw.json'
    localization.parent.mkdir(exist_ok=True)
    localization.write_text('[]')
    pipeline.run('validate_timing', command)
    pipeline.run('validate_timing', command)
    assert len(calls)==4
    localization.write_text('[{"asr_localizations": []}]')
    pipeline.run('validate_timing', command)
    assert len(calls)==5
    localization.unlink()
    pipeline.run('validate_timing', command)
    assert len(calls)==6


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
    assert calls == ['align_words', 'align_context']
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
    assert calls == ['align_words', 'align_phonemes', 'align_context']
    calls.clear()
    models[0].write_bytes(b'updated Whisper model')
    prepare()
    assert calls == ['align_words', 'transcribe']
    calls.clear()
    for filename, affected in (
            ('cjk_recovery.py', ['align_words', 'align_context']),
            ('align_lyric_phonemes.py', ['align_words', 'align_phonemes', 'align_context']),
            ('local_asr.py', ['align_words', 'transcribe']),
            ('align_lyrics_whisper.py', ['align_words'])):
        path = code/filename
        path.write_bytes(path.read_bytes()+b'updated')
        prepare()
        assert calls == affected
        calls.clear()
    (folder/'source-44100.wav').write_bytes(b'updated original audio')
    prepare()
    assert calls == ['align_words', 'transcribe', 'align_context']
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
    for path in pipeline.stage_spec('align_context', stages['align_context'])['code']:
        if not path.exists():
            path.write_text('input')
    prepare()
    calls.clear()
    for asset in (dictionary, g2p, models[3]):
        fingerprint = module.pipeline_fingerprint()
        asset.write_bytes(asset.read_bytes()+b'updated')
        assert module.pipeline_fingerprint() != fingerprint
        prepare()
        assert calls == (['align_words', 'align_context'] if asset == models[3] else ['align_context'])
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
    dict(plainLyrics='毫不相干的文字'), dict(plainLyrics='[Music]'),
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
    song['title'] = '晚風 - Evening Wind'
    assert lyric_candidate(dict(status='ok', results=[row]), song, transcription, 265.)[0]
    candidate, report = lyric_candidate(dict(status='ok', results=[dict(row, **change)]), song, transcription, 265.)
    assert candidate is None and report['status'] == 'no_supported_recording'


def test_lyric_candidate_preserves_sung_words_in_brackets():
    from karaoke.lyric_lookup import lyric_candidate
    body = ['Stay [with me] beside the quiet shore until the morning light returns (sing with me)',
            'Sing once more beneath the stars']
    row = dict(id=1, trackName='Evening', artistName='Singer', duration=30.,
               instrumental=False, plainLyrics='[Verse 1]\n'+'\n'.join(body))
    lines, report = lyric_candidate(dict(status='ok', results=[row]),
        dict(title='Evening', artist='Singer'), None, 30.)
    assert lines == body and report['excluded_credit_lines'] == [1]


def test_lyric_text_search_skips_a_title_only_result_before_alignment():
    from karaoke.lyric_lookup import lyric_candidate
    song = dict(title='Evening',artist='Singer',language='English')
    row = dict(id=1,trackName='Evening',artistName='Singer',duration=30.,instrumental=False,
               plainLyrics='Evening - Singer (Artist)')
    lyrics = 'Stars above the quiet shore\nI will sing once more'
    lines, report = lyric_candidate(dict(status='ok',results=[row,dict(row,id=2,plainLyrics=lyrics)]),song,None,30.)
    assert lines == lyrics.splitlines() and report['id'] == 2


@pytest.mark.parametrize('source_title,album,duration,reason', [
    ('Evening', 'Evening', 249., 'duration'),
    ('Evening (Live at Concert Hall)', 'Evening', 257., 'live_performance'),
    ('Evening (Live at Concert Hall)', 'Live at Other Hall', 257., 'live_performance')])
def test_published_text_can_cross_recordings_without_relaxing_asr_validation(source_title, album, duration, reason):
    from karaoke.lyric_lookup import lyric_candidate
    song = dict(title='Evening', artist='Singer')
    row = dict(id=1, trackName='Evening', artistName='Singer', albumName=album,
               duration=duration, instrumental=False, plainLyrics='\n'.join(['Stars above the quiet shore'] * 6))
    packet = dict(status='ok', results=[dict(row, artistName='Someone else'), row])
    assert lyric_candidate(packet, song, None, 257., source_title=source_title)[0] is None
    lines, report = lyric_candidate(packet, song, None, 257., source_title=source_title, allow_recording_mismatch=True)
    assert lines == row['plainLyrics'].splitlines() and reason in report['recording_mismatches']
    assert report['text_support'] is None and report['audio_word_coverage'] is None
    assert not report['provider_timestamps_used']
    with pytest.raises(ValueError, match='independent ASR'):
        lyric_candidate(packet, song, dict(words=[]), 257., allow_recording_mismatch=True)
    packet['results'].append(dict(row, id=2, duration=257., albumName=source_title))
    _, report = lyric_candidate(packet, song, None, 257., source_title=source_title, allow_recording_mismatch=True)
    assert report['id'] == 2 and not report['recording_mismatches']


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
    elif audio_only:
        with pytest.raises(StageFailure, match='ASR reached'):
            pipeline.prepare_text({}, 30, lead)
    else:
        with pytest.raises(StageFailure, match='未找到可靠'):
            pipeline.prepare_text({}, 30, lead)
        assert 'transcribe' not in stages


@pytest.mark.parametrize('title', ['晚風', '晚風 (Official Music Video)', '晚風 [Official Audio]'])
def test_lyrics_first_search_uses_metadata_duration_and_retains_review_status(tmp_path, monkeypatch, title):
    from karaoke import lyric_lookup, lyric_consensus
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
    monkeypatch.setattr(lyric_lookup, 'fetch_published_lyrics', lambda *a: (None, dict(status='unavailable')))
    monkeypatch.setattr(lyric_consensus, 'fetch_additional_lyrics', lambda provider,*a:
        (row['plainLyrics'].splitlines() if provider == 'youtube_music' else None,
         dict(provider=provider, language='Chinese')))
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', lambda *a: pytest.fail('Tagged recording needs no model'))
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL), dict(track=title, artist='林林', language='en'),
                                               30, StageCache(tmp_path), 'unused', None)
    assert len(queries) == 2 and result['lines'] == row['plainLyrics'].splitlines()
    assert result['language'] == 'Chinese'
    provenance = result['provenance']
    assert provenance['needs_review'] and provenance['route'] == 'lyrics_first'
    assert provenance['independent_asr_coverage'] is None
    assert provenance['lyric_lookup']['status'] == 'candidate'
    assert provenance['lyric_lookup']['audio_word_coverage'] is None
    assert provenance['lyric_lookup']['text_sha256'] == hashlib.sha256(row['plainLyrics'].encode()).hexdigest()
    assert provenance['lyric_lookup']['metadata_language_corrected'] == dict(original='English', resolved='Chinese')


@pytest.mark.parametrize('recovered', [True, False])
@pytest.mark.parametrize('explicit_identity', [True, False])
def test_unmatched_lyrics_retry_cached_native_identity_once_without_repeating_blocked_providers(
        tmp_path, monkeypatch, recovered, explicit_identity):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    calls, metadata_calls, limited_calls = [], [], []
    body = ['微風裡慢慢歌唱', '星光下靜靜傾聽']
    def extract(*args):
        metadata_calls.append(1)
        return dict(parsed=dict(title='晚風 Evening Wind', artist='林林 Lin Lin', lyric_line_numbers=[]))
    def published(song, *args):
        calls.append((song['title'], song['artist']))
        found = recovered and song['artist'] == '林林 Lin Lin'
        return (body if found else None), dict(provider='line_music',language='Chinese',
            status='candidate' if found else 'no_matching_published_text')
    def additional(provider, *args):
        if provider == 'netease':
            limited_calls.append(provider)
            return None, dict(provider=provider,status='unavailable',http_status=429,error='HTTPStatusError')
        return None, dict(provider=provider,status='unavailable')
    monkeypatch.setattr(lyric_lookup,'extract_song_metadata',extract)
    monkeypatch.setattr(lyric_lookup,'fetch_published_lyrics',published)
    monkeypatch.setattr(lyric_lookup,'fetch_lyrics',lambda *a:dict(status='no_supported_recording'))
    monkeypatch.setattr(lyric_consensus,'fetch_additional_lyrics',additional)
    request = dict(url=URL)
    if explicit_identity:
        request.update(title='Evening',artist='Lin')
    result = lyric_lookup.prepare_online_lyrics(request,
        dict(track='Evening',artist='Lin',description='Published album and artist information'),
        30,StageCache(tmp_path),'test-model',None)
    assert len(metadata_calls) == 1 and limited_calls == ['netease']
    assert calls == [('Evening','Lin')] + ([] if explicit_identity else [
        ('晚風 Evening Wind','Lin'), ('晚風 Evening Wind','林林 Lin Lin')])
    assert bool(result) == (recovered and not explicit_identity)
    if result:
        assert result['lines'] == body
        recovery = result['provenance']['lyric_lookup']['identity_recovery']
        assert recovery['recovered'] and recovery['previous_lookup']['crosscheck']['unavailable_sources']
    else:
        report = json.loads((tmp_path/'lyrics-first-lookup.json').read_text())
        assert not report.get('identity_recovery',{}).get('recovered',False)


@pytest.mark.parametrize('corroborated', [True, False])
def test_isolated_lyric_source_reuses_native_metadata_but_requires_corroboration(tmp_path, monkeypatch, corroborated):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    original=['古い街を歩いている', '知らない道を進んでいる']
    recovered=['星空に色を重ねて', '夜の向こうで歌っている']
    metadata_calls, limited_calls, queries=[],[],[]
    def extract(*args):
        metadata_calls.append(1)
        return dict(parsed=dict(title='夜の色',artist='はる',lyric_line_numbers=[]))
    def published(song, *args):
        queries.append(song['artist'])
        return (recovered if song['artist']=='はる' else None), dict(provider='line_music',language='Japanese')
    def lrclib(song, *args):
        body=original if song['artist']=='Haru' else recovered if corroborated else None
        return dict(status='ok',results=[] if body is None else [dict(id=1,trackName='夜の色',
            artistName=song['artist'],duration=30,plainLyrics='\n'.join(body),instrumental=False)])
    def additional(provider, *args):
        if provider=='netease':
            limited_calls.append(1)
            return None,dict(provider=provider,status='unavailable',http_status=429,error='HTTPStatusError')
        return None,dict(provider=provider,status='unavailable')
    monkeypatch.setattr(lyric_lookup,'extract_song_metadata',extract)
    monkeypatch.setattr(lyric_lookup,'fetch_published_lyrics',published)
    monkeypatch.setattr(lyric_lookup,'fetch_lyrics',lrclib)
    monkeypatch.setattr(lyric_consensus,'fetch_additional_lyrics',additional)
    result=lyric_lookup.prepare_online_lyrics(dict(url=URL),
        dict(track='夜の色',artist='Haru',description='はるの公開作品情報'),30,StageCache(tmp_path),'unused',None)
    assert result['lines']==(recovered if corroborated else original)
    assert result['artist']==('はる' if corroborated else 'Haru')
    assert queries==['Haru','はる'] and metadata_calls==[1] and limited_calls==[1]
    recovery=result['provenance']['lyric_lookup']['identity_recovery']
    assert recovery['recovered']==corroborated
    assert recovery['previous_lookup']['candidate_texts']==[original]
    assert recovery['attempts'][0]['lookup']['candidate_texts'][0]==recovered


@pytest.mark.parametrize('failure', ['timeout', 'server', 'long_backoff', 'authentication'])
def test_missing_lyrics_retry_transient_metadata_and_keep_confirmed_artist(tmp_path, monkeypatch, failure):
    import httpx
    from openai import APITimeoutError, InternalServerError, RateLimitError, AuthenticationError
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    request = httpx.Request('POST', 'https://example.test')
    response = httpx.Response(429, request=request, headers={'retry-after': '120'})
    error = {'timeout': APITimeoutError(request=request),
             'server': InternalServerError('server', response=httpx.Response(503, request=request), body=None),
             'long_backoff': RateLimitError('rate limit', response=response, body=None),
             'authentication': AuthenticationError('auth', response=response, body=None)}[failure]
    calls, waits, queries = [], [], []
    def extract(*args):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return dict(parsed=dict(title='晚風', artist='Singer; session musician', lyric_line_numbers=[]))
    def search(song, *args, **kwargs):
        queries.append((song['title'], song['artist']))
        if callable(args[3]):
            args[3]()
        found = song['title'] == '晚風' and song['artist'] == '林林'
        return (['微風裡慢慢歌唱'] if found else None), dict(provider='line_music', language='Chinese')
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', extract)
    monkeypatch.setattr(lyric_lookup.time, 'sleep', waits.append)
    monkeypatch.setattr(lyric_consensus, 'fetch_lyric_consensus', search)
    recovered = failure in ('timeout', 'server')
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL),
        dict(track='Evening', artist='林林', description='Published metadata'),
        30, StageCache(tmp_path), 'unused', None)
    assert bool(result) == recovered
    assert len(calls) == (2 if recovered else 1)
    assert len(waits) == int(recovered)
    if recovered:
        assert 0 <= waits[0] <= 30 and result['artist'] == '林林'
        assert queries == [('Evening','林林'), ('晚風','林林')]
    report = json.loads((tmp_path/'lyrics-first-lookup.json').read_text())
    assert report['metadata_retries'][0]['attempted'] == recovered


def test_published_lyrics_use_explicit_video_credits_without_a_text_model(tmp_path, monkeypatch):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    row = dict(trackId=123, trackTitle='晚風', artists=[dict(artistName='林林')],
               album=dict(albumTitle='春天'), lyrics='歌名：晚風\n詞曲：林林\n\n微風裡慢慢歌唱\n星光下靜靜傾聽')
    calls = []
    def fetch(url, *, headers):
        calls.append(url)
        assert headers and '/api/search/' in url
        return dict(response=dict(result=dict(tracks=[dict(row, artists=[dict(artistName='別人')]),
            dict(row, album=dict(albumTitle='現場演唱會')), row])))
    monkeypatch.setattr(lyric_lookup, 'read_document', fetch)
    monkeypatch.setattr(lyric_lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(lyric_consensus, 'fetch_additional_lyrics', lambda provider,*a:
        (['微風裡慢慢歌唱', '星光下靜靜傾聽'] if provider == 'youtube_music' else None,
         dict(provider=provider, language='Chinese')))
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', lambda *a: pytest.fail('Explicit credits need no model'))
    metadata = dict(title='林林 Lin Lin【晚風 Evening Wind】-Official Music Video', uploader='Unrelated channel')
    for _ in range(2):
        result = lyric_lookup.prepare_online_lyrics(dict(url=URL), metadata, 30,
                                                   StageCache(tmp_path), 'unused', None)
    assert len(calls) == 1
    assert result['lines'] == ['微風裡慢慢歌唱', '星光下靜靜傾聽']
    report = result['provenance']['lyric_lookup']
    assert report['provider'] == 'line_music' and report['excluded_credit_lines'] == [1, 2]
    assert report['url'].endswith('/123') and not report['provider_timestamps_used']
    assert result['provenance']['independent_asr_coverage'] is None


def test_known_video_credits_do_not_skip_lyrics_in_its_description(tmp_path, monkeypatch):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    metadata = dict(title='Lin Lin 林林【Evening Wind 晚風】Official Music Video',
        description='晚風\n詞曲：林林\n微風裡慢慢歌唱\n星光下靜靜傾聽\n#Official')
    calls = []
    def extract(evidence, *args):
        calls.append(evidence)
        return dict(parsed=dict(title='Wrong inferred title', artist='Wrong inferred artist', lyric_line_numbers=[3,4]))
    def consensus(song, metadata, duration, cache, description, preserve_title=False):
        assert song['title'] == 'Evening Wind 晚風' and song['artist'] == 'Lin Lin 林林'
        return description() if callable(description) else description
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', extract)
    monkeypatch.setattr(lyric_consensus, 'fetch_lyric_consensus', consensus)
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL), metadata, 30, StageCache(tmp_path), 'unused', None)
    assert len(calls) == 1 and result['lines'] == ['微風裡慢慢歌唱', '星光下靜靜傾聽']
    assert result['provenance']['lyric_lookup']['lyric_line_numbers'] == [3,4]


def test_lyric_sources_start_while_description_is_being_read(tmp_path, monkeypatch):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    source_started = threading.Event()
    text = ['A little song', 'A second line']
    def extract(*args):
        assert source_started.wait(5), 'Lyric sources waited for metadata inference'
        return dict(parsed=dict(title='Song', artist='Singer', lyric_line_numbers=[1, 2]))
    def published(*args):
        source_started.set()
        return text, dict(provider='line_music', language='English')
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', extract)
    monkeypatch.setattr(lyric_lookup, 'fetch_published_lyrics', published)
    monkeypatch.setattr(lyric_lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(lyric_consensus, 'fetch_additional_lyrics', lambda provider, *a:
        (None, dict(provider=provider, status='unavailable')))
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL),
        dict(title='Song', track='Song', artist='Singer', description='\n'.join(text)),
        30, StageCache(tmp_path), 'unused', None)
    assert result['lines'] == text
    crosscheck = result['provenance']['lyric_lookup']['crosscheck']
    assert crosscheck['status'] == 'corroborated' and crosscheck['independent_sources'] == 2
    assert {s['provider'] for s in crosscheck['sources']} == {'line_music', 'video_description'}


@pytest.mark.parametrize('case', ['agreement', 'minority', 'missing_tail', 'same_publisher', 'tie', 'word_boundary'])
def test_lyric_crosscheck_preserves_repetitions_and_requires_independent_agreement(case):
    from karaoke.lyric_consensus import compare_lyrics
    original = ['風裡慢慢唱', '星光陪著我', '風裡怎麼唱']
    candidates = [(original, dict(provider='one')),
                  (['风里慢慢唱，星光陪着我。风里怎么唱！'], dict(provider='two'))]
    if case == 'minority':
        candidates.append((['完全不同的文字'], dict(provider='three')))
    elif case == 'missing_tail':
        candidates[1] = (original[:-1], dict(provider='two'))
    elif case == 'same_publisher':
        candidates[0][1]['publisher'] = candidates[1][1]['publisher'] = 'one-publisher'
    elif case == 'tie':
        candidates.extend([(['别的歌词'], dict(provider=p)) for p in ('three','four')])
    elif case == 'word_boundary':
        candidates = [(['Now here'],dict(provider='one')), (['Nowhere'],dict(provider='two'))]
    lines, report = compare_lyrics(candidates)
    assert bool(lines) == (case in ('agreement','minority'))
    if lines:
        assert lines == original and report['independent_sources'] == 2
    else:
        assert report['status'] in ('conflicting', 'insufficient_sources')
    assert len(report['comparisons']) == len(candidates)*(len(candidates)-1)//2


@pytest.mark.parametrize('provider_fails', [False, True])
def test_lyric_sources_run_concurrently_and_one_failure_does_not_erase_agreement(tmp_path, monkeypatch, provider_fails):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    barrier = threading.Barrier(4)
    body = ['Stars above the quiet shore', 'Stars above the quiet shore']
    calls = []
    def published(*args):
        calls.append('line_music');barrier.wait(5)
        return body, dict(provider='line_music', language='English')
    def lrclib(*args):
        calls.append('lrclib');barrier.wait(5)
        return dict(status='ok', results=[dict(id=1, trackName='Evening', artistName='Singer',
            duration=30, instrumental=False, plainLyrics='\n'.join(body))])
    def additional(provider, *args):
        calls.append(provider);barrier.wait(5)
        if provider_fails and provider == 'netease':
            raise OSError('provider failed')
        return body, dict(provider=provider, language='English', publisher='youtube' if provider == 'youtube_music' else provider)
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', published)
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lrclib)
    monkeypatch.setattr(module, 'fetch_additional_lyrics', additional)
    lines, report = module.fetch_lyric_consensus(dict(title='Evening',artist='Singer'), {}, 30,
        StageCache(tmp_path), (body,dict(provider='video_description',publisher='youtube',language='English')))
    assert lines == body and set(calls) == {'line_music','lrclib','netease','youtube_music'}
    assert report['crosscheck']['independent_sources'] == (3 if provider_fails else 4)
    assert bool(report['crosscheck']['unavailable_sources']) == provider_fails


@pytest.mark.parametrize('missing_identity', [False, True])
def test_description_failure_keeps_other_published_lyrics(tmp_path, monkeypatch, missing_identity):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    body = ['Stars above the quiet shore', 'I will sing once more']
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (body, dict(provider='line_music', language='English')))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(module, 'fetch_additional_lyrics', lambda provider, *a:
        (None, dict(provider=provider, status='unavailable')))
    def broken_description(*args):
        raise TypeError('Malformed description response')
    song = dict(title='Evening', artist='' if missing_identity else 'Singer')
    lines, report = module.fetch_lyric_consensus(song, {}, 30, StageCache(tmp_path), broken_description)
    assert lines == body and report['provider'] == 'line_music'
    assert dict(provider='video_description', status='unavailable', error='TypeError') in report['crosscheck']['unavailable_sources']


def test_source_disagreements_keep_published_text_and_full_diagnostics(tmp_path, monkeypatch):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    published = ['Stars above the quiet shore', 'I will sing once more']
    monkeypatch.setattr(module.lookup,'fetch_published_lyrics',lambda *a:
        (published,dict(provider='line_music',language='English')))
    monkeypatch.setattr(module.lookup,'fetch_lyrics',lambda *a:dict(status='unavailable'))
    monkeypatch.setattr(module,'fetch_additional_lyrics',lambda provider,*a:
        (published[:-1] if provider == 'netease' else None,dict(provider=provider,language='English')))
    lines, report = module.fetch_lyric_consensus(dict(title='Evening',artist='Singer'),{},30,StageCache(tmp_path))
    assert lines == published and report['provider'] == 'line_music'
    assert report['crosscheck']['status'] == 'conflicting'
    assert report['crosscheck']['selection'] == 'published_source'
    assert report['crosscheck']['comparisons'][0]['differences']
    evidence = json.loads((tmp_path/'lyric-crosscheck.json').read_text())
    assert evidence['candidate_texts'] == [published,published[:-1]]


@pytest.mark.parametrize('fragment', ['林林 - 晚風', '晚風 / 林林', '這是僅有的一句', 'Singer — Evening'])
@pytest.mark.parametrize('all_short', [False, True])
def test_short_lyrics_retry_other_sources_before_alignment(tmp_path, monkeypatch, fragment, all_short):
    from karaoke import lyric_lookup, lyric_consensus
    body = ['微風裡我們輕輕慢慢地歌唱', '星光下我們靜靜細細地傾聽'] * 3
    monkeypatch.setattr(lyric_lookup, 'fetch_published_lyrics', lambda *a:
        ([fragment], dict(provider='line_music', language='Chinese')))
    monkeypatch.setattr(lyric_lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(lyric_consensus, 'fetch_additional_lyrics', lambda provider, *a:
        (([fragment] if all_short else body) if provider == 'youtube_music' else None,
         dict(provider=provider, language='Chinese')))
    pipeline = Pipeline(dict(url=URL, title='晚風', artist='林林'), tmp_path)
    if all_short:
        with pytest.raises(StageFailure, match='未找到可靠'):
            pipeline.prepare_text(dict(track='晚風', artist='林林'), 300)
        assert not (tmp_path/'lyrics.json').exists()
    else:
        result = pipeline.prepare_text(dict(track='晚風', artist='林林'), 300)
        assert result['lines'] == body
        assert result['provenance']['lyric_lookup']['provider'] == 'youtube_music'
    review = json.loads((tmp_path/'lyric-crosscheck.json').read_text())['crosscheck']
    assert len(review['rejected_sources']) == (2 if all_short else 1)
    assert all(r['reason'] == 'lyrics_too_short' for r in review['rejected_sources'])
    if all_short:
        assert review['status'] == 'lyrics_incomplete' and not review['agreeing_sources']


@pytest.mark.parametrize('other_performance', [False, True])
def test_lyric_excerpt_yields_to_fuller_source_for_same_recording(tmp_path, monkeypatch, other_performance):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    fragment = ['Stars above the quiet shore']
    body = ['We will sing as the night begins', *fragment, 'And once again we all sing together']
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (fragment, dict(provider='line_music', language='English')))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(module, 'fetch_additional_lyrics', lambda provider, *a:
        (body if provider == 'youtube_music' else None, dict(provider=provider, language='English',
            recording_mismatches=['live_performance'] if other_performance else [])))
    lines, report = module.fetch_lyric_consensus(dict(title='Evening', artist='Singer'), {}, 30, StageCache(tmp_path))
    assert lines == (fragment if other_performance else body)
    rejected = report['crosscheck']['rejected_sources']
    assert len(rejected) == (0 if other_performance else 1)
    if rejected:
        assert rejected[0]['reason'] == 'lyrics_fragment' and rejected[0]['fuller_source'] == 1


@pytest.mark.parametrize('independent', [True, False])
@pytest.mark.parametrize('other_performance', [False, True])
@pytest.mark.parametrize('with_description', [False, True])
def test_similar_lyric_majority_respects_recording_identity(tmp_path, monkeypatch, independent, other_performance, with_description):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    outlier = ['A different lyric in another language']
    body = 'Stars above the quiet shore and every night we sing and walk along the road back home again'
    variants = [body, body.replace('Stars', 'Lights'), body.replace('night', 'day')]
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (outlier, dict(provider='line_music', language='English')))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(module, 'fetch_additional_lyrics', lambda provider, *a:
        ([variants[0 if provider == 'netease' else 1]], dict(provider=provider, language='English',
            recording_mismatches=['live_performance'] if other_performance else [],
            publisher=provider if independent else 'shared')))
    lines, report = module.fetch_lyric_consensus(dict(title='Evening', artist='Singer'), {}, 30, StageCache(tmp_path),
        ([variants[2]], dict(provider='video_description', language='English', publisher='youtube' if independent else 'shared',
                            recording_mismatches=['live_performance'] if other_performance else [])) if with_description else None)
    review = report['crosscheck']
    assert review['status'] == 'conflicting' and not review['agreeing_sources']
    assert lines == ([variants[0 if with_description else 1]] if independent and not other_performance else outlier)
    assert review['selection'] == ('similarity_support' if independent and not other_performance else 'published_source')


def test_exact_lyric_agreement_from_other_performances_does_not_override_current_recording(tmp_path, monkeypatch):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    other, current = ['An extra verse at another show'], ['A shorter verse tonight']
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (other, dict(provider='line_music', language='English', recording_mismatches=['live_performance'])))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(module, 'fetch_additional_lyrics', lambda provider, *a:
        (current if provider == 'youtube_music' else other, dict(provider=provider, language='English',
         recording_mismatches=[] if provider == 'youtube_music' else ['live_performance'])))
    lines, report = module.fetch_lyric_consensus(dict(title='Evening',artist='Singer'), {}, 30, StageCache(tmp_path))
    assert lines == current and report['provider'] == 'youtube_music'
    assert report['crosscheck']['status'] == 'conflicting'
    assert report['crosscheck']['unselected_agreeing_sources'] == [0, 1]


def test_equally_supported_lyrics_choose_closest_whole_text_before_provider_priority(tmp_path, monkeypatch):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    body = ('Stars above the quiet shore and every night we sing and walk along the road back home again '
            'The morning brings another chance to find a melody that keeps us close together as the seasons turn')
    texts = {'line_music': [body.replace('quiet', 'distant').replace('night', 'day')],
             'youtube_music': [body], 'netease': [body.replace('Stars', 'Lights')]}
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (texts['line_music'], dict(provider='line_music', language='English')))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a: dict(status='unavailable'))
    monkeypatch.setattr(module, 'fetch_additional_lyrics', lambda provider, *a:
        (texts[provider], dict(provider=provider, language='English')))
    lines, report = module.fetch_lyric_consensus(dict(title='Evening',artist='Singer'), {}, 30, StageCache(tmp_path))
    assert lines == texts['youtube_music']
    assert report['candidate_texts'] == [texts[p] for p in ('line_music','netease','youtube_music')]
    assert report['crosscheck']['selection'] == 'similarity_support'
    assert len(report['crosscheck']['supporting_sources']) == 3
    assert all(p['similarity'] >= .9 for p in report['crosscheck']['comparisons'])
    assert report['crosscheck']['status'] == 'conflicting'  # Similarity is not exact agreement.


def test_netease_lyrics_reject_other_recordings_and_strip_lrc_metadata(tmp_path, monkeypatch):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    calls = []
    def read(url, **kwargs):
        calls.append(url)
        if 'cloudsearch' in url:
            return dict(code=200,result=dict(songs=[
                dict(id=i,name=title,ar=[dict(name=artist)],al=dict(name=album))
                for i,title,artist,album in [(1,'晚風','其他歌手','春天'),
                    (2,'晚風','林林','Live'),(3,'晚風','林林','春天')]]))
        assert 'id=3' in url
        return dict(lrc=dict(lyric='[ar:林林]\n[00:00.00] 作词：林林\n[00:01.00] 微風裡慢慢歌唱\n[00:05.00] 星光下靜靜傾聽'))
    monkeypatch.setattr(module, 'read_document', read)
    for _ in range(2):
        lines, report = module.fetch_additional_lyrics('netease',dict(title='晚風',artist='林林'),{},StageCache(tmp_path))
        assert lines == ['微風裡慢慢歌唱', '星光下靜靜傾聽']
        assert report['provider'] == 'netease' and report['url'].endswith('id=3')
        assert report['excluded_credit_lines'] == [1,2] and not report['provider_timestamps_used']
    assert len(calls) == 2


@pytest.mark.parametrize('album', ['Original release', None])
def test_netease_lyrics_use_release_metadata_before_provider_order(tmp_path, monkeypatch, album):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    from karaoke.song_sources import video_song_info
    calls = []
    def read(url, **kwargs):
        calls.append(url)
        if 'cloudsearch' in url:
            return dict(code=200, result=dict(songs=[
                dict(id=1, name='Evening', ar=[dict(name='Singer')], al=dict(name='Later release'), dt=310680),
                dict(id=2, name='Evening', ar=[dict(name='Singer')], al=dict(name='Original release'), dt=316933)]))
        assert 'id=2' in url
        return dict(lrc=dict(lyric='\n'.join(['Stars above the quiet shore']*8)))
    monkeypatch.setattr(module, 'read_document', read)
    metadata = dict(title='Evening',artist='Singer',album=album)
    for _ in range(2):
        lines, report = module.fetch_additional_lyrics('netease', video_song_info(metadata),
            metadata, StageCache(tmp_path), 316.933)
        assert lines and report['url'].endswith('id=2')
        assert report['duration_seconds'] == 316.933
    assert len(calls) == 2


def test_published_lyric_release_preference_keeps_complete_fallbacks():
    from karaoke.lyric_lookup import published_candidate
    body='\n'.join(['Stars above the quiet shore']*8)
    row=dict(trackId=1,trackTitle='Evening',artists=[dict(artistName='Singer')],
             album=dict(albumTitle='Other release'),duration_seconds=100.,lyrics=body)
    same=dict(row,trackId=2,album=dict(albumTitle='Original release'),duration_seconds=101.)
    song=dict(title='Evening',artist='Singer',album='Original release')
    _, report=published_candidate([row,same],song,'',duration=100.)
    assert report['url'].endswith('/2')
    for invalid in (dict(same,lyrics='Stars'),dict(same,duration_seconds=250.)):
        lines, report=published_candidate([invalid,row],song,'',duration=100.)
        assert lines and report['url'].endswith('/1')
    lines, report=published_candidate([row],song,'',duration=100.)
    assert lines and report['url'].endswith('/1')


@pytest.mark.parametrize('same_recording', [True, False])
@pytest.mark.parametrize('artist', ['Singer', ''])
def test_music_lyrics_are_bound_to_the_requested_video(tmp_path, monkeypatch, same_recording, artist):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    import ytmusicapi
    calls = []
    body = 'Stars above the quiet shore will guide me home tonight (sing with me)'
    class Music:
        def __init__(self, **kwargs): pass
        def get_watch_playlist(self, videoId, limit):
            calls.append(videoId)
            return dict(lyrics='lyrics-id',tracks=[dict(videoId=videoId if same_recording else 'another',
                title='Evening', artists=[dict(name='Confirmed singer')])])
        def get_lyrics(self, key):
            return dict(lyrics=body,source='Source: LyricFind')
    monkeypatch.setattr(ytmusicapi,'YTMusic',Music)
    for _ in range(2):
        lines, report = module.fetch_additional_lyrics('youtube_music',dict(title='Evening',artist=artist),
            dict(id='abcdefghijk'),StageCache(tmp_path))
        assert bool(lines) == same_recording
    assert len(calls) == 1
    if same_recording:
        assert lines == [body]
        assert report['publisher'] == 'lyricfind' and not report['provider_timestamps_used']
        assert report['recording_identity']['artist'] == 'Confirmed singer'
    else:
        assert not report.get('recording_identity')


@pytest.mark.parametrize('lyric_response', ['absent', 'failure'])
@pytest.mark.parametrize('manual_title', [None, 'User title'])
def test_source_identity_survives_missing_lyrics_and_enables_other_sources(tmp_path, monkeypatch, lyric_response, manual_title):
    from karaoke import lyric_lookup, lyric_consensus
    import ytmusicapi
    calls = []
    class Music:
        def __init__(self, **kwargs): pass
        def get_watch_playlist(self, videoId, limit):
            calls.append(videoId)
            return dict(lyrics='lyrics-id' if lyric_response == 'failure' else None,
                tracks=[dict(videoId=videoId,title='Evening',artists=[dict(name='Confirmed singer')])])
        def get_lyrics(self, key):
            raise ValueError('This lyric endpoint is unavailable')
    monkeypatch.setattr(ytmusicapi,'YTMusic',Music)
    monkeypatch.setattr(lyric_lookup, 'extract_song_metadata', lambda *a:pytest.fail('Source identity needs no model'))
    def published(song, *args):
        assert song['artist'] == 'Confirmed singer'
        assert song['title'] == (manual_title or 'Evening')
        return ['Stars above the quiet shore'], dict(provider='line_music',language='English')
    monkeypatch.setattr(lyric_lookup, 'fetch_published_lyrics', published)
    monkeypatch.setattr(lyric_lookup, 'fetch_lyrics', lambda *a:dict(status='unavailable'))
    additional = lyric_consensus.fetch_additional_lyrics
    monkeypatch.setattr(lyric_consensus, 'fetch_additional_lyrics', lambda provider,*args:
        additional(provider,*args) if provider=='youtube_music' else (None,dict(provider=provider,status='unavailable')))
    for _ in range(2):
        result = Pipeline(dict(url=URL,title=manual_title),tmp_path).prepare_text(
            dict(title='Channel - Evening (official video)',id='abcdefghijk'),30)
        assert result['artist'] == 'Confirmed singer' and result['lines'] == ['Stars above the quiet shore']
        assert result['title'] == (manual_title or 'Evening')
    assert calls == ['abcdefghijk']


@pytest.mark.parametrize('http_status', [401, 429])
def test_lyric_rate_limit_is_shared_between_jobs_without_blocking_other_sources(tmp_path, monkeypatch, http_status):
    import httpx
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    clock, calls = [1000.], []
    monkeypatch.setattr(module.time,'time',lambda:clock[0])
    def limited(url, **kwargs):
        calls.append('line_music')
        response = httpx.Response(http_status, headers={'Retry-After':'900'}, request=httpx.Request('GET',url))
        response.raise_for_status()
    monkeypatch.setattr(module.lookup,'read_document',limited)
    monkeypatch.setattr(module.lookup,'fetch_lyrics',lambda *a:dict(status='unavailable'))
    monkeypatch.setattr(module,'fetch_additional_lyrics',lambda provider,*a:
        (['The quiet shore'],dict(provider=provider,language='English')))
    for i in range(3):
        if i == 2:clock[0]=1901.
        (tmp_path/str(i)).mkdir()
        lines, report = module.fetch_lyric_consensus(dict(title='Evening',artist='Singer'),{},30,StageCache(tmp_path/str(i)))
        assert lines == ['The quiet shore']
    assert calls == (['line_music','line_music'] if http_status == 429 else ['line_music'])
    if http_status == 401:
        (tmp_path/'lyric-provider-backoff/line_music.json').unlink()
        module.lookup.fetch_published_lyrics(dict(title='Evening',artist='Singer'),None,StageCache(tmp_path/'0'),30)
        assert calls == ['line_music','line_music']  # New authorization retries a cached 401.


@pytest.mark.parametrize('http_status', [401, 429])
def test_music_backoff_never_supplies_another_songs_identity(tmp_path, monkeypatch, http_status):
    import httpx
    import ytmusicapi
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    calls, authorized = [], [False]
    class Music:
        def __init__(self, **kwargs): pass
        def get_watch_playlist(self, videoId, limit):
            calls.append(videoId)
            return dict(lyrics='lyrics-id',tracks=[dict(videoId=videoId,title='Song A',artists=[dict(name='Singer A')])])
        def get_lyrics(self, key):
            if authorized[0]:
                return dict(lyrics='Stars above the quiet shore')
            response = httpx.Response(http_status, headers={'Retry-After':'900'}, request=httpx.Request('GET','https://example.com'))
            response.raise_for_status()
    monkeypatch.setattr(ytmusicapi,'YTMusic',Music)
    monkeypatch.setattr(lyric_lookup,'read_document',lambda *a,**kw:dict(response=dict(result=dict(tracks=[]))))
    monkeypatch.setattr(lyric_lookup,'fetch_lyrics',lambda *a:dict(status='unavailable'))
    additional = lyric_consensus.fetch_additional_lyrics
    monkeypatch.setattr(lyric_consensus,'fetch_additional_lyrics',lambda provider,*a:
        additional(provider,*a) if provider=='youtube_music' else (None,dict(status='unavailable')))
    for name in ('a','b'):
        (tmp_path/name).mkdir()
    song_a = dict(title='Upload A',artist='')
    lyric_consensus.fetch_lyric_consensus(song_a,dict(title='Upload A',id='aaaaaaaaaaa'),30,StageCache(tmp_path/'a'))
    assert song_a == dict(title='Song A',artist='Singer A')
    song_b = dict(title='Upload B',artist='')
    lines, _ = lyric_consensus.fetch_lyric_consensus(song_b,dict(title='Upload B',id='bbbbbbbbbbb'),30,StageCache(tmp_path/'b'))
    assert lines is None and song_b == dict(title='Upload B',artist='')
    assert calls == ['aaaaaaaaaaa']
    if http_status == 401:
        authorized[0] = True
        (tmp_path/'lyric-provider-backoff/youtube_music.json').unlink()
        lines, _ = lyric_consensus.fetch_lyric_consensus(song_a,dict(title='Upload A',id='aaaaaaaaaaa'),30,StageCache(tmp_path/'a'))
        assert lines == ['Stars above the quiet shore'] and calls == ['aaaaaaaaaaa']


@pytest.mark.parametrize('complete_identity', [False, True])
@pytest.mark.parametrize('video_title', ['Channel - Evening (cover)', 'Evening'])
def test_source_metadata_precedes_model_and_keeps_description_vote(tmp_path, monkeypatch, complete_identity, video_title):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    calls, source_started = [], threading.Event()
    body = ['Stars above the quiet shore','Morning light will guide me home']
    def additional(provider,*args):
        if provider != 'youtube_music':
            return None, dict(provider=provider,status='unavailable')
        calls.append('source')
        return body, dict(provider=provider,language='English',publisher='lyricfind',
            recording_identity=dict(video_id='abcdefghijk',title='Evening',artist='Singer' if complete_identity else ''))
    def extract(*args):
        assert calls == ['source']
        calls.append('model')
        if complete_identity:
            assert source_started.wait(5), 'Description extraction blocked name-based providers'
        return dict(parsed=dict(title='Wrong inferred title',artist='Singer',lyric_line_numbers=[1,2]))
    def published(song,*args):
        assert song['title'] == 'Evening' and song['artist'] == 'Singer'
        source_started.set()
        return None, dict(provider='line_music',status='unavailable')
    monkeypatch.setattr(lyric_consensus,'fetch_additional_lyrics',additional)
    monkeypatch.setattr(lyric_lookup,'extract_song_metadata',extract)
    monkeypatch.setattr(lyric_lookup,'fetch_published_lyrics',published)
    monkeypatch.setattr(lyric_lookup,'fetch_lyrics',lambda *a:dict(status='unavailable'))
    result = lyric_lookup.prepare_online_lyrics(dict(url=URL),
        dict(id='abcdefghijk',title=video_title,description='\n'.join(body)),30,StageCache(tmp_path),'unused',None)
    assert result['title'] == 'Evening' and result['artist'] == 'Singer'
    assert result['provenance']['lyric_lookup']['crosscheck']['independent_sources'] == 2
    assert calls == ['source','model']


@pytest.mark.parametrize('provider', ['line_music','netease'])
def test_duration_reselects_cached_lyrics_and_can_fetch_a_later_candidate(tmp_path, monkeypatch, provider):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    song = dict(title='Evening',artist='Singer')
    body = 'Stars above the quiet shore\nMorning light will guide me home along the road (where we belong)'
    rows = [dict(trackId=i,trackTitle='Evening',artists=[dict(artistName='Singer')],
        album=dict(albumTitle='Night'),lyrics=text) for i,text in ((1,'Stars'),(2,body))]
    calls = []
    def read(url,**kwargs):
        calls.append(url)
        if provider == 'line_music':
            return dict(response=dict(result=dict(tracks=rows[:1] if 'query=Evening+Singer' in url else rows[1:])))
        if 'cloudsearch' in url:
            return dict(code=200,result=dict(songs=[dict(id=row['trackId'],name='Evening',ar=[dict(name='Singer')],
                al=dict(name='Night')) for row in rows]))
        return dict(lrc=dict(lyric='Stars' if 'id=1&' in url else body))
    monkeypatch.setattr(lyric_lookup if provider=='line_music' else lyric_consensus,'read_document',read)
    folder = tmp_path/'song'
    folder.mkdir()
    cache = StageCache(folder)
    def fetch(duration):
        return (lyric_lookup.fetch_published_lyrics(song,None,cache,duration) if provider=='line_music' else
                lyric_consensus.fetch_additional_lyrics(provider,song,{},cache,duration))
    assert fetch(10)[0] == ['Stars']
    assert len(calls) == (1 if provider=='line_music' else 2)
    lines, report = fetch(100)
    assert lines == body.splitlines() and report['url'].endswith('2')
    hold = tmp_path/'lyric-provider-backoff'/f'{provider}.json'
    hold.parent.mkdir(exist_ok=True)
    hold.write_text(json.dumps(dict(status='unavailable',http_status=401,requires_access=True)))
    assert fetch(10)[0] == ['Stars']
    assert len(calls) == (2 if provider=='line_music' else 3)
    assert len(set(calls)) == len(calls)


def test_music_cached_body_survives_duration_changes_and_other_song_backoff(tmp_path, monkeypatch):
    import ytmusicapi
    from karaoke import lyric_consensus
    from karaoke.stage_cache import StageCache
    calls = []
    class Music:
        def __init__(self, **kwargs): pass
        def get_watch_playlist(self, videoId, limit):
            calls.append('watch')
            return dict(lyrics='lyrics-id',tracks=[dict(videoId=videoId,title='Evening',artists=[dict(name='Singer')])])
        def get_lyrics(self, key):
            calls.append('lyrics')
            return dict(lyrics='Stars above the quiet shore')
    monkeypatch.setattr(ytmusicapi,'YTMusic',Music)
    folder = tmp_path/'song'
    folder.mkdir()
    cache = StageCache(folder)
    first = lyric_consensus.fetch_additional_lyrics('youtube_music',dict(title='Evening',artist='Singer'),dict(id='abcdefghijk'),cache,30)
    hold = tmp_path/'lyric-provider-backoff/youtube_music.json'
    hold.parent.mkdir()
    hold.write_text(json.dumps(dict(status='unavailable',http_status=401,requires_access=True,
        recording_identity=dict(video_id='other-video',title='Wrong song',artist='Wrong singer'))))
    second = lyric_consensus.fetch_additional_lyrics('youtube_music',dict(title='Evening',artist='Singer'),dict(id='abcdefghijk'),cache,40)
    assert first == second and calls == ['watch','lyrics']


@pytest.mark.parametrize('credits', [
    ('詞：林林', '曲：林林', '弦樂編寫：樂手', '錄音師：工程師'),
    ('词：林林', '曲：林林', '第一小提琴：乐手', '混音师：工程师'),
    ('Lyrics: Writer', 'Music: Composer', 'Guitar: Player', 'Mixed by: Engineer'),
    ('專輯：春天', '歌曲：晚風 (Album Version)', '作詞：林林', '作曲：林林'),
    ('OA/OC: A Writer / B Writer', 'ＯＰ：A Publisher', 'SP: B Publisher', 'Publisher: C Publisher')])
@pytest.mark.parametrize('heading', ['晚風', '晚風-林林', '晚風 - 林林', '晚風—林林', '晚風 – 林林'])
def test_published_lyrics_remove_credit_blocks_but_keep_sung_titles_and_colons(credits, heading):
    from karaoke.lyric_lookup import published_candidate, published_lyric_lines
    body = ['晚風', '我说：继续往前走', '晚風']
    raw = '\n'.join([heading, *credits[:2], *body, credits[2], 'A Player', credits[3]])
    row = dict(trackId=123, trackTitle='晚風', artists=[dict(artistName='林林')],
               album=dict(albumTitle='春天'), lyrics=raw)
    lines, report = published_candidate([row], dict(title='晚風', artist='林林'), '')
    assert lines == body
    assert report['excluded_credit_lines'] == [1, 2, 3, 7, 8, 9]
    assert report['text_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    assert published_lyric_lines('\n'.join(body), '晚風') == (body, [])


@pytest.mark.parametrize('heading', ['晚風-林林', '晚風 — 林林', '林林 - 晚風', '晚風-林林\n詞：林林\n曲：林林'])
def test_title_artist_credit_is_not_a_lyric_candidate(heading):
    from karaoke.lyric_lookup import published_candidate, published_lyric_lines, lyric_candidate
    song = dict(title='晚風', artist='林林')
    row = dict(trackId=123, trackTitle='晚風', artists=[dict(artistName='林林')],
        album=dict(albumTitle='春天'), lyrics=heading)
    assert published_candidate([row], song, '')[0] is None
    body = ['晚風', '我说：继续往前走', '晚風']
    assert published_candidate([row, dict(row, lyrics='\n'.join(body))], song, '')[0] == body
    assert published_lyric_lines('晚風', '晚風', ['林林'])[0] == ['晚風']
    packet = dict(status='ok', results=[dict(id=123, trackName='晚風', artistName='林林',
        albumName='春天', duration=300, instrumental=False, plainLyrics=heading)])
    assert lyric_candidate(packet, song, None, 300, allow_recording_mismatch=True)[0] is None


@pytest.mark.parametrize('title,artist,heading', [
    ('晚風', '林林', '晚風 - 林林'),
    ('晚風', '林林', '林林 — 晚風'),
    ('Evening Breeze', 'Singer', 'Singer - Evening Breeze 晚風'),
    ('Evening Breeze', 'Singer', 'Evening Breeze 晚風 – Singer')])
def test_published_lyrics_strip_only_leading_metadata_identity(title, artist, heading):
    from karaoke.lyric_lookup import published_lyric_lines
    body = [title, 'We keep singing through the night', heading, title]
    assert published_lyric_lines('\n'.join([heading, *body]), title, [artist]) == (body, [1])
    # A hyphen alone is not a header, nor is a different artist or title.
    for line in ('Another song - '+artist, title+' - Someone Else', 'You - and me'):
        assert published_lyric_lines('\n'.join([line, *body]), title, [artist]) == ([line, *body], [])


@pytest.mark.parametrize('credit', ['音频助理', '音頻工程師', '录音助理', 'Audio Engineer', 'Assistant Engineer',
    '配唱制作人', '人聲錄音師', '童声', '合唱', '和音', '伴唱编配', '主唱指导',
    '編曲 Arrangement', '制作人 Producer', '和声编写 Backing Vocal Arrangement',
    '配唱录音师 Recording Engineer (Vocals)', '母带后期处理工程师 Mastering Engineer', 'Music Production 音乐制作', '音乐制作 Music Production',
    '低音吉他', '器樂', 'Instrumental', '音乐制作助理', '录音工作室', '混音工作室', '母带后期制作人', '编曲协力'])
def test_translated_production_footer_cannot_change_the_lyric_language(credit):
    from karaoke.lyric_lookup import published_candidate
    body = ['Stars above the quiet shore', 'I will sing once more']
    row = dict(trackId=123, trackTitle='Evening', artists=[dict(artistName='Singer')],
        album=dict(albumTitle='Evening'), lyrics='\n'.join([*body, credit+' : A Person', 'Another Person']))
    lines, report = published_candidate([row], dict(title='Evening', artist='Singer'), '')
    assert lines == body and report['language'] == 'English'
    assert report['excluded_credit_lines'] == [3, 4]


def test_bilingual_credits_do_not_consume_lyric_prose_or_change_the_body():
    from karaoke.lyric_lookup import published_lyric_lines
    body = ['Music keeps us moving: through the night', '鼓 beat tonight: 我们继续唱']
    text = '\n'.join(['编曲 Arrangement：A Person', *body, '制作人 Producer：Someone'])
    assert published_lyric_lines(text, 'Evening') == (body, [1, 4])


@pytest.mark.parametrize('placeholder', ['[00:00.00]纯音乐，请欣赏', '純音樂，請您欣賞',
    '此歌曲为没有填词的纯音乐，请您欣赏', '暂无歌词', 'No lyrics available', 'Instrumental'])
def test_published_lyrics_skip_placeholder_and_try_the_next_release(placeholder):
    from karaoke.lyric_lookup import published_candidate, published_lyric_lines
    row = dict(trackId=123, trackTitle='Evening', artists=[dict(artistName='Singer')],
               album=dict(albumTitle='Evening'), lyrics=placeholder)
    song = dict(title='Evening', artist='Singer')
    assert published_candidate([row], song, '')[0] is None
    body = ['Stars above the quiet shore', 'I will sing once more']
    lines, report = published_candidate([row, dict(row, trackId=124, lyrics='\n'.join(body))], song, '')
    assert lines == body and report['url'].endswith('/124')
    # These words may occur in real lyrics; only a whole placeholder is rejected.
    assert published_lyric_lines(placeholder+'\n'+body[0], 'Evening')[0]


def test_published_lyric_html_entities_keep_the_source_words():
    from karaoke.lyric_lookup import published_lyric_lines
    assert published_lyric_lines('You&nbsp;&amp;&nbsp;me\nWe&#39;ll sing', 'Evening')[0] == [
        'You\u00a0&\u00a0me', "We'll sing"]


@pytest.mark.parametrize('status', [None, 401, 403, 429, 503])
def test_one_bad_lyric_candidate_does_not_discard_other_releases(status):
    import httpx
    from karaoke.lyric_lookup import published_candidate
    row = dict(trackTitle='Evening', artists=[dict(artistName='Singer')], album=dict(albumTitle='Album'))
    rows = [None, dict(row, trackId=1), dict(row, trackId=2)]
    calls = []
    def read(item):
        calls.append(item['trackId'])
        if item['trackId'] == 1:
            if status:
                response = httpx.Response(status, request=httpx.Request('GET','https://example.com/lyrics'))
                response.raise_for_status()
            raise TimeoutError('One release timed out')
        return 'Morning sunshine [again]\nEverybody sings along'
    if status in (401,403,429):
        with pytest.raises(httpx.HTTPStatusError):
            published_candidate(rows, dict(title='Evening',artist='Singer'), '', read_lyrics=read)
        assert calls == [1]
    else:
        lines, report = published_candidate(rows, dict(title='Evening',artist='Singer'), '', read_lyrics=read)
        assert lines == ['Morning sunshine [again]','Everybody sings along'] and calls == [1,2]
        assert report['rejected_candidates'][0]['track_id'] == '1'


def test_published_lyrics_parse_lrc_repetitions_without_reusing_their_timestamps():
    from karaoke.lyric_lookup import published_lyric_lines
    text = '[ar:Singer]\n[Chorus]\n[00:01.00][00:03.00]First [again]\n[00:02.00]Middle\n[00:04.00]Last'
    lines, excluded = published_lyric_lines(text, 'Evening')
    assert lines == ['First [again]', 'Middle', 'First [again]', 'Last']
    assert excluded == [1,2]
    # Do not guess the positions of missing timestamped repetitions.
    with pytest.raises(ValueError, match='fully timed'):
        published_lyric_lines('[00:01][00:03]First\nUnplaced verse', 'Evening')


@pytest.mark.parametrize('provider', ['line_music','lrclib'])
def test_lyric_result_pages_keep_valid_rows_beyond_expected_page_size(tmp_path, monkeypatch, provider):
    from karaoke import lyric_lookup
    from karaoke.stage_cache import StageCache
    song = dict(title='Evening', artist='Singer')
    metadata = tmp_path/'song.json'; metadata.write_text(json.dumps(song))
    body = 'Morning sunshine finds the door\nEverybody sings once more'
    if provider == 'line_music':
        row = dict(trackId=1,trackTitle='Evening',artists=[dict(artistName='Singer')],
                   album=dict(albumTitle='Album'),lyrics=body)
        response = dict(response=dict(result=dict(tracks=[None]*35+[row])))
        monkeypatch.setattr(lyric_lookup,'read_document',lambda *a,**k:response)
        lines, report = lyric_lookup.fetch_published_lyrics(song,'',StageCache(tmp_path))
        assert lines == body.splitlines() and report['status'] == 'candidate'
    else:
        response = [None]+[dict(id=i,trackName='Evening',artistName='Singer',plainLyrics=body) for i in range(105)]
        monkeypatch.setattr(lyric_lookup,'read_document',lambda *a,**k:response)
        result = lyric_lookup.fetch_lyrics(song,metadata,StageCache(tmp_path))
        assert result['status'] == 'ok' and len(result['results']) == 105


@pytest.mark.parametrize('change', [dict(trackTitle='別的歌'), dict(artists=[dict(artistName='別人')]),
    dict(album=dict(albumTitle='Live')), dict(album=dict(albumTitle='Remix')), dict(artists=None),
    dict(lyrics=None), dict(lyrics='歌名：晚風'), dict(lyrics='專輯：春天\n歌曲：晚風\n作詞：林林'),
    dict(trackId='../other')])
def test_published_lyrics_reject_wrong_credits_versions_and_empty_text(change):
    from karaoke.lyric_lookup import published_candidate
    row = dict(trackId=123, trackTitle='晚風', artists=[dict(artistName='林林')],
               album=dict(albumTitle='春天'), lyrics='微風裡慢慢歌唱')
    song = dict(title='晚風', artist='林林')
    assert published_candidate([row], song, 'Official Music Video')[0]
    assert published_candidate([dict(row, **change)], song, 'Official Music Video')[0] is None


@pytest.mark.parametrize('title,artist', [
    ('晚風 Evening Wind', '林林 Lin Lin'), ('Evening Wind 晚風', 'Lin Lin 林林'),
    ('晚風(Evening Wind)', '林林 Lin Lin')])
def test_published_lyrics_match_both_languages_of_bilingual_credits(title, artist):
    from karaoke.lyric_lookup import published_candidate
    row = dict(trackId=123, trackTitle=title, artists=[dict(artistName=artist)],
               album=dict(albumTitle='春天'), lyrics='微風裡慢慢歌唱')
    for title, artist in [('Evening Wind', 'Lin Lin'), ('晚風', '林林')]:
        lines, report = published_candidate([row], dict(title=title, artist=artist), 'Official Music Video')
        assert lines == [row['lyrics']] and report['url'].endswith('/123')
    for change in (dict(trackTitle='晚風 Live'), dict(artists=[dict(artistName='林林 Remix')])):
        assert published_candidate([dict(row, **change)], dict(title='Evening Wind', artist='Lin Lin'), '')[0] is None


@pytest.mark.parametrize('problem', [None, 'artist', 'text', 'short', 'repetitive', 'version'])
def test_alternate_titles_require_independent_lyric_body_support(problem):
    from karaoke.lyric_lookup import published_candidate
    body = ['微風吹過山谷看見遠方河流我們牽手走過漫長季節',
            '星光落在窗邊聽見朋友歌聲今夜等待清晨明亮天空']
    reference = ['啦啦啦啦啦', *body, '啦啦啦啦啦']
    row = dict(trackId=123, trackTitle='遠方之歌', artists=[dict(artistName='Singer')],
               album={}, lyrics='\n'.join(body))
    song = dict(title='Distant song', artist='Singer')
    if problem == 'artist':row['artists'] = [dict(artistName='Other singer')]
    if problem == 'text':row['lyrics'] = '另一首完全不同的歌曲' * 10
    if problem == 'short':reference = body[:1]
    if problem == 'repetitive':reference = ['啦' * 100]
    if problem == 'version':row['album'] = dict(albumTitle='Remix')
    assert published_candidate([row], song, '')[0] is None
    lines, report = published_candidate([row], song, '', reference_lines=reference)
    assert bool(lines) == (problem is None)
    if problem is None:
        assert ''.join(lines) == ''.join(body) and report['title_match'] == 'corroborated_lyric_text'
        assert report['identity_support']['matched_units'] == sum(map(len, body))
        exact = dict(row, trackId=124, trackTitle=song['title'])
        assert published_candidate([row, exact], song, '', reference_lines=reference)[1]['url'].endswith('/124')


@pytest.mark.parametrize('unavailable', [False, True])
def test_name_mismatch_reuses_source_results_and_keeps_reference_when_recovery_fails(tmp_path, monkeypatch, unavailable):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    body = ['微風吹過山谷看見遠方河流我們牽手走過漫長季節',
            '星光落在窗邊聽見朋友歌聲今夜等待清晨明亮天空']
    alternate = [body[0], body[1]+'啊']
    calls = []
    monkeypatch.setattr(module.lookup, 'fetch_published_lyrics', lambda *a:
        (body, dict(provider='line_music',language='Chinese')))
    monkeypatch.setattr(module.lookup, 'fetch_lyrics', lambda *a:dict(status='unavailable'))
    def additional(provider, *args, reference_lines=()):
        calls.append((provider, reference_lines))
        if provider != 'netease':return None, dict(provider=provider,status='unavailable')
        if not reference_lines:
            return None, dict(provider=provider,status='no_matching_published_text',
                              identity_candidates=[dict(title='遠方之歌',track_id='123')])
        assert reference_lines == body
        if unavailable:raise OSError('provider unavailable')
        return alternate, dict(provider=provider,language='Chinese',title_match='corroborated_lyric_text')
    monkeypatch.setattr(module, 'fetch_additional_lyrics', additional)
    lines, report = module.fetch_lyric_consensus(dict(title='Distant song',artist='Singer'),{},60,StageCache(tmp_path))
    assert lines == body
    assert report['candidate_texts'] == ([body] if unavailable else [body,alternate])
    assert calls.count(('netease',body)) == 1
    if unavailable:
        failed = next(r for r in report['crosscheck']['unavailable_sources'] if r['provider']=='netease')
        assert failed['error'] == 'OSError' and failed['status'] == 'unavailable'


def test_published_text_prefers_same_performance_before_studio_fallback():
    from karaoke.lyric_lookup import published_candidate
    song = dict(title='Evening', artist='Singer')
    row = dict(trackId=123, trackTitle='Evening', artists=[dict(artistName='Singer')],
               album=dict(albumTitle='Evening'), lyrics='Stars above the quiet shore')
    source_title = 'Evening (Live at Concert Hall)'
    assert published_candidate([row], song, source_title)[0] is None
    lines, report = published_candidate([row], song, source_title, allow_recording_mismatch=True)
    assert lines and set(report['recording_mismatches']) == {'version', 'live_performance'}
    for version in ('Remix', 'Karaoke', 'Instrumental'):
        assert published_candidate([dict(row, album=dict(albumTitle=version))], song, source_title,
            allow_recording_mismatch=True)[0] is None
    _, report = published_candidate([row, dict(row, trackId=124, album=dict(albumTitle=source_title))], song,
                                   source_title, allow_recording_mismatch=True)
    assert report['url'].endswith('/124') and not report['recording_mismatches']


def test_published_lyrics_match_joined_artist_scripts_without_matching_collaborators():
    from karaoke.lyric_lookup import published_candidate
    song = dict(title='晚風', artist='林林 Lin Lin')
    row = dict(trackId=123, trackTitle='晚風', album=dict(albumTitle='春天'), lyrics='微風裡慢慢歌唱')
    for name in ('ALPHA林林', 'ALPHA林林 Lin Lin'):
        assert published_candidate([dict(row, artists=[dict(artistName=name)])], song, '')[0]
    for name in ('ALPHA林楠', 'ALPHA & 林林', 'LIVE林林', '林林 Remix'):
        assert published_candidate([dict(row, artists=[dict(artistName=name)])], song, '')[0] is None


@pytest.mark.parametrize('credit,artists,accepted', [
    (None, ['林林', '苏雨'], False),
    ('林林 蘇雨', ['苏雨', '林林'], True),
    ('蘇雨 & 林林', ['林林', '苏雨'], True),
    ('蘇雨 feat. 林林', ['林林', '苏雨'], True),
    ('林林 · 蘇雨', ['苏雨 / 林林'], True),
    ('林林 蘇雨', ['林林'], False),
    ('林林 蘇雨', ['苏雨', '林林歌迷站'], False),
    ('林林 蘇雨', ['苏雨', '林林', '另一歌手'], False),
    ('North River feat. Evening Star', ['Evening Star', 'North River'], True),
    ('North River & Evening Star', ['North River'], False),
    ('Anna & Ann', ['Ann', 'Anna'], True),
])
def test_published_lyrics_match_complete_collaboration_credits(credit, artists, accepted):
    from karaoke.lyric_lookup import published_candidate, lyric_candidate
    body = 'Stars above the quiet shore'
    song = dict(title='Evening', artist=credit)
    row = dict(trackId=123, trackTitle='Evening', artists=[dict(artistName=a) for a in artists],
               album={}, lyrics=body)
    assert bool(published_candidate([row], song, '')[0]) is accepted
    flat = dict(id=123, trackName='Evening', artistName=' / '.join(artists),
                albumName='', duration=30, instrumental=False, plainLyrics=body)
    assert bool(lyric_candidate(dict(status='ok', results=[flat]), song, None, 30)[0]) is accepted


def test_netease_reads_duet_lyrics_after_matching_reordered_credit(tmp_path, monkeypatch):
    from karaoke import lyric_consensus as module
    from karaoke.stage_cache import StageCache
    calls = []
    def read(url, **kwargs):
        calls.append(url)
        if 'cloudsearch' in url:
            return dict(code=200, result=dict(songs=[
                dict(id=1, name='晚風', ar=[dict(name='林林')], al=dict(name='春天')),
                dict(id=2, name='晚風', ar=[dict(name='苏雨'), dict(name='林林')], al=dict(name='春天'))]))
        assert 'id=2&' in url
        return dict(lrc=dict(lyric='[00:01.00] 微風裡慢慢歌唱\n[00:05.00] 星光下靜靜傾聽'))
    monkeypatch.setattr(module, 'read_document', read)
    for _ in range(2):
        lines, report = module.fetch_additional_lyrics('netease', dict(title='晚風', artist='林林 蘇雨'),
                                                      {}, StageCache(tmp_path))
        assert lines == ['微風裡慢慢歌唱', '星光下靜靜傾聽']
        assert report['url'].endswith('id=2')
    assert len(calls) == 2


@pytest.mark.parametrize('first_hit', ['yes', 'wrong_identity', 'short', 'header'])
def test_published_lyric_search_uses_a_bounded_title_query_only_after_no_match(tmp_path, monkeypatch, first_hit):
    from urllib.parse import parse_qs, urlparse
    from karaoke import lyric_lookup
    from karaoke.stage_cache import StageCache
    calls = []
    row = dict(trackId=123, trackTitle='晚風', artists=[dict(artistName='ALPHA林林')],
               album=dict(albumTitle='春天'), lyrics='微風裡慢慢歌唱')
    def read(url, **kwargs):
        query = parse_qs(urlparse(url).query)['query'][0]
        calls.append(query)
        selected = row if first_hit == 'yes' or query == '晚風' else (
            dict(row, lyrics='晚風吹' if first_hit == 'short' else '林林 - 晚風')
            if first_hit in ('short', 'header') else dict(row, trackTitle='無關歌曲'))
        return dict(response=dict(result=dict(tracks=[selected])))
    monkeypatch.setattr(lyric_lookup, 'read_document', read)
    song = dict(title='晚風', artist='林林')
    for _ in range(2):
        lines, report = lyric_lookup.fetch_published_lyrics(song, '', StageCache(tmp_path), 60)
        assert lines == [row['lyrics']] and report['url'].endswith('/123')
        assert bool(report['rejected_candidates']) == (first_hit in ('short', 'header'))
    assert calls == (['晚風 林林'] if first_hit == 'yes' else ['晚風 林林', '晚風'])


@pytest.mark.parametrize('provider', ['line_music', 'lrclib', 'netease'])
@pytest.mark.parametrize('bad_text,reason', [('林林 - 晚風', 'empty_lyric_body'), ('晚風吹', 'lyrics_too_short')])
def test_short_lyric_release_does_not_hide_later_complete_release(tmp_path, monkeypatch, provider, bad_text, reason):
    from karaoke import lyric_lookup, lyric_consensus
    from karaoke.stage_cache import StageCache
    song = dict(title='晚風', artist='林林')
    body = '微風裡慢慢歌唱\n星光下靜靜傾聽\n' * 3
    row = dict(trackId=1, trackTitle='晚風', artists=[dict(artistName='林林')],
               album=dict(albumTitle='春天'), lyrics=bad_text)
    if provider == 'line_music':
        lines, report = lyric_lookup.published_candidate([row, dict(row, trackId=2, lyrics=body)],
            song, '', duration=300)
        assert report['url'].endswith('/2')
    elif provider == 'lrclib':
        row = dict(id=1, trackName='晚風', artistName='林林', duration=300, instrumental=False,
                   plainLyrics=row['lyrics'])
        lines, report = lyric_lookup.lyric_candidate(dict(status='ok', results=[row, dict(row, id=2, plainLyrics=body)]),
            song, None, 300)
        assert report['id'] == 2
    else:
        calls = []
        def read(url, **kwargs):
            calls.append(url)
            if 'cloudsearch' in url:
                return dict(code=200, result=dict(songs=[dict(id=i, name='晚風', ar=[dict(name='林林')],
                    al=dict(name='春天')) for i in (1, 2)]))
            return dict(lrc=dict(lyric=row['lyrics'] if 'id=1&' in url else body))
        monkeypatch.setattr(lyric_consensus, 'read_document', read)
        lines, report = lyric_consensus.fetch_additional_lyrics(provider, song, {}, StageCache(tmp_path), 300)
        assert len(calls) == 3 and report['url'].endswith('id=2')
    assert lines == body.splitlines()
    assert len(report['rejected_candidates']) == 1
    assert report['rejected_candidates'][0]['reason'] == reason
    assert report['rejected_candidates'][0]['text_sha256'] == hashlib.sha256(bad_text.encode()).hexdigest()
    if reason == 'empty_lyric_body':
        assert report['rejected_candidates'][0]['excluded_credit_lines'] == [1]


def test_published_lyric_search_does_not_issue_another_query_after_rate_limit(tmp_path, monkeypatch):
    import httpx
    from karaoke import lyric_lookup
    from karaoke.stage_cache import StageCache
    calls = []
    def limited(url, **kwargs):
        calls.append(url)
        response = httpx.Response(429, request=httpx.Request('GET', url))
        response.raise_for_status()
    monkeypatch.setattr(lyric_lookup, 'read_document', limited)
    folder = tmp_path/'song'
    folder.mkdir()
    lines, report = lyric_lookup.fetch_published_lyrics(dict(title='晚風',artist='林林'), '', StageCache(folder))
    assert lines is None and report['http_status'] == 429 and len(calls) == 1


@pytest.mark.parametrize('empty_anchor_file', [False, True])
def test_direct_lyrics_reach_acoustic_aligner_without_asr_anchors(tmp_path, monkeypatch, empty_anchor_file):
    import soundfile as sf
    from karaoke import align_lyrics_whisper as aligner
    class ReachedAlignment(Exception):
        pass
    def align(audio, text, **options):
        assert len(audio) == 32000 and text == 'Stars above the quiet shore'
        assert options['language'] == 'en' and options['original_split']
        raise ReachedAlignment
    model = SimpleNamespace(_buffers={'alignment_heads': None}, to=lambda device: None,
                            register_buffer=lambda *a, **kw: None, align=align)
    monkeypatch.setattr(aligner.stable_whisper, 'load_model', lambda *a, **kw: model)
    monkeypatch.setattr(aligner, 'sha', lambda path:'test-model-hash')
    sf.write(tmp_path/'audio.wav', np.ones(32000)*.1, 16000)
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='English', lines=['Stars above the quiet shore'])))
    job = dict(audio=str(tmp_path/'audio.wav'), lyrics=str(tmp_path/'lyrics.json'), output=str(tmp_path/'aligned'))
    if empty_anchor_file:
        (tmp_path/'anchors.json').write_text('[]')
        job['anchors'] = str(tmp_path/'anchors.json')
    (tmp_path/'jobs.json').write_text(json.dumps([job]))
    monkeypatch.setattr(sys, 'argv', ['align', '--jobs', str(tmp_path/'jobs.json'), '--device', 'cpu'])
    with pytest.raises(ReachedAlignment):
        aligner.main()


@pytest.mark.parametrize('failure', [None, 'changed_text', 'collapsed', 'outside', 'crossing', 'raised', 'empty_attention', 'oom'])
def test_direct_alignment_repairs_only_bad_lines_with_acoustic_results(failure):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['First phrase', 'Middle phrase', 'Last phrase']
    def segment(text, start):
        words = [dict(word=w, start=start+i*.4, end=start+i*.4+.3) for i,w in enumerate(text.split())]
        return dict(text=text, start=start, end=words[-1]['end'], words=words)
    segments = [segment(text, start) for text,start in zip(lines, [1., 4., 8.])]
    for word in segments[1]['words']:
        word.update(start=4., end=4.)
    original = dict(context=[0,3], range=[0,3], source_range=[0.,300.], result=dict(segments=segments))
    saved = deepcopy(original)
    calls = []
    def align(begin, end, text, **options):
        calls.append((begin,end,text))
        assert end-begin < 10 and options['dynamic_heads']
        if failure == 'raised':
            raise ValueError('No usable acoustic alignment')
        if failure in ('empty_attention', 'oom'):
            raise RuntimeError('selected index k out of range' if failure == 'empty_attention' else 'CUDA out of memory')
        candidate = [segment(line, start-begin) for line,start in zip(lines, [1.1,4.,8.1])]
        word = candidate[1]['words'][0]
        if failure == 'changed_text': word['word'] = 'Unrelated'
        if failure == 'collapsed': word['end'] = word['start']
        if failure == 'outside': word['end'] = end+10
        if failure == 'crossing': word['start'] = 0
        return dict(segments=candidate)
    if failure == 'oom':
        with pytest.raises(RuntimeError, match='CUDA out of memory'):
            recover_unresolved_lines(original,lines,300.,'English',align)
        return
    result = recover_unresolved_lines(original,lines,300.,'English',align)
    assert original == saved and len(calls) == 1
    assert result['line_recoveries'][0]['accepted'] == (failure is None)
    actual = result['result']['segments']
    assert actual[0] == segments[0] and actual[2] == segments[2]
    if failure is None:
        assert actual[1] == segment(lines[1],4.)
    else:
        assert actual == segments
    if failure == 'empty_attention':
        assert 'selected index k out of range' in result['line_recoveries'][0]['error']


def test_direct_alignment_has_a_shared_local_retry_budget():
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['first', 'second', 'third', 'fourth', 'fifth']
    segments = [dict(text=text, words=[dict(word=text, start=float(i*3+1),
        end=float(i*3+1 if i%2 else i*3+2))]) for i,text in enumerate(lines)]
    block = dict(context=[0,5], range=[0,5], source_range=[0.,20.], result=dict(segments=segments))
    calls = []
    def align(begin,end,text,**options):
        calls.append(text)
        return dict(segments=[dict(text=line, words=[dict(word=line, start=i*3+1-begin,
            end=i*3+2-begin)]) for i,line in enumerate(text.splitlines())])
    result = recover_unresolved_lines(block,lines,20.,'English',align,max_attempts=1)
    assert calls == ['\n'.join(lines[:3])]
    assert result['line_recoveries'][0]['accepted']
    assert result['line_recoveries'][1]['reason'] == 'local_retry_budget_exhausted'
    assert result['result']['segments'][3:] == segments[3:]


@pytest.mark.parametrize('failure', [None, 'changed_text', 'outside', 'collapsed', 'missing_neighbour',
                                   'acoustic', 'acoustic_missing', 'acoustic_outside', 'acoustic_oom'])
def test_isolated_gap_recovery_keeps_both_neighbours_and_prior_diagnostics(failure):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['Before', 'Quiet repeated phrase', 'After']
    segments = [dict(text=t, words=[dict(word=t, start=a, end=b)])
                for t,a,b in zip(lines, [1.,4.,8.], [2.,4.,9.])]
    if failure == 'missing_neighbour':
        segments[0]['words']=segments[2]['words']=[]
    block = dict(context=[0,3], range=[0,3], source_range=[0.,10.], result=dict(segments=segments),
                 line_recoveries=[dict(accepted=True,lines=[2,2],isolated=False)],
                 asr_localizations=[dict(contradicted_lines=[2],accepted_lines=[])])
    original = deepcopy(block)
    calls=[]
    def align(begin,end,text,**options):
        calls.append((begin,end,text))
        assert (begin,end,text) == (2.,8.,lines[1])
        assert not options['detect_language']
        words=[dict(word=t,start=s,end=s+.5) for t,s in zip(text.split(),[1.,2.,3.])]
        if failure == 'changed_text':words[0]['word']='Different'
        if failure == 'outside':words[-1]['end']=7.
        if failure == 'collapsed':words[-1]['end']=words[-1]['start']
        if failure and failure.startswith('acoustic'):
            words[0]['end']=words[0]['start']
        return dict(segments=[dict(text=text,words=words)])
    def localize(begin,end,text):
        assert (begin,end,text) == (2.,8.,lines[1])
        if failure == 'acoustic_oom':raise RuntimeError('CUDA out of memory')
        words=[dict(first=a,stop=b,text=text[a:b],start=t,end=t+.5)
               for a,b,t in ((0,5,3.),(6,14,4.),(15,21,5.))]
        if failure == 'acoustic_missing':words.pop()
        if failure == 'acoustic_outside':words[-1]['end']=9.
        return dict(model='test_acoustic_decoder', views=[[],words])
    if failure == 'acoustic_oom':
        with pytest.raises(RuntimeError, match='CUDA out of memory'):
            recover_unresolved_lines(block,lines,10.,'English',align,isolated=True,localize=localize)
        return
    result=recover_unresolved_lines(block,lines,10.,'English',align,isolated=True,
                                   localize=localize if failure and failure.startswith('acoustic') else None)
    assert block == original
    assert result['line_recoveries'][0] == block['line_recoveries'][0]
    assert result['line_recoveries'][-1]['accepted'] == (failure in (None,'acoustic'))
    assert len(calls) == (0 if failure == 'missing_neighbour' else 1)
    assert result['result']['segments'][0] == segments[0]
    assert result['result']['segments'][2] == segments[2]
    from karaoke.pipeline import alignment_recovery_summary
    assert alignment_recovery_summary([result])['unresolved_localization_lines'] == (
        [] if failure in (None,'acoustic') else [2])
    if failure in (None,'acoustic'):
        assert [w['start'] for w in result['result']['segments'][1]['words']] == [3.,4.,5.]
        if failure == 'acoustic':
            proof=result['line_recoveries'][-1]
            assert proof['acoustic_selected_view']==1
            assert proof['result']['segments'][0]['words'][0]['start']==proof['result']['segments'][0]['words'][0]['end']
    else:
        assert result['result'] == block['result']


@pytest.mark.parametrize('edge', ['opening', 'ending'])
def test_isolated_recovery_uses_recording_edge_only_as_an_audio_search_bound(edge):
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['First phrase', 'Last phrase']
    segments=[dict(text=t,words=[dict(word=t,start=a,end=b)])
              for t,a,b in zip(lines,[1.,8.],[3.,10.])]
    bad = 0 if edge=='opening' else 1
    segments[bad]['words'][0]['end']=segments[bad]['words'][0]['start']
    block=dict(context=[0,2],range=[0,2],source_range=[10.,30.],result=dict(segments=segments))
    calls=[]
    def align(begin,end,text,**options):
        calls.append((begin,end,text))
        return dict(segments=[dict(text=text,words=[dict(word=w,start=i+1.,end=i+1.5)
                                                  for i,w in enumerate(text.split())])])
    result=recover_unresolved_lines(block,lines,40.,'English',align,isolated=True)
    assert calls==[(10.,18.,lines[0])] if edge=='opening' else calls==[(13.,30.,lines[1])]
    assert result['line_recoveries'][-1]['accepted']
    assert result['result']['segments'][1-bad]==segments[1-bad]
    measured_start=result['result']['segments'][bad]['words'][0]['start']+10.
    assert measured_start==calls[0][0]+1.  # The model's onset, never the crop edge.


@pytest.mark.parametrize('edge', ['opening', 'ending'])
@pytest.mark.parametrize('fault', [None, 'interior_unlocalized', 'changed_text', 'ambiguous_repeat', 'outside_neighbour', 'missing_anchor', 'no_anchors'])
def test_isolated_recovery_reuses_localization_across_long_instrumentals(edge, fault):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['First phrase', 'Middle phrase', 'Another phrase', 'Last phrase']
    first, stop = (0, 3) if edge == 'opening' else (1, 4)
    segments = [dict(text=t, words=[]) for t in lines]
    neighbour = 3 if edge == 'opening' else 0
    segments[neighbour]['words'] = [dict(word=lines[neighbour], start=70. if edge=='opening' else 1.,
                                        end=71. if edge=='opening' else 2.)]
    anchors = [dict(status='candidate', text=lines[i], start=20.+3*j, end=21.+3*j)
               for j,i in enumerate(range(first, stop))]
    if fault == 'changed_text':anchors[0]['text'] = 'Another phrase'
    if fault == 'ambiguous_repeat':anchors[0]['status'] = 'ambiguous_repeat'
    if fault == 'outside_neighbour':anchors[0].update(start=5., end=6.)
    if fault == 'missing_anchor':anchors.pop()
    if fault == 'interior_unlocalized':anchors[1].update(status='unresolved_text',start=None,end=None)
    localization = dict(context=[first, stop], anchors=anchors)
    if fault == 'no_anchors':localization.pop('anchors')
    block = dict(context=[0,4], range=[0,4], source_range=[10.,160.], result=dict(segments=segments),
                 asr_localizations=[localization])
    original = deepcopy(block)
    calls = []
    def align(begin, end, text, **options):
        calls.append((begin, end, text))
        return dict(segments=[dict(text=line, words=[dict(word=w, start=1.+i*2+j, end=1.4+i*2+j)
             for j,w in enumerate(line.split())]) for i,line in enumerate(text.splitlines())])
    result = recover_unresolved_lines(block, lines, 170., 'English', align, isolated=True)
    assert block == original
    assert result['result']['segments'][neighbour] == segments[neighbour]
    proof = result['line_recoveries'][-1]
    if fault in (None, 'interior_unlocalized'):
        assert calls == [(19.4,27.6,'\n'.join(lines[first:stop]))]
        assert proof['accepted']
        assert proof['search_anchors'] == anchors
        assert result['result']['segments'][first]['words'][0]['start']+10. == pytest.approx(20.4)
    else:
        assert not calls and proof['reason'] == 'requires_coarse_localization'
        assert result['result'] == block['result']


@pytest.mark.parametrize('fault', [None, 'first_window_fails', 'budget', 'unlocated_split'])
@pytest.mark.parametrize('ending', [False, True])
def test_isolated_recovery_splits_long_runs_at_measured_phrase_boundaries(fault, ending):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['Before', 'First chorus', 'Second chorus', 'First verse', 'Second verse', 'After']
    starts = [5., 20., 24., 75., 79., 95.]
    segments = [dict(text=t, words=[dict(word=t, start=s, end=s+1.)] if i in (0,5) else [])
                for i,(t,s) in enumerate(zip(lines, starts))]
    anchors = [dict(text=lines[i], status='candidate', start=starts[i], end=starts[i]+2.) for i in range(1,5)]
    if fault == 'unlocated_split':
        for anchor in anchors[1:3]:
            anchor.update(status='unresolved_text', start=None, end=None)
    if ending:
        lines, segments = lines[:-1], segments[:-1]
    block = dict(context=[0,len(lines)], range=[0,len(lines)], source_range=[0.,100.], result=dict(segments=segments),
                 asr_localizations=[dict(context=[1,5], anchors=anchors)])
    original, calls = deepcopy(block), []
    def align(begin, end, text, **options):
        calls.append((begin,end,text.splitlines()))
        if fault == 'first_window_fails' and len(calls)==1:
            return None
        return dict(segments=[dict(text=line, words=[dict(word=word,
            start=starts[lines.index(line)]+.2+j*.5-begin,
            end=starts[lines.index(line)]+.5+j*.5-begin) for j,word in enumerate(line.split())])
            for line in text.splitlines()])
    result = recover_unresolved_lines(block, lines, 100., 'English', align, isolated=True,
                                      max_attempts=1 if fault=='budget' else 8)
    assert block == original
    assert result['result']['segments'][0] == segments[0]
    if not ending:
        assert result['result']['segments'][5] == segments[5]
    if fault == 'unlocated_split':
        assert not calls and result['result'] == block['result']
        return
    assert calls[0] == (19.4,26.6,lines[1:3])
    assert len(calls) == (1 if fault=='budget' else 2)
    if fault != 'budget':
        assert calls[1] == (74.4,81.6,lines[3:5])
    for i in range(1,5):
        words = result['result']['segments'][i]['words']
        accepted = not (fault=='first_window_fails' and i<3 or fault=='budget' and i>=3)
        assert bool(words) == accepted
        if accepted:
            assert words[0]['start'] == pytest.approx(starts[i]+.2)


@pytest.mark.parametrize('far_neighbour', [False, True])
def test_direct_alignment_defers_unanchored_or_oversized_repairs(far_neighbour):
    from karaoke.align_lyrics_whisper import recover_unresolved_lines
    lines = ['First phrase', 'Last phrase']
    segments = [dict(text=text, words=[dict(word=text,start=0.,end=0.)]) for text in lines]
    if far_neighbour:
        segments[1]['words'][0].update(start=200.,end=201.)
    original = dict(context=[0,2],range=[0,2],source_range=[0.,300.],result=dict(segments=segments))
    def align(*args,**kwargs):
        raise AssertionError('Unsafe whole-song retry')
    result = recover_unresolved_lines(original,lines,300.,'English',align)
    assert result['result'] == original['result']
    assert result['line_recoveries'][0]['reason'] == (
        'requires_coarse_localization' if far_neighbour else 'no_measured_neighbour')


@pytest.mark.parametrize('failure', [None, 'wrong_words', 'outside_window', 'outside_anchor', 'no_speech', 'boundary_overlap', 'only_overlap', 'collapsed', 'empty_attention', 'oom'])
def test_local_recognition_locates_missing_line_without_replacing_lyrics_or_neighbours(failure):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie'], [1., 4., 8.]
    if failure == 'collapsed':lines[1] = 'bravo delta echo foxtrot'
    segments = [dict(text=text, start=t, end=t+.8, words=[dict(word=text,start=t,end=t+.8)])
                for text,t in zip(lines,starts)]
    if failure != 'only_overlap':
        segments[1]['words'][0]['end'] = 4.
    if failure in ('boundary_overlap', 'only_overlap'):
        segments[2]['words'][0]['start'] = 3.8
    block = dict(range=[0,3],context=[0,3],source_range=[0.,12.],result=dict(segments=segments))
    original = deepcopy(block)
    def recognize(begin,end):
        assert end-begin <= 12
        return dict(segments=[dict(no_speech_prob=1 if failure=='no_speech' else 0,
            words=[dict(word=text,start=t-begin,end=t+.8-begin) for text,t in zip(lines,starts)])])
    attempts = []
    def align(begin,end,text,**kwargs):
        attempts.append((begin,end))
        if failure == 'oom':
            raise RuntimeError('CUDA out of memory')
        if failure == 'empty_attention' and len(attempts) == 1:
            raise RuntimeError('selected index k out of range')
        assert text.splitlines() == lines
        result = [dict(text=line,words=[dict(word=line,start=t-begin,end=t+.8-begin)])
                  for line,t in zip(lines,starts)]
        if failure == 'wrong_words':result[1]['words'][0]['word'] = 'invented'
        if failure == 'outside_window':result[1]['words'][0]['end'] = end-begin+10
        if failure == 'outside_anchor':result[1]['words'][0].update(start=7.-begin,end=7.8-begin)
        if failure == 'collapsed':result[1]['words'][0]['end'] = result[1]['words'][0]['start']+.06
        return dict(segments=result)
    if failure == 'oom':
        with pytest.raises(RuntimeError, match='CUDA out of memory'):
            recover_with_local_asr(block,lines,np.zeros(12*16000),'English',recognize,align)
        return
    result = recover_with_local_asr(block,lines,np.zeros(12*16000),'English',recognize,align)
    assert block == original
    assert result['result']['segments'][0] == segments[0]
    if failure not in ('boundary_overlap', 'only_overlap'):
        assert result['result']['segments'][2] == segments[2]
    assert result['asr_localizations'][0]['accepted_lines'] == (
        [2,3] if failure in ('boundary_overlap', 'only_overlap') else [2] if failure is None else [])
    if failure in (None, 'boundary_overlap', 'only_overlap', 'empty_attention'):
        assert result['result']['segments'][1]['words'] == [dict(word='bravo',start=4.,end=4.8)]
        assert result['result']['segments'][2]['words'] == [dict(word='charlie',start=8.,end=8.8)]
    else:
        assert result['result'] == block['result']
    if failure == 'empty_attention':
        assert 'selected index k out of range' in result['asr_localizations'][0]['attempts'][0]['error']
        assert result['asr_localizations'][-1]['accepted_lines'] == [2]
    good = dict(block,result=dict(segments=[dict(text=t,words=[dict(word=t,start=s,end=s+.8)])
        for t,s in zip(lines,starts)]))
    assert recover_with_local_asr(good,lines,np.zeros(12*16000),'English',
        lambda *a:pytest.fail('A complete alignment ran ASR'),align) is good


def test_missing_recognition_uses_original_audio_before_spending_alignment_retries():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie'], [1.,4.,8.]
    segments = [dict(text=t,start=s,end=s+.8,words=[dict(word=t,start=s,end=s+.8)])
                for t,s in zip(lines,starts)]
    segments[1]['words'][0]['end'] = 4.
    block = dict(range=[0,3],context=[0,3],source_range=[0.,12.],result=dict(segments=segments))
    calls=[]
    def original(begin,end):
        calls.append((begin,end))
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
                                         for t,s in zip(lines,starts)])])
    def align(begin,end,text,**options):
        assert options['detect_language']
        return dict(segments=[dict(text=t,words=[dict(word=t,start=s-begin,end=s+.8-begin)])
                              for t,s in zip(lines,starts)])
    result = recover_with_local_asr(block,lines,np.zeros(12*16000),'English',
        lambda *args:dict(segments=[]),align,recognize_original=original)
    assert len(calls)==1
    assert result['result']['segments'][1]['words']==[dict(word='bravo',start=4.,end=4.8)]
    assert result['asr_localizations'][0]['recognition_source']=='original'
    assert result['asr_localizations'][1]['recognition']==dict(segments=[])


@pytest.mark.parametrize('support', ['agree', 'disagree', 'unavailable', 'original', 'original_wrong', 'original_outside'])
def test_local_unrecognized_phrase_uses_acoustic_evidence_without_repeating_recognition(support):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie'], [1.,12.,25.]
    def segment(text, begin, collapsed=False):
        start=starts[lines.index(text)]-begin
        return dict(words=[dict(word=text,start=start,end=start+(0 if collapsed else .8))])
    original=[segment(t,0.,i==1) for i,t in enumerate(lines)]
    block=dict(range=[0,3],context=[0,3],source_range=[0.,30.],result=dict(segments=original))
    recognition=[]
    def recognize(begin,end):
        recognition.append((begin,end))
        return dict(segments=[segment(lines[i],begin) for i in (0,2)])
    def align(begin,end,text,**options):
        return dict(segments=[segment(t,begin,text== '\n'.join(lines) and t==lines[1])
                              for t in text.splitlines()])
    acoustic=[]
    def localize(begin,end,text):
        acoustic.append((begin,end))
        assert recognition[0]==(begin,end) and text=='\n'.join(lines)
        if support=='unavailable' or support.startswith('original'):raise ValueError('No pronunciation evidence')
        words,offset=[],0
        for t,s in zip(lines,starts):
            words.append(dict(text=t,first=offset,stop=offset+len(t),start=s,end=s+.8))
            offset+=len(t)+1
        shift=1 if support=='disagree' else .1
        return dict(views=[words,[dict(w,start=w['start']+shift,end=w['end']+shift) for w in words]])
    originals=[]
    def original_mix(begin,end,text,**options):
        originals.append((begin,end))
        assert recognition[-1][0]<=begin<end<=recognition[-1][1] and text=='\n'.join(lines)
        result=[segment(t,begin) for t in lines]
        if support=='original_wrong':result[1]['words'][0]['word']='unrelated'
        if support=='original_outside':result[1]['words'][0]['end']=end-begin+1
        return dict(segments=result)
    result=recover_with_local_asr(block,lines,np.zeros(30*16000),'English',recognize,align,
        localize=localize,max_alignment_attempts=8,align_original=original_mix if support.startswith('original') else None)
    assert acoustic or originals
    if originals:assert originals[0]==recognition[0]
    assert recognition.count(recognition[0])==1
    assert result['result']['segments'][0]==original[0] and result['result']['segments'][2]==original[2]
    if support in ('agree','original'):
        assert result['result']['segments'][1]['words']==segment('bravo',0.)['words']
    else:
        assert result['result']==block['result']


def test_narrowed_acoustic_search_finishes_before_unrelated_regions_spend_its_budget():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines,starts=['alpha','bravo','charlie','delta','omega'],[1.,12.,25.,45.,60.]
    def segment(text,begin,collapsed=False):
        start=starts[lines.index(text)]-begin
        return dict(words=[dict(word=text,start=start,end=start+(0 if collapsed else .8))])
    original=[segment(t,0.,i in (1,3)) for i,t in enumerate(lines)]
    block=dict(range=[0,5],context=[0,5],source_range=[0.,65.],result=dict(segments=original))
    def recognize(begin,end):
        return dict(segments=[segment(lines[i],begin)for i in (0,2,4) if begin<=starts[i]<end])
    calls=[]
    def align(begin,end,text,**options):
        calls.append(text)
        return dict(segments=[segment(t,begin,len(text.splitlines())>1 and t in ('bravo','delta'))
                              for t in text.splitlines()])
    def localize(begin,end,text):
        units,offset=[],0
        for t in text.splitlines():
            start=starts[lines.index(t)]
            units.append(dict(text=t,first=offset,stop=offset+len(t),start=start,end=start+.8))
            offset+=len(t)+1
        return dict(views=[units,units])
    result=recover_with_local_asr(block,lines,np.zeros(65*16000),'English',recognize,align,
        localize=localize,force_global=True,max_alignment_attempts=5)
    assert calls==['\n'.join(lines[:3]),'bravo']
    assert result['result']['segments'][1]['words']==segment('bravo',0.)['words']
    assert result['result']['segments'][3]==original[3]
    assert result['asr_localizations'][0]['accepted_lines']==[2]


def test_local_recognition_cannot_search_outside_its_observed_crop():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie', 'delta', 'omega'], [10.,30.,31.,32.,50.]
    segments = [dict(words=[dict(word=t,start=s,end=s+(0 if i==2 else .4))])
                for i,(t,s) in enumerate(zip(lines,starts))]
    block = dict(range=[0,5],context=[0,5],source_range=[0.,60.],result=dict(segments=segments))
    observed = []
    def recognize(begin,end):
        observed.append((begin,end))
        # Only the right neighbour is recognized; the left search boundary
        # must still be the crop start, not the start of the whole recording.
        return dict(segments=[dict(words=[dict(word='delta',start=32.-begin,end=32.4-begin)])])
    def align(begin,end,text,**options):
        assert observed[-1][0] <= begin < end <= observed[-1][1]
        return dict(segments=[dict(words=[dict(word=t,start=starts[lines.index(t)]-begin,
            end=starts[lines.index(t)]+.4-begin)]) for t in text.splitlines()])
    result = recover_with_local_asr(block,lines,np.zeros(60*16000),'English',recognize,align)
    assert len(observed) == 1
    assert result['result']['segments'][2]['words'][0]['start'] == 31.
    assert result['result']['segments'][0] == segments[0]
    assert result['result']['segments'][4] == segments[4]


def test_isolated_missing_lines_preserve_budget_for_global_localization():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, missing = ['line'+str(i) for i in range(30)], {3,9,15,21}
    starts = [i*3.+1 for i in range(30)]
    segments = [dict(words=[dict(word=t,start=s,end=s+(0 if i in missing else .8))])
                for i,(t,s) in enumerate(zip(lines,starts))]
    block = dict(range=[0,30],context=[0,30],source_range=[0.,100.],result=dict(segments=segments))
    recognition, alignment = [], []
    def recognize(begin,end):
        recognition.append((begin,end))
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
            for i,(t,s) in enumerate(zip(lines,starts)) if i not in missing and begin<=s and s+.8<=end])])
    def align(begin,end,text,**options):
        alignment.append((begin,end,text))
        return None
    result = recover_with_local_asr(block,lines,np.zeros(100*16000),'English',recognize,align)
    assert (0.,100.) in recognition and len(alignment) <= 24
    assert result['result'] == block['result']


@pytest.mark.parametrize('force_global', [False, True])
def test_missing_lyric_timing_takes_priority_over_collapsed_refinement(force_global):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    from karaoke.lyric_progress import partial_alignment_cues
    lines = ['line'+str(i) for i in range(12)]
    lines[1] = 'early collapsed phrase here'
    starts = [1.+i*8 for i in range(12)]
    segments = [dict(words=[dict(word=t,start=s,end=s+(.8 if i!=8 else 0))])
                for i,(t,s) in enumerate(zip(lines,starts))]
    segments[1]['words'] = [dict(word=word+' ',start=starts[1],end=starts[1]+(.12 if i==3 else 0))
                            for i,word in enumerate(lines[1].split())]
    block = dict(range=[0,12],context=[0,12],source_range=[0.,100.],result=dict(segments=segments))
    original = deepcopy(block)
    cues = partial_alignment_cues(lines,[block],100.,'English')
    assert cues[1]['status']=='collapsed_timing' and cues[8]['status']=='unresolved_timing'
    recognized, aligned = [], []
    def recognize(begin,end):
        recognized.append((begin,end))
        # Old local bounds have no lyric evidence; a global pass can locate
        # the neighbours without guessing the missing or collapsed phrases.
        return dict(segments=[dict(words=[dict(word=t,start=s,end=s+.8)
            for i,(t,s) in enumerate(zip(lines,starts)) if i not in (1,8)])]
            if (begin,end)==(0.,100.) else [])
    def align(begin,end,text,**options):
        aligned.append(text.splitlines())
        if lines[8] not in text.splitlines():
            return None
        return dict(segments=[dict(words=[dict(word=t,start=starts[lines.index(t)]-begin,
            end=starts[lines.index(t)]+.8-begin)]) for t in text.splitlines()])
    result = recover_with_local_asr(block,lines,np.zeros(100*16000),'English',recognize,align,
                                   force_global=force_global,max_alignment_attempts=1)
    assert block==original
    assert aligned==[lines[7:10]]
    assert recognized[-1]==(0.,100.) and len(recognized)==(1 if force_global else 2)
    assert result['result']['segments'][8]['words'][0]==dict(word=lines[8],start=65.,end=65.8)
    assert result['result']['segments'][1]==segments[1]


def test_regional_split_keeps_missing_line_ahead_of_optional_refinement():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['line'+str(i) for i in range(12)]
    lines[7] = 'earlier collapsed phrase here'
    starts = [1.+8*i for i in range(12)]
    segments = [dict(words=[dict(word=t,start=s,end=s+(.8 if i!=8 else 0))])
                for i,(t,s) in enumerate(zip(lines,starts))]
    segments[7]['words'] = [dict(word=w+' ',start=57.,end=57.+(.12 if i==3 else 0))
                            for i,w in enumerate(lines[7].split())]
    block = dict(range=[0,12],context=[0,12],source_range=[0.,100.],result=dict(segments=segments))
    recognized, aligned = [], []
    def recognize(begin,end):
        recognized.append((begin,end))
        omitted = (7,8) if (begin,end)==(0.,100.) else (7,)
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
            for i,(t,s) in enumerate(zip(lines,starts)) if i not in omitted and begin<=s<s+.8<=end])])
    def align(begin,end,text,**options):
        aligned.append(text.splitlines())
        if text!=lines[8]:
            return None
        return dict(segments=[dict(words=[dict(word=text,start=65.-begin,end=65.8-begin)])])
    result = recover_with_local_asr(block,lines,np.zeros(100*16000),'English',recognize,align,
                                   force_global=True,max_alignment_attempts=3)
    assert len(recognized)==2 and aligned==[lines[6:10],lines[8:9]]
    assert result['result']['segments'][8]['words'][0]==dict(word=lines[8],start=65.,end=65.8)
    assert result['result']['segments'][7]==segments[7]


@pytest.mark.parametrize('located', [1, 3])
def test_local_recognition_keeps_context_predictions_through_global_retry(located):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie', 'delta', 'omega'], [1.,4.,8.,12.,16.]
    segments = [dict(words=[dict(word=t,start=s,end=s+(0 if i in (1,3) else .8))])
                for i,(t,s) in enumerate(zip(lines,starts))]
    block = dict(range=[0,5],context=[0,5],source_range=[0.,20.],result=dict(segments=segments))
    selected = slice(located-1, located+2)
    def recognize(begin,end):
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
            for t,s in zip(lines[selected],starts[selected])])]
            if (end < 10 if located==1 else begin > 5) else [])
    def align(begin,end,text,**options):
        if text.splitlines() != lines[selected]:
            return None
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)])
                             for t,s in zip(lines[selected],starts[selected])])
    result = recover_with_local_asr(block,lines,np.zeros(20*16000),'English',recognize,align)
    assert any(r['global_localization'] for r in result['asr_localizations'])
    assert result['result']['segments'][located]['words'][0] == dict(
        word=lines[located],start=starts[located],end=starts[located]+.8)
    assert result['result']['segments'][4-located] == segments[4-located]
    assert len(result['recovery_contexts']) == 1
    assert result['recovery_contexts'][0]['context'] == [located-1,located+2]


@pytest.mark.parametrize('outside_old_context', [False, True])
def test_local_recognition_repairs_a_contradicted_neighbour_too(outside_old_context):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['intro', 'alpha', 'bravo', 'omega']
    old = [1., 4., 12., 16.]
    actual = [1., 10., 12., 16.]
    if outside_old_context:
        lines = ['intro', 'alpha', 'bravo', 'charlie', 'delta', 'omega']
        old, actual = [1.,4.,6.,8.,16.,20.], [1.,10.,12.,14.,16.,20.]
    segments = [dict(words=[dict(word=t, start=s, end=s+(.8 if i != len(lines)-2 else 0))])
                for i,(t,s) in enumerate(zip(lines,old))]
    block = dict(range=[0,len(lines)],context=[0,len(lines)],source_range=[0.,25.],result=dict(segments=segments))
    original = deepcopy(block)
    def recognize(begin,end):
        assert 0 < begin < 10 < end < 25
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
            for t,s in zip(lines[1:],actual[1:])])])
    def align(begin,end,text,**options):
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)])
            for t,s in zip(lines,actual) if t in text.splitlines()])
    result = recover_with_local_asr(block,lines,np.zeros(25*16000),'English',recognize,align)
    assert block == original
    assert result['result']['segments'][0] == segments[0]
    assert result['result']['segments'][-1] == segments[-1]
    assert result['result']['segments'][1]['words'][0]['start'] == 10.
    report = result['asr_localizations'][0]
    assert not report['global_localization']
    assert report['contradicted_lines'] == ([2,3,4] if outside_old_context else [2])
    assert report['accepted_lines'] == list(range(2,len(lines)))


@pytest.mark.parametrize('evidence', ['relocated', 'same_time', 'near_boundary', 'unrecognized', 'short_gap'])
def test_sentence_inside_nonspeech_only_requests_independent_localization(evidence):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, old = ['alpha', 'bravo', 'charlie'], [1.,4.,15.]
    starts = [1.,10.,15.] if evidence=='relocated' else old
    segments = [dict(words=[dict(word=t,start=s,end=s+1.2)]) for t,s in zip(lines,old)]
    block = dict(range=[0,3],context=[0,3],source_range=[0.,20.],result=dict(segments=segments,
        nonspeech_sections=[dict(start=4.,end=4.2 if evidence=='short_gap' else 5.2)]))
    original = deepcopy(block)
    windows, attempts = [], []
    def recognize(begin,end):
        windows.append((begin,end))
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+1.2-begin)
            for t,s in zip(lines,[1.,2.5,15.] if evidence=='near_boundary' else starts)
            if evidence!='unrecognized' or t!='bravo'])])
    def align(begin,end,text,**kwargs):
        attempts.append(text)
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+1.2-begin)])
            for t,s in zip(lines,starts) if t in text.splitlines()])
    result = recover_with_local_asr(block,lines,np.zeros(20*16000),'English',recognize,align)
    assert block == original
    assert windows == ([] if evidence=='short_gap' else [(0.,20.)])
    assert [s['words'][0]['start'] for s in result['result']['segments']] == starts
    assert bool(attempts) == (evidence=='relocated')
    if evidence!='short_gap':
        assert result['asr_localizations'][0]['activity_screened_lines'] == [2]
        assert result['asr_localizations'][0]['accepted_lines'] == ([2] if evidence=='relocated' else [])
        assert result['asr_localizations'][0]['contradicted_lines'] == ([2] if evidence=='relocated' else [])


@pytest.mark.parametrize('missing_middle', [False, True])
def test_ambiguous_global_match_cannot_displace_aligned_neighbouring_phrases(missing_middle):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['All the kids sing', 'A broken phrase', 'Hum dee la', 'Hum dee la', 'Kids']
    old = [0., 3., 10., 15., 22.]
    segments = [dict(words=[dict(word=t, start=s, end=s+(0 if i==1 or missing_middle and i in (2,3) else 1))])
                for i,(t,s) in enumerate(zip(lines,old))]
    block = dict(range=[0,5],context=[0,5],source_range=[0.,25.],result=dict(segments=segments))
    def recognize(begin,end):
        return dict(segments=[dict(words=[dict(word=lines[i],start=s-begin,end=s+1-begin)
                    for i,s in [(0,0.),(1,5.),(4,9.)] if begin <= s < s+1 <= end])])
    def align(begin,end,text,**kwargs):
        seen = 0
        result = []
        for line in text.splitlines():
            start = 10.+5*seen if line=='Hum dee la' else {lines[0]:0.,lines[1]:5.,lines[4]:22. if end>=23 else 9.}[line]
            seen += line=='Hum dee la'
            result.append(dict(words=[dict(word=line,start=start-begin,end=start+1-begin)]))
        return dict(segments=result)
    result = recover_with_local_asr(block,lines,np.zeros(25*16000),'English',recognize,align,force_global=True)
    assert [s['words'][0]['start'] for s in result['result']['segments']] == [0.,5.,10.,15.,22.]
    if not missing_middle:
        assert result['result']['segments'][2:] == segments[2:]
    assert result['asr_localizations'][0]['contradicted_lines'] == []


@pytest.mark.parametrize('scenario', ['stable', 'stale_neighbour', 'cascading_conflicts', 'late_discovery'])
def test_global_recovery_revisits_neighbours_newly_contradicted_by_repaired_phrases(scenario):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['Opening verse', 'Drink water', 'Fall asleep then dream', 'Drink water', 'Asleep and dream']
    actual = [0., 5., 12., 18., 24.]
    old = [0., 3., 6., 9., 10.] if scenario!='stable' else [0., 3., 6., 18., 24.]
    heard = [*lines[:4], 'Asleep dream']
    if scenario in ('cascading_conflicts', 'late_discovery'):
        lines = ['Opening verse', 'Middle melody', 'Quiet morning light', 'Gentle evening star', 'Asleep and dream']
        heard = [*lines[:2], 'morning light', 'evening star', 'Asleep dream']
        actual, old = [0.,10.,14.,18.,24.], [0.,3.,4.,5.,12.]
    segments = [dict(words=[dict(word=t, start=s, end=s+(0 if i==1 else 1))])
                for i,(t,s) in enumerate(zip(lines, old))]
    block = dict(range=[0,5], context=[0,5], source_range=[0.,30.], result=dict(segments=segments))
    def recognize(begin, end):
        # This partial match cannot itself invalidate a timed neighbour.
        return dict(segments=[dict(words=[dict(word=heard[i],
            start=s-begin, end=s+1-begin) for i,(t,s) in enumerate(zip(lines, actual))
            if begin <= s < s+1 <= end])])
    def align(begin, end, text, **options):
        selected = text.splitlines()
        first = next((i for i in range(len(lines)) if lines[i:i+len(selected)] == selected
                      and begin <= actual[i] < actual[i+len(selected)-1]+1 <= end), None)
        if first is None:
            return None
        return dict(segments=[dict(words=[dict(word=lines[i], start=actual[i]-begin,
            end=actual[i]+1-begin)]) for i in range(first, first+len(selected))])
    result = recover_with_local_asr(block, lines, np.zeros(30*16000), 'English',
                                    recognize, align, force_global=True,
                                    max_alignment_attempts=1 if scenario=='late_discovery' else 24)
    assert [s['words'][0]['start'] for s in result['result']['segments']] == actual
    if scenario != 'stable':
        assert result['asr_localizations'][0]['unresolved_segments']
    else:
        assert result['result']['segments'][3:] == segments[3:]


def test_local_recognition_broadens_once_when_old_neighbours_miss_the_verse():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines, starts = ['alpha', 'bravo', 'charlie'], [30., 40., 50.]
    block = dict(range=[0,3],context=[0,3],source_range=[0.,60.],result=dict(segments=[
        dict(words=[dict(word=t,start=s,end=s+(.8 if i!=1 else 0))])
        for i,(t,s) in enumerate(zip(lines,[1.,4.,8.]))]))
    windows=[]
    def recognize(begin,end):
        windows.append((begin,end))
        words=[] if end<30 else [dict(word=t,start=s,end=s+.8) for t,s in zip(lines,starts)]
        return dict(segments=[dict(words=words)])
    def align(begin,end,text,**kwargs):
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)])
                             for t,s in zip(lines,starts) if t in text.splitlines()])
    result=recover_with_local_asr(block,lines,np.zeros(60*16000),'English',recognize,align)
    assert len(windows)==2 and windows[0][1]<10 and windows[1]==(0.,60.)
    assert result['asr_localizations'][-1]['accepted_lines']==[1,2,3]
    assert [s['words'][0]['start'] for s in result['result']['segments']]==starts


def test_recognition_windows_keep_language_switches_and_boundary_words():
    from karaoke.align_lyrics_whisper import recognize_in_windows
    calls=[]
    tokens=[('morning',1.,2.),('bonjour',19.,21.),('soir',30.,31.),('night',41.,42.)]
    def recognize(begin,end):
        calls.append((begin,end))
        return dict(language='fr' if begin>0 else 'en',segments=[dict(words=[
            dict(word=word,start=start-begin,end=stop-begin)
            for word,start,stop in tokens if begin<=start<stop<=end])])
    words,reports=recognize_in_windows(recognize,0.,60.,3)
    assert words==[dict(word=word,start=start,end=end) for word,start,end in tokens]
    assert calls==[(0.,25.),(15.,45.),(35.,60.)]
    assert [r['recognition']['language'] for r in reports]==['en','fr','fr']


def test_local_recovery_attempts_later_regions_before_deep_retries():
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['early missing phrase', 'middle missing phrase', 'late missing phrase']
    starts = [1.,30.,60.]
    block = dict(range=[0,3],context=[0,3],source_range=[0.,70.],result=dict(segments=[
        dict(words=[dict(word=t,start=0.,end=0.)]) for t in lines]))
    calls = []
    def recognize(begin,end):
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+.8-begin)
            for t,s in zip(lines,starts) if begin <= s < end])])
    def align(begin,end,text,**options):
        calls.append(text)
        i = lines.index(text)
        return dict(segments=[dict(words=[dict(word=text,start=starts[i]-begin,
                                               end=starts[i]-begin+(0 if i==0 else .8))])])
    result = recover_with_local_asr(block,lines,np.zeros(70*16000),'English',recognize,align,
        force_global=True,max_alignment_attempts=3,align_original=align)
    assert calls == lines
    assert result['result']['segments'][-1]['words'] == [dict(word=lines[-1],start=60.,end=60.8)]
    assert result['result']['segments'][0]['words'][0]['start'] == result['result']['segments'][0]['words'][0]['end']


@pytest.mark.parametrize('omitted_repeat', [1, 3])
@pytest.mark.parametrize('drifted_middle', [False, True])
def test_incomplete_refrain_recognition_cannot_overwrite_bounded_acoustic_alignment(omitted_repeat, drifted_middle):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['alpha', 'sing again', 'la la', 'sing again', 'bravo', 'missing', 'charlie']
    starts = [1., 5., 8., 12., 15., 18., 21.]
    segments = [dict(words=[dict(word=text, start=t, end=t+.8)]) for text,t in zip(lines,starts)]
    segments[5]['words'][0]['end'] = 18.
    if drifted_middle:
        segments[2]['words'][0].update(start=28., end=29.)
    block = dict(range=[0,7], context=[0,7], source_range=[0.,30.], result=dict(segments=segments))
    original = deepcopy(block)
    def recognize(begin, end):
        # The intervening scat and one refrain are absent. Text alone cannot
        # tell which occurrence the single recognized refrain belongs to.
        return dict(segments=[dict(words=[dict(word=text, start=t, end=t+.8)
            for i,(text,t) in enumerate(zip(lines,starts)) if i not in (2, omitted_repeat)])])
    def align(begin, end, text, **options):
        selected = text.splitlines()
        offset = next(i for i in range(len(lines)) if lines[i:i+len(selected)] == selected)
        return dict(segments=[dict(words=[dict(word=line, start=t-begin, end=t+.8-begin)])
            for line,t in zip(selected,starts[offset:])])
    result = recover_with_local_asr(block,lines,np.zeros(30*16000),'English',recognize,align,force_global=True)
    assert block == original
    assert result['result']['segments'][1]['words'] == segments[1]['words']
    assert result['result']['segments'][3]['words'] == segments[3]['words']
    assert result['result']['segments'][5]['words'][0] == dict(word='missing',start=18.,end=18.8)
    assert result['result']['segments'][2]['words'][0] == dict(word='la la',start=8.,end=8.8)
    report = result['asr_localizations'][0]
    assert report['contradicted_lines'] == ([3] if drifted_middle else [])
    assert [report['anchors'][i]['status'] for i in (1,3)] == ['ambiguous_repeat']*2


def test_broad_recognition_anchor_cannot_erase_earlier_aligned_lyrics():
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['morning', 'echo', 'bright horizon', 'clear sky', 'missing phrase', 'finale']
    starts = [1., 4., 14., 18., 22., 26.]
    def segment(index, begin=0., rough=False):
        words = [dict(word=word, start=starts[index]+j*.5-begin, end=starts[index]+j*.5+.4-begin)
                 for j, word in enumerate(lines[index].split())]
        if rough and index == 2:
            # Recognition can attach the first syllable to a much earlier sound.
            words[0]['start'] = 2.-begin
        return dict(words=words)
    segments = [segment(i) for i in range(len(lines))]
    segments[4]['words'] = []
    block = dict(range=[0,6], context=[0,6], source_range=[0.,30.], result=dict(segments=segments))
    original = deepcopy(block)
    def recognize(begin, end):
        return dict(segments=[segment(i,begin,True) for i in range(2,6)])
    def align(begin, end, text, **options):
        return dict(segments=[segment(lines.index(line),begin,True) for line in text.splitlines()])
    result = recover_with_local_asr(block,lines,np.zeros(30*16000),'English',recognize,align,
                                   force_global=True)
    assert block == original
    assert result['result']['segments'][:4] == segments[:4]
    assert result['result']['segments'][4]['words'] == segment(4)['words']
    assert result['asr_localizations'][0]['contradicted_lines'] == []


def test_ambiguous_refrains_need_independent_bounds_and_cannot_bootstrap_them():
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recovery_anchors
    recognized = [dict(text=t, start=s, end=e, status='candidate' if s is not None else 'unresolved_text')
        for t,s,e in [('echo',1.,2.),('echo',None,None),('omega',20.,21.)]]
    original = deepcopy(recognized)
    anchors = recovery_anchors(recognized)
    assert recognized == original
    assert [a['status'] for a in anchors] == ['ambiguous_repeat','ambiguous_repeat','candidate']


@pytest.mark.parametrize('duration,budget,recovers', [(20.,24,True),(70.,2,True),(70.,1,False),(110.,24,False)])
def test_unrecognized_run_does_not_use_its_drifted_times_as_search_bounds(duration,budget,recovers):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['alpha', 'la la', 'na na', 'ta ta', 'omega']
    starts = [1.,duration*.2,duration*.35,duration*.5,duration-5]
    def segment(text, start, length=.8):
        return dict(words=[dict(word=text, start=start, end=start+length)])
    old = [1.,duration*.5,duration*.6,duration*.65,duration-5]
    segments = [segment(t,s,0 if i==2 else .8) for i,(t,s) in enumerate(zip(lines,old))]
    block = dict(range=[0,5], context=[0,5], source_range=[0.,duration], result=dict(segments=segments))
    calls = []
    def recognize(begin,end):
        return dict(segments=[dict(words=[dict(word=lines[i], start=starts[i], end=starts[i]+.8) for i in (0,4)])])
    def align(begin,end,text,**options):
        calls.append((begin,end,text))
        assert begin < 1 and end > starts[-1]  # Old predictions cannot crop the real phrase away.
        return dict(segments=[segment(t,s-begin) for t,s in zip(lines,starts)])
    result = recover_with_local_asr(block,lines,np.zeros(int(duration*16000)),'English',recognize,align,
                                   force_global=True,max_alignment_attempts=budget)
    if not recovers:
        assert calls == [] and result['result']['segments'] == segments
        assert result['asr_localizations'][0]['attempts'][0]['reason'] == (
            'window_exceeds_limit' if duration > 90 else 'local_retry_budget_exhausted')
        return
    assert len(calls) == 1
    assert result['result']['segments'][0] == segments[0] and result['result']['segments'][-1] == segments[-1]
    assert [s['words'][0]['start'] for s in result['result']['segments']] == starts
    assert result['asr_localizations'][0]['accepted_lines'] == [2,3,4]
    assert result['recovery_contexts']


@pytest.mark.parametrize('edge', ['opening', 'ending'])
@pytest.mark.parametrize('budget,view', [(1,'vocals'), (2,'vocals'), (3,'original'), (4,'original')])
def test_unrecognized_edge_uses_the_same_bounded_window_budget_as_an_internal_verse(edge,budget,view):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines=['anchor phrase','first missing verse','second missing verse']
    starts=[20.,82.,94.]
    if edge=='opening':
        lines=lines[1:]+lines[:1]
        starts=[36.,50.,110.]
    known=lines.index('anchor phrase')
    segments=[dict(words=[dict(word=t,start=s if i==known else 22.,end=s+1 if i==known else 22.)])
              for i,(t,s) in enumerate(zip(lines,starts))]
    block=dict(range=[0,3],context=[0,3],source_range=[0.,130.],result=dict(segments=segments))
    original=deepcopy(block)
    def recognize(begin,end):
        return dict(segments=[dict(words=[dict(word=lines[known],start=starts[known],end=starts[known]+1)])])
    calls=[]
    def align(begin,end,text,original=False,**options):
        calls.append((begin,end,text,original))
        assert 45 < end-begin <= 90
        if view=='original' and not original:
            return None
        if any(not begin<=s<s+1<=end for s in starts):
            return None
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+1-begin)])
                              for t,s in zip(lines,starts)])
    result=recover_with_local_asr(block,lines,np.ones(130*16000)*.1,'English',recognize,align,
                                 force_global=True,max_alignment_attempts=budget,
                                 align_original=lambda a,b,t,**kw:align(a,b,t,original=True,**kw))
    assert block==original
    if budget==1:
        assert calls==[] and result['result']['segments']==segments
        assert result['asr_localizations'][0]['attempts'][0]['reason']=='local_retry_budget_exhausted'
    elif budget==3:
        assert len(calls)==1 and not calls[0][-1]
        assert result['result']['segments']==segments
    else:
        assert len(calls)==(2 if view=='original' else 1)
        assert [s['words'][0]['start'] for s in result['result']['segments']]==starts
        assert result['result']['segments'][known]==segments[known]


@pytest.mark.parametrize('regional_support', ['full', 'none', 'outside', 'error'])
@pytest.mark.parametrize('budget', [2, 8])
def test_failed_global_alignment_relocalizes_only_the_independently_bounded_region(regional_support,budget):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines=['anchor phrase','quiet verse begins','quiet verse ends']
    starts=[20.,82.,94.]
    def segments(texts, begin, collapsed=False):
        return [dict(words=[dict(word=t,start=starts[lines.index(t)]-begin,
                    end=starts[lines.index(t)]-begin+(0 if collapsed and t!=lines[0] else 1.))])
                for t in texts]
    block=dict(range=[0,3],context=[0,3],source_range=[0.,130.],result=dict(segments=segments(lines,0.,True)))
    original=deepcopy(block)
    recognition=[]
    def recognize(begin,end):
        recognition.append((begin,end))
        if (begin,end)==(0.,130.):
            return dict(segments=segments(lines[:1],begin))
        assert 0 < begin < 20 and 95 < end < 130
        if regional_support=='error':
            raise ValueError('region unavailable')
        return dict(segments=segments(lines if regional_support!='none' else [],
                    begin-200 if regional_support=='outside' else begin))
    def align(begin,end,text,**options):
        texts=text.splitlines()
        return dict(segments=segments(texts,begin,collapsed=texts==lines))
    result=recover_with_local_asr(block,lines,np.zeros(130*16000),'English',recognize,align,
                                 force_global=True,max_alignment_attempts=budget)
    assert block==original and result['result']['segments'][0]==original['result']['segments'][0]
    assert len(recognition)==(2 if budget==8 else 1)
    if regional_support=='full' and budget==8:
        assert [s['words'] for s in result['result']['segments']]==[s['words'] for s in segments(lines,0.)]
        assert result['asr_localizations'][0]['accepted_lines']==[2,3]
    else:
        assert result['result']==original['result']


@pytest.mark.parametrize('support', ['complete', 'partial', 'missing', 'unavailable', 'acoustic', 'acoustic_partial', 'acoustic_tail',
                                     'acoustic_disagrees', 'acoustic_interior_disagrees', 'acoustic_invalid', 'acoustic_unavailable',
                                     'original', 'original_partial', 'original_missing', 'original_outside', 'original_error', 'original_unneeded',
                                     'original_budget'])
def test_failed_long_recognition_uses_bounded_overlapping_context_without_duplicating_words(support):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines=['anchor phrase','quiet verse begins','quiet verse ends']
    starts=[20.,72.,84.]
    original=[dict(words=[dict(word=t,start=s,end=s+(1 if i==0 else 0))])
              for i,(t,s) in enumerate(zip(lines,starts))]
    block=dict(range=[0,3],context=[0,3],source_range=[0.,120.],result=dict(segments=original))
    narrow, acoustic_calls, original_calls=[], [], []
    def recognize(begin,end):
        if end-begin>30:
            indexes = [0,2] if support in ('partial','original_partial') and begin>0 else [0]
            return dict(segments=[dict(words=[dict(word=lines[i],start=starts[i]-begin,end=starts[i]+1-begin)
                                             for i in indexes])])
        narrow.append((begin,end))
        assert 19 <= begin < end <= 110 and end-begin<=30
        if support=='unavailable':
            raise ValueError('short recognition unavailable')
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=s+1-begin)
            for t,s in zip(lines,starts) if (support in ('complete','partial','original_unneeded') or
                support=='acoustic_interior_disagrees' and t==lines[1]) and begin<=s<s+1<=end])])
    def recognize_original(begin,end):
        original_calls.append((begin,end))
        assert 19 <= begin < end <= 110 and end-begin<=30
        if support=='original_error':
            raise ValueError('Original recognition unavailable')
        mix_starts=[20.,72.,95.] if support=='original_partial' else starts
        return dict(segments=[dict(words=[dict(word=t,
            start=s-begin-(200 if support=='original_outside' else 0),
            end=s+1-begin-(200 if support=='original_outside' else 0))
            for t,s in zip(lines,mix_starts) if support!='original_missing' and begin<=s<s+1<=end])])
    def align(begin,end,text,**kwargs):
        texts=text.splitlines()
        return dict(segments=[dict(words=[dict(word=t,start=starts[lines.index(t)]-begin,
            end=starts[lines.index(t)]-begin+(0 if texts==lines and t!=lines[0] else 1))]) for t in texts])
    def localize(begin,end,text):
        acoustic_calls.append((begin,end,text))
        if support=='acoustic_unavailable':
            raise ValueError('Independent localization unavailable')
        words,offset=[],0
        for t,s in zip(lines,starts):
            words.append(dict(text=t,first=offset,stop=offset+len(t),start=s-.2,end=s+1.2))
            offset+=len(t)+1
        if support=='acoustic_interior_disagrees':
            unit=words[1]
            split=unit['first']+len('quiet')
            words[1:2]=[dict(unit,text='quiet',stop=split,end=72.4),
                        dict(unit,text='verse begins',first=split+1,start=72.4)]
        shift=2 if support=='acoustic_disagrees' else .1
        second=[dict(w,start=w['start']+shift,end=w['end']+shift) for w in words]
        if support=='acoustic_interior_disagrees':
            second[2].update(start=74.4,end=75.2)
        if support=='acoustic_tail':
            second=[dict(w,end=w['end']+1.) for w in second]
        if support=='acoustic_invalid':
            second[1]['text']='Unrelated words'
        if support=='acoustic_partial':
            words,second=words[1:],second[1:]
        return dict(views=[words,second])
    result=recover_with_local_asr(block,lines,np.zeros(120*16000),'English',recognize,align,
                                 force_global=True,max_alignment_attempts=12 if support=='original_budget' else 24,
                                 localize=localize if support.startswith('acoustic') else None,
                                 recognize_original=recognize_original if support.startswith('original') else None)
    if support in ('acoustic','acoustic_partial','acoustic_tail'):
        assert not narrow  # Known-text localization succeeded; no need to transcribe again.
    else:
        assert narrow and len(narrow)<=5
    assert all(a[1]>b[0] for a,b in zip(narrow,narrow[1:]))
    assert result['result']['segments'][0]==original[0]
    assert len(acoustic_calls)==int(support.startswith('acoustic'))
    assert bool(original_calls)==(support.startswith('original') and support not in ('original_unneeded','original_budget'))
    assert len(original_calls)<=5
    if support in ('complete','partial','acoustic','acoustic_partial','acoustic_tail','acoustic_interior_disagrees',
                   'original','original_partial','original_unneeded'):
        assert [s['words'][0]['start'] for s in result['result']['segments']]==starts
        assert result['asr_localizations'][0]['accepted_lines']==[2,3]
        assert all(s['words'][0]['end']>s['words'][0]['start'] for s in result['result']['segments'])
    else:
        assert result['result']['segments']==original


@pytest.mark.parametrize('prefix', ['recognized', 'scattered', 'too_short'])
def test_partial_recognition_narrows_acoustic_search_after_a_long_instrumental(prefix):
    from karaoke.anchors_from_asr import anchors_from_words
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['前一句', '风吹过this long English phrase', '后一句']
    words = [dict(word='前一句',start=1.,end=2.)]
    for i,word in enumerate('风吹过' if prefix != 'too_short' else '风吹'):
        words.append(dict(word=word,start=30+i*.2,end=30+(i+1)*.2))
        if prefix == 'scattered':
            words.append(dict(word='噪声',start=30+(i+1)*.2,end=30+(i+1)*.2+.1))
    words.append(dict(word='后一句',start=35.,end=36.))
    anchors = anchors_from_words(lines,words,'Chinese',allow_missing=True)
    assert anchors[1]['status'] == 'unresolved_text' and anchors[1]['start'] is None
    if prefix == 'recognized':
        assert anchors[1]['search_start'] == 30.
    else:
        assert 'search_start' not in anchors[1]
    segments = [dict(words=[dict(word=t,start=s,end=e)]) for t,s,e in
                zip(lines,[1.,4.,35.],[2.,4.,36.])]
    block = dict(range=[0,3],context=[0,3],source_range=[0.,40.],result=dict(segments=segments))
    calls=[]
    def align(begin,end,text,**kwargs):
        calls.append((begin,end,text))
        if begin < 29.:
            return None
        return dict(segments=[dict(words=[dict(word=t,start=s-begin,end=e-begin)])
            for t,s,e in zip(lines[1:],[30.,35.],[33.,36.])])
    result = recover_with_local_asr(block,lines,np.zeros(40*16000),'Chinese',
        lambda begin,end:dict(segments=[dict(words=[dict(w,start=w['start']-begin,end=w['end']-begin) for w in words])]),align)
    if prefix == 'recognized':
        assert len(calls) == 1
        begin,end,text = calls[0]
        assert 29. <= begin < 30. and 36. <= end <= 40. and text == '\n'.join(lines[1:])
        assert result['asr_localizations'][0]['accepted_lines'] == [2]
        assert result['result']['segments'][1]['words'] == [dict(word=lines[1],start=30.,end=33.)]
    else:
        assert all(begin < 29. for begin,end,text in calls)
        assert result['result'] == block['result']
    assert result['result']['segments'][0] == segments[0]
    assert result['result']['segments'][2] == segments[2]


@pytest.mark.parametrize('partly_timed', [False, True])
@pytest.mark.parametrize('asr_gap', ['none', 'long_intro', 'missing_middle', 'boundary_overlap', 'boundary_unresolved', 'unheard_opening', 'long_phrase_window', 'reuse_context'])
def test_widespread_alignment_collapse_is_relocalized_from_recognized_words(partly_timed, asr_gap):
    from karaoke.align_lyrics_whisper import recover_with_local_asr
    lines = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel']
    starts = {line:(50. if asr_gap=='unheard_opening' else 30.)+i*6 for i,line in enumerate(lines)}
    if asr_gap=='long_phrase_window':
        starts = {t:s+(48 if i>=2 else 0) for i,(t,s) in enumerate(starts.items())}
    duration = 120. if asr_gap=='unheard_opening' else 90.
    if asr_gap=='long_phrase_window':duration=140.
    ends = {line:start+1 for line,start in starts.items()}
    if asr_gap == 'reuse_context':
        ends = {line:start+2.5 for line,start in starts.items()}
    block = dict(range=[0,8],context=[0,8],source_range=[0.,duration],result=dict(segments=[
        dict(text=t,words=[dict(word=t,start=float(i),end=float(i if i<5 else i+1))])
        for i,t in enumerate(lines)]))
    if partly_timed:
        for i,s in enumerate(block['result']['segments'][:5]):
            tokens = [*s['text'][:3],s['text'][3:]]
            s['words'] = [dict(word=t,start=float(i),end=i+(.05 if j==3 else 0)) for j,t in enumerate(tokens)]
    if asr_gap.startswith('boundary_'):
        starts['echo'] = 58.5
        ends['echo'] = 59.
        for i in range(5,8):
            start = starts[lines[i]]-(2 if i==5 else 0)
            block['result']['segments'][i] = dict(text=lines[i],words=[dict(word=lines[i],start=start,end=ends[lines[i]])])
    calls=[]
    def recognize(begin,end):
        calls.append((begin,end))
        words = [dict(word=t,start=starts[t],end=ends[t]) for t in lines
                 if not (asr_gap == 'missing_middle' and t == 'golf')
                 and not (asr_gap == 'unheard_opening' and t in lines[:2])]
        if asr_gap == 'long_intro':words[0]['start'] = 0.
        if asr_gap == 'long_phrase_window':words[2]['start'] = 42.
        return dict(segments=[dict(words=words)])
    aligned_texts = []
    def align(begin,end,text,**kwargs):
        aligned_texts.append(text)
        assert end-begin <= (90 if asr_gap=='unheard_opening' or asr_gap=='long_phrase_window' and text==lines[2] else 45)
        if asr_gap=='boundary_unresolved' and text.splitlines()==lines[2:8]:
            return None
        segments = [dict(text=t,words=[dict(word=t,start=starts[t]-begin,end=ends[t]-begin)])
                    for t in text.splitlines()]
        if asr_gap == 'boundary_unresolved':
            for segment in segments:
                if segment['text'] == 'foxtrot':
                    segment['words'][0]['start'] -= 2  # Still crosses the preceding word onset.
        if asr_gap == 'reuse_context' and text.splitlines() == lines[2:7]:
            segments[1]['words'][0]['start'] = starts['charlie']-.5-begin
        return dict(segments=segments)
    result = recover_with_local_asr(block,lines,np.zeros(int(duration*16000)),'English',recognize,align)
    assert calls == [(0.,duration)]*(2 if asr_gap=='boundary_unresolved' else 1)
    last = 6 if asr_gap=='boundary_unresolved' else 7 if asr_gap=='boundary_overlap' else 9
    assert result['asr_localizations'][0]['accepted_lines'] == list(range(1,last))
    assert [s['words'][0]['start'] for s in result['result']['segments'] if s['words']] == [v for i,v in enumerate(starts.values()) if asr_gap!='boundary_unresolved' or i!=5]
    if asr_gap=='boundary_unresolved':
        assert result['result']['segments'][5]['words'] == []
        assert result['asr_localizations'][0]['unresolved_segments'][6] == block['result']['segments'][5]
    if asr_gap.startswith('boundary_'):
        assert result['result']['segments'][6:] == block['result']['segments'][6:]
        assert result['asr_localizations'][0]['boundary_reconciliations']
    if asr_gap == 'reuse_context':
        assert len(aligned_texts) == 3  # Reuse the existing joint prediction, no new inference.
        assert result['asr_localizations'][0]['boundary_reconciliations'][0]['selected_block'] > 0


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
    tmp_path = tmp_path/'song'
    tmp_path.mkdir()
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
    assert len(calls) == 2  # A changed source cannot bypass the provider's active hold.
    clock[0] = 2802.
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


@pytest.mark.parametrize('damage', ['nonfinite', 'length', 'channels', 'rate', 'subtype', 'empty', 'truncated'])
def test_soak_checks_actual_stem_integrity_without_exact_sum(tmp_path, damage):
    import soundfile as sf
    from karaoke.tools.soak import verify_stems
    original = np.full((100, 2), 1e-30)
    lead = np.full_like(original, .000288816)
    paths = {name: tmp_path/(name+'.wav') for name in ('lead_vocals','backing_vocals','instrumental')}
    for path, audio in zip(paths.values(), (lead, original-lead, np.zeros_like(original))):
        sf.write(path, audio, 8000, subtype='DOUBLE')
    assert verify_stems(paths, 100/8000) == dict(sample_rate=8000, channels=2, frames=100)
    audio = {'nonfinite': np.full_like(lead, np.nan), 'length': lead[:-10],
             'channels': lead[:, :1], 'empty': lead[:0]}.get(damage, lead)
    for path in (paths.values() if damage == 'truncated' else [paths['lead_vocals']]):
        sf.write(path, lead[:10] if damage == 'truncated' else audio,
                 16000 if damage == 'rate' else 8000, subtype='FLOAT' if damage == 'subtype' else 'DOUBLE')
    with pytest.raises(ValueError, match='Non-finite|incompatible|duration'):
        verify_stems(paths, 100/8000)


@pytest.mark.parametrize('diagnostic,reason', [
    ("Unable to connect to proxy: Tunnel connection failed: 403 Forbidden", '网络代理'),
    ("Sign in to confirm you're not a bot", '登录验证'),
    ('HTTP Error 429: Too Many Requests', '限流（429）'),
    ("HTTP Error 429: Too Many Requests\nSign in to confirm you're not a bot", '限流（429），并要求登录验证'),
    ('ERROR: Video unavailable', '无法访问'),
    ('ERROR: unable to download video data: HTTP Error 403: Forbidden', '拒绝下载（403）'),
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


@pytest.mark.parametrize('first_stage,next_stage,competing_stage', [
    ('separate_music', 'align_words', 'separate_lead'), ('render', 'mux_video', 'render')])
def test_stage_pipeline_overlaps_different_work_but_serializes_separation(tmp_path, monkeypatch,
                                                                        first_stage, next_stage, competing_stage):
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
        stage(first, first_stage, 'first-separation')
        stage(first, next_stage, 'first-alignment')
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first_song)
        two = None
        try:
            wait(lambda: (tmp_path/'first-separation.started').exists())
            two = pool.submit(stage, second, competing_stage, 'second-separation')
            wait(lambda: second.stage_progress.get(competing_stage, {}).get('waiting'))
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


@pytest.mark.parametrize('stage,capacity', [('separate_music', 2), ('align_words', 2),
                                             ('package_three_stems', 2), ('download', 1)])
def test_compute_slots_overlap_up_to_capacity_and_keep_downloads_serial(tmp_path, monkeypatch, stage, capacity):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv('KARAOKE_STAGE_LOCK_DIR', str(tmp_path/'slots'))
    monkeypatch.setenv('KARAOKE_COMPUTE_CONCURRENCY', '2')
    pipelines = [Pipeline(dict(url=URL), tmp_path/str(i)) for i in range(3)]
    code = ("import pathlib,sys,time;pathlib.Path(sys.argv[1]).touch();"
            "\nwhile not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)")
    def run(i):
        pipelines[i].run_process(stage, [sys.executable, '-c', code,
            str(tmp_path/f'{i}.started'), str(tmp_path/f'{i}.release')], timeout=10)
    def wait_for(check):
        deadline = time.monotonic()+8
        while not check() and time.monotonic() < deadline:
            time.sleep(.01)
        assert check()
    def started(i):
        return (tmp_path/f'{i}.started').exists()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        try:
            futures.append(pool.submit(run, 0))
            wait_for(lambda: started(0))
            futures.append(pool.submit(run, 1))
            wait_for(lambda: started(1) if capacity == 2 else pipelines[1].stage_progress.get(stage, {}).get('waiting'))
            futures.append(pool.submit(run, 2))
            wait_for(lambda: pipelines[2].stage_progress.get(stage, {}).get('waiting'))
            assert not started(2)
            assert started(1) == (capacity == 2)
            (tmp_path/'0.release').touch()
            if capacity == 1:
                wait_for(lambda: started(1))
                assert not started(2)
                (tmp_path/'1.release').touch()
            wait_for(lambda: started(2))
        finally:
            for i in range(3):
                (tmp_path/f'{i}.release').touch()
        for future in futures:
            future.result()
    assert all(not pipeline.active for pipeline in pipelines)


def test_started_pipelines_respect_provider_backoff_before_taking_download_lane(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from karaoke.job_schema import download_backoff
    slots = tmp_path/'slots'
    monkeypatch.setenv('KARAOKE_STAGE_LOCK_DIR', str(slots))
    first = Pipeline(dict(url=URL), tmp_path/'first')
    with pytest.raises(StageFailure, match='429'):
        first.run_process('download', [sys.executable, '-c',
            "import sys;print('HTTP Error 429: Too Many Requests');print('Retry-After: 125');sys.exit(1)"])
    assert 120 < download_backoff(URL)-time.time() <= 125
    second = Pipeline(dict(url=URL), tmp_path/'second')
    other = Pipeline(dict(url='https://www.bilibili.com/video/BV1bK411W797'), tmp_path/'other')
    def download(pipeline):
        with pipeline.stage_slot('download_video'):
            return pipeline.output.name
    with ThreadPoolExecutor(max_workers=2) as pool:
        waiting = pool.submit(download, second)
        try:
            assert pool.submit(download, other).result(timeout=3) == 'other'
            assert not waiting.done()
        finally:
            # Expire the recorded limit; the existing child resumes naturally.
            (slots/'youtube.retry.json').write_text('0')
        assert waiting.result(timeout=3) == 'second'


def test_waiting_stages_follow_playlist_priority_without_interrupting_work(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from karaoke.api import Worker
    worker = Worker(Store(tmp_path), concurrency=None)
    worker.prioritize(['next', 'later'])
    monkeypatch.setenv('KARAOKE_STAGE_LOCK_DIR', str(tmp_path/'stage-locks'))
    blocker, later, next_song = [Pipeline(dict(url=URL), tmp_path/name) for name in ('running', 'later', 'next')]
    started, release = threading.Event(), threading.Event()
    order = []
    def run(pipeline):
        with pipeline.stage_slot('separate_music'):
            order.append(pipeline.output.name)
            if pipeline is next_song:
                started.set()
                assert release.wait(5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with blocker.stage_slot('separate_lead'):
            a = pool.submit(run, later)
            b = pool.submit(run, next_song)
            deadline = time.monotonic()+5
            while not (later.stage_progress and next_song.stage_progress) and time.monotonic() < deadline:
                time.sleep(.01)
            assert later.stage_progress and next_song.stage_progress and order == []
        try:
            assert started.wait(5) and order == ['next']
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert order == ['next', 'later']


def test_download_metadata_is_available_before_transfer_finishes(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    pipeline.source.parent.mkdir()
    info = dict(title='Song', artist='Singer', track='Song', duration=30)
    (pipeline.source.parent/'source.info.json').write_text(json.dumps(info))
    (tmp_path/'progress.json').write_text('{}')
    called = []
    def ready(metadata, duration):
        called.append((metadata, duration))
        (tmp_path/'metadata-observed').touch()
    pipeline.metadata_ready = ready
    code = '''import pathlib,sys,time
p=pathlib.Path(sys.argv[1])
print('KARAOKE_SOURCE_READY',flush=True)
deadline=time.monotonic()+4
while not (p/'metadata-observed').exists() and time.monotonic()<deadline: time.sleep(.01)
assert (p/'metadata-observed').exists(), 'metadata waited for download completion'
print('KARAOKE_SOURCE_READY',flush=True)
'''
    pipeline.run('download', [sys.executable, '-u', '-c', code, str(tmp_path)], timeout=5)
    assert called == [(info, 30)]
    event = pipeline.timeline['download']
    assert event['ready'] <= event['resource_acquired'] <= event['started'] <= event['finished']
    assert event['status'] == 'completed'


@pytest.mark.parametrize('status', [None, 403, 429])
def test_fresh_metadata_downloads_stream_without_reextracting_webpage(tmp_path, status):
    from functools import partial
    video = tmp_path/'stream.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=blue:s=64x64:d=0.2',
                    '-c:v', 'libx264', str(video)], check=True)
    requests = []
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if status:
                self.send_error(status)
            else:
                super().do_GET()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    media = '/stream.mp4'
    metadata = dict(id='sample', title='Example', extractor_key='Generic',
        webpage_url=base+'/never-reextract', formats=[dict(format_id='v', url=base+media,
            ext='mp4', vcodec='avc1', acodec='none', height=64)])
    info = tmp_path/'info.json'
    info.write_text(json.dumps(metadata))
    try:
        result = subprocess.run([sys.executable, '-m', 'karaoke.download_from_info', '--ignore-config',
            '--load-info-json', str(info), '-f', 'bv', '--retries', '0', '-o', str(tmp_path/'out.mp4')],
            capture_output=True, text=True)
        assert (result.returncode != 0) is (status is not None), result.stderr
        assert requests == [media]
        assert (tmp_path/'out.mp4').exists() is (status is None)
    finally:
        server.shutdown()
        thread.join()



@pytest.mark.parametrize('stage,next_stage', [('render', 'render'), ('download', 'download_video')])
def test_interactive_probe_resumes_new_songs_and_shares_stage_locks(tmp_path, stage, next_stage):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from karaoke.tools.soak import run_probe
    source = str(Path(__file__).resolve().parent.parent)
    code = r"""import json,pathlib,sys,time
sys.path.insert(0,sys.argv[2])
from karaoke.pipeline import Pipeline
p=pathlib.Path(sys.argv[1])
request=json.loads((p/'request.json').read_text())
stage=request['title'] if request.get('title') in ('render','download','download_video') else 'separate_music'
pipeline=Pipeline(request,p)
with pipeline.stage_slot(stage):
    pipeline.phase(stage,True)
    (p/'entered').touch()
    while not (p.parent/'release-all').exists() and not (p/'release').exists(): time.sleep(.01)
    pipeline.phase(stage,False)
(p/'media').write_text('ok')
(p/'result.json').write_text(json.dumps({'artifacts':{n:{'path':'media','bytes':2}
    for n in ['mv','instrumental','backing_vocals','lead_vocals']}}))
"""
    command = lambda folder: [sys.executable, '-u', '-c', code, str(folder), source]
    pause = tmp_path/'preparation.pause'
    pause.write_text('interactive repair owns this pause')
    app = create_app(tmp_path, worker_command=command, version='test', worker_concurrency=None)
    def wait(check):
        deadline = time.monotonic()+10
        while not check() and time.monotonic() < deadline:
            time.sleep(.01)
        assert check()
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        repair = client.post('/v1/room/entries', json={'url':URL, 'title':stage}).json()['entries'][-1]['job_id']
        folder = tmp_path/repair
        future = pool.submit(run_probe, client, 'http://testserver', folder, command(folder),
                             resume_preparation=pause.read_text())
        try:
            wait(lambda: (folder/'entered').exists() and not pause.exists())
            wait(lambda: app.state.store.read(repair).get('active_stages') == [stage])
            new = client.post('/v1/room/entries', json={'url':URL, 'title':'separation'}).json()['entries'][-1]['job_id']
            wait(lambda: (tmp_path/new/'entered').exists())
            assert not future.done() and app.state.store.read(new)['status'] == 'running'
            assert repair not in app.state.worker.processes  # no duplicate launch
            other = client.post('/v1/room/entries', json={'url':URL, 'title':next_stage, 'force':True}).json()['entries'][-1]['job_id']
            wait(lambda: app.state.store.read(other).get('stage_progress', {}).get(next_stage, {}).get('waiting'))
            assert not (tmp_path/other/'entered').exists()
            (folder/'release').touch()
            assert future.result(timeout=10) == 0
            wait(lambda: (tmp_path/other/'entered').exists())
        finally:
            (tmp_path/'release-all').touch()
            future.result(timeout=15)
        assert finished(client, new)['status'] == 'completed'
        assert finished(client, other)['status'] == 'completed'
        assert not pause.exists()


def test_interactive_probe_cannot_resume_another_owners_pause(tmp_path):
    from karaoke.tools.soak import run_probe
    pause = tmp_path/'preparation.pause'
    pause.write_text('user pause')
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        job = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][-1]['job_id']
        with pytest.raises(ValueError, match='owner'):
            run_probe(client, 'http://testserver', tmp_path/job, fake_worker(tmp_path/job),
                      resume_preparation='repair pause')
        assert pause.read_text() == 'user pause'
        assert app.state.store.read(job)['status'] == 'queued'


def test_external_repair_notifies_subscribers_without_approving_results(tmp_path):
    from karaoke.api import Store
    (tmp_path/'preparation.pause').touch()
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        job = client.post('/v1/room/entries', json={'url':URL}).json()['entries'][-1]['job_id']
        events = app.state.room.updates.events(app.state.room.snapshot)
        client.portal.call(anext, events)
        Store(tmp_path).update(job, status='running', active_stages=['render'])
        pending = client.portal.start_task_soon(anext, events)
        try:
            assert client.post('/v1/room/events', json={'status':'completed'}).status_code == 204
            snapshot = json.loads(pending.result(timeout=2).split('data: ', 1)[1])
            assert snapshot['entries'][0]['active_stages'] == ['render']
            assert snapshot['entries'][0]['status'] == 'running'
            assert app.state.store.read(job)['result'] is None
        finally:
            pending.cancel()
            client.portal.call(events.aclose)


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


def test_poster_uses_thumbnail_or_video_frame_and_failure_is_optional(tmp_path, monkeypatch):
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
            for malformed in ({'url':'invalid'}, 123, ['invalid']):
                assert create_poster(video, poster, thumbnail=malformed, seconds=0)
            from pathlib import Path
            unlink = Path.unlink
            def deny_optional_cleanup(path, *args, **kwargs):
                if path.name == '.poster.tmp.jpg':
                    raise PermissionError('cleanup unavailable')
                return unlink(path, *args, **kwargs)
            monkeypatch.setattr(Path, 'unlink', deny_optional_cleanup)
            assert create_poster(video, poster, seconds=0)
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
            title='A new video', uploader='Not necessarily the singer', duration=120,
            thumbnail='https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg', **tags)))

    probe = dict(streams=[dict(codec_type='video'), dict(codec_type='audio')], format=dict(duration='120'))
    with patch.object(pipeline, 'run', side_effect=download), patch(
            'karaoke.pipeline.subprocess.check_output', return_value=json.dumps(probe).encode()):
        pipeline.download()
    pipeline.phase('separate_music', True)
    events = [json.loads(line.removeprefix('KARAOKE_PROGRESS '))
              for line in capsys.readouterr().out.splitlines()]
    info = dict(title=expected[0], artist=expected[1], duration_seconds=120,
                thumbnail='https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg')
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
    pipeline = Pipeline(dict(url=URL, language=requested), tmp_path, audio_only=True)
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


@pytest.mark.parametrize('video_issue', [None, 'identity', 'duration'])
@pytest.mark.parametrize('audio_codec', ['pcm_s16le', 'libopus'])
def test_audio_download_promotes_to_video_without_replacing_tested_audio(tmp_path, monkeypatch, video_issue, audio_codec):
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
            args += [f'color=blue:s=160x90:r=30:d={4 if video_issue == "duration" else 2}', '-c:v', 'libx264']
        subprocess.run(args+[str(path)], check=True)
        path.with_suffix('.info.json').write_text(json.dumps(dict(id='wrong' if video_issue=='identity' and name!='download' else 'same',
            extractor_key='Youtube', title='Example', duration=2, format_id='a1' if name=='download' else 'v1')))
    monkeypatch.setattr(pipeline, 'run', download)
    with patch('karaoke.pipeline.create_poster', side_effect=AssertionError('Audio test downloaded artwork')):
        pipeline.download()
        pipeline.download()
    before = pcm_sha(decode_source(pipeline.source)[0])
    assert calls == ['download'] and not (tmp_path/'result.json').exists()
    pipeline.audio_only = False
    if video_issue:
        with pytest.raises(StageFailure, match='来源不同' if video_issue == 'identity' else '时长不匹配'):
            pipeline.download(artwork=False)
        assert json.loads((tmp_path/'source.json').read_text())['audio_only']
    else:
        metadata, _ = pipeline.download(artwork=False)
        assert metadata['format_id'] == 'v1+a1'
        pipeline.download(artwork=False)
        assert json.loads((tmp_path/'source.json').read_text())['audio_only']
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-of', 'json', str(pipeline.video)]))
        assert {stream['codec_type'] for stream in probe['streams']} == {'video'}
        assert not (tmp_path/'video-download/with-audio.mkv').exists()
        assert pipeline.video != pipeline.source
    assert calls == ['download', 'download_video']
    assert pcm_sha(decode_source(pipeline.source)[0]) == before


@pytest.mark.parametrize('mode,expected', [('full', '720+audio'), ('promote', '720'), ('audio', 'audio')])
@pytest.mark.parametrize('custom_cookies', [False, True])
def test_download_selects_720p_without_reducing_audio_quality(tmp_path, monkeypatch, mode, expected, custom_cookies):
    from yt_dlp import YoutubeDL, parse_options
    cookie_file = tmp_path/('configured/session.txt' if custom_cookies else 'private/youtube-cookies.txt')
    pipeline = Pipeline(dict(url=URL), tmp_path/'job', audio_only=mode == 'audio',
                        cookies_file=str(cookie_file) if custom_cookies else None)
    commands = []
    class DownloadSelected(Exception):
        pass
    def capture(name, command, **kwargs):
        commands.append(command)
        raise DownloadSelected
    monkeypatch.setattr(pipeline, 'run', capture)
    if mode == 'promote':
        pipeline.source.parent.mkdir()
        pipeline.source.write_bytes(b'cached audio')
    # Each invocation rechecks the private file; missing files use anonymous
    # download, including when an exported session is removed between requests.
    for present in (False, True, False):
        if present:
            cookie_file.parent.mkdir()
            cookie_file.write_text('# Netscape HTTP Cookie File\n')
        else:
            cookie_file.unlink(missing_ok=True)
        with pytest.raises(DownloadSelected):
            if mode == 'promote':
                pipeline.promote_video({}, {})
            else:
                pipeline.download(artwork=False)
        command = commands[-1]
        assert ('--cookies' in command) is present
        if present:
            assert str(command[command.index('--cookies')+1]) == str(cookie_file)
    formats = [dict(format_id=str(height), url=f'https://media.example/{height}.mp4', ext='mp4',
                    height=height, vcodec='avc1', acodec='none') for height in (360, 720, 1080, 2160)]
    formats += [dict(format_id=name, url=f'https://media.example/{name}.webm', ext='webm',
                     vcodec='none', acodec='opus', abr=rate) for name, rate in [('small-audio', 64), ('audio', 160)]]
    options = parse_options([str(arg) for arg in commands[0][3:]]).ydl_opts
    with YoutubeDL(dict(options, quiet=True, no_warnings=True)) as downloader:
        assert 'node' in downloader.params['js_runtimes']
        result = downloader.process_ie_result(dict(id='sample', title='sample', duration=120,
            is_live=False, formats=formats), download=False)
    assert result['format_id'] == expected


@pytest.mark.parametrize('mode', ['full', 'audio', 'playback_audio'])
@pytest.mark.parametrize('failure', ['transcribe', 'prepare_lyrics', 'align_words', 'align_phonemes', 'no_subtitles', 'separate_lead', 'package_three_stems', 'validate_timing', 'overlap', 'dependency_overlap', 'render_overlap', 'video_overlap'])
def test_playback_survives_text_failure_but_audio_and_acceptance_remain_strict(tmp_path, monkeypatch, mode, failure):
    import soundfile as sf
    audio_only, strict = mode != 'full', mode == 'audio'
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=audio_only, device='cpu',
                        playback_policy=mode == 'playback_audio')
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
    (tmp_path/'source.json').write_text(json.dumps(dict(audio_only=failure == 'video_overlap')))
    (tmp_path/'poster.jpg').write_bytes(b'poster fixture')
    (tmp_path/'alignment').mkdir()
    coarse = [dict(text='A little song', start=.1, end=.9, status='candidate')]
    (tmp_path/'alignment/cues.json').write_text(json.dumps(coarse))
    pipeline.alignment_outputs.add(tmp_path/'alignment')
    monkeypatch.setenv('OPENAI_API_KEY', 'unit-test-placeholder')
    monkeypatch.setattr(pipeline, 'download', lambda: ({'format_id': 'fixture'}, 1))
    alignment_started = threading.Event()
    picture_started = threading.Event()
    lyrics_started, resampled, packaged = (threading.Event() for _ in range(3))
    def fail(stage):
        if stage == 'separate_lead' and failure == 'overlap':
            assert alignment_started.wait(10), 'Alignment waited for independent lead separation'
        if stage == 'package_three_stems' and failure == 'render_overlap' and not audio_only:
            assert picture_started.wait(5), 'Picture rendering waited for MP3 packaging'
        if failure == 'dependency_overlap':
            if stage == 'separate_music' and not strict:
                assert lyrics_started.wait(5), 'Lyric lookup waited for separation'
            elif stage == 'package_two_stems':
                assert resampled.wait(5), 'Resampling waited for independent stem packaging'
            elif stage == 'resample_vocals':
                resampled.set()
            elif stage == 'package_three_stems':
                packaged.set()
        if stage == failure:
            raise StageFailure(stage, 'fixture failure')
    def run(stage, command, **kwargs):
        if stage == 'resample_vocals' and failure == 'dependency_overlap':
            subprocess.run(command, check=True, capture_output=True)
        fail(stage)
    monkeypatch.setattr(pipeline, 'run', run)
    def prepare(*args):
        lyrics_started.set()
        if failure == 'dependency_overlap':
            assert packaged.wait(5), 'Stem packaging waited for independent lyric lookup'
        fail('transcribe')
        fail('prepare_lyrics')
        if failure != 'overlap' and len(args) > 2:
            args[-1].result()
        return lyrics
    monkeypatch.setattr(pipeline, 'prepare_text', prepare)
    def align(*args):
        alignment_started.set()
        (tmp_path/'quality.json').write_text('{}')
        (tmp_path/'cues.json').write_text(json.dumps(coarse))
        if failure == 'no_subtitles':
            pipeline.alignment_outputs.clear()
            raise StageFailure('align_words', 'fixture failure')
        fail('align_words')
        fail('align_phonemes')
    monkeypatch.setattr(pipeline, 'align', align)
    validated = []
    def validate(*args):
        validated.append(True)
        fail('validate_timing')
    monkeypatch.setattr(pipeline, 'validate_alignment', validate)
    monkeypatch.setattr(pipeline, 'render_picture', lambda result: picture_started.set())
    def promote(metadata, source):
        assert alignment_started.wait(5), 'Alignment waited for the picture download'
        return metadata
    monkeypatch.setattr(pipeline, 'promote_video', promote)
    monkeypatch.setattr(pipeline, 'render_video', lambda result: (tmp_path/'mv.mp4').write_bytes(b'video fixture'))
    if failure.endswith('overlap'):
        result = pipeline.execute()
        assert result['validation_scope'] == ('audio_and_lyrics' if audio_only else 'full_mv')
        if failure == 'dependency_overlap':
            assert sf.info(tmp_path/'vocals-16k.wav').samplerate == 16000
            assert (tmp_path/'vocals-asr.mp3').exists() == strict
    elif strict or failure in ('separate_lead', 'package_three_stems', 'no_subtitles', 'validate_timing'):
        with pytest.raises(StageFailure) as caught:
            pipeline.execute()
        assert not (tmp_path/'result.json').exists()
        assert not (tmp_path/'audio-result.json').exists()
        if failure == 'no_subtitles' and not strict:
            assert caught.value.stage == 'subtitles'
            assert not (tmp_path/'mv.mp4').exists()
            assert (tmp_path/'stems/lead.wav').is_file()
            assert json.loads((tmp_path/'playback-warning.json').read_text())['omitted_lines'] == [1]
        else:
            assert str(caught.value) == 'fixture failure'
    else:
        result = pipeline.execute()
        assert result['validation_scope'] == 'playback_with_lyrics_review'
        assert result['quality']['lyrics_provenance']['needs_review']
        assert not result['quality']['playback_fallback']['word_alignment_passed']
        assert result['quality']['playback_fallback']['error']['code'] == failure+'_failed'
        fallback = json.loads((tmp_path/'cues.json').read_text())
        assert fallback[0]['units'] == [dict(text='A little song', first=0, stop=13, start=.1, end=.9,
                                           timing_source='linear_fallback', bounds_source='neighbor_or_phrase')]
        assert ('mv' in result['artifacts']) == (not audio_only)
        assert result['shared_preview_gain'] == .8
        assert (tmp_path/'audio-result.json').exists() == audio_only
        assert validated == [True], 'Playback fallback bypassed the common timing gate'


@pytest.mark.parametrize('change', [None, 'url', 'metadata', 'code', 'lyrics_source'])
def test_prepared_lyrics_preserve_provenance_and_reject_different_inputs(tmp_path, monkeypatch, change):
    from karaoke.lyric_lookup import prepared_lyrics_identity
    metadata = dict(title='Example', artist='Singer')
    lyrics = dict(title='Example', artist='Singer', language='English', lines=['A little song'],
                  source_url=URL, provenance=dict(route='lyrics_first', kind='online_lyric_text',
                  lyric_lookup=dict(provider='line_music', url='https://music-tw.line.me/track/123')))
    packet = dict(identity=prepared_lyrics_identity(URL, metadata), lyrics=lyrics)
    if change == 'url':
        packet['identity']['url'] = 'https://www.youtube.com/watch?v=abcdefghijk'
    elif change == 'metadata':
        metadata['title'] = 'Another song'
    elif change == 'code':
        packet['identity']['code_sha256']['lyric_lookup.py'] = 'old implementation'
    elif change == 'lyrics_source':
        packet['lyrics']['source_url'] = 'https://www.youtube.com/watch?v=abcdefghijk'
    prepared = tmp_path/'prepared.json'
    prepared.write_text(json.dumps(packet))
    pipeline = Pipeline(dict(url=URL), tmp_path/'job', audio_only=True, playback_policy=True,
                        prepared_lyrics=prepared)
    monkeypatch.setenv('KARAOKE_ASR_BACKEND', 'local')
    with patch('karaoke.lyric_lookup.prepare_online_lyrics', side_effect=AssertionError('Repeated network lookup')):
        if change:
            with pytest.raises(RuntimeError, match='Prepared'):
                pipeline.prepare_text(metadata, 10)
        else:
            assert pipeline.prepare_text(metadata, 10) == lyrics
            assert json.loads((pipeline.output/'lyrics.json').read_text()) == lyrics
            assert json.loads((pipeline.output/'anchors.json').read_text()) == []


@pytest.mark.parametrize('mode', ['success', 'other_recording', 'all_fail', 'resource_failure', 'stale', 'supplied', 'soft',
                                'missing_language', 'null_language', 'cached_credits', 'primary_credits', 'credits_only'])
@pytest.mark.parametrize('prepared', [False, True])
def test_failed_lyric_source_retries_only_whole_eligible_texts(tmp_path, monkeypatch, mode, prepared):
    from karaoke.lyric_lookup import prepared_lyrics_identity
    primary, alternative = ['First phrase', 'Missing phrase'], ['First phrase', 'Different phrase']
    lyrics = dict(title='Evening', artist='Singer', language='English', lines=primary,
        provenance=dict(kind='user_supplied' if mode=='supplied' else 'online_lyric_text',
                        route='lyrics_first', lyric_lookup=dict(provider='primary')))
    packet = dict(candidate_texts=[primary, alternative, alternative, ['Rejected fragment']],
        crosscheck=dict(selected_source=0, eligible_sources=[0,1,2], status='conflicting',
            rejected_sources=[dict(source=3, reason='lyrics_fragment')],
            sources=[dict(provider=p,language='English') for p in ('primary','alternate','duplicate','rejected')]))
    if mode == 'other_recording':
        packet['crosscheck']['eligible_sources'] = [0]
        packet['crosscheck']['sources'][1]['recording_mismatches'] = ['duration']
    if mode == 'missing_language': packet['crosscheck']['sources'][1].pop('language')
    if mode == 'null_language': packet['crosscheck']['sources'][1]['language'] = None
    if mode == 'cached_credits': packet['candidate_texts'][1] = ['OP: A Publisher', *alternative]
    if mode == 'primary_credits':
        lyrics['lines'] = packet['candidate_texts'][0] = ['OA/OC: A Writer', *primary]
    if mode == 'credits_only':
        lyrics['lines'] = packet['candidate_texts'][0] = ['OP: A Publisher']
        packet['candidate_texts'][1:3] = [['SP: A Publisher'], ['OA/OC: A Writer']]
    if mode == 'stale':packet['candidate_texts'][0] = ['Old lyrics']
    prepared_path = None
    if prepared:
        lyrics['source_url'] = URL
        lyrics['provenance']['lyric_lookup'].update(packet)
        prepared_path = tmp_path/'prepared.json'
        prepared_path.write_text(json.dumps(dict(identity=prepared_lyrics_identity(URL, {}), lyrics=lyrics)))
    else:
        (tmp_path/'lyric-crosscheck.json').write_text(json.dumps(packet))
    (tmp_path/'lyrics.json').write_text(json.dumps(lyrics))
    pipeline = Pipeline(dict(url=URL), tmp_path, audio_only=prepared, playback_policy=prepared,
                        prepared_lyrics=prepared_path)
    if prepared:
        lyrics = pipeline.prepare_text({}, 6.)
        assert not (tmp_path/'lyric-crosscheck.json').exists()
    calls = []
    def align(candidate, duration):
        calls.append(candidate['lines'])
        assert candidate['language'] == 'English'
        if mode == 'resource_failure':raise StageFailure('align_words', 'CUDA out of memory')
        (pipeline.checkpoints.directory/'align_words.json').write_text('{}')
        cues = [dict(text=t,start=1.+i*2,end=2.+i*2,status='candidate')
                for i,t in enumerate(candidate['lines'])]
        failing = candidate['lines']==primary or mode=='all_fail'
        if failing and mode != 'soft':cues[-1] = dict(text=candidate['lines'][-1],status='unresolved_timing')
        folder = tmp_path/'alignment';folder.mkdir(exist_ok=True)
        (folder/'cues.json').write_text(json.dumps(cues))
        pipeline.alignment_outputs.add(folder)
        (tmp_path/'quality.json').write_text('{}')
        (tmp_path/'cues.json').write_text(json.dumps(cues))
        if failing:raise ValueError('Unresolved alignment')
    monkeypatch.setattr(pipeline, 'align', align)
    if mode in ('success', 'other_recording', 'soft', 'missing_language', 'null_language', 'cached_credits', 'primary_credits'):
        selected, error = pipeline.align_candidates(lyrics, {}, 6.)
        assert selected['lines'] == (primary if mode=='soft' else alternative)
        assert calls == ([primary] if mode=='soft' else [primary,alternative])
        if mode in ('success', 'other_recording', 'missing_language', 'null_language', 'cached_credits', 'primary_credits'):
            from karaoke.tools.playback_sweep import soft_failures
            quality=json.loads((tmp_path/'quality.json').read_text())
            assert soft_failures(dict(quality=quality), {}) == [dict(module='lyrics',
                code='lyric_source_fallback', rejected_sources=1, selected_provider='alternate')]
            rejected=quality['lyric_source_fallback']['rejected']
            assert len(rejected)==1 and rejected[0]['source']['provider']=='primary'
            saved=tmp_path/rejected[0]['diagnostics']
            assert json.loads((saved/'lyrics.json').read_text())['lines']==primary
            assert json.loads((saved/'playback-warning.json').read_text())['omitted_lines']==[2]
            assert json.loads((saved/'failure.json').read_text())['stage']=='subtitles'
            assert json.loads((tmp_path/'lyrics.json').read_text())['lines']==alternative
    else:
        with pytest.raises(StageFailure):pipeline.align_candidates(lyrics, {}, 6.)
        assert calls == ([] if mode=='credits_only' else [primary,alternative] if mode=='all_fail' else [primary])


@pytest.mark.parametrize('complete_source', [True, False])
@pytest.mark.parametrize('compression_ratio', [1., 3.2])
def test_observed_missing_refrain_selects_complete_published_source(tmp_path, monkeypatch, complete_source, compression_ratio):
    refrain = ['glowing red horizon', 'bright blue ocean']
    short, complete = ['opening',*refrain,'closing'], ['opening',*refrain,*refrain,'closing']
    if complete_source:
        complete.insert(-1, 'silver stars appear behind the evening clouds')
    packet = dict(candidate_texts=[short,complete], crosscheck=dict(selected_source=0,
        eligible_sources=[0,1], sources=[dict(provider=p,language='English') for p in ('short','complete')]))
    if not complete_source:
        packet = dict(candidate_texts=[short], crosscheck=dict(selected_source=0,
            eligible_sources=[0], sources=[dict(provider='short',language='English')]))
    lyrics = dict(title='Song',artist='Singer',language='English',lines=short,
        provenance=dict(kind='online_lyric_text',route='lyrics_first',lyric_lookup=packet))
    pipeline = Pipeline(dict(url=URL),tmp_path)
    (tmp_path/'lyrics.json').write_text(json.dumps(lyrics))
    raw = tmp_path/'alignment/raw.json';raw.parent.mkdir()
    heard = [1.,3.,5.,8.,10.,*([11.] if complete_source else []),12.]
    calls=[]
    def align(candidate,duration):
        calls.append(candidate['lines'])
        starts = ([1.,3.,5.,12.] if candidate['lines']==short else heard
                  if candidate['lines']==complete else [1.,3.,5.,8.,10.,12.])
        cues = [dict(text=t,start=s,end=s+.8,status='candidate') for t,s in zip(candidate['lines'],starts)]
        (tmp_path/'cues.json').write_text(json.dumps(cues))
        (tmp_path/'quality.json').write_text('{}')
        (pipeline.checkpoints.directory/'align_words.json').write_text('{}')
        raw.write_text(json.dumps([dict(asr_localizations=[dict(context=[0,len(candidate['lines'])],
            source_range=[0.,15.],recognition=dict(segments=[dict(compression_ratio=compression_ratio, words=[
                dict(word=t,start=s,end=s+.8) for t,s in zip(complete,heard)])]))])]))
    monkeypatch.setattr(pipeline,'align',align)
    selected,error=pipeline.align_candidates(lyrics,{},15.)
    assert selected['lines']==complete and error is None
    assert calls==[short,complete]
    evidence=json.loads((tmp_path/'localization-validation.json').read_text())
    assert all(not e['missing_repetitions'] and not e['conflicts'] for e in evidence)
    rejected=json.loads((tmp_path/'quality.json').read_text())['lyric_source_fallback']['rejected']
    saved=json.loads((tmp_path/rejected[0]['diagnostics']/'localization-validation.json').read_text())
    assert saved[0]['missing_repetitions'][0]['source_range']==[1,3]
    if not complete_source:
        assert selected['provenance']['repetitions'][0]['source_lines']==refrain
        assert json.loads((tmp_path/'quality.json').read_text())['restored_repetitions']==1
        # Keeping the copied words is insufficient: they must be aligned to
        # the separately heard repetition, not reused at the first occurrence.
        wrong=json.loads((tmp_path/'cues.json').read_text())
        wrong[3].update(start=3.,end=3.8)
        with pytest.raises(StageFailure,match='重复唱段'):
            pipeline.validate_localization(selected,15.,wrong)


@pytest.mark.parametrize('gap_kind', ['missing', 'covered', 'unrelated'])
def test_caption_gap_review_requires_substantial_existing_lyric_evidence(gap_kind):
    from copy import deepcopy
    from karaoke.align_lyrics_whisper import review_uncovered_lyrics
    line = 'Glowing lanterns guide us through the silent forest'
    cues = [dict(text=line,start=1.,end=3.,status='candidate'),
            dict(text='The boats return across the silver ocean',start=30.,end=34.,status='candidate')]
    if gap_kind=='covered':cues.insert(1,dict(cues[0],start=10.,end=13.))
    before=deepcopy(cues)
    words=[dict(word='zqx '*20 if gap_kind=='unrelated' else line,start=10.,end=13.)]
    calls=[]
    def recognize(a,b):
        calls.append((a,b));return dict(segments=[])
    result=review_uncovered_lyrics(cues,[words],'English',40.,recognize)
    assert cues==before
    assert bool(result)==bool(calls)==(gap_kind=='missing')
    if result:
        assert result[0]['gap']==[3.,30.]
        assert calls==[(.4,34.6)]
        assert result[0]['accepted_lines']==[]  # Review does not assign timings or lyric text.


@pytest.mark.parametrize('heard_prefix', [False, True])
def test_repeat_proposal_keeps_source_interior_between_observed_outer_phrases(heard_prefix):
    from karaoke.lyric_consensus import observed_repetition_candidate
    lines=['Blue morning (ooh)', 'We trace the river through the trees',
           'Silver lanterns carry our dreams into the night', 'Closing phrase']
    lyrics=dict(lines=lines,language='English',provenance=dict(kind='online_lyric_text'))
    cues=[dict(text=t,start=s,end=s+1,status='candidate') for t,s in zip(lines,[1.,3.,5.,28.])]
    words=[dict(word='Blue morning' if heard_prefix else 'qxz',start=10.,end=12.),
           dict(word='qxz qxz qxz',start=13.,end=16.),dict(word=lines[2],start=17.,end=21.)]
    observed=[dict(context=[0,4],source_range=[0.,30.],recognition=dict(segments=[dict(words=words)]))]
    candidate=observed_repetition_candidate(lyrics,cues,observed,30.)
    copy=candidate['provenance']['repetitions'][-1]
    assert copy['source_lines']==(lines[:3] if heard_prefix else lines[2:3])
    assert copy['boundary_anchors'][0]['start']==(10. if heard_prefix else 17.)
    assert lyrics['lines']==lines  # This only proposes original text; alignment must validate it.


@pytest.mark.parametrize('observed', ['repeat', 'unrelated', 'spelling_noise'])
@pytest.mark.parametrize('compression_ratio', [1., 3.2])
def test_published_source_retry_requires_substantial_observed_content(observed, compression_ratio):
    from karaoke.lyric_consensus import observed_source_alternative
    first, refrain, last = 'We leave beside the morning river', 'The golden lanterns guide us home', 'Until the quiet summer ends'
    lyrics = dict(lines=[first,refrain,last], language='English')
    alternative = dict(lyrics, lines=[first,refrain,refrain,last])
    performed = alternative['lines'] if observed=='repeat' else lyrics['lines']
    if observed=='unrelated':
        performed = [first,refrain,'Remember to bring the umbrellas tomorrow',last]
    if observed=='spelling_noise':
        alternative = dict(lyrics,lines=[first,refrain.replace('lanterns','lantern'),last])
    observations = [dict(source_range=[0.,20.],recognition=dict(segments=[dict(compression_ratio=compression_ratio, words=[
        dict(word=text,start=1.+i*3,end=2.+i*3) for i,text in enumerate(performed)])]))]
    observations.insert(0, dict(reason='followup_did_not_improve_coverage', accepted_lines=[]))
    result = observed_source_alternative(lyrics,[alternative],observations)
    assert bool(result) == (observed=='repeat')
    if result:
        assert result['alternative_matched_units']-result['current_matched_units'] >= 24
    assert observed_source_alternative(lyrics,[alternative],[]) is None


@pytest.mark.parametrize('overlap', [.02, .1, .16, .32, .75])
def test_playback_fallback_keeps_overlapping_word_tails_without_moving_onsets(tmp_path, overlap):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['First phrase', 'Next phrase']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='English',lines=lines,provenance={})))
    cues = [dict(text=text,start=start,end=end,status='candidate',units=[
        dict(text=text,first=0,stop=len(text),start=start,end=end)])
        for text,start,end in [(lines[0],1.,2.),(lines[1],2.-overlap,3.)]]
    (tmp_path/'word-cues.json').write_text(json.dumps(cues))
    pipeline.alignment_outputs.add(tmp_path/'word-cues.json')
    pipeline.playback_fallback({},4,StageFailure('align_words','fixture'))
    result = json.loads((tmp_path/'cues.json').read_text())
    assert [c['text'] for c in result] == lines
    assert [c['units'][0]['start'] for c in result] == [1.,2.-overlap]
    assert result[0]['end'] == result[1]['start']
    assert [c['aligned_start'] for c in result] == [1.,2.-overlap]
    warning = json.loads((tmp_path/'playback-warning.json').read_text())
    assert warning['omitted_lines'] == [] and len(warning['overlap_adjustments']) == 1


@pytest.mark.parametrize('language,middle', [('English', 'I want to sing with you now'),
    ('Japanese', 'ここでもう一度歌おう'), ('Korean', '여기서우리다시노래해')])
@pytest.mark.parametrize('seconds', [.06, .8])
def test_playback_fallback_rejects_collapsed_phrase_without_rejecting_fast_lyrics(tmp_path, seconds, language, middle):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['First phrase', middle, 'Last phrase']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language=language,lines=lines,provenance={})))
    cues = [dict(text=text,start=float(i*2),end=float(i*2)+(seconds if i==1 else 1.),status='candidate')
            for i,text in enumerate(lines)]
    (tmp_path/'alignment').mkdir()
    (tmp_path/'alignment/cues.json').write_text(json.dumps(cues))
    pipeline.alignment_outputs.add(tmp_path/'alignment')
    if seconds == .06:
        with pytest.raises(StageFailure,match='1 行歌词未能定位'):
            pipeline.playback_fallback({},6,StageFailure('align_words','fixture'))
    else:
        pipeline.playback_fallback({},6,StageFailure('align_words','fixture'))
    result=json.loads((tmp_path/'cues.json').read_text())
    assert [c['text'] for c in result] == ([lines[0],lines[2]] if seconds==.06 else lines)


@pytest.mark.parametrize('bad', ['text', 'onset_order', 'missing', 'nan'])
def test_playback_fallback_omits_only_phrases_without_usable_times(tmp_path, bad):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(title='Example', artist='', language='English',
        lines=['First line', 'Next line'], provenance={})))
    (tmp_path/'alignment').mkdir()
    cues = [dict(text=text, start=start, end=end, status='candidate')
            for text, start, end in [('First line', 0, 1), ('Next line', 2, 3)]]
    if bad == 'text': cues[0]['text'] = 'Wrong lyric'
    if bad == 'onset_order': cues[1]['start'] = 0.
    if bad == 'missing': cues[0]['start'] = None
    if bad == 'nan': cues[0]['start'] = float('nan')
    (tmp_path/'alignment/cues.json').write_text(json.dumps(cues))
    pipeline.alignment_outputs.add(tmp_path/'alignment')
    with pytest.raises(StageFailure, match='1 行歌词未能定位'):
        pipeline.playback_fallback({}, 4, StageFailure('align_words', 'unresolved'))
    output = json.loads((tmp_path/'cues.json').read_text())
    retained = 0 if bad == 'onset_order' else 1
    assert len(output) == 1 and output[0]['text'] == cues[retained]['text']
    assert output[0]['start'] == cues[retained]['start'] and output[0]['end'] == cues[retained]['end']
    warning = json.loads((tmp_path/'quality.json').read_text())['playback_fallback']
    assert warning['subtitle_mode'] == 'estimated_word'
    assert warning['omitted_lines'] == [2 if bad == 'onset_order' else 1]


@pytest.mark.parametrize('bad', ['merged', 'text', 'span', 'nan', 'outside', 'backward'])
def test_playback_fallback_preserves_other_lines_highlights(tmp_path, bad):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['一起唱', '今夜']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(title='Example', artist='', language='Chinese',
        lines=lines, provenance={})))
    cues = [dict(text=text, start=start, end=start+1, status='candidate',
                 units=[dict(text=char, first=i, stop=i+1, start=start+i*.2, end=start+(i+1)*.2)
                        for i, char in enumerate(text)]) for text, start in zip(lines, [0, 2])]
    if bad == 'merged': cues[0]['units'] = [dict(text=lines[0], first=0, stop=3, start=0, end=1)]
    if bad == 'text': cues[0]['units'][0]['text'] = '错'
    if bad == 'span': cues[0]['units'][0]['stop'] = 2
    if bad == 'nan': cues[0]['units'][0]['start'] = float('nan')
    if bad == 'outside': cues[0]['units'][0]['start'] = -.1
    if bad == 'backward': cues[0]['units'][0]['start'] = .3
    pipeline.aligned_cues = cues
    pipeline.playback_fallback({}, 4, StageFailure('align_words', 'unresolved'))
    output = json.loads((tmp_path/'cues.json').read_text())
    assert output[0]['units'][0]['timing_source'] == 'linear_fallback'
    if bad != 'merged':
        assert output[0]['units'][1:] == cues[0]['units'][1:]
        assert output[0]['units'][0]['start'] == 0
        assert output[0]['units'][0]['end'] == .2
    assert output[1] == cues[1]
    quality = json.loads((tmp_path/'quality.json').read_text())
    assert quality['units'] == (3 if bad == 'merged' else 5)
    assert quality['playback_fallback']['subtitle_mode'] == 'estimated_word'
    assert [s['line'] for s in quality['playback_fallback']['estimated_spans']] == [1]
    assert quality['playback_fallback']['unhighlighted_lines'] == []
    assert quality['playback_fallback']['word_alignment_passed'] is False
    assert quality['onset_accuracy'] is None


@pytest.mark.parametrize('old_time', [True, False])
@pytest.mark.parametrize('strict_audio', [True, False])
@pytest.mark.parametrize('need_lead', [None, False, True])
def test_publication_rechecks_acoustic_constraints_instead_of_trusting_a_success_flag(tmp_path, monkeypatch, old_time, strict_audio, need_lead):
    pipeline = Pipeline(dict(url=URL), tmp_path, device='cpu', audio_only=strict_audio)
    lyrics = dict(lines=['hello'], language='English')
    cue = dict(text='hello', start=1., end=5., units=[dict(text='hello', first=0, stop=5, start=1., end=5.)])
    (tmp_path/'cues.json').write_text(json.dumps([cue]))
    (tmp_path/'quality.json').write_text('{}')
    calls = []
    lead = Mock() if need_lead is not None else None
    master = tmp_path/'stems/lead.wav'
    def finish_packaging():
        master.parent.mkdir(exist_ok=True)
        master.write_bytes(b'delivered lead after native-rate conversion')
    if lead is not None:
        lead.result.side_effect = finish_packaging
    def run(*args):
        calls.append(args[1])
        assert ('--playback-fallback' in args[1]) is not strict_audio
        assert ('--defer-constraints' in args[1]) is (lead is not None and len(calls)==1)
        assert ('--lead' in args[1]) is (len(calls)==2)
        if lead is not None and len(calls)==1:
            lead.result.assert_not_called()
            assert not master.exists()
        if len(calls)==2:
            source = args[1][args[1].index('--lead')+1]
            assert source == master and source.read_bytes() == b'delivered lead after native-rate conversion'
        output=tmp_path/'timing-checked';output.mkdir(exist_ok=True)
        checked=json.loads(json.dumps(cue))
        if not old_time and (not need_lead or len(calls)==2):checked['units'][0]['start']=4.
        if not strict_audio:
            checked['units'][0].update(timing_source='linear_fallback',
                                      bounds_source='cross_model_onset_prior_acoustic_end')
        (output/'cues.json').write_text(json.dumps([checked]))
        (output/'report.json').write_text(json.dumps(dict(
            pending_review_lines=[1] if need_lead and len(calls)==1 else [],
            lines=[dict(line=1, status='reviewed', repairs=[]),
                   dict(line=1, status='reviewed', repairs=[dict(accepted=True)])], unresolved=[],constraints=[
            dict(line=1,text='hello',first=0,stop=5,start_min=3.6,start_max=4.4,onset=4.)])))
    monkeypatch.setattr(pipeline,'run',run)
    if old_time:
        with pytest.raises(StageFailure, match='字幕时间校验失败'):
            pipeline.validate_alignment(lyrics,6,lead)
        assert json.loads((tmp_path/'cues.json').read_text())==[cue]
    else:
        pipeline.validate_alignment(lyrics,6,lead)
        assert json.loads((tmp_path/'cues.json').read_text())[0]['units'][0]['start']==4.
        review = json.loads((tmp_path/'quality.json').read_text())['timing_review']
        assert len(review['onset_constraints']) == review['reviewed_lines'] == review['repaired_ranges'] == 1
        assert review['review_attempts'] == 2
        if not strict_audio:
            from karaoke.tools.playback_sweep import soft_failures
            assert len(review['estimated_spans'])==1
            assert soft_failures(dict(quality=dict(timing_review=review)),{})[0]['code']=='timing_group_fallback'
    assert len(calls) == (2 if need_lead else 1)
    if lead is not None:
        assert lead.result.call_count == int(need_lead)


@pytest.mark.parametrize('displaced', [False, True])
def test_publication_checks_observed_lyrics_even_after_playback_fallback(tmp_path, monkeypatch, displaced):
    pipeline = Pipeline(dict(url=URL), tmp_path, device='cpu')
    lyrics = dict(lines=['blue river'], language='English')
    start = 30. if displaced else 20.
    cue = dict(text='blue river', start=start, end=start+4., units=[
        dict(text='blue', first=0, stop=4, start=start, end=start+2., timing_source='linear_fallback', acoustic_end=None),
        dict(text='river', first=5, stop=10, start=start+2., end=start+4., timing_source='linear_fallback')])
    (tmp_path/'cues.json').write_text(json.dumps([cue]))
    (tmp_path/'quality.json').write_text('{}')
    raw = tmp_path/'alignment/raw.json'; raw.parent.mkdir()
    raw.write_text(json.dumps([dict(asr_localizations=[dict(context=[0,1], source_range=[10.,40.],
        recognition=dict(segments=[dict(words=[dict(word='blue river', start=10., end=14.)])]))])]))
    def run(*args):
        output = tmp_path/'timing-checked'; output.mkdir()
        (output/'cues.json').write_text(json.dumps([cue]))
        (output/'report.json').write_text(json.dumps(dict(lines=[], unresolved=[], constraints=[])))
    monkeypatch.setattr(pipeline, 'run', run)
    if displaced:
        with pytest.raises(StageFailure, match='字幕与已定位的唱段不一致'):
            pipeline.validate_alignment(lyrics, 40.)
    else:
        pipeline.validate_alignment(lyrics, 40.)
    evidence = json.loads((tmp_path/'localization-validation.json').read_text())
    assert evidence[0]['checked_lines'] == [1]
    assert bool(evidence[0]['conflicts']) is displaced
    assert json.loads((tmp_path/'cues.json').read_text()) == [cue]


@pytest.mark.parametrize('displaced', [False, True])
def test_phrase_validation_reconciles_dialogue_with_ordered_recording_views(tmp_path, displaced):
    lines = ['A long opening phrase about the morning', 'Can you hear me', 'Another long phrase about the evening']
    lyrics = dict(lines=lines, language='English')
    cues = [dict(text=t,start=s,end=s+2) for t,s in zip(lines,[40.,80. if displaced else 50.,60.])]
    primary = [dict(word=lines[1],start=5.,end=7.), *[
        dict(word=t,start=s,end=s+2.) for t,s in zip(lines,[40.,50.,60.])]]
    alternate = [dict(word=lines[1],start=5.,end=7.),dict(word=lines[2],start=60.,end=62.)]
    raw = tmp_path/'alignment/raw.json';raw.parent.mkdir()
    raw.write_text(json.dumps([dict(asr_localizations=[dict(context=[0,3],source_range=[0.,90.],
        recognition=dict(segments=[dict(words=words)])) for words in [primary,alternate]])]))
    pipeline = Pipeline(dict(url=URL),tmp_path)
    if displaced:
        with pytest.raises(StageFailure,match='字幕与已定位的唱段不一致'):
            pipeline.validate_localization(lyrics,90.,cues)
    else:
        evidence = pipeline.validate_localization(lyrics,90.,cues)
        assert not any(e['conflicts'] for e in evidence)
        assert evidence[1]['competing_view_conflicts'][0]['observed_range']==[5.,7.]
        assert 2 in evidence[0]['joint_view_validation']['supported_lines']


@pytest.mark.parametrize('extra_frames', [0, 4])
def test_timing_review_accepts_one_sample_of_resampling_but_rejects_different_lengths(tmp_path, monkeypatch, extra_frames):
    import soundfile as sf
    from karaoke import timing_validation as module
    cue = dict(text='hello', start=.1, end=.3,
               units=[dict(text='hello', first=0, stop=5, start=.1, end=.3)])
    (tmp_path/'cues.json').write_text(json.dumps([cue]))
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(lines=['hello'], language='English')))
    audio, lead = tmp_path/'audio.wav', tmp_path/'lead.wav'
    sf.write(audio, np.ones((44101, 2))*.05, 44100, subtype='FLOAT')
    sf.write(lead, np.ones((48000+extra_frames, 2))*.05, 48000, subtype='DOUBLE')
    review = Mock()
    monkeypatch.setattr(module, 'AcousticReview', lambda *args: review)
    monkeypatch.setattr(module, 'acoustic_model_files', lambda *args: [])
    def reconcile(cues, observations, *args, **kwargs):
        assert len(list(observations)) == 2
        return cues, dict(lines=[], constraints=[], unresolved=[])
    monkeypatch.setattr(module, 'reconcile_views', reconcile)
    monkeypatch.setattr(sys, 'argv', ['timing_validation', '--input', str(tmp_path/'cues.json'),
        '--lyrics', str(tmp_path/'lyrics.json'), '--audio', str(audio), '--original', str(audio),
        '--lead', str(lead), '--output', str(tmp_path/'checked'), '--device', 'cpu'])
    if extra_frames:
        with pytest.raises(ValueError, match='does not match the source duration'):
            module.main()
    else:
        module.main()
        assert json.loads((tmp_path/'checked/cues.json').read_text()) == [cue]


@pytest.mark.parametrize('changed', [None, 'code', 'models', 'protocol', 'legacy', 'lead'])
def test_timing_constraints_reuse_only_the_same_evidence_method(tmp_path, monkeypatch, changed):
    import soundfile as sf
    from karaoke import timing_validation as module
    cue = dict(text='hello', start=.1, end=.3,
               units=[dict(text='hello', first=0, stop=5, start=.1, end=.3)])
    (tmp_path/'cues.json').write_text(json.dumps([cue]))
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(lines=['hello'], language='English')))
    audio = tmp_path/'audio.wav'
    sf.write(audio, np.ones((44100, 2))*.05, 44100, subtype='FLOAT')
    proof = dict(line=1, text='hello', first=0, stop=5, onset=.1, start_min=.05, start_max=.15)
    seen = []
    def reconcile(cues, observations, protocol, language, constraints, **kwargs):
        seen.append(constraints)
        return cues, dict(lines=[], constraints=[proof], unresolved=[])
    monkeypatch.setattr(module, 'AcousticReview', Mock())
    monkeypatch.setattr(module, 'acoustic_model_files', lambda *args: [])
    monkeypatch.setattr(module, 'reconcile_views', reconcile)
    monkeypatch.setattr(sys, 'argv', ['timing_validation', '--input', str(tmp_path/'cues.json'),
        '--lyrics', str(tmp_path/'lyrics.json'), '--audio', str(audio), '--original', str(audio),
        '--output', str(tmp_path/'checked'), '--device', 'cpu'])
    module.main()
    path = tmp_path/'timing-constraints.json'
    old = json.loads(path.read_text())
    if changed == 'lead':
        old['lead_sha256'] = 'different-audio'
    elif changed == 'legacy':
        old.pop('identity')
    elif changed:
        old['identity'][changed] = {'earlier_method': True}
    path.write_text(json.dumps(old))
    module.main()
    assert seen == [[], [proof] if changed is None else []]
    if changed:
        archived = list((tmp_path/'timing-history').glob('*.json'))
        assert len(archived) == 1 and json.loads(archived[0].read_text()) == old
    else:
        assert not (tmp_path/'timing-history').exists()


def test_playback_fallback_recovers_each_phrase_from_available_stages(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['第一句', '第二句', '第三句']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='Chinese', lines=lines, provenance={})))
    coarse = [dict(text=text, start=2*i, end=2*i+1, status='candidate') for i, text in enumerate(lines)]
    (tmp_path/'alignment').mkdir()
    (tmp_path/'alignment/cues.json').write_text(json.dumps(coarse))
    refined = [dict(c, units=[dict(text=x, first=i, stop=i+1, start=c['start']+.2*i,
                                   end=c['start']+.2*(i+1)) for i, x in enumerate(c['text'])]) for c in coarse]
    refined[1]['start'] = None
    (tmp_path/'word-cues.json').write_text(json.dumps(refined))
    pipeline.alignment_outputs.update([tmp_path/'word-cues.json', tmp_path/'alignment'])
    # A later stage left by an earlier attempt was not revalidated this run.
    (tmp_path/'context/song').mkdir(parents=True)
    stale = [dict(c, start=c['start']+.1) for c in coarse]
    (tmp_path/'context/song/cues.json').write_text(json.dumps(stale))
    pipeline.playback_fallback({}, 6, StageFailure('align_phonemes', 'fixture'))
    output = json.loads((tmp_path/'cues.json').read_text())
    assert output[0] == refined[0] and output[2] == refined[2]
    assert output[1]['units'][0]['timing_source'] == 'linear_fallback'
    assert output[1]['start'] == 2 and output[1]['end'] == 3
    warning = json.loads((tmp_path/'playback-warning.json').read_text())
    assert warning['omitted_lines'] == []
    assert [x['source'] for x in warning['phrase_timing_sources']] == [
        'word-cues.json', 'alignment/cues.json', 'word-cues.json']


@pytest.mark.parametrize('missing_last', [False, True])
def test_playback_fallback_backtracks_refinements_that_block_later_lyrics(tmp_path, missing_last):
    from copy import deepcopy
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['甲乙', '丙丁', '戊己']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='Chinese',lines=lines,provenance={})))
    def cue(text, a, b):
        return dict(text=text,start=a,end=b+.2,status='candidate',units=[
            dict(text=c,first=i,stop=i+1,start=t,end=t+.2) for i,(c,t) in enumerate(zip(text,[a,b]))])
    coarse = [cue(t,s,s+.4) for t,s in zip(lines,[1.,3.,5.])]
    refined = [cue(lines[0],4.2,4.6),cue(lines[1],4.4,5.1),deepcopy(coarse[2])]
    if missing_last:
        coarse[2]['status'] = refined[2]['status'] = 'unresolved_timing'
    for name,values in [('word-cues.json',coarse),('ordered-cues.json',refined)]:
        path=tmp_path/name;path.write_text(json.dumps(values));pipeline.alignment_outputs.add(path)
    if missing_last:
        with pytest.raises(StageFailure,match='未能定位'):
            pipeline.playback_fallback({},8,StageFailure('align_phonemes','fixture'))
    else:
        pipeline.playback_fallback({},8,StageFailure('align_phonemes','fixture'))
        assert json.loads((tmp_path/'cues.json').read_text()) == coarse
        warning=json.loads((tmp_path/'playback-warning.json').read_text())
        assert warning['omitted_lines'] == []
        assert [p['source'] for p in warning['phrase_timing_sources']] == [
            'word-cues.json','word-cues.json','ordered-cues.json']


@pytest.mark.parametrize('shift', [.1, 2.])
def test_failed_refinement_cannot_take_priority_in_an_unrelated_gap(tmp_path, shift):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='English',lines=['first','last'],provenance={})))
    def cue(text,start):
        return dict(text=text,start=start,end=start+.5,status='candidate',
                    units=[dict(text=text,first=0,stop=len(text),start=start,end=start+.5)])
    coarse=[cue('first',1.),cue('last',5.)];refined=[cue('first',1.+shift),cue('last',5.)]
    for name,values in [('word-cues.json',coarse),('ordered-cues.json',refined)]:
        path=tmp_path/name;path.write_text(json.dumps(values));pipeline.alignment_outputs.add(path)
    pipeline.playback_fallback({},8.,StageFailure('align_phonemes','fixture'))
    actual=json.loads((tmp_path/'cues.json').read_text())
    assert actual == (refined if shift<.5 else coarse)


@pytest.mark.parametrize('fresh', [True, False])
@pytest.mark.parametrize('missing_words', [True, False])
def test_playback_fallback_retains_raw_aligned_lines_when_whole_block_text_check_fails(tmp_path, fresh, missing_words):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['First phrase', 'Missing phrase', 'Last phrase']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines, provenance={})))
    (tmp_path/'alignment').mkdir()
    raw = [dict(range=[0,3], context=[0,3], source_range=[10,20], result=dict(segments=[
        dict(words=[dict(word=text, start=i*2, end=i*2+1)])
        for i,text in enumerate(['First phrase','incorrect model text','Last phrase'])]))]
    if missing_words:
        raw[0]['result']['segments'][1].pop('words')
    (tmp_path/'alignment/raw.json').write_text(json.dumps(raw))
    if fresh:
        pipeline.alignment_outputs.add(tmp_path/'alignment')
    if fresh:
        with pytest.raises(StageFailure, match='1 行歌词未能定位'):
            pipeline.playback_fallback({}, 25, StageFailure('align_words', 'block failed'))
    else:
        with pytest.raises(StageFailure, match='未生成任何可用字幕'):
            pipeline.playback_fallback({}, 25, StageFailure('align_words', 'block failed'))
    output = json.loads((tmp_path/'cues.json').read_text())
    assert [c['text'] for c in output] == ([lines[0], lines[2]] if fresh else [])
    assert [c['start'] for c in output] == ([10, 14] if fresh else [])
    warning = json.loads((tmp_path/'playback-warning.json').read_text())
    assert warning['omitted_lines'] == ([2] if fresh else [1,2,3])
    assert not warning['word_alignment_passed']


@pytest.mark.parametrize('contradicted', [False, True])
@pytest.mark.parametrize('earlier_result', [False, True])
def test_playback_fallback_prefers_existing_word_times_to_coarse_lines(tmp_path, contradicted, earlier_result):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    lines = ['First phrase', 'Last phrase']
    (tmp_path/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines, provenance={})))
    (tmp_path/'alignment').mkdir()
    coarse = [dict(text=text, start=10+i*2, end=11+i*2, status='candidate') for i, text in enumerate(lines)]
    (tmp_path/'alignment/cues.json').write_text(json.dumps(coarse))
    raw = [dict(range=[0, 2], context=[0, 2], source_range=[10, 15], result=dict(segments=[
        dict(words=[dict(word=w, start=i*2+j*.4, end=i*2+j*.4+.3)
                    for j,w in enumerate(text.split())]) for i,text in enumerate(lines)]))]
    raw[0]['asr_localizations'] = [dict(source_range=[10,15], accepted_lines=[] if contradicted else [2],
                                      contradicted_lines=[2],
                                      recognition=dict(segments=[]))]
    if earlier_result:
        raw[0]['asr_localizations'].insert(0, dict(source_range=[10,15],
            accepted_lines=[2] if contradicted else [], contradicted_lines=[2],
            recognition=dict(segments=[])))
    (tmp_path/'alignment/raw.json').write_text(json.dumps(raw))
    pipeline.alignment_outputs.add(tmp_path/'alignment')
    if contradicted:
        with pytest.raises(StageFailure, match='1 行歌词未能定位'):
            pipeline.playback_fallback({}, 20, StageFailure('align_words', 'unresolved'))
    else:
        pipeline.playback_fallback({}, 20, StageFailure('align_words', 'unresolved'))
    cues = json.loads((tmp_path/'cues.json').read_text())
    assert [u['start'] for c in cues for u in c['units']] == ([10., 10.4] if contradicted else [10., 10.4, 12., 12.4])
    warning = json.loads((tmp_path/'playback-warning.json').read_text())
    assert warning['estimated_spans'] == [] and warning['omitted_lines'] == ([2] if contradicted else [])
    assert warning['requires_review'] == contradicted
    quality = json.loads((tmp_path/'quality.json').read_text())
    assert quality['asr_localization_windows'] == 1
    assert quality['recovered_alignment_lines'] == ([] if contradicted else [2])
    assert quality['unresolved_localization_lines'] == ([2] if contradicted else [])
    from karaoke.tools.playback_sweep import soft_failures
    events = soft_failures(dict(quality=quality),{})
    assert next(e for e in events if e['code']=='alignment_localized_with_asr')['windows'] == 1


@pytest.mark.parametrize('long_credits', [False, True])
def test_intro_fits_long_metadata_without_truncating_lyrics(monkeypatch, long_credits):
    from karaoke import subtitle_style as style
    info = dict(title='夜空の歌', artist='歌手', version='Studio')
    if long_credits:
        info = {key: value*40 for key, value in info.items()}
    original = dict(info)
    caption = style.caption_image
    accepted = []
    def record(text, **kwargs):
        image = caption(text, **kwargs)
        accepted.append(text)
        assert image.getbbox() is not None
        return image
    monkeypatch.setattr(style, 'caption_image', record)
    image = style.song_intro_image(info, width=640)
    assert image.mode == 'RGBA' and image.getbbox() is not None
    assert image.getpixel((0, 0))[3] == 0 and info == original
    if long_credits:
        assert any(text.endswith('…') and text != '…' for text in accepted)
        with pytest.raises(style.CaptionOverflow):
            caption(info['title'], width=640)
    else:
        assert accepted == [info['title'], '演唱 · '+info['artist'], info['version']]


@pytest.mark.parametrize('info', ['{invalid', '{}', json.dumps(dict(title='\U0010ffff')), 'font_failure'])
def test_auxiliary_intro_and_long_preview_cannot_abort_valid_lyric_video(tmp_path, info, monkeypatch):
    from pathlib import Path
    source, output = tmp_path/'source.mp4', tmp_path/'output.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=30:d=0.4',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.4', '-c:v', 'libx264', '-c:a', 'aac', str(source)], check=True)
    cues = [dict(text=text, start=i*.2, end=(i+1)*.2,
                 units=[dict(first=0, stop=len(text), start=i*.2, end=(i+1)*.2)])
            for i, text in enumerate(['Hello', '星'*42])]
    (tmp_path/'cues.json').write_text(json.dumps(cues))
    import karaoke.render_lyric_progress as renderer
    if info == 'font_failure':
        info = json.dumps(dict(title='Example'))
        def broken_font(*args, **kwargs):
            raise StopIteration('Optional font not found')
        monkeypatch.setattr(renderer, 'song_intro_image', broken_font)
    (tmp_path/'info.json').write_text(info)
    monkeypatch.setattr(sys, 'argv', [str(Path(__file__).with_name('render_lyric_progress.py')),
        '--video', str(source), '--cues', str(tmp_path/'cues.json'), '--song-info', str(tmp_path/'info.json'),
        '--language', 'Japanese', '--duration', '0.4', '--bottom-margin', '32', '--output', str(output)])
    renderer.main()
    report = json.loads(output.with_suffix('.render.json').read_text())
    assert report['song_intro']['status'] == 'omitted' and report['song_intro']['reason']
    assert report['shown_lines'] == [1, 2] and report['decode_check'] == 'passed'
    assert json.loads((tmp_path/'cues.json').read_text()) == cues


@pytest.mark.parametrize('radius', [1, 3, 6])
def test_caption_outline_preserves_antialiasing_and_canvas_edges(radius):
    from PIL import Image, ImageDraw, ImageFilter
    from karaoke.subtitle_style import outline_mask
    for region in (None, (17, 13, 47, 26), (0, 0, 7, 9), (56, 31, 63, 39)):
        mask = Image.new('L', (64, 40))
        if region:
            draw = ImageDraw.Draw(mask)
            draw.rectangle(region, fill=173)
            draw.point(region[:2], fill=255)
        assert outline_mask(mask, radius).tobytes() == mask.filter(
            ImageFilter.MaxFilter(2*radius+1)).tobytes()


@pytest.mark.parametrize('timing_source', [None, 'cross_model_phonemes', 'linear_fallback'])
def test_linear_playback_sweeps_locally_and_preserves_measured_onsets(monkeypatch, timing_source):
    from PIL import Image
    import karaoke.render_lyric_progress as renderer
    def caption(text, *, layout_out, **kwargs):
        # The middle span crosses two display rows. All glyphs are visible white
        # squares so exact pixels distinguish gradual fill from a whole-word jump.
        layout_out.update(mask=Image.new('L', (60, 50), 255), characters=[
            dict(index=i, left=3+20*(i%2), right=13+20*(i%2), top=3+25*(i//2),
                 baseline=13+25*(i//2), line=i//2, character=x) for i, x in enumerate(text)])
        return Image.new('RGBA', (60, 50), (255, 255, 255, 255))
    monkeypatch.setattr(renderer, 'caption_image', caption)
    cue = dict(text='甲乙丙丁', units=[
        dict(first=0, stop=1, start=1., end=2.),
        dict(first=1, stop=3, start=2., end=4., **({'timing_source':timing_source} if timing_source else {})),
        dict(first=3, stop=4, start=5., end=6.),
    ])
    caption = renderer.Caption(cue, 'cinema_focus', 'Chinese')
    white, amber = (255, 255, 255, 255), caption.color+(255,)
    assert caption.frame(.9, 'word').getpixel((5, 5)) == white
    assert caption.frame(1., 'word').getpixel((5, 5)) == white
    partial = caption.frame(1.5, 'word')
    assert partial.getpixel((5, 5)) == amber and partial.getpixel((12, 5)) == white
    assert caption.frame(1.99, 'word').getpixel((12, 5)) == amber
    assert caption.frame(2., 'word').getpixel((25, 5)) == white
    middle = caption.frame(3., 'word')
    assert middle.getpixel((25, 5)) == amber and middle.getpixel((5, 30)) == white
    assert caption.frame(4., 'word').getpixel((5, 30)) == amber
    assert caption.frame(4.9, 'word').getpixel((25, 30)) == white
    assert caption.frame(5., 'word').getpixel((25, 30)) == white
    assert caption.frame(6., 'word').getpixel((25, 30)) == amber
    # Seeking back must not reuse the cached fully highlighted frame.
    assert caption.frame(2., 'word').getpixel((25, 5)) == white
    assert caption.frame(3., 'word').tobytes() == middle.tobytes()
    assert caption.frame(1.5, 'word').tobytes() == partial.tobytes()
    assert caption.frame(3., 'static').getpixel((5, 5)) == white
    assert len(caption.cache) <= 3

    # A later onset must not finish a still-sounding earlier unit. Nor may a
    # completed later unit freeze that earlier sweep in the frame cache.
    overlapping = renderer.Caption(dict(cue, units=[dict(cue['units'][0], end=5.), *cue['units'][1:]]),
                                   'cinema_focus', 'Chinese')
    assert overlapping.frame(2., 'word').getpixel((12, 5)) == white
    assert overlapping.frame(4., 'word').getpixel((12, 5)) == white
    assert overlapping.frame(4.9, 'word').getpixel((12, 5)) == amber
    assert overlapping.frame(4., 'word').getpixel((12, 5)) == white


@pytest.mark.parametrize('spaced', [False, True])
def test_long_lyric_pages_preserve_all_text_and_only_switch_at_measured_onsets(spaced):
    import copy
    from karaoke.render_lyric_progress import Caption
    text = '春日暖阳照过山川河流'*16
    if spaced:
        text = '（ '+' '.join(text)+' ）'
    cue = dict(text=text, units=[dict(first=i, stop=i+1, text=c, start=1+i*.2, end=1.2+i*.2)
                                  for i,c in enumerate(text) if c.isalnum()])
    before = copy.deepcopy(cue)
    caption = Caption(cue,'cinema_focus','Chinese')
    assert len(caption.pages) > 1 and cue == before
    assert ''.join(p.cue['text'] for p in caption.pages) == text
    assert [u['start'] for p in caption.pages for u in p.cue['units']] == [u['start'] for u in cue['units']]
    for i,page in enumerate(caption.pages):
        start = page.starts[0]
        assert caption.frame(start,'word').tobytes() == page.frame(start,'word').tobytes()
        if i:
            assert caption.frame(start-.001,'word').tobytes() == caption.pages[i-1].frame(start-.001,'word').tobytes()
    assert caption.frame(0,'word').tobytes() == caption.pages[0].base.tobytes()
    second = caption.pages[1]
    assert caption.frame(0,'word',page_at=second.starts[0]).tobytes() == second.base.tobytes()


def test_playback_fallback_keeps_unverified_asr_draft_out_of_subtitles(tmp_path):
    pipeline = Pipeline(dict(url=URL), tmp_path)
    draft = dict(title='Example', artist='', language='Chinese', lines=['猜測的文字'],
                 provenance=dict(kind='asr_draft', needs_review=True))
    (tmp_path/'lyrics.json').write_text(json.dumps(draft))
    cue = dict(text=draft['lines'][0], start=0, end=2, status='candidate', units=[])
    pipeline.aligned_cues = [cue]
    (tmp_path/'alignment').mkdir()
    (tmp_path/'alignment/cues.json').write_text(json.dumps([cue]))
    (tmp_path/'anchors.json').write_text(json.dumps([cue]))
    with pytest.raises(StageFailure, match='未生成任何可用字幕'):
        pipeline.playback_fallback({}, 3, StageFailure('align_words', 'unresolved'))
    assert json.loads((tmp_path/'cues.json').read_text()) == []
    assert json.loads((tmp_path/'lyrics.json').read_text()) == draft
    warning = json.loads((tmp_path/'playback-warning.json').read_text())
    assert warning['subtitle_mode'] == 'unavailable' and warning['unverified_asr_text_omitted']


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
        track = (tone*scale).astype('float64')
        if name == 'lead':
            track[-1] += 1e-16  # A harmless rounding difference must not block acceptance.
        sf.write(artifacts/f'{name}.wav', track, sample_rate, subtype='DOUBLE')
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
    monkeypatch.setattr(pipeline, 'prepare_text', lambda *args: dict(
        title='Example', artist='', language='English', lines=['Hello']))
    monkeypatch.setattr(pipeline, 'align', lambda *args: stages.append('align'))
    monkeypatch.setattr(pipeline, 'validate_alignment', lambda *args: stages.append('validate_timing'))
    monkeypatch.setattr(pipeline, 'render_video', lambda *args: pytest.fail('Audio-only test rendered a video'))
    result=pipeline.execute()
    assert {'separate_music','separate_lead','package_three_stems','align','validate_timing'}<=set(stages)
    assert result['validation_scope']=='audio_and_lyrics' and 'mv' not in result['artifacts']
    assert not (folder/'result.json').exists()
    evidence=verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    assert evidence['validation_scope']=='audio_and_lyrics' and len(evidence['artifact_sha256'])==7
    assert evidence['audio_format'] == dict(sample_rate=sample_rate, channels=1, frames=len(tone))
    (folder/'lyrics.json').write_text(json.dumps(dict(lines=['Hello'])))
    playback = verify_outputs(None, '', tmp_path, {'id':'job'}, audio_only=True, playback_policy=True)
    assert playback['validation_scope'] == 'playback_audio'
    # A recorded timing fallback does not authorize dropping lyric lines.
    result['quality']['playback_fallback'] = dict(subtitle_mode='estimated_word')
    (folder/'audio-result.json').write_text(json.dumps(result))
    assert verify_outputs(None, '', tmp_path, {'id':'job'}, audio_only=True,
                          playback_policy=True)['subtitle_lines'] == 1
    (folder/'lyrics.json').write_text(json.dumps(dict(lines=['Hello', 'Missing verse'])))
    with pytest.raises(ValueError, match='omit supplied lyric text'):
        verify_outputs(None, '', tmp_path, {'id':'job'}, audio_only=True, playback_policy=True)
    result['quality'].pop('playback_fallback')
    (folder/'lyrics.json').write_text(json.dumps(dict(lines=['Hello'])))
    (folder/'audio-result.json').write_text(json.dumps(result))
    original_master = (artifacts/'lead.wav').read_bytes()
    (artifacts/'lead.wav').write_bytes(original_master[:-1] + bytes([original_master[-1] ^ 1]))
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True,playback_policy=True)
    (artifacts/'lead.wav').write_bytes(original_master)
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
    subtitle_path.write_text('[]')
    result['artifacts']['subtitles'].update(bytes=subtitle_path.stat().st_size, sha256=file_hash(subtitle_path))
    result['quality']['playback_fallback'] = dict(subtitle_mode='unavailable')
    (folder/'audio-result.json').write_text(json.dumps(result))
    for policy in (True, False):
        with pytest.raises(ValueError, match='No usable subtitles'):
            verify_outputs(None, '', tmp_path, {'id':'job'}, audio_only=True, playback_policy=policy)
    result['quality'].pop('playback_fallback')
    subtitle_path.write_text(json.dumps([dict(text='Hello world', start=.1, end=.12,
        units=[dict(text='Hello world', first=0, stop=11, start=.1, end=.12)])]))
    result['artifacts']['subtitles'].update(bytes=subtitle_path.stat().st_size, sha256=file_hash(subtitle_path))
    (folder/'audio-result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='independent highlights'):
        verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True)
    subtitle_path.write_text(json.dumps([dict(text='We can sing together', start=.1, end=.16,
        units=[dict(text='We can sing together', first=0, stop=20, start=.1, end=.16)])]))
    result['artifacts']['subtitles'].update(bytes=subtitle_path.stat().st_size, sha256=file_hash(subtitle_path))
    (folder/'audio-result.json').write_text(json.dumps(result))
    for policy in (True, False):
        with pytest.raises(ValueError, match='Collapsed lyric phrase'):
            verify_outputs(None,'',tmp_path,{'id':'job'},audio_only=True,playback_policy=policy)
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


@pytest.mark.parametrize('cache_state', ['valid', 'missing', 'corrupt', 'different_source', 'bad_json', 'bad_metadata', 'missing_field'])
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
        if cache_state == 'bad_json':
            (tmp_path/'source.json').write_text('{')
        elif cache_state == 'bad_metadata':
            (tmp_path/'download/source.info.json').write_text('[]')
        elif cache_state == 'missing_field':
            (tmp_path/'source.json').write_text(json.dumps(dict(url=URL)))
    if cache_state == 'valid':
        assert pipeline.download() == ({}, 10)
    else:
        with pytest.raises(StageFailure, match='离线处理'):
            pipeline.download()
    if cache_state != 'missing':
        assert pipeline.source.read_bytes() == b'cached source'


@pytest.mark.parametrize('broken_file', ['source.json','download/source.info.json'])
def test_invalid_download_metadata_is_rebuilt_without_reusing_wrong_source(tmp_path, monkeypatch, broken_file):
    pipeline = Pipeline(dict(url=URL),tmp_path,audio_only=True)
    (tmp_path/'download').mkdir()
    pipeline.source.write_bytes(b'old source')
    (tmp_path/'source.json').write_text(json.dumps(dict(url=URL,sha256=file_hash(pipeline.source),duration=2)))
    (tmp_path/'download/source.info.json').write_text('{}')
    (tmp_path/broken_file).write_text('{')
    calls = []
    def download(*args,**kwargs):
        calls.append('download')
        pipeline.source.write_bytes(b'new source')
        (tmp_path/'download/source.info.json').write_text(json.dumps(dict(title='New source',duration=2)))
    monkeypatch.setattr(pipeline,'run',download)
    monkeypatch.setattr('karaoke.pipeline.subprocess.check_output',lambda *a,**k:json.dumps(
        dict(streams=[dict(codec_type='audio')],format=dict(duration=2))).encode())
    assert pipeline.download()[1] == 2
    assert pipeline.download()[1] == 2 and calls == ['download']
    assert pipeline.source.read_bytes() == b'new source'
    assert json.loads((tmp_path/'download-cache-error.json').read_text())['error'] == 'JSONDecodeError'


def test_context_refinement_keeps_tightly_spaced_but_ordered_baseline(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import soundfile as sf
    from timing_validation import validate_timeline
    spec = importlib.util.spec_from_file_location('context_audit', Path(__file__).parent/'experiments/cjk-onsets-v003/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cue = dict(text='あい', start=.1, end=.6, aligned_start=.1, aligned_end=.6, status='candidate', units=[
        dict(text='あ',first=0,stop=1,start=.1,end=.11),dict(text='い',first=1,stop=2,start=.11,end=.6)])
    (tmp_path/'cues.json').write_text(json.dumps([cue]))
    sf.write(tmp_path/'audio.wav',np.ones(44100)*.01,44100)
    protocol=json.loads((Path(__file__).parent/'experiments/cjk-onsets-v003/protocol.json').read_text())
    def unavailable(*args,**kwargs):raise ValueError('Optional refinement unavailable')
    monkeypatch.setattr(module,'fixed_units',unavailable)
    module.process(dict(song='audit',cues=str(tmp_path/'cues.json'),audio=str(tmp_path/'audio.wav'),
        output=str(tmp_path/'result'),duration=1.,language='Japanese'),SimpleNamespace(checkpoint='test'),protocol)
    actual=json.loads((tmp_path/'result/cues.json').read_text())
    validate_timeline(actual,['あい'],1.,language='Japanese')
    assert [u['start'] for u in actual[0]['units']] == [.1,.11]


@pytest.mark.parametrize('problem', ['source_hash','shape','nan','extra'])
def test_separation_contract_survives_optimized_python(tmp_path, problem):
    script = '''
import json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from karaoke.benchmark_neural import run_cases,sha
root=Path(sys.argv[1]); problem=sys.argv[2]
x=np.ones((441,2),dtype=np.float32)*.1
source=root/'source.wav'; sf.write(source,x,44100,subtype='FLOAT')
manifest=root/'cases.json'
manifest.write_text(json.dumps([dict(id='song',input=str(source),input_sha256='bad' if problem=='source_hash' else sha(source))]))
def infer(audio):
    a=audio.copy(); b=audio.copy(); extras={}
    if problem=='shape': a=a[:-1]
    if problem=='nan': a[0,0]=np.nan
    if problem=='extra': extras['other']=audio[:-1]
    return a,b,extras
try: run_cases(infer,root/'output',{},manifest)
except ValueError as error: print(json.dumps(dict(rejected=str(error))))
else: print(json.dumps(dict(rejected=False)))
'''
    result = subprocess.run([sys.executable,'-O','-c',script,str(tmp_path),problem],
                            text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)['rejected']


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


@pytest.mark.parametrize('concurrency', [1, 2, None])
def test_shutdown_stops_current_job_and_preserves_the_queue(tmp_path, concurrency):
    def slow_worker(folder):
        return [sys.executable, '-c', "import time;time.sleep(30)"]
    app = create_app(tmp_path, worker_command=slow_worker, version='test', worker_concurrency=concurrency,
                     cache_limit_bytes=1024**2)  # Sleeping fixtures do not need audio scratch space.
    active_count = concurrency or 5
    with TestClient(app) as client:
        current = [client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id']
                   for _ in range(active_count)]
        deadline = time.monotonic()+5
        while (len(app.state.worker.processes) < active_count or
               any(p is None for p in app.state.worker.processes.values())) and time.monotonic() < deadline:
            time.sleep(.02)
        (tmp_path/'preparation.pause').touch()
        queued = client.post('/v1/jobs', json={'url': 'https://youtu.be/2Vv-BfVoq4g'}).json()['id']
        processes = list(app.state.worker.processes.values())
        assert len(processes) == active_count and all(p is not None for p in processes)
        assert client.get(f'/v1/jobs/{queued}').json()['status'] == 'queued'
    assert all(p.returncode is not None for p in processes)
    assert all(app.state.store.read(j)['error']['code'] == 'interrupted' for j in current)
    assert app.state.store.read(queued)['status'] == 'queued'
    (tmp_path/'preparation.pause').unlink()
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        assert finished(client, queued)['status'] == 'completed'


@pytest.mark.parametrize('concurrency', [2, None])
def test_workers_keep_progress_results_and_failure_isolated_and_honor_pause(tmp_path, concurrency):
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
    app = create_app(tmp_path, worker_command=controlled_worker, version='test', worker_concurrency=concurrency,
                     cache_limit_bytes=1024**2)  # Tiny fixture files; storage limits are tested separately.
    active_count = concurrency or 5
    with TestClient(app) as client:
        ids = [client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id'] for _ in range(active_count)]
        deadline = time.monotonic()+5
        while not all((tmp_path/j/'started').exists() for j in ids) and time.monotonic() < deadline:
            time.sleep(.02)
        hold = tmp_path/'preparation.pause'
        hold.touch()
        ids.append(client.post('/v1/jobs', json={'url': URL, 'force': True}).json()['id'])
        health = client.get('/health').json()
        assert health['running_jobs'] == ids[:-1] and health['running_job'] == ids[0]
        assert health['preparation_concurrency'] == concurrency and health['queued'] == 1
        assert client.get('/v1/room').json()['preparation_jobs'] == ids[:-1]
        assert not (tmp_path/ids[-1]/'started').exists()
        for j in ids[:-1]:
            job = app.state.store.read(j)
            assert job['active_stages'] == ['render'] and job['song_info']['title'] == j
        (tmp_path/ids[0]/'fail').touch()
        (tmp_path/ids[0]/'release').touch()
        failed = finished(client, ids[0])
        assert failed['error']['message'] == ids[0]
        assert client.get('/health').json()['running_jobs'] == ids[1:-1]
        assert app.state.store.read(ids[1])['status'] == 'running'
        for j in ids[1:-1]:
            (tmp_path/j/'release').touch()
            assert finished(client, j)['status'] == 'completed'
        assert app.state.store.read(ids[-1])['status'] == 'queued'
        hold.unlink()
        (tmp_path/ids[-1]/'release').touch()
        assert finished(client, ids[-1])['status'] == 'completed'
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


def test_metadata_inference_is_reused_after_media_eviction_and_new_job(tmp_path):
    import shutil
    from karaoke.song_sources import extract_song_metadata
    from karaoke.stage_cache import StageCache
    store = Store(tmp_path)
    old, _ = store.create(JobRequest(url=URL), 'old')
    evidence = dict(title='Song', description_lines=[dict(number=1, text='A little song')])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        parsed=SongText(title='Song', artist='Singer', lyric_line_numbers=[1])))], model='test', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        parse = client.return_value.beta.chat.completions.parse
        parse.return_value = response
        first = extract_song_metadata(evidence, 'test', StageCache(tmp_path/old['id']))
        store.update(old['id'], status='failed', song_info=dict(title='Song', artist='Singer'))
        shutil.rmtree(tmp_path/old['id'])
        new, _ = store.create(JobRequest(url=URL), 'new')
        assert new['id'] != old['id'] and new['song_info']['artist'] == 'Singer'
        parse.side_effect = AssertionError('Metadata was fetched twice')
        second = extract_song_metadata(evidence, 'test', StageCache(tmp_path/new['id']))
        assert second == first and parse.call_count == 1


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


@pytest.mark.parametrize('split_line', [False, True])
@pytest.mark.parametrize('padding', ['distant', 'continuous'])
def test_repeated_passage_detection_ignores_layout_and_unrelated_padding(split_line, padding):
    from karaoke.song_sources import missing_repetitions
    refrain = ['Golden lanterns in the window', 'Every sparrow flies back home']
    lines = ['Opening', *(refrain if split_line else [' '.join(refrain)])]
    cues = [dict(text=text,start=1.+i,end=1.8+i,status='candidate') for i,text in enumerate(lines)]
    words = [dict(word=text,start=10.+i,end=10.8+i) for i,text in enumerate(refrain)]
    if padding == 'continuous':
        words.insert(0, dict(word='The presenter is discussing something unrelated', start=9.,end=9.8))
    words.append(dict(word='Thank you for watching this presentation',
                      start=28. if padding == 'distant' else 12., end=29. if padding == 'distant' else 13.))
    found = missing_repetitions(lines,cues,words,'English',30.)
    assert len(found)==1 and found[0]['source_range']==[1,len(lines)]
    assert found[0]['boundary_anchors'][0]['start']==10.
    assert found[0]['boundary_anchors'][-1]['end']==11.8


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


@pytest.mark.parametrize('cached_native', [False, True])
@pytest.mark.parametrize('previews', [False, True])
@pytest.mark.parametrize('silent', [False, True])
def test_master_only_package_still_builds_verified_three_stem_previews(tmp_path, previews, silent, cached_native):
    import soundfile as sf
    from karaoke import exact_stems
    from karaoke.three_stems import build, verify

    rate, frames = 48000, 48013
    tone = (.2*np.sin(2*np.pi*440*np.arange(frames)/rate)).astype('float32')
    audio = np.column_stack([tone, -tone])
    source, vocals, lead = (tmp_path/f'{name}.wav' for name in ('source', 'vocals', 'lead'))
    for path, scale in ((source, 1), (vocals, .5), (lead, .25)):
        track = audio*scale
        # Reproduce quiet-source cancellation in both levels of the split.
        track[-3:] = (1e-30 if path == source else
                      np.array([[.000288816], [9.1246e-5], [.000112535]]) if path == vocals else .1)
        if path == vocals:
            track[-4] = 1e-30
        if silent:
            track[:] = 0
        sf.write(path, track, rate, subtype='DOUBLE')
    two, three = tmp_path/'two', tmp_path/'three'
    native = tmp_path/'native.wav' if cached_native else None
    if native:
        exact_stems.cache_source(source, native)
        np.testing.assert_array_equal(exact_stems.source_samples(source, native)[0], exact_stems.decode_source(source)[0])
    subprocess.run([sys.executable, exact_stems.__file__, 'build', '--source', str(source),
        '--vocals-estimate', str(vocals), '--output', str(two), *([] if previews else ['--no-previews']),
        *(['--source-pcm', str(native)] if native else [])],
        check=True, capture_output=True)
    assert {p.name for p in two.glob('*.mp3')} == ({'vocals.mp3', 'backing.mp3'} if previews else set())
    assert (exact_stems.verify(two)['max_absolute_error'] > 0) is not silent
    np.testing.assert_array_equal(sf.read(two/'vocals.wav')[0], sf.read(vocals)[0])
    result = build(two, lead, three, source_pcm=native, karaoke_preview=previews)
    assert (result['reconstruction']['vocal_first']['max_absolute_error'] > 0) is not silent
    assert verify(three) == result['reconstruction']
    np.testing.assert_array_equal(sf.read(three/'lead.wav')[0], sf.read(lead)[0])
    assert exact_stems.sha(three/'instrumental.wav') == exact_stems.sha(two/'backing.wav')
    assert set(result['previews']) == {'lead', 'backing-vocals', 'instrumental'} | ({'karaoke'} if previews else set())
    assert (three/'karaoke.mp3').exists() == previews
    assert not (three/'.karaoke-mix.wav').exists()
    masters = {name: sf.read(three/spec['file'], always_2d=True)[0] for name, spec in result['masters'].items()}
    peak = max(float(np.max(np.abs(track))) for track in (
        *masters.values(), sf.read(source, always_2d=True)[0], masters['instrumental']+masters['backing-vocals']))
    assert result['preview_shared_gain'] == min(1., .84/max(peak, 1e-12))
    for name in result['previews']:
        assert validate_preview(three/f'{name}.mp3', frames, 2, rate)['decoded_length_delta_samples'] == 0
    if native:
        with native.open('ab') as stream:
            stream.write(b'corrupt')
        with pytest.raises(ValueError, match='Cached native audio'):
            exact_stems.source_samples(source, native)
    with (three/'lead.wav').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='Master changed'):
        verify(three)
    with (two/'vocals.wav').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='Master file has changed'):
        exact_stems.verify(two)


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
    nested = folder/'diagnostics'
    nested.mkdir()
    (nested/'keep.json').write_text('{}')
    (nested/'oversized.json').write_bytes(b'x'*(2*1024**2+1))
    outside = tmp_path/'outside.json'
    outside.write_text('{"unmanaged":true}')
    (nested/'linked.json').symlink_to(outside)
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
    assert (nested/'keep.json').read_text() == '{}'
    assert not (nested/'oversized.json').exists() and not (nested/'linked.json').exists()
    assert outside.read_text() == '{"unmanaged":true}'
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
    app = create_app(tmp_path, worker_command=failed_download, version='test')
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
    assert not cached and duplicate_cached and not restored['restore_video']
    assert restored['pipeline_version'] == 'new-renderer'
    assert restored['result'] == original['result']
    changed, _ = store.create(request.model_copy(update={'mv_audio':'instrumental'}), 'new-renderer')
    assert changed['id'] != original['id']
    forced, _ = store.create(request.model_copy(update={'force':True}), 'new-renderer')
    assert forced['id'] != original['id']


def test_completed_version_refresh_keeps_valid_stage_checkpoints(tmp_path):
    from karaoke.stage_cache import StageCache
    store = Store(tmp_path)
    old = cache_fixture(store, '1', restorable=True)
    folder = tmp_path/old['id']
    source, separated, aligned, code = [folder/name for name in ('input', 'separated', 'aligned', 'align.py')]
    source.write_bytes(b'audio')
    code.write_text('version = 1')
    cache = StageCache(folder)
    with cache.stage('separate', inputs=[source], outputs=[separated]):
        separated.write_bytes(b'computed stem')
    with cache.stage('align', inputs=[separated], outputs=[aligned], code=[code]):
        aligned.write_bytes(b'old timings')
    refreshed, hit = store.create(JobRequest(**{**old['input'],'force':False}), 'new')
    assert not hit and refreshed['id']==old['id'] and refreshed['status']=='queued'
    code.write_text('version = 2')
    with StageCache(folder).stage('separate', inputs=[source], outputs=[separated]) as hit:
        assert hit and separated.read_bytes()==b'computed stem'
    with StageCache(folder).stage('align', inputs=[separated], outputs=[aligned], code=[code]) as hit:
        assert not hit and not aligned.exists()
        aligned.write_bytes(b'new timings')


@pytest.mark.parametrize('queued', [False, True])
def test_outdated_video_restore_revalidates_stages_before_completion(tmp_path, queued):
    def check_refresh(folder):
        job = app.state.store.read(folder.name)
        assert not job['restore_video'] and job['pipeline_version'] == 'new'
        return fake_worker(folder)
    app = create_app(tmp_path, worker_command=check_refresh, version='new')
    store = app.state.store
    old = cache_fixture(store, '1', restorable=True)
    store.update(old['id'], status='queued' if queued else 'video_evicted', restore_video=True)
    (tmp_path/old['id']/'mv.mp4').unlink()
    with TestClient(app) as client:
        if not queued:
            response = client.post('/v1/room/entries', json={'job_id':old['id']})
            assert response.status_code == 201
        fresh = finished(client, old['id'])
        assert fresh['status'] == 'completed' and fresh['pipeline_version'] == 'new'
        assert not fresh['restore_video']


@pytest.mark.parametrize('source_matches', [True, False])
def test_video_only_pipeline_uses_retained_audio_and_cues(tmp_path, monkeypatch, source_matches):
    import shutil
    from pathlib import Path
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
    result['artifacts']['poster'] = dict(path='missing-poster.jpg', bytes=100, sha256='missing')
    (folder/'result.json').write_text(json.dumps(result))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    pipeline = Pipeline(job['input'], folder, separator_python='/missing/python')
    def download(**kwargs):
        pipeline.source.parent.mkdir()
        shutil.copyfile(source, pipeline.source)
    monkeypatch.setattr(pipeline, 'download', download)
    if source_matches:
        restored = pipeline.restore_video()
        assert set(restored['video_restore']['timings']) == {'verify_video', 'render', 'mux_video'}
        assert (folder/'mv.mp4').stat().st_size > 2048
        assert restored['artifacts']['mv']['sha256'] == file_hash(folder/'mv.mp4')
        opening = restored['quality']['opening_countdown']
        assert opening['start'] == -3.5 and opening['onset'] == .5
        assert opening['dots'] == 3 and 0 < opening['y'] < opening['height']
        # Splitting picture encoding from audio muxing must not change either
        # decoded stream compared with the established all-at-once renderer.
        mixed = folder/'expected-mix.wav'
        samples, rate = sf.read(folder/'instrumental', dtype='float32', always_2d=True)
        if result['mv_audio'] == 'instrumental_backing':
            samples += sf.read(folder/'backing_vocals', dtype='float32', always_2d=True)[0]
        sf.write(mixed, samples, rate, subtype='FLOAT')
        baseline = folder/'baseline.mp4'
        subprocess.run([sys.executable, str(Path(__file__).with_name('render_lyric_progress.py')),
            '--video', str(source), '--audio', str(mixed), '--cues', str(folder/'subtitles'),
            '--song-info', str(folder/'song-info.json'), '--language', 'Latin', '--duration', '2',
            '--gain', str(result['shared_preview_gain']), '--bottom-margin', '96', '--output', str(baseline)],
            check=True, capture_output=True)
        for stream in ('v:0', 'a:0'):
            hashes = [subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:'+stream,
                '-f', 'hash', '-hash', 'sha256', '-']) for path in (baseline, folder/'mv.mp4')]
            assert hashes[0] == hashes[1]
    else:
        with pytest.raises(StageFailure, match='音频与缓存版本不同'):
            pipeline.restore_video()
        assert not (folder/'mv.mp4').exists()
    assert all((folder/name).read_bytes() == data for name, data in before.items())


def test_pipeline_materializes_all_media_before_ready(tmp_path, monkeypatch):
    """Real codecs/packagers around fixed model outputs exercise the full DAG."""
    import soundfile as sf
    from karaoke.three_stems import verify
    monkeypatch.setenv('KARAOKE_REQUIRE_CACHED_SOURCE', '1')
    pipeline = Pipeline(dict(url=URL, title='Fixture', artist='Test', language='English',
                             lyrics=['We sing together']), tmp_path, device='cpu', catalog_root=None)
    pipeline.source.parent.mkdir()
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=30:d=2',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-ac', '2', '-c:v', 'libx264',
        '-c:a', 'pcm_s16le', str(pipeline.source)], check=True)
    (tmp_path/'source.json').write_text(json.dumps(dict(url=URL, duration=2,
        sha256=file_hash(pipeline.source), audio_only=False)))
    (pipeline.source.parent/'source.info.json').write_text(json.dumps(dict(
        title='Fixture', artist='Test', duration=2, format_id='fixture')))
    original_run = pipeline.run
    def run(stage, command, **kwargs):
        if stage.startswith('separate_'):
            path = pipeline.vocals if stage == 'separate_music' else tmp_path/'lead-separation/song/vocals.wav'
            path.parent.mkdir(parents=True)
            samples, rate = sf.read(tmp_path/'source-44100.wav' if stage == 'separate_music' else pipeline.vocals,
                                    dtype='float32', always_2d=True)
            sf.write(path, samples*.5, rate, subtype='FLOAT')
        else:
            original_run(stage, command, **kwargs)
    monkeypatch.setattr(pipeline, 'run', run)
    def align(*args):
        (tmp_path/'cues.json').write_text(json.dumps([dict(text='We sing together', start=.1, end=1.9,
            units=[dict(text=word, start=start, end=end, first=first, stop=stop) for word,start,end,first,stop in
                   [('We',.1,.6,0,2), ('sing',.6,1.1,3,7), ('together',1.1,1.9,8,16)]])]))
        (tmp_path/'quality.json').write_text('{}')
    monkeypatch.setattr(pipeline, 'align', align)
    result = pipeline.execute()
    assert {'mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'instrumental_preview',
            'backing_vocals_preview', 'lead_vocals_preview', 'subtitles'} <= result['artifacts'].keys()
    assert json.loads((tmp_path/'result.json').read_text()) == result
    for artifact in result['artifacts'].values():
        path = tmp_path/artifact['path']
        assert path.stat().st_size == artifact['bytes'] and file_hash(path) == artifact['sha256']
    assert verify(tmp_path/'stems')['vocal_first'] == result['reconstruction']
    assert all(result['stage_timeline'][name]['status'] == 'completed' for name in
               ('decode_native', 'package_two_stems', 'package_three_stems', 'render', 'mux_video', 'verify_video'))


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


@pytest.mark.parametrize('failure', [None, 'value', 'fatal'])
def test_local_recognizer_isolates_libraries_reuses_model_and_cleans_up(tmp_path, monkeypatch, failure):
    import os
    from karaoke.local_asr import isolated_recognizer
    (tmp_path/'faster_whisper.py').write_text('''
import os, sys
from dataclasses import dataclass
@dataclass
class Segment:
    text: str
    words: list
class WhisperModel:
    def __init__(self, *args, **kwargs):
        assert 'torch' not in sys.modules
        self.calls = 0
    def detect_language(self, **kwargs):
        return 'en', 1., []
class BatchedInferencePipeline:
    def __init__(self, model): self.model = model
    def transcribe(self, samples, **kwargs):
        if samples[0] == -1: raise ValueError('unusable crop')
        if samples[0] == -2: raise RuntimeError('model failed')
        self.model.calls += 1
        text = f'{os.getpid()}:{self.model.calls}:{len(samples)}:{samples.sum()}'
        return iter([Segment(text, [])]), None
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    config = dict(model='test', compute_type='float32', batch_size=8)
    with isolated_recognizer(config, 'cpu', tmp_path) as recognize:
        first = recognize(np.ones(16000, dtype=np.float32))
        pid, calls, size, total = first['text'].split(':')
        pid = int(pid)
        assert pid != os.getpid() and (calls,size,total) == ('1','16000','16000.0')
        if failure:
            with pytest.raises(ValueError if failure == 'value' else RuntimeError,
                               match='unusable crop' if failure == 'value' else 'Local recognizer exited'):
                recognize(np.full(16000, -1 if failure == 'value' else -2, dtype=np.float32))
        if failure != 'fatal':
            second = recognize(np.ones(8000, dtype=np.float32))
            assert second['text'] == f'{pid}:2:8000:8000.0'
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert not list(tmp_path.glob('recognizer-*'))


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


@pytest.mark.parametrize('status', ['source_ready', 'audio_ready', 'completed', 'video_evicted'])
def test_new_survey_reuses_verified_ready_sources(tmp_path, status):
    store = Store(tmp_path)
    original, _ = store.create(JobRequest(url=URL), 'old')
    source = tmp_path/original['id']
    (source/'download').mkdir()
    media = source/'download/source.mkv'
    media.write_bytes(b'reusable original audio')
    (source/'source.json').write_text(json.dumps(dict(url=URL, sha256=file_hash(media))))
    (source/'download/source.info.json').write_text(json.dumps(dict(id='n5sjpCQhTDk')))
    store.update(original['id'], status=status)
    new, _ = store.create(JobRequest(url=URL, force=True), 'new')
    target = tmp_path/new['id']
    store.cache.reuse_download(target, URL)
    assert (target/'download/source.mkv').read_bytes() == media.read_bytes()
    assert json.loads((target/'source.json').read_text())['input_url'] == URL


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
    crosscheck = dict(candidate_texts=[['Stars above the quiet shore']], crosscheck=dict(status='corroborated'))
    (folder/'lyric-crosscheck.json').write_text(json.dumps(crosscheck))
    source_failure = folder/'lyric-source-attempts/first/lyrics.json'
    source_failure.parent.mkdir(parents=True)
    source_failure.write_text('{"lines":["Original rejected text"]}')
    render_report = dict(song_intro=dict(status='omitted', reason='Missing title glyph'))
    (folder/'captioned-picture.render.json').write_text(json.dumps(render_report))
    outside = tmp_path/'precious.wav'
    outside.write_bytes(b'keep me')
    (folder/'raw/link').symlink_to(outside)
    store.cache.compact(job)
    assert (folder/'mv.mp4').read_bytes() == b'x'*2048
    assert (folder/'request.json').exists() and (folder/'worker.log').stat().st_size == 65536
    assert not (folder/'raw').exists() and outside.read_bytes() == b'keep me'
    assert json.loads((folder/'lyric-crosscheck.json').read_text()) == crosscheck
    assert json.loads(source_failure.read_text())['lines'] == ['Original rejected text']
    assert json.loads((folder/'captioned-picture.render.json').read_text()) == render_report
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


@pytest.mark.parametrize('limit_gib', [2, None])
def test_cache_limit_setting_is_validated_and_survives_restart(tmp_path, limit_gib):
    expected = None if limit_gib is None else limit_gib*GIB
    with TestClient(create_app(tmp_path,version='test')) as client:
        assert client.post('/v1/storage',json={'limit_gib':0}).status_code == 422
        saved = client.post('/v1/storage',json={'limit_gib':limit_gib}).json()
        assert saved['storage']['limit_bytes'] == expected
    with TestClient(create_app(tmp_path,version='test')) as client:
        assert client.get('/v1/storage').json()['limit_bytes'] == expected
        assert client.post('/v1/storage',json={'limit_gib':10}).json()['storage']['limit_bytes'] == 10*GIB


def test_unlimited_cache_keeps_outputs_and_is_scoped_to_its_data_directory(tmp_path, monkeypatch):
    store = Store(tmp_path/'remote', cache_limit_bytes=1024)
    job = cache_fixture(store, '1')
    folder = store.root/job['id']
    store.cache.set_limit(None)
    assert store.cache.enforce() and (folder/'mv.mp4').exists()
    assert store.cache.snapshot()['used_bytes'] > 1024
    assert Store(tmp_path/'mac').cache.limit == 10*GIB
    assert Store(store.root).cache.limit is None
    monkeypatch.setenv('KARAOKE_CACHE_MAX_GIB', 'none')
    assert Store(tmp_path/'new-remote').cache.limit is None
    assert Store(tmp_path/'mac').cache.limit == 10*GIB


@pytest.mark.parametrize('unlimited', [False, True])
def test_low_free_disk_blocks_processing_without_evicting_the_playlist(tmp_path, unlimited):
    def must_not_start(folder):
        raise AssertionError('The worker must not start without disk headroom')
    app = create_app(tmp_path, worker_command=must_not_start, version='test')
    if unlimited:
        app.state.store.cache.set_limit(None)
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
    item = dict(url=URL, title='Video title', duration=180, channel_is_official_artist=True,
                thumbnails=[{'url':'https://i.ytimg.com/cover.jpg'}])
    results = search_results('youtube', [item, item, dict(item, live_status='is_live'),
        dict(item, url='http://127.0.0.1/private'), dict(item, url='https://youtube.com.attacker.test/'),
        dict(item, url='https://www.bilibili.com/video/BV1bK411W797')])
    assert len(results) == 3 and all(row['url'] == URL for row in results)
    assert results[0]['thumbnail'] == 'https://i.ytimg.com/cover.jpg'
    assert search_results('youtube', [dict(item, thumbnail='https://localhost/cover')])[0]['thumbnail'] is None


@pytest.mark.parametrize('catalogue_section', ['songs', 'videos'])
def test_music_catalogue_keeps_provider_order_versions_and_distinct_artist_ids(monkeypatch, catalogue_section):
    from karaoke.video_search import music_entries
    artist_id = 'UC'+'a'*22
    namesake_id = 'UC'+'b'*22
    rows = [dict(title='Evening' if i % 2 else 'Evening (Live)', videoId=f'{i:011d}',
        artists=[dict(name='Singer', id=artist_id)], duration_seconds=180,
        views='2.5M' if i == 0 else None, videoType='MUSIC_VIDEO_TYPE_ATV',
        album=dict(name='An album'), thumbnails=[dict(url='https://lh3.googleusercontent.com/cover')]) for i in range(35)]
    calls = []
    class Music:
        def __init__(self, **kwargs):
            pass
        def _send_request(self, endpoint, body, *params):
            if endpoint == 'search':
                calls.append(('search', body['query']))
                entries = ([dict(artist='SINGER', browseId=namesake_id), dict(artist='Singer', browseId=artist_id)]
                           if body['query'] == 'Singer' else rows)
                return dict(contents=dict(musicShelfRenderer=dict(contents=[dict(musicResponsiveListItemRenderer=e) for e in entries],
                    continuations=[dict(nextContinuationData=dict(continuation='search-next'))])))
            return dict(contents=dict(musicPlaylistShelfRenderer=dict(contents=[dict(continuationItemRenderer=dict(
                continuationEndpoint=dict(continuationCommand=dict(token='playlist-next'))))])))
        def get_artist(self, identifier):
            calls.append(('artist', identifier))
            artist = dict(name='Singer', songs=dict(browseId=None), monthlyListeners='8M')
            artist[catalogue_section] = dict(browseId='VLpopular', results=rows[:5])
            return artist
        def get_playlist(self, identifier, limit, **kwargs):
            assert identifier == 'VLpopular' and limit == 0
            self._send_request('browse', dict(browseId=identifier))
            return dict(tracks=rows)
    monkeypatch.setattr('ytmusicapi.YTMusic', Music)
    monkeypatch.setattr('ytmusicapi.parsers.search.parse_search_results', lambda contents, kind: [row['musicResponsiveListItemRenderer'] for row in contents])
    page = music_entries('songs', 'Evening')
    assert page['continuation'] == dict(params='&ctoken=search-next&continuation=search-next')
    songs = page['results']
    assert len(songs) == 35 and [s['title'] for s in songs] == [r['title'] for r in rows]
    assert songs[0]['version'] == '专辑音频' and songs[0]['views'] == '2.5M' and songs[1]['views'] is None
    assert songs[0]['artists'] == [dict(name='Singer', id=artist_id)]
    artists = music_entries('artists', 'Singer')['results']
    assert [a['browse_id'] for a in artists] == [namesake_id, artist_id]
    catalogue = music_entries('artist', artists[1]['browse_id'])
    assert calls[-1] == ('artist', artist_id)
    assert catalogue['complete'] and catalogue['catalogue_section'] == catalogue_section
    assert catalogue['order'] == ('youtube_music_top_songs' if catalogue_section == 'songs' else 'youtube_music_videos')
    assert catalogue['results'] == songs
    assert catalogue['continuation']['token'] == 'playlist-next'


@pytest.mark.parametrize('section', ['songs', 'videos', 'releases', 'empty'])
def test_artist_without_playlist_keeps_native_available_works(monkeypatch, section):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    artist = dict(name='Singer', songs=dict(browseId=None),
                  albums=dict(results=[dict(title='Album', browseId='MPREalbum')]),
                  singles=dict(results=[dict(title='Single', browseId='MPREsingle')]))
    if section in ('songs', 'videos'):
        artist[section] = dict(results=[dict(title='Native track', videoId='abcdefghijk')])
    if section == 'songs':
        artist['videos'] = dict(results=[dict(title='Other shelf', videoId='12345678901')])
    if section == 'empty':
        artist.pop('albums'); artist.pop('singles')
    monkeypatch.setattr(YTMusic, 'get_artist', lambda *args: artist)
    monkeypatch.setattr(YTMusic, 'get_playlist', lambda *a, **kw: pytest.fail('No playlist link was provided'))
    page = music_entries('artist', 'UC'+'a'*22)
    assert page['complete'] is False and page['continuation'] is None
    if section in ('songs', 'videos'):
        assert page['catalogue_section'] == section
        assert [r['title'] for r in page['results']] == ['Native track']
        assert page['results'][0]['url'] == 'https://www.youtube.com/watch?v=abcdefghijk'
    else:
        assert page['catalogue_section'] == 'releases'
        assert [r['title'] for r in page['results']] == ([] if section == 'empty' else ['Album', 'Single'])
        assert all(r['kind'] == 'album' and r['uploader'] == 'Singer' for r in page['results'])


def test_music_metadata_does_not_expose_arbitrary_links_or_invent_missing_counts():
    from karaoke.video_search import music_results
    row = dict(title='Evening', videoId='abcdefghijk', artists=[dict(name='Singer', id='not-a-channel')],
               thumbnails=[dict(url='https://googleusercontent.com.attacker.test/image')], duration_seconds=float('nan'))
    songs = music_results([row, dict(row, videoId='../../private')], 'songs')
    assert len(songs) == 1 and songs[0]['url'] == 'https://www.youtube.com/watch?v=abcdefghijk'
    assert songs[0]['thumbnail'] is None and songs[0]['duration'] == 0 and songs[0]['views'] is None
    assert songs[0]['artists'][0]['id'] is None
    assert music_results([dict(artist='Singer', browseId='../../private')], 'artists') == []
    unavailable = music_results([dict(row, isAvailable=False)], 'songs')
    assert len(unavailable) == 1 and unavailable[0]['available'] is False


def test_unfiltered_music_search_keeps_native_sections_translated_titles_and_collections(monkeypatch):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    artist_id = 'UC'+'a'*22
    artist_run = dict(text='Singer', navigationEndpoint=dict(browseEndpoint=dict(browseId=artist_id)))
    def row(title, browse_id):
        return dict(musicResponsiveListItemRenderer=dict(
            navigationEndpoint=dict(browseEndpoint=dict(browseId=browse_id)),
            flexColumns=[dict(musicResponsiveListItemFlexColumnRenderer=dict(text=dict(runs=[dict(text=title)]))),
                dict(musicResponsiveListItemFlexColumnRenderer=dict(text=dict(runs=[
                    dict(text='专辑'), dict(text=' • '), artist_run, dict(text=' • '), dict(text='2001年')])))]))
    watch = dict(watchEndpoint=dict(videoId='abcdefghijk', watchEndpointMusicSupportedConfigs=dict(
        watchEndpointMusicConfig=dict(musicVideoType='MUSIC_VIDEO_TYPE_OMV'))))
    card = dict(musicCardShelfRenderer=dict(onTap=watch, title=dict(runs=[dict(text='Top song')]),
        subtitle=dict(runs=[dict(text='视频'), dict(text=' • '), artist_run, dict(text=' • '), dict(text='52M views')])))
    artist_card = dict(musicCardShelfRenderer=dict(title=dict(runs=[artist_run]), subtitle=dict(runs=[dict(text='歌手')])))
    sections = [card, dict(itemSectionRenderer=dict(contents=[row('The One and Only', 'MPREexample')])),
        dict(musicShelfRenderer=dict(contents=[row('Another album', 'MPREanother')])),
        dict(itemSectionRenderer=dict(contents=[row('Singer', artist_id)])), artist_card]
    calls = []
    def send(self, endpoint, body, *params):
        calls.append((endpoint, body))
        return dict(contents=dict(sectionListRenderer=dict(contents=sections)))
    monkeypatch.setattr(YTMusic, '_send_request', send)
    page = music_entries('songs', '唯一')
    assert calls == [('search', dict(query='唯一'))]
    assert page['continuation'] is None
    assert [s['title'] for s in page['results']] == ['Top song', 'The One and Only', 'Another album', 'Singer', 'Singer']
    assert [s['kind'] for s in page['results']] == ['song', 'album', 'album', 'artist', 'artist']
    assert page['results'][1]['browse_id'] == 'MPREexample'
    assert page['results'][1].get('views') is None  # A release year is not a play count.


@pytest.mark.parametrize('first_page', [True, False])
def test_artist_song_play_counts_survive_native_playlist_parsing(monkeypatch, first_page):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    artist_id = 'UC'+'a'*22
    def column(text, endpoint=None):
        run = dict(text=text)
        if endpoint:
            run['navigationEndpoint'] = endpoint
        return dict(musicResponsiveListItemFlexColumnRenderer=dict(text=dict(runs=[run])))
    watch = dict(watchEndpoint=dict(videoId='abcdefghijk'))
    artist = dict(browseEndpoint=dict(browseId=artist_id, browseEndpointContextSupportedConfigs=dict(
        browseEndpointContextMusicConfig=dict(pageType='MUSIC_PAGE_TYPE_ARTIST'))))
    rows = [dict(musicResponsiveListItemRenderer=dict(
        flexColumns=[column('A song', watch), column('Singer', artist), column(count)],
        overlay=dict(musicItemThumbnailOverlayRenderer=dict(content=dict(musicPlayButtonRenderer=dict(playNavigationEndpoint=watch))))))
        for count in ['播放次数：5163万', '52M plays', '2001', '3:21']]
    monkeypatch.setattr(YTMusic, 'get_artist', lambda self, query: dict(name='Singer', songs=dict(browseId='VLpopular')))
    def playlist(self, *args, **kwargs):
        self.response = dict(contents=dict(musicPlaylistShelfRenderer=dict(contents=rows)))
        return dict(tracks=[])
    monkeypatch.setattr(YTMusic, 'get_playlist', playlist)
    monkeypatch.setattr(YTMusic, '_send_request', lambda *a, **k: dict(onResponseReceivedActions=[
        dict(appendContinuationItemsAction=dict(continuationItems=rows))]))
    page = music_entries('artist', artist_id, None if first_page else dict(token='next'))
    assert [s['views'] for s in page['results']] == ['5163万', '52M', None, None]
    assert all(s['artists'] == [dict(name='Singer', id=artist_id)] for s in page['results'])


def test_music_album_browse_keeps_every_track_and_its_order(monkeypatch):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    tracks = [dict(title=f'Track {i}', videoId=f'{i:011d}') for i in range(32)]
    monkeypatch.setattr(YTMusic, 'get_album', lambda self, query: dict(title='An album',
        artists=[dict(name='Singer', id='UC'+'a'*22)], tracks=tracks))
    page = music_entries('album', 'MPREexample')
    assert [s['title'] for s in page['results']] == [s['title'] for s in tracks]
    assert all(s['album'] == 'An album' and s['uploader'] == 'Singer' for s in page['results'])
    assert page['continuation'] is None


def test_artist_playlist_preserves_richer_shelf_metadata_by_video_id(monkeypatch):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    shelf = [dict(videoId='abcdefghijk', title='Same title', views='1136万次观看'),
             dict(videoId='00000000000', title='Same title', views='8万次观看')]
    monkeypatch.setattr(YTMusic, 'get_artist', lambda *a: dict(name='Singer', videos=dict(browseId='VLlist', results=shelf)))
    monkeypatch.setattr(YTMusic, 'get_playlist', lambda *a, **kw: dict(tracks=[
        dict(videoId='00000000000', title='Renamed', views=0, duration_seconds=180),
        dict(videoId='abcdefghijk', title='Same title', views='', duration_seconds=240),
        dict(videoId='11111111111', title='Same title')]))
    page = music_entries('artist', 'UC'+'a'*22)
    assert [s['views'] for s in page['results']] == [0, '1136万次观看', None]
    assert [s['duration'] for s in page['results']] == [180, 240, 0]
    assert [s['title'] for s in page['results']] == ['Renamed', 'Same title', 'Same title']


def test_music_episodes_remain_visible_without_becoming_songs():
    from karaoke.video_search import music_results
    rows = [dict(title='Conversation', videoId='abcdefghijk', resultType='episode'),
            dict(title='Song', videoId='00000000000', resultType='song')]
    results = music_results(rows, 'songs')
    assert [s['kind'] for s in results] == ['episode', 'song']
    assert results[0]['url'] == 'https://www.youtube.com/watch?v=abcdefghijk'


def test_search_api_failure_is_distinct_from_empty_and_creates_no_jobs(tmp_path, monkeypatch):
    from karaoke import video_search
    async def fetch(provider, query, continuation=None):
        if query == 'failure':
            raise RuntimeError('provider unavailable')
        return dict(results=[], continuation=None)
    monkeypatch.setattr(video_search, 'fetch_results', fetch)
    app = create_app(tmp_path, version='test')
    with TestClient(app) as client:
        found = client.get('/v1/search', params=dict(provider='youtube', q='  some song  '))
        assert found.status_code == 200 and found.json()['results'] == []
        assert found.json()['query'] == 'some song'
        assert client.get('/v1/search', params=dict(provider='youtube', q='failure')).status_code == 502
        assert client.get('/v1/search', params=dict(provider='bilibili', q='some song')).status_code == 422
        assert client.get('/v1/music/search', params=dict(q='song')).json()['results'] == []
        for kind in ('songs', 'artists'):
            assert client.get('/v1/music/search', params=dict(kind=kind, q='song')).json()['results'] == []
            assert client.get('/v1/music/search', params=dict(kind=kind, q='failure')).status_code == 502
            assert client.get('/v1/music/search', params=dict(kind=kind, q=' ')).status_code == 422
        assert client.get('/v1/music/search', params=dict(kind='other', q='song')).status_code == 422
        assert client.get('/v1/music/artists/not-an-artist').status_code == 422
        assert client.get('/v1/music/artists/UC'+'a'*22).status_code == 200
        assert client.get('/v1/music/albums/MPREexample').status_code == 200
        assert client.get('/v1/music/albums/not-an-album').status_code == 422
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
        async def fetch(provider, query, continuation=None):
            nonlocal active, peak
            calls.append((provider, query)); active += 1; peak = max(peak, active)
            started.set()
            try:
                await release.wait()
                return dict(results=[dict(title=query)], continuation=None)
            finally:
                active -= 1
        monkeypatch.setattr(video_search, 'fetch_results', fetch)
        one = asyncio.create_task(service.search('youtube', 'same'))
        await started.wait()
        two = asyncio.create_task(service.search('youtube', 'same'))
        await asyncio.sleep(0)
        one.cancel()
        await asyncio.gather(one, return_exceptions=True)
        others = [asyncio.create_task(service.search('youtube', str(i))) for i in range(7)]
        await asyncio.sleep(0)
        with pytest.raises(OverflowError):
            await service.search('youtube', 'overflow')
        release.set()
        assert (await two)['results'] == [dict(title='same', rank=1)]
        await asyncio.gather(*others)
        assert peak <= 2 and calls.count(('youtube', 'same')) == 1
        assert (await service.search('youtube', 'same'))['results'] == [dict(title='same', rank=1)]
        assert len(calls) == 8 and not service.pending
        # Expired entries refresh, and the metadata cache has a fixed size.
        service.cache[('youtube', 'same', None)] = (0, dict(results=[]))
        assert (await service.search('youtube', 'same'))['results'] == [dict(title='same', rank=1)]
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


def test_search_keeps_provider_order_when_original_has_no_channel_badge():
    from karaoke.video_search import search_results
    def item(index, **extra):
        return dict(url=f'https://www.youtube.com/watch?v={index:011d}', title=f'Cover {index}',
                    channel_is_official_artist=True, **extra)
    found = search_results('youtube', [item(i) for i in range(12)])
    assert [x['title'] for x in found] == [f'Cover {i}' for i in range(12)]
    original = dict(url=URL, title='The song (Official music video)', channel='A broadcaster',
                    channel_is_verified=None, channel_is_official_artist=False)
    cover = dict(item(1), title='The song (Cover)')
    found = search_results('youtube', [original, cover], 'The song')
    assert [x['url'] for x in found] == [URL, cover['url']]


@pytest.mark.parametrize('artist_badge', [True, False])
def test_youtube_translated_titles_are_not_removed_by_literal_query_matching(artist_badge):
    from karaoke.video_search import search_results
    original = dict(url=URL, title='Nightfall - Official music video',
                    channel='Original artist', channel_is_official_artist=artist_badge)
    fan_edit = dict(url='https://www.youtube.com/watch?v=00000000001', title='【MAD】夜色')
    cover = dict(url='https://www.youtube.com/watch?v=00000000002', title='夜色 (Cover)')
    results = search_results('youtube', [original, fan_edit, cover], '夜色')
    assert [row['url'] for row in results] == [URL, fan_edit['url'], cover['url']]


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




def test_youtube_search_fetches_one_native_page_and_resumes_its_continuation(monkeypatch):
    from karaoke.video_search import youtube_entries
    from yt_dlp.extractor.youtube import YoutubeSearchIE
    calls = []
    def respond(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            assert kwargs['query']['continuation'] == 'another-page'
            assert kwargs['query']['query'] == 'anything'
            return {'onResponseReceivedCommands': [{'appendContinuationItemsAction': {'continuationItems': [
                {'itemSectionRenderer': {'contents': [
                    {'videoRenderer': {'videoId': '99999999999', 'title': {'simpleText': 'Last video'}}}]}}]}}]}
        return {'contents': {'twoColumnSearchResultsRenderer': {'primaryContents': {
            'sectionListRenderer': {'contents': [{'itemSectionRenderer': {'contents': [
                {'videoRenderer': {'videoId': f'{i:011d}', 'title': {'simpleText': f'Video {i}'}}}
                for i in range(35)]}}, {'continuationItemRenderer': {'continuationEndpoint': {
                    'continuationCommand': {'token': 'another-page'}}}}]}}}}}
    monkeypatch.setattr(YoutubeSearchIE, '_extract_response', respond)
    monkeypatch.setattr(YoutubeSearchIE, '_download_ytcfg', lambda *a, **k: {})
    found = youtube_entries('anything')
    assert [row['title'] for row in found['results']] == [f'Video {i}' for i in range(35)]
    assert len(calls) == 1 and found['continuation']
    second = youtube_entries('anything', found['continuation'])
    assert [row['title'] for row in second['results']] == ['Last video']
    assert len(calls) == 2 and second['continuation'] is None


@pytest.mark.parametrize(('kind', 'first_page'), [
    ('songs', True), ('artists', True), ('songs', False), ('artists', False), ('artist', False)])
def test_music_pages_keep_localized_names_and_request_only_the_requested_page(monkeypatch, kind, first_page):
    from ytmusicapi import YTMusic
    from karaoke.video_search import music_entries
    artist_id = 'UC'+'a'*22
    title_run = dict(text='Next song', navigationEndpoint=dict(watchEndpoint=dict(videoId='abcdefghijk')))
    artist_run = dict(text='Singer', navigationEndpoint=dict(browseEndpoint=dict(browseId=artist_id,
        browseEndpointContextSupportedConfigs=dict(browseEndpointContextMusicConfig=dict(pageType='MUSIC_PAGE_TYPE_ARTIST')))))
    row = dict(musicResponsiveListItemRenderer=dict(
        navigationEndpoint=(dict(browseEndpoint=dict(browseId=artist_id)) if kind == 'artists' else dict(watchEndpoint=dict(videoId='abcdefghijk'))),
        flexColumns=[dict(musicResponsiveListItemFlexColumnRenderer=dict(text=dict(runs=[title_run if kind != 'artists' else artist_run]))),
                     dict(musicResponsiveListItemFlexColumnRenderer=dict(text=dict(runs=[artist_run])))],
        overlay=dict(musicItemThumbnailOverlayRenderer=dict(content=dict(musicPlayButtonRenderer=dict(
            playNavigationEndpoint=dict(watchEndpoint=dict(videoId='abcdefghijk'))))))))
    calls = []
    def send(self, endpoint, body, *params):
        calls.append((endpoint, body, params))
        artist_run['text'] = '孙燕姿' if self.language == 'zh_CN' else 'Sun Yanzi'
        if kind == 'artist':
            assert endpoint == 'browse' and body == dict(continuation='second-page')
            return dict(onResponseReceivedActions=[dict(appendContinuationItemsAction=dict(continuationItems=[row]))])
        assert endpoint == 'search' and body['query'] == 'Singer'
        assert bool(body.get('params')) == (kind == 'artists')
        if first_page:
            shelf = dict(musicShelfRenderer=dict(
                title=dict(runs=[dict(text='歌曲' if kind == 'songs' else '歌手')]), contents=[row]))
            sections = dict(sectionListRenderer=dict(contents=[shelf]))
            return dict(contents=dict(tabbedSearchResultsRenderer=dict(tabs=[dict(tabRenderer=dict(content=sections))])))
        assert params == ('&ctoken=second-page&continuation=second-page',)
        return dict(continuationContents=dict(musicShelfContinuation=dict(contents=[row])))
    monkeypatch.setattr(YTMusic, '_send_request', send)
    continuation = dict(token='second-page') if kind == 'artist' else dict(params='&ctoken=second-page&continuation=second-page')
    page = music_entries(kind, artist_id if kind == 'artist' else 'Singer', None if first_page else continuation)
    assert len(calls) == 1 and page['continuation'] is None
    assert len(page['results']) == 1
    assert page['results'][0]['title'] == ('孙燕姿' if kind == 'artists' else 'Next song')
    if kind != 'artists':
        assert page['results'][0]['artists'] == [dict(name='孙燕姿', id=artist_id)]


def test_search_pages_bind_cursors_deduplicate_retries_and_reject_expiry(tmp_path, monkeypatch):
    from karaoke import video_search
    calls = []
    fail = True
    async def fetch(provider, query, continuation=None):
        nonlocal fail
        calls.append((provider, query, continuation))
        if continuation and fail:
            fail = False
            raise RuntimeError('temporary provider failure')
        return dict(results=[dict(title='second' if continuation else 'first')],
                    continuation=None if continuation else dict(token='native-next-page'))
    monkeypatch.setattr(video_search, 'fetch_results', fetch)
    app = create_app(tmp_path, version='test')
    with TestClient(app) as client:
        for path, params in [('/v1/search', dict(provider='youtube', q='Singer')),
                             ('/v1/music/search', dict(kind='songs', q='Singer')),
                             ('/v1/music/artists/UC'+'a'*22, {})]:
            first = client.get(path, params=params).json()
            cursor = first['next_cursor']
            assert cursor and 'continuation' not in first and 'native-next-page' not in cursor
            assert client.get(path, params=params).json() == first
            assert client.get('/v1/music/search', params=dict(kind='artists', q='Singer', cursor=cursor)).status_code == 410
            fail = True
            request = dict(params, cursor=cursor)
            assert client.get(path, params=request).status_code == 502
            second = client.get(path, params=request).json()
            assert second['results'] == [dict(title='second', rank=2)] and second['next_cursor'] is None
            assert client.get(path, params=request).json() == second
            assert client.get(path, params=dict(params, cursor='forged')).status_code == 410
            # Each cursor has an expiry independent of the much shorter page cache.
            from types import SimpleNamespace
            with monkeypatch.context() as clock_patch:
                clock_patch.setattr(video_search, 'time', SimpleNamespace(monotonic=lambda: 1e20))
                assert client.get(path, params=request).status_code == 410
        assert app.state.store.list() == []
        assert len(calls) == 9  # first page, failed continuation, successful retry for each mode


def test_playback_survey_classifies_empty_subtitles_without_rewriting_recorded_attempts():
    from karaoke.tools.playback_sweep import summarize
    event = dict(code='align_words_failed', subtitle_mode='unavailable')
    attempts = [
        dict(outcome='soft_fail', evidence=dict(subtitle_lines=0), soft_failures=[event]),
        dict(outcome='clean', evidence=dict(quality=dict(playback_fallback=dict(subtitle_mode='unavailable')))),
        dict(outcome='soft_fail', evidence=dict(subtitle_lines=1),
             soft_failures=[dict(event, subtitle_mode='estimated_word')]),
        dict(outcome='clean', evidence=dict(subtitle_lines=10)),
        dict(outcome='running')]
    original = json.dumps(attempts, sort_keys=True)
    result = summarize([dict(audio_attempts=[a]) for a in attempts])
    assert result['outcomes'] == dict(hard_fail=2, soft_fail=1, clean=1, running=1)
    assert result['hard_failures'] == dict(subtitles_failed=2)
    assert result['soft_failures'] == dict(align_words_failed=1)
    assert json.dumps(attempts, sort_keys=True) == original


def test_timeline_publication_check_works_without_development_pythonpath(tmp_path):
    from pathlib import Path
    # The real `python -m karaoke.tools.playback_sweep` entry imports a package;
    # PYTHONPATH=karaoke in the legacy test command must not hide script imports.
    root = str(Path(__file__).resolve().parents[1])
    code = f'''import sys
sys.path.insert(0, {root!r})
from karaoke.timing_validation import validate_timeline
cue = dict(text='Hello', start=1., end=2., units=[dict(text='Hello', first=0, stop=5, start=1., end=2.)])
validate_timeline([cue], ['Hello'], 3.)
try:
    validate_timeline([cue], ['Missing text'], 3.)
except ValueError:
    pass
else:
    raise AssertionError('The publication guard was bypassed')
'''
    subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, check=True, capture_output=True)


def test_playback_survey_records_source_disagreements_and_actual_provider_failures():
    from karaoke.tools.playback_sweep import soft_failures
    crosscheck = dict(status='conflicting',sources=[dict(provider='line_music'),dict(provider='lrclib')],
        unavailable_sources=[dict(provider='netease',status='unavailable',http_status=429,error='HTTPStatusError'),
                             dict(provider='youtube_music',status='missing_video_identity')])
    packet = dict(lookup=dict(status='candidate',provider='line_music',crosscheck=crosscheck))
    events = soft_failures({},packet)
    assert [e['code'] for e in events] == ['lyric_source_disagreement','lyric_provider_unavailable']
    assert events[0]['sources'] == ['line_music','lrclib']
    assert events[1]['provider'] == 'netease' and events[1]['http_status'] == 429
    crosscheck['status'] = 'corroborated'
    assert [e['code'] for e in soft_failures({},packet)] == ['lyric_provider_unavailable']
    crosscheck.update(status='insufficient_sources',
        unavailable_sources=[dict(provider='youtube_music',status='unavailable')])
    assert soft_failures({},packet) == []  # A missing second source is not a processing exception.
    packet['lookup']['recording_mismatches'] = ['duration', 'live_performance']
    assert soft_failures({},packet) == [dict(module='lyrics', code='lyric_recording_mismatch',
        provider='line_music', reasons=['duration', 'live_performance'])]
    del packet['lookup']['recording_mismatches']
    recovered = dict(provider='line_music', status='candidate', rejected_candidates=[
        dict(reason='candidate_unavailable', track_id='123', error='ReadTimeout'),
        dict(reason='text_too_short', track_id='456')])
    packet['lookup']['published_lookup'] = recovered
    crosscheck['sources'] = [recovered]
    assert soft_failures({},packet) == [dict(module='lyrics', code='lyric_candidate_unavailable',
        provider='line_music', track_id='123', http_status=None, error_type='ReadTimeout')]
    packet['lookup'].update(identity_recovery=dict(original={'artist':'Latin name'},
        resolved={'artist':'Native name'},recovered=True),
        metadata_language_corrected=dict(original='English',resolved='Chinese'))
    assert [e['code'] for e in soft_failures({},packet)] == [
        'lyric_identity_recovery','lyric_metadata_language_corrected','lyric_candidate_unavailable']
    spans = [dict(line=2,first=0,stop=2,start=3.,end=4.2,timing_source='linear_fallback')]
    review = dict(estimated_spans=spans, pending_review_lines=[3])
    assert soft_failures(dict(quality=dict(timing_review=review)), {}) == [
        dict(module='alignment',code='timing_group_fallback',spans=spans),
        dict(module='alignment',code='timing_review_unconfirmed',lines=[3])]


@pytest.mark.parametrize('hard_code', ['separate_music_failed', 'subtitles_failed'])
@pytest.mark.parametrize('has_full_mv', [False, True])
@pytest.mark.parametrize('targeted', [False, True])
@pytest.mark.parametrize('mode', ['serial', 'parallel', 'cache_limited', 'pause', 'skip_large', 'unlimited'])
def test_playback_survey_runs_each_case_once_and_keeps_user_queue(tmp_path, monkeypatch, hard_code, has_full_mv, targeted, mode):
    from contextlib import nullcontext
    from karaoke.tools import playback_sweep as sweep
    data = tmp_path/'data'
    data.mkdir()
    (data/'preparation.pause').write_text('test survey')
    app = create_app(data, version='one-version')
    if mode == 'unlimited':
        from types import SimpleNamespace
        monkeypatch.setattr('karaoke.cache.shutil.disk_usage', lambda path: SimpleNamespace(free=1000*GIB))
        app.state.store.cache.set_limit(None)
    cases = [dict(url='https://www.youtube.com/watch?v='+key, language='English', title=key)
             for key in ['aaaaaaaaaaa','bbbbbbbbbbb','ccccccccccc']]
    selection = [dict(url=c['url'], language=c['language']) for c in cases]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=cases, selection_sha256=hashlib.sha256(
        json.dumps(selection,sort_keys=True).encode()).hexdigest())))
    original_manifest = manifest.read_bytes()
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    source_jobs = []
    for case in cases:
        job, _ = app.state.store.create(JobRequest(url=case['url']), 'old')
        source_jobs.append(job['id'])
        folder = data/job['id']
        (folder/'download').mkdir()
        media = folder/'download/source.mkv'
        media.write_bytes(case['url'].encode())
        (folder/'source.json').write_text(json.dumps(dict(url=case['url'], sha256=file_hash(media))))
        (folder/'download/source.info.json').write_text('{}')
        app.state.store.update(job['id'], status='source_ready')
        (prepared/(hashlib.sha256(case['url'].encode()).hexdigest()+'.json')).write_text(json.dumps(
            dict(seconds=1, source_sha256=file_hash(media))))
    calls = []
    import threading
    barrier = threading.Barrier(3 if mode == 'unlimited' and not targeted else 2)
    counter_lock = threading.Lock()
    active, maximum = 0, 0
    peak = 20*GIB if mode == 'unlimited' else app.state.store.cache.limit if mode == 'cache_limited' else 1024**2
    monkeypatch.setattr(sweep, 'working_bytes', lambda store, case:
        store.cache.limit if mode == 'skip_large' and case['url'] == cases[1]['url'] else peak)
    def process(client, base, folder, command, **kwargs):
        nonlocal active, maximum
        with counter_lock:
            calls.append(folder.name)
            active += 1
            maximum = max(active, maximum)
            number = len(calls)
        if mode == 'unlimited' or (mode in ('parallel', 'pause', 'skip_large') and number <= 2):
            barrier.wait(timeout=5)
            if mode == 'pause':
                report.with_suffix('.pause').touch()
        else:
            time.sleep(.05)
        with counter_lock:
            active -= 1
        url = json.loads((folder/'request.json').read_text())['url']
        if url == cases[1]['url']:
            (folder/'error.json').write_text(json.dumps(dict(code=hard_code, message='test')))
            return 1
        quality = dict(playback_fallback=dict(error=dict(code='align_words_failed', message='test'),
            subtitle_mode='estimated_word', estimated_spans=[{}])) if url == cases[2]['url'] else {}
        (folder/'audio-result.json').write_text(json.dumps(dict(title='Song', artist='Singer',
            language='English', duration_seconds=10, quality=quality)))
        return 0
    monkeypatch.setattr(sweep, 'run_probe', process)
    monkeypatch.setattr(sweep, 'verify_outputs', lambda *a,**k:dict(validation_scope='playback_audio'))
    monkeypatch.setattr(sweep, 'record_failure', lambda *a:None)
    report = tmp_path/'report.json'
    requested = [cases[0]['url'], cases[2]['url']] if targeted else []
    with TestClient(app) as api:
        health_disconnected = False
        if mode == 'parallel':
            import httpx
            original_get = api.get
            def interrupted_health(url, *args, **kwargs):
                nonlocal health_disconnected
                if str(url).endswith('/health') and calls and not health_disconnected:
                    health_disconnected = True
                    raise httpx.ReadError('Connection reset by peer')
                return original_get(url, *args, **kwargs)
            monkeypatch.setattr(api, 'get', interrupted_health)
        if has_full_mv:
            app.state.store.update(source_jobs[0], status='completed', result=dict(artifacts={}))
        user_queue = api.post('/v1/room/entries',json=dict(url=URL)).json()['entries']
        monkeypatch.setattr(sweep.httpx,'Client',lambda **k:nullcontext(api))
        concurrency = 1 if mode == 'serial' else 4 if mode == 'unlimited' else 2
        sweep.run_once(manifest,data,report,prepared,'http://testserver',case_urls=requested, concurrency=concurrency)
        sweep.run_once(manifest,data,report,prepared,'http://testserver',case_urls=requested, concurrency=concurrency)
        assert len(calls) == (2 if targeted or mode == 'pause' else 3)
        if mode == 'parallel':
            assert health_disconnected
        # Worker admission order does not determine thread start order.
        called = {json.loads((data/job/'request.json').read_text())['url']:job for job in calls}
        first_job = called[cases[0]['url']]
        assert set(calls)-{first_job} == set(source_jobs[2:] if targeted else source_jobs[1:2] if mode == 'pause' else source_jobs[1:])
        assert maximum == ((2 if targeted else 3) if mode == 'unlimited' else 2 if mode in ('parallel', 'pause', 'skip_large') else 1)
        if mode == 'skip_large' and not targeted:
            assert calls.index(source_jobs[2]) < calls.index(source_jobs[1])
        assert (first_job != source_jobs[0]) == has_full_mv
        if has_full_mv:
            assert app.state.store.read(source_jobs[0])['status'] == 'completed'
        assert [e['id'] for e in api.get('/v1/room').json()['entries']] == [e['id'] for e in user_queue]
        result = json.loads(report.read_text())
        assert result['fixed_selection_size'] == 3
        assert [c['url'] for c in result['cases']] == (requested or [c['url'] for c in cases])
        assert result['summary']['outcomes'] == (dict(clean=1,soft_fail=1) if targeted else
            dict(clean=1,hard_fail=1,pending=1) if mode == 'pause' else dict(clean=1,hard_fail=1,soft_fail=1))
        assert result['summary']['hard_failures'] == ({} if targeted else {hard_code:1})
        assert result['summary']['soft_failures'] == ({} if mode == 'pause' and not targeted else dict(align_words_failed=1))
        assert all(len(c['audio_attempts']) == (0 if mode == 'pause' and not targeted and c['url'] == cases[2]['url'] else 1) for c in result['cases'])
        with pytest.raises(ValueError,match='requested cases'):
            sweep.run_once(manifest,data,report,prepared,'http://testserver',case_urls=[cases[1]['url']])
        with pytest.raises(ValueError,match='outside the fixed selection'):
            sweep.run_once(manifest,data,report,prepared,'http://testserver',case_urls=[URL])
        result['cases'][0]['audio_attempts'][0]['outcome']='running'
        report.write_text(json.dumps(result))
        with pytest.raises(ValueError,match='Reconcile'):
            sweep.run_once(manifest,data,report,prepared,'http://testserver',case_urls=requested, concurrency=1 if mode == 'serial' else 2)
        assert len(calls) == (2 if targeted or mode == 'pause' else 3)
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize('budget', ['normal', 'quota_full', 'unlimited', 'disk_full', 'transient_disconnect', 'disconnected'])
def test_probe_session_overlaps_children_and_stops_them_together(tmp_path, budget):
    from concurrent.futures import ThreadPoolExecutor
    import httpx
    import os
    from karaoke.tools.soak import ProbeSession, run_probe
    folders = [tmp_path/name for name in ('one', 'two')]
    for folder in folders:
        folder.mkdir()
        (folder/'checkpoint').write_bytes(b'keep')
    pids = [folder/'pid' for folder in folders]
    disconnected = False
    def respond(request):
        nonlocal disconnected
        if request.url.path == '/health':
            data = dict(preparation_paused=True, running_job=None)
        elif request.url.path == '/v1/storage':
            both = all(p.exists() for p in pids)
            if both and (budget == 'disconnected' or (budget == 'transient_disconnect' and not disconnected)):
                disconnected = True
                raise httpx.ReadError('Connection reset by peer', request=request)
            data = dict(used_bytes=200 if both else 20,
                limit_bytes=None if budget in ('unlimited', 'disk_full') else 100 if budget == 'quota_full' else 1000,
                can_prepare=not (budget == 'disk_full' and both))
        else:
            data = dict(status='queued')
        return httpx.Response(200, json=data)
    def command(folder):
        code = ('import os,pathlib,time;'
                f'pathlib.Path({str(folder/"pid")!r}).write_text(str(os.getpid()));'
                'time.sleep(2)')
        return [sys.executable, '-c', code]
    with httpx.Client(transport=httpx.MockTransport(respond)) as client, ProbeSession(tmp_path) as session:
        with pytest.raises(BlockingIOError):
            run_probe(client, 'http://test', folders[0], command(folders[0]))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_probe, client, 'http://test', folder, command(folder), session=session)
                       for folder in folders]
            for future in futures:
                if budget in ('quota_full', 'disk_full', 'disconnected'):
                    expected = (ValueError, httpx.ReadError) if budget == 'disconnected' else ValueError
                    with pytest.raises(expected, match='cache or free-space|infrastructure failure|Connection reset'):
                        future.result(timeout=10)
                else:
                    assert future.result(timeout=10) == 0
    assert all(p.exists() for p in pids)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid.read_text()), 0)
    assert all((folder/'checkpoint').read_bytes() == b'keep' for folder in folders)
    if budget == 'transient_disconnect':
        assert any('Probe monitor retry after ReadError' in (folder/'probe.log').read_text() for folder in folders)
    with pytest.raises(ValueError, match='does not own'):
        session.check(folders[0])


@pytest.mark.parametrize('failed_endpoint', [None, '/health', '/v1/storage'])
def test_probe_session_shares_fresh_monitoring_with_the_controller(tmp_path, monkeypatch, failed_endpoint):
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    import httpx
    from karaoke.tools import soak
    now = [0.]
    monkeypatch.setattr(soak.time, 'monotonic', lambda: now[0])
    calls = Counter()
    state = dict(version='before', full=False)
    def respond(request):
        path = request.url.path
        calls[path] += 1
        assert request.method == 'GET'
        if path == failed_endpoint and calls[path] == 1:
            raise httpx.ReadError('Connection reset by peer', request=request)
        data = (dict(used_bytes=10, limit_bytes=None, can_prepare=not state['full'])
                if path == '/v1/storage' else dict(pipeline_version=state['version']))
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client, soak.ProbeSession(tmp_path) as session:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: session.check_resources(client, 'http://test'), range(8)))
        assert all(health['pipeline_version'] == 'before' for health, _ in results)
        assert calls == {'/v1/storage': 2 if failed_endpoint else 1,
                         '/health': 2 if failed_endpoint == '/health' else 1}
        before = calls.copy()
        state['version'] = 'after'
        now[0] = .5
        assert session.check_resources(client, 'http://test')[0]['pipeline_version'] == 'before'
        assert calls == before
        # Admission and final acceptance need current state, even within a sample interval.
        assert session.check_resources(client, 'http://test', refresh=True)[0]['pipeline_version'] == 'after'
        assert calls == before + Counter({'/health': 1, '/v1/storage': 1})
        state['full'] = True
        now[0] = 2.
        with pytest.raises(ValueError, match='cache or free-space'):
            session.check_resources(client, 'http://test')
        assert session.cancelled.is_set()
        state['full'] = False
        with pytest.raises(ValueError, match='infrastructure failure'):
            session.check_resources(client, 'http://test')
