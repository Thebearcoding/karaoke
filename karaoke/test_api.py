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
        assert attempt['status'] == 'external' and attempt['next_retry_at'] > time.time()+7100
        assert eligible(case, 'v2', attempt['next_retry_at']+1)
        attempt['external_attempt'] = 3
        assert not eligible(case, 'v3', attempt['next_retry_at']+1)


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
    (folder/'repeated-lyrics').mkdir()
    (folder/'repeated-lyrics/run.json').write_text('{"attempts": []}')
    (folder/'vocals.mp3').write_bytes(b'private audio')
    attempt = dict(external_attempt=1)
    error = dict(code='align_repeats_failed', message='Generic stage failure')
    record_failure(attempt, dict(id='job', error=error), folder)
    shutil.rmtree(folder)

    with gzip.open(attempt['reproduction_file'], 'rt') as stream:
        saved = json.load(stream)
    assert saved['align_repeats.log'].endswith(traceback)
    assert len(saved['align_repeats.log'].encode()) <= 16384
    assert saved['asr/transcription.json'] == {'language': 'chinese'}
    assert saved['repeated-lyrics/run.json'] == {'attempts': []}
    assert 'vocals.mp3' not in saved
    assert traceback in attempt['diagnostic_tail']
    assert attempt['status'] == 'failed' and attempt['error'] == error


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
            pipeline_version='v2', submitted_at=f'2026-10-01T00:00:0{1-i}Z', entry_id=str(i))])
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
            data = dict(id=job_id, status='failed', error={'code': 'processing_failed', 'message': 'test'})
        return httpx.Response(200, json=data)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(soak.httpx, 'Client', lambda **kwargs: client)
    soak.run(manifest, tmp_path, 'http://test')
    assert observed == ['1', '0']
    saved = json.loads(manifest.read_text())['cases']
    assert [c['url'] for c in saved] == [c['url'] for c in cases]
    assert all(c['attempts'][-1]['status'] == 'failed' for c in saved)


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


def test_language_classification_uses_audio_metadata_not_subtitle_language():
    metadata = dict(title='English song (日本語字幕)', subtitles={'ja': []})
    assert 'language' not in video_song_info(metadata)
    metadata['requested_formats'] = [dict(acodec='none', language='ja'), dict(acodec='opus', language='en-US')]
    assert video_song_info(metadata)['language'] == 'English'


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


def test_shutdown_stops_current_job_and_preserves_the_queue(tmp_path):
    def slow_worker(folder):
        return [sys.executable, '-c', "import time;time.sleep(30)"]
    app = create_app(tmp_path, worker_command=slow_worker, version='test')
    with TestClient(app) as client:
        current = client.post('/v1/jobs', json={'url': URL}).json()['id']
        queued = client.post('/v1/jobs', json={'url': 'https://youtu.be/2Vv-BfVoq4g'}).json()['id']
        deadline = time.monotonic()+5
        while app.state.worker.process is None and time.monotonic() < deadline:
            time.sleep(.02)
        process = app.state.worker.process
        assert process is not None
        assert client.get(f'/v1/jobs/{queued}').json()['status'] == 'queued'
    assert process.returncode is not None
    assert app.state.store.read(current)['error']['code'] == 'interrupted'
    assert app.state.store.read(queued)['status'] == 'queued'
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        assert finished(client, queued)['status'] == 'completed'


def test_successful_process_without_required_outputs_is_not_completed(tmp_path):
    def incomplete_worker(folder):
        return [sys.executable, '-c',
            "import pathlib,sys;pathlib.Path(sys.argv[1],'result.json').write_text('{\"artifacts\": {}}')", str(folder)]
    with TestClient(create_app(tmp_path, worker_command=incomplete_worker, version='test')) as client:
        job_id = client.post('/v1/jobs', json={'url': URL}).json()['id']
        assert finished(client, job_id)['status'] == 'failed'


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
    ids = [store.create(request, version)[0]['id'] for request, version in cases]
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
    parsed = SongText(title='Detected', artist='Singer', language='English', lines=['One short line', 'One short line'])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], model='test-model', usage=None)
    with patch('karaoke.song_sources.OpenAI') as client:
        client.return_value.beta.chat.completions.parse.return_value = response
        result = prepare_song_text(dict(url=URL, title='User title'), {},
            dict(language='english', segments=[dict(text='One short line', start=1, end=3)]),
            120, catalog_root=None, model='test-model')
    assert result['title'] == 'User title' and result['lines'] == ['One short line', 'One short line']
    assert result['provenance']['needs_review'] is True


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
    from karaoke.three_stems import encode_preview

    tone = .1*np.sin(2*np.pi*440*np.arange(frames)/rate)
    source, output = tmp_path/'master.wav', tmp_path/'preview.mp3'
    sf.write(source, np.repeat(tone[:,None], channels, axis=1), rate, subtype='DOUBLE')
    report = encode_preview(source, output, .5, frames, channels)
    assert report['decoded_length_delta_samples'] == 0
    assert report['padding_peak'] == 0
    assert .04 < report['decoded_peak'] < .06
    assert sf.info(source).frames == frames and sf.info(source).subtype == 'DOUBLE'


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
                                        'short_window_recovery', 'short_tail_recovery'])
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
    data = dict(language='chinese', segments=[dict(start=i*4, end=i*4+3, text='唱到这一句，接着唱下一句') for i in range(8)],
                words=[dict(start=1,end=1,word='一'), dict(start=1,end=1,word='起'),
                       dict(start=1,end=2,word='唱'), dict(start=1.9,end=2.5,word='歌')])
    assert not repetition_loop(data)
    audio = tmp_path/'audio.mp3'
    audio.write_bytes(b'input')
    with patch('karaoke.transcribe_lyrics.OpenAI') as client:
        client.audio.transcriptions.create.return_value.model_dump.return_value = data
        actual, recovery = transcribe(client, audio, language, '', tmp_path)
        assert actual == data and not recovery['triggered']
        client.audio.transcriptions.create.assert_called_once()


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
