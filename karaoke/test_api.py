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
from karaoke.cache import CacheFull, GIB
from karaoke.pipeline import Pipeline, StageFailure, create_poster
from karaoke.stage_progress import PREFIX, parse_progress, progress_items


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
            pipeline.run('separate_lead', [sys.executable, '-c', code, str(release)], timeout=5)
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
        pipeline.run('align_words', [sys.executable, '-c',
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


def cache_fixture(store, suffix):
    job, _ = store.create(JobRequest(url='https://youtu.be/0000000000'+suffix, title=suffix, force=True), 'test')
    (store.root/job['id']/'mv.mp4').write_bytes(b'x'*2048)
    return store.update(job['id'], status='completed', last_accessed_at='2020-01-0'+suffix,
        result=dict(title=suffix, duration_seconds=10,
                    artifacts={'mv':dict(path='mv.mp4',bytes=2048,sha256='fixture')}))


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
    store.cache.compact(store.read(job['id']))
    assert not (folder/'mv.mp4').exists()
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
        duration='1:02:03', pic='//i0.hdslb.com/cover.jpg', author='Uploader')])
    assert bili == [dict(url='https://www.bilibili.com/video/BV1bK411W797', title='歌曲 & 现场',
        duration=3723, thumbnail='https://i0.hdslb.com/cover.jpg', uploader='Uploader', provider='bilibili')]
    item = dict(url=URL, title='Video title', duration=180, thumbnails=[{'url':'https://i.ytimg.com/cover.jpg'}])
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
