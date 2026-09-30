import asyncio
import sys
import time

from fastapi.testclient import TestClient

from karaoke.api import Store, Worker, create_app
from karaoke.job_schema import JobRequest
from karaoke.room import Updates
from karaoke.test_api import URL, fake_worker, finished


def ready_job(store, url=URL):
    job, _ = store.create(JobRequest(url=url), 'test')
    folder = store.root/job['id']
    artifacts = {}
    for name in ['mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview']:
        (folder/name).write_bytes(b'media')
        artifacts[name] = dict(path=name, bytes=5, sha256='fixture')
    return store.update(job['id'], status='completed', result=dict(title='Test song', artist='Singer',
                        duration_seconds=10, artifacts=artifacts))


def test_playlist_duplicate_song_order_and_stale_end_event(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        ids = []
        for _ in range(3):
            data = client.post('/v1/room/entries', json={'job_id':job['id']}).json()
            ids.append(data['entries'][-1]['id'])
        assert len(set(ids)) == 3
        assert len(app.state.store.list()) == 1
        first = client.post('/v1/room/commands', json={'action':'play'}).json()
        assert first['current_id'] == ids[0] and first['started_at'] is None
        playing = client.post('/v1/room/commands', json={'action':'ready','entry_id':ids[0]}).json()
        assert playing['started_at'] is not None
        response = client.post('/v1/room/commands', json={'action':'reorder','order':[ids[0],ids[2],ids[1]],'revision':playing['revision']})
        assert response.status_code == 200
        invalid = client.post('/v1/room/commands', json={'action':'reorder','order':[ids[1],ids[0],ids[2]]})
        assert invalid.status_code == 409
        following = client.post('/v1/room/commands', json={'action':'next','entry_id':ids[0]}).json()
        assert following['current_id'] == ids[2]
        stale = client.post('/v1/room/commands', json={'action':'next','entry_id':ids[0]})
        assert stale.status_code == 409
        assert client.get('/v1/room').json()['current_id'] == ids[2]
        assert client.post('/v1/room/commands', json={'action':'reorder','order':[ids[2],ids[2]]}).status_code == 422


def test_wait_for_preparation_then_play_pause_seek_mix_and_persist(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        data = client.post('/v1/room/entries', json={'url':URL,'title':'New song'}).json()
        entry = data['entries'][0]
        client.post('/v1/room/commands', json={'action':'play'})
        assert client.get('/v1/room').json()['started_at'] is None
        assert finished(client,entry['job_id'])['status'] == 'completed'
        # A ready audio renderer, not elapsed preparation time, starts playback.
        assert client.get('/v1/room').json()['started_at'] is None
        app.state.store.update(entry['job_id'], result={**app.state.store.read(entry['job_id'])['result'], 'duration_seconds':10})
        client.post('/v1/room/commands', json={'action':'ready','entry_id':entry['id']})
        client.post('/v1/room/commands', json={'action':'seek','entry_id':entry['id'],'position':6})
        paused = client.post('/v1/room/commands', json={'action':'pause','entry_id':entry['id']}).json()
        assert 6 <= paused['position'] < 7 and not paused['playing']
        mixed = client.post('/v1/room/commands', json={'action':'mix','mix':'instrumental'}).json()
        assert mixed['position'] == paused['position'] and mixed['mix'] == 'instrumental'
        assert client.post('/v1/room/commands', json={'action':'seek','entry_id':'old','position':9}).status_code == 409
    with TestClient(create_app(tmp_path, worker_command=fake_worker, version='test')) as client:
        restored = client.get('/v1/room').json()
        assert restored['current_id'] == entry['id'] and restored['position'] == paused['position']
        assert restored['mix'] == 'instrumental' and not restored['playing']
        empty = client.post('/v1/room/commands', json={'action':'remove','entry_id':entry['id']}).json()
        assert empty['entries'] == [] and empty['current_id'] is None


def test_independent_levels_preserve_playback_persist_and_presets_reset_them(tmp_path):
    app = create_app(tmp_path, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        client.post('/v1/room/commands', json={'action':'play'})
        before = client.post('/v1/room/commands', json={'action':'ready','entry_id':entry['id']}).json()
        changed = client.post('/v1/room/commands', json={'action':'levels','levels':{'backing_vocals':.35}}).json()
        assert changed['levels'] == dict(instrumental=1,backing_vocals=.35,lead_vocals=0)
        assert changed['started_at'] == before['started_at'] and changed['position'] == before['position']
        changed = client.post('/v1/room/commands', json={'action':'levels','levels':{'instrumental':.8,'lead_vocals':.2}}).json()
        assert changed['mix'] == 'custom' and changed['levels']['backing_vocals'] == .35
        assert client.post('/v1/room/commands', json={'action':'levels','levels':{'lead_vocals':1.1}}).status_code == 422
        assert client.post('/v1/room/commands', json={'action':'levels','levels':{}}).status_code == 422
    with TestClient(create_app(tmp_path, version='test')) as client:
        assert client.get('/v1/room').json()['levels'] == changed['levels']
        original = client.post('/v1/room/commands', json={'action':'mix','mix':'original'}).json()
        assert original['levels'] == dict(instrumental=1,backing_vocals=1,lead_vocals=1)
        muted = client.post('/v1/room/commands', json={'action':'levels', 'levels':dict(instrumental=0,backing_vocals=0,lead_vocals=0)}).json()
        assert all(value == 0 for value in muted['levels'].values())


def test_pending_preparation_follows_playlist_without_interrupting_current(tmp_path):
    store = Store(tmp_path)
    worker = Worker(store)
    worker.current = 'already-running'
    for job in ['a','b','c','standalone']:
        worker.queue.put_nowait(job)
    worker.prioritize(['c','a','c','b'])
    assert [worker.queue.get_nowait() for _ in range(4)] == ['c','a','b','standalone']
    assert worker.current == 'already-running'


def test_restored_playlist_priority_is_applied_before_starting_worker(tmp_path):
    app = create_app(tmp_path, worker_command=lambda folder: [sys.executable, '-c', 'import time;time.sleep(30)'], version='test')
    first, _ = app.state.store.create(JobRequest(url=URL), 'test')
    second, _ = app.state.store.create(JobRequest(url='https://youtu.be/2Vv-BfVoq4g'), 'test')
    state = app.state.room.read()
    state['entries'] = [dict(id='second',job_id=second['id']), dict(id='first',job_id=first['id'])]
    app.state.room.save(state)
    with TestClient(app) as client:
        assert client.get('/health').json()['running_job'] == second['id']
        assert app.state.store.read(first['id'])['status'] == 'queued'


def test_removed_pending_song_does_not_create_a_new_job(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        assert client.post('/v1/room/entries',json={'url':'http://127.0.0.1/secret'}).status_code == 422
        assert client.post('/v1/room/entries',json={'job_id':'missing'}).status_code == 404
        data = client.post('/v1/room/entries',json={'url':URL}).json()
        entry = data['entries'][0]
        client.post('/v1/room/commands',json={'action':'remove','entry_id':entry['id']})
        assert finished(client,entry['job_id'])['status'] == 'completed'
        assert client.get('/v1/room').json()['entries'] == []
        assert client.get('/').status_code == 200
        assert client.get('/assets/app.js').status_code == 200


def test_sse_initial_snapshot_push_coalescing_and_disconnect():
    async def check():
        updates = Updates()
        data = {'revision':0}
        stream = updates.events(lambda: data.copy())
        assert '"revision": 0' in await anext(stream)
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        data['revision'] = 1
        updates.publish()
        assert '"revision": 1' in await asyncio.wait_for(waiting,1)
        for n in range(2,50):
            data['revision'] = n
            updates.publish()
        assert '"revision": 49' in await anext(stream)
        await stream.aclose()
        assert not updates.subscribers
    asyncio.run(check())


def test_progress_is_received_before_worker_exits(tmp_path):
    def event_worker(folder):
        return [sys.executable,'-u','-c',
                'import time;print(\'KARAOKE_PROGRESS {"active_stages":["transcribe"],"timings":{}}\',flush=True);time.sleep(30)']
    with TestClient(create_app(tmp_path, worker_command=event_worker, version='test')) as client:
        job = client.post('/v1/jobs',json={'url':URL}).json()
        end = time.monotonic()+3
        while time.monotonic() < end:
            state = client.get('/v1/jobs/'+job['id']).json()
            if state['active_stages'] == ['transcribe']:
                break
            time.sleep(.01)
        assert state['active_stages'] == ['transcribe'] and state['status'] == 'running'
