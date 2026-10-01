import asyncio
import json
import sys
import time

from fastapi.testclient import TestClient

from karaoke.api import Store, Worker, create_app
from karaoke.job_schema import JobRequest
from karaoke.room import AddSong, RoomCommand, Updates
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


def test_poster_survives_cleanup_and_browsing_does_not_refresh_lru(tmp_path):
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job = ready_job(store)
    folder = store.root/job['id']
    (folder/'poster.jpg').write_bytes(b'cover')
    job['result']['artifacts']['poster'] = dict(path='poster.jpg', bytes=5, sha256='fixture')
    store.update(job['id'], result=job['result'], last_accessed_at='2000-01-01T00:00:00Z')
    store.cache.compact(store.read(job['id']))
    with TestClient(app) as client:
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['job_id'] == job['id'])
        poster = song['poster']
        assert poster == f"/v1/jobs/{job['id']}/artifacts/poster"
        response = client.get(poster)
        assert response.status_code == 200 and response.content == b'cover'
        assert response.headers['content-type'] == 'image/jpeg'
        assert store.read(job['id'])['last_accessed_at'] == '2000-01-01T00:00:00Z'
        assert not store.cache.leases
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert entry['poster'] == poster
        store.update(job['id'], status='evicted')
        assert client.get(poster).status_code == 410
        song = next(s for s in client.get('/v1/library').json()['songs'] if s['id'] == job['id'])
        assert song['poster'] is None and song['job_id'] is None


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


def test_soak_adds_all_cases_to_room_and_only_removes_finished_owned_entries(tmp_path):
    from karaoke.tools.soak import queue_cases, release_finished_entries
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    user_job = ready_job(app.state.store)
    state = {'provider_backoff': {}, 'cases': [dict(url=f'https://www.youtube.com/watch?v={i:011d}',
        title=f'Test {i}', provider='youtube', attempts=[]) for i in range(3)]}
    with TestClient(app) as client:
        user = client.post('/v1/room/entries', json={'job_id': user_job['id']}).json()['entries'][0]
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        entries = client.get('/v1/room').json()['entries']
        assert len(entries) == 4  # The abandoned queue=2 idea must not become an app limit.
        attempts = [c['attempts'][0] for c in state['cases']]
        assert {a['entry_id'] for a in attempts} == {e['id'] for e in entries[1:]}
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        assert len(client.get('/v1/room').json()['entries']) == 4
        attempts[0]['status'] = 'passed'
        attempts[1]['status'] = 'failed'
        client.post('/v1/room/commands', json={'action':'select','entry_id':attempts[0]['entry_id']})
        release_finished_entries(client, 'http://testserver', state)
        remaining = {e['id'] for e in client.get('/v1/room').json()['entries']}
        assert remaining == {user['id'], attempts[0]['entry_id'], attempts[2]['entry_id']}


def test_room_can_force_full_revalidation_without_reusing_a_completed_job(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    old = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'url':URL,'force':True}).json()['entries'][0]
        assert entry['job_id'] != old['id']
        assert finished(client, entry['job_id'])['status'] == 'completed'


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
        shifted = client.post('/v1/room/commands', json={'action':'pitch','pitch_semitones':3}).json()
        assert shifted['pitch_semitones'] == 3
        assert shifted['started_at'] == changed['started_at'] and shifted['position'] == changed['position']
        assert shifted['levels'] == changed['levels']
        for value in (-7, 7, .5, True, None):
            assert client.post('/v1/room/commands', json={'action':'pitch','pitch_semitones':value}).status_code == 422
    with TestClient(create_app(tmp_path, version='test')) as client:
        assert client.get('/v1/room').json()['levels'] == changed['levels']
        assert client.get('/v1/room').json()['pitch_semitones'] == 3
        original = client.post('/v1/room/commands', json={'action':'mix','mix':'original'}).json()
        assert original['levels'] == dict(instrumental=1,backing_vocals=1,lead_vocals=1)
        assert original['pitch_semitones'] == 3
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


def test_library_lists_queued_and_running_songs_and_reuses_their_jobs(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room, worker = app.state.store, app.state.room, app.state.worker
    job, _ = store.create(JobRequest(url=URL, language='en'), 'test')
    worker.queue.put_nowait(job['id'])
    room.add(AddSong(job_id=job['id']))
    snapshot = room.add(AddSong(job_id=job['id']))
    song = next(s for s in snapshot['library'] if s['url'] == URL)
    assert song['status'] == 'queued' and song['language'] == 'English'
    assert song['job_id'] == job['id'] and worker.queue.qsize() == 1
    assert len(store.list()) == 1 and len(snapshot['entries']) == 2
    counters = {'download': dict(completed=50, total=100, unit='bytes')}
    store.update(job['id'], status='running', active_stages=['download'], stage_progress=counters,
                 song_info=dict(title='Video title', artist='Artist'))
    song = next(s for s in room.library() if s['url'] == URL)
    assert song['title'] == 'Video title' and song['stage_progress'] == counters
    assert song['status'] == 'running'  # Reusable job ID does not mean ready to play.
    room.add(AddSong(job_id=song['job_id']))
    assert len(store.list()) == 1


def test_library_keeps_one_card_across_failed_jobs_and_retries(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    job, _ = store.create(JobRequest(url=URL, title='Song'), 'test')
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    store.update(job['id'], status='failed', song_info=dict(language='Japanese'),
                 error=dict(code='test', message='Failed'))
    song = next(s for s in room.library() if s['url'] == URL)
    assert song['status'] == 'failed' and song['language'] == 'Japanese'
    assert song['job_id'] is None
    snapshot = room.command(RoomCommand(action='retry', entry_id=first['id']))
    songs = [s for s in snapshot['library'] if s['url'] == URL]
    assert len(songs) == 1 and songs[0]['status'] == 'queued'
    assert songs[0]['job_id'] != job['id']


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
    info = dict(title='New video title', artist='Singer', duration_seconds=123)
    counters = {'separate_lead': dict(completed=3, total=10, unit='chunks')}
    def event_worker(folder):
        return [sys.executable,'-u','-c',
                'import sys,time;print("KARAOKE_PROGRESS "+sys.argv[1],flush=True);'
                'print("KARAOKE_PROGRESS "+sys.argv[2],flush=True);time.sleep(30)',
                json.dumps(dict(song_info=info)), json.dumps(dict(active_stages=['transcribe', 'separate_lead'],
                    stage_progress=counters, timings={}))]
    with TestClient(create_app(tmp_path, worker_command=event_worker, version='test')) as client:
        entry = client.post('/v1/room/entries',json={'url':URL}).json()['entries'][0]
        end = time.monotonic()+3
        while time.monotonic() < end:
            state = client.get('/v1/jobs/'+entry['job_id']).json()
            if state.get('stage_progress') == counters:
                break
            time.sleep(.01)
        assert state['active_stages'] == ['transcribe', 'separate_lead'] and state['status'] == 'running'
        # A later stage-only event must not erase the metadata or alter the input.
        assert state['song_info'] == info and state['result'] is None
        assert state['input']['title'] is None and state['input']['artist'] is None
        visible = client.get('/v1/room').json()['entries'][0]
        assert (visible['title'], visible['artist'], visible['duration']) == ('New video title', 'Singer', 123)
        assert visible['stage_progress'] == counters


def test_song_info_updates_push_to_room_and_preserve_user_overrides(tmp_path):
    async def check():
        app = create_app(tmp_path, version='test')
        room, store = app.state.room, app.state.store
        job, _ = store.create(JobRequest(url=URL), 'test')
        state = room.read()
        state['entries'] = [dict(id='entry', job_id=job['id'])]
        room.save(state)
        stream = room.updates.events(room.snapshot)

        def entry(event):
            return json.loads(event.split('data: ', 1)[1])['entries'][0]

        assert entry(await anext(stream))['title'] == '新点的歌'
        for done, title in enumerate(('Video title', 'Resolved song title'), 1):
            counters = {'separate_music': dict(completed=done, total=4, unit='chunks')}
            store.update(job['id'], status='running', song_info=dict(title=title, artist='Singer', language='English'),
                         active_stages=['separate_music'], stage_progress=counters)
            event = await asyncio.wait_for(anext(stream), 1)
            visible = entry(event)
            assert visible['title'] == title and visible['stage_progress'] == counters
            song = next(s for s in json.loads(event.split('data: ', 1)[1])['library'] if s['url'] == URL)
            assert song['title'] == title and song['language'] == 'English'
            assert song['stage_progress'] == counters
        store.update(job['id'], status='failed', error=dict(code='test', message='Test failure'))
        visible = entry(await anext(stream))
        assert visible['title'] == 'Resolved song title' and visible['stage_progress'] == {}
        store.update(job['id'], status='completed', result=dict(title='Final title', artist='Final singer'))
        assert entry(await anext(stream))['title'] == 'Final title'
        await stream.aclose()

        manual, _ = store.create(JobRequest(url=URL, title='My title', artist='My singer'), 'test')
        store.update(manual['id'], song_info=dict(title='Detected title', artist='Detected singer'),
                     result=dict(title='Final title', artist='Final singer'))
        visible = room.describe(dict(id='manual', job_id=manual['id']))
        assert (visible['title'], visible['artist']) == ('My title', 'My singer')
    asyncio.run(check())


def test_search_title_is_immediate_but_does_not_override_metadata_or_cache(tmp_path):
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'url':URL, 'source_title':'Video title (Official MV)'}).json()['entries'][0]
        assert entry['title'] == 'Video title (Official MV)'
        job = app.state.store.read(entry['job_id'])
        assert job['input']['title'] is None
        assert finished(client, job['id'])['status'] == 'completed'
        app.state.store.update(job['id'], song_info={'title':'Clean song title', 'artist':'Singer'})
        room = client.get('/v1/room').json()
        assert room['entries'][0]['title'] == 'Clean song title'
        assert room['library'][0]['title'] == 'Clean song title'
        again = client.post('/v1/room/entries', json={'url':URL, 'source_title':'A different search title'}).json()
        assert again['entries'][-1]['job_id'] == job['id']
        assert again['entries'][-1]['title'] == 'Clean song title'
        assert len(app.state.store.list()) == 1
