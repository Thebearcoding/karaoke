import asyncio
import json
import sys
import time

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from karaoke.api import Store, Worker, create_app
from karaoke.job_schema import JobRequest
from karaoke.room import AddSong, Room, RoomCommand, Updates
from karaoke.test_api import URL, cache_fixture, fake_worker, finished


def ready_job(store, url=URL):
    job, _ = store.create(JobRequest(url=url), 'test')
    folder = store.root/job['id']
    artifacts = {}
    for name in ['mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview']:
        (folder/name).write_bytes(b'media')
        artifacts[name] = dict(path=name, bytes=5, sha256='fixture')
    return store.update(job['id'], status='completed', result=dict(title='Test song', artist='Singer',
                        duration_seconds=10, artifacts=artifacts))


def test_room_distinguishes_missing_lyrics_from_alignment_review(tmp_path):
    app = create_app(tmp_path, version='test')
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert not entry['lyrics_unavailable']
        for mode in ('unavailable', 'partial_word', 'line'):
            result = dict(job['result'], quality=dict(lyrics_provenance=dict(needs_review=True),
                                                     playback_fallback=dict(subtitle_mode=mode)))
            app.state.store.update(job['id'], result=result)
            song = client.get('/v1/room').json()['entries'][0]
            assert song['lyrics_need_review']
            assert song['lyrics_unavailable'] == (mode == 'unavailable')


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


def test_opening_countdown_waits_for_ready_and_keeps_original_song_clock(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    store = app.state.store
    job = ready_job(store)
    opening = dict(start=-3.369, onset=.631, dots=3, step_seconds=1., settle_seconds=1.)
    job['result']['quality'] = dict(opening_countdown=opening)
    store.update(job['id'], result=job['result'])
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        assert entry['opening_countdown'] == opening
        def command(action, **args):
            return client.post('/v1/room/commands', json=dict(action=action, entry_id=entry['id'], **args)).json()
        waiting = command('play')
        assert waiting['started_at'] is None and waiting['position'] == 0
        playing = command('ready')
        assert playing['started_at'] == pytest.approx(103.369)
        assert command('ready')['started_at'] == playing['started_at']
        now[0] = 102.
        assert app.state.room.position(app.state.room.read()) == 0
        paused = command('pause')
        assert paused['position'] == 0 and paused['started_at'] is None
        command('play')
        assert command('ready')['started_at'] == pytest.approx(105.369)
        now[0] = 106.
        assert command('pause')['position'] == pytest.approx(.631)
        command('play')
        assert command('ready')['started_at'] == 106.  # Resuming never inserts time into the song.
        command('seek', position=6.)
        assert command('pause')['position'] == 6.
        command('play')
        assert command('ready')['started_at'] == 106.
        assert command('seek', position=0.)['started_at'] is None
        assert command('ready')['started_at'] == pytest.approx(109.369)


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


def test_two_clients_share_queue_without_lost_adds_or_accidental_duplicates(tmp_path):
    import httpx

    app = create_app(tmp_path, version='test')
    jobs = [ready_job(app.state.store, url) for url in
            (URL, 'https://www.youtube.com/watch?v=abcdefghijk', 'https://www.youtube.com/watch?v=12345678901')]

    async def check():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as alice, \
                httpx.AsyncClient(transport=transport, base_url='http://test') as bob:
            first, same = await asyncio.gather(
                alice.post('/v1/room/entries', json={'job_id':jobs[0]['id'], 'if_absent':True}),
                bob.post('/v1/room/entries', json={'url':'https://youtu.be/n5sjpCQhTDk?t=10', 'if_absent':True}))
            assert first.status_code == same.status_code == 201
            initial = first.json()
            assert len(initial['entries']) == 1
            assert same.json()['entries'] == initial['entries']
            added = await asyncio.gather(*[
                client.post('/v1/room/entries', json={'job_id':job['id'], 'if_absent':True})
                for client, job in zip((alice, bob), jobs[1:])])
            assert all(response.status_code == 201 for response in added)
            for client in (alice, bob):
                snapshot = (await client.get('/v1/room')).json()
                assert len(snapshot['entries']) == 3
                assert snapshot['entries'][0]['job_id'] == jobs[0]['id']
                assert {e['job_id'] for e in snapshot['entries']} == {job['id'] for job in jobs}
            stale = await alice.post('/v1/room/commands', json={
                'action':'reorder', 'order':[e['id'] for e in initial['entries']], 'revision':initial['revision']})
            assert stale.status_code == 409
            await asyncio.gather(
                alice.post('/v1/room/commands', json={'action':'levels','levels':{'lead_vocals':.3}}),
                bob.post('/v1/room/commands', json={'action':'levels','levels':{'backing_vocals':.6}}))
            for client in (alice, bob):
                snapshot = (await client.get('/v1/room')).json()
                assert len(snapshot['entries']) == 3
                assert snapshot['levels'] == dict(instrumental=1, backing_vocals=.6, lead_vocals=.3)
            repeat = await bob.post('/v1/room/entries', json={'job_id':jobs[0]['id'], 'if_absent':False})
            assert repeat.status_code == 201
            assert len(repeat.json()['entries']) == 4
            assert repeat.json()['entries'][-1]['id'] != initial['entries'][0]['id']
    asyncio.run(check())


def test_sung_history_keeps_performances_newest_first_across_restart_and_requeue(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    first = room.add(AddSong(job_id=job['id']))['entries'][0]
    second = room.add(AddSong(job_id=job['id']))['entries'][1]
    room.command(RoomCommand(action='play'))
    room.command(RoomCommand(action='ready', entry_id=first['id']))
    now[0] = 105.
    room.command(RoomCommand(action='next', entry_id=first['id']))
    room.command(RoomCommand(action='ready', entry_id=second['id']))
    now[0] = 110.
    room.pause_on_shutdown()
    now[0] = 115.
    done = room.command(RoomCommand(action='next', entry_id=second['id']))

    assert done['entries'] == [] and done['current_id'] is None
    assert [entry['id'] for entry in done['history']] == [second['id'], first['id']]
    assert [entry['played_at'] for entry in done['history']] == [105., 100.]
    assert [entry['finished_at'] for entry in done['history']] == [115., 105.]
    assert all(entry['title'] == 'Test song' and entry['artist'] == 'Singer' and entry['url'] == URL
               for entry in done['history'])
    assert job['id'] not in store.cache.protected(store.list())
    with pytest.raises(HTTPException) as stale:
        room.command(RoomCommand(action='next', entry_id=second['id']))
    assert stale.value.status_code == 409

    restored = Room(store, app.state.worker, 'test', Updates())
    assert restored.snapshot()['history'] == done['history']
    again = restored.add(AddSong(job_id=job['id']))
    assert again['history'] == done['history']
    assert again['entries'][0]['id'] not in (first['id'], second['id'])
    assert again['entries'][0]['job_id'] == job['id'] and len(store.list()) == 1
    restored.command(RoomCommand(action='remove', entry_id=again['entries'][0]['id']))
    store.update(job['id'], status='evicted', result=None, song_info={})
    assert restored.snapshot()['history'] == done['history']


@pytest.mark.parametrize('mode', ['loading', 'unprepared', 'opening', 'paused_opening', 'seek_without_play'])
def test_unplayed_entries_do_not_enter_sung_history(tmp_path, monkeypatch, mode):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    job['result']['quality'] = dict(opening_countdown=dict(start=-4.))
    store.update(job['id'], result=job['result'], status='queued' if mode == 'unprepared' else 'completed')
    entry = room.add(AddSong(job_id=job['id']))['entries'][0]
    room.command(RoomCommand(action='play'))
    if mode in ('opening', 'paused_opening', 'unprepared'):
        room.command(RoomCommand(action='ready', entry_id=entry['id']))
    now[0] = 101.
    if mode == 'paused_opening':
        room.command(RoomCommand(action='pause', entry_id=entry['id']))
        now[0] = 110.
    if mode == 'seek_without_play':
        room.command(RoomCommand(action='seek', entry_id=entry['id'], position=5.))
    done = room.command(RoomCommand(action='next', entry_id=entry['id']))
    assert done['entries'] == [] and done['history'] == []


def test_select_archives_sung_current_and_keeps_unplayed_entries_in_queue(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr('karaoke.room.time.time', lambda: now[0])
    app = create_app(tmp_path, version='test')
    room, store = app.state.room, app.state.store
    job = ready_job(store)
    for _ in range(3):
        room.add(AddSong(job_id=job['id']))
    first, second, third = room.read()['entries']
    room.command(RoomCommand(action='play'))
    room.command(RoomCommand(action='ready', entry_id=first['id']))
    now[0] = 101.
    selected = room.command(RoomCommand(action='select', entry_id=second['id']))
    assert [e['id'] for e in selected['entries']] == [second['id'], third['id']]
    assert [e['id'] for e in selected['history']] == [first['id']]
    selected = room.command(RoomCommand(action='select', entry_id=third['id']))
    assert [e['id'] for e in selected['entries']] == [third['id'], second['id']]
    assert [e['id'] for e in selected['history']] == [first['id']]
    room.command(RoomCommand(action='ready', entry_id=third['id']))
    now[0] = 102.
    removed = room.command(RoomCommand(action='remove', entry_id=third['id']))
    assert [e['id'] for e in removed['history']] == [third['id'], first['id']]
    assert removed['current_id'] == second['id']


def test_soak_adds_all_cases_to_room_and_only_removes_finished_owned_entries(tmp_path):
    from karaoke.tools.soak import queue_cases, release_finished_entries
    app = create_app(tmp_path, worker_command=fake_worker, version='test')
    user_job = ready_job(app.state.store)
    state = {'provider_backoff': {}, 'cases': [dict(url=f'https://www.youtube.com/watch?v={i:011d}',
        title=f'Test {i}', provider='youtube', attempts=[]) for i in range(3)]}
    original_urls = [case['url'] for case in state['cases']]
    for case, status in zip(state['cases'], ('passed', 'failed', 'failed')):
        case['attempts'].append(dict(status=status, pipeline_version='previous'))
    with TestClient(app) as client:
        user = client.post('/v1/room/entries', json={'job_id': user_job['id']}).json()['entries'][0]
        state['provider_backoff']['openai'] = time.time()+60
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        assert len(client.get('/v1/room').json()['entries']) == 1
        state['provider_backoff'].clear()
        queue_cases(client, 'http://testserver', state, 'test', lambda: None)
        entries = client.get('/v1/room').json()['entries']
        assert len(entries) == 4  # The abandoned queue=2 idea must not become an app limit.
        attempts = [c['attempts'][-1] for c in state['cases']]
        assert {a['entry_id'] for a in attempts} == {e['id'] for e in entries[1:]}
        assert [e['id'] for e in entries[1:]] == [attempts[i]['entry_id'] for i in (1, 2, 0)]
        assert [case['url'] for case in state['cases']] == original_urls
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


def test_buffering_freezes_room_until_ready_and_rejects_old_playback_events(tmp_path):
    app = create_app(tmp_path, version='test')
    app.state.store.cache.minimum_free = 0
    job = ready_job(app.state.store)
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json={'job_id':job['id']}).json()['entries'][0]
        def command(action, **fields):
            return client.post('/v1/room/commands', json={'action':action, 'entry_id':entry['id'], **fields})
        command('play')
        playing = command('ready').json()
        request = dict(position=4, playback_started_at=playing['started_at'])
        waiting = command('buffering', **request).json()
        assert waiting['playing'] and waiting['started_at'] is None and waiting['position'] == 4
        assert client.get('/v1/room').json()['position'] == 4
        resumed = command('ready', revision=waiting['revision']).json()
        assert resumed['started_at'] is not None and resumed['position'] == 4
        assert command('buffering', **request).status_code == 409
        seeked = command('seek', position=7).json()
        assert command('buffering', position=4, playback_started_at=resumed['started_at']).status_code == 409
        command('pause')
        assert command('buffering', position=7, playback_started_at=seeked['started_at']).status_code == 409
        assert command('buffering', entry_id='old', **request).status_code == 409


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
    worker.processes['already-running'] = None
    jobs = {name:store.create(JobRequest(url=f'https://youtu.be/{i:011d}'),'test')[0]['id']
            for i,name in enumerate(['a','b','c','standalone'])}
    for job in [*jobs.values(), jobs['a'], 'missing-job']:
        worker.queue.put_nowait(job)
    worker.prioritize([jobs[name] for name in ['c','a','c','b']])
    assert [worker.queue.get_nowait() for _ in range(4)] == [jobs[name] for name in ['c','a','b','standalone']]
    assert worker.queue.empty()
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


@pytest.mark.parametrize('new_status', ['failed', 'source_ready', 'audio_ready', 'queued', 'running'])
@pytest.mark.parametrize('video_evicted', [False, True])
def test_library_keeps_existing_media_after_playlist_removal_and_new_attempt(tmp_path, new_status, video_evicted):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    old = cache_fixture(store, '1', restorable=True)
    entry = room.add(AddSong(job_id=old['id']))['entries'][0]
    room.command(RoomCommand(action='remove', entry_id=entry['id']))
    folder = tmp_path/old['id']
    assert (folder/'mv.mp4').exists()
    if video_evicted:
        (folder/'mv.mp4').unlink()
        store.update(old['id'], status='video_evicted', restore_video=True)
    newer, _ = store.create(JobRequest(**{**old['input'], 'force': True}), 'new-version')
    store.update(newer['id'], status=new_status)
    cards = [s for s in room.library() if s['url'] == old['input']['url']]
    assert len(cards) == 1
    assert cards[0]['job_id'] == old['id']
    assert cards[0]['status'] == ('video_evicted' if video_evicted else 'completed')
    assert room.read()['entries'] == []
    added = room.add(AddSong(job_id=cards[0]['job_id']))['entries'][0]
    assert added['job_id'] == old['id']
    assert (folder/'instrumental').read_bytes() == b'keep me'


def test_library_does_not_prefer_an_incomplete_old_result(tmp_path):
    app = create_app(tmp_path, version='test')
    store, room = app.state.store, app.state.room
    old = cache_fixture(store, '1', restorable=True)
    (tmp_path/old['id']/'instrumental').unlink()
    store.update(old['id'], status='video_evicted', restore_video=True)
    newer, _ = store.create(JobRequest(**{**old['input'], 'force': True}), 'new-version')
    card = next(s for s in room.library() if s['url'] == old['input']['url'])
    assert card['job_id'] == newer['id'] and card['status'] == 'queued'


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
                    stage_progress=counters, timings={}, song_info={'language': 'English'}))]
    with TestClient(create_app(tmp_path, worker_command=event_worker, version='test')) as client:
        entry = client.post('/v1/room/entries',json={'url':URL}).json()['entries'][0]
        end = time.monotonic()+3
        while time.monotonic() < end:
            state = client.get('/v1/jobs/'+entry['job_id']).json()
            if state.get('stage_progress') == counters:
                break
            time.sleep(.01)
        assert state['active_stages'] == ['transcribe', 'separate_lead'] and state['status'] == 'running'
        # Learning the language in a later stage must not erase earlier metadata.
        assert state['song_info'] == {**info, 'language': 'English'} and state['result'] is None
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

        queued = entry(await anext(stream))
        assert queued['title'] == '新点的歌' and queued['preparation_started_at'] is None
        started_at = '2026-10-05T22:00:00+00:00'
        store.update(job['id'], status='running', started_at=started_at)
        assert entry(await anext(stream))['preparation_started_at'] == started_at
        for done, title in enumerate(('Video title', 'Resolved song title'), 1):
            counters = {'separate_music': dict(completed=done, total=4, unit='chunks')}
            store.update(job['id'], status='running', song_info=dict(title=title, artist='Singer', language='English'),
                         active_stages=['separate_music'], stage_progress=counters)
            event = await asyncio.wait_for(anext(stream), 1)
            visible = entry(event)
            assert visible['title'] == title and visible['stage_progress'] == counters
            assert visible['preparation_started_at'] == started_at
            song = next(s for s in json.loads(event.split('data: ', 1)[1])['library'] if s['url'] == URL)
            assert song['title'] == title and song['language'] == 'English'
            assert song['stage_progress'] == counters
            assert song['preparation_started_at'] == started_at
        # Reconnecting/reloading reads the persisted start, not the last progress update.
        assert room.snapshot()['entries'][0]['preparation_started_at'] == started_at
        store.update(job['id'], status='failed', error=dict(code='test', message='Test failure'))
        visible = entry(await anext(stream))
        assert visible['title'] == 'Resolved song title' and visible['stage_progress'] == {}
        assert visible['preparation_started_at'] is None
        # A queued retry must not inherit the previous attempt's clock.
        store.update(job['id'], status='queued')
        assert entry(await anext(stream))['preparation_started_at'] is None
        retry_started_at = '2026-10-05T22:10:00+00:00'
        store.update(job['id'], status='running', started_at=retry_started_at)
        assert entry(await anext(stream))['preparation_started_at'] == retry_started_at
        store.update(job['id'], status='completed', result=dict(title='Final title', artist='Final singer'))
        completed = entry(await anext(stream))
        assert completed['title'] == 'Final title' and completed['preparation_started_at'] is None
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
        for partial in ({}, {'language': 'English'}, {'title': '', 'artist': None}, None):
            app.state.store.update(job['id'], song_info=partial)
            room = client.get('/v1/room').json()
            assert room['entries'][0]['title'] == 'Video title (Official MV)'
            assert room['library'][0]['title'] == 'Video title (Official MV)'
        app.state.store.update(job['id'], song_info={'title':'Clean song title', 'artist':'Singer'})
        room = client.get('/v1/room').json()
        assert room['entries'][0]['title'] == 'Clean song title'
        assert room['library'][0]['title'] == 'Clean song title'
        assert room['library'][0]['language'] == 'English'
        again = client.post('/v1/room/entries', json={'url':URL, 'source_title':'A different search title'}).json()
        assert again['entries'][-1]['job_id'] == job['id']
        assert again['entries'][-1]['title'] == 'Clean song title'
        assert len(app.state.store.list()) == 1


@pytest.mark.parametrize('recreate', ['force', 'version', 'evicted'])
def test_recreated_song_keeps_known_metadata_before_any_worker_runs(tmp_path, recreate):
    app = create_app(tmp_path, version='new')
    (tmp_path/'preparation.pause').touch()
    old = ready_job(app.state.store)
    if recreate == 'evicted':
        import shutil
        shutil.rmtree(tmp_path/old['id'])
        app.state.store.update(old['id'], status='evicted')
    with TestClient(app) as client:
        entry = client.post('/v1/room/entries', json=dict(
            url=URL, force=recreate == 'force', source_title='Less useful search title')).json()['entries'][0]
        assert entry['job_id'] != old['id'] and entry['status'] == 'queued'
        assert (entry['title'], entry['artist'], entry['duration']) == ('Test song', 'Singer', 10)
        from karaoke.pipeline import Pipeline
        pipeline = Pipeline(dict(url=URL), tmp_path/entry['job_id'])
        assert pipeline.song_info['title'] == 'Test song'
        other = client.post('/v1/room/entries', json=dict(url='https://www.youtube.com/watch?v=abcdefghijk')).json()['entries'][-1]
        assert other['title'] == '新点的歌'
