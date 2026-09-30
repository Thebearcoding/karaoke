"""Local persistent jobs API. Run one server worker for one GPU queue."""
import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import signal
import sqlite3
import sys
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cache import CacheFull, GIB, SongCache
from .job_schema import JobRequest
from .room import AddSong, Room, RoomCommand, Updates


ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root, cache_limit_bytes=None):
        self.on_change = lambda: None
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'jobs.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, cache_key TEXT, status TEXT, created_at TEXT, doc TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_cache ON jobs(cache_key, status)')
        self.cache = SongCache(self, cache_limit_bytes)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def read(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT doc FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id, **fields):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT doc FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            doc = json.loads(row[0])
            doc.update(fields, updated_at=now())
            db.execute('UPDATE jobs SET status=?, doc=? WHERE id=?',
                       (doc['status'], json.dumps(doc, ensure_ascii=False), job_id))
        self.on_change()
        return doc

    def touch(self, job_id):
        # Listing or receiving SSE snapshots is not use of a cached song.
        with self.connect() as db:
            db.execute("UPDATE jobs SET doc=json_set(doc, '$.last_accessed_at', ?) WHERE id=?", (now(), job_id))

    def create(self, payload, version):
        key_payload = payload.model_dump(exclude={'force'})
        key = hashlib.sha256(json.dumps([version, key_payload], sort_keys=True).encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not payload.force:
                rows = db.execute("SELECT doc FROM jobs WHERE cache_key=? AND status IN ('queued','running','completed') ORDER BY created_at DESC", (key,)).fetchall()
                for row in rows:
                    doc = json.loads(row[0])
                    if doc['status'] != 'completed' or all(
                            (self.root/doc['id']/item['path']).is_file()
                            for item in doc['result']['artifacts'].values()):
                        doc['last_accessed_at'] = now()
                        db.execute('UPDATE jobs SET doc=? WHERE id=?', (json.dumps(doc, ensure_ascii=False), doc['id']))
                        return doc, True
            job_id, stamp = str(uuid4()), now()
            doc = dict(id=job_id, status='queued', active_stages=[], created_at=stamp, updated_at=stamp,
                       last_accessed_at=stamp, input=payload.model_dump(), result=None, error=None, pipeline_version=version)
            folder = self.root/job_id
            folder.mkdir()
            (folder/'request.json').write_text(json.dumps(payload.model_dump(), ensure_ascii=False)+'\n')
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)',
                       (job_id, key, 'queued', stamp, json.dumps(doc, ensure_ascii=False)))
        return doc, False

    def list(self, limit=20, status=None):
        with self.connect() as db:
            rows = (db.execute('SELECT doc FROM jobs WHERE status=? ORDER BY created_at LIMIT ?', (status, limit))
                    if status else db.execute('SELECT doc FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,)))
            return [json.loads(row[0]) for row in rows]


class Worker:
    def __init__(self, store, command=None):
        self.store, self.command = store, command
        self.queue, self.current, self.process = asyncio.Queue(), None, None

    def prioritize(self, job_ids):
        pending = []
        while not self.queue.empty():
            pending.append(self.queue.get_nowait())
            self.queue.task_done()
        rank = {job_id: index for index, job_id in reversed(list(enumerate(job_ids)))}
        for job_id in sorted(pending, key=lambda key: rank.get(key, len(rank))):
            self.queue.put_nowait(job_id)

    async def start(self):
        for job in self.store.list(100000, 'running'):
            self.store.update(job['id'], status='failed', active_stages=[],
                error=dict(code='interrupted', message='上次服务在处理时退出；可重新提交链接'))
        for job in self.store.list(100000):
            if job['status'] in ('completed', 'failed'):
                self.store.cache.compact(job)
        self.store.cache.enforce()
        for job in self.store.list(100000, 'queued'):
            self.queue.put_nowait(job['id'])
        self.task = asyncio.create_task(self.consume())

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass

    async def terminate(self):
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()

    async def consume(self):
        while True:
            job_id = await self.queue.get()
            self.current = job_id
            folder = self.store.root/job_id
            self.store.update(job_id, status='running', started_at=now())
            try:
                if not self.store.cache.enforce(self.store.cache.headroom):
                    raise CacheFull('歌曲缓存或硬盘剩余空间不足；请移除待唱歌曲、提高上限或释放磁盘后重试')
                command = self.command(folder) if self.command else [sys.executable, '-u', '-m',
                    'karaoke.pipeline', '--request', str(folder/'request.json'), '--output', str(folder)]
                with (folder/'worker.log').open('wb') as log:
                    self.process = await asyncio.create_subprocess_exec(*command, cwd=ROOT.parent,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                        env={**os.environ, 'KARAOKE_EVENTS_STDOUT': '1'}, start_new_session=True)
                    await self.supervise(log, job_id)
                result_path = folder/'result.json'
                if self.process.returncode == 0 and result_path.exists():
                    result = json.loads(result_path.read_text())
                    if not {'mv', 'instrumental', 'backing_vocals', 'lead_vocals'} <= result['artifacts'].keys():
                        raise ValueError('Pipeline omitted a required output')
                    for artifact in result['artifacts'].values():
                        path = (folder/artifact['path']).resolve()
                        if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size != artifact['bytes']:
                            raise ValueError('Invalid result artifact')
                    self.store.update(job_id, status='completed', active_stages=[], result=result,
                                      completed_at=now(), last_accessed_at=now())
                else:
                    error_path = folder/'error.json'
                    error = json.loads(error_path.read_text()) if error_path.exists() else dict(
                        code='worker_failed', message='处理进程退出；请检查该任务的 worker.log')
                    self.store.update(job_id, status='failed', active_stages=[], error=error)
            except CacheFull as error:
                await self.terminate()
                self.store.update(job_id, status='failed', active_stages=[],
                                  error=dict(code='cache_full', message=str(error)))
            except asyncio.CancelledError:
                await self.terminate()
                self.store.update(job_id, status='failed', active_stages=[],
                    error=dict(code='interrupted', message='服务关闭，任务已中止；可重新提交链接'))
                raise
            except Exception:
                logging.getLogger(__name__).exception('Job %s failed output/worker validation', job_id)
                await self.terminate()
                self.store.update(job_id, status='failed', active_stages=[],
                    error=dict(code='worker_error', message='任务执行或输出验证失败；请检查服务器日志'))
            finally:
                self.process, self.current = None, None
                self.store.cache.compact(self.store.read(job_id))
                self.store.cache.enforce()
                self.queue.task_done()

    async def supervise(self, log, job_id):
        async def output():
            async for line in self.process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(b'KARAOKE_PROGRESS '):
                    progress = json.loads(line[len(b'KARAOKE_PROGRESS '):])
                    self.store.update(job_id, **{key: progress[key]
                        for key in ('active_stages', 'timings', 'song_info') if key in progress})
            await self.process.wait()

        async def budget():
            while True:
                await asyncio.sleep(1)
                if not self.store.cache.enforce(self.store.cache.headroom):
                    raise CacheFull('预处理达到缓存或磁盘空间限制，已停止并清理临时文件；请释放空间后重试')

        tasks = [asyncio.create_task(output()), asyncio.create_task(budget())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
            if not self.store.cache.enforce():
                raise CacheFull('生成文件超过缓存容量；请提高上限或移除待唱歌曲后重试')
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class StorageLimit(BaseModel):
    limit_gib: float = Field(ge=1, le=4096, allow_inf_nan=False)


class LeasedFileResponse(FileResponse):
    def __init__(self, *args, cache, job_id, record_access=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache, self.job_id = cache, job_id
        cache.acquire(job_id, record_access=record_access)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cache.release(self.job_id)


def create_app(data_dir=None, *, worker_command=None, version=None, cache_limit_bytes=None):
    store = Store(data_dir or os.getenv('KARAOKE_DATA_DIR', str(ROOT/'service-data')), cache_limit_bytes)
    if version is None:
        from .pipeline import pipeline_fingerprint
        settings = {key: os.getenv(key) for key in ('KARAOKE_TEXT_MODEL', 'KARAOKE_DEVICE',
            'KARAOKE_DISABLE_LYRIC_CATALOG', 'KARAOKE_FONT_FILE', 'KARAOKE_SEPARATOR_PYTHON')}
        version = hashlib.sha256((pipeline_fingerprint()+json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    worker = Worker(store, worker_command)
    updates = Updates()
    store.on_change = updates.publish
    room = Room(store, worker, version, updates)

    @asynccontextmanager
    async def lifespan(app):
        with (store.root/'worker.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('此任务目录已有服务在运行。每个 GPU 队列只运行一个 API worker。') from error
            await worker.start()
            worker.prioritize([entry['job_id'] for entry in room.read()['entries']])
            try:
                yield
            finally:
                room.pause_on_shutdown()
                await worker.stop()

    app = FastAPI(title='Karaoke preparation API', version='1.0', lifespan=lifespan,
        description='提交视频链接，后台生成带字幕 MV、背景音乐、伴唱和主唱。默认 MV 保留伴唱。')
    origins = os.getenv('KARAOKE_CORS_ORIGINS', 'http://127.0.0.1:8745,http://localhost:8745')
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins.split(',') if o.strip()],
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    app.state.store, app.state.worker, app.state.room = store, worker, room

    def public_job(doc, request):
        result = json.loads(json.dumps(doc))
        result['status_url'] = str(request.url_for('get_job', job_id=doc['id']))
        if result['result']:
            for name, artifact in result['result']['artifacts'].items():
                artifact['url'] = str(request.url_for('get_artifact', job_id=doc['id'], name=name))
                artifact.pop('path', None)
        return result

    @app.get('/health')
    async def health():
        return dict(status='ok', queued=worker.queue.qsize(), running_job=worker.current,
                    pipeline_version=version, device=os.getenv('KARAOKE_DEVICE', 'auto'))

    @app.get('/v1/library')
    async def library():
        return dict(songs=room.library())

    @app.get('/v1/storage')
    async def storage():
        store.cache.enforce()
        return store.cache.snapshot()

    @app.post('/v1/storage')
    async def storage_limit(payload: StorageLimit):
        try:
            store.cache.set_limit(round(payload.limit_gib*GIB))
        except CacheFull as error:
            raise HTTPException(409, str(error)) from error
        return room.snapshot()

    @app.get('/v1/room')
    async def room_state():
        return room.snapshot()

    @app.post('/v1/room/entries', status_code=201)
    async def add_song(payload: AddSong):
        return room.add(payload)

    @app.post('/v1/room/commands')
    async def room_command(payload: RoomCommand):
        return room.command(payload)

    @app.get('/v1/room/events')
    async def room_events():
        return StreamingResponse(updates.events(room.snapshot), media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post('/v1/jobs', status_code=202)
    async def submit(payload: JobRequest, request: Request, response: Response):
        doc, cached = store.create(payload, version)
        if not cached:
            worker.queue.put_nowait(doc['id'])
            updates.publish()
        elif doc['status'] == 'completed':
            response.status_code = 200
        return dict(**public_job(doc, request), cache_hit=cached)

    @app.get('/v1/jobs')
    async def list_jobs(request: Request, limit: int = Query(20, ge=1, le=100)):
        return dict(jobs=[public_job(doc, request) for doc in store.list(limit)])

    @app.get('/v1/jobs/{job_id}', name='get_job')
    async def get_job(job_id: UUID, request: Request):
        doc = store.read(str(job_id))
        if doc is None:
            raise HTTPException(404, 'Unknown job')
        return public_job(doc, request)

    @app.get('/v1/jobs/{job_id}/artifacts/{name}', name='get_artifact')
    async def get_artifact(job_id: UUID, name: str):
        doc = store.read(str(job_id))
        if doc is None:
            raise HTTPException(404, 'Unknown job')
        if doc['status'] == 'evicted':
            raise HTTPException(410, '歌曲缓存已回收，请重新点歌')
        if doc['status'] != 'completed':
            raise HTTPException(409, 'Artifacts are available when the job completes')
        artifact = doc['result']['artifacts'].get(name)
        if artifact is None:
            raise HTTPException(404, 'Unknown artifact')
        folder = store.root/str(job_id)
        path = (folder/artifact['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise HTTPException(404, 'Artifact is unavailable')
        return LeasedFileResponse(path, cache=store.cache, job_id=str(job_id),
                            record_access=name != 'poster',
                            media_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                            filename=path.name, content_disposition_type='inline')

    app.mount('/assets', StaticFiles(directory=ROOT/'web'), name='assets')

    @app.get('/', include_in_schema=False)
    async def home():
        return FileResponse(ROOT/'web/index.html', headers={'Cache-Control': 'no-cache'})

    return app


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host=os.getenv('KARAOKE_HOST', '127.0.0.1'),
                port=int(os.getenv('KARAOKE_PORT', '8746')), timeout_graceful_shutdown=5)
