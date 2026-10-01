"""Bounded song cache. Only managed job directories are eligible for deletion."""
from collections import Counter
import os
from pathlib import Path
import shutil
import time
from uuid import UUID

GIB = 1024**3


def can_restore_video(folder, result):
    """Only discard an MV when its expensive inputs and source identity survive."""
    try:
        artifacts = result['artifacts']
        required = {'mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'subtitles',
                    'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'}
        if not required <= artifacts.keys() or not result['reconstruction']['source_pcm_sha256']:
            return False
        if not result['language'] or not 0 < result['shared_preview_gain'] <= 1 or result['duration_seconds'] <= 0:
            return False
        video = (folder/artifacts['mv']['path']).resolve()
        if not video.is_relative_to(folder):
            return False
        for name, artifact in artifacts.items():
            if name == 'mv':
                continue
            path = (folder/artifact['path']).resolve()
            if path == video or not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size != artifact['bytes']:
                return False
        return True
    except (KeyError, TypeError, OSError):
        return False


class CacheFull(RuntimeError):
    pass


class SongCache:
    def __init__(self, store, limit_bytes=None):
        self.store = store
        self.leases = Counter()
        self.minimum_free = GIB
        self.stats = {}
        self.revision = 0
        default = limit_bytes if limit_bytes is not None else int(float(os.getenv('KARAOKE_CACHE_MAX_GIB', '10'))*GIB)
        if default <= 0:
            raise ValueError('KARAOKE_CACHE_MAX_GIB must be positive')
        with store.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS cache_settings (id INTEGER PRIMARY KEY, limit_bytes INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO cache_settings VALUES (1, ?)', (default,))
            self.limit = db.execute('SELECT limit_bytes FROM cache_settings WHERE id=1').fetchone()[0]

    @property
    def headroom(self):
        return min(256*1024**2, self.limit//20)

    def folder(self, job_id):
        # Do not follow a replaced job directory into another part of the disk.
        if str(UUID(job_id)) != job_id:
            raise ValueError('Invalid cache job ID')
        path = self.store.root/job_id
        if path.is_symlink():
            raise ValueError('Cache job directory cannot be a symlink')
        return path

    @staticmethod
    def size(folder):
        total = 0
        for parent, _, names in os.walk(folder, followlinks=False):
            for name in names:
                try:
                    total += (Path(parent)/name).lstat().st_size
                except FileNotFoundError:  # A running worker may have just removed a temporary file.
                    pass
        return total

    def protected(self, jobs):
        ids = {j['id'] for j in jobs if j['status'] in ('queued', 'running')}
        ids.update(self.leases)
        with self.store.connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='room'").fetchone():
                import json
                state = json.loads(db.execute('SELECT state FROM room WHERE id=1').fetchone()[0])
                ids.update(e['job_id'] for e in state['entries'])
        return ids

    def compact(self, job):
        """Keep published outputs and bounded diagnostics; discard working audio."""
        folder = self.folder(job['id'])
        keep = {'request.json', 'result.json', 'error.json', 'worker.log'}
        keep.update(a['path'] for name, a in (job.get('result') or {}).get('artifacts', {}).items()
                    if name != 'mv' or not job.get('restore_video'))
        if not folder.exists():
            return
        for path in sorted(folder.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink():
                path.unlink()
            elif path.is_file():
                if relative not in keep and path.suffix != '.log':
                    path.unlink()
                elif path.suffix == '.log' and path.stat().st_size > 65536:
                    with path.open('rb') as stream:
                        stream.seek(-65536, 2)
                        tail = stream.read()
                    path.write_bytes(tail)
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    def enforce(self, reserve=0):
        jobs = self.store.list(-1)
        protected = self.protected(jobs)
        sizes = {j['id']: self.size(self.folder(j['id'])) for j in jobs}
        used = sum(sizes.values())
        free = shutil.disk_usage(self.store.root).free
        candidates = sorted((j for j in jobs if j['id'] not in protected and
                             j['status'] in ('completed', 'video_evicted', 'failed', 'evicted')),
                            key=lambda j: (j.get('last_accessed_at', j['created_at']), j['id']))
        # Exhaust cheap-to-rebuild video before discarding any computed audio/timings.
        for job in candidates:
            if used + reserve <= self.limit and free >= self.minimum_free + reserve:
                break
            folder = self.folder(job['id'])
            if job['status'] != 'completed' or not can_restore_video(folder, job.get('result')):
                continue
            video = folder/job['result']['artifacts']['mv']['path']
            size = video.stat().st_size if video.is_file() else 0
            video.unlink(missing_ok=True)
            used -= size
            sizes[job['id']] -= size
            free = shutil.disk_usage(self.store.root).free
            job.update(status='video_evicted', restore_video=True)
            self.store.update(job['id'], status='video_evicted', restore_video=True, evicted_at=time.time())
        for job in candidates:
            if used + reserve <= self.limit and free >= self.minimum_free + reserve:
                break
            size = sizes[job['id']]
            if not size:
                continue
            folder = self.folder(job['id'])
            shutil.rmtree(folder)
            used -= size
            sizes[job['id']] = 0
            free = shutil.disk_usage(self.store.root).free
            self.store.update(job['id'], status='evicted' if job.get('result') else job['status'],
                              restore_video=False, active_stages=[], evicted_at=time.time())
        evictable = sum(sizes[j['id']] for j in candidates)
        stats = dict(used_bytes=used, limit_bytes=self.limit, evictable_bytes=evictable,
                     protected_bytes=sum(sizes.get(i, 0) for i in protected),
                     free_bytes=free, can_prepare=used-evictable+self.headroom <= self.limit and
                     free+evictable >= self.minimum_free+self.headroom)
        # Disk free-space changes from unrelated apps need not generate SSE traffic.
        if any(stats[k] != self.stats.get(k) for k in stats if k != 'free_bytes'):
            self.revision += 1
            self.stats = stats
            self.store.on_change()
        else:
            self.stats = stats
        return used + reserve <= self.limit and free >= self.minimum_free + reserve

    def snapshot(self):
        return {**self.stats, 'revision': self.revision}

    def set_limit(self, value):
        self.enforce()
        if value < self.stats['protected_bytes']:
            raise CacheFull('上限小于正在处理、下载或待唱的歌曲占用；请先移除一些待唱歌曲')
        with self.store.connect() as db:
            db.execute('UPDATE cache_settings SET limit_bytes=? WHERE id=1', (value,))
        self.limit = value
        self.enforce()

    def acquire(self, job_id, *, record_access=True):
        self.leases[job_id] += 1
        if record_access:
            self.store.touch(job_id)

    def release(self, job_id):
        self.leases[job_id] -= 1
        if self.leases[job_id] <= 0:
            del self.leases[job_id]
