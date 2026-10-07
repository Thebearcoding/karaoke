"""Bounded song cache. Only managed job directories are eligible for deletion."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from uuid import UUID

from .stage_cache import retained_files

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
            if name in ('mv', 'poster'):
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

    def source_files(self, job):
        """Retain offline inputs and diagnostics; reuse still verifies the hash."""
        if os.getenv('KARAOKE_REQUIRE_CACHED_SOURCE') != '1':
            return set()
        folder = self.folder(job['id'])
        required = {'source.json', 'download/source.mkv', 'download/source.info.json'}
        try:
            source = json.loads((folder/'source.json').read_text())
            if (not isinstance(source, dict) or
                    source.get('input_url', source.get('url')) != job['input']['url'] or
                    not all((folder/p).is_file() and not (folder/p).is_symlink() for p in required)):
                return set()
            return required | {'request.json'} | {
                p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                and (p.suffix == '.log' or (p.suffix == '.json' and p.stat().st_size <= 2*1024**2
                                          and p.name not in ('result.json', 'audio-result.json')))}
        except (OSError, ValueError, KeyError):
            return set()

    def compact(self, job, *, source_only=False):
        """Keep published outputs and bounded diagnostics; discard working audio."""
        folder = self.folder(job['id'])
        keep = {'request.json', 'result.json', 'error.json', 'worker.log', 'lyric-crosscheck.json',
                'captioned-picture.render.json',
                'song-metadata.json', 'checkpoints/extract_song_metadata.json',
                'lyrics.json', 'timing-constraints.json', 'timing-checked/report.json'}
        if (job['status'] in ('failed', 'audio_ready', 'source_ready') or
                (job.get('result') or {}).get('validation_scope') == 'playback_audio_preview'):
            keep.update(retained_files(folder))
            keep.update(p.relative_to(folder).as_posix() for p in folder.rglob('*.json'))
            if (folder/'source.json').is_file():
                keep.update({'source.json', 'download/source.mkv', 'download/source.info.json', 'poster.jpg'})
        keep.update(a['path'] for name, a in (job.get('result') or {}).get('artifacts', {}).items()
                    if name != 'mv' or not job.get('restore_video'))
        if (job.get('result') or {}).get('quality', {}).get('playback_fallback'):
            keep.update(p.relative_to(folder).as_posix() for p in folder.rglob('*.json'))
        if source_only:
            keep = self.source_files(job)
            if not keep:
                raise ValueError('Cannot retain a missing offline source')
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

    def reuse_download(self, folder, url):
        """A retry can reuse a verified local download even if the website is unavailable."""
        for job in reversed(self.store.list(-1)):
            if job['input']['url'] != url or job['status'] not in (
                    'failed', 'source_ready', 'audio_ready', 'completed', 'video_evicted'):
                continue
            source = self.folder(job['id'])
            try:
                metadata = json.loads((source/'source.json').read_text())
                info = json.loads((source/'download/source.info.json').read_text())
                video = source/'download/source.mkv'
                if video.is_symlink():
                    continue
                with video.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata['sha256']:
                        continue
            except (OSError, KeyError, ValueError):
                continue
            (folder/'download').mkdir()
            os.link(video, folder/'download/source.mkv')
            metadata['input_url'] = url
            (folder/'source.json').write_text(json.dumps(metadata))
            (folder/'download/source.info.json').write_text(json.dumps(info))
            return

    def enforce(self, reserve=0):
        jobs = self.store.list(-1)
        protected = self.protected(jobs)
        sizes = {j['id']: self.size(self.folder(j['id'])) for j in jobs}
        used = sum(sizes.values())
        free = shutil.disk_usage(self.store.root).free
        candidates = sorted((j for j in jobs if j['id'] not in protected and
                             j['status'] in ('completed', 'video_evicted', 'failed', 'evicted', 'audio_ready', 'source_ready')),
                            key=lambda j: (j.get('last_accessed_at', j['created_at']), j['id']))
        retained = {}
        for job in candidates:
            folder = self.folder(job['id'])
            retained[job['id']] = sum(min((folder/name).stat().st_size, 65536) if name.endswith('.log')
                else (folder/name).stat().st_size for name in self.source_files(job) if (folder/name).is_file())
        # Exhaust cheap-to-rebuild video before discarding any computed audio/timings.
        for job in candidates:
            if used + reserve <= self.limit and free >= self.minimum_free + reserve:
                break
            folder = self.folder(job['id'])
            if job['status'] in ('failed', 'audio_ready', 'source_ready') and (folder/'download/source.mkv').is_file():
                source_info = folder/'source.json'
                try:
                    audio_only = json.loads(source_info.read_text()).get('audio_only')
                except (OSError, ValueError):
                    audio_only = False
                if audio_only or retained[job['id']]:
                    continue  # Keep pure audio and every required offline input.
                video = folder/'download/source.mkv'
            elif job['status'] == 'completed' and can_restore_video(folder, job.get('result')):
                video = folder/job['result']['artifacts']['mv']['path']
            else:
                continue
            size = video.stat().st_size if video.is_file() else 0
            video.unlink(missing_ok=True)
            used -= size
            sizes[job['id']] -= size
            free = shutil.disk_usage(self.store.root).free
            if job['status'] == 'completed':
                job.update(status='video_evicted', restore_video=True)
                self.store.update(job['id'], status='video_evicted', restore_video=True, evicted_at=time.time())
        for job in candidates:
            if used + reserve <= self.limit and free >= self.minimum_free + reserve:
                break
            size = sizes[job['id']]
            if size <= retained[job['id']]:
                continue
            folder = self.folder(job['id'])
            if retained[job['id']]:
                self.compact(job, source_only=True)
                remaining = self.size(folder)
                retained[job['id']] = remaining
                self.store.update(job['id'], status='source_ready', result=None, audio_result=None,
                                  restore_video=False, active_stages=[], evicted_at=time.time())
            else:
                shutil.rmtree(folder)
                remaining = 0
                self.store.update(job['id'], status='evicted' if job.get('result') else job['status'],
                                  restore_video=False, active_stages=[], evicted_at=time.time())
            used -= size-remaining
            sizes[job['id']] = remaining
            free = shutil.disk_usage(self.store.root).free
        evictable = sum(max(0, sizes[j['id']]-retained[j['id']]) for j in candidates)
        stats = dict(used_bytes=used, limit_bytes=self.limit, evictable_bytes=evictable,
                     protected_bytes=sum(sizes.get(i, 0) for i in protected)+sum(retained.values()),
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
            raise CacheFull('上限小于正在处理、下载、待唱或离线源文件的占用')
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
