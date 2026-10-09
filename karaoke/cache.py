"""Song cache with an optional quota and mandatory disk-space protection."""
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import time
from uuid import UUID

from .stage_cache import retained_files
from .job_schema import playback_stems

GIB = 1024**3


def can_restore_video(folder, result):
    """Only discard an MV when its expensive inputs and source identity survive."""
    try:
        artifacts = result['artifacts']
        required = {'mv', 'subtitles', *playback_stems(artifacts)}
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
        configured = os.getenv('KARAOKE_CACHE_MAX_GIB', '10').strip().lower()
        default = limit_bytes if limit_bytes is not None else (0 if configured == 'none' else int(float(configured)*GIB))
        if default < 0 or (default == 0 and (limit_bytes is not None or configured != 'none')):
            raise ValueError('KARAOKE_CACHE_MAX_GIB must be positive')
        with store.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS cache_settings (id INTEGER PRIMARY KEY, limit_bytes INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO cache_settings VALUES (1, ?)', (default,))
            # Zero is the persisted no-quota sentinel; existing databases are NOT NULL.
            self.limit = db.execute('SELECT limit_bytes FROM cache_settings WHERE id=1').fetchone()[0] or None

    @property
    def headroom(self):
        return 256*1024**2 if self.limit is None else min(256*1024**2, self.limit//20)

    def within_limit(self, size):
        return self.limit is None or size <= self.limit

    def folder(self, job_id):
        # Do not follow a replaced job directory into another part of the disk.
        if str(UUID(job_id)) != job_id:
            raise ValueError('Invalid cache job ID')
        path = self.store.root/job_id
        if path.is_symlink():
            raise ValueError('Cache job directory cannot be a symlink')
        return path

    @staticmethod
    def size(folder, seen=None):
        seen = set() if seen is None else seen
        total = 0
        items = [(folder.parent, [], [folder.name])] if folder.is_file() else os.walk(folder, followlinks=False)
        for parent, _, names in items:
            for name in names:
                try:
                    info = os.stat(os.path.join(parent, name), follow_symlinks=False)
                    key = info.st_dev, info.st_ino
                    if key not in seen:
                        total += info.st_size
                        seen.add(key)
                except FileNotFoundError:  # A running worker may have just removed a temporary file.
                    pass
        return total

    def total_size(self, folders):
        seen = set()
        return sum(self.size(folder, seen) for folder in folders)

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
            keep = required | {'request.json'}
            for parent, _, names in os.walk(folder, followlinks=False):
                for name in names:
                    is_log = name.endswith('.log')
                    if not is_log and (not name.endswith('.json') or name in ('result.json', 'audio-result.json')):
                        continue
                    path = os.path.join(parent, name)
                    try:
                        info = os.stat(path, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(info.st_mode) and (is_log or info.st_size <= 2*1024**2):
                        keep.add(os.path.relpath(path, folder))
            return keep
        except (OSError, ValueError, KeyError):
            return set()

    def compact(self, job, *, source_only=False):
        """Keep published outputs and bounded diagnostics; discard working audio."""
        folder = self.folder(job['id'])
        keep = {'request.json', 'result.json', 'error.json', 'worker.log', 'lyric-crosscheck.json',
                'captioned-picture.render.json',
                'song-metadata.json', 'checkpoints/extract_song_metadata.json',
                'lyrics.json', 'timing-constraints.json', 'timing-checked/report.json',
                'quality.json', 'alignment/raw.json', 'lyric-coverage-validation.json',
                'highlight-validation.json', 'storage-migration.json'}
        keep.update(p.relative_to(folder).as_posix() for p in (folder/'lyric-source-attempts').rglob('*.json'))
        # Published playback depends only on final artifacts. Keep the compact
        # source for explicit reprocessing, not every intermediate calculation.
        if job['status'] in ('completed', 'video_evicted'):
            keep.update({'source.json', 'download/source.mkv', 'download/source.info.json'})
        if (job['status'] in ('failed', 'cancelled', 'audio_ready', 'source_ready') or
                (job['status'] == 'queued' and job.get('download_retry')) or
                (job.get('result') or {}).get('validation_scope') == 'playback_audio_preview'):
            keep.update(retained_files(folder))
            keep.update(p.relative_to(folder).as_posix() for p in folder.rglob('*.json'))
            if (folder/'source.json').is_file():
                keep.update({'source.json', 'download/source.mkv', 'download/source.info.json', 'poster.jpg'})
        keep.update(a['path'] for name, a in (job.get('result') or {}).get('artifacts', {}).items()
                    if name != 'mv' or not job.get('restore_video'))
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
        if not source_only and job['status'] == 'completed':
            self.deduplicate(job)

    def deduplicate(self, job):
        """Share immutable media by verified bytes, never by song title."""
        folder = self.folder(job['id'])
        others = [other for other in self.store.list(-1)
                  if other['id'] != job['id'] and other['status'] in ('completed', 'video_evicted')]
        for artifact in (job.get('result') or {}).get('artifacts', {}).values():
            path = folder/artifact['path']
            if (not path.resolve().is_relative_to(folder) or path.suffix not in ('.flac', '.mp4')
                    or not path.is_file() or path.is_symlink()):
                continue
            for other in others:
                match = next((a for a in (other.get('result') or {}).get('artifacts', {}).values()
                              if a.get('sha256') and a.get('sha256') == artifact.get('sha256')
                              and a.get('bytes') == artifact.get('bytes')), None)
                if not match:
                    continue
                source = self.folder(other['id'])/match['path']
                if (not source.resolve().is_relative_to(self.folder(other['id'])) or source.is_symlink()
                        or not source.is_file()):
                    continue
                if os.path.samefile(source, path):
                    break
                if any(self.media_hash(p) != artifact['sha256'] for p in (source, path)):
                    continue
                temporary = path.with_suffix(path.suffix+'.shared-tmp')
                try:
                    temporary.unlink(missing_ok=True)
                    os.link(source, temporary)
                    temporary.replace(path)
                except OSError:
                    logging.getLogger(__name__).warning('Could not share media %s', path, exc_info=True)
                finally:
                    temporary.unlink(missing_ok=True)
                break

    @staticmethod
    def media_hash(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def reuse_download(self, folder, url):
        """A retry can reuse a verified local download even if the website is unavailable."""
        for job in reversed(self.store.list(-1)):
            if job['input']['url'] != url or job['status'] not in (
                    'failed', 'cancelled', 'source_ready', 'audio_ready', 'completed', 'video_evicted'):
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
        used = self.total_size(self.folder(j['id']) for j in jobs)
        free = shutil.disk_usage(self.store.root).free
        candidates = sorted((j for j in jobs if j['id'] not in protected and
                             j['status'] in ('completed', 'video_evicted', 'failed', 'evicted', 'audio_ready', 'source_ready')),
                            key=lambda j: (j.get('last_accessed_at', j['created_at']), j['id']))
        retained = {}
        for job in candidates:
            folder = self.folder(job['id'])
            retained[job['id']] = 0
            for name in self.source_files(job):
                try:
                    size = os.stat(os.path.join(folder, name), follow_symlinks=False).st_size
                except FileNotFoundError:
                    continue
                retained[job['id']] += min(size, 65536) if name.endswith('.log') else size
        # Exhaust cheap-to-rebuild video before discarding any computed audio/timings.
        for job in candidates:
            if self.within_limit(used + reserve) and free >= self.minimum_free + reserve:
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
            used = self.total_size(self.folder(j['id']) for j in jobs)
            sizes[job['id']] -= size
            free = shutil.disk_usage(self.store.root).free
            if job['status'] == 'completed':
                job.update(status='video_evicted', restore_video=True)
                self.store.update(job['id'], status='video_evicted', restore_video=True, evicted_at=time.time())
        for job in candidates:
            if self.within_limit(used + reserve) and free >= self.minimum_free + reserve:
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
            used = self.total_size(self.folder(j['id']) for j in jobs)
            sizes[job['id']] = remaining
            free = shutil.disk_usage(self.store.root).free
        candidate_ids = {j['id'] for j in candidates}
        retained_paths = [self.folder(j['id'])/name for j in candidates for name in self.source_files(j)]
        blocked = self.total_size([self.folder(j['id']) for j in jobs if j['id'] not in candidate_ids] + retained_paths)
        evictable = max(0, used-blocked)
        stats = dict(used_bytes=used, limit_bytes=self.limit, evictable_bytes=evictable,
                     protected_bytes=self.total_size([self.folder(i) for i in protected] + retained_paths),
                     free_bytes=free, can_prepare=self.within_limit(used-evictable+self.headroom) and
                     free+evictable >= self.minimum_free+self.headroom)
        # Disk free-space changes from unrelated apps need not generate SSE traffic.
        if any(stats[k] != self.stats.get(k) for k in stats if k != 'free_bytes'):
            self.revision += 1
            self.stats = stats
            self.store.on_change()
        else:
            self.stats = stats
        return self.within_limit(used + reserve) and free >= self.minimum_free + reserve

    def snapshot(self):
        return {**self.stats, 'revision': self.revision}

    def set_limit(self, value):
        if value is not None:
            self.enforce()
            if value < self.stats['protected_bytes']:
                raise CacheFull('上限小于正在处理、下载、待唱或离线源文件的占用')
        with self.store.connect() as db:
            db.execute('UPDATE cache_settings SET limit_bytes=? WHERE id=1', (value or 0,))
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
