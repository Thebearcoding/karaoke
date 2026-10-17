"""Bounded, source-bound lyric checks for discovery; never a preparation verdict."""
import asyncio
import json
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


def lookup(video_id, folder):
    # Use the preparation parser, including its credit/heading removal. An MV
    # without its own lyrics stays unconfirmed; do not borrow another recording.
    from .lyric_consensus import fetch_additional_lyrics
    from .lyric_lookup import short_lyric_text
    from .stage_cache import StageCache

    lines, report = fetch_additional_lyrics('youtube_music', {}, {'id': video_id}, StageCache(Path(folder)))
    duration = report.get('recording_identity', {}).get('duration_seconds')
    if lines and duration and not short_lyric_text(lines, duration):
        return dict(status='available', provider='youtube_music', source=report['url'],
                    text_sha256=report['text_sha256'])
    return dict(status='unknown', reason='lookup_failed' if report.get('error') else 'not_found',
                http_status=report.get('http_status'), retry_at=report.get('retry_at'))


async def fetch_availability(video_id):
    # The parent owns cleanup, including when a timeout kills the child.
    with tempfile.TemporaryDirectory(prefix='karaoke-lyrics-') as folder:
        process = None
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, '-m', 'karaoke.lyric_availability', video_id, folder,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL))
        try:
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                raise
            output, _ = await process.communicate()
            if process.returncode:
                raise RuntimeError('Lyric lookup failed')
            return json.loads(output)
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()


class LyricAvailability:
    def __init__(self):
        self.cache, self.pending = OrderedDict(), {}
        self.slots = asyncio.Semaphore(2)
        self.retry_at = 0

    async def check(self, video_id):
        cached = self.cache.get(video_id)
        if cached and cached['expires_at'] > time.time():
            self.cache.move_to_end(video_id)
            return cached
        if video_id not in self.pending:
            if self.retry_at > time.time() or len(self.pending) >= 8:
                return dict(status='unknown', reason='busy', expires_at=max(self.retry_at, time.time()+30))
            self.pending[video_id] = asyncio.create_task(self._load(video_id))
            self.pending[video_id].add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return await asyncio.shield(self.pending[video_id])

    async def _load(self, video_id):
        result: dict[str, Any]
        try:
            try:
                async with asyncio.timeout(25):
                    async with self.slots:
                        if self.retry_at > time.time():
                            result = dict(status='unknown', reason='busy')
                        else:
                            result = await fetch_availability(video_id)
                if result.get('http_status') in (401, 403, 429):
                    self.retry_at = max(self.retry_at, result.get('retry_at') or time.time()+60)
            except Exception:
                result = dict(status='unknown', reason='lookup_failed')
            ttl = 3600 if result['status'] == 'available' else 300 if result.get('reason') == 'not_found' else 60
            result.update(checked_at=time.time(), expires_at=max(time.time()+ttl, self.retry_at))
            self.cache[video_id] = result
            self.cache.move_to_end(video_id)
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
            return result
        finally:
            self.pending.pop(video_id, None)

    async def close(self):
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    print(json.dumps(lookup(sys.argv[1], sys.argv[2]), ensure_ascii=False))
