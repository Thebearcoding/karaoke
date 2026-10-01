"""Bounded, metadata-only video search. Searching never creates a preparation job."""
import asyncio
from collections import OrderedDict
from html import unescape
import json
import math
import re
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from .job_schema import video_url


def search_results(provider, entries):
    results, seen = [], set()
    for item in entries:
        if not isinstance(item, dict) or item.get('live_status') in ('is_live', 'is_upcoming'):
            continue
        try:
            raw_url = (f"https://www.bilibili.com/video/{item['bvid']}" if provider == 'bilibili'
                       else item.get('url') or item['webpage_url'])
            url = video_url(raw_url)
            expected_host = 'www.bilibili.com' if provider == 'bilibili' else 'www.youtube.com'
            if urlsplit(url).hostname != expected_host or url in seen:
                continue
            title = str(item.get('title') or '').strip()
            if provider == 'bilibili':
                title = unescape(re.sub(r'<[^>]*>', '', title))
            if not title:
                continue
            duration = item.get('duration') or 0
            if isinstance(duration, str) and ':' in duration:
                duration = sum(float(n)*60**i for i, n in enumerate(reversed(duration.split(':'))))
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0:
                duration = 0
            thumbnail = item.get('pic') or item.get('thumbnail')
            if not thumbnail:
                thumbnail = next((t.get('url') for t in reversed(item.get('thumbnails') or []) if t.get('url')), '')
            thumbnail = 'https:' + thumbnail if thumbnail.startswith('//') else thumbnail
            host = urlsplit(thumbnail).hostname or ''
            if urlsplit(thumbnail).scheme != 'https' or not any(host == h or host.endswith('.'+h) for h in ('ytimg.com', 'hdslb.com')):
                thumbnail = None
            results.append(dict(url=url, title=title[:500], provider=provider, duration=duration,
                                uploader=str(item.get('author') or item.get('uploader') or item.get('channel') or '')[:200],
                                thumbnail=thumbnail))
            seen.add(url)
        except (KeyError, TypeError, ValueError):
            continue
    return results[:8]


async def fetch_results(provider, query):
    if provider == 'bilibili':
        async with httpx.AsyncClient(timeout=12, headers={
                'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com/'},
                cookies={'buvid3': f'{uuid4()}infoc'}) as client:
            response = await client.get('https://api.bilibili.com/x/web-interface/search/type',
                params={'search_type': 'video', 'keyword': query, 'page': 1})
            response.raise_for_status()
            data = response.json()
            if data.get('code') != 0 or not isinstance(data.get('data'), dict):
                raise RuntimeError('Bilibili search unavailable')
            return search_results(provider, data['data'].get('result') or [])
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'yt_dlp',
        '--ignore-config', '--flat-playlist', '--dump-single-json', '--skip-download',
        '--socket-timeout', '12', '--retries', '0', '--extractor-retries', '0',
        '--', f'ytsearch8:{query}', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await process.communicate()
        if process.returncode:
            raise RuntimeError('YouTube search unavailable')
        return search_results(provider, json.loads(output)['entries'])
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class VideoSearch:
    def __init__(self):
        self.cache, self.pending = OrderedDict(), {}
        self.slots = asyncio.Semaphore(2)

    async def search(self, provider, query):
        key = provider, query
        cached = self.cache.get(key)
        if cached and time.monotonic()-cached[0] < 120:
            self.cache.move_to_end(key)
            return cached[1]
        if key not in self.pending:
            if len(self.pending) >= 8:
                raise OverflowError('Too many pending searches')
            self.pending[key] = asyncio.create_task(self._load(key))
            self.pending[key].add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        # A browser disconnect must not cancel another client's identical query.
        return await asyncio.shield(self.pending[key])

    async def _load(self, key):
        try:
            async with asyncio.timeout(25):
                async with self.slots:
                    result = await fetch_results(*key)
            self.cache[key] = time.monotonic(), result
            self.cache.move_to_end(key)
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
            return result
        finally:
            self.pending.pop(key, None)

    async def close(self):
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
