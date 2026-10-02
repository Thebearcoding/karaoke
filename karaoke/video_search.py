"""Bounded, metadata-only video search. Searching never creates a preparation job."""
import asyncio
from collections import OrderedDict
from html import unescape
import json
from itertools import islice
import math
import re
import sys
import time
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from opencc import OpenCC

from .job_schema import video_url


_RELEASE = re.compile(r'\bofficial\b|官方|オフィシャル', re.I)
_UNOFFICIAL = re.compile(r'\b(?:fan[ -]?made|unofficial)\b|饭制|飯製|搬运|搬運|非官方', re.I)
_PUBLISHER = re.compile(r'music|records?|recordings?|entertainment|vevo|音[乐樂]|唱片|娱[乐樂]|娛樂|演[艺藝]|[卫衛][视視]|[广廣]播|[电電][视視]', re.I)
_FAN_CHANNEL = re.compile(r'\bfan\b|lyrics?|covers?|搬运|搬運|歌[词詞]|粉[丝絲]|后援|後援', re.I)
_ARTIST = re.compile(r'歌手|唱作人|音[乐樂]人|音[乐樂]制作人|音樂製作人|[乐樂][队隊团團]|[组組]合|singer|songwriter|musician|band', re.I)
_bilibili_identities = OrderedDict()
_bilibili_retry_at = 0
_simplified = OpenCC('t2s')


class SourceVerificationUnavailable(RuntimeError):
    pass


def matches_query(query, item):
    def normalize(value):
        return _simplified.convert(unicodedata.normalize('NFKC', value)).casefold()
    title = unescape(re.sub(r'<[^>]*>', '', str(item.get('title') or '')))
    text = normalize(f"{title} {item.get('channel') or item.get('author') or item.get('uploader') or ''}")
    text = ''.join(char for char in text if char.isalnum())
    terms = re.findall(r'[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+', normalize(query))
    for term in terms:
        if term in ('official', 'mv', '官方') or term in text:
            continue
        # Also accept an unspaced Chinese artist + title in either display order.
        if re.fullmatch(r'[\u3400-\u9fff]{4,}', term) and any(
                term[:i] in text and term[i:] in text for i in range(2, len(term)-1)):
            continue
        return False
    return True


def official_source(provider, item):
    """Conservative provenance annotation; a release title alone is never evidence."""
    title = unescape(re.sub(r'<[^>]*>', '', str(item.get('title') or '')))
    if _UNOFFICIAL.search(title):
        return None
    if provider == 'youtube':
        if item.get('channel_is_official_artist') is True:
            return 'youtube_official_artist_channel'
        channel = item.get('channel') or item.get('uploader') or ''
        if (item.get('channel_is_verified') is True and _RELEASE.search(title)
                and _PUBLISHER.search(channel) and not _FAN_CHANNEL.search(channel)):
            return 'youtube_verified_music_channel_and_release_title'
    else:
        identity = item.get('official_identity') or {}
        description = identity.get('desc') or ''
        if identity.get('type') == 0 and _ARTIST.search(description):
            return 'bilibili_certified_artist'
        if identity.get('type') == 1 and _PUBLISHER.search(description) and not _FAN_CHANNEL.search(description):
            return 'bilibili_certified_music_publisher'
    return None


def youtube_entries(query):
    from yt_dlp import YoutubeDL
    from yt_dlp.extractor.youtube import YoutubeSearchIE

    class ArtistSearch(YoutubeSearchIE):
        def _extract_video(self, renderer):
            result = super()._extract_video(renderer)
            # yt-dlp folds artist badges into the generic verified flag; retain the distinction.
            result['channel_is_official_artist'] = any(
                badge.get('metadataBadgeRenderer', {}).get('style') == 'BADGE_STYLE_TYPE_VERIFIED_ARTIST'
                for badge in (renderer.get('ownerBadges') or []))
            return result

    with YoutubeDL({'quiet': True, 'no_warnings': True, 'socket_timeout': 12, 'extractor_retries': 0}) as ydl:
        return list(islice(ArtistSearch(ydl)._search_results(query), 20))


async def bilibili_identity(client, mid, slots):
    global _bilibili_retry_at
    mid = str(mid)
    if not mid.isdecimal():
        return {}
    cached = _bilibili_identities.get(mid)
    if cached and time.monotonic()-cached[0] < 3600:
        _bilibili_identities.move_to_end(mid)
        return cached[1]
    try:
        async with slots:
            if time.monotonic() < _bilibili_retry_at:
                return None
            response = await client.get('https://api.bilibili.com/x/web-interface/card', params={'mid': mid})
            if response.status_code == 429:
                _bilibili_retry_at = time.monotonic()+60
            response.raise_for_status()
            data = response.json()
            if data.get('code') == -352:
                _bilibili_retry_at = time.monotonic()+60
        if data.get('code') != 0:
            return None
        identity = data['data']['card']['official_verify']
        _bilibili_identities[mid] = time.monotonic(), identity
        _bilibili_identities.move_to_end(mid)
        while len(_bilibili_identities) > 256:
            _bilibili_identities.popitem(last=False)
        return identity
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        # Unknown provenance never falls back to a title or an uploader's self-description.
        return None


def search_results(provider, entries, query=''):
    results, seen = [], set()
    for item in entries:
        if not isinstance(item, dict) or item.get('live_status') in ('is_live', 'is_upcoming'):
            continue
        evidence = official_source(provider, item)
        # YouTube omits badges from some search results, including official releases.
        # Keep its relevance order; unknown provenance must not promote a cover over them.
        if (provider == 'bilibili' and not evidence) or not matches_query(query, item):
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
                                thumbnail=thumbnail, official_evidence=evidence))
            seen.add(url)
        except (KeyError, TypeError, ValueError):
            continue
    return results[:3]


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
            entries = (data['data'].get('result') or [])[:20]
            mids = list(dict.fromkeys(str(item.get('mid')) for item in entries))
            slots = asyncio.Semaphore(2)
            identities = dict(zip(mids, await asyncio.gather(*(bilibili_identity(client, mid, slots) for mid in mids))))
            for item in entries:
                item['official_identity'] = identities.get(str(item.get('mid')), {})
            results = search_results(provider, entries, query)
            if not results and any(identity is None for identity in identities.values()):
                raise SourceVerificationUnavailable()
            return results
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'karaoke.video_search', query,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await process.communicate()
        if process.returncode:
            raise RuntimeError('YouTube search unavailable')
        return search_results(provider, json.loads(output), query)
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


if __name__ == '__main__':
    print(json.dumps(youtube_entries(sys.argv[1]), ensure_ascii=False))
