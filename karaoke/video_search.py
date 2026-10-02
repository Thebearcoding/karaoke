"""YouTube search metadata in provider order, without application ranking."""
import asyncio
from collections import OrderedDict
from html import unescape
import json
import math
import re
import sys
import time
import unicodedata
from urllib.parse import urlsplit

from opencc import OpenCC

from .job_schema import video_url


_RELEASE = re.compile(r'\bofficial\b|官方|オフィシャル', re.I)
_UNOFFICIAL = re.compile(r'\b(?:fan[ -]?made|unofficial)\b|饭制|飯製|搬运|搬運|非官方', re.I)
_PUBLISHER = re.compile(r'music|records?|recordings?|entertainment|vevo|音[乐樂]|唱片|娱[乐樂]|娛樂|演[艺藝]|[卫衛][视視]|[广廣]播|[电電][视視]', re.I)
_FAN_CHANNEL = re.compile(r'\bfan\b|lyrics?|covers?|搬运|搬運|歌[词詞]|粉[丝絲]|后援|後援', re.I)
_ARTIST = re.compile(r'歌手|唱作人|音[乐樂]人|音[乐樂]制作人|音樂製作人|[乐樂][队隊团團]|[组組]合|singer|songwriter|musician|band', re.I)
_simplified = OpenCC('t2s')


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
        def _extract_entries(self, renderer, continuation_list):
            # Read the complete native page; do not fetch an unbounded search.
            yield from super()._extract_entries(renderer, continuation_list)
            continuation_list[:] = [None]

        def _extract_video(self, renderer):
            result = super()._extract_video(renderer)
            # yt-dlp folds artist badges into the generic verified flag; retain the distinction.
            result['channel_is_official_artist'] = any(
                badge.get('metadataBadgeRenderer', {}).get('style') == 'BADGE_STYLE_TYPE_VERIFIED_ARTIST'
                for badge in (renderer.get('ownerBadges') or []))
            return result

    with YoutubeDL({'quiet': True, 'no_warnings': True, 'socket_timeout': 12, 'extractor_retries': 0}) as ydl:
        return list(ArtistSearch(ydl)._search_results(query))


def search_results(provider, entries, query=''):
    results = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        try:
            url = video_url(item.get('url') or item['webpage_url'])
            if urlsplit(url).hostname != 'www.youtube.com':
                continue
            title = str(item.get('title') or '').strip()
            if not title:
                continue
            duration = float(item.get('duration') or 0)
            if not math.isfinite(duration) or duration < 0:
                duration = 0
            thumbnail = item.get('thumbnail') or next((t.get('url') for t in
                reversed(item.get('thumbnails') or []) if t.get('url')), '')
            host = urlsplit(thumbnail).hostname or ''
            if urlsplit(thumbnail).scheme != 'https' or not (host == 'ytimg.com' or host.endswith('.ytimg.com')):
                thumbnail = None
            results.append(dict(url=url, title=title, provider=provider, duration=duration,
                                uploader=str(item.get('uploader') or item.get('channel') or ''),
                                thumbnail=thumbnail))
        except (KeyError, TypeError, ValueError):
            continue
    return results


async def fetch_results(provider, query):
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
