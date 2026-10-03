"""YouTube video search and YouTube Music catalogues in provider order."""
import asyncio
from collections import OrderedDict
from html import unescape
import json
import math
import re
import secrets
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


def youtube_entries(query, continuation=None):
    from yt_dlp import YoutubeDL
    from yt_dlp.extractor.youtube import YoutubeSearchIE

    class ArtistSearch(YoutubeSearchIE):
        next_page = None
        visitor_data = None

        def _extract_response(self, **kwargs):
            if continuation:
                kwargs['query'].update(continuation['query'])
                kwargs['headers'] = self.generate_api_headers(ytcfg=kwargs['ytcfg'],
                    visitor_data=continuation.get('visitor_data'), default_client=kwargs['default_client'])
            response = super()._extract_response(**kwargs)
            self.visitor_data = self._extract_visitor_data(response)
            return response

        def _extract_entries(self, renderer, continuation_list):
            yield from super()._extract_entries(renderer, continuation_list)
            self.next_page = continuation_list[0]
            continuation_list[:] = [None]  # One native page per request.

        def _extract_video(self, renderer):
            result = super()._extract_video(renderer)
            # yt-dlp folds artist badges into the generic verified flag; retain the distinction.
            result['channel_is_official_artist'] = any(
                badge.get('metadataBadgeRenderer', {}).get('style') == 'BADGE_STYLE_TYPE_VERIFIED_ARTIST'
                for badge in (renderer.get('ownerBadges') or []))
            return result

    with YoutubeDL({'quiet': True, 'no_warnings': True, 'socket_timeout': 12, 'extractor_retries': 0}) as ydl:
        extractor = ArtistSearch(ydl)
        entries = list(extractor._search_results(query))
        return dict(results=search_results('youtube', entries), continuation=(
            dict(query=extractor.next_page, visitor_data=extractor.visitor_data) if extractor.next_page else None))


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
                                thumbnail=thumbnail, views=item.get('view_count')))
        except (KeyError, TypeError, ValueError):
            continue
    return results


def music_thumbnail(item):
    for thumb in reversed(item.get('thumbnails') or []):
        url = thumb.get('url', '')
        parsed = urlsplit(url)
        if parsed.scheme == 'https' and any(parsed.hostname == host or
                (parsed.hostname or '').endswith('.'+host) for host in ('ytimg.com', 'googleusercontent.com')):
            return url
    return None


def music_results(entries, kind):
    results = []
    for rank, item in enumerate(entries, 1):
        if not isinstance(item, dict):
            continue
        title = item.get('artist') if kind == 'artists' else item.get('title')
        if not isinstance(title, str) or not title.strip():
            continue
        common = dict(title=title, provider='youtube_music', thumbnail=music_thumbnail(item), rank=rank)
        if kind == 'artists':
            browse_id = item.get('browseId') or ''
            if re.fullmatch(r'UC[\w-]{22}', browse_id):
                results.append(dict(common, kind='artist', browse_id=browse_id,
                    url='https://music.youtube.com/channel/'+browse_id))
            continue
        video_id = item.get('videoId') or ''
        if not re.fullmatch(r'[\w-]{11}', video_id):
            continue
        artists = [dict(name=a['name'], id=a.get('id') if re.fullmatch(r'UC[\w-]{22}', a.get('id') or '') else None)
                   for a in (item.get('artists') or []) if isinstance(a, dict) and isinstance(a.get('name'), str)]
        duration = item.get('duration_seconds')
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            duration = 0
        results.append(dict(common, kind='song', url='https://www.youtube.com/watch?v='+video_id,
            artists=artists, uploader=' · '.join(a['name'] for a in artists), duration=duration,
            album=(item.get('album') or {}).get('name'), views=item.get('views'),
            version={'MUSIC_VIDEO_TYPE_ATV': '专辑音频', 'MUSIC_VIDEO_TYPE_OMV': 'MV'}.get(item.get('videoType'), '歌曲'),
            available=item.get('isAvailable') is not False))
    return results


def _renderer(value, name):
    """Find the provider shelf while leaving parsing to the pinned provider library."""
    if isinstance(value, dict):
        if name in value:
            return value[name]
        value = value.values()
    elif not isinstance(value, list):
        return None
    for child in value:
        found = _renderer(child, name)
        if found is not None:
            return found
    return None


def music_entries(kind, query, continuation=None):
    from ytmusicapi import YTMusic
    from ytmusicapi.continuations import get_continuation_token, get_continuation_params
    from ytmusicapi.parsers.search import get_search_params, parse_search_results
    from ytmusicapi.parsers.playlists import parse_playlist_items

    class PagedMusic(YTMusic):
        response = {}

        def _send_request(self, *args, **kwargs):
            self.response = super()._send_request(*args, **kwargs)
            return self.response

    music = PagedMusic(language='en')
    metadata = {}
    if kind in ('songs', 'artists'):
        if continuation:
            music._send_request('search', dict(query=query, params=get_search_params(kind, None, False)),
                                continuation['params'])
            shelf = music.response['continuationContents']['musicShelfContinuation']
            entries = parse_search_results(shelf.get('contents', []), kind[:-1])
        else:
            # Zero suppresses automatic continuation, not results on this native page.
            entries = music.search(query, filter=kind, limit=0)
            shelf = _renderer(music.response, 'musicShelfRenderer') or {}
        next_page = dict(params=get_continuation_params(shelf)) if shelf.get('continuations') else None
    elif kind == 'artist' and re.fullmatch(r'UC[\w-]{22}', query):
        if continuation:
            music._send_request('browse', dict(continuation=continuation['token']))
            contents = music.response['onResponseReceivedActions'][0]['appendContinuationItemsAction']['continuationItems']
            entries = parse_playlist_items(contents)
        else:
            artist = music.get_artist(query)
            shelf = artist.get('songs') or {}
            playlist_id = shelf.get('browseId')
            metadata = dict(title=artist['name'], browse_id=query, thumbnail=music_thumbnail(artist),
                url='https://music.youtube.com/channel/'+query, monthly_listeners=artist.get('monthlyListeners'),
                order='youtube_music_top_songs', complete=bool(playlist_id))
            entries = (music.get_playlist(playlist_id, limit=0, related=False, suggestions_limit=0)['tracks']
                       if playlist_id else shelf.get('results', []))
            contents = (_renderer(music.response, 'musicPlaylistShelfRenderer') or {}).get('contents', []) if playlist_id else []
        token = get_continuation_token(contents) if contents else None
        next_page = dict(token=token) if token else None
    else:
        raise ValueError('Invalid music catalogue request')
    return dict(metadata, results=music_results(entries, 'songs' if kind == 'artist' else kind), continuation=next_page)


async def fetch_results(provider, query, continuation=None):
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'karaoke.video_search', query, provider,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await process.communicate(json.dumps(continuation).encode())
        if process.returncode:
            raise RuntimeError('YouTube search unavailable')
        return json.loads(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class SearchCursorExpired(ValueError):
    pass


class VideoSearch:
    def __init__(self):
        self.cache, self.pending, self.cursors = OrderedDict(), {}, OrderedDict()
        self.slots = asyncio.Semaphore(2)

    async def search(self, provider, query, cursor=None):
        continuation, offset = None, 0
        if cursor:
            saved = self.cursors.get(cursor)
            if not saved or saved[1:3] != (provider, query) or time.monotonic()-saved[0] >= 1800:
                raise SearchCursorExpired('Search cursor expired or does not match this query')
            _, _, _, continuation, offset = saved
            self.cursors.move_to_end(cursor)
        key = provider, query, cursor
        cached = self.cache.get(key)
        if cached and time.monotonic()-cached[0] < 120:
            self.cache.move_to_end(key)
            return cached[1]
        if key not in self.pending:
            if len(self.pending) >= 8:
                raise OverflowError('Too many pending searches')
            self.pending[key] = asyncio.create_task(self._load(key, continuation, offset))
            self.pending[key].add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        # A browser disconnect must not cancel another client's identical query.
        return await asyncio.shield(self.pending[key])

    async def _load(self, key, continuation, offset):
        try:
            async with asyncio.timeout(25):
                async with self.slots:
                    result = await fetch_results(*key[:2], continuation)
            next_page = result.pop('continuation', None)
            result['next_cursor'] = None
            for index, row in enumerate(result['results'], offset+1):
                row['rank'] = index
            if next_page and next_page != continuation:
                cursor = secrets.token_urlsafe(24)
                self.cursors[cursor] = (time.monotonic(), *key[:2], next_page, offset+len(result['results']))
                while len(self.cursors) > 256:
                    self.cursors.popitem(last=False)
                result['next_cursor'] = cursor
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
    provider = sys.argv[2] if len(sys.argv) > 2 else 'youtube'
    continuation = json.loads(sys.stdin.read() or 'null')
    print(json.dumps(youtube_entries(sys.argv[1], continuation) if provider == 'youtube' else
                     music_entries(provider, sys.argv[1], continuation), ensure_ascii=False))
