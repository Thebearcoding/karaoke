"""Compare published lyric text from distinct sources before acoustic alignment."""
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import functools
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlencode

from opencc import OpenCC
import httpx
import requests

from .job_schema import retry_after_seconds
from . import lyric_lookup as lookup
from .song_sources import transcript_lines
from .source_captions import read_document


def compare_lyrics(candidates):
    """One vote per publisher; choose a complete source, never splice lyrics."""
    traditional, simplified = OpenCC('s2t'), OpenCC('tw2s')
    texts = [tuple(re.findall(r"[a-z\u00c0-\u024f]+(?:'[a-z\u00c0-\u024f]+)*|\d+|[^\W_]",
        simplified.convert(traditional.convert(unicodedata.normalize('NFKC',
            ' '.join(lines)))).casefold().replace('’', "'"))) for lines, _ in candidates]
    groups = {}
    for i, text in enumerate(texts):
        if text:
            groups.setdefault(text, []).append(i)
    votes = lambda indexes: {candidates[i][1].get('publisher') or candidates[i][1]['provider'] for i in indexes}
    ordered = sorted(groups.values(), key=lambda indexes: len(votes(indexes)), reverse=True)
    agreed = bool(ordered and len(votes(ordered[0])) >= 2 and
                  (len(ordered) == 1 or len(votes(ordered[0])) > len(votes(ordered[1]))))
    comparisons = []
    for a, b in combinations(range(len(candidates)), 2):
        match = SequenceMatcher(None, texts[a], texts[b], autojunk=False)
        differences = [dict(kind=kind, left=[i,j], right=[k,l])
            for kind,i,j,k,l in match.get_opcodes() if kind != 'equal']
        comparisons.append(dict(sources=[a,b], identical=not differences,
                                similarity=round(match.ratio(), 4), differences=differences))
    report = dict(status='corroborated' if agreed else 'conflicting' if len(groups) > 1 else 'insufficient_sources',
                  sources=[r for _, r in candidates], comparisons=comparisons,
                  agreeing_sources=ordered[0] if agreed else [],
                  independent_sources=len(votes(ordered[0])) if agreed else 0, provider_timestamps_used=False,
                  comparison_unit='latin_words_and_other_characters')
    return (candidates[ordered[0][0]][0] if agreed else None), report


def fetch_additional_lyrics(provider, song, metadata, cache):
    """Read source-bound Music lyrics or an identity-matched NetEase release."""
    if not song.get('title') or not song.get('artist'):
        return None, dict(provider=provider, status='missing_identity')
    if provider == 'youtube_music' and not re.fullmatch(r'[A-Za-z0-9_-]{11}', metadata.get('id') or ''):
        return None, dict(provider=provider, status='missing_video_identity')
    path, stage = cache.root/(provider+'-lyrics.json'), 'fetch_'+provider+'_lyrics'
    try:
        previous = json.loads(path.read_text())
        if previous.get('retry_at') is not None and previous['retry_at'] <= time.time():
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__)], outputs=[path],
                     config=dict(song=song, video_id=metadata.get('id'), source_title=metadata.get('title')),
                     code=[Path(__file__), Path(__file__).with_name('lyric_lookup.py'),
                           Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            try:
                if provider == 'netease':
                    query = ' '.join(lookup.names(song[k])[-1] for k in ('title','artist'))
                    packet = read_document('https://music.163.com/api/cloudsearch/pc?'+urlencode(
                        dict(s=query, type=1, limit=30, offset=0)), timeout=10)
                    if packet.get('code') != 200:
                        raise ValueError('NetEase lyric search rejected the request')
                    rows = [dict(trackId=r['id'], trackTitle=r['name'],
                        artists=[dict(artistName=a['name']) for a in r['ar']],
                        album=dict(albumTitle=r['al']['name'])) for r in packet['result'].get('songs', [])]
                    def lyrics(row):
                        if not str(row['trackId']).isdigit():
                            return None
                        data = read_document('https://music.163.com/api/song/lyric?'+urlencode(
                            dict(id=row['trackId'], lv=-1, kv=-1, tv=-1)), timeout=10)
                        return data.get('lrc', {}).get('lyric')
                    lines, report = lookup.published_candidate(rows, song, metadata.get('title'),
                        read_lyrics=lyrics, allow_recording_mismatch=True)
                    if lines:
                        report['url'] = 'https://music.163.com/#/song?id='+report['url'].rsplit('/',1)[-1]
                else:
                    from ytmusicapi import YTMusic
                    video = metadata.get('id', '')
                    with requests.Session() as session:
                        session.request = functools.partial(session.request, timeout=10)
                        session.hooks['response'].append(lambda response, *a, **kw: response.raise_for_status())
                        music = YTMusic(requests_session=session)
                        listing = music.get_watch_playlist(videoId=video, limit=1)
                        if not listing['tracks'] or listing['tracks'][0]['videoId'] != video:
                            raise ValueError('Music lyrics refer to another recording')
                        result = music.get_lyrics(listing['lyrics']) if listing.get('lyrics') else None
                    text = (result or {}).get('lyrics')
                    if text is not None and (not isinstance(text, str) or len(text) > 30000):
                        raise ValueError('Invalid Music lyric text')
                    lines, excluded = lookup.published_lyric_lines(text or '', song['title'])
                    language = song.get('language') or lookup.text_language('\n'.join(lines))
                    lines = transcript_lines(dict(text='\n'.join(lines)), language) if language else None
                    if lines and (len(lines) > 1000 or any(len(line) > 200 or '[' in line or ']' in line for line in lines)):
                        raise ValueError('Invalid Music lyric lines')
                    report = dict(status='candidate' if lines else 'unavailable', language=language,
                        url='https://music.youtube.com/watch?v='+video, source_credit=(result or {}).get('source'),
                        text_sha256=hashlib.sha256((text or '').encode()).hexdigest(), excluded_credit_lines=excluded,
                        provider_timestamps_used=False)
                    credit = (report['source_credit'] or '').casefold()
                    report['publisher'] = next((p for p in ('lyricfind','musixmatch') if p in credit), 'youtube')
                result = dict(lines=lines, report=dict(report, provider=provider))
            except Exception as error:
                response = getattr(error, 'response', None)
                status = getattr(response, 'status_code', None)
                transient = isinstance(error, (httpx.TransportError, requests.ConnectionError, requests.Timeout)) or status == 429 or (status or 0) >= 500
                retry_at = time.time()+max(30, retry_after_seconds(error) or 0) if transient else None
                result = dict(lines=None, report=dict(provider=provider, status='unavailable',
                    error=type(error).__name__, http_status=status, retry_at=retry_at), retry_at=retry_at)
            path.write_text(json.dumps(result, ensure_ascii=False)+'\n')
    result = json.loads(path.read_text())
    return result['lines'], result['report']


def fetch_lyric_consensus(song, metadata, duration, cache, description=None):
    """Five possible routes, queried together; unavailable sources never vote."""
    def lrclib():
        packet = lookup.fetch_lyrics(song, cache.root/'download/source.info.json', cache)
        lines, report = lookup.lyric_candidate(packet, song, None, duration,
            source_title=metadata.get('title'), allow_recording_mismatch=True)
        return lines, dict(report, provider='lrclib')
    sources = [('line_music', lambda: lookup.fetch_published_lyrics(song, metadata.get('title'), cache)), ('lrclib', lrclib),
               ('netease', lambda: fetch_additional_lyrics('netease', song, metadata, cache)),
               ('youtube_music', lambda: fetch_additional_lyrics('youtube_music', song, metadata, cache))]
    def fetch(source):
        name, request = source
        hold = cache.root.parent/'lyric-provider-backoff'/f'{name}.json'
        previous = {}
        try:
            previous = json.loads(hold.read_text())
            if previous.get('requires_access') or previous.get('retry_at', 0) > time.time():
                return None, previous
        except (OSError, ValueError, TypeError):
            pass
        try:
            lines, report = request()
            if report.get('http_status') in (401, 429):
                status = report['http_status']
                pending = dict(report, provider=name, requires_access=status == 401,
                    retry_at=max(previous.get('retry_at') or 0, report.get('retry_at') or time.time()+30))
                hold.parent.mkdir(parents=True, exist_ok=True)
                temporary = hold.with_suffix(f'.{os.getpid()}.{time.time_ns()}.tmp')
                temporary.write_text(json.dumps(pending)+'\n')
                temporary.replace(hold)
            return lines, dict(report, provider=name)
        except Exception as error:
            return None, dict(provider=name, status='unavailable', error=type(error).__name__)
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        results = list(pool.map(fetch, sources))
    if description:
        results.append(description)
    candidates = [(lines, report) for lines, report in results if lines]
    lines, review = compare_lyrics(candidates)
    review['unavailable_sources'] = [report for text, report in results if not text]
    if candidates:
        # A majority from another performance must not outrank text attached
        # to this recording. Compare agreement within the eligible versions.
        preferred = ('line_music', 'youtube_music', 'netease', 'video_description', 'lrclib')
        eligible = {i for i, (_, r) in enumerate(candidates) if not r.get('recording_mismatches')}
        eligible = eligible or set(range(len(candidates)))
        review['eligible_sources'] = sorted(eligible)
        index = min(eligible, key=lambda i: preferred.index(candidates[i][1]['provider']))
        selection = 'published_source'
        agreeing = eligible.intersection(review['agreeing_sources'])
        if agreeing:
            index, selection = min(agreeing, key=lambda i: preferred.index(candidates[i][1]['provider'])), 'agreement'
        else:
            # Three close, independently published texts outweigh an outlier.
            # Similarity only selects a whole candidate; differences stay soft
            # failures and are never reported as exact agreement or accuracy.
            support = [{i} for i in range(len(candidates))]
            for comparison in review['comparisons']:
                if comparison['similarity'] >= .9:
                    a, b = comparison['sources']
                    support[a].add(b)
                    support[b].add(a)
            votes = [len({candidates[i][1].get('publisher') or candidates[i][1]['provider'] for i in group & eligible})
                     for group in support]
            if max(votes[i] for i in eligible) >= 3:
                index = min(eligible, key=lambda i: (-votes[i], preferred.index(candidates[i][1]['provider'])))
                review['supporting_sources'] = sorted(support[index] & eligible)
                selection = 'similarity_support'
        lines, selected = candidates[index]
        if review['status'] == 'corroborated' and index not in review['agreeing_sources']:
            review.update(status='conflicting', unselected_agreeing_sources=review['agreeing_sources'],
                          agreeing_sources=[], independent_sources=0)
        review['selected_source'] = index
        review['selection'] = selection
        report = dict(selected, crosscheck=review)
    else:
        report = dict(status=review['status'], crosscheck=review)
    (cache.root/'lyric-crosscheck.json').write_text(json.dumps(dict(report,
        candidate_texts=[text for text, _ in candidates]), ensure_ascii=False)+'\n')
    return lines, report
