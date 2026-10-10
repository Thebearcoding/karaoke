"""Compare published lyric text from distinct sources before acoustic alignment."""
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import functools
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode

import requests
from rapidfuzz.distance import Levenshtein

from . import lyric_lookup as lookup
from .source_captions import read_document


def serialized_reading_variants(candidates):
    """Identify text that differs only by a source's explicit ruby annotations."""
    keys = [lookup.comparison_key(''.join(lines)) for lines, _ in candidates]
    variants = {}
    for source, (_, report) in enumerate(candidates):
        readings = {(r['after'], r['text']) for r in report.get('reading_annotations', [])}
        if not readings:
            continue
        for index, key in enumerate(keys):
            if index == source or index in variants:
                continue
            changes = [o for o in Levenshtein.opcodes(keys[source], key) if o.tag != 'equal']
            if changes and all(o.tag == 'insert' and
                               (o.src_start, key[o.dest_start:o.dest_end]) in readings for o in changes):
                variants[index] = dict(source=index, plain_source=source,
                    reading_ranges=[[o.dest_start, o.dest_end] for o in changes])
    return list(variants.values())


def compare_lyrics(candidates):
    """One vote per publisher; choose a complete source, never splice lyrics."""
    texts = [tuple(re.findall(r"[a-z\u00c0-\u024f]+(?:'[a-z\u00c0-\u024f]+)*|\d+|[^\W_]",
        lookup.canonical_script(' '.join(lines)).casefold().replace('’', "'"))) for lines, _ in candidates]
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
                                similarity=round(match.ratio(), 4), differences=differences,
                                matched_units=sum(block.size for block in match.get_matching_blocks())))
    report = dict(status='corroborated' if agreed else 'conflicting' if len(groups) > 1 else 'insufficient_sources',
                  sources=[r for _, r in candidates], comparisons=comparisons,
                  agreeing_sources=ordered[0] if agreed else [],
                  independent_sources=len(votes(ordered[0])) if agreed else 0, provider_timestamps_used=False,
                  text_units=[len(text) for text in texts],
                  comparison_unit='latin_words_and_other_characters')
    return (candidates[ordered[0][0]][0] if agreed else None), report


def resolve_recording_artists(recording, cache):
    """Read names by the artist IDs on this recording, never by name search."""
    from ytmusicapi import YTMusic
    resolved = []
    with requests.Session() as session:
        session.request = functools.partial(session.request, timeout=10)
        session.hooks['response'].append(lambda response, *a, **kw: response.raise_for_status())
        music = YTMusic(language='zh_CN', requests_session=session)
        for artist in recording.get('artists') or []:
            row = dict(artist)
            if row.get('id'):
                try:
                    page = lookup.cached_lyric_request('youtube_music',
                        dict(method='artist', browse_id=row['id'], locale='zh_CN'), cache,
                        lambda: music.get_artist(row['id']))
                    if isinstance(page.get('name'), str) and page['name'].strip():
                        # The response's channelId is the subscription channel,
                        # not necessarily the artist browse ID we requested.
                        row.update(name=page['name'].strip(), original_name=artist.get('name'),
                                   subscription_channel_id=page.get('channelId'))
                except Exception as error:
                    row['error'] = lookup.lyric_request_error(error)
            resolved.append(row)
    name = ' · '.join(row['name'] for row in resolved if row.get('name'))
    return name or recording.get('artist'), dict(provider='youtube_music', artists=resolved)


def fetch_additional_lyrics(provider, song, metadata, cache, duration=None, *, reference_lines=()):
    """Read source-bound Music lyrics or an identity-matched NetEase release."""
    if provider != 'youtube_music' and (not song.get('title') or not song.get('artist')):
        return None, dict(provider=provider, status='missing_identity')
    if provider == 'utaten':
        try:
            return lookup.fetch_reading_lyrics(song, cache, duration, reference_lines, metadata.get('title'))
        except Exception as error:
            return None, dict(lookup.lyric_request_error(error), provider=provider)
    if provider == 'youtube_music' and not re.fullmatch(r'[A-Za-z0-9_-]{11}', metadata.get('id') or ''):
        return None, dict(provider=provider, status='missing_video_identity')
    path, stage = cache.root/(provider+'-lyrics.json'), 'fetch_'+provider+'_lyrics'
    try:
        previous = json.loads(path.read_text())
        if lookup.lyric_retry_due(previous.get('report') or {}, provider, cache):
            (cache.directory/(stage+'.json')).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    with cache.stage(stage, inputs=[Path(__file__)], outputs=[path],
                     config=dict(song=song, video_id=metadata.get('id'), source_title=metadata.get('title'), duration=duration,
                                 locale='zh_CN' if provider == 'youtube_music' else None, reference_lines=reference_lines),
                     code=[Path(__file__), Path(__file__).with_name('lyric_lookup.py'),
                           Path(__file__).with_name('source_captions.py')]) as reused:
        if not reused:
            recording_identity = {}
            try:
                if provider == 'netease':
                    def lyrics(row):
                        if not str(row['trackId']).isdigit():
                            return None
                        url = 'https://music.163.com/api/song/lyric?'+urlencode(dict(id=row['trackId'], lv=-1, kv=-1, tv=-1))
                        data = lookup.cached_lyric_request(provider, url, cache, lambda: read_document(url, timeout=10))
                        return data.get('lrc', {}).get('lyric')
                    title, artist = (lookup.song_names(song, k)[-1] for k in ('title','artist'))
                    attempted, rejected = [], []
                    for query in dict.fromkeys([title+' '+artist, title]):
                        url = 'https://music.163.com/api/cloudsearch/pc?'+urlencode(dict(s=query, type=1, limit=30, offset=0))
                        packet = lookup.cached_lyric_request(provider, url, cache, lambda: read_document(url, timeout=10))
                        if packet.get('code') != 200:
                            raise ValueError('NetEase lyric search rejected the request')
                        rows = [dict(trackId=r.get('id'), trackTitle=r.get('name'), trackAliases=(r.get('alia') or []) + (r.get('tns') or []),
                            artists=[dict(artistName=a.get('name'), artistAliases=(a.get('alias') or []) + (a.get('tns') or []))
                                     for a in r.get('ar') or [] if isinstance(a, dict)],
                            album=dict(albumTitle=(r['al'].get('name') or '') if isinstance(r.get('al'), dict) else ''),
                            duration_seconds=r['dt']/1000 if isinstance(r.get('dt'), (int, float)) else None)
                            for r in (packet.get('result') or {}).get('songs') or [] if isinstance(r, dict)]
                        attempted.append(query)
                        lines, report = lookup.published_candidate(rows, song, metadata.get('title'),
                            read_lyrics=lyrics, allow_recording_mismatch=True, duration=duration, reference_lines=reference_lines)
                        rejected.extend(report.get('rejected_candidates', []))
                        if lines:
                            break
                    report.update(queries=attempted, rejected_candidates=rejected)
                    if lines:
                        report['url'] = 'https://music.163.com/#/song?id='+report['url'].rsplit('/',1)[-1]
                else:
                    from ytmusicapi import YTMusic
                    video = metadata.get('id', '')
                    lyric_video, release = video, None
                    with requests.Session() as session:
                        session.request = functools.partial(session.request, timeout=10)
                        session.hooks['response'].append(lambda response, *a, **kw: response.raise_for_status())
                        music = YTMusic(language='zh_CN', requests_session=session)
                        listing = lookup.cached_lyric_request(provider, dict(video_id=video, method='watch', locale='zh_CN'), cache,
                            lambda: music.get_watch_playlist(videoId=video, limit=1))
                        if not listing['tracks'] or listing['tracks'][0]['videoId'] != video:
                            raise ValueError('Music lyrics refer to another recording')
                        track = listing['tracks'][0]
                        recording_identity = dict(title=track.get('title') or '', video_id=video,
                            artist=' · '.join(a['name'] for a in track.get('artists', []) if a.get('name')),
                            artists=[dict(name=a.get('name'), id=a.get('id')) for a in track.get('artists') or []])
                        result = (lookup.cached_lyric_request(provider,
                            dict(video_id=video, method='lyrics', browse_id=listing['lyrics'], locale='zh_CN'), cache,
                            lambda: music.get_lyrics(listing['lyrics'])) if listing.get('lyrics') else None)
                        if not (result or {}).get('lyrics') and song.get('title') and song.get('artist'):
                            # An MV/live upload may lack the lyrics tab even
                            # though the same work's album track has it. Read
                            # that published text, retaining the release change.
                            music = YTMusic(language='en', requests_session=session)
                            query = ' '.join(lookup.song_names(song, key)[-1] for key in ('title', 'artist'))
                            rows = lookup.cached_lyric_request(provider,
                                dict(method='search', query=query, locale='en'), cache,
                                lambda: music.search(query, filter='songs', limit=20))
                            matching = []
                            for row in rows:
                                if not isinstance(row, dict):
                                    continue
                                artists = [a.get('name') for a in row.get('artists') or [] if isinstance(a, dict)]
                                title_match = any(lookup.identity(name, title=True) & lookup.identity(row.get('title'), title=True)
                                                  for name in lookup.song_names(song, 'title'))
                                artist_match = any(lookup.artist_credit_matches(name, artists, partial=True)
                                                   for name in lookup.song_names(song, 'artist'))
                                album = row.get('album') if isinstance(row.get('album'), dict) else {}
                                mismatch = lookup.recording_mismatches(song, metadata.get('title'), row.get('title'), album.get('name'))
                                if title_match and artist_match and mismatch is not None and row.get('videoId') and row['videoId'] != video:
                                    if not any(lookup.artist_credit_matches(name, artists) for name in lookup.song_names(song, 'artist')):
                                        mismatch.append('performers')
                                    matching.append((mismatch, row))
                            for mismatch, row in sorted(matching, key=lambda item: len(item[0]))[:5]:
                                candidate_video = row['videoId']
                                candidate = lookup.cached_lyric_request(provider,
                                    dict(video_id=candidate_video, method='watch', locale='en'), cache,
                                    lambda: music.get_watch_playlist(videoId=candidate_video, limit=1))
                                if (not candidate.get('tracks') or candidate['tracks'][0].get('videoId') != candidate_video
                                        or not candidate.get('lyrics')):
                                    continue
                                alternative = lookup.cached_lyric_request(provider,
                                    dict(video_id=candidate_video, method='lyrics', browse_id=candidate['lyrics'], locale='en'), cache,
                                    lambda: music.get_lyrics(candidate['lyrics']))
                                if (alternative or {}).get('lyrics'):
                                    result, lyric_video, release = alternative, candidate_video, row
                                    track = candidate['tracks'][0]
                                    release_mismatches = mismatch
                                    break
                    text = (result or {}).get('lyrics')
                    if text is not None and (not isinstance(text, str) or len(text) > 30000):
                        raise ValueError('Invalid Music lyric text')
                    # Parse a publisher's header using that same recording's
                    # names, which may differ from the UI's localized labels.
                    title = track.get('title') or song.get('title') or ''
                    artists = [a['name'] for a in track.get('artists') or [] if a.get('name')]
                    album = track.get('album')
                    lines, excluded = lookup.published_lyric_lines(text or '', title, artists or [song.get('artist')],
                        album=album.get('name') if isinstance(album, dict) else None)
                    language = song.get('language') or lookup.text_language('\n'.join(lines))
                    if lines and (len(lines) > 1000 or any(len(line) > 200 for line in lines)):
                        raise ValueError('Invalid Music lyric lines')
                    report = dict(status='candidate' if lines else 'unavailable', language=language,
                        recording_identity=recording_identity,
                        url='https://music.youtube.com/watch?v='+lyric_video, source_credit=(result or {}).get('source'),
                        text_sha256=hashlib.sha256((text or '').encode()).hexdigest(), excluded_credit_lines=excluded,
                        provider_timestamps_used=False)
                    credit = (report['source_credit'] or '').casefold()
                    report['publisher'] = next((p for p in ('lyricfind','musixmatch') if p in credit), 'youtube')
                    if release:
                        report['release_fallback'] = dict(requested_video_id=video, selected_video_id=lyric_video,
                            title=release.get('title'), artists=release.get('artists'),
                            duration_seconds=release.get('duration_seconds'))
                        report['recording_mismatches'] = release_mismatches
                result = dict(lines=lines, report=dict(report, provider=provider))
            except Exception as error:
                report = lookup.lyric_request_error(error)
                result = dict(lines=None, report=dict(report, provider=provider,
                    recording_identity=recording_identity), retry_at=report.get('retry_at'))
            path.write_text(json.dumps(result, ensure_ascii=False)+'\n')
    result = json.loads(path.read_text())
    return result['lines'], result['report']


def alternative_lyric_candidates(lyrics, packet):
    """Reuse whole, eligible published texts after the selected text fails alignment."""
    from .subtitles import normalized
    if lyrics.get('provenance', {}).get('kind') != 'online_lyric_text':
        return
    review = packet.get('crosscheck', {})
    texts, sources = packet.get('candidate_texts', []), review.get('sources', [])
    selected = review.get('selected_source')
    if not isinstance(selected, int) or not 0 <= selected < len(texts) or texts[selected] != lyrics['lines']:
        return  # A stale lookup must not replace the current input.
    seen = {normalized(''.join(lyrics['lines'])).casefold()}
    rejected = {item['source'] for item in review.get('rejected_sources', [])}
    # Recording agreement is a preference, not a ban on another release's
    # complete text after the preferred version fails acoustic alignment.
    order = dict.fromkeys([*review.get('eligible_sources', []), *range(len(texts))])
    for index in order:
        if index in rejected or not 0 <= index < min(len(texts), len(sources)):
            continue
        key = normalized(''.join(texts[index])).casefold()
        if key in seen:
            continue
        seen.add(key)
        selected_review = dict(review, selected_source=index, selection='alignment_fallback')
        if review.get('status') == 'corroborated' and index not in review.get('agreeing_sources', []):
            selected_review.update(status='conflicting', agreeing_sources=[], independent_sources=0)
        report = dict(sources[index], crosscheck=selected_review)
        language = report.get('language') or lyrics['language']
        if lyrics['language'] == 'Cantonese' and language == 'Chinese':
            language = 'Cantonese'
        yield dict(lyrics, lines=texts[index], language=language, provenance=dict(
            lyrics['provenance'], lyric_lookup=report, text_transform=report.get('text_transform', 'whitespace_only')))


def observed_source_alternative(lyrics, alternatives, observations, *, cues=None, duration=None):
    """Use existing recognition to request a better-supported published text.

    After a usable alignment, require an observed uncovered passage as well.
    Extra matching ad-libs over an existing sung line cannot invalidate it.
    """
    from .anchors_from_asr import text_units
    from .align_lyrics_whisper import recognized_words
    from .song_sources import missing_repetitions
    from rapidfuzz.distance import Levenshtein

    language = lyrics['language']
    recognized = (o for o in observations if o.get('recognition'))
    for observation in sorted(recognized, key=lambda o:o['source_range'][1]-o['source_range'][0], reverse=True):
        begin, end = observation['source_range']
        words = recognized_words(begin, observation['recognition'], end)
        heard = [u for word in words for u in text_units(word['word'], language)]
        owners = [word for word in words for _ in text_units(word['word'], language)]
        def support(candidate):
            expected = text_units(''.join(candidate['lines']), language)
            return {index for block in Levenshtein.opcodes(expected, heard) if block.tag == 'equal'
                    for index in range(block.dest_start, block.dest_end)}
        baseline = support(lyrics)
        for index, candidate in enumerate(alternatives):
            matched = support(candidate)
            if len(matched)-len(baseline) >= max(24, len(heard)*.05):
                missing = None if cues is None else missing_repetitions(
                    candidate['lines'], [dict(c, status='candidate') for c in cues], words, language, duration)
                if cues is not None and not missing:
                    # A misspelled observed word need not prevent trying a
                    # published alternative. Require substantial matching text
                    # in a real caption gap, using the coverage floor shared
                    # with lyric_coverage. This proposes text, never timings.
                    edges = [dict(end=0.), *cues, dict(start=duration)]
                    for left, right in zip(edges, edges[1:]):
                        low, high = left['end'], right['start']
                        if high <= low:
                            continue
                        inside = {i for i,w in enumerate(owners)
                                  if low-.001 <= w['start'] < w['end'] <= high+.001}
                        count, prior = len(matched & inside), len(baseline & inside)
                        if count >= max(24, len(inside)*.55) and count > prior:
                            missing.append(dict(gap=[low,high], recognized_units=len(inside),
                                current_matched_units=prior, alternative_matched_units=count,
                                basis='source_overlap_in_caption_gap'))
                if cues is not None and not missing:
                    continue
                return dict(alternative=index, source_range=[begin,end], recognized_units=len(heard),
                            current_matched_units=len(baseline), alternative_matched_units=len(matched),
                            uncovered_passages=missing)
    return None


def observed_repetition_candidate(lyrics, cues, observations, duration):
    """Propose exact source-text copies; their times still need full alignment."""
    from .align_lyrics_whisper import recognized_words
    from .anchors_from_asr import anchors_from_words
    from .song_sources import missing_repetitions
    from .subtitles import normalized

    provenance = lyrics.get('provenance', {})
    if provenance.get('kind') not in ('online_lyric_text', 'video_description') or provenance.get('repetition_passes', 0) >= 4:
        return None
    lines, language = lyrics['lines'], lyrics['language']
    proposals = []
    for observation in observations:
        if not observation.get('recognition'):
            continue
        begin, end = observation['source_range']
        words = recognized_words(begin, observation['recognition'], end)
        found = missing_repetitions(lines, [dict(c,status='candidate') for c in cues], words, language, duration)
        if not found or any(f['ambiguous'] for f in found):
            continue
        if any(r['output_range'][0] < f['insertion_index'] < r['output_range'][1]
               for f in found for r in provenance.get('repetitions', []) if 'output_range' in r):
            continue
        copies = []
        for item in found:
            first, stop = item['source_range']
            anchors = item['boundary_anchors']
            covered = None
            # A recognizer can hear the outer phrases while garbling the
            # interior. Use them to propose the complete source passage; the
            # next acoustic alignment must still verify every copied line.
            # Include neighbouring tails: an old held subtitle can overlap
            # the first words of the omitted passage.
            insertion = item['insertion_index']
            low = max(0.,cues[insertion-1]['start']-.6) if insertion else 0.
            high = min(duration,cues[insertion]['end']+.6) if insertion<len(cues) else duration
            gap_words = [w for w in words if low-.001 <= w['start'] and w['end'] <= high+.001]
            context = anchors_from_words(lines[:stop],gap_words,language,allow_missing=True)
            def strong(a):
                return a['status']=='candidate' and (a['matched_fraction']>=.8 or
                    a.get('match_basis') in ('complete_phrase','complete_clause'))
            starts = [i for i,a in enumerate(context[:first]) if strong(a) and a['end']<=anchors[-1]['end'] and not any(
                normalized(c['text']).casefold()==normalized(a['text']).casefold() and
                c['start']-2 <= a['end'] and a['start'] <= c['end']+2 for c in cues)]
            if starts and strong(context[-1]):
                first = min(starts)
                anchors = [context[first],context[-1]]
                covered = sum(a['matched_fraction']*len(a['text']) for a in context[first:])/max(1,sum(map(len,lines[first:stop])))
            copy = dict(item, source_range=[first,stop], source_lines=lines[first:stop],
                        boundary_anchors=anchors, observation_range=[begin,end],
                        seed_source_range=item['source_range'], seed_similarity=item['similarity'],
                        context_coverage=covered)
            copy.pop('similarity')
            copies.append(copy)
        copies.sort(key=lambda c:c['boundary_anchors'][0]['start'])
        if any(a['boundary_anchors'][-1]['end'] > b['boundary_anchors'][0]['start']
               for a,b in zip(copies,copies[1:])):
            continue
        expanded, positions, additions = [], {}, []
        for index in range(len(lines)+1):
            for copy in copies:
                if copy['insertion_index'] == index:
                    first = len(expanded)
                    expanded.extend(copy['source_lines'])
                    additions.append(dict(copy, output_range=[first,len(expanded)]))
            if index < len(lines):
                positions[index] = len(expanded)
                expanded.append(lines[index])
        if len(expanded)>1000:
            continue
        previous = [dict(r, output_range=[positions[r['output_range'][0]], positions[r['output_range'][1]-1]+1])
                    for r in provenance.get('repetitions', [])]
        candidate = dict(lyrics, lines=expanded, provenance=dict(provenance, text_transform='source_repetitions',
            repetition_passes=provenance.get('repetition_passes',0)+1, repetitions=[*previous,*additions]))
        proposals.append(candidate)
    return max(proposals, key=lambda c:len(normalized(''.join(c['lines']))), default=None)


def fetch_lyric_consensus(song, metadata, duration, cache, description=None, preserve_title=False, *, prefetched=None):
    """Resolve source identity before name searches; unavailable sources never vote."""
    def lrclib():
        packet = lookup.fetch_lyrics(song, cache.root/'download/source.info.json', cache)
        lines, report = lookup.lyric_candidate(packet, song, None, duration,
            source_title=metadata.get('title'), allow_recording_mismatch=True)
        return lines, dict(report, provider='lrclib')
    sources = [('line_music', lambda: lookup.fetch_published_lyrics(song, metadata.get('title'), cache, duration)), ('lrclib', lrclib),
               ('netease', lambda: fetch_additional_lyrics('netease', song, metadata, cache, duration)),
               ('youtube_music', lambda: fetch_additional_lyrics('youtube_music', song, metadata, cache, duration)),
               ('lyricsweb', lambda: lookup.fetch_structured_lyrics(song, cache, duration, metadata.get('title')))]
    prefetched = dict(prefetched or {})
    def fetch(source):
        name, request = source
        if name in prefetched:
            return prefetched[name]
        try:
            result = request()
            if result is None:
                return None, dict(provider=name, status='unavailable')
            lines, report = result
            return lines, dict(report, provider=name)
        except Exception as error:
            return None, dict(lookup.lyric_request_error(error), provider=name)
    raw_title = not preserve_title and not metadata.get('track') and song.get('title') == metadata.get('title')
    upload_artist = (not metadata.get('artist') and not metadata.get('artists') and
                     song.get('artist') == lookup.video_song_info(metadata).get('artist'))
    recording = {}
    source_title_preferred = False
    if not song.get('artist') or not song.get('title') or raw_title or upload_artist:
        source = next(source for source in sources if source[0] == 'youtube_music')
        prefetched['youtube_music'] = fetch(source)
        # Identity remains useful when this provider has no lyrics. Reuse the
        # same response, then let name-based providers search confirmed metadata.
        recording = prefetched['youtube_music'][1].get('recording_identity') or {}
        if not recording.get('video_id') or recording['video_id'] != metadata.get('id'):
            recording = {}
        source_title_preferred = bool(not preserve_title and recording.get('title') and
            len(recording['title']) < len(song.get('title') or '') and
            lookup.identity(recording['title'], title=True) &
            lookup.identity(lookup.names(song['title'], title=True)[-1], title=True))
        for key in ('title', 'artist'):
            replace = (not song.get(key) or key == 'title' and (raw_title or source_title_preferred) or key == 'artist' and
                       upload_artist and recording.get('title') != metadata.get('title'))
            if recording.get(key) and replace:
                if key == 'artist' and song.get(key) and song[key] != recording[key]:
                    song.setdefault(key+'_aliases', []).append(song[key])
                song[key] = recording[key]
    missing_identity = [key for key in ('title', 'artist') if not song.get(key) or
        key == 'artist' and upload_artist and source_title_preferred and not recording.get('artist') or
        key == 'title' and raw_title and song[key] == metadata.get('title') and not recording.get('title')]
    if callable(description) and missing_identity:
        # Only unresolved identity needs the model before name-based queries.
        description = fetch(('video_description', lambda: description(missing_identity)))
    with ThreadPoolExecutor(max_workers=len(sources)+1) as pool:
        pending = pool.submit(fetch, ('video_description', description)) if callable(description) else None
        results = list(pool.map(fetch, sources))
        if pending is not None:
            description = pending.result()
    if description:
        results.append(description)
    reference = next((lines for lines, _ in results if lines), None)
    if not description or not description[0]:
        source_text = lookup.description_lyrics(metadata, reference or ())
        if source_text:
            results = [result for result in results if result[1].get('provider') != 'video_description']
            results.append(source_text)
            reference = reference or source_text[0]
    if reference:
        # Search engines may return a native/translated title that does not
        # share characters with ours. Confirm that alias from another source's
        # actual lyric body, rather than guessing a translation or pronunciation.
        for index, (lines, report) in enumerate(results):
            if lines or not report.get('identity_candidates'):
                continue
            provider = report.get('provider')
            try:
                if provider == 'line_music':
                    retry = lookup.fetch_published_lyrics(song, metadata.get('title'), cache, duration,
                                                         reference_lines=reference)
                elif provider == 'netease':
                    retry = fetch_additional_lyrics(provider, song, metadata, cache, duration, reference_lines=reference)
                elif provider == 'lyricsweb':
                    retry = lookup.fetch_structured_lyrics(song, cache, duration, metadata.get('title'),
                                                          reference_lines=reference)
                else:
                    continue
            except Exception as error:
                results[index] = (None, dict(lookup.lyric_request_error(error), provider=provider,
                                            identity_candidates=report['identity_candidates']))
                continue
            results[index] = retry
    if (any(lookup.text_language(''.join(text)) == 'Japanese' for text, _ in results if text) or
            not reference and re.search(r'[\u3040-\u30ff\u3400-\u9fff]', song.get('title') or '')):
        results.append(fetch(('utaten', lambda: fetch_additional_lyrics('utaten', song, metadata, cache, duration,
                                                                       reference_lines=reference or ()))))
    from .lyric_readings import separate_inline_readings
    reading_evidence = [source for _, report in results for source in report.get('reading_evidence', [])]
    candidates = []
    for lines, report in results:
        if not lines:
            continue
        cleaned, annotations = separate_inline_readings(lines, reading_evidence)
        if annotations:
            report = dict(report, inline_readings=annotations,
                          text_transform=(report.get('text_transform') or '')+'; separate_publisher_confirmed_ruby')
        candidates.append((cleaned, report))
    _, review = compare_lyrics(candidates)
    lines = None
    review['unavailable_sources'] = [report for text, report in results if not text]
    rejected = {i: issue for i, (text, _) in enumerate(candidates)
                if (issue := lookup.short_lyric_text(text, duration))}
    review['rejected_sources'] = [dict(issue, source=i) for i, issue in sorted(rejected.items())]
    review['serialized_reading_variants'] = serialized_reading_variants(candidates)
    annotated = {r['source'] for r in review['serialized_reading_variants']}
    usable = set(range(len(candidates))) - rejected.keys()
    if usable:
        # A known different performance remains a weaker fallback. Duration
        # alone cannot establish that: music videos add intros and credits.
        preferred = ('utaten', 'line_music', 'youtube_music', 'netease', 'lyricsweb', 'video_description', 'lrclib')
        eligible = {i for i in usable - annotated if not set(candidates[i][1].get('recording_mismatches') or []) - {'duration'}}
        eligible = eligible or usable
        review['eligible_sources'] = sorted(eligible)
        index = min(eligible, key=lambda i: preferred.index(candidates[i][1]['provider']))
        selection = 'published_source'
        agreeing = eligible.intersection(review['agreeing_sources'])
        if agreeing:
            index, selection = min(agreeing), 'agreement'
        else:
            # A majority of close, independently published texts outweighs an outlier.
            # Similarity only selects a whole candidate; differences stay soft
            # failures and are never reported as exact agreement or accuracy.
            support = [{i} for i in range(len(candidates))]
            similarities = [{} for _ in candidates]
            for comparison in review['comparisons']:
                if comparison['similarity'] >= .9:
                    a, b = comparison['sources']
                    support[a].add(b)
                    support[b].add(a)
                    for i, j in ((a,b),(b,a)):
                        own = candidates[i][1].get('publisher') or candidates[i][1]['provider']
                        peer = candidates[j][1].get('publisher') or candidates[j][1]['provider']
                        if j in eligible and peer != own:
                            similarities[i][peer] = max(similarities[i].get(peer, 0), comparison['similarity'])
            votes = [len({candidates[i][1].get('publisher') or candidates[i][1]['provider'] for i in group & eligible})
                     for group in support]
            publishers = {candidates[i][1].get('publisher') or candidates[i][1]['provider'] for i in eligible}
            if max(votes[i] for i in eligible) >= max(2, len(publishers)//2+1):
                # Equally supported versions can still differ in completeness.
                # Prefer the closest whole text, counting each publisher once.
                agreement = [sum(s.values())/len(s) if s else 0 for s in similarities]
                index = min(eligible, key=lambda i: (-votes[i], -agreement[i], preferred.index(candidates[i][1]['provider'])))
                review['supporting_sources'] = sorted(support[index] & eligible)
                review['supporting_similarity'] = agreement[index]
                selection = 'similarity_support'
        lines, selected = candidates[index]
        if review['status'] == 'corroborated' and index not in review['agreeing_sources']:
            review.update(status='conflicting', unselected_agreeing_sources=review['agreeing_sources'],
                          agreeing_sources=[], independent_sources=0)
        review['selected_source'] = index
        review['selection'] = selection
        report = dict(selected, crosscheck=review)
    else:
        if rejected or any(r.get('rejected_candidates') for r in review['unavailable_sources']):
            review.update(status='lyrics_incomplete', agreeing_sources=[], independent_sources=0)
        report = dict(status=review['status'], crosscheck=review)
    # Keep eligible alternatives with the source-bound packet, so preparing
    # lyrics on another host does not silently remove alignment fallback.
    report = dict(report, candidate_texts=[text for text, _ in candidates])
    (cache.root/'lyric-crosscheck.json').write_text(json.dumps(report, ensure_ascii=False)+'\n')
    return lines, report
