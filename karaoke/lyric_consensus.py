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
import unicodedata
from urllib.parse import urlencode

from opencc import OpenCC
import requests

from . import lyric_lookup as lookup
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
                                similarity=round(match.ratio(), 4), differences=differences,
                                matched_units=sum(block.size for block in match.get_matching_blocks())))
    report = dict(status='corroborated' if agreed else 'conflicting' if len(groups) > 1 else 'insufficient_sources',
                  sources=[r for _, r in candidates], comparisons=comparisons,
                  agreeing_sources=ordered[0] if agreed else [],
                  independent_sources=len(votes(ordered[0])) if agreed else 0, provider_timestamps_used=False,
                  text_units=[len(text) for text in texts],
                  comparison_unit='latin_words_and_other_characters')
    return (candidates[ordered[0][0]][0] if agreed else None), report


def fetch_additional_lyrics(provider, song, metadata, cache, duration=None, *, reference_lines=()):
    """Read source-bound Music lyrics or an identity-matched NetEase release."""
    if provider != 'youtube_music' and (not song.get('title') or not song.get('artist')):
        return None, dict(provider=provider, status='missing_identity')
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
                    query = ' '.join(lookup.names(song[k], title=k == 'title')[-1] for k in ('title','artist'))
                    url = 'https://music.163.com/api/cloudsearch/pc?'+urlencode(dict(s=query, type=1, limit=30, offset=0))
                    packet = lookup.cached_lyric_request(provider, url, cache, lambda: read_document(url, timeout=10))
                    if packet.get('code') != 200:
                        raise ValueError('NetEase lyric search rejected the request')
                    rows = [dict(trackId=r['id'], trackTitle=r['name'],
                        artists=[dict(artistName=a['name']) for a in r['ar']],
                        album=dict(albumTitle=r['al']['name']),
                        duration_seconds=r['dt']/1000 if isinstance(r.get('dt'), (int, float)) else None)
                        for r in packet['result'].get('songs', [])]
                    def lyrics(row):
                        if not str(row['trackId']).isdigit():
                            return None
                        url = 'https://music.163.com/api/song/lyric?'+urlencode(dict(id=row['trackId'], lv=-1, kv=-1, tv=-1))
                        data = lookup.cached_lyric_request(provider, url, cache, lambda: read_document(url, timeout=10))
                        return data.get('lrc', {}).get('lyric')
                    lines, report = lookup.published_candidate(rows, song, metadata.get('title'),
                        read_lyrics=lyrics, allow_recording_mismatch=True, duration=duration, reference_lines=reference_lines)
                    if lines:
                        report['url'] = 'https://music.163.com/#/song?id='+report['url'].rsplit('/',1)[-1]
                else:
                    from ytmusicapi import YTMusic
                    video = metadata.get('id', '')
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
                            artist=' · '.join(a['name'] for a in track.get('artists', []) if a.get('name')))
                        result = (lookup.cached_lyric_request(provider,
                            dict(video_id=video, method='lyrics', browse_id=listing['lyrics'], locale='zh_CN'), cache,
                            lambda: music.get_lyrics(listing['lyrics'])) if listing.get('lyrics') else None)
                    text = (result or {}).get('lyrics')
                    if text is not None and (not isinstance(text, str) or len(text) > 30000):
                        raise ValueError('Invalid Music lyric text')
                    title = song.get('title') or track.get('title') or ''
                    artist = song.get('artist') or recording_identity['artist']
                    lines, excluded = lookup.published_lyric_lines(text or '', title, [artist])
                    language = song.get('language') or lookup.text_language('\n'.join(lines))
                    lines = lines if language else None
                    if lines and (len(lines) > 1000 or any(len(line) > 200 for line in lines)):
                        raise ValueError('Invalid Music lyric lines')
                    report = dict(status='candidate' if lines else 'unavailable', language=language,
                        recording_identity=recording_identity,
                        url='https://music.youtube.com/watch?v='+video, source_credit=(result or {}).get('source'),
                        text_sha256=hashlib.sha256((text or '').encode()).hexdigest(), excluded_credit_lines=excluded,
                        provider_timestamps_used=False)
                    credit = (report['source_credit'] or '').casefold()
                    report['publisher'] = next((p for p in ('lyricfind','musixmatch') if p in credit), 'youtube')
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
        yield dict(lyrics, lines=texts[index], language=report.get('language') or lyrics['language'], provenance=dict(
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
        def support(candidate):
            expected = text_units(''.join(candidate['lines']), language)
            return sum(block.src_end-block.src_start for block in Levenshtein.opcodes(expected, heard)
                       if block.tag == 'equal')
        baseline = support(lyrics)
        for index, candidate in enumerate(alternatives):
            matched = support(candidate)
            if matched-baseline >= max(24, len(heard)*.05):
                missing = None if cues is None else missing_repetitions(
                    candidate['lines'], [dict(c, status='candidate') for c in cues], words, language, duration)
                if cues is not None and not missing:
                    continue
                return dict(alternative=index, source_range=[begin,end], recognized_units=len(heard),
                            current_matched_units=baseline, alternative_matched_units=matched,
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
               ('youtube_music', lambda: fetch_additional_lyrics('youtube_music', song, metadata, cache, duration))]
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
            return None, dict(provider=name, status='unavailable', error=type(error).__name__)
    raw_title = not preserve_title and not metadata.get('track') and song.get('title') == metadata.get('title')
    recording = {}
    if not song.get('artist') or not song.get('title') or raw_title:
        source = next(source for source in sources if source[0] == 'youtube_music')
        prefetched['youtube_music'] = fetch(source)
        # Identity remains useful when this provider has no lyrics. Reuse the
        # same response, then let name-based providers search confirmed metadata.
        recording = prefetched['youtube_music'][1].get('recording_identity') or {}
        if not recording.get('video_id') or recording['video_id'] != metadata.get('id'):
            recording = {}
        for key in ('title', 'artist'):
            if recording.get(key) and (not song.get(key) or key == 'title' and raw_title):
                song[key] = recording[key]
    missing_identity = [key for key in ('title', 'artist') if not song.get(key) or
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
                else:
                    continue
            except Exception as error:
                results[index] = (None, dict(lookup.lyric_request_error(error), provider=provider,
                                            identity_candidates=report['identity_candidates']))
                continue
            results[index] = retry
    candidates = [(lines, report) for lines, report in results if lines]
    _, review = compare_lyrics(candidates)
    lines = None
    review['unavailable_sources'] = [report for text, report in results if not text]
    rejected = {i: issue for i, (text, _) in enumerate(candidates)
                if (issue := lookup.short_lyric_text(text, duration))}
    for comparison in review['comparisons']:
        for i, j in (comparison['sources'], comparison['sources'][::-1]):
            short, long = review['text_units'][i], review['text_units'][j]
            left, right = candidates[i][1], candidates[j][1]
            if (i not in rejected and j not in rejected and short < long * .5 and
                    comparison['matched_units'] >= short * .9 and
                    not left.get('recording_mismatches') and not right.get('recording_mismatches') and
                    (left.get('publisher') or left['provider']) != (right.get('publisher') or right['provider'])):
                rejected[i] = dict(reason='lyrics_fragment', units=short, fuller_source=j, fuller_units=long)
    review['rejected_sources'] = [dict(issue, source=i) for i, issue in sorted(rejected.items())]
    usable = set(range(len(candidates))) - rejected.keys()
    if usable:
        # A majority from another performance must not outrank text attached
        # to this recording. Compare agreement within the eligible versions.
        preferred = ('line_music', 'youtube_music', 'netease', 'video_description', 'lrclib')
        eligible = {i for i in usable if not candidates[i][1].get('recording_mismatches')}
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
