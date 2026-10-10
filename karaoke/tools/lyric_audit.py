"""Resumable, lyric-only audit of frozen real inputs; never marks consensus as accuracy."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback
import unicodedata

from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

from .. import lyric_lookup
from ..stage_cache import StageCache


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def text_key(lines):
    text = OpenCC('tw2s').convert(unicodedata.normalize('NFKC', '\n'.join(lines)).casefold())
    return ''.join(c for c in text if c.isalnum())


def assess(lyrics, lookup, reference=None):
    """Keep retrieval, agreement and independent reference evidence separate."""
    review = lookup.get('crosscheck') or {}
    sources = review.get('sources') or []
    candidates = lookup.get('candidate_texts') or []
    selected = text_key((lyrics or {}).get('lines') or [])
    comparisons = []
    for source, lines in zip(sources, candidates):
        candidate = text_key(lines)
        similarity = Levenshtein.normalized_similarity(selected, candidate) if selected and candidate else 0.
        comparisons.append(dict(provider=source.get('provider'), publisher=source.get('publisher') or source.get('provider'),
            url=source.get('url'), units=len(candidate), similarity=round(similarity, 6),
            exact=bool(selected) and candidate == selected))
    exact = {c['publisher'] for c in comparisons if c['exact']}
    close = {c['publisher'] for c in comparisons if c['similarity'] >= .9}
    if not selected:
        status = 'no_lyrics'
    elif len(exact) >= 2:
        status = 'corroborated_text'
    elif len(close) >= 2:
        status = 'similar_text'
    elif len({c['publisher'] for c in comparisons}) >= 2:
        status = 'conflicting_text'
    else:
        status = 'single_source'
    independent = dict(status='not_reviewed')
    if reference:
        publisher = reference.get('publisher')
        used = {c['publisher'] for c in comparisons}
        if not publisher or publisher in used:
            independent = dict(status='not_independent', publisher=publisher)
        else:
            key = text_key(reference['lines'])
            independent = dict(status='exact' if key and selected == key else 'different', publisher=publisher,
                url=reference.get('url'), similarity=Levenshtein.normalized_similarity(selected,key),
                reference_sha256=digest(reference))
    return dict(status=status, selected_units=len(selected), selected_lines=len((lyrics or {}).get('lines') or []),
        selected_provider=lookup.get('provider'), selected_url=lookup.get('url'), comparisons=comparisons,
        source_disagreement=any(not c['exact'] for c in comparisons),
        recording_mismatches=lookup.get('recording_mismatches') or [],
        identity_recovery=lookup.get('identity_recovery'),
        rejected_sources=review.get('rejected_sources') or [],
        unavailable_sources=review.get('unavailable_sources') or [], independent_reference=independent)


@contextmanager
def paced_requests(interval, protected_root=None):
    """Pace benchmark calls only, preserving real fetch/selection and provider backoff."""
    original = lyric_lookup.cached_lyric_request
    locks, next_calls, guard = {}, {}, threading.Lock()

    def cached(provider, identity, cache, request, minimum_delay=30):
        with guard:
            lock = locks.setdefault(provider, threading.Lock())

        def paced():
            with lock:
                if protected_root:
                    path = Path(protected_root)/'lyric-provider-backoff'/f'{provider}.json'
                    if path.exists():
                        packet = json.loads(path.read_text())
                        if packet.get('requires_access') or (packet.get('retry_at') or 0) > time.time():
                            raise lyric_lookup.LyricRequestError(packet)
                time.sleep(max(0, next_calls.get(provider,0)-time.monotonic()))
                next_calls[provider] = time.monotonic()+interval
            return request()
        return original(provider,identity,cache,paced,minimum_delay)

    lyric_lookup.cached_lyric_request = cached
    try:
        yield
    finally:
        lyric_lookup.cached_lyric_request = original


def run_case(case, output, model):
    folder = output/'cases'/case['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(input_sha256=digest(case), **lyric_lookup.prepared_lyrics_identity(case['url'],case['metadata']))
    result_path = folder/'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('identity') == identity:
            return old
        raise ValueError(f"{case['case_id']}: changed input/code requires a new output directory")
    metadata = folder/'download/source.info.json'
    write(metadata,case['metadata'])
    started = time.time()
    lyrics, lookup, error = None, {}, None
    try:
        lyrics = lyric_lookup.prepare_online_lyrics(case.get('request') or dict(url=case['url']),
            case['metadata'],case['duration'],StageCache(folder),model,None)
        path = folder/'lyrics-first-lookup.json'
        if path.exists():
            lookup = json.loads(path.read_text())
        elif lyrics:
            lookup = lyrics.get('provenance',{}).get('lyric_lookup') or {}
    except Exception:
        error = traceback.format_exc()
    result = dict(case_id=case['case_id'], url=case['url'], title=case.get('title'), artist=case.get('artist'),
        stratum=case.get('stratum'), origin=case.get('origin'), identity=identity,
        started_at=started, finished_at=time.time(), seconds=round(time.time()-started,3),
        lyrics=lyrics, lookup=lookup, error=error, assessment=assess(lyrics,lookup,case.get('reference')))
    if error:
        result['assessment']['status'] = 'exception'
    write(result_path,result)
    return result


def summarize(cases, results):
    counts = Counter(r['assessment']['status'] for r in results)
    return dict(total=len(cases), completed=len(results), pending=len(cases)-len(results), counts=dict(counts),
        source_disagreement=sum(r['assessment']['source_disagreement'] for r in results),
        independent_reference=dict(Counter(r['assessment']['independent_reference']['status'] for r in results)),
        by_stratum={group:dict(Counter(r['assessment']['status'] for r in results if r.get('stratum') == group))
                    for group in sorted({c.get('stratum','unknown') for c in cases})},
        interpretation='Source agreement is evidence, not independent lyric or recording accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--interval',type=float,default=.4)
    parser.add_argument('--model',default='gpt-5-mini')
    parser.add_argument('--protected-root',type=Path)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if digest(cases) != manifest['selection_sha256'] or len({c['url'] for c in cases}) != len(cases):
        raise ValueError('Frozen selection hash or unique recording identity mismatch')
    selected = cases[:args.limit] if args.limit else cases
    args.output.mkdir(parents=True,exist_ok=True)
    results = []
    with (args.output/'run.lock').open('w') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with paced_requests(args.interval,args.protected_root), ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case,c,args.output,args.model) for c in selected]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                summary = summarize(cases,results)
                write(args.output/'report.json',dict(selection_sha256=manifest['selection_sha256'],
                    updated_at=time.time(),summary=summary,
                    results=[{k:v for k,v in r.items() if k not in ('lyrics','lookup')} for r in results]))
                print(len(results),result['case_id'],result['assessment']['status'],result['title'],flush=True)
        write(args.output/'exit.json',dict(exit_code=0,completed=len(results),finished_at=time.time()))


if __name__ == '__main__':
    main()
