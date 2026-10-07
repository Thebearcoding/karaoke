"""One pass over a frozen corpus using the normal playback policy, without video."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import subprocess
import threading
import time

import httpx

from karaoke.api import Store
from karaoke.stage_cache import digest
from karaoke.tools.soak import ProbeSession, record_failure, run_probe, save, stamp, verify_outputs


def soft_failures(result, prepared):
    quality = result.get('quality', {})
    events = []
    fallback = quality.get('playback_fallback')
    if fallback and fallback['subtitle_mode'] != 'unavailable':
        events.append(dict(module='alignment',
            **fallback['error'], subtitle_mode=fallback['subtitle_mode'],
            estimated_spans=len(fallback.get('estimated_spans', [])),
            omitted_lines=fallback.get('omitted_lines', [])))
    if quality.get('fallback_units') or quality.get('line_order_fallback_lines') or quality.get('recovered_alignment_lines'):
        events.append(dict(module='alignment', code='alignment_recovered',
            fallback_units=quality.get('fallback_units', 0),
            line_order_fallback_lines=quality.get('line_order_fallback_lines', []),
            recovered_alignment_lines=quality.get('recovered_alignment_lines', [])))
    if quality.get('asr_localization_windows'):
        events.append(dict(module='alignment', code='alignment_localized_with_asr',
                           windows=quality['asr_localization_windows']))
    if spans := quality.get('timing_review', {}).get('estimated_spans'):
        events.append(dict(module='alignment', code='timing_group_fallback', spans=spans))
    if fallback := quality.get('lyric_source_fallback'):
        events.append(dict(module='lyrics', code='lyric_source_fallback',
            rejected_sources=len(fallback['rejected']), selected_provider=fallback['selected']['provider']))
    lookup = prepared.get('lookup') or {}
    for key in ('identity_recovery', 'metadata_language_corrected'):
        recovery = lookup.get(key)
        if recovery and (key != 'identity_recovery' or recovery.get('recovered')):
            events.append(dict(module='metadata', code='lyric_'+key,
                original=recovery['original'], resolved=recovery['resolved']))
    if lookup.get('recording_mismatches'):
        events.append(dict(module='lyrics', code='lyric_recording_mismatch',
            provider=lookup.get('provider'), reasons=lookup['recording_mismatches']))
    metadata_error = (prepared.get('lyrics') or {}).get('provenance', {}).get('metadata_error') or lookup.get('metadata_error')
    if metadata_error:
        events.append(dict(module='metadata', code='metadata_lookup_failed', error_type=metadata_error))
    crosscheck = lookup.get('crosscheck')
    if crosscheck and crosscheck.get('status') == 'conflicting':
        events.append(dict(module='lyrics', code='lyric_source_disagreement',
            provider=lookup.get('provider'), sources=[s['provider'] for s in crosscheck.get('sources', [])]))
    sources = (crosscheck.get('unavailable_sources', []) if crosscheck else
               (lookup, lookup.get('published_lookup', {})))
    for source in sources:
        if source.get('status') == 'unavailable' and source.get('error'):
            events.append(dict(module='lyrics', code='lyric_provider_unavailable',
                provider=source.get('provider'), http_status=source.get('http_status'), error_type=source.get('error')))
    reports = [lookup, lookup.get('published_lookup', {})]
    if crosscheck:
        reports += crosscheck.get('sources', []) + crosscheck.get('unavailable_sources', [])
    seen = set()
    for source in reports:
        for rejected in source.get('rejected_candidates', []):
            if rejected.get('reason') != 'candidate_unavailable':
                continue
            event = dict(module='lyrics', code='lyric_candidate_unavailable', provider=source.get('provider'),
                track_id=rejected.get('track_id'), http_status=rejected.get('http_status'), error_type=rejected.get('error'))
            key = json.dumps(event, sort_keys=True)
            if key not in seen:
                events.append(event)
                seen.add(key)
    return events


def review_attempt(attempt):
    """Apply the no-empty-subtitles rule to evidence without rewriting old attempts."""
    evidence = attempt.get('evidence', {})
    fallback = evidence.get('quality', {}).get('playback_fallback', {})
    if attempt.get('outcome') in ('clean', 'soft_fail') and (
            evidence.get('subtitle_lines') == 0 or fallback.get('subtitle_mode') == 'unavailable'):
        return dict(attempt, outcome='hard_fail', error=dict(code='subtitles_failed',
            message='未生成任何可用字幕；音轨和诊断已保留', cause=fallback.get('error')),
            soft_failures=[e for e in attempt.get('soft_failures', []) if e.get('subtitle_mode') != 'unavailable'])
    return attempt


def summarize(cases):
    outcomes = Counter()
    hard, soft, soft_songs = Counter(), Counter(), Counter()
    for case in cases:
        attempt = review_attempt((case.get('audio_attempts') or [{}])[-1])
        outcomes[attempt.get('outcome', 'pending')] += 1
        if attempt.get('outcome') == 'hard_fail':
            hard[(attempt.get('error') or {}).get('code', 'unknown')] += 1
        for event in attempt.get('soft_failures', []):
            soft[event['code']] += 1
        soft_songs.update({e['code'] for e in attempt.get('soft_failures', [])})
    return dict(total=len(cases), outcomes=dict(outcomes), hard_failures=dict(hard), soft_failures=dict(soft),
                soft_failure_songs=dict(soft_songs))


def working_bytes(store, case):
    """Conservative audio-only peak for admission; the live cache remains authoritative."""
    for job in store.list(-1):
        if job['input']['url'] != case['url']:
            continue
        folder = store.root/job['id']
        if not (folder/'download/source.mkv').is_file():
            continue
        try:
            # Reuse local decode metadata; never make a provider request.
            if (folder/'source-native.json').is_file():
                native = json.loads((folder/'source-native.json').read_text())
                rate, channels = native['sample_rate'], native['channels']
                duration = native['frames']/rate
            else:
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                    '-show_entries', 'stream=sample_rate,channels:format=duration', '-of', 'json',
                    str(folder/'download/source.mkv')], timeout=30))
                rate, channels = int(probe['streams'][0]['sample_rate']), probe['streams'][0]['channels']
                duration = float(probe['format']['duration'])
            if not all(math.isfinite(v) and v > 0 for v in (rate, channels, duration)):
                return None
            # Six native float64 files + two temporary int24 previews; five
            # stereo float32 separation files; compressed previews/local clips.
            return math.ceil(duration*(rate*channels*54 + 44100*2*20 + 256000) + 64*1024**2)
        except (OSError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, subprocess.SubprocessError):
            return None  # Unknown size runs alone, with the ordinary live monitor.
    return None


def run_once(manifest, data_dir, report, prepared_dir, base, *, case_urls=(), concurrency=1):
    if concurrency < 1:
        raise ValueError('Survey concurrency must be positive')
    frozen = json.loads(manifest.read_text())
    selection = [dict(url=c['url'], language=c['language']) for c in frozen['cases']]
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if selection_hash != frozen['selection_sha256']:
        raise ValueError('Fixed selection changed')
    if set(case_urls)-{c['url'] for c in selection}:
        raise ValueError('Requested cases are outside the fixed selection')
    selected = [dict(c, title=original['title']) for c, original in zip(selection, frozen['cases'])
                if not case_urls or c['url'] in case_urls]
    with manifest.with_suffix('.lock').open('a') as lock, httpx.Client(timeout=30) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = client.get(base+'/health').raise_for_status().json()
        version = health['pipeline_version']
        if not health.get('preparation_paused') or health.get('running_job'):
            raise ValueError('Pause the normal worker before the one-pass survey')
        state = json.loads(report.read_text()) if report.exists() else dict(
            run_id=report.parent.name, started_at=stamp(), selection_sha256=selection_hash,
            pipeline_version=version, controller_sha256=digest(Path(__file__)), scope='playback_audio',
            fixed_selection_size=len(selection), cases=[dict(c, audio_attempts=[]) for c in selected])
        if (state['selection_sha256'] != selection_hash or state['pipeline_version'] != version or
                state['controller_sha256'] != digest(Path(__file__)) or
                [c['url'] for c in state['cases']] != [c['url'] for c in selected]):
            raise ValueError('A survey must keep its frozen selection, requested cases and pipeline version')
        if any((c.get('audio_attempts') or [{}])[-1].get('outcome') == 'running' for c in state['cases']):
            raise ValueError('Reconcile the interrupted attempt; never resubmit it automatically')
        store = Store(data_dir)
        state_lock = threading.RLock()
        pending, peaks = {}, {}
        remaining = [c for c in state['cases'] if not c['audio_attempts']]

        def run_case(case, packet_path, packet):
            started = time.perf_counter()
            with state_lock:
                attempt = dict(outcome='running', submitted_at=stamp(), pipeline_version=version,
                               prepared_lyrics_sha256=digest(packet_path), lookup_seconds=packet['seconds'])
                case['audio_attempts'].append(attempt)
                state.update(summary=summarize(state['cases']), updated_at=stamp())
                save(report, state)
            entry, job, folder, probe_finished = None, None, None, False
            try:
                with state_lock:
                    room = client.get(base+'/v1/room').raise_for_status().json()
                    existing = {e['id'] for e in room['entries']}
                    # Reuse audio checkpoints, but leave an existing full MV intact.
                    force = any(j['input']['url'] == case['url'] and
                                (j['status'] == 'completed' or j.get('restore_video'))
                                for j in store.list(-1))
                    room = client.post(base+'/v1/room/entries', json=dict(
                        url=case['url'], source_title=case['title'], force=force)).raise_for_status().json()
                    entry = next(e for e in room['entries'] if e['id'] not in existing)
                    job = client.get(f'{base}/v1/jobs/{entry["job_id"]}').raise_for_status().json()
                    folder = data_dir/job['id']
                    attempt.update(job_id=job['id'], entry_id=entry['id'])
                    save(report, state)
                    if not (folder/'download/source.mkv').is_file():
                        store.cache.reuse_download(folder, case['url'])
                    if digest(folder/'download/source.mkv') != packet['source_sha256']:
                        raise RuntimeError('Cached audio does not match the prepared lyric source')
                    shutil.copy2(packet_path, folder/'prepared-lyrics.json')
                command = [sys.executable, '-u', '-m', 'karaoke.pipeline', '--request', str(folder/'request.json'),
                           '--output', str(folder), '--playback-audio', '--prepared-lyrics', str(folder/'prepared-lyrics.json')]
                code = run_probe(client, base, folder, command, session=session)
                probe_finished = True
                if client.get(base+'/health').raise_for_status().json()['pipeline_version'] != version:
                    probe_finished = False
                    raise RuntimeError('Pipeline changed during this case')
                if code:
                    raise RuntimeError('Pipeline failed; see retained diagnostics')
                evidence = verify_outputs(client, base, data_dir, job, audio_only=True, playback_policy=True)
                result = json.loads((folder/'audio-result.json').read_text())
                events = soft_failures(result, packet)
                with state_lock:
                    attempt.update(outcome='soft_fail' if events else 'clean', soft_failures=events,
                                   evidence=evidence, finished_at=stamp())
                    store.update(job['id'], status='audio_ready', pipeline_version=version, audio_result=result,
                        song_info={k:result[k] for k in ('title','artist','language','duration_seconds')}, active_stages=[], error=None)
            except Exception as error:
                with state_lock:
                    details = json.loads((folder/'error.json').read_text()) if folder and (folder/'error.json').is_file() else dict(
                        code='verification_failed' if probe_finished else 'survey_infrastructure_failed', message=str(error)[:600])
                    attempt.update(outcome='hard_fail', error=details, finished_at=stamp())
                    quality = json.loads((folder/'quality.json').read_text()) if folder and (folder/'quality.json').is_file() else {}
                    attempt['soft_failures'] = soft_failures(dict(quality=quality), packet)
                    if job:
                        store.update(job['id'], status='failed', error=details, active_stages=[])
            finally:
                with state_lock:
                    if folder:
                        diagnostic = {}
                        record_failure(diagnostic, dict(id=job['id'], error=attempt.get('error') or dict(
                            code='survey_diagnostic', message=attempt['outcome'])), folder)
                        attempt.update({k:diagnostic[k] for k in ('reproduction_file','diagnostic_sha256') if k in diagnostic})
                    if entry:
                        room = client.get(base+'/v1/room').raise_for_status().json()
                        if room['current_id'] != entry['id']:
                            client.post(base+'/v1/room/commands', json=dict(action='remove', entry_id=entry['id'],
                                revision=room['revision'])).raise_for_status()
                    state.update(summary=summarize(state['cases']), updated_at=stamp())
                    save(report, state)
                    print(json.dumps(dict(title=case['title'], **attempt, summary=state['summary']), ensure_ascii=False), flush=True)
            if not probe_finished:
                session.cancelled.set()

            with state_lock:
                attempt['wall_seconds'] = round(time.perf_counter()-started, 3)
                save(report, state)

        with ProbeSession(data_dir) as session, ThreadPoolExecutor(max_workers=concurrency) as pool:
            state['concurrency'] = concurrency
            try:
                while remaining or pending:
                    for future in list(pending):
                        if future.done():
                            future.result()
                            del pending[future]
                    if session.cancelled.is_set() or report.with_suffix('.pause').exists():
                        break
                    if client.get(base+'/health').raise_for_status().json()['pipeline_version'] != version:
                        session.cancelled.set()
                        raise ValueError('Pipeline changed during the one-pass survey')
                    if remaining and len(pending) < concurrency:
                        with state_lock:
                            old_wait = (state.get('waiting_for_lyrics'), state.get('waiting_for_cache'))
                            state.pop('waiting_for_lyrics', None)
                            state.pop('waiting_for_cache', None)
                            available = None
                            if pending:
                                sizes = []
                                for active, estimate in pending.values():
                                    attempt = (active['audio_attempts'] or [{}])[-1]
                                    size = store.cache.size(data_dir/attempt['job_id']) if attempt.get('job_id') else 0
                                    sizes.append((estimate, size))
                                # Read storage after sizes so ongoing writes over-reserve.
                                storage = client.get(base+'/v1/storage').raise_for_status().json()
                                reserved = sum(max(0, (estimate or storage['limit_bytes'])-size)
                                               for estimate, size in sizes)+store.cache.headroom
                                available = min(storage['limit_bytes']-storage['protected_bytes']-reserved,
                                                storage['free_bytes']-store.cache.minimum_free-reserved)
                            admitted = False
                            for case in remaining:
                                packet_path = prepared_dir/(hashlib.sha256(case['url'].encode()).hexdigest()+'.json')
                                if not packet_path.exists():
                                    state.setdefault('waiting_for_lyrics', case['url'])
                                    continue
                                if concurrency > 1 and case['url'] not in peaks:
                                    peaks[case['url']] = working_bytes(store, case)
                                peak = peaks.get(case['url'])
                                if pending and (peak is None or peak > available):
                                    state.setdefault('waiting_for_cache', case['url'])
                                    continue
                                packet = json.loads(packet_path.read_text())
                                future = pool.submit(run_case, case, packet_path, packet)
                                pending[future] = (case, peak)
                                remaining.remove(case)
                                admitted = True
                                break
                            if admitted or old_wait != (state.get('waiting_for_lyrics'), state.get('waiting_for_cache')):
                                state.update(summary=summarize(state['cases']), updated_at=stamp())
                                save(report, state)
                        if admitted:
                            continue
                        if not pending and (prepared_dir/'aborted.json').exists():
                            break
                    if pending:
                        wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    elif remaining:
                        time.sleep(1)
                # A pause stops admission; already-started songs finish normally.
                # Infrastructure cancellation is observed by every probe monitor.
                for future in pending:
                    future.result()
            except BaseException:
                session.cancelled.set()
                raise
        if all(c['audio_attempts'] and c['audio_attempts'][-1]['outcome'] != 'running' for c in state['cases']):
            state['finished_at'] = stamp()
        state.pop('waiting_for_lyrics', None)
        state.pop('waiting_for_cache', None)
        state.update(summary=summarize(state['cases']), updated_at=stamp())
        save(report, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-dir', 'report', 'prepared-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8746')
    parser.add_argument('--case-url', action='append', default=[], help='Recheck only this URL from the unchanged fixed manifest')
    parser.add_argument('--concurrency', type=int, default=1, help='Maximum in-flight songs; shared cache and stage locks still apply')
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    run_once(args.manifest, args.data_dir, args.report, args.prepared_dir, args.base_url.rstrip('/'), case_urls=args.case_url, concurrency=args.concurrency)


if __name__ == '__main__':
    main()
