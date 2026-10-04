"""One pass over a frozen corpus using the normal playback policy, without video."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import httpx

from karaoke.api import Store
from karaoke.stage_cache import digest
from karaoke.tools.soak import record_failure, run_probe, save, stamp, verify_outputs


def soft_failures(result, prepared):
    quality = result.get('quality', {})
    events = []
    fallback = quality.get('playback_fallback')
    if fallback and fallback['subtitle_mode'] != 'unavailable':
        events.append(dict(module='alignment' if fallback['error']['code'].startswith('align') else 'lyrics',
            **fallback['error'], subtitle_mode=fallback['subtitle_mode'],
            estimated_spans=len(fallback.get('estimated_spans', [])),
            omitted_lines=fallback.get('omitted_lines', [])))
    if quality.get('fallback_units') or quality.get('line_order_fallback_lines'):
        events.append(dict(module='alignment', code='alignment_recovered',
            fallback_units=quality.get('fallback_units', 0),
            line_order_fallback_lines=quality.get('line_order_fallback_lines', [])))
    lookup = prepared.get('lookup') or {}
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


def run_once(manifest, data_dir, report, prepared_dir, base):
    frozen = json.loads(manifest.read_text())
    selection = [dict(url=c['url'], language=c['language']) for c in frozen['cases']]
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if selection_hash != frozen['selection_sha256']:
        raise ValueError('Fixed selection changed')
    with manifest.with_suffix('.lock').open('a') as lock, httpx.Client(timeout=30) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = client.get(base+'/health').raise_for_status().json()
        version = health['pipeline_version']
        if not health.get('preparation_paused') or health.get('running_job'):
            raise ValueError('Pause the normal worker before the one-pass survey')
        state = json.loads(report.read_text()) if report.exists() else dict(
            run_id=report.parent.name, started_at=stamp(), selection_sha256=selection_hash,
            pipeline_version=version, controller_sha256=digest(Path(__file__)), scope='playback_audio',
            cases=[dict(c, title=original['title'], audio_attempts=[]) for c, original in zip(selection, frozen['cases'])])
        if (state['selection_sha256'] != selection_hash or state['pipeline_version'] != version or
                state['controller_sha256'] != digest(Path(__file__))):
            raise ValueError('A survey must keep its frozen selection and pipeline version')
        if any((c.get('audio_attempts') or [{}])[-1].get('outcome') == 'running' for c in state['cases']):
            raise ValueError('Reconcile the interrupted attempt; never resubmit it automatically')
        store = Store(data_dir)
        for case in state['cases']:
            if case['audio_attempts']:
                continue
            if report.with_suffix('.pause').exists():
                break
            if client.get(base+'/health').raise_for_status().json()['pipeline_version'] != version:
                raise ValueError('Pipeline changed during the one-pass survey')
            packet_path = prepared_dir/(hashlib.sha256(case['url'].encode()).hexdigest()+'.json')
            while not packet_path.exists():
                state.update(waiting_for_lyrics=case['url'], summary=summarize(state['cases']), updated_at=stamp())
                save(report, state)
                if report.with_suffix('.pause').exists() or (prepared_dir/'aborted.json').exists():
                    return
                time.sleep(5)
            state.pop('waiting_for_lyrics', None)
            packet = json.loads(packet_path.read_text())
            attempt = dict(outcome='running', submitted_at=stamp(), pipeline_version=version,
                           prepared_lyrics_sha256=digest(packet_path), lookup_seconds=packet['seconds'])
            case['audio_attempts'].append(attempt)
            state.update(summary=summarize(state['cases']), updated_at=stamp())
            save(report, state)
            entry, job, folder, probe_finished = None, None, None, False
            try:
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
                code = run_probe(client, base, folder, command)
                probe_finished = True
                if client.get(base+'/health').raise_for_status().json()['pipeline_version'] != version:
                    probe_finished = False
                    raise RuntimeError('Pipeline changed during this case')
                if code:
                    raise RuntimeError('Pipeline failed; see retained diagnostics')
                evidence = verify_outputs(client, base, data_dir, job, audio_only=True, playback_policy=True)
                result = json.loads((folder/'audio-result.json').read_text())
                events = soft_failures(result, packet)
                attempt.update(outcome='soft_fail' if events else 'clean', soft_failures=events,
                               evidence=evidence, finished_at=stamp())
                store.update(job['id'], status='audio_ready', pipeline_version=version, audio_result=result,
                    song_info={k:result[k] for k in ('title','artist','language','duration_seconds')}, active_stages=[], error=None)
            except Exception as error:
                details = json.loads((folder/'error.json').read_text()) if folder and (folder/'error.json').is_file() else dict(
                    code='verification_failed' if probe_finished else 'survey_infrastructure_failed', message=str(error)[:600])
                attempt.update(outcome='hard_fail', error=details, finished_at=stamp())
                quality = json.loads((folder/'quality.json').read_text()) if folder and (folder/'quality.json').is_file() else {}
                attempt['soft_failures'] = soft_failures(dict(quality=quality), packet)
                if job:
                    store.update(job['id'], status='failed', error=details, active_stages=[])
            finally:
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
                break
        if all(c['audio_attempts'] and c['audio_attempts'][-1]['outcome'] != 'running' for c in state['cases']):
            state['finished_at'] = stamp()
        state.update(summary=summarize(state['cases']), updated_at=stamp())
        save(report, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-dir', 'report', 'prepared-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8746')
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    run_once(args.manifest, args.data_dir, args.report, args.prepared_dir, args.base_url.rstrip('/'))


if __name__ == '__main__':
    main()
