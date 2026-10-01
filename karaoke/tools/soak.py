"""Reproducible whole-song acceptance runs through the room's song queue."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time

import httpx

from karaoke.job_schema import provider_error, retry_after_seconds, video_url


REQUIRED = {'mv', 'instrumental', 'backing_vocals', 'lead_vocals', 'subtitles',
            'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'}
GROUPS = {
    'Chinese': (40, ['周杰倫', '五月天', '林俊傑', '孫燕姿', '蔡依林', '鄧紫棋',
                     '張惠妹', '梁靜茹', '田馥甄', '李榮浩', '鳳凰傳奇', '周深']),
    'Japanese': (30, ['米津玄師', 'YOASOBI', 'Aimer', '宇多田ヒカル', 'Official髭男dism',
                      'King Gnu', 'あいみょん', 'ヨルシカ', 'Mrs. GREEN APPLE']),
    'English': (30, ['Adele', 'Ed Sheeran', 'Taylor Swift', 'Coldplay', 'Bruno Mars',
                     'Billie Eilish', 'Dua Lipa', 'Imagine Dragons', 'Linkin Park', 'Eminem']),
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def select_cases(pool, quotas, seed):
    """Freeze selection before any processing outcome is known."""
    rng, seen, chosen = random.Random(seed), set(), []
    for language, count in quotas.items():
        candidates = sorted((item for item in pool if item['language'] == language),
                            key=lambda item: item['url'])
        rng.shuffle(candidates)
        selected = []
        for item in candidates:
            url = video_url(item['url'])
            # Brackets often enclose the actual song title, so retain their contents.
            title = re.sub(r'official|music video|lyric video|lyrics|video|audio|ultrahd|高清|高畫質|完整版|官方|華納|mv|hd|4k', '', item['title'], flags=re.I)
            if item['language'] == 'Chinese':
                chinese = ''.join(re.findall(r'[\u3400-\u9fff]', title))
                artist = (item.get('query') or '').split(' official')[0]
                chinese = chinese.replace(artist, '')
                if chinese:
                    title = artist+chinese
            key = ''.join(c for c in title.casefold() if c.isalnum())
            if url in seen or key in seen:
                continue
            seen.update((url, key))
            selected.append(dict(item, url=url, attempts=[]))
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'{language}: only {len(selected)} distinct candidates for {count} slots')
        chosen.extend(selected)
    rng.shuffle(chosen)
    return chosen


def sample(path, data_dir, seed):
    from karaoke.video_search import youtube_entries, official_source, matches_query
    if path.exists():
        raise ValueError('The fixed manifest already exists; refusing to replace its cases')
    pool_path = path.with_name('candidate-pool.json')
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {'queries': {}, 'items': []}
    with sqlite3.connect(f'file:{data_dir}/jobs.sqlite3?mode=ro', uri=True) as db:
        excluded = {json.loads(row[0])['input']['url'] for row in db.execute('SELECT doc FROM jobs')}
    for lyrics in (data_dir.parent/'lyrics').glob('*.json'):
        value = json.loads(lyrics.read_text()).get('source_url')
        if value:
            try:
                excluded.add(video_url(value))
            except ValueError:  # Older research sources can be local files or lyric sites.
                pass
    for language, (_, artists) in GROUPS.items():
        for artist in artists:
            query = artist + ' official music video'
            if query in pool['queries']:
                continue
            items = youtube_entries(query)
            added = 0
            for item in items:
                evidence = official_source('youtube', item)
                duration = item.get('duration') or 0
                if (not evidence or not matches_query(artist, item) or not 90 <= duration <= 480
                        or item.get('live_status') in ('is_live', 'is_upcoming')):
                    continue
                url = video_url(item['url'])
                if url in excluded or any(x['url'] == url for x in pool['items']):
                    continue
                pool['items'].append(dict(url=url, title=item['title'], language=language,
                    duration=duration, channel=item.get('channel'), evidence=evidence,
                    provider='youtube', query=query))
                added += 1
            pool['queries'][query] = dict(at=stamp(), added=added)
            save(pool_path, pool)
            print(json.dumps({'query': query, 'added': added}, ensure_ascii=False), flush=True)
            time.sleep(1)
    queries = {artist+' official music video' for _, artists in GROUPS.values() for artist in artists}
    cases = select_cases([item for item in pool['items'] if item['query'] in queries],
                         {k: v[0] for k, v in GROUPS.items()}, seed)
    selection = [dict(url=c['url'], language=c['language']) for c in cases]
    save(path, dict(schema_version=1, seed=seed, created_at=stamp(),
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
        cache_limit_bytes=10*1024**3, cases=cases, provider_backoff={}))


def eligible(case, version, now):
    attempts = case.get('attempts', [])
    if not attempts:
        return True
    last = attempts[-1]
    if last['status'] in ('submitted', 'running', 'blocked'):
        return False
    if last['status'] == 'external':
        return now >= last['next_retry_at'] and last.get('external_attempt', 1) < 3
    if last['status'] == 'cache_evicted':
        return True
    # A software failure is retried only after a fix. Prior passes must also
    # pass on the final deployed version, so fixes are exercised together.
    return last['pipeline_version'] != version


def retry_deadline(state, case):
    # Song metadata is optional; its text-model quota must not block ASR.
    required = (case['provider'], 'openai', 'openai:whisper-1',
                'openai:'+os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe'))
    return max((state['provider_backoff'].get(key, 0) for key in required), default=0)


def retry_fields(error, diagnostic, number, *, failed_at):
    """Scope retries to the failure and respect explicit server delays."""
    exponent = 2**min(10, max(0, number-1))
    fields = dict(status='failed', provider_blocked=False, next_retry_at=None)
    if error.get('provider') == 'openai':
        scope = 'openai'+(':'+error['model'] if error.get('model') else '')
        fields.update(status='external' if error['retryable'] else 'blocked', retry_provider=scope,
                      provider_blocked=error['code'] == 'openai_rate_limit')
        if error['retryable']:
            fields['next_retry_at'] = failed_at+max(error.get('retry_after_seconds', 0), min(900, 30*exponent))
        return fields
    if error['code'] == 'interrupted' or re.search(r'^ImportError: dlopen\(.*system is shutting down', diagnostic, re.M):
        return dict(fields, status='external', retry_provider='local', next_retry_at=failed_at+min(900, 60*exponent))
    if error['code'] != 'download_failed':
        return fields
    text = (str(error)+' '+diagnostic).lower()
    if any(s in text for s in ('sign in', 'login', '登录', 'risk control', '-352', 'captcha',
                                'video unavailable', 'private video', '无法访问')):
        return dict(fields, status='blocked')
    rate_limited = bool(re.search(r'(?:http error|error code:|http/\d(?:\.\d)?)\s*429\b', text))
    forbidden = bool(re.search(r'(?:http error|error code:|http/\d(?:\.\d)?)\s*403\b', text))
    if rate_limited or forbidden or any(s in text for s in ('timed out', 'connection reset',
            'proxy', '代理', '超时', 'failed to resolve', 'temporary failure in name resolution',
            'nodename nor servname', 'name or service not known', 'dns')):
        delay = max(retry_after_seconds(diagnostic) or 0, min(1800, (300 if forbidden else 60)*exponent))
        return dict(fields, status='external', provider_blocked=rate_limited, next_retry_at=failed_at+delay)
    return fields


def verify_stems(paths, expected_hash):
    import numpy as np
    import soundfile as sf
    digest = hashlib.sha256()
    with ExitStack() as stack:
        files = [stack.enter_context(sf.SoundFile(paths[name]))
                 for name in ('lead_vocals', 'backing_vocals', 'instrumental')]
        formats = {(f.samplerate, f.channels, len(f), f.subtype) for f in files}
        if len(formats) != 1 or files[0].subtype != 'DOUBLE' or not len(files[0]):
            raise ValueError('Three stems have incompatible formats or are not float64 masters')
        while True:
            blocks = [f.read(65536, dtype='float64', always_2d=True) for f in files]
            if not len(blocks[0]):
                break
            if any(not np.isfinite(block).all() for block in blocks):
                raise ValueError('Non-finite stem samples')
            combined = np.ascontiguousarray((blocks[0]+blocks[1])+blocks[2], dtype='<f8')
            digest.update(memoryview(combined).cast('B'))
    if digest.hexdigest() != expected_hash:
        raise ValueError('Three stems do not reconstruct the decoded source bitwise')
    return digest.hexdigest()


def verify_outputs(client, base, data_dir, job, *, audio_only=False):
    from karaoke.subtitles import normalized
    folder = (data_dir/job['id']).resolve()
    result = json.loads((folder/('audio-result.json' if audio_only else 'result.json')).read_text())
    required = REQUIRED-{'mv'} if audio_only else REQUIRED
    if not required <= result['artifacts'].keys():
        raise ValueError('Missing required output artifacts')
    paths, hashes = {}, {}
    for name, artifact in result['artifacts'].items():
        path = (folder/artifact['path']).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size != artifact['bytes']:
            raise ValueError(f'Invalid artifact: {name}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != artifact['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
        for start in (() if audio_only else (0, max(0, artifact['bytes']-1024))):
            end = min(artifact['bytes']-1, start+1023)
            response = client.get(f'{base}/v1/jobs/{job["id"]}/artifacts/{name}',
                                  headers={'Range': f'bytes={start}-{end}'})
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise ValueError(f'Artifact range playback failed: {name}, {response.status_code}')
            with path.open('rb') as stream:
                stream.seek(start)
                if response.content != stream.read(end-start+1):
                    raise ValueError(f'Artifact range content mismatch: {name}')
        paths[name], hashes[name] = path, digest
    reconstruction = verify_stems(paths, result['reconstruction']['source_pcm_sha256'])
    cues = json.loads(paths['subtitles'].read_text())
    units = [u for cue in cues for u in cue['units']]
    duration = result['duration_seconds']
    if not units or any(not 0 <= u['start'] < u['end'] <= duration+.05 for u in units):
        raise ValueError('Invalid or empty subtitle intervals')
    if any(a['start'] >= b['start'] for a, b in zip(units, units[1:])):
        raise ValueError('Non-monotonic word onsets')
    if any(normalized(''.join(u['text'] for u in c['units'])) != normalized(c['text']) for c in cues):
        raise ValueError('Subtitle units omit lyric text')
    for name in ('mv', 'instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'):
        if audio_only and name == 'mv':
            continue
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(paths[name])], timeout=30))
        needed = {'audio', 'video'} if name == 'mv' else {'audio'}
        streams = [s for s in probe['streams'] if s['codec_type'] in needed]
        if {s['codec_type'] for s in streams} != needed or any(
                abs(float(s.get('duration', probe['format']['duration']))-duration) > .15 for s in streams):
            raise ValueError(f'Truncated media or missing stream: {name}')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(paths[name])]
        command += ['-map', '0:v:0', '-map', '0:a:0'] if name == 'mv' else ['-map', '0:a:0']
        subprocess.run(command+['-f', 'null', '-'], check=True, capture_output=True, timeout=300)
    return dict(validation_scope='audio_and_lyrics' if audio_only else 'full_mv',
                artifact_sha256=hashes, reconstructed_pcm_sha256=reconstruction,
                subtitle_lines=len(cues), subtitle_units=len(units),
                processing_seconds=result['processing_seconds'], timings=result['timings'],
                quality=result['quality'], device=result['device'], reused_stages=result.get('reused_stages', []))


def wait_job(client, base, job_id):
    job = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
    if job['status'] not in ('queued', 'running'):
        return job
    with client.stream('GET', base+'/v1/room/events') as events:
        events.raise_for_status()
        for line in events.iter_lines():
            if not line.startswith('data: '):
                continue
            snapshot = json.loads(line[6:])
            library = snapshot.get('library', [])
            if any(song.get('url') == job['input']['url'] and song.get('status') not in ('queued', 'running')
                   for song in library):
                current = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
                if current['status'] not in ('queued', 'running'):
                    return current
    raise ConnectionError('Room event stream ended before completion')


def run_probe(client, base, folder, command):
    """Run one registered job under the live service's shared cache budget."""
    health = client.get(base+'/health').raise_for_status().json()
    job = client.get(f'{base}/v1/jobs/{folder.name}').raise_for_status().json()
    if not health.get('preparation_paused') or health.get('running_job') or job['status'] != 'queued':
        raise ValueError('Pause preparation and register a queued job before a focused probe')
    with (folder.parent/'focused-probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder/'probe.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=folder.parent.parent.parent,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    storage = client.get(base+'/v1/storage').raise_for_status().json()
                    if storage['used_bytes'] > storage['limit_bytes'] or not storage['can_prepare']:
                        raise ValueError('Focused probe reached the shared cache or free-space limit')
                    health = client.get(base+'/health').raise_for_status().json()
                    if not health.get('preparation_paused') or health.get('running_job'):
                        raise ValueError('Preparation resumed during the focused probe')
                    result = process.poll()
                    if result is not None:
                        return result
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def probe(path, data_dir, base, job_id):
    with httpx.Client(timeout=30) as client:
        job = client.get(f'{base}/v1/jobs/{job_id}').raise_for_status().json()
        state = json.loads(path.read_text())
        case = next(c for c in state['cases'] if video_url(c['url']) == job['input']['url'])
        retry_at = max(retry_deadline(state, case),
                       (case.get('attempts') or [{}])[-1].get('next_retry_at') or 0)
        if time.time() < retry_at:
            raise ValueError(f'Provider backoff lasts until {datetime.fromtimestamp(retry_at, timezone.utc).isoformat()}')
        folder = (data_dir/job['id']).resolve()
        command = [sys.executable, '-u', '-m', 'karaoke.pipeline', '--request', str(folder/'request.json'),
                   '--output', str(folder), '--audio-only']
        version = client.get(base+'/health').raise_for_status().json()['pipeline_version']
        code = run_probe(client, base, folder, command)
        if code:
            raise RuntimeError(f'Audio probe failed ({code}); inspect {folder / "error.json"} and probe.log')
        evidence = verify_outputs(client, base, data_dir, job, audio_only=True)
        if version != client.get(base+'/health').raise_for_status().json()['pipeline_version']:
            raise ValueError('Pipeline changed during the focused probe')
        evidence['pipeline_version'] = version
        save(folder/'audio-verification.json', evidence)
        print(json.dumps(evidence, ensure_ascii=False), flush=True)


def record_failure(attempt, job, folder):
    error = job.get('error') or dict(code='verification_failed', message='Output validation failed')
    diagnostics = {}
    for path in sorted(folder.glob('*.log')):
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-16384))
                diagnostics[path.name] = stream.read().decode(errors='replace')
        except FileNotFoundError:  # Cache eviction may race a completed job's inspection.
            continue
    names = ('worker.log', 'download.log', 'transcribe.log', error['code'].removesuffix('_failed')+'.log')
    logs = [diagnostics[name][-4000:] for name in dict.fromkeys(names) if name in diagnostics]
    # Older workers reported these as generic stage failures. Classify their
    # retained logs too, so a code rollout does not cause inappropriate retries.
    parsed = provider_error('\n'.join(logs))
    error = {**error, **parsed} if parsed else error
    attempt.update(status='failed', error=error, finished_at=stamp(),
        error_fingerprint=hashlib.sha256(json.dumps(error, sort_keys=True).encode()).hexdigest())
    attempt['diagnostic_tail'] = '\n'.join(logs)
    for pattern in ('asr/*.json', 'alignment/*.json', 'phonemes/*.json', 'context/**/*.json',
                    'repeated-lyrics/*.json', 'english-onsets/*.json',
                    'lyrics.json', 'anchors.json', 'word-cues.json', 'ordered-cues.json', 'quality.json'):
        for path in folder.glob(pattern):
            try:
                if path.stat().st_size <= 2*1024**2:
                    diagnostics[str(path.relative_to(folder))] = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
    if diagnostics:
        target = folder.parent/'soak-100'/f'{job["id"]}-{attempt["error_fingerprint"][:12]}.json.gz'
        target.parent.mkdir(exist_ok=True)
        with gzip.open(target, 'wt', encoding='utf-8') as output:
            json.dump(diagnostics, output, ensure_ascii=False)
        attempt['reproduction_file'] = str(target)
    attempt.update(retry_fields(error, '\n'.join(logs), attempt.get('external_attempt', 1), failed_at=time.time()))


def queue_cases(client, base, state, version, persist):
    """Use the same room API as a person's add button; no test-only queue limit."""
    room = client.get(base+'/v1/room').raise_for_status().json()
    # Retry software regressions first without changing the fixed selection or room order.
    cases = sorted(state['cases'], key=lambda case: not (
        case['attempts'] and case['attempts'][-1]['status'] == 'failed'))
    for case in cases:
        if len(room['entries']) >= 100:
            break
        last = case['attempts'][-1] if case['attempts'] else None
        if last and last['status'] in ('submitted', 'running'):
            if last.get('entry_id'):
                continue
            job = client.get(f'{base}/v1/jobs/{last["job_id"]}').raise_for_status().json()
            if job['status'] not in ('queued', 'running', 'completed'):
                continue
            payload = dict(job_id=last['job_id'])
        else:
            if not eligible(case, version, time.time()) or time.time() < retry_deadline(state, case):
                continue
            payload = dict(url=case['url'], source_title=case['title'],
                force=bool(last and last['status']=='passed' and last['pipeline_version']!=version))
        before = {e['id'] for e in room['entries']}
        response = client.post(base+'/v1/room/entries', json=payload)
        if response.status_code == 409:
            break  # The user may have filled the last available queue slot.
        room = response.raise_for_status().json()
        entry = next(e for e in room['entries'] if e['id'] not in before)
        if not last or last['status'] not in ('submitted', 'running'):
            job = client.get(f'{base}/v1/jobs/{entry["job_id"]}').raise_for_status().json()
            last = dict(job_id=job['id'], pipeline_version=job['pipeline_version'], status='submitted',
                        submitted_at=stamp(), external_attempt=1+sum(a['status']=='external' for a in case['attempts']))
            case['attempts'].append(last)
        last['entry_id'] = entry['id']
        persist()


def release_finished_entries(client, base, state):
    room = client.get(base+'/v1/room').raise_for_status().json()
    owned = {a.get('entry_id') for c in state['cases'] for a in c['attempts']
             if a['status'] not in ('submitted', 'running')}
    for entry in list(room['entries']):
        if entry['id'] not in owned or entry['id'] == room['current_id']:
            continue
        response = client.post(base+'/v1/room/commands', json=dict(action='remove',
            entry_id=entry['id'], revision=room['revision']))
        if response.status_code == 409:
            return
        room = response.raise_for_status().json()


def run(path, data_dir, base):
    # Only this controller writes the manifest; it submits to the existing worker.
    with path.with_suffix('.lock').open('w') as lock, httpx.Client(timeout=40) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if path.with_suffix('.pause').exists():
                print('Paused by controller file', flush=True)
                return
            state = json.loads(path.read_text())
            selection = [dict(url=c['url'], language=c['language']) for c in state['cases']]
            if hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest() != state['selection_sha256']:
                raise ValueError('Fixed case selection changed')
            health = client.get(base+'/health').raise_for_status().json()
            version = health['pipeline_version']
            state['current_pipeline_version'] = version
            state['updated_at'] = stamp()
            state['passed_current_version'] = sum(bool(c['attempts'] and c['attempts'][-1]['status'] == 'passed'
                and c['attempts'][-1]['pipeline_version'] == version) for c in state['cases'])
            release_finished_entries(client, base, state)
            failed = any(c['attempts'] and c['attempts'][-1]['status']=='failed' and
                         c['attempts'][-1]['pipeline_version']==version for c in state['cases'])
            storage = client.get(base+'/v1/storage').raise_for_status().json()
            if storage['limit_bytes'] != state['cache_limit_bytes']:
                raise RuntimeError('Cache limit changed; refusing to silently change it')
            if not failed and storage['can_prepare']:
                queue_cases(client, base, state, version, lambda: save(path, state))
            active = min(((c, c['attempts'][-1]) for c in state['cases'] if c['attempts'] and
                          c['attempts'][-1]['status'] in ('submitted', 'running')),
                         key=lambda item: item[1]['submitted_at'], default=None)
            if active:
                case, attempt = active
                try:
                    job = wait_job(client, base, attempt['job_id'])
                except (httpx.HTTPError, ConnectionError) as error:
                    print(f'Waiting for API reconnect: {type(error).__name__}', flush=True)
                    time.sleep(15)
                    continue
                attempt['pipeline_version'] = job.get('pipeline_version', attempt['pipeline_version'])
                if job['status'] == 'completed':
                    try:
                        evidence = verify_outputs(client, base, data_dir, job)
                        attempt.update(status='passed', finished_at=stamp(), evidence=evidence)
                    except Exception as error:
                        job['error'] = dict(code='verification_failed', message=str(error)[:1000])
                        record_failure(attempt, job, data_dir/job['id'])
                elif job['status'] in ('evicted', 'video_evicted'):
                    attempt.update(status='cache_evicted', finished_at=stamp())
                else:
                    record_failure(attempt, job, data_dir/job['id'])
                if attempt['status'] == 'external' and attempt['provider_blocked']:
                    provider = attempt.get('retry_provider', case['provider'])
                    state['provider_backoff'][provider] = max(
                        state['provider_backoff'].get(provider, 0), attempt['next_retry_at'])
                save(path, state)
                print(json.dumps({'title': case['title'], 'job_id': job['id'], 'status': attempt['status'],
                                  'error': attempt.get('error')}, ensure_ascii=False), flush=True)
                continue
            save(path, state)
            if state['passed_current_version'] == len(state['cases']):
                print(f'All {len(state["cases"])} cases passed on {version}', flush=True)
                return
            # Stop on reproducible software failures until the maintainer deploys a fix.
            if failed:
                print('Needs repair; retained failed job diagnostics', flush=True)
                return
            if not storage['can_prepare']:
                print('Waiting for cache space; user queue remains protected', flush=True)
                time.sleep(30)
                continue
            candidates = [c for c in state['cases'] if eligible(c, version, time.time()) and
                          time.time() >= retry_deadline(state, c)]
            if not candidates:
                print('No eligible cases; external failures are on backoff or need user action', flush=True)
                return
            time.sleep(15)  # User-owned entries may currently fill the room.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sample', 'run', 'probe'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8746')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--job-id', help='A registered, queued job for an audio-only probe')
    args = parser.parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'sample':
        sample(args.manifest, args.data_dir, args.seed)
    elif args.action == 'probe':
        if not args.job_id:
            parser.error('probe requires --job-id')
        probe(args.manifest, args.data_dir, args.base_url.rstrip('/'), args.job_id)
    else:
        run(args.manifest, args.data_dir, args.base_url.rstrip('/'))


if __name__ == '__main__':
    main()
