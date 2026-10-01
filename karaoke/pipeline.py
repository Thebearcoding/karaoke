"""URL-to-karaoke adapter for the existing, versioned separation/alignment stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import selectors
import subprocess
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
# The research stages predate the package API and use sibling imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .job_schema import JobRequest, provider_error, song_language, video_url
from .song_sources import prepare_song_text, video_song_info
from .stage_progress import parse_progress
from .cache import can_restore_video
from .stage_cache import StageCache


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pipeline_fingerprint():
    digest = hashlib.sha256(b'karaoke-url-api-v1')
    for path in [p for p in sorted(ROOT.glob('*.py'))
                 if not p.name.startswith('test_') and p.name not in ('api.py', 'room.py', 'cache.py')] + [
            ROOT/'subtitle-styles.json', ROOT/'experiments/cjk-onsets-v003/run.py',
            ROOT/'experiments/cjk-onsets-v003/protocol.json']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    # Model/config changes invalidate the cache without hashing GBs on each submission.
    for path in sorted((ROOT/'models/pretrained').glob('*roformer*')):
        stat = path.stat()
        digest.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for path in sorted((ROOT/'models/alignment').rglob('*')):
        if path.is_file() and path.suffix in ('.ckpt', '.pt', '.yaml', '.json'):
            stat = path.stat()
            digest.update(f'{path.relative_to(ROOT)}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    for path in sorted((ROOT/'lyrics').glob('*.json')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class StageFailure(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def processing_error(error, output):
    details = provider_error(error)
    if details is None and isinstance(error, StageFailure) and error.stage == 'transcribe':
        saved = Path(output)/'asr/error.json'
        if saved.is_file():
            details = json.loads(saved.read_text())
        path = Path(output)/'transcribe.log'
        if details is None and path.is_file():
            with path.open('rb') as log:
                log.seek(max(0, path.stat().st_size-16384))
                details = provider_error(log.read().decode(errors='replace'))
    return details or dict(code=getattr(error, 'stage', 'processing')+'_failed',
        message=str(error)[:600] if isinstance(error, (StageFailure, ValueError)) else
        '处理失败；请查看任务目录中的日志，或提供本版本歌词后重试')


def download_failure(log):
    """Expose actionable causes without publishing raw URLs, headers or credentials."""
    diagnostic = log.lower()
    if any(text in diagnostic for text in ('unable to connect to proxy', 'proxyerror', 'tunnel connection failed')):
        return '下载被网络代理拒绝，请检查服务的代理设置后重试'
    if any(text in diagnostic for text in ('sign in', 'login required', 'login_required')):
        return '视频站点要求登录验证，请配置下载登录信息后重试'
    if any(text in diagnostic for text in ('video unavailable', 'private video', 'video is unavailable')):
        return '视频当前无法访问，请换一个可公开播放的链接'
    if any(text in diagnostic for text in ('failed to resolve', 'temporary failure in name resolution',
                                           'nodename nor servname', 'name or service not known')):
        return '无法解析视频站点地址，请检查网络或 DNS 后重试'
    if any(text in diagnostic for text in ('timed out', 'connection reset')):
        return '连接视频站点超时或中断，请稍后重试'
    return '视频下载失败，请稍后重试或换一个视频链接'


def create_poster(source, output, *, thumbnail=None, seconds=30):
    """Cache the video thumbnail, falling back to a frame; artwork is optional."""
    output = Path(output)
    temporary = output.with_name('.poster.tmp.jpg')
    candidates = [(thumbnail, [])] if thumbnail and thumbnail.startswith(('https://', 'http://')) else []
    candidates.append((source, ['-ss', str(seconds)]))
    for media, options in candidates:
        try:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', *options,
                '-i', str(media), '-frames:v', '1', '-vf',
                'scale=960:540:force_original_aspect_ratio=decrease', '-q:v', '3', str(temporary)],
                capture_output=True, timeout=15)
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size:
                temporary.replace(output)
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return False


class Pipeline:
    def __init__(self, request, output, *, separator_python=None, device='auto', text_model='gpt-5-mini',
                 max_duration=900, catalog_root=ROOT, cookies_file=None, audio_only=False):
        self.request = JobRequest.model_validate(request).model_dump()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        default_separator = ROOT.parent.parent/'audio/.venv-separator-local/bin/python'
        self.separator_python = separator_python or (str(default_separator) if default_separator.exists() else sys.executable)
        self.device, self.text_model = device, text_model
        self.max_duration, self.catalog_root, self.cookies_file = max_duration, catalog_root, cookies_file
        self.audio_only = audio_only
        self.active, self.timings, self.lock = set(), {}, threading.Lock()
        self.stage_progress = {}
        self.song_info = {}
        self.source = self.output/'download/source.mkv'
        self.video = self.source
        self.vocals = self.output/'music-separation/song/vocals.wav'
        self.checkpoints = StageCache(self.output)

    def stage_spec(self, name, command):
        """Explicit inputs/outputs prevent a downstream fix from discarding upstream work."""
        alignment_code = [ROOT/p for p in ('align_lyrics_whisper.py', 'align_lyric_phonemes.py',
            'align_english_onsets.py', 'align_repeated_lyrics.py', 'phoneme_progress.py',
            'phonetic_align.py', 'anchors_from_asr.py', 'subtitles.py', 'lyric_progress.py',
            'onset_consensus.py', 'experiments/cjk-onsets-v003/run.py',
            'experiments/cjk-onsets-v003/protocol.json')]
        english = str(ROOT/'align_english_onsets.py') in [str(a) for a in command]
        specs = {
            'decode_audio': (['download/source.mkv'], ['source-44100.wav'], []),
            'separate_music': (['source-44100.wav'], ['music-separation'], [ROOT/'benchmark_neural.py']),
            'package_two_stems': (['download/source.mkv', 'music-separation/song/vocals.wav'],
                                  ['two-stems'], [ROOT/'exact_stems.py']),
            'resample_vocals': (['music-separation/song/vocals.wav'], ['vocals-16k.wav', 'vocals-asr.mp3'], []),
            'transcribe': (['vocals-asr.mp3'], ['asr'], [ROOT/p for p in
                ('transcribe_lyrics.py', 'anchors_from_asr.py', 'subtitles.py', 'job_schema.py')]),
            'separate_lead': (['music-separation/song/vocals.wav'], ['lead-separation'], [ROOT/'benchmark_neural.py']),
            'package_three_stems': (['two-stems', 'lead-separation/song/vocals.wav'],
                                    ['stems'], [ROOT/'exact_stems.py', ROOT/'three_stems.py']),
            'align_words': (['vocals-16k.wav', 'lyrics.json', 'anchors.json', 'alignment-jobs.json'],
                            ['alignment'], alignment_code),
            'align_phonemes': ((['vocals-16k.wav', 'lyrics.json', 'anchors.json', 'english-jobs.json'] if english else
                                ['music-separation/song/vocals.wav', 'word-cues.json', 'phoneme-jobs.json']),
                               ['english-onsets' if english else 'phonemes'], alignment_code),
            'align_repeats': (['music-separation/song/vocals.wav', 'phonemes/cues.json', 'word-cues.json', 'repeat-jobs.json'],
                              ['repeated-lyrics'], alignment_code),
            'align_context': (['music-separation/song/vocals.wav', 'ordered-cues.json', 'context-jobs.json'],
                              ['context'], alignment_code),
        }
        if name not in specs:
            return None
        inputs, outputs, code = specs[name]
        config = dict(command=[str(a) for a in command])
        if name in ('separate_music', 'separate_lead'):
            model = 'vocals_mel_band_roformer' if name == 'separate_music' else 'karaoke_bs_roformer_anvuew'
            code = code+[ROOT/f'models/pretrained/{model}.yaml']
            weights = ROOT/f'models/pretrained/{model}.ckpt'
            config['weights'] = (weights.stat().st_size, weights.stat().st_mtime_ns)
        if name.startswith('align_'):
            config['weights'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted((ROOT/'models/alignment').rglob('*')) if p.is_file() and p.suffix in ('.ckpt', '.pt')]
        if name == 'transcribe':
            config['fallback_model'] = os.getenv('KARAOKE_ASR_FALLBACK_MODEL', 'gpt-4o-transcribe')
        return dict(inputs=[self.output/p for p in inputs], outputs=[self.output/p for p in outputs],
                    config=config, code=code)

    def phase(self, name=None, active=False, *, song_info=None):
        with self.lock:
            if name is not None:
                self.active.add(name) if active else self.active.discard(name)
                self.stage_progress.pop(name, None)
            if song_info is not None:
                self.song_info.update(song_info)
            self.publish_progress()

    def publish_progress(self):
        # Caller holds the lock: parallel stages publish one coherent snapshot.
        progress = dict(active_stages=sorted(self.active), stage_progress=self.stage_progress,
            updated_at=datetime.now(timezone.utc).isoformat(), timings=self.timings,
            song_info=self.song_info)
        write_json(self.output/'progress.json', progress)
        if os.getenv('KARAOKE_EVENTS_STDOUT') == '1':
            print('KARAOKE_PROGRESS ' + json.dumps(progress), flush=True)

    def run(self, name, command, timeout=3600):
        spec = self.stage_spec(name, command)
        if spec is None:
            return self.run_process(name, command, timeout)
        started = time.perf_counter()
        with self.checkpoints.stage(name, **spec) as reused:
            if reused:
                with self.lock:
                    self.timings['reuse_'+name] = round(time.perf_counter()-started, 3)
                    self.publish_progress()
            else:
                self.run_process(name, command, timeout)

    def run_process(self, name, command, timeout=3600):
        self.phase(name, True)
        started = time.perf_counter()
        try:
            with (self.output/f'{name}.log').open('wb') as log, selectors.DefaultSelector() as selector:
                with subprocess.Popen([str(a) for a in command], cwd=ROOT.parent,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as result:
                    selector.register(result.stdout, selectors.EVENT_READ)
                    pending = b''
                    try:
                        while True:
                            remaining = timeout-(time.perf_counter()-started)
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(command, timeout)
                            if not selector.select(remaining):
                                raise subprocess.TimeoutExpired(command, timeout)
                            block = result.stdout.read1(65536)
                            log.write(block)
                            log.flush()
                            lines = (pending+block).split(b'\n')
                            pending = lines.pop() if block else b''
                            for line in lines:
                                value = parse_progress(line.decode(errors='replace'))
                                if value is not None:
                                    with self.lock:
                                        self.stage_progress[name] = value
                                        self.publish_progress()
                            if not block:
                                break
                        result.wait(timeout=max(.001, timeout-(time.perf_counter()-started)))
                    except BaseException:
                        result.kill()
                        raise
            if result.returncode:
                with (self.output/f'{name}.log').open('rb') as log:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell()-16384))
                    diagnostic = log.read().decode(errors='replace')
                if name in ('download', 'download_video'):
                    raise StageFailure(name, download_failure(diagnostic))
                if name == 'align_words' and any(message in diagnostic for message in (
                        'Empty alignment audio', 'FFmpeg failed to read "[]"')):
                    raise StageFailure(name, '歌词对齐失败：歌词时间范围内没有有效音频，请检查歌词与视频版本')
                raise StageFailure(name, f'{name} 阶段失败；具体错误已保存在任务日志中')
        except subprocess.TimeoutExpired as error:
            raise StageFailure(name, f'{name} 阶段超过允许的处理时间') from error
        finally:
            with self.lock:
                self.timings[name] = round(time.perf_counter()-started, 3)
            self.phase(name, False)

    def inputs(self, audio, name):
        import soundfile as sf
        write_json(self.output/name, [dict(id='song', input=str(audio), input_sha256=file_hash(audio),
                                          duration_seconds=sf.info(audio).duration)])

    def download(self, *, artwork=True, format_id=None):
        artwork = artwork and not self.audio_only
        cached = self.output/'source.json'
        if cached.exists() and self.source.is_file() and (self.output/'download/source.info.json').is_file():
            started = time.perf_counter()
            source = json.loads(cached.read_text())
            if source.get('input_url', source['url']) == self.request['url'] and source['sha256'] == file_hash(self.source):
                if not 0 < source['duration'] <= self.max_duration+1:
                    raise StageFailure('download', '缓存视频超过当前时长上限')
                metadata = json.loads((self.output/'download/source.info.json').read_text())
                if source.get('audio_only') and not self.audio_only:
                    metadata = self.promote_video(metadata, source)
                self.phase(song_info=video_song_info(metadata))
                if artwork and not (self.output/'poster.jpg').exists():
                    create_poster(self.video, self.output/'poster.jpg', seconds=min(30, source['duration']*.1))
                self.timings['reuse_download'] = round(time.perf_counter()-started, 3)
                return metadata, source['duration']
        if (self.output/'download').exists():
            shutil.rmtree(self.output/'download')
        (self.output/'download').mkdir()
        url = self.request['url']
        if url.startswith('https://b23.tv/'):
            import httpx
            for _ in range(5):
                response = httpx.get(url, follow_redirects=False, timeout=20)
                if not response.is_redirect:
                    break
                from urllib.parse import urljoin
                url = video_url(urljoin(url, response.headers['location']))
            if url.startswith('https://b23.tv/'):
                raise StageFailure('download', '无法解析 b23.tv 短链，请使用完整 Bilibili 视频链接')
        command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist',
            '--newline', '--progress', '--progress-delta', '.5', '--progress-template',
            'download:KARAOKE_STAGE_PROGRESS {"completed":%(progress.downloaded_bytes)j,'
            '"total":%(progress.total_bytes)j,"unit":"bytes","vcodec":%(info.vcodec)j}',
            '--progress-template', 'postprocess:KARAOKE_STAGE_PROGRESS {"detail":"合并视频"}',
            '--write-info-json', '--no-overwrites', '--retries', '2', '--fragment-retries', '2',
            '--socket-timeout', '25', '--max-filesize', '2G', '--match-filter',
            f'!is_live & duration <= {self.max_duration}', '-f', format_id or (
                'ba' if self.audio_only else 'bv[height<=1080]+ba/b[height<=1080]'),
            '--merge-output-format', 'mkv', '--remux-video', 'mkv',
            '-o', self.output/'download/source.%(ext)s']
        if self.cookies_file:
            command.extend(['--cookies', self.cookies_file])
        command.append(url)
        self.run('download', command, timeout=900)
        info_path = self.output/'download/source.info.json'
        if not self.source.exists() or not info_path.exists():
            raise StageFailure('download', '没有下载到完整视频；请检查链接、直播状态、时长或站点登录要求')
        metadata = json.loads(info_path.read_text())
        self.phase(song_info=video_song_info(metadata))
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(self.source)]))
        required = {'audio'} if self.audio_only else {'audio', 'video'}
        if not required <= {s['codec_type'] for s in probe['streams']}:
            raise StageFailure('download', '输入必须同时包含视频和音频')
        duration = float(probe['format']['duration'])
        if not 0 < duration <= self.max_duration+1:
            raise StageFailure('download', f'视频时长必须介于 0 和 {self.max_duration} 秒之间')
        if artwork:
            create_poster(self.source, self.output/'poster.jpg', thumbnail=metadata.get('thumbnail'),
                          seconds=min(30, duration*.1))
        write_json(self.output/'source.json', dict(url=url, input_url=self.request['url'], sha256=file_hash(self.source),
            duration=duration, probe=probe, audio_only='video' not in {s['codec_type'] for s in probe['streams']}))
        return metadata, duration

    def promote_video(self, metadata, source):
        """Add this video's picture while preserving the already tested audio stream."""
        from .exact_stems import decode_source, pcm_sha
        folder = self.output/'video-download'
        video, info = folder/'source.mkv', folder/'source.info.json'
        with self.checkpoints.stage('download_video', inputs=[self.source], outputs=[video, info],
                config=dict(url=self.request['url'], format='bv[height<=1080]/b[height<=1080]')) as reused:
            if not reused:
                folder.mkdir(exist_ok=True)
                command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist',
                    '--write-info-json', '--no-overwrites', '--retries', '2', '--fragment-retries', '2',
                    '--socket-timeout', '25', '--max-filesize', '2G', '--match-filter',
                    f'!is_live & duration <= {self.max_duration}', '-f', 'bv[height<=1080]/b[height<=1080]',
                    '--remux-video', 'mkv', '-o', folder/'source.%(ext)s']
                if self.cookies_file:
                    command.extend(['--cookies', self.cookies_file])
                self.run('download_video', command+[self.request['url']], timeout=900)
        added = json.loads(info.read_text())
        if not metadata.get('id') or (added.get('id'), added.get('extractor_key')) != (
                metadata['id'], metadata.get('extractor_key')):
            raise StageFailure('verify_video', '补充的视频与已测试音频来源不同')
        merged = folder/'with-audio.mkv'
        with self.checkpoints.stage('prepare_video', inputs=[self.source, video, info],
                outputs=[merged], code=[ROOT/'pipeline.py']) as reused:
            if not reused:
                try:
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video),
                        '-i', str(self.source), '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', str(merged)], check=True)
                    original, rate, _ = decode_source(self.source)
                    promoted, new_rate, _ = decode_source(merged)
                    if rate != new_rate or pcm_sha(original) != pcm_sha(promoted):
                        raise StageFailure('verify_video', '补充视频改变了已测试音频，保留原音频及检查点')
                    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                        '-show_format', '-of', 'json', str(merged)]))
                    if abs(float(probe['format']['duration'])-source['duration']) > 1:
                        raise StageFailure('verify_video', '补充视频与已测试音频时长不匹配')
                except BaseException:
                    merged.unlink(missing_ok=True)
                    raise
        self.video = merged
        return dict(metadata, format_id=added['format_id']+'+'+metadata['format_id'])

    def prepare_text(self, metadata, duration):
        self.run('package_two_stems', [sys.executable, ROOT/'exact_stems.py', 'build',
            '--source', self.source, '--vocals-estimate', self.vocals, '--output', self.output/'two-stems'])
        self.run('resample_vocals', ['ffmpeg', '-v', 'error', '-nostdin', '-i', self.vocals,
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', self.output/'vocals-16k.wav',
            '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '64k', self.output/'vocals-asr.mp3'])
        self.run('transcribe', [sys.executable, ROOT/'transcribe_lyrics.py', '--audio',
            self.output/'vocals-asr.mp3', '--language', self.request.get('language') or 'auto',
            '--output', self.output/'asr'])
        self.phase('prepare_lyrics', True)
        started = time.perf_counter()
        try:
            catalog = list((self.catalog_root/'lyrics').glob('*.json')) if self.catalog_root else []
            with self.checkpoints.stage('prepare_lyrics',
                    inputs=[self.output/'asr/transcription.json', self.output/'vocals-16k.wav',
                            *sorted((self.output/'asr').glob('initial-transcription.json'))],
                    outputs=[self.output/'lyrics.json', self.output/'anchors.json', self.output/'asr/filtered.json',
                             self.output/'asr/anchor-recovery.json'],
                    config=dict(request=self.request,
                                metadata={k: metadata.get(k) for k in ('title', 'track', 'artist', 'artists', 'uploader', 'description')},
                                duration=duration, model=self.text_model,
                                implementation=inspect.getsource(type(self).prepare_lyrics)),
                    code=[ROOT/p for p in ('song_sources.py', 'anchors_from_asr.py', 'transcribe_lyrics.py', 'anchor_recovery.py',
                                           'subtitles.py', 'job_schema.py')] + catalog) as reused:
                if not reused:
                    self.prepare_lyrics(metadata, duration)
            lyrics = json.loads((self.output/'lyrics.json').read_text())
            self.phase(song_info=dict(title=lyrics['title'], artist=lyrics['artist'],
                                      duration_seconds=duration, language=lyrics['language']))
        finally:
            with self.lock:
                self.timings['prepare_lyrics'] = round(time.perf_counter()-started, 3)
            self.phase('prepare_lyrics', False)
        return lyrics

    def prepare_lyrics(self, metadata, duration):
        from anchors_from_asr import anchors_from_words
        from transcribe_lyrics import filter_silent_transcription
        from anchor_recovery import recover_anchor_words
        import soundfile as sf
        transcription = json.loads((self.output/'asr/transcription.json').read_text())
        language = song_language(transcription.get('language'))
        if language:
            self.phase(song_info=dict(language=language))
        audio, rate = sf.read(self.output/'vocals-16k.wav', dtype='float32')
        transcription, rejected = filter_silent_transcription(transcription, audio, rate)
        write_json(self.output/'asr/filtered.json', dict(transcription=transcription, rejected=rejected))
        lyrics = prepare_song_text(self.request, metadata, transcription, duration,
            catalog_root=self.catalog_root, model=self.text_model)
        candidates = {'transcription': transcription.get('words') or []}
        initial_path = self.output/'asr/initial-transcription.json'
        if lyrics['provenance']['kind'] != 'asr_draft' and initial_path.is_file():
            # Recovery of one bad passage need not discard usable whole-song
            # anchors. Only compare them against independently supplied lyrics.
            initial = json.loads(initial_path.read_text())
            if song_language(initial.get('language')) == lyrics['language']:
                initial, _ = filter_silent_transcription(initial, audio, rate)
                words = [w for w in initial.get('words') or [] if 0 <= w['start'] < w['end'] <= duration]
                if words and all(a['start'] <= b['start'] for a, b in zip(words, words[1:])):
                    candidates['initial_transcription'] = words
        matches = {}
        for name, words in candidates.items():
            cues = anchors_from_words(lyrics['lines'], words, lyrics['language'], allow_missing=True)
            coverage = sum(c['matched_fraction']*len(c['text']) for c in cues)/max(1, sum(len(c['text']) for c in cues))
            matches[name] = (cues, coverage)
        selected = max(matches, key=lambda name: matches[name][1])
        anchors, covered = matches[selected]
        recovery_path = self.output/'asr/anchor-recovery.json'
        write_json(recovery_path, dict(attempts=[]))
        if covered < .55 and lyrics['provenance']['kind'] != 'asr_draft':
            _, anchors, covered = recover_anchor_words(None,
                self.output/'vocals-16k.wav', lyrics['lines'], candidates[selected], lyrics['language'],
                recovery_path, self.checkpoints, .55, duration)
        write_json(self.output/'asr/filtered.json', dict(transcription=transcription, rejected=rejected,
            anchor_source=selected, anchor_coverage={name: match[1] for name, match in matches.items()},
            final_anchor_coverage=covered))
        write_json(self.output/'lyrics.json', lyrics)
        write_json(self.output/'anchors.json', anchors)
        if covered < .55 or not any(c['status'] == 'candidate' for c in anchors):
            raise StageFailure('prepare_lyrics', '歌词与这个视频的演唱匹配不足；请提交本版本的 lyrics 后重试')

    def align(self, lyrics, duration):
        from lyric_progress import build
        from subtitles import presentation_cues, countdown_events, normalized
        language = lyrics['language']
        output = self.output
        write_json(output/'alignment-jobs.json', [dict(audio=str(output/'vocals-16k.wav'),
            lyrics=str(output/'lyrics.json'), anchors=str(output/'anchors.json'), output=str(output/'alignment'))])
        self.run('align_words', [sys.executable, ROOT/'align_lyrics_whisper.py', '--jobs',
            output/'alignment-jobs.json', '--device', self.device])
        raw = json.loads((output/'alignment/raw.json').read_text())
        coarse, _ = presentation_cues(json.loads((output/'alignment/cues.json').read_text()),
                                     duration, lead=0, tail=0, bridge_gap=0)
        words, mapping = build(coarse, raw, language)
        if mapping['fallback_lines']:
            raise StageFailure('align_words', '部分歌词无法生成时间轴；请检查本版本歌词')
        write_json(output/'word-cues.json', words)
        order_fallbacks = []
        if language in ('Chinese', 'Japanese'):
            job = dict(song='song', language=language, audio=str(self.vocals),
                       cues=str(output/'word-cues.json'), output=str(output/'phonemes'))
            write_json(output/'phoneme-jobs.json', [job])
            self.run('align_phonemes', [sys.executable, ROOT/'align_lyric_phonemes.py',
                '--jobs', output/'phoneme-jobs.json', '--device', self.device])
            job.update(cues=str(output/'phonemes/cues.json'), coarse_cues=str(output/'word-cues.json'),
                       output=str(output/'repeated-lyrics'))
            write_json(output/'repeat-jobs.json', [job])
            self.run('align_repeats', [sys.executable, ROOT/'align_repeated_lyrics.py',
                '--jobs', output/'repeat-jobs.json', '--device', self.device])
            from align_lyric_phonemes import protect_line_order
            ordered, order_fallbacks = protect_line_order(
                json.loads((output/'repeated-lyrics/cues.json').read_text()), words)
            write_json(output/'ordered-cues.json', ordered)
            job.update(cues=str(output/'ordered-cues.json'), duration=duration,
                       output=str(output/'context/song'))
            write_json(output/'context-jobs.json', [job])
            self.run('align_context', [sys.executable, ROOT/'experiments/cjk-onsets-v003/run.py',
                '--jobs', output/'context-jobs.json', '--output', output/'context', '--device', self.device])
            cues = json.loads((output/'context/song/cues.json').read_text())
        else:
            from onset_consensus import blend_cues
            job = dict(audio=str(output/'vocals-16k.wav'), lyrics=str(output/'lyrics.json'),
                       anchors=str(output/'anchors.json'), output=str(output/'english-onsets'))
            write_json(output/'english-jobs.json', [job])
            self.run('align_phonemes', [sys.executable, ROOT/'align_english_onsets.py', '--method', 'singing',
                '--onset-ratio', '.02', '--full-song', '--jobs', output/'english-jobs.json', '--device', self.device])
            primary = json.loads((output/'english-onsets/cues.json').read_text())
            cues = blend_cues(primary, words, max_disagreement=.2, primary_weight=.75)
            cues = [dict(c, status='candidate') for c in cues]
            cues, _ = presentation_cues(cues, duration, lead=0, tail=0, bridge_gap=0)
        units = [u for c in cues for u in c['units']]
        if not units or [c['text'] for c in cues] != lyrics['lines']:
            raise StageFailure('align_words', '最终字幕没有完整覆盖输入歌词')
        if any(not 0 <= u['start'] < u['end'] <= duration+.05 for u in units) or any(
                a['start'] >= b['start'] for a, b in zip(units, units[1:])):
            raise StageFailure('align_words', '歌词时间轴仍有冲突；需要核对歌词与实际演唱')
        for cue in cues:
            if normalized(''.join(u['text'] for u in cue['units'])) != normalized(cue['text']):
                raise StageFailure('align_words', '逐字时间轴遗漏了歌词')
        write_json(output/'cues.json', cues)
        write_json(output/'quality.json', dict(lines=len(cues), units=len(units),
            lyrics_provenance=lyrics['provenance'], onset_accuracy=None, separation_purity=None,
            fallback_units=sum(u.get('timing_source') == 'prior_word_fallback' for u in units),
            line_order_fallback_lines=order_fallbacks,
            countdown_events=countdown_events(cues)))

    def execute(self):
        started = time.perf_counter()
        if not os.getenv('OPENAI_API_KEY'):
            raise StageFailure('configuration', '请在启动服务前设置 OPENAI_API_KEY，用于 ASR 和歌词整理')
        if not shutil.which(self.separator_python):
            raise StageFailure('configuration', '找不到分离环境的 Python，请设置 KARAOKE_SEPARATOR_PYTHON')
        import soundfile as sf
        import torch
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        metadata, source_duration = self.download()
        audio = self.output/'source-44100.wav'
        self.run('decode_audio', ['ffmpeg', '-v', 'error', '-nostdin', '-i', self.source,
            '-map', '0:a:0', '-ar', '44100', '-ac', '2', '-c:a', 'pcm_f32le', audio])
        self.inputs(audio, 'mixture-inputs.json')
        duration = sf.info(audio).duration
        common = [self.separator_python, '-u', ROOT/'benchmark_neural.py', '--device', self.device]
        self.run('separate_music', common + ['--model', 'vocals_mel_band_roformer.ckpt', '--config',
            ROOT/'models/pretrained/vocals_mel_band_roformer.yaml', '--inputs', self.output/'mixture-inputs.json',
            '--output', self.output/'music-separation'])
        self.inputs(self.vocals, 'vocal-inputs.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            text = pool.submit(self.prepare_text, metadata, source_duration)
            self.run('separate_lead', common + ['--model', 'karaoke_bs_roformer_anvuew.ckpt', '--config',
                ROOT/'models/pretrained/karaoke_bs_roformer_anvuew.yaml', '--target-stem', 'Vocals',
                '--inputs', self.output/'vocal-inputs.json', '--output', self.output/'lead-separation'])
            lyrics = text.result()
        with ThreadPoolExecutor(max_workers=2) as pool:
            packaging = pool.submit(self.run, 'package_three_stems', [sys.executable, ROOT/'three_stems.py',
                'build', '--existing', self.output/'two-stems', '--lead-estimate',
                self.output/'lead-separation/song/vocals.wav', '--output', self.output/'stems'])
            self.align(lyrics, duration)
            packaging.result()
        stems = json.loads((self.output/'stems/manifest.json').read_text())
        artifacts = dict(instrumental='stems/instrumental.wav',
            backing_vocals='stems/backing-vocals.wav', lead_vocals='stems/lead.wav',
            instrumental_preview='stems/instrumental.mp3', backing_vocals_preview='stems/backing-vocals.mp3',
            lead_vocals_preview='stems/lead.mp3', subtitles='cues.json')
        if not self.audio_only:
            artifacts['mv'] = 'mv.mp4'
        if (self.output/'poster.jpg').is_file():
            artifacts['poster'] = 'poster.jpg'
        result = dict(title=lyrics['title'], artist=lyrics['artist'], language=lyrics['language'],
            duration_seconds=duration, mv_audio=self.request['mv_audio'], shared_preview_gain=stems['preview_shared_gain'],
            source_format=metadata.get('format_id'), artifacts={key: dict(path=value) for key, value in artifacts.items()},
            quality=json.loads((self.output/'quality.json').read_text()),
            reconstruction=stems['reconstruction']['vocal_first'], device=self.device)
        if not self.audio_only:
            self.render_video(result)
        for artifact in result['artifacts'].values():
            path = self.output/artifact['path']
            artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))
        result.update(processing_seconds=round(time.perf_counter()-started, 3), timings=self.timings,
                      reused_stages=self.checkpoints.reused,
                      validation_scope='audio_and_lyrics' if self.audio_only else 'full_mv')
        write_json(self.output/('audio-result.json' if self.audio_only else 'result.json'), result)
        return result

    def render_video(self, result):
        paths = {name: self.output/a['path'] for name, a in result['artifacts'].items()}
        with self.checkpoints.stage('render',
                inputs=[self.video, paths['instrumental'], paths['backing_vocals'], paths['subtitles']],
                outputs=[paths['mv']],
                config={**{key: result[key] for key in ('title', 'artist', 'language', 'mv_audio',
                    'duration_seconds', 'shared_preview_gain')},
                    'implementation': inspect.getsource(type(self).render_uncached)},
                code=[ROOT/p for p in ('render_lyric_progress.py', 'render_subtitles.py', 'subtitles.py',
                                      'subtitle_style.py', 'subtitle-styles.json')]) as reused:
            if not reused:
                self.render_uncached(result)

    def render_uncached(self, result):
        import soundfile as sf
        paths = {name: self.output/a['path'] for name, a in result['artifacts'].items()}
        instrumental, rate = sf.read(paths['instrumental'], dtype='float32', always_2d=True)
        if result['mv_audio'] == 'instrumental_backing':
            backing, _ = sf.read(paths['backing_vocals'], dtype='float32', always_2d=True)
            instrumental += backing
        mixed = self.output/'render-mix.wav'
        sf.write(mixed, instrumental, rate, subtype='FLOAT')
        write_json(self.output/'song-info.json', dict(title=result['title'], original_artist=result['artist']))
        rendered = self.output/'rendered-mv.mp4'
        rendered.unlink(missing_ok=True)
        self.run('render', [sys.executable, ROOT/'render_lyric_progress.py', '--video', self.video,
            '--audio', mixed, '--cues', paths['subtitles'], '--song-info', self.output/'song-info.json',
            '--language', {'English': 'Latin'}.get(result['language'], result['language']),
            '--duration', result['duration_seconds'], '--gain', result['shared_preview_gain'],
            '--bottom-margin', '96', '--next-line', '--countdown', '--output', rendered])
        rendered.replace(paths['mv'])
        mixed.unlink()

    def restore_video(self):
        """Rebuild only the discarded video; never run a model or change cached audio/cues."""
        from .exact_stems import decode_source, pcm_sha
        started = time.perf_counter()
        result = json.loads((self.output/'result.json').read_text())
        if not can_restore_video(self.output, result):
            raise StageFailure('restore_video', '保留的音轨或字幕不完整，无法补回视频')
        self.download(artwork=False, format_id=result.get('source_format'))
        self.phase('verify_video', True)
        verified = time.perf_counter()
        try:
            audio, _, _ = decode_source(self.source)
            if pcm_sha(audio) != result['reconstruction']['source_pcm_sha256']:
                raise StageFailure('verify_video', '下载的视频音频与缓存版本不同，已保留原音轨和字幕；需重新完整准备该版本')
            del audio
        finally:
            self.timings['verify_video'] = round(time.perf_counter()-verified, 3)
            self.phase('verify_video', False)
        self.render_video(result)
        artifact = result['artifacts']['mv']
        path = self.output/artifact['path']
        artifact.update(bytes=path.stat().st_size, sha256=file_hash(path))
        result['video_restore'] = dict(processing_seconds=round(time.perf_counter()-started, 3), timings=self.timings)
        write_json(self.output/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore-video', action='store_true')
    parser.add_argument('--audio-only', action='store_true', help='Test all audio/subtitle stages without downloading or rendering video')
    args = parser.parse_args()
    pipeline = Pipeline(json.loads(args.request.read_text()), args.output,
        separator_python=os.getenv('KARAOKE_SEPARATOR_PYTHON'), device=os.getenv('KARAOKE_DEVICE', 'auto'),
        text_model=os.getenv('KARAOKE_TEXT_MODEL', 'gpt-5-mini'),
        max_duration=int(os.getenv('KARAOKE_MAX_DURATION_SECONDS', '900')),
        catalog_root=None if os.getenv('KARAOKE_DISABLE_LYRIC_CATALOG') == '1' else ROOT,
        cookies_file=os.getenv('KARAOKE_COOKIES_FILE'), audio_only=args.audio_only)
    try:
        pipeline.restore_video() if args.restore_video else pipeline.execute()
    except Exception as error:
        write_json(args.output/'error.json', processing_error(error, args.output))
        raise


if __name__ == '__main__':
    main()
