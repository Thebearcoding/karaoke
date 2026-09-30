"""URL-to-karaoke adapter for the existing, versioned separation/alignment stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
# The research stages predate the package API and use sibling imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .job_schema import JobRequest, video_url
from .song_sources import prepare_song_text, video_song_info


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


def download_failure(log):
    """Expose actionable causes without publishing raw URLs, headers or credentials."""
    diagnostic = log.lower()
    if any(text in diagnostic for text in ('unable to connect to proxy', 'proxyerror', 'tunnel connection failed')):
        return '下载被网络代理拒绝，请检查服务的代理设置后重试'
    if any(text in diagnostic for text in ('sign in', 'login required', 'login_required')):
        return '视频站点要求登录验证，请配置下载登录信息后重试'
    if any(text in diagnostic for text in ('video unavailable', 'private video', 'video is unavailable')):
        return '视频当前无法访问，请换一个可公开播放的链接'
    if any(text in diagnostic for text in ('timed out', 'connection reset', 'temporary failure in name resolution')):
        return '连接视频站点超时或中断，请稍后重试'
    return '视频下载失败，请稍后重试或换一个视频链接'


class Pipeline:
    def __init__(self, request, output, *, separator_python=None, device='auto', text_model='gpt-5-mini',
                 max_duration=900, catalog_root=ROOT, cookies_file=None):
        self.request = JobRequest.model_validate(request).model_dump()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        default_separator = ROOT.parent.parent/'audio/.venv-separator-local/bin/python'
        self.separator_python = separator_python or (str(default_separator) if default_separator.exists() else sys.executable)
        self.device, self.text_model = device, text_model
        self.max_duration, self.catalog_root, self.cookies_file = max_duration, catalog_root, cookies_file
        self.active, self.timings, self.lock = set(), {}, threading.Lock()
        self.song_info = {}
        self.source = self.output/'download/source.mkv'
        self.vocals = self.output/'music-separation/song/vocals.wav'

    def phase(self, name=None, active=False, *, song_info=None):
        with self.lock:
            if name is not None:
                self.active.add(name) if active else self.active.discard(name)
            if song_info is not None:
                self.song_info = song_info
            progress = dict(active_stages=sorted(self.active),
                updated_at=datetime.now(timezone.utc).isoformat(), timings=self.timings,
                song_info=self.song_info)
            write_json(self.output/'progress.json', progress)
            if os.getenv('KARAOKE_EVENTS_STDOUT') == '1':
                print('KARAOKE_PROGRESS ' + json.dumps(progress), flush=True)

    def run(self, name, command, timeout=3600):
        self.phase(name, True)
        started = time.perf_counter()
        try:
            with (self.output/f'{name}.log').open('w') as log:
                result = subprocess.run([str(a) for a in command], cwd=ROOT.parent,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if result.returncode:
                with (self.output/f'{name}.log').open('rb') as log:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell()-16384))
                    diagnostic = log.read().decode(errors='replace')
                if name == 'download':
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

    def download(self):
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
        command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--no-progress',
            '--write-info-json', '--no-overwrites', '--retries', '2', '--fragment-retries', '2',
            '--socket-timeout', '25', '--max-filesize', '2G', '--match-filter',
            f'!is_live & duration <= {self.max_duration}', '-f', 'bv[height<=1080]+ba/b[height<=1080]',
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
        if not {'audio', 'video'} <= {s['codec_type'] for s in probe['streams']}:
            raise StageFailure('download', '输入必须同时包含视频和音频')
        duration = float(probe['format']['duration'])
        if not 0 < duration <= self.max_duration+1:
            raise StageFailure('download', f'视频时长必须介于 0 和 {self.max_duration} 秒之间')
        write_json(self.output/'source.json', dict(url=url, sha256=file_hash(self.source),
                                                 duration=duration, probe=probe))
        return metadata, duration

    def prepare_text(self, metadata, duration):
        from anchors_from_asr import anchors_from_words
        from transcribe_lyrics import filter_silent_transcription
        import soundfile as sf
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
            transcription = json.loads((self.output/'asr/transcription.json').read_text())
            audio, rate = sf.read(self.output/'vocals-16k.wav', dtype='float32')
            transcription, rejected = filter_silent_transcription(transcription, audio, rate)
            write_json(self.output/'asr/filtered.json', dict(transcription=transcription, rejected=rejected))
            lyrics = prepare_song_text(self.request, metadata, transcription, duration,
                catalog_root=self.catalog_root, model=self.text_model)
            self.phase(song_info=dict(title=lyrics['title'], artist=lyrics['artist'],
                                      duration_seconds=duration))
            anchors = anchors_from_words(lyrics['lines'], transcription.get('words') or [],
                                         lyrics['language'], allow_missing=True)
            covered = sum(c['matched_fraction']*len(c['text']) for c in anchors)/max(1, sum(len(c['text']) for c in anchors))
            if covered < .55 or not any(c['status'] == 'candidate' for c in anchors):
                raise StageFailure('prepare_lyrics', '歌词与这个视频的演唱匹配不足；请提交本版本的 lyrics 后重试')
            write_json(self.output/'lyrics.json', lyrics)
            write_json(self.output/'anchors.json', anchors)
            write_json(self.output/'song-info.json', dict(title=lyrics['title'], original_artist=lyrics['artist']))
        finally:
            with self.lock:
                self.timings['prepare_lyrics'] = round(time.perf_counter()-started, 3)
            self.phase('prepare_lyrics', False)
        return lyrics

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
        gain = stems['preview_shared_gain']
        instrumental, rate = sf.read(self.output/'stems/instrumental.wav', dtype='float32', always_2d=True)
        if self.request['mv_audio'] == 'instrumental_backing':
            backing, _ = sf.read(self.output/'stems/backing-vocals.wav', dtype='float32', always_2d=True)
            instrumental += backing
        mixed = self.output/'render-mix.wav'
        sf.write(mixed, instrumental, rate, subtype='FLOAT')
        self.run('render', [sys.executable, ROOT/'render_lyric_progress.py', '--video', self.source,
            '--audio', mixed, '--cues', self.output/'cues.json', '--song-info', self.output/'song-info.json',
            '--language', {'English': 'Latin'}.get(lyrics['language'], lyrics['language']),
            '--duration', duration, '--gain', gain, '--bottom-margin', '96', '--next-line', '--countdown',
            '--output', self.output/'mv.mp4'])
        mixed.unlink()
        artifacts = dict(mv='mv.mp4', instrumental='stems/instrumental.wav',
            backing_vocals='stems/backing-vocals.wav', lead_vocals='stems/lead.wav',
            instrumental_preview='stems/instrumental.mp3', backing_vocals_preview='stems/backing-vocals.mp3',
            lead_vocals_preview='stems/lead.mp3', subtitles='cues.json')
        result = dict(title=lyrics['title'], artist=lyrics['artist'], language=lyrics['language'],
            duration_seconds=duration, mv_audio=self.request['mv_audio'], shared_preview_gain=gain,
            artifacts={key: dict(path=value, bytes=(self.output/value).stat().st_size,
                       sha256=file_hash(self.output/value)) for key, value in artifacts.items()},
            quality=json.loads((self.output/'quality.json').read_text()),
            reconstruction=stems['reconstruction']['vocal_first'], device=self.device,
            processing_seconds=round(time.perf_counter()-started, 3), timings=self.timings)
        write_json(self.output/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline = Pipeline(json.loads(args.request.read_text()), args.output,
        separator_python=os.getenv('KARAOKE_SEPARATOR_PYTHON'), device=os.getenv('KARAOKE_DEVICE', 'auto'),
        text_model=os.getenv('KARAOKE_TEXT_MODEL', 'gpt-5-mini'),
        max_duration=int(os.getenv('KARAOKE_MAX_DURATION_SECONDS', '900')),
        catalog_root=None if os.getenv('KARAOKE_DISABLE_LYRIC_CATALOG') == '1' else ROOT,
        cookies_file=os.getenv('KARAOKE_COOKIES_FILE'))
    try:
        pipeline.execute()
    except Exception as error:
        write_json(args.output/'error.json', dict(code=getattr(error, 'stage', 'processing')+'_failed',
            message=str(error)[:600] if isinstance(error, (StageFailure, ValueError)) else
            '处理失败；请查看任务目录中的日志，或提供本版本歌词后重试'))
        raise


if __name__ == '__main__':
    main()
