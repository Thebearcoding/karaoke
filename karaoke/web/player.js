// Schedule all stems on the render clock; visuals follow the device output clock.
import {createPitchShift} from './pitch-shift.js';
import {StemStream} from './stem-stream.js';

function videoReady(video, signal) {
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer);
      video.removeEventListener('loadeddata', check);
      video.removeEventListener('canplay', check);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
      error ? reject(error) : resolve();
    };
    const check = () => { if (video.readyState >= 2) finish(); };
    const failed = () => finish(new Error('视频加载失败，请重试'));
    const aborted = () => finish();
    const timer = setTimeout(() => finish(new Error('视频加载超时，请重试')), 30000);
    video.addEventListener('loadeddata', check);
    video.addEventListener('canplay', check);
    video.addEventListener('error', failed);
    signal.addEventListener('abort', aborted, {once:true});
    if (signal.aborted) aborted();
    else if (video.error) failed();
    else check();
  });
}

export class Player {
  constructor(video, onTick, onEnd, pitchFactory = createPitchShift) {
    this.video = video;
    this.onTick = onTick;
    this.onEnd = onEnd;
    this.sources = [];
    this.volume = 0.8;
    this.anchor = null;
    this.unlocked = false;
    this.mode = 'karaoke';
    this.levels = {instrumental:1, backing_vocals:1, lead_vocals:0};
    this.pitch = 0;
    this.pitchFactory = pitchFactory;
    const tick = () => {
      if (this.anchor !== null && this.context.state === 'running') {
        this.pump();
        if (this.anchor === null) { requestAnimationFrame(tick); return; }
        const outputTime = this.outputTime();
        const elapsed = outputTime - this.anchor;
        const position = Math.min(this.scheduledUntil,
          this.startPosition === 0 ? elapsed : Math.max(this.startPosition, elapsed));
        if (this.videoPending && outputTime >= this.audibleWhen) {
          this.videoPending = false;
          const signature = this.signature;
          this.video.play().catch(() => {
            if (this.signature !== signature) return;
            this.stop(); this.unlocked = false; this.onTick(position);
          });
        }
        this.syncVideo(Math.max(0, position));
        this.onTick(position);
        if (position >= this.entry.duration - 0.025) {
          const id = this.entry.id;
          this.stop();
          this.onEnd(id);
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  outputTime() {
    const context = this.context, stamp = context.getOutputTimestamp?.();
    if (Number.isFinite(stamp?.contextTime) && Number.isFinite(stamp?.performanceTime) && stamp.performanceTime > 0) {
      return Math.max(0, Math.min(context.currentTime,
        stamp.contextTime + (performance.now() - stamp.performanceTime) / 1000));
    }
    return Math.max(0, context.currentTime - (context.baseLatency || 0) - (context.outputLatency || 0));
  }

  syncVideo(position) {
    const video = this.video;
    if (this.videoPending || video.paused || video.seeking || video.readyState < 2) return;
    const drift = position - video.currentTime;
    // Let a seek finish and the decoder run before correcting again. Repeated
    // currentTime writes can keep a slow decoder permanently seeking.
    if (this.context.currentTime - this.lastVideoSeek < 1) return;
    if (Math.abs(drift) > .25) {
      video.playbackRate = 1;
      video.currentTime = Math.min(position, this.entry.duration);
      this.lastVideoSeek = this.context.currentTime;
    } else {
      video.playbackRate = Math.abs(drift) < .04 ? 1 : 1 + Math.max(-.1, Math.min(.1, drift));
    }
  }

  ensureContext() {
    if (this.context) return;
    this.context = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.context.createGain();
    this.master.gain.value = this.volume;
    this.master.connect(this.context.destination);
    this.gains = [0, 1, 2].map(() => {
      const gain = this.context.createGain();
      gain.connect(this.master);
      return gain;
    });
    this.setLevels(this.levels);
  }

  async unlock() {
    this.ensureContext();
    await this.context.resume();
    await this.preparePitch();
    this.unlocked = this.context.state === 'running';
  }

  preparePitch() {
    if (!this.pitchLoading) {
      this.pitchLoading = this.pitchFactory(this.context, this.master).then(chain => {
        this.pitchChain = chain;
        for (const gain of this.gains) { gain.disconnect(); gain.connect(chain.input); }
        chain.setPitch(this.pitch);
      }).catch(error => { this.pitchError = error; });
    }
    return this.pitchLoading;
  }

  setPitch(value) {
    this.pitch = Math.max(-6, Math.min(6, Math.round(value || 0)));
    if (this.pitch && this.pitchError) {
      this.stop();
      throw new Error(this.pitchError.message || '升降调初始化失败，请刷新播放页');
    }
    this.pitchChain?.setPitch(this.pitch);
  }

  setVolume(value) {
    this.volume = value;
    if (this.master) this.master.gain.setTargetAtTime(value, this.context.currentTime, 0.02);
  }

  setMix(mode) {
    this.mode = mode;
    this.setLevels({instrumental:1, backing_vocals:mode === 'instrumental' ? 0 : 1, lead_vocals:mode === 'original' ? 1 : 0});
  }

  setLevels(levels) {
    for (const name of ['instrumental', 'backing_vocals', 'lead_vocals']) {
      if (Number.isFinite(levels[name])) this.levels[name] = Math.min(1, Math.max(0, levels[name]));
    }
    if (!this.gains) return;
    const values = ['instrumental', 'backing_vocals', 'lead_vocals'].map(name => this.levels[name]);
    values.forEach((value, i) => this.gains[i].gain.setTargetAtTime(value, this.context.currentTime, 0.02));
  }

  stop() {
    const wasPlaying = this.anchor !== null;
    for (const source of this.sources) {
      try { source.stop(); } catch { /* Already ended. */ }
      source.disconnect();
    }
    this.sources = [];
    this.anchor = null;
    this.signature = null;
    this.videoPending = false;
    this.video.playbackRate = 1;
    if (wasPlaying) this.pitchChain?.stop();
    this.video.pause();
  }

  clear() {
    this.stop();
    this.abort?.abort();
    this.key = null;
    this.entry = null;
    this.stream = null;
    this.ready = false;
    this.video.removeAttribute('src');
    this.video.load();
  }

  matches(entry) {
    return Boolean(entry && this.key === entry.id && this.entry?.job_id === entry.job_id &&
      ['mv', 'instrumental', 'backing_vocals', 'lead_vocals'].every(
        name => this.entry.artifacts[name] === entry.artifacts[name]));
  }

  async load(entry, position = 0) {
    const generation = this.loadGeneration = (this.loadGeneration || 0) + 1;
    if (this.matches(entry)) {
      const stream = this.stream;
      try {
        await this.loading;
        if (!stream.signal.aborted) await stream.prepare(position);
      } catch (error) {
        if (stream.signal.aborted) return;
        if (generation === this.loadGeneration) this.key = null;
        throw error;
      }
      return;
    }
    this.stop();
    this.abort?.abort();
    this.abort = new AbortController();
    const {signal} = this.abort;
    this.key = entry.id;
    this.entry = entry;
    this.ready = false;
    this.ensureContext();
    this.video.preload = 'auto';
    this.video.src = entry.artifacts.mv;
    this.video.load();
    this.stream = new StemStream(this.context, `/v1/jobs/${entry.job_id}/playback`, signal, () => this.pump());
    this.loading = Promise.all([this.stream.prepare(position), videoReady(this.video, signal)]).then(() => {
      if (signal.aborted) return;
      this.ready = true;
    }).catch(error => {
      if (signal.aborted) return;
      this.key = null;
      throw error;
    });
    return this.loading;
  }

  apply(state, clockOffset) {
    this.setPitch(state.pitch_semitones || 0);
    if (state.levels) this.setLevels(state.levels);
    else this.setMix(state.mix);
    if (!this.ready || this.entry?.id !== state.current_id) return;
    const position = Math.min(this.entry.duration, state.position +
      (state.playing && state.started_at ? Math.max(0, Date.now()/1000 + clockOffset - state.started_at) : 0));
    if (!state.playing || !state.started_at || !this.unlocked) {
      this.stop();
      // An unopened playback tab may receive another device's playing clock.
      // Keep its preview still instead of seeking on every preparation event.
      const preview = this.unlocked ? position : state.position;
      if (Math.abs(this.video.currentTime - preview) > .04) this.video.currentTime = preview;
      this.onTick(position);
      return;
    }
    const signature = [state.current_id, state.position, state.started_at].join(':');
    if (signature === this.signature) return;
    this.stop();
    if (!this.hasAudio(position)) { this.onBuffering?.(position, state.started_at); return; }
    this.signature = signature;
    this.startedAt = state.started_at;
    const delay = Math.max(0, state.started_at - (Date.now()/1000 + clockOffset));
    this.audibleWhen = this.context.currentTime + Math.max(delay, 0.02 + (this.pitchChain?.latency || 0));
    this.startPosition = position;
    this.anchor = this.audibleWhen - position;
    this.nextWindow = this.stream.index(position);
    this.scheduledUntil = position;
    this.pump();
    this.pitchChain?.start(this.audibleWhen);
    this.video.currentTime = position;
    this.lastVideoSeek = this.context.currentTime;
    this.videoPending = true;
  }

  hasAudio(position) { return this.ready && Boolean(this.stream?.at(position)); }

  pump() {
    if (this.anchor === null || !this.ready) return;
    const stream = this.stream;
    const position = Math.max(this.startPosition, this.outputTime() - this.anchor);
    const inputPosition = Math.max(this.startPosition,
      this.context.currentTime - this.anchor + (this.pitchChain?.latency || 0));
    stream.around(inputPosition);
    while (this.nextWindow < stream.count) {
      const audio = stream.windows.get(this.nextWindow)?.audio;
      if (!audio) break;
      const {start, end, decodedStart} = stream.bounds(this.nextWindow++);
      const first = Math.max(start, this.startPosition);
      const when = this.anchor - (this.pitchChain?.latency || 0) + first;
      // Never start a late window partway through a song. Freeze the shared
      // room clock below and resume all stems from the same buffered position.
      if (when < this.context.currentTime - .005) { this.nextWindow--; break; }
      for (const [i, buffer] of audio.entries()) {
        const source = this.context.createBufferSource();
        source.buffer = buffer;
        source.connect(this.gains[i]);
        source.onended = () => {
          source.disconnect();
          this.sources = this.sources.filter(item => item !== source);
          if (stream === this.stream) this.pump();
        };
        source.start(when, first - decodedStart, end - first);
        this.sources.push(source);
      }
      this.scheduledUntil = end;
    }
    if (position >= this.scheduledUntil && this.scheduledUntil < stream.duration) {
      const pausedAt = this.scheduledUntil, startedAt = this.startedAt;
      this.stop();
      this.onTick(pausedAt);
      this.onBuffering?.(pausedAt, startedAt);
    }
  }
}
