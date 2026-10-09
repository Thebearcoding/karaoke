// Use one AudioContext clock for all stems and latency-adjusted visuals.
import {createPitchShift} from './pitch-shift.js';
import {StemStream} from './stem-stream.js';

function videoReady(video, signal, position) {
  return new Promise((resolve, reject) => {
    const events = ['loadedmetadata', 'loadeddata', 'canplay', 'seeked'];
    let positioned = false;
    const finish = error => {
      clearTimeout(timer);
      events.forEach(name => video.removeEventListener(name, check));
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
      error ? reject(error) : resolve();
    };
    const check = () => {
      if (video.readyState >= 1 && !positioned) {
        positioned = true;
        if (Math.abs(video.currentTime - position) > .04) video.currentTime = position;
      }
      if (video.readyState >= 2 && !video.seeking) finish();
    };
    const failed = () => finish(new Error('视频加载失败，请重试'));
    const aborted = () => finish();
    const timer = setTimeout(() => finish(new Error('视频加载超时，请重试')), 30000);
    events.forEach(name => video.addEventListener(name, check));
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
    video.muted = true;
    video.playsInline = true;
    this.onTick = onTick;
    this.onEnd = onEnd;
    this.sources = [];
    this.volume = 1;
    this.anchor = null;
    this.unlocked = false;
    this.mode = 'karaoke';
    this.levels = {instrumental:1, backing_vocals:1, lead_vocals:0};
    this.pitch = 0;
    this.pitchFactory = pitchFactory;
    const tick = () => {
      if (this.disposed) return;
      if (this.anchor !== null && this.context.state === 'running') {
        if (this.pendingStart) {
          const pending = this.pendingStart;
          this.onTick(pending.position === 0 ? Math.min(0, this.outputTime() - this.anchor) : pending.position);
          requestAnimationFrame(tick); return;
        }
        this.pump();
        if (this.anchor === null) { requestAnimationFrame(tick); return; }
        const outputTime = this.outputTime();
        const elapsed = outputTime - this.anchor;
        const position = Math.min(this.scheduledUntil,
          this.startPosition === 0 ? elapsed : Math.max(this.startPosition, elapsed));
        if (this.videoPending && outputTime >= this.audibleWhen) {
          this.videoPending = false;
          this.lastVideoTime = this.video.currentTime;
          this.lastVideoAdvance = this.context.currentTime;
          const run = this.videoRun;
          this.video.play().catch(error => {
            if (this.videoRun === run) this.failPlayback(error, position);
          });
        }
        if (position >= this.entry.duration - 0.025) {
          const id = this.entry.id;
          this.stop();
          this.onEnd(id);
        } else {
          this.syncVideo(Math.max(0, position));
          if (this.anchor !== null) this.onTick(position);
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  outputTime() {
    // Keep visuals on the clock that actually schedules audio. Some WebKit
    // devices report a fresh getOutputTimestamp() with a nearly frozen time,
    // which would leave the video waiting forever while audio keeps playing.
    const context = this.context;
    return Math.max(0, context.currentTime - (context.baseLatency || 0) - (context.outputLatency || 0));
  }

  syncVideo(position) {
    const video = this.video;
    if (this.videoPending) return;
    if (video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = video.currentTime;
      this.lastVideoAdvance = this.context.currentTime;
    }
    // readyState can briefly drop while frames still advance. Freeze only for
    // an actual pause/stall, not a momentary change in the decoder's buffer.
    if (video.paused || this.context.currentTime - this.lastVideoAdvance > 1) {
      this.waitForMedia(position);
      return;
    }
    if (video.seeking) return;
    const drift = position - video.currentTime;
    // Never chase a moving audio target with seeks. Stop the common clock and
    // prepare one fixed position if the video needs to load or catch up.
    if (Math.abs(drift) > 1) {
      this.waitForMedia(position);
    }
    // Leave normal playback at native speed. On WebKit, changing rate stalls
    // the decoder and shifts its clock; correcting that shift repeats the
    // stall. Small timestamp differences must not create playback interruptions.
  }

  waitForMedia(position) {
    const startedAt = this.startedAt;
    this.stop();
    this.onTick(position);
    this.onBuffering?.(position, startedAt);
  }

  failPlayback(error, position) {
    this.unlocked = false;
    this.waitForMedia(position);
    this.onError?.(error);
  }

  prepareVideo(position, signal = this.abort.signal) {
    const existing = this.videoPreparation;
    if (existing?.signal === signal && Math.abs(existing.position - position) < .04) return existing.promise;
    const preparation = this.videoPreparation = {position, signal};
    preparation.promise = this.primeVideo(position, signal).finally(() => {
      if (this.videoPreparation === preparation) this.videoPreparation = null;
    });
    return preparation.promise;
  }

  async primeVideo(position, signal) {
    await videoReady(this.video, signal, position);
    if (signal.aborted || (this.videoPrepared != null && this.video.paused &&
        Math.abs(this.videoPrepared - position) < .04)) return;
    let timer;
    try {
      await Promise.race([this.video.play(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('视频启动超时，请重试')), 15000);
      })]);
      if (signal.aborted) {
        if (!this.signature) this.video.pause();
        return;
      }
      if (this.video.paused) throw new Error('视频未能开始播放，请重试');
      this.video.pause();
      await videoReady(this.video, signal, position);
      if (!signal.aborted) this.videoPrepared = position;
    } finally { clearTimeout(timer); }
  }

  async startVideo(pending) {
    try {
      // Normal startup primes before the room's ready command. Also protect
      // direct seek/resume callers before scheduling any audio.
      await this.prepareVideo(pending.position, pending.abort.signal);
      if (this.pendingStart !== pending) return;
      // Mobile browsers may suspend an idle context while the next song loads.
      // Reuse the user's existing play permission before scheduling that song.
      if (this.unlocked && this.context.state !== 'running') await this.context.resume();
      if (this.pendingStart !== pending) return;
      if (this.context.state !== 'running') throw new Error('声音尚未启动，请点播放重试');
      this.pendingStart = null;
      this.videoRun = pending;
      this.videoPending = true;
      this.audibleWhen = this.context.currentTime + Math.max(pending.when - this.outputTime(),
        .02 + (this.pitchChain?.latency || 0));
      this.anchor = this.audibleWhen - pending.position;
      this.lastVideoTime = this.video.currentTime;
      this.lastVideoAdvance = this.context.currentTime;
      this.pump();
      this.pitchChain?.start(this.audibleWhen);
    } catch (error) {
      if (this.pendingStart !== pending) return;
      this.failPlayback(error, pending.position);
    }
  }

  ensureContext() {
    if (this.context) return;
    this.context = new (window.AudioContext || window.webkitAudioContext)();
    this.context.onstatechange = () => {
      if (this.context.state !== 'running' && this.anchor !== null) {
        const position = Math.max(0, Math.min(this.scheduledUntil, this.outputTime() - this.anchor));
        this.failPlayback(new Error('声音已中断，请点播放继续'), position);
      }
    };
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
    // iOS otherwise treats Web Audio as ambient audio, which the ringer switch mutes.
    try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { /* Optional API. */ }
    this.ensureContext();
    this.unlocked = false;
    await this.context.resume();
    await this.preparePitch();
    this.unlocked = this.context.state === 'running';
    if (!this.unlocked) throw new Error('声音尚未启动，请点播放重试');
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
    this.pendingStart?.abort.abort();
    this.pendingStart = null;
    this.videoRun = null;
    this.videoPending = false;
    if (this.video.playbackRate !== 1) this.video.playbackRate = 1;
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

  dispose() {
    this.disposed = true;
    this.clear();
    this.context?.close();
  }

  matches(entry) {
    return Boolean(entry && this.key === entry.id && this.entry?.job_id === entry.job_id &&
      ['mv', 'accompaniment', 'instrumental', 'backing_vocals', 'lead_vocals'].every(
        name => this.entry.artifacts[name] === entry.artifacts[name]));
  }

  async load(entry, position = 0) {
    const generation = this.loadGeneration = (this.loadGeneration || 0) + 1;
    if (this.matches(entry)) {
      const stream = this.stream;
      try {
        await this.loading;
        if (!stream.signal.aborted) await Promise.all([stream.prepare(position),
          this.anchor === null ? videoReady(this.video, stream.signal, position) : Promise.resolve()]);
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
    this.videoPrepared = null;
    this.ensureContext();
    this.video.preload = 'auto';
    this.video.src = entry.artifacts.mv;
    this.video.load();
    this.stream = new StemStream(this.context, `/v1/jobs/${entry.job_id}/playback`, signal, () => this.pump());
    this.loading = Promise.all([this.stream.prepare(position), videoReady(this.video, signal, position)]).then(() => {
      if (signal.aborted) return;
      this.ready = true;
    }).catch(error => {
      if (signal.aborted) return;
      this.key = null;
      this.abort.abort();
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
    this.audibleWhen = this.outputTime() + delay;
    this.startPosition = position;
    this.anchor = this.audibleWhen - position;
    this.nextWindow = this.stream.index(position);
    this.scheduledUntil = position;
    const pending = this.pendingStart = {position, when:this.audibleWhen, abort:new AbortController()};
    return this.startVideo(pending);
  }

  hasAudio(position) { return this.ready && Boolean(this.stream?.at(position)); }

  pump() {
    if (this.anchor === null || !this.ready || this.pendingStart) return;
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
        const name = stream.stems[i];
        source.connect(this.gains[name === 'lead_vocals' ? 2 : name === 'backing_vocals' ? 1 : 0]);
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
