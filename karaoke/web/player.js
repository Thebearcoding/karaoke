// All three stems share one AudioContext clock. Video follows that clock.
export class Player {
  constructor(video, onTick, onEnd) {
    this.video = video;
    this.onTick = onTick;
    this.onEnd = onEnd;
    this.sources = [];
    this.volume = 0.8;
    this.anchor = null;
    this.unlocked = false;
    this.mode = 'karaoke';
    this.levels = {instrumental:1, backing_vocals:1, lead_vocals:0};
    const tick = () => {
      if (this.anchor !== null && this.context.state === 'running') {
        const position = Math.max(0, this.context.currentTime - this.anchor);
        if (this.video.readyState >= 2 && Math.abs(this.video.currentTime - position) > 0.08) {
          this.video.currentTime = Math.min(position, this.entry.duration);
        }
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
    this.unlocked = this.context.state === 'running';
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
    for (const source of this.sources) {
      try { source.stop(); } catch { /* Already ended. */ }
      source.disconnect();
    }
    this.sources = [];
    this.anchor = null;
    this.signature = null;
    this.video.pause();
  }

  clear() {
    this.stop();
    this.abort?.abort();
    this.key = null;
    this.entry = null;
    this.buffers = null;
    this.video.removeAttribute('src');
    this.video.load();
  }

  load(entry) {
    if (this.key === entry.id) return this.loading;
    this.stop();
    this.abort?.abort();
    this.abort = new AbortController();
    const {signal} = this.abort;
    this.key = entry.id;
    this.entry = entry;
    this.buffers = null;
    this.ensureContext();
    this.video.src = entry.artifacts.mv;
    this.video.load();
    this.loading = Promise.all(['instrumental_preview', 'backing_vocals_preview', 'lead_vocals_preview'].map(async key => {
      const url = entry.artifacts[key];
      if (!url) throw new Error('这首歌的试听音轨不完整，请重新准备');
      const response = await fetch(url, {signal});
      if (!response.ok) throw new Error('音轨加载失败，请重试');
      return this.context.decodeAudioData(await response.arrayBuffer());
    })).then(buffers => {
      if (signal.aborted) return;
      this.buffers = buffers;
    }).catch(error => {
      if (signal.aborted) return;
      this.key = null;
      throw error;
    });
    return this.loading;
  }

  apply(state, clockOffset) {
    if (state.levels) this.setLevels(state.levels);
    else this.setMix(state.mix);
    if (!this.buffers || this.entry?.id !== state.current_id) return;
    const position = Math.min(this.entry.duration, state.position +
      (state.playing && state.started_at ? Math.max(0, Date.now()/1000 + clockOffset - state.started_at) : 0));
    if (!state.playing || !state.started_at || !this.unlocked) {
      this.stop();
      this.video.currentTime = position;
      this.onTick(position);
      return;
    }
    const signature = [state.current_id, state.position, state.started_at].join(':');
    if (signature === this.signature) return;
    this.stop();
    this.signature = signature;
    const when = this.context.currentTime + 0.02;
    this.sources = this.buffers.map((buffer, i) => {
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.gains[i]);
      source.start(when, Math.min(position, buffer.duration));
      return source;
    });
    this.anchor = when - position;
    this.video.currentTime = position;
    this.video.play().catch(() => {
      this.stop();
      this.unlocked = false;
      this.onTick(position);
    });
  }
}
