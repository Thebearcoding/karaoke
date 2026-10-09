// Room revisions order commands; progress and cache updates can share a revision.
export function isStaleRoomSnapshot(incoming, current) {
  return incoming.revision < current.revision ||
    (incoming.revision === current.revision && incoming.server_time < current.server_time);
}

export function queueToTop(entries, id, playingId = null) {
  const ids = entries.map(entry => entry.id);
  if (!ids.includes(id) || id === playingId) return null;
  const first = ids.includes(playingId) ? [playingId, id] : [id];
  const order = [...first, ...ids.filter(item => !first.includes(item))];
  return order.some((item, index) => item !== ids[index]) ? order : null;
}

export function preparationRetry(entry) {
  if (entry?.status === 'failed' || (entry?.status === 'queued' && entry.next_retry_at))
    return {action:'retry', label:'重试'};
  return null;
}

const bounded = (value, min, max, fallback) => Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;

// One tab owns audio. Saved progress from an older owner cannot overwrite a
// newer tab's commands; storage is scoped to this browser, never the room.
export class PlaybackTabs {
  constructor(storage, id, fallback = {}) {
    this.storage = storage; this.id = id; this.key = 'karaokeBrowserPlayback';
    this.sequence = 0;
    this.record = {revision:0, change:null, session:null, owner:null, playback:fallback};
    this.read();
  }
  read() {
    try {
      const saved = JSON.parse(this.storage?.getItem(this.key) || 'null');
      // Storage, not a locally incremented number, decides the last writer.
      // Two tabs can legitimately write the same revision after a concurrent read.
      if (saved && Number.isSafeInteger(saved.revision) && saved.playback)
        this.record = {...saved, change:saved.change || `${saved.owner}:${saved.revision}`,
          session:saved.session || saved.owner};
    } catch { /* Private storage restrictions must not disable local playback. */ }
    return this.record;
  }
  owns(session) {
    const current = this.read();
    return current.owner === this.id && (session === undefined || current.session === session);
  }
  write(playback, {claim = false, release = false} = {}) {
    const previous = this.record.change, current = this.read();
    if (!claim && (current.owner !== this.id || current.change !== previous)) return false;
    const change = `${this.id}:${++this.sequence}`;
    this.record = {revision:current.revision+1, change, session:claim ? change : current.session,
      owner:release ? null : this.id, playback};
    try { this.storage?.setItem(this.key, JSON.stringify(this.record)); } catch {}
    return this.read().change === change;
  }
}

// Only the playlist and preparation results belong to the room. Transport is
// private to this browser. Only an already-running performance may outlive a
// queue removal on another device; paused/restored selections must be current.
export class LocalPlayback {
  constructor(saved = {}) {
    this.entry = saved?.entry?.id && saved.entry.job_id ? saved.entry : null;
    this.state = {playing:saved?.playing === true, position:bounded(saved?.position, 0, Infinity, 0), started_at:null,
      levels:{instrumental:1, backing_vocals:1, lead_vocals:bounded(saved?.levels?.lead_vocals, 0, 1, 0)},
      pitch_semitones:Math.round(bounded(saved?.pitch_semitones, -6, 6, 0))};
    this.playedAt = saved?.played_at || null;
    this.leadRestore = this.state.levels.lead_vocals || bounded(saved?.lead_restore, 0, 1, 1) || 1;
    this.pendingFinishes = saved?.pending_finishes || (saved?.finished_ids || []).map(entry_id =>
      ({entry_id, playback_started_at:null}));
  }

  reconcile(room) {
    const first = room.entries.find(entry => !this.isFinished(entry.id)) || null;
    // An idle preselection follows the queue. Pause, buffering and a saved seek
    // position do not end an existing performance.
    const idle = !this.state.playing && this.playedAt === null && this.state.position === 0;
    if (!this.entry || (idle && this.entry.id !== first?.id)) this.select(first, this.state.playing);
    else {
      const queued = room.entries.find(entry => entry.id === this.entry.id);
      const watched = room.playback_entry?.job_id === this.entry.job_id ? room.playback_entry : null;
      const updated = queued || watched || room.library?.find(entry => entry.job_id === this.entry.job_id);
      if (!queued && (!this.state.playing || this.state.started_at === null))
        this.select(first, this.state.playing);
      else if (updated) this.entry = {...updated, id:this.entry.id};
    }
    if (this.state.playing && this.entry?.status !== 'completed') {
      const ready = room.entries.find(entry => entry.status === 'completed' && !this.isFinished(entry.id));
      if (ready) this.select(ready, true);
    }
    return {...room, ...this.state, current_id:this.entry?.id || null};
  }

  select(entry, playing) {
    this.entry = entry;
    // Playing is this browser's intent. An empty queue waits without a clock;
    // finishing the last song must not discard the request to keep singing.
    Object.assign(this.state, {position:0, playing, started_at:null, pitch_semitones:0});
    this.playedAt = null;
  }

  position() {
    const {position, playing, started_at} = this.state;
    return position + (playing && started_at !== null ? Math.max(0, Date.now()/1000 - started_at) : 0);
  }

  markPlayed() {
    if (this.state.playing && this.state.started_at !== null && Date.now()/1000 > this.state.started_at)
      this.playedAt ||= this.state.started_at;
  }

  isFinished(id) { return this.pendingFinishes.some(item => item.entry_id === id); }

  // Entry IDs make finish idempotent on the server. Persist the original command
  // until acknowledged, including across refresh or an audio-owner handoff.
  async flushFinishes(send, accept, owns = () => true) {
    if (this.flushing) return this.flushing;
    this.flushing = (async () => {
      while (this.pendingFinishes.length && owns()) {
        const finished = this.pendingFinishes[0];
        const updated = await send({action:'finish', ...finished});
        if (!owns()) return;
        this.pendingFinishes = this.pendingFinishes.filter(item => item !== finished);
        accept(updated);
      }
    })();
    try { await this.flushing; } finally { this.flushing = null; }
  }

  command(action, extra, entries, clockOffset = 0) {
    const state = this.state;
    if (extra.entry_id && !['lead','toggle_lead','pitch'].includes(action) && extra.entry_id !== this.entry?.id) return {};
    this.markPlayed();
    let finished;
    if (['next','remove'].includes(action)) {
      const target = entries.find(entry => entry.id !== this.entry?.id && !this.isFinished(entry.id));
      if (this.entry) finished = {entry_id:this.entry.id,
        playback_started_at:this.playedAt === null ? null : this.playedAt + clockOffset};
      if (finished) this.pendingFinishes.push(finished);
      this.select(target || null, action === 'remove' ? state.playing : true);
    } else if (action === 'lead' || action === 'toggle_lead') {
      const level = action === 'toggle_lead' ? (state.levels.lead_vocals > 0 ? 0 : this.leadRestore) : extra.level;
      state.levels.lead_vocals = bounded(level, 0, 1, state.levels.lead_vocals);
      if (state.levels.lead_vocals > 0) this.leadRestore = state.levels.lead_vocals;
    } else if (action === 'pitch') {
      state.pitch_semitones = extra.pitch_semitones;
    } else if (action === 'play') state.playing = true;
    else if (action === 'pause') Object.assign(state, {position:this.position(), playing:false, started_at:null});
    else if (this.entry) {
      if (action === 'restart') Object.assign(state, {position:0, started_at:null, pitch_semitones:0});
      else if (action === 'seek') Object.assign(state, {position:bounded(extra.position, 0, Math.max(0, (this.entry.duration || 0)-.05), 0), started_at:null});
      else if (action === 'ready' && state.playing && state.started_at === null && this.entry.status === 'completed') {
        const delay = state.position === 0 ? Math.max(0, -(this.entry.opening_countdown?.start || 0)) : 0;
        state.started_at = Date.now()/1000 + delay;
      } else if (action === 'buffering' && state.playing && state.started_at !== null && extra.playback_started_at === state.started_at) {
        Object.assign(state, {position:extra.position, started_at:null});
      }
    }
    return {finished};
  }

  save(position = this.position()) {
    this.markPlayed();
    return {entry:this.entry, playing:this.state.playing, position:Math.max(0, position), levels:this.state.levels,
      pitch_semitones:this.state.pitch_semitones, lead_restore:this.leadRestore,
      played_at:this.playedAt, pending_finishes:this.pendingFinishes};
  }
}
