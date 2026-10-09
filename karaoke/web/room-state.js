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
  return entry?.status === 'completed' ? {action:'reprepare', label:'重新准备'} : null;
}

const bounded = (value, min, max, fallback) => Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;

// One tab owns audio. Saved progress from an older owner cannot overwrite a
// newer tab's commands; storage is scoped to this browser, never the room.
export class PlaybackTabs {
  constructor(storage, id, fallback = {}) {
    this.storage = storage; this.id = id; this.key = 'karaokeBrowserPlayback';
    this.record = {revision:0, owner:null, playback:fallback};
    this.read();
  }
  read() {
    try {
      const saved = JSON.parse(this.storage?.getItem(this.key) || 'null');
      if (saved && Number.isSafeInteger(saved.revision) && saved.revision >= this.record.revision && saved.playback)
        this.record = saved;
    } catch { /* Private storage restrictions must not disable local playback. */ }
    return this.record;
  }
  owns() { return this.read().owner === this.id; }
  write(playback, {claim = false, release = false} = {}) {
    const previous = this.record.revision, current = this.read();
    if (!claim && (current.owner !== this.id || current.revision !== previous)) return false;
    this.record = {revision:current.revision+1, owner:release ? null : this.id, playback};
    try { this.storage?.setItem(this.key, JSON.stringify(this.record)); } catch {}
    return true;
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
    this.finished = new Set(saved?.finished_ids || []);
  }

  reconcile(room) {
    const first = room.entries.find(entry => !this.finished.has(entry.id)) || null;
    // Queue order chooses the next performance. Reordering while paused must
    // update the player too, without interrupting an active performance.
    if (!this.entry || (!this.state.playing && this.entry.id !== first?.id)) this.select(first, this.state.playing);
    else {
      const queued = room.entries.find(entry => entry.id === this.entry.id);
      const watched = room.playback_entry?.job_id === this.entry.job_id ? room.playback_entry : null;
      const updated = queued || watched || room.library?.find(entry => entry.job_id === this.entry.job_id);
      if (!queued && (!this.state.playing || this.state.started_at === null))
        this.select(first, this.state.playing);
      else if (updated) this.entry = {...updated, id:this.entry.id};
    }
    if (this.state.playing && this.entry?.status !== 'completed') {
      const ready = room.entries.find(entry => entry.status === 'completed' && !this.finished.has(entry.id));
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

  command(action, extra, entries) {
    const state = this.state;
    if (extra.entry_id && !['select','lead','pitch'].includes(action) && extra.entry_id !== this.entry?.id) return {};
    this.markPlayed();
    let finished;
    if (['next','select','remove'].includes(action)) {
      const target = action === 'select' ? entries.find(entry => entry.id === extra.entry_id) : entries.find(entry => entry.id !== this.entry?.id && !this.finished.has(entry.id));
      if (action === 'select' && (!target || target.id === this.entry?.id)) return {};
      if (this.entry && (action !== 'select' || this.playedAt)) finished = {entry_id:this.entry.id, playback_started_at:this.playedAt};
      if (finished) this.finished.add(finished.entry_id);
      this.select(target || null, action === 'remove' ? state.playing : true);
    } else if (action === 'lead') {
      state.levels.lead_vocals = bounded(extra.level, 0, 1, state.levels.lead_vocals);
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
      pitch_semitones:this.state.pitch_semitones, played_at:this.playedAt, finished_ids:[...this.finished]};
  }
}
