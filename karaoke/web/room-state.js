// Room revisions order commands; progress and cache updates can share a revision.
export function isStaleRoomSnapshot(incoming, current) {
  return incoming.revision < current.revision ||
    (incoming.revision === current.revision && incoming.server_time < current.server_time);
}

const presets = {
  instrumental:{instrumental:1, backing_vocals:0, lead_vocals:0},
  karaoke:{instrumental:1, backing_vocals:1, lead_vocals:0},
  original:{instrumental:1, backing_vocals:1, lead_vocals:1},
};
const bounded = (value, min, max, fallback) => Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;

// Only the playlist and preparation results belong to the room. Transport is
// private to this browser, including when a room update removes its active song.
export class LocalPlayback {
  constructor(saved = {}) {
    this.entry = saved?.entry?.id && saved.entry.job_id ? saved.entry : null;
    this.state = {playing:false, position:bounded(saved?.position, 0, Infinity, 0), started_at:null,
      levels:{...presets.karaoke}, pitch_semitones:Math.round(bounded(saved?.pitch_semitones, -6, 6, 0))};
    for (const key in this.state.levels) this.state.levels[key] = bounded(saved?.levels?.[key], 0, 1, this.state.levels[key]);
    this.updateMix();
    this.playedAt = saved?.played_at || null;
    this.finished = new Set(saved?.finished_ids || []);
  }

  updateMix() {
    this.state.mix = Object.keys(presets).find(name => Object.keys(presets[name]).every(
      key => presets[name][key] === this.state.levels[key])) || 'custom';
  }

  reconcile(room) {
    if (!this.entry) this.select(room.entries.find(entry => !this.finished.has(entry.id)) || null, false);
    else {
      const queued = room.entries.find(entry => entry.id === this.entry.id);
      const watched = room.playback_entry?.job_id === this.entry.job_id ? room.playback_entry : null;
      const updated = queued || watched || room.library?.find(entry => entry.job_id === this.entry.job_id);
      if (updated) this.entry = {...updated, id:this.entry.id};
    }
    return {...room, ...this.state, current_id:this.entry?.id || null};
  }

  select(entry, playing) {
    this.entry = entry;
    Object.assign(this.state, {position:0, playing:playing && Boolean(entry), started_at:null});
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
    if (extra.entry_id && !['select','mix','levels','pitch'].includes(action) && extra.entry_id !== this.entry?.id) return {};
    this.markPlayed();
    let finished;
    if (action === 'next' || action === 'select') {
      const target = action === 'select' ? entries.find(entry => entry.id === extra.entry_id) : entries.find(entry => entry.id !== this.entry?.id && !this.finished.has(entry.id));
      if (action === 'select' && (!target || target.id === this.entry?.id)) return {};
      if (this.entry && (action === 'next' || this.playedAt)) finished = {entry_id:this.entry.id, playback_started_at:this.playedAt};
      if (finished) this.finished.add(finished.entry_id);
      this.select(target || null, true);
    } else if (action === 'mix') {
      state.levels = {...presets[extra.mix]}; this.updateMix();
    } else if (action === 'levels') {
      Object.assign(state.levels, extra.levels); this.updateMix();
    } else if (action === 'pitch') {
      state.pitch_semitones = extra.pitch_semitones;
    } else if (this.entry) {
      if (action === 'play') state.playing = true;
      else if (action === 'pause') Object.assign(state, {position:this.position(), playing:false, started_at:null});
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
    return {entry:this.entry, position:Math.max(0, position), levels:this.state.levels,
      pitch_semitones:this.state.pitch_semitones, played_at:this.playedAt, finished_ids:[...this.finished]};
  }
}
