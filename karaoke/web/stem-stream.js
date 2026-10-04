// Two short windows, all three stems together. Frame indices, not fetch timing,
// determine where each window belongs on the shared AudioContext timeline.
const stems = ['instrumental', 'backing_vocals', 'lead_vocals'];

export class StemStream {
  constructor(context, url, signal, changed) {
    Object.assign(this, {context, url, signal, changed});
    this.windows = new Map();
    this.loading = this.read(url, signal).then(response => response.json()).then(info => {
      this.info = info;
      this.duration = info.frames / info.sample_rate;
      this.count = Math.ceil(info.frames / info.chunk_frames);
    });
  }

  async read(url, signal) {
    const response = await fetch(`${url}?sample_rate=${this.context.sampleRate}`, {signal});
    if (!response.ok) throw new Error('音轨加载失败，请重试');
    return response;
  }

  index(position) {
    return Math.min(this.count - 1, Math.floor((Math.max(0, position) * this.info.sample_rate + 1e-6) / this.info.chunk_frames));
  }

  at(position) { return this.info && this.windows.get(this.index(position))?.audio; }

  request(index) {
    if (this.windows.has(index)) return this.windows.get(index);
    const controller = new AbortController();
    const abort = () => controller.abort();
    this.signal.addEventListener('abort', abort, {once:true});
    if (this.signal.aborted) abort();
    const item = {controller};
    this.windows.set(index, item);
    const timer = setTimeout(abort, 20000);
    item.loading = Promise.all(stems.map(async stem => {
      const response = await this.read(`${this.url}/${stem}/${index}`, controller.signal);
      return this.context.decodeAudioData(await response.arrayBuffer());
    })).then(audio => {
      if (controller.signal.aborted) throw new Error('音轨加载超时，请重试');
      item.audio = audio;
      this.changed();
      return audio;
    }).catch(error => {
      item.error = error;
      throw error;
    }).finally(() => {
      clearTimeout(timer);
      this.signal.removeEventListener('abort', abort);
    });
    // Prefetch failures are surfaced when that window is needed; never retry
    // every animation frame or discard the audio that is still playing.
    item.loading.catch(() => {});
    return item;
  }

  around(position) {
    const index = this.index(position);
    for (const [key, item] of this.windows) {
      if (key < index || key > index + 1) {
        item.controller.abort();
        this.windows.delete(key);
      }
    }
    this.request(index);
    if (index + 1 < this.count) this.request(index + 1);
  }

  async prepare(position) {
    await this.loading;
    if (this.signal.aborted) return;
    this.around(position);
    return this.request(this.index(position)).loading;
  }

  bounds(index) {
    const {sample_rate:rate, chunk_frames:chunk, frames, padding_frames:padding} = this.info;
    return {start:index * chunk / rate, end:Math.min(frames, (index + 1) * chunk) / rate,
      decodedStart:Math.max(0, index * chunk - padding) / rate};
  }
}
