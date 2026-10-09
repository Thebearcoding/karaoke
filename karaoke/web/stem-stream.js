// Download ahead to the end of the song; decode only nearby windows.
// Frame indices place every published stem on the shared AudioContext timeline.
const decodeAheadSeconds = 24;

async function readMedia(url, signal, consume) {
  for (let attempt = 0; ; attempt++) {
    let retryAfter = 0;
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener('abort', abort, {once:true});
    if (signal.aborted) abort();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; abort(); }, 20000);
    try {
      const response = await fetch(url, {signal:controller.signal});
      if (response.ok) return await consume(response);
      const header = response.headers?.get('Retry-After');
      if (header) retryAfter = Math.max(0, Number.isFinite(Number(header)) ? Number(header) * 1000 : Date.parse(header) - Date.now()) || 0;
      throw Object.assign(new Error('音轨加载失败，请重试'), {status:response.status});
    } catch (error) {
      const temporary = timedOut || error instanceof TypeError || error.status === 408 || error.status === 429 || (error.status >= 500 && error.status < 600);
      if (signal.aborted || !temporary || attempt >= 2) throw error;
      // Retry the failed resource only; leaving the song cancels backoff.
      // Animation frames never start network requests.
      await new Promise((resolve, reject) => {
        const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(signal.reason); };
        const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, Math.max(retryAfter, 250 * 4 ** attempt));
        signal.addEventListener('abort', abort, {once:true});
        if (signal.aborted) abort();
      });
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
    }
  }
}

export class StemStream {
  constructor(context, url, signal, changed) {
    Object.assign(this, {context, url, signal, changed});
    this.windows = new Map();
    this.encoded = new Map();
    this.loading = this.read(url, signal, response => response.json()).then(info => {
      this.info = info;
      this.stems = info.stems || ['instrumental', 'backing_vocals', 'lead_vocals'];
      this.duration = info.frames / info.sample_rate;
      this.count = Math.ceil(info.frames / info.chunk_frames);
    });
  }

  read(url, signal, consume) {
    return readMedia(`${url}?sample_rate=${this.context.sampleRate}`, signal, consume);
  }

  index(position) {
    return Math.min(this.count - 1, Math.floor((Math.max(0, position) * this.info.sample_rate + 1e-6) / this.info.chunk_frames));
  }

  at(position) { return this.info && this.windows.get(this.index(position))?.audio; }

  download(index) {
    if (!this.encoded.has(index)) {
      this.encoded.set(index, Promise.all(this.stems.map(stem =>
        this.read(`${this.url}/${stem}/${index}`, this.signal, response => response.arrayBuffer()))));
    }
    return this.encoded.get(index);
  }

  async downloadAhead() {
    // Start with the requested position, then keep fetching the entire song.
    // A seek can fetch its own target without waiting behind distant audio.
    const first = this.first;
    for (let offset = 0; offset < this.count && !this.signal.aborted; offset++) {
      try { await this.download((first + offset) % this.count); }
      catch { /* The affected window reports its bounded retry failure. */ }
    }
  }

  request(index) {
    if (this.windows.has(index)) return this.windows.get(index);
    const controller = new AbortController();
    const abort = () => controller.abort();
    this.signal.addEventListener('abort', abort, {once:true});
    if (this.signal.aborted) abort();
    const item = {controller};
    this.windows.set(index, item);
    let timer;
    item.loading = this.download(index).then(data => {
      if (controller.signal.aborted) throw new Error('音轨加载已取消');
      timer = setTimeout(abort, 20000);
      return Promise.all(data.map(bytes => this.context.decodeAudioData(bytes.slice(0))));
    }).then(audio => {
      if (controller.signal.aborted) throw new Error('音轨加载超时，请重试');
      item.audio = audio;
      this.prefetch();
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
    this.first = index;
    this.last = this.index(position + decodeAheadSeconds);
    for (const [key, item] of this.windows) {
      if (key < index || key > this.last) {
        item.controller.abort();
        this.windows.delete(key);
      }
    }
    this.request(index);
    this.prefetch();
    this.downloading ||= this.downloadAhead();
  }

  prefetch() {
    // Decode the current window first, then the reserve, all from local data.
    // Compressed copies survive seeks; decoded windows stay bounded.
    for (let index = this.first; index <= this.last; index++) {
      const item = this.windows.get(index);
      if (!item) { this.request(index); return; }
      if (!item.audio) return;
    }
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
