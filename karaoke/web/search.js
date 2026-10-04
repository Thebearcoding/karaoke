// Local results remain available while YouTube loads; stale responses cannot repaint.
export function createVideoSearch(onChange, fetcher = fetch, getLocal = () => []) {
  let controller, generation = 0, active;
  function cancel() {
    generation++;
    controller?.abort();
    active = null;
  }
  async function search(query, mode = 'youtube') {
    cancel();
    controller = new AbortController();
    const providers = {local:{loading:false, results:getLocal(query), error:''}};
    if (mode !== 'local') providers[mode] = {loading:true, results:[], error:''};
    active = {query, mode, providers, generation, signal:controller.signal};
    onChange(providers);
    if (mode !== 'local') await loadPage(active, false);
  }
  async function loadMore() {
    const info = active?.providers[active.mode];
    if (!info?.nextCursor || info.loading || info.loadingMore) return;
    await loadPage(active, true);
  }
  async function loadPage(request, append) {
    const {query, mode, providers, signal} = request;
    const info = providers[mode];
    const cursor = append ? info.nextCursor : null;
    const notify = () => { if (request.generation === generation) onChange(providers); };
    if (append) {
      info.loadingMore = true; info.moreError = ''; info.expired = false;
      notify();
    }
    try {
      const params = new URLSearchParams(mode === 'youtube' ? {q:query, provider:'youtube'} : {q:query, kind:mode});
      if (cursor) params.set('cursor', cursor);
      const url = ['artist','album'].includes(mode) ? `/v1/music/${mode}s/${encodeURIComponent(query)}${cursor ? '?'+new URLSearchParams({cursor}) : ''}`
        : `${mode === 'youtube' ? '/v1/search' : '/v1/music/search'}?${params}`;
      const response = await fetcher(url, {signal});
      const data = await response.json();
      if (!response.ok) throw Object.assign(new Error(data.detail || '搜索失败，请重试'), {expired:response.status === 410});
      if (!Array.isArray(data.results)) throw new Error('搜索结果暂不可用，请重试');
      if (signal.aborted || request.generation !== generation) return;
      // Suppress only overlap with earlier pages, keeping distinct source versions.
      const seen = new Set(append ? info.results.map(song => song.browse_id || song.url).filter(Boolean) : []);
      const added = data.results.filter(song => !seen.has(song.browse_id || song.url));
      info.results = append ? [...info.results, ...added] : added;
      info.metadata = {...info.metadata, ...data};
      info.nextCursor = data.next_cursor && data.next_cursor !== cursor ? data.next_cursor : null;
      info.autoBlocked = append && added.length === 0;
      info.loading = false; info.loadingMore = false;
      info.error = ''; info.moreError = '';
    } catch (error) {
      if (signal.aborted || request.generation !== generation) return;
      if (append) {
        info.loadingMore = false; info.moreError = error.message; info.expired = !!error.expired;
      } else {
        info.loading = false; info.error = error.message;
      }
    }
    notify();
  }
  function refreshLocal() {
    if (!active) return;
    const results = getLocal(active.query);
    if (JSON.stringify(results) === JSON.stringify(active.providers.local.results)) return;
    active.providers.local = {loading:false, results, error:''};
    onChange(active.providers);
  }
  return {search, loadMore, cancel, refreshLocal};
}

export function searchLibrary(library, query) {
  const normalize = text => String(text || '').normalize('NFKC').toLocaleLowerCase();
  const terms = normalize(query).trim().split(/\s+/);
  return library.filter(song => terms.every(term => normalize(`${song.title} ${song.artist}`).includes(term)))
    .sort((a,b) => Number(b.status === 'completed')-Number(a.status === 'completed'))
    .map(song => ({...song, provider:'local', thumbnail:song.poster, uploader:song.artist}));
}

export function visibleResults(songs, source) {
  return songs.filter(song => song.provider === source);
}

export function playCount(value) {
  if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null;
  const match = String(value || '').normalize('NFKC').replaceAll(',', '').trim().match(/^(?:播放次数:\s*)?(\d+(?:\.\d+)?)\s*([KMB千万萬亿億])?(?:\s*(?:plays|views|次观看|次播放))?$/i);
  return match ? Number(match[1])*({K:1e3, M:1e6, B:1e9, 千:1e3, 万:1e4, 萬:1e4, 亿:1e8, 億:1e8}[match[2]?.toUpperCase()] || 1) : null;
}

export function orderResults(songs, order) {
  if (order === 'popular') return [...songs].sort((a,b) => (playCount(b.views) ?? -1)-(playCount(a.views) ?? -1));
  if (order === 'ready') return [...songs].sort((a,b) => Number(b.status === 'completed')-Number(a.status === 'completed'));
  if (order === 'title') return [...songs].sort((a,b) => a.title.localeCompare(b.title, 'zh-CN'));
  if (order === 'relevance') {
    // Stable partition: only explicit performance labels lower a result's priority.
    const live = song => /(?:[\[(（【]\s*live\b|\blive(?:\s+(?:at|in|from|on|version|performance|\d{4})\b|\s*[|/·—–-]|[\s)\]）】]*$))|[现現][场場]|演唱[会會]|演奏[会會]|ライブ/i.test(song.title || '') &&
      !['artist','profile'].includes(song.kind);
    return [...songs].sort((a,b) => Number(live(a))-Number(live(b)));
  }
  return songs;
}

export function previewURL(song, origin) {
  const url = new URL(song.url);
  if (['local','youtube','youtube_music'].includes(song.provider) && url.hostname === 'www.youtube.com') {
    const id = url.searchParams.get('v');
    if (/^[\w-]{11}$/.test(id || '')) return `https://www.youtube.com/embed/${id}?${new URLSearchParams({autoplay:'0', playsinline:'1', origin})}`;
  }
  if (['local','bilibili'].includes(song.provider) && url.hostname === 'www.bilibili.com') {
    const id = url.pathname.match(/^\/video\/(BV[\w]{10})$/)?.[1];
    if (id) return `https://player.bilibili.com/player.html?bvid=${id}&autoplay=0&danmaku=0`;
  }
  return '';
}
