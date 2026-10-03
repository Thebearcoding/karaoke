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
    const current = generation;
    controller = new AbortController();
    const {signal} = controller;
    const providers = {local:{loading:false, results:getLocal(query), error:''}};
    if (mode !== 'local') providers[mode] = {loading:true, results:[], error:''};
    active = {query, providers};
    const notify = () => { if (current === generation) onChange(providers); };
    notify();
    if (mode === 'local') return;
    try {
      const url = mode === 'youtube' ? `/v1/search?${new URLSearchParams({q:query, provider:'youtube'})}`
        : mode === 'artist' ? `/v1/music/artists/${encodeURIComponent(query)}`
        : `/v1/music/search?${new URLSearchParams({q:query, kind:mode})}`;
      const response = await fetcher(url, {signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '搜索失败，请重试');
      providers[mode] = {loading:false, results:data.results, error:'', metadata:data};
    } catch (error) {
      if (signal.aborted) return;
      providers[mode] = {loading:false, results:[], error:error.message};
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
  return {search, cancel, refreshLocal};
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
  const match = String(value || '').replaceAll(',', '').trim().match(/^(\d+(?:\.\d+)?)\s*([KMB])?$/i);
  return match ? Number(match[1])*({K:1e3, M:1e6, B:1e9}[match[2]?.toUpperCase()] || 1) : null;
}

export function orderResults(songs, order) {
  if (order === 'popular') return [...songs].sort((a,b) => (playCount(b.views) ?? -1)-(playCount(a.views) ?? -1));
  if (order === 'ready') return [...songs].sort((a,b) => Number(b.status === 'completed')-Number(a.status === 'completed'));
  if (order === 'title') return [...songs].sort((a,b) => a.title.localeCompare(b.title, 'zh-CN'));
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
