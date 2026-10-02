// Local results remain available while YouTube loads; stale responses cannot repaint.
export function createVideoSearch(onChange, fetcher = fetch, getLocal = () => []) {
  let controller, generation = 0, active;
  function cancel() {
    generation++;
    controller?.abort();
    active = null;
  }
  async function search(query) {
    cancel();
    const current = generation;
    controller = new AbortController();
    const {signal} = controller;
    const providers = {local:{loading:false, results:getLocal(query), error:''},
      youtube:{loading:true, results:[], error:''}};
    active = {query, providers};
    const notify = () => { if (current === generation) onChange(providers); };
    notify();
    try {
      const response = await fetcher(`/v1/search?${new URLSearchParams({q:query, provider:'youtube'})}`, {signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '搜索失败，请重试');
      providers.youtube = {loading:false, results:data.results, error:''};
    } catch (error) {
      if (signal.aborted) return;
      providers.youtube = {loading:false, results:[], error:error.message};
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

export function previewURL(song, origin) {
  const url = new URL(song.url);
  if (['local','youtube'].includes(song.provider) && url.hostname === 'www.youtube.com') {
    const id = url.searchParams.get('v');
    if (/^[\w-]{11}$/.test(id || '')) return `https://www.youtube.com/embed/${id}?${new URLSearchParams({autoplay:'0', playsinline:'1', origin})}`;
  }
  if (['local','bilibili'].includes(song.provider) && url.hostname === 'www.bilibili.com') {
    const id = url.pathname.match(/^\/video\/(BV[\w]{10})$/)?.[1];
    if (id) return `https://player.bilibili.com/player.html?bvid=${id}&autoplay=0&danmaku=0`;
  }
  return '';
}
