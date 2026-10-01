// Each provider can finish independently. Old queries may never repaint the UI.
export function createVideoSearch(onChange, fetcher = fetch) {
  let controller, generation = 0;
  function cancel() {
    generation++;
    controller?.abort();
  }
  async function search(query) {
    cancel();
    const current = generation;
    controller = new AbortController();
    const {signal} = controller;
    const providers = Object.fromEntries(['youtube', 'bilibili'].map(name => [name, {loading:true, results:[], error:''}]));
    const notify = () => { if (current === generation) onChange(providers); };
    notify();
    await Promise.all(Object.keys(providers).map(async provider => {
      try {
        const response = await fetcher(`/v1/search?${new URLSearchParams({q:query, provider})}`, {signal});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '搜索失败，请重试');
        providers[provider] = {loading:false, results:data.results, error:''};
      } catch (error) {
        if (signal.aborted) return;
        providers[provider] = {loading:false, results:[], error:error.message};
      }
      notify();
    }));
  }
  return {search, cancel};
}

export function previewURL(song, origin) {
  const url = new URL(song.url);
  if (song.provider === 'youtube' && url.hostname === 'www.youtube.com') {
    const id = url.searchParams.get('v');
    if (/^[\w-]{11}$/.test(id || '')) return `https://www.youtube.com/embed/${id}?${new URLSearchParams({autoplay:'0', playsinline:'1', origin})}`;
  }
  if (song.provider === 'bilibili' && url.hostname === 'www.bilibili.com') {
    const id = url.pathname.match(/^\/video\/(BV[\w]{10})$/)?.[1];
    if (id) return `https://player.bilibili.com/player.html?bvid=${id}&autoplay=0&danmaku=0`;
  }
  return '';
}
