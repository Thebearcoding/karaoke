import { Player } from './player.js';
import { createVideoSearch, previewURL, searchLibrary, playCount, orderResults } from './search.js';

const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const time = seconds => `${Math.floor(Math.max(0, seconds || 0)/60)}:${String(Math.floor(Math.max(0, seconds || 0)%60)).padStart(2,'0')}`;
const remote = new URLSearchParams(location.search).get('remote') === '1';
const stages = {download:'下载视频', verify_video:'核对视频版本', decode_audio:'提取音频', separate_music:'分离伴奏', separate_lead:'分离主唱', package_two_stems:'整理音轨', package_three_stems:'整理音轨', resample_vocals:'整理人声', transcribe:'识别歌词', prepare_lyrics:'整理歌词', align_words:'对齐歌词', align_phonemes:'对齐起唱时间', align_repeats:'检查重复段落', align_context:'校准字幕', render:'生成字幕 MV'};
let state = {entries:[], current_id:null, playing:false, position:0, started_at:null, mix:'karaoke', levels:{instrumental:1,backing_vocals:1,lead_vocals:0}, revision:0};
let library = [], filter = 'all', view = 'library', clockOffset = 0, readySent = null, loading = false, loadError = '', seeking = false;
let toastTimer, librarySignature = '', queueSignature = '';
let pendingLevels = {}, sendingLevels = {}, levelTimer = null, levelsBusy = false;
let levelRequest = Promise.resolve();
let pitchRequest = Promise.resolve();
let searchTimer, searchQuery = '', searchSongs = [], previewSong = null, skipping = false, composing = false;
let videoSource = 'songs', searchOrder = 'relevance', searchSnapshot = null, catalogueSelection = null;
const addingVideos = new Set();
const videoSearch = createVideoSearch(renderVideoResults, fetch, query => searchLibrary(library, query));
const current = () => state.entries.find(e => e.id === state.current_id);
const effectiveLevels = () => ({...state.levels, ...sendingLevels, ...pendingLevels});

function renderMixer() {
  const levels = effectiveLevels();
  player.setLevels(levels);
  document.querySelectorAll('[data-level]').forEach(input => {
    input.value = levels[input.dataset.level];
    const percentage = `${Math.round(Number(input.value)*100)}%`;
    $(`value-${input.dataset.level}`).textContent = percentage;
    input.setAttribute('aria-valuetext', percentage);
  });
}

async function flushLevels() {
  levelTimer = null;
  if (levelsBusy || !Object.keys(pendingLevels).length) return;
  levelsBusy = true;
  sendingLevels = pendingLevels;
  pendingLevels = {};
  levelRequest = command('levels', {levels:sendingLevels});
  await levelRequest;
  sendingLevels = {};
  levelsBusy = false;
  renderMixer();
  if (Object.keys(pendingLevels).length) levelTimer = setTimeout(flushLevels, 80);
}

function renderStorage() {
  const storage = state.storage;
  if (!storage?.limit_bytes) return;
  const gib = bytes => (bytes/1024**3).toLocaleString('zh-CN', {maximumFractionDigits:2});
  $('storage-usage').textContent = `${gib(storage.used_bytes)} / ${gib(storage.limit_bytes)} GiB`;
  $('storage-meter').max = storage.limit_bytes;
  $('storage-meter').value = storage.used_bytes;
  $('storage-note').hidden = storage.can_prepare;
  if (document.activeElement !== $('storage-limit')) $('storage-limit').value = storage.limit_bytes/1024**3;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').hidden = true, 3800);
}

async function request(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求没有成功，请检查输入后重试');
  return data;
}

async function command(action, extra = {}) {
  try {
    const updated = await request('/v1/room/commands', {action, entry_id:state.current_id, ...extra});
    acceptState(updated);
    return updated;
  } catch (error) { toast(error.message); return null; }
}

function switchView(next) {
  view = next;
  const mobile = matchMedia('(max-width:760px)').matches;
  $('library-view').hidden = next !== 'library';
  $('player-view').hidden = next !== 'player';
  $('queue-panel').classList.toggle('mobile-open', next === 'queue');
  if (next === 'queue' && !mobile) { view = 'library'; $('library-view').hidden = false; }
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('selected', button.dataset.view === view);
    button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false');
  });
}

function renderLibrary() {
  const query = $('song-input').value.trim();
  const isURL = /^https?:\/\//i.test(query);
  $('title-row').hidden = !isURL;
  $('submit-song').innerHTML = `${isURL ? '点歌' : '搜索'} ${icon('arrow')}`;
  const search = isURL ? '' : query.toLocaleLowerCase();
  const songs = library.filter(song => (!search || search.split(/\s+/).every(word => `${song.title} ${song.artist}`.toLocaleLowerCase().includes(word))) &&
    (search || filter === 'all' || (filter === 'ready' ? ['completed', 'video_evicted'].includes(song.status) : song.language === filter)));
  const preparedView = filter === 'ready' && !search;
  const missingVideo = songs.filter(song => song.status === 'video_evicted').length;
  $('library-count').textContent = preparedView && missingVideo
    ? `${songs.length-missingVideo} 首可播放 · ${missingVideo} 首待补视频`
    : `${songs.length} 首${preparedView ? '已准备' : '歌曲'}`;
  $('library-heading').textContent = search ? '曲库里的歌' : '曲库';
  $('local-library').hidden = Boolean(searchQuery || (search && !songs.length));
  document.querySelector('#local-library .filters').hidden = Boolean(search);
  const signature = JSON.stringify([songs.map(({stage_progress, active_stages, ...song}) => song), filter, query]);
  if (signature !== librarySignature) {
    librarySignature = signature;
    $('song-grid').innerHTML = songs.map(song => {
      const ready = song.status === 'completed';
      const label = ready ? '● 已准备' : song.status === 'video_evicted' ? '待补视频' : song.status === 'queued' ? '等待准备' : song.status === 'failed' ? '准备失败' : song.status === 'running' ? '' : '需准备';
      return `<article class="song-card" data-card="${escape(song.id)}">
    <div class="song-poster">${song.poster ? `<img src="${escape(song.poster)}" alt="" loading="lazy">` : `<span class="poster-initial">${escape(song.title.slice(0,1))}</span>`}
      <span class="poster-duration">${time(song.duration)}</span>
      <button class="add-song" data-song="${escape(song.id)}" aria-label="点歌：${escape(song.title)}" title="点歌">${icon('plus')}</button></div>
    <div class="song-detail"><h3>${escape(song.title)}</h3><div class="song-info-row"><span class="song-artist">${escape(song.artist)}</span>${label ? `<span class="${ready ? 'ready-label' : 'prepare-label'}">${label}</span>` : ''}</div>${song.status === 'running' ? `<div class="preparation-slot">${preparationHTML(song)}</div>` : ''}</div></article>`;
    }).join('');
  } else {
    updatePreparationSlots($('song-grid'), songs);
  }
  $('search-empty').hidden = songs.length > 0;
}

function renderVideoResults(providers) {
  searchSnapshot = providers;
  $('library-view').classList.add('searching');
  $('video-results').hidden = false;
  $('local-library').hidden = true;
  const info = providers[videoSource];
  if (!info) return;
  const artists = videoSource === 'artists', album = videoSource === 'album', catalogue = album || videoSource === 'artist';
  const metadata = info.metadata || catalogueSelection || {};
  const popularityAvailable = info.results.some(song => playCount(song.views) !== null);
  if (searchOrder === 'popular' && !popularityAvailable) searchOrder = 'relevance';
  $('search-order').value = searchOrder;
  $('search-order').querySelector('[value="popular"]').disabled = !popularityAvailable;
  $('search-order').querySelector('[value="relevance"]').textContent = album ? '专辑顺序' : '默认';
  $('search-order-label').hidden = artists || info.loading || !info.results.length;
  $('search-heading').textContent = album ? '专辑曲目' : catalogue ? '热门歌曲' : artists ? '找一位歌手' : '搜索结果';
  $('search-order-note').textContent = artists ? '选一位歌手，看看有哪些歌想唱' :
    searchOrder === 'popular' ? '已加载结果按播放量排序 · YouTube 提供' : searchOrder === 'ready' ? '已加载结果中，已准备歌曲排在前面' :
    searchOrder === 'title' ? '已加载结果按歌名排序' : album ? 'YouTube Music 专辑曲目顺序' :
    videoSource === 'local' ? '本地曲库 · 现场版靠后' : 'YouTube 原顺序 · 现场版靠后';
  $('search-status').innerHTML = `<span class="${info.error ? 'search-error' : ''}">${info.loading ? '正在查找…' : info.error ? escape(info.error) : info.results.length ? `${videoSource === 'local' ? '' : '已加载 '}${info.results.length} ${artists ? '位歌手' : videoSource === 'youtube' ? '个视频' : videoSource === 'songs' ? '项' : '首'}${catalogue && info.metadata?.complete === false ? ' · 歌手页精选' : ''}` : '没有找到，试试另一个关键词或搜索类型'}${info.error ? '<button data-search-retry>重试</button>' : ''}</span>`;
  $('artist-summary').hidden = !catalogue;
  if (catalogue) $('artist-summary').innerHTML = `${metadata.thumbnail ? `<img src="${escape(metadata.thumbnail)}" alt="" referrerpolicy="no-referrer">` : ''}<div><p>${album ? '专辑' : '歌手'}</p><h3>${escape(metadata.title)}</h3>${metadata.monthly_listeners ? `<p>${escape(metadata.monthly_listeners)} · YouTube Music</p>` : ''}<button class="text-button" data-artist-back>返回搜索</button></div>`;
  renderSearchMore(info);
  searchSongs = orderResults(info.results.map(song => ({...song,
    status:library.find(item => item.url === song.url)?.status || song.status})), album && searchOrder === 'relevance' ? 'album' : searchOrder);
  const videos = videoSource === 'youtube', local = videoSource === 'local';
  $('search-list').className = artists ? '' : `search-list${videos ? ' video-list' : local ? ' local-list' : ''}`;
  $('search-list').setAttribute('role', artists ? 'group' : 'table');
  $('search-list').setAttribute('aria-label', artists ? '歌手列表' : videos ? '视频列表' : '歌曲列表');
  $('music-columns').hidden = artists || !searchSongs.length;
  $('music-title-heading').textContent = videos ? '视频' : videoSource === 'songs' ? '名称' : '歌曲';
  $('music-artist-heading').textContent = videos ? '频道' : '歌手';
  $('music-alternative-heading').textContent = videos ? '来源' : '找 MV';
  $('video-grid').className = artists ? 'artist-grid' : 'music-list';
  if (artists) $('video-grid').removeAttribute('role');
  else $('video-grid').setAttribute('role', 'rowgroup');
  const rows = searchSongs.map((song, index) => {
    const thumbnail = song.thumbnail ? `<img src="${escape(song.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : `<span class="music-initial">${escape(song.title.slice(0,1))}</span>`;
    if (artists) return `<button class="artist-card" data-artist-result="${index}">${thumbnail}<strong>${escape(song.title)}</strong><span>热门歌曲 ${icon('arrow')}</span></button>`;
    const playable = !song.kind || song.kind === 'song';
    const catalogueAction = ['artist','album'].includes(song.kind) ? `data-catalogue="${index}"` : '';
    const open = playable ? `data-preview="${index}"` : catalogueAction || `href="${escape(song.url)}" target="_blank" rel="noopener noreferrer"`;
    const tag = playable || catalogueAction ? 'button' : 'a';
    const action = playable ? `<button class="music-add" data-add-video="${index}" aria-label="点歌：${escape(song.title)}">${icon('plus')}<span>点歌</span></button>` :
      `<${tag} class="music-add" ${open}>${catalogueAction ? '查看歌曲' : '打开'} ${icon('arrow')}</${tag}>`;
    const credits = song.artists?.length ? song.artists.map(a => a.id ? `<button data-artist-id="${escape(a.id)}" data-artist-name="${escape(a.name)}">${escape(a.name)}</button>` : escape(a.name)).join(' · ') : escape(song.uploader || '—');
    return `<article class="music-row" data-result="${index}" role="row"><span class="music-rank" role="cell">${String(index+1).padStart(2,'0')}</span>
      <div class="music-song" role="cell"><${tag} class="music-cover" ${open} aria-label="${playable ? '预览' : '打开'}：${escape(song.title)}">${thumbnail}<span>${icon(playable ? 'play' : 'arrow')}</span></${tag}>
        <div class="music-title"><h3 title="${escape(song.title)}">${escape(song.title)}</h3><div class="music-meta"><span>${escape(song.version || (local ? '本地歌曲' : 'YouTube 视频'))}</span><span class="music-availability"></span></div></div></div>
      <div class="music-credits" role="cell" title="${escape(song.artists?.map(a => a.name).join(' · ') || song.uploader || '')}">${credits}</div>
      <div class="music-album" role="cell" title="${escape(song.album || '')}">${escape(song.album || '—')}</div>
      <div class="music-plays" role="cell">${playCount(song.views) !== null ? `${formatPlays(song.views)}<small>播放</small>` : '—'}</div>
      <div class="music-duration" role="cell">${song.duration ? time(song.duration) : '—'}</div>
      <div class="music-actions" role="cell">${action}</div>
      <div class="music-alternative" role="cell">${song.provider === 'youtube_music' && playable ? `<button class="find-mv" data-find-mv="${index}" aria-label="找 MV：${escape(song.title)}">找 MV</button>` : videos ? `<a class="find-mv" href="${escape(song.url)}" target="_blank" rel="noopener noreferrer">YouTube ↗</a>` : ''}</div></article>`;
  });
  updateSearchRows(rows);
  document.querySelectorAll('[data-video-source]').forEach(button => {
    const selected = button.dataset.videoSource === (album ? 'songs' : catalogue ? 'artists' : videoSource);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  updateVideoButtons();
}

function updateSearchRows(rows) {
  const grid = $('video-grid');
  const existing = new Map([...grid.children].map(node => [node.dataset.searchKey, node]));
  const occurrences = new Map();
  searchSongs.forEach((song, index) => {
    const source = song.browse_id || song.url;
    const occurrence = occurrences.get(source) || 0;
    occurrences.set(source, occurrence+1);
    const key = `${source}:${occurrence}`;
    const content = JSON.stringify([song.kind, song.title, song.thumbnail, song.uploader, song.artists, song.version, song.album, song.views, song.duration]);
    let node = existing.get(key);
    existing.delete(key);
    if (node && node.searchContent !== content) { node.remove(); node = null; }
    if (!node) {
      const template = document.createElement('template');
      template.innerHTML = rows[index];
      node = template.content.firstElementChild;
      node.dataset.searchKey = key;
      node.searchContent = content;
    }
    // Keep existing cards, images and focused buttons alive when appending a page.
    for (const attribute of ['data-result', 'data-artist-result', 'data-catalogue', 'data-preview', 'data-add-video', 'data-find-mv']) {
      if (node.hasAttribute(attribute)) node.setAttribute(attribute, index);
      node.querySelectorAll(`[${attribute}]`).forEach(button => button.setAttribute(attribute, index));
    }
    const rank = node.querySelector('.music-rank');
    if (rank) rank.textContent = String(index+1).padStart(2,'0');
    if (grid.children[index] !== node) grid.insertBefore(node, grid.children[index] || null);
  });
  existing.forEach(node => node.remove());
}

function renderSearchMore(info) {
  const more = $('search-more'), button = $('search-more-button');
  more.hidden = videoSource === 'local' || info.loading || !!info.error || (!info.results.length && !info.nextCursor);
  $('search-more-status').textContent = info.loadingMore ? '正在加载更多…' : info.moreError ||
    (info.nextCursor ? '下拉继续看' : '已经到底了');
  $('search-more-status').classList.toggle('search-error', !!info.moreError);
  button.hidden = !info.nextCursor;
  button.disabled = !!info.loadingMore;
  button.textContent = info.expired ? '重新搜索' : info.moreError ? '重试加载' : '加载更多';
  // Reobserve after a page arrives, including when it was too short to fill the screen.
  searchObserver?.unobserve(more);
  if (!more.hidden && info.nextCursor && !info.loadingMore && !info.moreError && !info.autoBlocked) searchObserver?.observe(more);
}

function formatPlays(value) {
  return new Intl.NumberFormat('zh-CN', {notation:'compact', maximumFractionDigits:1}).format(playCount(value));
}

function openCatalogue(item) {
  const query = $('song-input').value.trim();
  clearVideoSearch();
  const returnMode = catalogueSelection?.returnMode || videoSource;
  catalogueSelection = {...item, returnMode};
  videoSource = item.kind || 'artist'; searchOrder = 'relevance';
  searchQuery = query;
  videoSearch.search(item.browse_id, videoSource);
}

function changeSearchMode(mode) {
  videoSource = mode; searchOrder = 'relevance'; catalogueSelection = null;
  searchVideos(true);
}

function updateVideoButtons() {
  document.querySelectorAll('[data-add-video], #preview-add').forEach(button => {
    const song = button.id === 'preview-add' ? previewSong : searchSongs[button.dataset.addVideo];
    if (!song) return;
    const queued = state.entries.some(entry => entry.url === song.url);
    const adding = addingVideos.has(song.url);
    button.disabled = queued || adding || song.available === false;
    const label = button.querySelector('span') || button;
    label.textContent = queued ? '已点 ✓' : adding ? '点歌中…' : song.available === false ? '暂不可用' : '点歌';
  });
  document.querySelectorAll('.music-availability').forEach(label => {
    const song = searchSongs[label.closest('[data-result]').dataset.result];
    const cached = library.find(item => item.url === song.url);
    label.textContent = cached?.status === 'completed' ? '已准备' : cached?.status === 'running' ? '准备中' : cached?.status === 'queued' ? '等待准备' : cached?.status === 'video_evicted' ? '待补视频' : '';
    label.classList.toggle('is-ready', cached?.status === 'completed');
  });
}

function clearVideoSearch() {
  clearTimeout(searchTimer);
  videoSearch.cancel();
  searchQuery = '';
  searchSongs = [];
  searchSnapshot = null;
  $('library-view').classList.remove('searching');
  $('video-results').hidden = true;
  searchObserver?.unobserve($('search-more'));
  $('search-more').hidden = true;
  $('video-grid').replaceChildren();
  $('search-status').replaceChildren();
}

function searchVideos(force = false) {
  clearTimeout(searchTimer);
  const query = $('song-input').value.trim();
  if (!query || /^https?:\/\//i.test(query) || (!force && query === searchQuery)) return;
  clearVideoSearch();
  searchQuery = query;
  if (query.length > 160) { toast('搜索关键词请控制在 160 字以内'); return; }
  videoSearch.search(['artist','album'].includes(videoSource) ? catalogueSelection.browse_id : query, videoSource);
}

function editSearch() {
  if (['artist','album'].includes(videoSource)) { videoSource = catalogueSelection?.returnMode || 'songs'; catalogueSelection = null; }
  clearVideoSearch();
  renderLibrary();
  if (!composing) searchTimer = setTimeout(searchVideos, 700);
}

async function addVideo(song) {
  if (!song || addingVideos.has(song.url)) return;
  addingVideos.add(song.url); updateVideoButtons();
  $('preview-error').hidden = true;
  try {
    const cached = library.find(item => item.url === song.url && item.job_id);
    acceptState(await request('/v1/room/entries', cached ? {job_id:cached.job_id} : {url:song.url, source_title:song.title}));
    if ($('video-preview').open) $('video-preview').close();
    toast('已点歌');
  } catch (error) {
    toast(error.message);
    $('preview-error').textContent = error.message;
    $('preview-error').hidden = false;
  } finally { addingVideos.delete(song.url); updateVideoButtons(); }
}

function previewVideo(song) {
  previewSong = song;
  $('preview-title').textContent = song.title;
  $('preview-source').href = song.url;
  $('preview-source').textContent = `在 ${song.url.includes('youtube.com') ? 'YouTube' : 'B站'} 观看 ↗`;
  $('preview-error').hidden = true;
  const cached = library.find(item => item.url === song.url && item.status === 'completed' && item.job_id);
  if (cached) {
    const video = document.createElement('video');
    video.src = `/v1/jobs/${cached.job_id}/artifacts/mv`;
    video.controls = true;
    video.preload = 'metadata';
    $('preview-player').replaceChildren(video);
    $('video-preview').showModal();
    updateVideoButtons();
    return;
  }
  const iframe = document.createElement('iframe');
  iframe.src = previewURL(song, location.origin);
  iframe.title = song.title;
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  iframe.allowFullscreen = true;
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';
  $('preview-player').replaceChildren(iframe);
  updateVideoButtons();
  $('video-preview').showModal();
}

function statusText(entry) {
  if (entry.status === 'failed') return entry.error?.message || '准备失败，请重试';
  if (entry.status === 'completed') return entry.id === state.current_id ? (state.playing ? '正在播放' : '已暂停') : '已准备';
  if (entry.status === 'queued') return '等待准备';
  return (entry.active_stages || []).map(s => stages[s] || '正在准备').join(' · ') || '开始准备';
}

function preparationHTML(entry) {
  if (entry?.status !== 'running') return '';
  return `<div class="preparation">${(entry.active_stages?.length ? entry.active_stages : ['starting']).map(stage => {
    const p = entry.stage_progress?.[stage] || {};
    const measured = Number.isFinite(p.completed) && Number.isFinite(p.total) && p.total > 0;
    const percent = measured ? Math.min(100, Math.floor(100*p.completed/p.total)) : null;
    const label = `${stages[stage] || '开始准备'}${p.detail ? ` · ${p.detail}` : ''}`;
    const amount = percent !== null ? `${percent}%` : p.unit === 'bytes' && Number.isFinite(p.completed) ? `${(p.completed/1024**2).toFixed(1)} MB` : '';
    return `<div class="preparation-stage"><div class="preparation-label"><span>${escape(label)}</span><span>${amount}</span></div><progress max="100" ${percent === null ? '' : `value="${percent}"`} aria-label="${escape(label)}"></progress></div>`;
  }).join('')}</div>`;
}

function updatePreparationSlots(container, songs) {
  for (const element of container.children) {
    const slot = element.querySelector('.preparation-slot');
    if (slot) slot.innerHTML = preparationHTML(songs.find(s => s.id === (element.dataset.entry || element.dataset.card)));
  }
}

function renderQueue() {
  const signature = JSON.stringify([state.entries.map(({stage_progress, active_stages, ...entry}) => entry), state.current_id, state.playing]);
  $('queue-count').textContent = String(state.entries.length).padStart(2,'0');
  $('nav-count').textContent = state.entries.length;
  $('queue-empty').hidden = state.entries.length > 0;
  const preparing = state.entries.filter(e => ['running','queued'].includes(e.status)).length;
  $('queue-badge').textContent = `${preparing} 首准备中`;
  $('queue-badge').hidden = !preparing;
  if (signature === queueSignature) {
    // Progress pushes must not replace focused buttons or interrupt a click.
    updatePreparationSlots($('queue-list'), state.entries);
    return;
  }
  queueSignature = signature;
  $('queue-list').innerHTML = state.entries.map((entry, i) => {
    const active = entry.id === state.current_id;
    const preparing = ['running','queued'].includes(entry.status);
    return `<article class="queue-entry ${active ? 'current' : ''}" data-entry="${escape(entry.id)}"><span class="entry-number">${active && state.playing ? '♫' : String(i+1).padStart(2,'0')}</span>
      ${entry.poster ? `<img class="entry-art" src="${escape(entry.poster)}" alt="">` : `<span class="entry-art">♪</span>`}
      <div class="entry-info"><h3 class="entry-title">${escape(entry.title)}</h3><p class="entry-artist">${escape(entry.artist || (entry.url.includes('bili') ? 'Bilibili 视频' : 'YouTube 视频'))}</p>${entry.status === 'running' ? `<div class="preparation-slot">${preparationHTML(entry)}</div>` : `<p class="entry-status ${preparing ? 'preparing' : entry.status === 'failed' ? 'failed' : ''}">${escape(statusText(entry))}</p>`}</div>
      <div class="entry-tools">${entry.status === 'failed' ? `<button data-action="retry">重试</button>` : ''}${!active ? `<button data-action="select">${icon('play')}现在唱</button><button data-action="up" aria-label="上移：${escape(entry.title)}" ${i <= (state.current_id ? 1 : 0) ? 'disabled' : ''}>${icon('up')}</button><button data-action="down" aria-label="下移：${escape(entry.title)}" ${i === state.entries.length-1 ? 'disabled' : ''}>${icon('down')}</button>` : ''}<button data-action="remove" aria-label="移除：${escape(entry.title)}">${icon('trash')}</button></div></article>`;
  }).join('');
}

function updateTransport() {
  const song = current();
  $('now-title').textContent = song?.title || '未播放';
  $('now-artist').textContent = song?.artist || '';
  $('now-art').style.backgroundImage = song?.poster ? `url("${song.poster}")` : '';
  $('now-art').querySelector('span').hidden = Boolean(song?.poster);
  $('player-title').textContent = song?.title || '未播放';
  $('player-artist').textContent = song?.artist || '';
  $('play-icon').setAttribute('href', state.playing && (remote || player.unlocked) ? '#i-pause' : '#i-play');
  $('play-toggle').setAttribute('aria-label', state.playing && (remote || player.unlocked) ? '暂停' : '播放');
  $('play-toggle').disabled = state.entries.length === 0;
  document.querySelectorAll('[data-next]').forEach(button => button.disabled = skipping || !state.current_id);
  $('restart').disabled = !song;
  $('transport-restart').disabled = !song;
  $('seek').disabled = !song || song.status !== 'completed';
  $('seek').max = song?.duration || 1;
  $('duration').textContent = time(song?.duration);
  $('playing-dot').hidden = !state.playing;
  $('quality-note').hidden = !song?.lyrics_need_review && !song?.lyrics_unavailable;
  $('quality-note').textContent = song?.lyrics_unavailable ? '歌词暂缺' : '歌词待校对';
  const pitch = state.pitch_semitones || 0;
  $('pitch-value').textContent = pitch ? `${pitch > 0 ? '+' : '−'}${Math.abs(pitch)}` : '原调';
  $('pitch-value').setAttribute('aria-label', `当前${pitch ? `${pitch > 0 ? '升' : '降'} ${Math.abs(pitch)} 个半音` : '原调'}`);
  $('pitch-down').disabled = pitch <= -6;
  $('pitch-up').disabled = pitch >= 6;
  document.querySelectorAll('[data-mix]').forEach(button => {
    button.classList.toggle('selected', button.dataset.mix === state.mix);
    button.setAttribute('aria-pressed', String(button.dataset.mix === state.mix));
  });
  renderMixer();
  updateCover();
}

function updateCover() {
  const song = current();
  $('screen-cover').hidden = Boolean(song?.status === 'completed' && !loading && !loadError && !remote && player.unlocked && player.buffers);
  $('screen-status').textContent = remote ? '遥控模式' : loadError || (!song ? '请先点歌' : song.status !== 'completed' ? statusText(song) : loading ? '加载音轨…' : '');
  $('screen-status').hidden = !$('screen-status').textContent || song?.status === 'running';
  $('screen-start').hidden = remote || !song || song.status !== 'completed' || loading;
  $('screen-start').firstChild.textContent = loadError ? '重试加载 ' : '开始唱 ';
}

function updateTime(position) {
  if (!seeking) $('seek').value = position;
  $('current-time').textContent = time(position);
  updateCover();
}

const player = new Player($('video'), updateTime, id => command('next', {entry_id:id}));

async function syncPlayback() {
  if (remote) return;
  const song = current();
  if (!song || song.status !== 'completed') {
    player.clear(); readySent = null; loading = false; loadError = '';
    updateCover(); return;
  }
  try {
    loading = !player.buffers || player.key !== song.id;
    updateCover();
    await player.load(song);
    if (state.current_id !== song.id) return;
    loading = false; loadError = '';
    if (state.playing && !state.started_at && player.unlocked && readySent !== song.id) {
      readySent = song.id;
      await command('ready', {entry_id:song.id});
    }
    if (!state.playing || state.started_at) readySent = null;
    player.apply({...state, levels:effectiveLevels()}, clockOffset);
    updateTransport();
  } catch (error) {
    if (state.current_id !== song.id) return;
    loading = false; loadError = error.message;
    updateCover(); toast(loadError);
  }
}

function acceptState(updated) {
  if (updated.revision < state.revision) return;
  state = updated;
  library = updated.library || [];
  renderLibrary();
  videoSearch.refreshLocal();
  $('screen-preparation').innerHTML = preparationHTML(current());
  clockOffset = state.server_time - Date.now()/1000;
  renderQueue(); updateTransport(); renderStorage();
  updateVideoButtons();
  if (!player.anchor) updateTime(state.position);
  syncPlayback();
}

async function activate() {
  if (!remote) {
    try { await player.unlock(); }
    catch { toast('浏览器还没有启用声音，请再点一次播放'); return false; }
  }
  return true;
}

async function togglePlay() {
  const wasUnlocked = player.unlocked;
  if (!(await activate())) return;
  if (state.playing && !wasUnlocked && !remote) { await syncPlayback(); return; }
  await command(state.playing ? 'pause' : 'play');
}

document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => switchView(button.dataset.view));
document.querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
  filter = button.dataset.filter;
  document.querySelectorAll('[data-filter]').forEach(b => b.classList.toggle('selected', b === button));
  renderLibrary();
});
$('song-input').oninput = editSearch;
document.querySelectorAll('[data-video-source]').forEach(button => button.onclick = () => changeSearchMode(button.dataset.videoSource));
$('search-order').onchange = () => { searchOrder = $('search-order').value; if (searchSnapshot) renderVideoResults(searchSnapshot); };
$('artist-summary').onclick = event => { if (event.target.closest('[data-artist-back]')) changeSearchMode(catalogueSelection?.returnMode || 'songs'); };
$('song-input').oncompositionstart = () => { composing = true; clearVideoSearch(); };
$('song-input').oncompositionend = () => { composing = false; editSearch(); };
$('clear-search').onclick = () => { $('song-input').value = ''; editSearch(); $('song-input').focus(); };
const searchObserver = typeof IntersectionObserver === 'function' ? new IntersectionObserver(entries => {
  const info = searchSnapshot?.[videoSource];
  if (entries.some(entry => entry.isIntersecting) && view === 'library' && info && !info.moreError && !info.autoBlocked) videoSearch.loadMore();
}, {rootMargin:'0px 0px 320px 0px'}) : null;
$('search-more-button').onclick = () => {
  if (searchSnapshot?.[videoSource]?.expired) searchVideos(true);
  else videoSearch.loadMore();
};
$('search-status').onclick = event => { if (event.target.closest('[data-search-retry]')) searchVideos(true); };
$('video-grid').onclick = event => {
  const catalogue = event.target.closest('[data-catalogue]');
  if (catalogue) { openCatalogue(searchSongs[catalogue.dataset.catalogue]); return; }
  const artist = event.target.closest('[data-artist-result], [data-artist-id]');
  if (artist) { openCatalogue(artist.dataset.artistId ? {browse_id:artist.dataset.artistId, title:artist.dataset.artistName} : searchSongs[artist.dataset.artistResult]); return; }
  const mv = event.target.closest('[data-find-mv]');
  if (mv) { const song = searchSongs[mv.dataset.findMv]; $('song-input').value = `${song.uploader} ${song.title}`; changeSearchMode('youtube'); return; }
  const preview = event.target.closest('[data-preview]');
  const add = event.target.closest('[data-add-video]');
  if (preview) previewVideo(searchSongs[preview.dataset.preview]);
  if (add) addVideo(searchSongs[add.dataset.addVideo]);
};
$('preview-add').onclick = () => addVideo(previewSong);
$('close-preview').onclick = () => $('video-preview').close();
$('video-preview').onclose = () => { $('preview-player').replaceChildren(); previewSong = null; };
$('video-preview').onclick = event => { if (event.target === $('video-preview')) $('video-preview').close(); };
$('add-form').onsubmit = async event => {
  event.preventDefault();
  if (composing) return;
  const url = $('song-input').value.trim();
  if (!/^https?:\/\//i.test(url)) { searchVideos(); if (!url) $('song-input').focus(); return; }
  $('submit-song').disabled = true;
  try {
    acceptState(await request('/v1/room/entries', {url, title:$('title-input').value.trim() || null}));
    $('song-input').value = ''; $('title-input').value = ''; editSearch();
    toast('已点歌');
  } catch (error) { toast(error.message); }
  finally { $('submit-song').disabled = false; }
};
$('song-grid').onclick = async event => {
  const button = event.target.closest('[data-song]');
  if (!button) return;
  const song = library.find(s => s.id === button.dataset.song);
  button.disabled = true;
  try { acceptState(await request('/v1/room/entries', song.job_id ? {job_id:song.job_id} : {url:song.url,title:song.title})); toast(`《${song.title}》已点歌`); }
  catch (error) { toast(error.message); }
  finally { button.disabled = false; }
};
$('queue-list').onclick = async event => {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  const id = button.closest('[data-entry]').dataset.entry;
  const action = button.dataset.action;
  if (['select','remove'].includes(action) && !(await activate())) return;
  if (action === 'up' || action === 'down') {
    const order = state.entries.map(e => e.id), index = order.indexOf(id), target = index + (action === 'up' ? -1 : 1);
    [order[index],order[target]] = [order[target],order[index]];
    await command('reorder', {order, revision:state.revision});
  } else {
    await command(action, {entry_id:id});
    if (action === 'select') switchView('player');
  }
};
$('play-toggle').onclick = togglePlay;
for (const [id, step] of [['pitch-down', -1], ['pitch-up', 1], ['pitch-reset', null]]) {
  $(id).onclick = () => {
    pitchRequest = pitchRequest.then(async () => {
      if (!(await activate())) return;
      if (!remote && player.pitchError) { toast(player.pitchError.message); return; }
      const pitch = step === null ? 0 : Math.max(-6, Math.min(6, (state.pitch_semitones || 0) + step));
      await command('pitch', {pitch_semitones:pitch});
    });
  };
}
$('screen-start').onclick = async () => { if (await activate()) { if (!state.playing) await command('play'); else await syncPlayback(); } };
async function nextSong() {
  if (skipping || !state.current_id) return;
  const entry_id = state.current_id;
  skipping = true; updateTransport();
  try { await activate(); await command('next', {entry_id}); }
  finally { skipping = false; updateTransport(); }
}
document.querySelectorAll('[data-next]').forEach(button => button.onclick = nextSong);
$('now-song').onclick = () => switchView('player');
for (const id of ['restart','transport-restart']) $(id).onclick = () => command('seek', {position:0});
document.querySelectorAll('[data-mix]').forEach(button => button.onclick = async () => {
  clearTimeout(levelTimer); levelTimer = null; pendingLevels = {};
  await levelRequest;
  await command('mix', {mix:button.dataset.mix});
});
document.querySelectorAll('[data-level]').forEach(input => input.oninput = () => {
  pendingLevels[input.dataset.level] = Number(input.value);
  renderMixer();
  if (!levelTimer) levelTimer = setTimeout(flushLevels, 80);
});
$('storage-form').onsubmit = async event => {
  event.preventDefault();
  $('storage-save').disabled = true;
  try {
    acceptState(await request('/v1/storage', {limit_gib:Number($('storage-limit').value)}));
    toast('缓存上限已保存，旧缓存会自动清理');
  } catch (error) { toast(error.message); }
  finally { $('storage-save').disabled = false; }
};
$('seek').oninput = () => { seeking = true; $('current-time').textContent = time(Number($('seek').value)); };
$('seek').onchange = async () => { await command('seek', {position:Number($('seek').value)}); seeking = false; };
$('volume').oninput = () => { player.setVolume(Number($('volume').value)); $('volume-value').textContent = Math.round(player.volume*100); };
$('mute').onclick = () => { $('volume').value = player.volume ? 0 : 0.8; $('volume').oninput(); };
async function fullscreen() {
  switchView('player');
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('player-stage').requestFullscreen(); }
  catch { toast('当前浏览器不支持全屏播放'); }
}
$('fullscreen').onclick = fullscreen;
$('exit-fullscreen').onclick = fullscreen;
// Move the same timeline so fullscreen keeps its clock, seek handlers and state.
const seekRow = $('seek').closest('.seek-row');
const seekHome = seekRow.parentElement;
document.addEventListener('fullscreenchange', () => {
  const target = document.fullscreenElement === $('player-stage') ? $('fullscreen-progress') : seekHome;
  target.append(seekRow);
});
document.addEventListener('keydown', event => {
  if ($('video-preview').open) return;
  if (['INPUT','TEXTAREA','BUTTON','SELECT'].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
  if (event.key.toLowerCase() === 'n') nextSong();
  if (event.key.toLowerCase() === 'f') fullscreen();
});
window.addEventListener('resize', () => switchView(view));
const events = new EventSource('/v1/room/events');
events.addEventListener('room', event => {
  $('connection-dot').classList.add('online');
  $('connection-text').textContent = remote ? '遥控模式' : '在线';
  acceptState(JSON.parse(event.data));
});
events.onerror = () => { $('connection-dot').classList.remove('online'); $('connection-text').textContent = '正在重连'; };
if (remote) {
  $('remote-link').textContent = '打开播放页 ↗'; $('remote-link').href = '/';
  document.querySelector('.volume-control').hidden = true;
  const remoteTick = () => {
    updateTime(state.position + (state.playing && state.started_at
      ? Math.max(0, Date.now()/1000 + clockOffset - state.started_at) : 0));
    requestAnimationFrame(remoteTick);
  };
  requestAnimationFrame(remoteTick);
}
updateTransport();
