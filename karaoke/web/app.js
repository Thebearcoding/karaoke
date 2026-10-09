import { Player } from './player.js';
import { mountPreview } from './preview.js';
import { isStaleRoomSnapshot, LocalPlayback, PlaybackTabs, queueToTop, preparationRetry } from './room-state.js';
import { createVideoSearch, createViewHistory, previewURL, searchLibrary, librarySongs, paginateSongs, filterSongs, playCount, orderResults, queueKey, queuePosition, historySuggestions } from './search.js';

const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const time = seconds => `${Math.floor(Math.max(0, seconds || 0)/60)}:${String(Math.floor(Math.max(0, seconds || 0)%60)).padStart(2,'0')}`;
let savedPlayback;
try { savedPlayback = JSON.parse(sessionStorage.getItem('karaokePlayback')); } catch {}
let tabStorage;
try { tabStorage = localStorage; } catch {}
const tabs = new PlaybackTabs(tabStorage, `${Date.now()}-${Math.random()}`, savedPlayback);
let playback = new LocalPlayback(tabs.read().playback);
if (!tabs.record.owner) tabs.write(playback.save(), {claim:true});
let tabChange = tabs.record.change;
let finishRetry, finishSending, finishDelay = 1000, queueSyncError = false, connected = false;
let roomState = {entries:[], revision:0}, receivedRoom = false, viewRevision = 0, lastPlaybackSave = 0;
const stages = {download:'下载音频', download_video:'下载视频', verify_video:'校验视频', decode_audio:'提取音频', decode_native:'准备音频母版', separate_music:'分离伴奏', separate_lead:'分离主唱', package_two_stems:'整理音轨', package_three_stems:'整理音轨', resample_vocals:'整理人声', transcribe:'识别歌词', prepare_lyrics:'整理歌词', align_words:'对齐歌词', align_phonemes:'对齐起唱时间', align_repeats:'检查重复段落', align_context:'校准字幕', validate_timing:'核对起唱时间', render:'生成字幕画面', mux_video:'合成字幕 MV'};
let state = {entries:[], current_id:null, playing:false, position:0, started_at:null, levels:{instrumental:1,backing_vocals:1,lead_vocals:0}, revision:0};
let library = [], filter = 'ready', view = 'library', clockOffset = 0, readySent = null, loading = false, loadError = '', seeking = false;
let toastTimer, librarySignature = '', queueSignature = '', historySignature = '', discoverySignature = '';
let queueTab = 'upcoming';
let libraryPage = 1;
let pitchRequest = Promise.resolve();
let searchTimer, searchQuery = '', searchSongs = [], previewSong = null, skipping = false, composing = false;
let videoSource = 'songs', searchOrder = 'relevance', searchSnapshot = null, catalogueSelection = null;
const addingVideos = new Set();
const videoSearch = createVideoSearch(renderVideoResults, fetch, query => searchLibrary(library, query));
const navigation = createViewHistory(history, () => ({query:$('song-input').value, title:$('title-input').value,
  filter, view, queueTab, libraryPage, historyQuery:$('history-query').value,
  videoSource, searchOrder, catalogueSelection, search:videoSearch.snapshot(), scroll:scrollY}), restoreView);
const current = () => playback.entry;

function renderRange(input) {
  const span = Number(input.max) - Number(input.min);
  const fill = `${span > 0 ? 100 * (Number(input.value) - Number(input.min)) / span : 0}%`;
  if (input.style.getPropertyValue('--range-fill') !== fill) input.style.setProperty('--range-fill', fill);
}

function renderMixer() {
  player.setLevels(state.levels);
  const level = state.levels.lead_vocals;
  $('lead-toggle').textContent = `主唱：${level > 0 ? '开' : '关'}`;
  $('lead-toggle').setAttribute('aria-pressed', String(level > 0));
  $('lead-value').textContent = `${Math.round(level * 100)}%`;
  $('lead-down').disabled = level <= 0;
  $('lead-up').disabled = level >= 1;
}

function renderStorage() {
  const storage = state.storage;
  if (!storage) return;
  const gib = bytes => (bytes/1024**3).toLocaleString('zh-CN', {maximumFractionDigits:2});
  const limited = storage.limit_bytes != null;
  $('storage-usage').textContent = limited ? `${gib(storage.used_bytes)} / ${gib(storage.limit_bytes)} GiB` : `${gib(storage.used_bytes)} GiB · 不限`;
  $('storage-meter').hidden = !limited;
  $('storage-meter').max = storage.limit_bytes || 1;
  $('storage-meter').value = storage.used_bytes;
  $('storage-note').hidden = storage.can_prepare;
  $('storage-limit').placeholder = limited ? '' : '不限';
  if (document.activeElement !== $('storage-limit')) $('storage-limit').value = limited ? storage.limit_bytes/1024**3 : '';
}

function toast(message, entry) {
  $('toast').textContent = message;
  if (entry) {
    const button = document.createElement('button');
    button.textContent = '查看待唱 →';
    button.onclick = () => showQueuedSong(entry.id);
    $('toast').append(button);
  }
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').hidden = true, entry ? 6500 : 3800);
}

async function request(path, body, timeout = 0) {
  const controller = new AbortController();
  const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
  try {
    const response = await fetch(path, {signal:controller.signal, ...(body ?
      {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {})});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求没有成功，请检查输入后重试');
    return data;
  } finally { clearTimeout(timer); }
}

async function command(action, extra = {}, claim = !['ready','buffering'].includes(action)) {
  try {
    if (['play','pause','seek','restart','next','lead','toggle_lead','pitch','ready','buffering'].includes(action) ||
        (action === 'remove' && extra.entry_id === state.current_id)) {
      if (!claim && !tabs.owns()) return null;
      if (extra.revision !== undefined && extra.revision !== state.revision) return null;
      const claimed = claim && claimPlayback();
      const session = tabs.record.session;
      if (claimed && playback.state.playing && action !== 'pause')
        await player.unlock();
      if (!tabs.owns(session)) { followPlayback(); return null; }
      playback.command(action, extra, state.entries, clockOffset);
      renderState();
      savePlayback();
      syncFinished();
      return state;
    }
    const updated = await request('/v1/room/commands', {action, ...extra});
    acceptState(updated);
    return updated;
  } catch (error) { toast(error.message); return null; }
}

function renderConnection() {
  $('connection-dot').classList.toggle('online', connected && !queueSyncError);
  $('connection-text').textContent = queueSyncError ? '队列待同步' : connected ? '在线' : '正在重连';
}

async function syncFinished() {
  if (finishSending || finishRetry || !tabs.owns() || !playback.pendingFinishes.length) return;
  const sending = finishSending = playback, session = tabs.record.session;
  const owns = () => playback === sending && tabs.owns(session);
  try {
    await sending.flushFinishes(body => request('/v1/room/commands', body, 10000), updated => {
      acceptState(updated);
      savePlayback();
      queueSyncError = false; finishDelay = 1000;
    }, owns);
  } catch {
    if (owns()) {
      queueSyncError = true;
      finishRetry = setTimeout(() => { finishRetry = null; syncFinished(); }, finishDelay);
      finishDelay = Math.min(30000, finishDelay * 2);
    }
  } finally {
    finishSending = null;
    renderConnection();
    if (playback !== sending) syncFinished();
  }
}

function retryFinished() {
  clearTimeout(finishRetry); finishRetry = null;
  syncFinished();
}

function switchView(next, {scroll = true} = {}) {
  view = next;
  const mobile = matchMedia('(max-width:1100px)').matches;
  $('library-view').hidden = next !== 'library';
  $('player-view').hidden = next !== 'player';
  $('queue-panel').classList.toggle('mobile-open', next === 'queue');
  if (next === 'queue' && !mobile) { view = 'library'; $('library-view').hidden = false; }
  document.body.dataset.view = view;
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('selected', button.dataset.view === view);
    button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false');
  });
  if (scroll) window.scrollTo(0, 0);
  navigation.save();
}

function restoreView(saved) {
  clearTimeout(scrollSave);
  clearVideoSearch();
  $('song-input').value = saved.query || '';
  $('title-input').value = saved.title || '';
  filter = [...document.querySelectorAll('[data-filter]')].some(button => button.dataset.filter === saved.filter) ? saved.filter : 'ready';
  libraryPage = saved.libraryPage || 1;
  videoSource = saved.videoSource || 'songs';
  searchOrder = saved.searchOrder || 'relevance';
  queueTab = saved.queueTab === 'history' ? 'history' : 'upcoming';
  $('history-query').value = saved.historyQuery || '';
  renderQueue();
  catalogueSelection = saved.catalogueSelection || null;
  searchQuery = saved.search ? saved.query.trim() : '';
  document.querySelectorAll('[data-filter]').forEach(button => button.classList.toggle('selected', button.dataset.filter === filter));
  renderLibrary();
  videoSearch.restore(saved.search);
  switchView(saved.view || 'library', {scroll:false});
  requestAnimationFrame(() => window.scrollTo(0, saved.scroll || 0));
}

function renderLibrary() {
  const query = $('song-input').value.trim();
  const isURL = /^https?:\/\//i.test(query);
  $('title-row').hidden = !isURL;
  $('submit-song').innerHTML = `${isURL ? '点歌' : '搜索'} ${icon('arrow')}`;
  const search = isURL ? '' : query.toLocaleLowerCase();
  const songs = librarySongs(library, search ? 'all' : filter, search);
  const page = paginateSongs(songs, libraryPage);
  if (receivedRoom) libraryPage = page.page;
  const preparedView = filter === 'ready' && !search;
  $('library-count').textContent = `${songs.length} 首`;
  $('library-heading').textContent = search ? '曲库里的歌' : preparedView ? '现在就能唱' : '曲库';
  document.querySelectorAll('[data-filter]').forEach(button => {
    button.classList.toggle('selected', button.dataset.filter === filter);
    button.setAttribute('aria-pressed', String(button.dataset.filter === filter));
    const count = button.querySelector('span');
    if (count) count.textContent = librarySongs(library, button.dataset.filter).length;
  });
  $('local-library').hidden = Boolean(searchQuery || (search && !songs.length));
  document.querySelector('#local-library .filters').hidden = Boolean(search);
  document.querySelectorAll('.library-pagination').forEach(nav => {
    nav.hidden = page.pages < 2;
    const select = nav.querySelector('select');
    if (select.options.length !== page.pages)
      select.innerHTML = Array.from({length:page.pages}, (_, i) => `<option value="${i+1}">${i+1} / ${page.pages}</option>`).join('');
    select.value = String(page.page);
    nav.querySelector('[data-library-step="-1"]').disabled = page.page === 1;
    nav.querySelector('[data-library-step="1"]').disabled = page.page === page.pages;
  });
  const signature = JSON.stringify([page.songs.map(({stage_progress, active_stages, ...song}) => song), page.page, filter, query]);
  if (signature !== librarySignature) {
    librarySignature = signature;
    $('song-grid').innerHTML = page.songs.map((song, index) => songRow({...song, provider:'local', thumbnail:song.poster, uploader:song.artist}, page.offset+index, {inLibrary:true, local:true})).join('');
  } else {
    updatePreparationSlots($('song-grid'), page.songs);
  }
  $('search-empty').hidden = songs.length > 0;
  $('search-empty').querySelector('h3').textContent = preparedView ? '还没有可以直接唱的歌' : '这里还没有歌曲';
  $('library-list').hidden = !songs.length;
  renderDiscovery();
  updateAddButtons();
}

function renderDiscovery() {
  const recent = historySuggestions(state.history || []).recent.slice(0,4);
  $('home-discovery').hidden = !recent.length || Boolean($('song-input').value.trim() || searchQuery);
  const cached = new Map(library.map(song => [queueKey(song), song]));
  const signature = JSON.stringify([recent, recent.map(song => {
    return cached.get(queueKey(song))?.poster;
  })]);
  if (signature === discoverySignature) return;
  discoverySignature = signature;
  $('recent-songs').innerHTML = recent.map(song => {
    const local = cached.get(queueKey(song));
    return `<div class="recent-song"><button class="recent-preview" data-preview-history="${escape(song.id)}" aria-label="预览：${escape(song.title)}">
      ${local?.poster ? `<img src="${escape(local.poster)}" alt="" loading="lazy">` : `<span class="recent-initial">${escape(song.title.slice(0,1))}</span>`}
      <span class="recent-info"><strong>${escape(song.title)}</strong><span>${escape(song.artist)}</span></span></button>
      <button class="music-add" data-recent-song="${escape(song.id)}" aria-label="再点：${escape(song.title)}">${icon('plus')}<span>再点</span></button></div>`;
  }).join('');
}

function renderVideoResults(providers, mode = videoSource) {
  videoSource = mode;
  if (mode === 'artist' && providers.artist.selection) catalogueSelection = {...providers.artist.selection, returnMode:'artists'};
  else if (mode === 'artists') catalogueSelection = null;
  searchSnapshot = providers;
  $('library-view').classList.add('searching');
  $('video-results').hidden = false;
  $('local-library').hidden = true;
  $('home-discovery').hidden = true;
  const info = providers[videoSource];
  if (!info) return;
  const artists = videoSource === 'artists', album = videoSource === 'album', catalogue = album || videoSource === 'artist';
  const metadata = info.metadata || catalogueSelection || {};
  const catalogueHeading = metadata.catalogue_section === 'videos' ? '歌手视频' : metadata.catalogue_section === 'releases' ? '歌手作品' : '热门歌曲';
  $('clear-search').textContent = catalogue || (artists && providers.artist?.selection) ? '返回上一级' : '返回曲库';
  const popularityAvailable = info.results.some(song => playCount(song.views) !== null);
  if (searchOrder === 'popular' && !popularityAvailable) searchOrder = 'relevance';
  $('search-order').value = searchOrder;
  $('search-order').querySelector('[value="popular"]').disabled = !popularityAvailable;
  $('search-order').querySelector('[value="relevance"]').textContent = album ? '专辑顺序' : '默认排序';
  $('search-order-label').hidden = artists || info.loading || !info.results.length;
  $('video-results').classList.toggle('is-catalogue', catalogue);
  $('search-heading').textContent = catalogue ? metadata.title || (album ? '专辑曲目' : catalogueHeading) : artists ? '找一位歌手' : '搜索结果';
  $('search-order-note').textContent = artists ? '选一位歌手，看看有哪些歌想唱' :
    searchOrder === 'popular' ? '已加载结果按播放量排序 · YouTube 提供' : searchOrder === 'ready' ? '已加载结果中，已准备歌曲排在前面' :
    searchOrder === 'title' ? '已加载结果按歌名排序' : album ? 'YouTube Music 专辑曲目顺序' :
    videoSource === 'local' ? '本地曲库 · 现场版靠后' : videoSource === 'songs' ? '歌曲优先 · 现场版靠后' : 'YouTube 原顺序 · 现场版靠后';
  $('search-order-note').hidden = searchOrder !== 'popular';
  $('search-order').title = $('search-order-note').textContent;
  $('search-status').innerHTML = `<span class="${info.error ? 'search-error' : ''}">${info.loading ? '正在查找…' : info.error ? escape(info.error) : info.results.length ? `${videoSource === 'local' ? '' : '已加载 '}${info.results.length} ${artists ? '位歌手' : videoSource === 'youtube' || metadata.catalogue_section === 'videos' ? '个视频' : videoSource === 'songs' || metadata.catalogue_section === 'releases' ? '项' : '首'}${catalogue && info.metadata?.complete === false ? ' · 歌手页精选' : ''}` : catalogue ? '这个页面暂未提供可点播作品' : '没有找到，试试另一个关键词或搜索类型'}${info.error ? '<button data-search-retry>重试</button>' : ''}</span>`;
  $('artist-summary').hidden = !catalogue;
  const otherArtists = !album && (providers.artists?.results.length > 1 || providers.artists?.nextCursor);
  if (catalogue) $('artist-summary').innerHTML = `${metadata.thumbnail ? `<img src="${escape(metadata.thumbnail)}" alt="" referrerpolicy="no-referrer">` : ''}<div><p>${escape(metadata.monthly_listeners || metadata.artist || (album ? '专辑曲目' : catalogueHeading))} · YouTube Music</p>${otherArtists ? '<button class="text-button" data-other-artists>其他歌手 →</button>' : ''}</div>`;
  renderSearchMore(info);
  searchSongs = orderResults(info.results.map(song => ({...song,
    status:library.find(item => item.url === song.url)?.status || song.status})), album && searchOrder === 'relevance' ? 'album' : searchOrder);
  const videos = videoSource === 'youtube', local = videoSource === 'local';
  $('search-list').className = artists ? '' : `search-list${videos ? ' video-list' : local ? ' local-list' : ''}`;
  $('search-list').classList.toggle('has-albums', !videos && !local && searchSongs.some(song => song.album));
  $('search-list').setAttribute('role', artists ? 'group' : 'table');
  $('search-list').setAttribute('aria-label', artists ? '歌手列表' : videos ? '视频列表' : '歌曲列表');
  $('music-columns').hidden = artists || !searchSongs.length;
  $('music-title-heading').textContent = videos ? '视频' : videoSource === 'songs' ? '名称' : '歌曲';
  $('music-artist-heading').textContent = videos ? '频道' : '歌手';
  $('music-alternative-heading').textContent = videos ? '来源' : '找 MV';
  $('video-grid').className = artists ? 'artist-grid' : 'music-list';
  if (artists) $('video-grid').removeAttribute('role');
  else $('video-grid').setAttribute('role', 'rowgroup');
  const rows = searchSongs.map((song, index) => songRow(song, index, {artists, videos, local}));
  updateSearchRows(rows);
  document.querySelectorAll('[data-video-source]').forEach(button => {
    const selected = button.dataset.videoSource === (album ? 'songs' : catalogue ? 'artists' : videoSource);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  updateAddButtons();
  navigation.save();
}

function songRow(song, index, {artists = false, videos = false, local = false, inLibrary = false} = {}) {
  const thumbnail = song.thumbnail ? `<img src="${escape(song.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : `<span class="music-initial">${escape(song.title.slice(0,1))}</span>`;
  if (artists) return `<button class="artist-card" data-artist-result="${index}">${thumbnail}<strong>${escape(song.title)}</strong><span>热门歌曲 ${icon('arrow')}</span></button>`;
  const playable = !song.kind || song.kind === 'song';
  const catalogueAction = ['artist','album'].includes(song.kind) ? `data-catalogue="${index}"` : '';
  const open = playable ? (inLibrary ? `data-preview-song="${escape(song.id)}"` : `data-preview="${index}"`) : catalogueAction || `href="${escape(song.url)}" target="_blank" rel="noopener noreferrer"`;
  const tag = playable || catalogueAction ? 'button' : 'a';
  const action = playable ? `<button class="music-add" ${inLibrary ? `data-song="${escape(song.id)}"` : `data-add-video="${index}"`} aria-label="点歌：${escape(song.title)}">${icon('plus')}<span>点歌</span></button>` :
    `<${tag} class="music-add" ${open}>${song.kind === 'album' ? '查看专辑' : catalogueAction ? '查看歌曲' : '打开'} ${icon('arrow')}</${tag}>`;
  const credits = song.artists?.length ? song.artists.map(a => a.id ? `<button data-artist-id="${escape(a.id)}" data-artist-name="${escape(a.name)}">${escape(a.name)}</button>` : escape(a.name)).join(' · ') : local && song.uploader ? `<button data-search-artist="${escape(song.uploader)}">${escape(song.uploader)}</button>` : escape(song.uploader || '—');
  const kind = song.kind === 'album' ? '专辑' : song.kind === 'artist' ? '歌手' : song.version || '';
  const status = inLibrary && song.status === 'failed' ? '<span class="library-status is-failed">准备失败</span>' : '';
  const meta = kind || status ? `<div class="music-meta">${kind ? `<span${song.kind === 'album' ? ' class="music-kind"' : ''}>${escape(kind)}</span>` : ''}${status}</div>` : '';
  return `<article class="music-row" ${inLibrary ? `data-card="${escape(song.id)}"` : `data-result="${index}"`} role="row"><span class="music-rank" role="cell">${String(index+1).padStart(2,'0')}</span>
    <div class="music-song" role="cell"><${tag} class="music-cover" ${open} aria-label="${playable ? '预览' : '打开'}：${escape(song.title)}">${thumbnail}<span>${icon(playable ? 'play' : 'arrow')}</span></${tag}>
      <div class="music-title"><h3><${tag} ${open} title="${escape(song.title)}">${escape(song.title)}</${tag}></h3>${meta}${inLibrary && song.status === 'running' ? `<div class="preparation-slot">${preparationHTML(song)}</div>` : ''}</div></div>
    <div class="music-credits" role="cell" title="${escape(song.artists?.map(a => a.name).join(' · ') || song.uploader || '')}">${inLibrary ? `<div class="song-info-row">${credits}${preparationElapsedHTML(song)}</div>` : credits}</div>
    <div class="music-album" role="cell" title="${escape(song.album || '')}">${escape(song.album || '—')}</div>
    <div class="music-facts"><div class="music-plays${playCount(song.views) === null ? ' is-empty' : ''}" role="cell">${playCount(song.views) !== null ? `${formatPlays(song.views)}<small>播放</small>` : '—'}</div>
    <div class="music-duration${!song.duration ? ' is-empty' : ''}" role="cell">${song.duration ? time(song.duration) : '—'}</div></div>
    <div class="music-actions" role="cell">${action}</div>
    <div class="music-alternative" role="cell">${(local || song.provider === 'youtube_music') && playable ? `<button class="find-mv" ${inLibrary ? `data-find-mv-song="${escape(song.id)}"` : `data-find-mv="${index}"`} aria-label="找 MV：${escape(song.title)}">找 MV</button>` : videos ? `<a class="find-mv" href="${escape(song.url)}" target="_blank" rel="noopener noreferrer">YouTube ↗</a>` : ''}</div></article>`;
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
  navigation.push();
  if ((item.kind || 'artist') === 'artist' && searchSnapshot?.artists?.results.some(artist => artist.browse_id === item.browse_id)) {
    searchOrder = 'relevance';
    videoSearch.selectArtist(item.browse_id);
    window.scrollTo(0, 0);
    return;
  }
  const query = $('song-input').value.trim();
  clearVideoSearch();
  const returnMode = catalogueSelection?.returnMode || videoSource;
  catalogueSelection = {...item, returnMode};
  videoSource = item.kind || 'artist'; searchOrder = 'relevance';
  searchQuery = query;
  videoSearch.search(item.browse_id, videoSource);
  window.scrollTo(0, 0);
}

function changeSearchMode(mode, query) {
  navigation.push();
  if (query !== undefined) $('song-input').value = query;
  videoSource = mode; searchOrder = 'relevance'; catalogueSelection = null;
  switchView('library');
  searchVideos(true);
  window.scrollTo(0, 0);
}

function updateAddButtons() {
  document.querySelectorAll('[data-add-video], [data-song], #preview-add, #submit-song').forEach(button => {
    const form = button.id === 'submit-song';
    const song = form ? {url:$('song-input').value.trim(), title:$('title-input').value.trim()} :
      button.id === 'preview-add' ? previewSong : button.hasAttribute('data-song')
        ? library.find(item => item.id === button.dataset.song) : searchSongs[button.dataset.addVideo];
    if (!song) return;
    if (form && !/^https?:\/\//i.test(song.url)) { button.disabled = false; button.classList.remove('is-queued'); button.removeAttribute('aria-label'); return; }
    const position = queuePosition(song, state.entries), adding = addingVideos.has(queueKey(song));
    const label = position ? '已点' : adding ? '点歌中…' : song.available === false ? '暂不可用' : '点歌';
    button.disabled = Boolean(position || adding || song.available === false);
    button.classList.toggle('is-queued', position > 0);
    button.setAttribute('aria-label', `${label}${position ? `，第${position}首` : ''}：${song.title || song.url}`);
    const markup = `${position ? '' : icon('plus')}<span>${label}${position ? `<small>第${position}首</small>` : ''}</span>`;
    if (button.innerHTML !== markup) button.innerHTML = markup;
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
  if (force && query === searchQuery && videoSource === 'artist' && searchSnapshot?.artist?.selection) {
    videoSearch.selectArtist(searchSnapshot.artist.selection.browse_id, true);
    return;
  }
  clearVideoSearch();
  searchQuery = query;
  if (query.length > 160) { toast('搜索关键词请控制在 160 字以内'); return; }
  videoSearch.search(['artist','album'].includes(videoSource) ? catalogueSelection.browse_id : query, videoSource);
}

function editSearch() {
  libraryPage = 1;
  if (['artist','album'].includes(videoSource)) { videoSource = catalogueSelection?.returnMode || 'songs'; catalogueSelection = null; }
  clearVideoSearch();
  renderLibrary();
  navigation.save();
  if (!composing) searchTimer = setTimeout(searchVideos, 700);
}

async function addVideo(song, {title, repeat = false} = {}) {
  if (!song || addingVideos.has(queueKey(song)) || song.available === false) return;
  const queued = !repeat && state.entries[queuePosition(song, state.entries)-1];
  if (queued) return;
  const key = queueKey(song);
  addingVideos.add(key); updateAddButtons();
  $('preview-error').hidden = true;
  try {
    const cached = library.find(item => queueKey(item) === key && item.job_id);
    const previous = new Set(state.entries.map(entry => entry.id));
    const updated = await request('/v1/room/entries', {if_absent:!repeat, source_thumbnail:song.thumbnail, ...(cached ? {job_id:cached.job_id} :
      {url:song.url, ...(title !== undefined ? {title} : {source_title:song.title})})});
    acceptState(updated);
    if ($('video-preview').open) $('video-preview').close();
    const added = updated.entries.find(entry => !previous.has(entry.id) && queueKey(entry) === key);
    const position = added ? updated.entries.findIndex(entry => entry.id === added.id)+1 : queuePosition(song, updated.entries);
    toast(`已点${position ? ` · 第${position}首` : ''}`, updated.entries[position-1]);
  } catch (error) {
    toast(error.message);
    $('preview-error').textContent = error.message;
    $('preview-error').hidden = false;
  } finally { addingVideos.delete(key); updateAddButtons(); }
}

function showQueuedSong(id) {
  if (!state.entries.some(song => song.id === id)) return;
  $('toast').hidden = true;
  if ($('video-preview').open) $('video-preview').close();
  queueTab = 'upcoming'; renderQueue(); switchView('queue');
  document.querySelector(`[data-entry="${CSS.escape(id)}"]`)?.scrollIntoView({block:'nearest'});
}

let closeCachedPreview = null;
function previewVideo(song) {
  if (current() && queuePosition(song, [current()])) { switchView('player'); return; }
  previewSong = song;
  $('preview-title').textContent = song.title;
  $('preview-source').href = song.url;
  $('preview-source').textContent = `在 ${song.url.includes('youtube.com') ? 'YouTube' : 'B站'} 观看 ↗`;
  $('preview-error').hidden = true;
  const cached = library.find(item => item.url === song.url && item.status === 'completed' && item.job_id);
  closeCachedPreview?.(); closeCachedPreview = null;
  if (cached) {
    closeCachedPreview = mountPreview($('preview-player'), cached, error => {
      $('preview-error').textContent = error.message;
      $('preview-error').hidden = false;
    });
    $('video-preview').showModal();
    updateAddButtons();
    return;
  }
  const iframe = document.createElement('iframe');
  iframe.src = previewURL(song, location.origin);
  iframe.title = song.title;
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  iframe.allowFullscreen = true;
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';
  $('preview-player').replaceChildren(iframe);
  updateAddButtons();
  $('video-preview').showModal();
}

function queuedStatus(entry) {
  return entry.next_retry_at ? '等待自动重试' : '等待准备';
}

function statusText(entry) {
  if (entry.status === 'failed') return entry.error?.message || '准备失败，请重试';
  if (entry.status === 'completed') return '';
  if (entry.status === 'queued') return queuedStatus(entry);
  return (entry.active_stages || []).map(s => stages[s] || '正在准备').join(' · ') || '开始准备';
}

function preparationElapsedHTML(entry) {
  if (entry?.status !== 'running') return '';
  const started = Date.parse(entry.preparation_started_at)/1000;
  return Number.isFinite(started) ? `<span class="preparation-elapsed" data-preparation-start="${started}">${preparationTime(started)}</span>` : '';
}

function preparationHTML(entry) {
  if (entry?.status !== 'running') return '';
  return `<div class="preparation">${(entry.active_stages?.length ? entry.active_stages : ['starting']).map(stage => {
    const p = entry.stage_progress?.[stage] || {};
    const measured = Number.isFinite(p.completed) && Number.isFinite(p.total) && p.total > 0;
    const percent = measured ? Math.min(100, Math.floor(100*p.completed/p.total)) : null;
    const label = `${stages[stage] || '开始准备'}${p.detail ? ` · ${p.detail}` : ''}`;
    const amount = percent !== null ? `${percent}%` : p.unit === 'bytes' && Number.isFinite(p.completed) ? `${(p.completed/1024**2).toFixed(1)} MB` : '';
    return `<div class="preparation-stage"><div class="preparation-label"><span title="${escape(label)}">${escape(label)}</span><span class="preparation-metrics">${amount}</span></div><progress max="100" ${percent === null ? '' : `value="${percent}"`} aria-label="${escape(label)}"></progress></div>`;
  }).join('')}</div>`;
}

function preparationTime(started) {
  return `已用时 ${time(Date.now()/1000 + clockOffset - started)}`;
}

setInterval(() => {
  for (const label of document.querySelectorAll('[data-preparation-start]')) {
    const text = preparationTime(Number(label.dataset.preparationStart));
    if (label.textContent !== text) label.textContent = text;
  }
}, 1000);

function updatePreparationSlots(container, songs) {
  for (const element of container.children) {
    const slot = element.querySelector('.preparation-slot');
    if (slot) slot.innerHTML = preparationHTML(songs.find(s => s.id === (element.dataset.entry || element.dataset.card)));
  }
}

function renderQueue() {
  document.querySelectorAll('[data-queue-tab]').forEach(button => {
    const selected = button.dataset.queueTab === queueTab;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $('upcoming-panel').hidden = queueTab !== 'upcoming';
  $('history-panel').hidden = queueTab !== 'history';
  renderHistory();
  const signature = JSON.stringify([state.entries.map(({stage_progress, active_stages, ...entry}) => entry), state.current_id, state.playing, playback.pendingFinishes]);
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
    const finishing = playback.isFinished(entry.id);
    const retry = preparationRetry(entry);
    const status = entry.status === 'running' || !statusText(entry) ? '' : `<span class="entry-status ${entry.status === 'queued' ? 'preparing' : ''}">${escape(statusText(entry))}</span>`;
    return `<article class="queue-entry ${active ? 'current' : ''}" data-entry="${escape(entry.id)}"><span class="entry-number">${active && state.playing ? '♫' : String(i+1).padStart(2,'0')}</span>
      <button class="entry-cover" data-action="preview" aria-label="${active ? '查看播放' : '预览'}：${escape(entry.title)}">${entry.poster ? `<img class="entry-art" src="${escape(entry.poster)}" alt="">` : `<span class="entry-art">♪</span>`}</button>
      <div class="entry-info"><h3 class="entry-title"><button data-action="preview">${escape(entry.title)}</button></h3><div class="song-info-row"><p class="entry-artist">${escape(entry.artist || 'YouTube 视频')}</p>${finishing ? '<span class="entry-status">等待同步移除</span>' : entry.status === 'failed' ? '' : preparationElapsedHTML(entry) || status}</div>${entry.status === 'running' ? `<div class="preparation-slot">${preparationHTML(entry)}</div>` : entry.status === 'failed' ? `<p class="entry-error">${escape(statusText(entry))}</p>` : ''}</div>
      <div class="entry-tools">${retry && !finishing ? `<button data-action="${retry.action}" aria-label="${retry.label}：${escape(entry.title)}">${icon('restart')}${retry.label}</button>` : ''}<button data-action="top" aria-label="置顶：${escape(entry.title)}" ${!finishing && queueToTop(state.entries, entry.id, state.playing ? state.current_id : null) ? '' : 'disabled'}>${icon('up')}置顶</button><button data-action="remove" aria-label="移除：${escape(entry.title)}" ${finishing ? 'disabled' : ''}>${icon('trash')}移除</button></div></article>`;
  }).join('');
}

function renderHistory() {
  const history = state.history || [];
  const songs = filterSongs(history, $('history-query').value);
  $('history-count').textContent = String(history.length).padStart(2,'0');
  $('history-empty').hidden = songs.length > 0;
  $('history-empty').querySelector('h3').textContent = history.length ? '没有找到匹配的已唱歌曲' : '还没有唱过的歌';
  const posters = new Map(library.map(song => [song.url, song.poster]));
  const signature = JSON.stringify(songs.map(song => [song, posters.get(song.url)]));
  if (signature === historySignature) return;
  historySignature = signature;
  const date = new Intl.DateTimeFormat('zh-CN', {month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit'});
  $('history-list').innerHTML = songs.map((song, index) => {
    const poster = posters.get(song.url);
    return `<article class="queue-entry history-entry" data-history-entry="${escape(song.id)}"><span class="entry-number">${String(index+1).padStart(2,'0')}</span>
      ${poster ? `<img class="entry-art" src="${escape(poster)}" alt="">` : `<span class="entry-art">♪</span>`}
      <div class="entry-info"><h3 class="entry-title">${escape(song.title)}</h3><p class="entry-artist">${escape(song.artist)}</p></div>
      <div class="entry-tools"><time datetime="${new Date(song.finished_at*1000).toISOString()}">${escape(date.format(song.finished_at*1000))}</time><button data-repeat-song ${song.url ? '' : 'disabled'} aria-label="再点：${escape(song.title)}">${icon('plus')}再点</button></div></article>`;
  }).join('');
}

function updateTransport() {
  const song = current();
  const idleTitle = state.playing ? '等待下一首' : '未播放';
  $('now-title').textContent = song?.title || idleTitle;
  $('now-artist').textContent = song?.artist || '';
  $('now-art').style.backgroundImage = song?.poster ? `url("${song.poster}")` : '';
  $('now-art').querySelector('span').hidden = Boolean(song?.poster);
  $('player-title').textContent = song?.title || idleTitle;
  $('player-artist').textContent = song?.artist || '';
  $('player-elapsed').innerHTML = preparationElapsedHTML(song);
  const playingHere = tabs.owns() && state.playing && player.unlocked;
  $('play-icon').setAttribute('href', playingHere ? '#i-pause' : '#i-play');
  $('play-toggle').setAttribute('aria-label', playingHere ? '暂停' : '播放');
  $('play-toggle').disabled = !song && !state.playing;
  document.querySelectorAll('[data-next]').forEach(button => button.disabled = skipping || !state.current_id);
  $('transport-restart').disabled = !song;
  $('seek').disabled = !song || song.status !== 'completed';
  $('seek').max = song?.duration || 1;
  $('duration').textContent = time(song?.duration);
  $('playing-dot').hidden = !state.playing;
  const pitch = state.pitch_semitones || 0;
  $('pitch-value').textContent = pitch ? `${pitch > 0 ? '+' : '−'}${Math.abs(pitch)}` : '原调';
  $('pitch-value').setAttribute('aria-label', `当前${pitch ? `${pitch > 0 ? '升' : '降'} ${Math.abs(pitch)} 个半音` : '原调'}`);
  $('pitch-down').disabled = pitch <= -6;
  $('pitch-up').disabled = pitch >= 6;
  renderMixer();
  updateCover();
}

function updateCover() {
  const song = current();
  const startingVideo = Boolean(player.pendingStart || player.videoPreparation);
  if (player.anchor === null || player.entry?.id !== song?.id)
    $('opening-countdown').toggleAttribute('hidden', true);
  $('screen-cover').toggleAttribute('hidden', Boolean(song?.status === 'completed' && !loading && !startingVideo && !loadError && player.unlocked && player.ready));
  const status = loadError || (!song ? state.playing ? '准备好后自动接唱' : '请先点歌' : song.status !== 'completed' ? statusText(song) : startingVideo ? '加载画面…' : loading ? '加载音轨…' : '');
  if ($('screen-status').textContent !== status) $('screen-status').textContent = status;
  $('screen-status').toggleAttribute('hidden', !status || song?.status === 'running');
  $('screen-start').toggleAttribute('hidden', !song || song.status !== 'completed' || loading || startingVideo);
  const label = loadError ? '重试加载 ' : '开始唱 ';
  if ($('screen-start').firstChild.textContent !== label) $('screen-start').firstChild.textContent = label;
  const retry = preparationRetry(song);
  $('screen-retry').toggleAttribute('hidden', !retry || retry.action !== 'retry' || !state.entries.some(e => e.id === song.id));
}

let countdownFrame = '';
function updateTime(position) {
  if (Date.now() - lastPlaybackSave > 1000) savePlayback(position);
  const seek = $('seek');
  const value = Math.min(Number(seek.max), Math.round(Math.max(0, position) * 10) / 10);
  // The slider already displays tenths and the label displays seconds. Keep
  // their DOM unchanged between visible updates; audio keeps its precise clock.
  if (!seeking && Number(seek.value) !== value) seek.value = value;
  renderRange(seek);
  const label = time(Math.max(0, position));
  if ($('current-time').textContent !== label) $('current-time').textContent = label;
  const opening = current()?.opening_countdown;
  const overlay = $('opening-countdown');
  const visible = position < 0 && opening && state.playing && player.anchor !== null;
  overlay.toggleAttribute('hidden', !visible);
  if (visible) {
    const remaining = Math.max(0, Math.min(opening.dots,
      Math.ceil((opening.onset-opening.settle_seconds-position)/opening.step_seconds-1e-9)));
    const signature = JSON.stringify([opening, remaining]);
    if (signature !== countdownFrame) {
      countdownFrame = signature;
      overlay.setAttribute('viewBox', `0 0 ${opening.width} ${opening.height}`);
      // Match the frozen first frame's rendered dots; the video takes over at t=0.
      overlay.replaceChildren(...Array.from({length:opening.dots}, (_, i) => {
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        for (const [name, value] of Object.entries({cx:opening.width/2+(i-(opening.dots-1)/2)*opening.spacing,
          cy:opening.y, r:opening.radius, fill:i >= opening.dots-remaining ? opening.color : 'transparent',
          stroke:i >= opening.dots-remaining ? opening.color : '#f6f4eca0', 'stroke-width':2})) dot.setAttribute(name, value);
        return dot;
      }));
    }
  } else countdownFrame = '';
  updateCover();
}

const player = new Player($('video'), updateTime, id => command('next', {entry_id:id}, false));
let bufferingRequest;
player.onBuffering = (position, startedAt, entryId = player.entry?.id) => {
  const token = `${entryId}:${startedAt}`;
  if (bufferingRequest === token) return;
  bufferingRequest = token;
  loading = true;
  updateCover();
  command('buffering', {entry_id:entryId, position, playback_started_at:startedAt})
    .finally(() => { if (bufferingRequest === token) bufferingRequest = null; });
};
player.onError = error => {
  if (!tabs.owns()) return;
  loadError = error.message || '视频播放失败，请重试';
  loading = false;
  updateCover(); toast(loadError);
};

async function syncPlayback() {
  const song = current();
  if (!tabs.owns() || !song || song.status !== 'completed') {
    player.clear(); readySent = null; loading = false; loadError = '';
    updateCover(); return;
  }
  const revision = state.revision;
  try {
    if (!state.playing || !state.started_at) player.stop();
    const position = state.position + (state.playing && state.started_at ?
      Math.max(0, Date.now()/1000 - state.started_at) : 0);
    if (state.playing && state.started_at && player.unlocked && (!player.matches(song) || !player.hasAudio(position))) {
      player.stop();
      player.onBuffering(position, state.started_at, song.id);
      return;
    }
    loading = !player.hasAudio(position) || !player.matches(song);
    updateCover();
    await player.load(song, position);
    if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision || !player.matches(song)) return;
    if (state.playing && !state.started_at && player.unlocked) {
      const preparing = player.prepareVideo(position);
      updateCover();
      await preparing;
      if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision) return;
    }
    loading = false; loadError = '';
    if (state.playing && !state.started_at && player.unlocked && readySent !== song.id) {
      readySent = song.id;
      if (!await command('ready', {entry_id:song.id, revision})) readySent = null;
    }
    if (!state.playing || state.started_at) readySent = null;
    player.apply(state, 0);
    updateTransport();
  } catch (error) {
    if (!tabs.owns() || state.current_id !== song.id || state.revision !== revision) return;
    player.unlocked = false;
    loading = false; loadError = error.message;
    updateCover(); toast(loadError);
  }
}

function acceptState(updated) {
  if (isStaleRoomSnapshot(updated, roomState)) return;
  roomState = updated;
  receivedRoom = true;
  clockOffset = updated.server_time - Date.now()/1000;
  renderState();
  syncFinished();
}

function renderState() {
  if (!receivedRoom) return;
  const wasReady = current()?.status === 'completed';
  const previous = current()?.id;
  state = {...playback.reconcile(roomState), revision:++viewRevision};
  if (current()?.id !== previous) savePlayback();
  connectEvents();
  library = roomState.library || [];
  renderLibrary();
  videoSearch.refreshLocal();
  $('screen-preparation').innerHTML = preparationHTML(current());
  renderQueue(); updateTransport(); renderStorage();
  updateAddButtons();
  if (!wasReady && state.playing && current()?.status === 'completed') switchView('player');
  if (!player.anchor) updateTime(state.position);
  syncPlayback();
}

async function activate() {
  loadError = '';
  claimPlayback();
  const session = tabs.record.session;
  try { await player.unlock(); }
  catch { toast('浏览器还没有启用声音，请再点一次播放'); return false; }
  if (!tabs.owns(session)) { followPlayback(); return false; }
  return true;
}

async function togglePlay() {
  const wasUnlocked = tabs.owns() && player.unlocked;
  if (!(await activate())) return;
  if (state.playing && !wasUnlocked) { await syncPlayback(); return; }
  await command(state.playing ? 'pause' : 'play');
}

document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => switchView(button.dataset.view));
document.querySelector('.brand').onclick = event => {
  event.preventDefault();
  navigation.push();
  $('song-input').value = ''; $('title-input').value = '';
  filter = 'ready'; libraryPage = 1; videoSource = 'songs'; searchOrder = 'relevance'; catalogueSelection = null;
  clearVideoSearch(); renderLibrary(); switchView('library');
  $('song-input').focus({preventScroll:true});
};
document.querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
  filter = button.dataset.filter;
  libraryPage = 1;
  document.querySelectorAll('[data-filter]').forEach(b => b.classList.toggle('selected', b === button));
  renderLibrary();
  navigation.save();
});
function changeLibraryPage(page) {
  navigation.push();
  libraryPage = page;
  renderLibrary();
  window.scrollTo(0, 0);
  navigation.save();
}
document.querySelectorAll('[data-library-step]').forEach(button => button.onclick = () =>
  changeLibraryPage(libraryPage + Number(button.dataset.libraryStep)));
document.querySelectorAll('.library-pagination select').forEach(select => select.onchange = () =>
  changeLibraryPage(Number(select.value)));
$('song-input').oninput = () => { if (view !== 'library') switchView('library'); editSearch(); };
document.querySelectorAll('[data-video-source]').forEach(button => button.onclick = () => changeSearchMode(button.dataset.videoSource));
$('search-order').onchange = () => { searchOrder = $('search-order').value; if (searchSnapshot) renderVideoResults(searchSnapshot); };
$('song-input').oncompositionstart = () => { composing = true; clearVideoSearch(); };
$('song-input').oncompositionend = () => { composing = false; editSearch(); };
$('clear-search').onclick = () => {
  if (videoSource === 'artists' && searchSnapshot?.artist?.selection) {
    if (!navigation.back()) videoSearch.selectArtist(searchSnapshot.artist.selection.browse_id);
  } else if (['artist','album'].includes(videoSource)) {
    if (!navigation.back()) changeSearchMode(videoSource === 'artist' ? 'songs' : catalogueSelection?.returnMode || 'songs');
  } else {
    navigation.push(); $('song-input').value = ''; editSearch(); $('song-input').focus();
  }
};
$('artist-summary').onclick = event => {
  if (!event.target.closest('[data-other-artists]')) return;
  navigation.push();
  videoSearch.showArtists();
  window.scrollTo(0, 0);
};
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
  if (mv) { const song = searchSongs[mv.dataset.findMv]; changeSearchMode('youtube', `${song.uploader} ${song.title}`); return; }
  const preview = event.target.closest('[data-preview]');
  const add = event.target.closest('[data-add-video]');
  if (preview) previewVideo(searchSongs[preview.dataset.preview]);
  if (add) addVideo(searchSongs[add.dataset.addVideo]);
};
$('preview-add').onclick = () => addVideo(previewSong);
$('close-preview').onclick = () => $('video-preview').close();
$('video-preview').onclose = () => {
  closeCachedPreview?.(); closeCachedPreview = null;
  $('preview-player').replaceChildren(); previewSong = null;
};
$('video-preview').onclick = event => { if (event.target === $('video-preview')) $('video-preview').close(); };
$('add-form').onsubmit = async event => {
  event.preventDefault();
  if (composing) return;
  const url = $('song-input').value.trim();
  if (!/^https?:\/\//i.test(url)) { switchView('library'); searchVideos(); if (!url) $('song-input').focus(); return; }
  await addVideo({url}, {title:$('title-input').value.trim() || null});
};
$('song-grid').onclick = async event => {
  const button = event.target.closest('[data-song]');
  if (!button || button.disabled) return;
  const song = library.find(s => s.id === button.dataset.song);
  await addVideo(song, {title:song.title});
};
$('library-view').addEventListener('click', async event => {
  const artist = event.target.closest('[data-search-artist]');
  if (artist) { changeSearchMode('artists', artist.dataset.searchArtist); return; }
  const historyPreview = event.target.closest('[data-preview-history]');
  if (historyPreview) {
    const song = state.history.find(item => item.id === historyPreview.dataset.previewHistory);
    if (song) previewVideo({...song, provider:'local'});
    return;
  }
  const repeat = event.target.closest('[data-recent-song]');
  if (repeat && !repeat.disabled) {
    repeat.disabled = true;
    try { await addVideo(state.history.find(song => song.id === repeat.dataset.recentSong), {repeat:true}); }
    finally { repeat.disabled = false; }
    return;
  }
  const button = event.target.closest('[data-preview-song], [data-find-mv-song]');
  if (!button) return;
  const song = library.find(song => song.id === (button.dataset.previewSong || button.dataset.findMvSong));
  if (button.hasAttribute('data-preview-song')) previewVideo({...song, provider:'local'});
  else changeSearchMode('youtube', `${song.artist} ${song.title}`.trim());
});
$('show-history').onclick = () => {
  queueTab = 'history'; renderQueue(); switchView('queue'); $('history-tab').focus();
};
$('queue-list').onclick = async event => {
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  const id = button.closest('[data-entry]').dataset.entry;
  const action = button.dataset.action;
  if (action === 'retry' || action === 'reprepare') return retryPreparation(id, action);
  if (action === 'preview') return previewVideo(state.entries.find(entry => entry.id === id));
  if (action === 'top') {
    const order = queueToTop(state.entries, id, state.playing ? state.current_id : null);
    if (order) await command('reorder', {order, revision:roomState.revision});
  } else {
    await command(action, {entry_id:id});
  }
};
for (const button of document.querySelectorAll('[data-queue-tab]')) {
  button.onclick = () => { queueTab = button.dataset.queueTab; renderQueue(); navigation.save(); };
  button.onkeydown = event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'upcoming' : event.key === 'End' ? 'history' : queueTab === 'upcoming' ? 'history' : 'upcoming';
    const tab = $(`${next}-tab`);
    tab.click(); tab.focus();
  };
}
$('history-query').oninput = () => { renderHistory(); navigation.save(); };
$('history-list').onclick = async event => {
  const button = event.target.closest('[data-repeat-song]');
  if (!button || button.disabled) return;
  const song = state.history.find(song => song.id === button.closest('[data-history-entry]').dataset.historyEntry);
  button.disabled = true;
  try { await addVideo(song, {repeat:true}); }
  finally { button.disabled = false; }
};
$('play-toggle').onclick = togglePlay;
for (const [id, step] of [['pitch-down', -1], ['pitch-up', 1]]) {
  $(id).onclick = () => {
    pitchRequest = pitchRequest.then(async () => {
      if (!(await activate())) return;
      if (player.pitchError) { toast(player.pitchError.message); return; }
      const pitch = Math.max(-6, Math.min(6, (state.pitch_semitones || 0) + step));
      await command('pitch', {pitch_semitones:pitch});
    });
  };
}
$('screen-start').onclick = async () => { if (await activate()) { if (!state.playing) await command('play'); else await syncPlayback(); } };
$('screen-retry').onclick = () => retryPreparation(state.current_id, 'retry');
const preparationRequests = new Set();
async function retryPreparation(id, action) {
  if (preparationRequests.has(id)) return;
  preparationRequests.add(id);
  document.querySelectorAll(`[data-entry="${CSS.escape(id)}"] [data-action="${action}"]`).forEach(b => b.disabled = true);
  $('screen-retry').disabled = true;
  try {
    const updated = await command(action, {entry_id:id});
    if (updated) {
      toast(action === 'reprepare' ? '已提交重新准备' : '已提交重试');
      if (action === 'reprepare') switchView('queue');
    }
  } finally {
    preparationRequests.delete(id);
    $('screen-retry').disabled = false;
    queueSignature = ''; renderQueue();
  }
}
async function nextSong() {
  if (skipping || !state.current_id) return;
  const entry_id = state.current_id;
  skipping = true; updateTransport();
  try { if (await activate()) await command('next', {entry_id}); }
  finally { skipping = false; updateTransport(); }
}
document.querySelectorAll('[data-next]').forEach(button => button.onclick = nextSong);
$('now-song').onclick = () => switchView('player');
$('transport-restart').onclick = () => command('restart');
$('lead-toggle').onclick = () => command('toggle_lead');
for (const [id, step] of [['lead-down', -5], ['lead-up', 5]]) {
  $(id).onclick = () => command('lead', {level:(Math.round(state.levels.lead_vocals * 100) + step) / 100});
}
$('storage-form').onsubmit = async event => {
  event.preventDefault();
  $('storage-save').disabled = true;
  try {
    acceptState(await request('/v1/storage', {limit_gib:Number($('storage-limit').value)}));
    toast('缓存上限已保存，旧缓存会自动清理');
  } catch (error) { toast(error.message); }
  finally { $('storage-save').disabled = false; }
};
$('seek').oninput = () => { renderRange($('seek')); seeking = true; $('current-time').textContent = time(Number($('seek').value)); };
$('seek').onchange = async () => { await command('seek', {position:Number($('seek').value)}); seeking = false; };
try { sessionStorage.removeItem('karaokeVolume'); } catch {}
async function fullscreen() {
  switchView('player');
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('player-stage').requestFullscreen(); }
  catch { toast('当前浏览器不支持全屏播放'); }
}
$('fullscreen').onclick = fullscreen;
$('exit-fullscreen').onclick = fullscreen;
// One control panel, including one timeline, in both normal and fullscreen views.
const transport = document.querySelector('.transport'), transportHome = transport.parentElement;
document.addEventListener('fullscreenchange', () => {
  const fullscreen = document.fullscreenElement === $('player-stage');
  (fullscreen ? $('player-stage') : transportHome).append(transport);
});
document.addEventListener('keydown', event => {
  if ($('video-preview').open) return;
  if (['INPUT','TEXTAREA','BUTTON','SELECT','SUMMARY'].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
  if (event.key.toLowerCase() === 'n') nextSong();
  if (event.key.toLowerCase() === 'f') fullscreen();
});
window.addEventListener('resize', () => switchView(view, {scroll:false}));
for (const [selector, variable] of [['.topbar', '--header-height'], ['.transport', '--transport-height']]) {
  new ResizeObserver(([entry]) => document.documentElement.style.setProperty(variable, `${entry.target.getBoundingClientRect().height}px`)).observe(document.querySelector(selector));
}
history.scrollRestoration = 'manual';
window.addEventListener('popstate', event => navigation.restore(event.state));
window.addEventListener('pagehide', () => navigation.save());
let scrollSave;
window.addEventListener('scroll', () => { clearTimeout(scrollSave); scrollSave = setTimeout(() => navigation.save(), 150); }, {passive:true});
let events, watchedJob;
function connectEvents() {
  const job = playback.entry?.job_id || '';
  if (events && job === watchedJob) return;
  events?.close();
  watchedJob = job;
  const source = events = new EventSource('/v1/room/events' + (job ? `?job_id=${encodeURIComponent(job)}` : ''));
  source.addEventListener('room', event => {
    if (source !== events) return;
    connected = true; renderConnection();
    acceptState(JSON.parse(event.data));
  });
  source.onopen = () => { if (source === events) retryFinished(); };
  source.onerror = () => {
    if (source !== events) return;
    connected = false; renderConnection();
  };
}
function savePlayback(position) {
  lastPlaybackSave = Date.now();
  const saved = playback.save(position);
  if (!tabs.write(saved)) { followPlayback(); return; }
  tabChange = tabs.record.change;
  try { sessionStorage.setItem('karaokePlayback', JSON.stringify(saved)); } catch {}
}
function followPlayback() {
  const record = tabs.read();
  if (record.change === tabChange) return;
  tabChange = record.change;
  player.clear(); readySent = null;
  playback = new LocalPlayback(record.playback);
  queueSyncError = false;
  clearTimeout(finishRetry); finishRetry = null;
  renderState(); renderConnection();
}
function claimPlayback() {
  followPlayback();
  if (tabs.owns()) return false;
  playback.reconcile(roomState);
  playback = new LocalPlayback(playback.save());
  if (!tabs.write(playback.save(), {claim:true})) { followPlayback(); return false; }
  tabChange = tabs.record.change;
  renderState();
  syncFinished();
  return true;
}
window.addEventListener('storage', event => { if (event.key === tabs.key) followPlayback(); });
window.addEventListener('online', retryFinished);
window.addEventListener('pagehide', () => {
  if (tabs.owns()) { tabs.write(playback.save(), {release:true}); player.stop(); }
  events?.close();
});
function resumePage() {
  followPlayback();
  events?.close(); events = null; connectEvents();
  retryFinished();
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) resumePage(); });
window.addEventListener('pageshow', event => { if (event.persisted) resumePage(); });
connectEvents();
updateTransport();
if (!navigation.restore()) navigation.save();
// A refresh can retain playback intent, but the browser still owns permission
// to produce audio. Until unlock succeeds, no ready event starts the clock.
if (tabs.owns() && playback.state.playing) player.unlock().then(() => renderState()).catch(() => updateTransport());
