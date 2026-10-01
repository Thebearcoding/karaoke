import { Player } from './player.js';
import { createVideoSearch, previewURL, searchLibrary, visibleResults } from './search.js';

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
let videoSource = 'all';
let youtubePending = false;
const addingVideos = new Set();
const providerName = provider => ({local:'本地', youtube:'YouTube', bilibili:'B站'})[provider];
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
  $('submit-song').innerHTML = `${isURL ? '加入歌单' : '搜索'} ${icon('arrow')}`;
  const search = isURL ? '' : query.toLocaleLowerCase();
  const songs = library.filter(song => (!search || search.split(/\s+/).every(word => `${song.title} ${song.artist}`.toLocaleLowerCase().includes(word))) &&
    (search || filter === 'all' || (filter === 'ready' ? song.status === 'completed' : song.language === filter)));
  $('library-count').textContent = `${songs.length} 首${filter === 'ready' ? '已准备' : '歌曲'}`;
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
      <button class="add-song" data-song="${escape(song.id)}" aria-label="点歌：${escape(song.title)}" title="加入歌单">${icon('plus')}</button></div>
    <div class="song-detail"><h3>${escape(song.title)}</h3><div class="song-info-row"><span class="song-artist">${escape(song.artist)}</span>${label ? `<span class="${ready ? 'ready-label' : 'prepare-label'}">${label}</span>` : ''}</div>${song.status === 'running' ? `<div class="preparation-slot">${preparationHTML(song)}</div>` : ''}</div></article>`;
    }).join('');
  } else {
    updatePreparationSlots($('song-grid'), songs);
  }
  $('search-empty').hidden = songs.length > 0;
}

function renderVideoResults(providers) {
  youtubePending = providers.youtube.loading;
  $('video-results').hidden = false;
  $('local-library').hidden = true;
  $('search-status').innerHTML = Object.entries(providers).map(([name, info]) => `<span class="${info.error ? 'search-error' : ''}">${providerName(name)} · ${info.loading ? '搜索中…' : info.error ? escape(info.error) : info.results.length ? `${info.results.length} ${name === 'local' ? '首' : '个视频'}` : name === 'local' ? '未找到' : '未找到官方版本'}${info.error ? '<button data-search-retry>重试</button>' : ''}</span>`).join('');
  const songs = Object.values(providers).flatMap(info => info.results);
  const order = {local:0, youtube:1, bilibili:2};
  // Keep provider ranking independent of response order, without replacing cards.
  for (const song of songs) {
    const existing = searchSongs.findIndex(old => old.url === song.url && old.provider === song.provider);
    if (existing >= 0) { searchSongs[existing] = song; continue; }
    const index = searchSongs.push(song)-1;
    const before = [...$('video-grid').children].find(card => order[card.dataset.provider] > order[song.provider]);
    (before || $('video-grid')).insertAdjacentHTML(before ? 'beforebegin' : 'beforeend', `<article class="video-card" data-provider="${song.provider}" data-result="${index}">
      <button class="video-thumbnail" data-preview="${index}" aria-label="预览：${escape(song.title)}">${song.thumbnail ? `<img src="${escape(song.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : ''}<span class="preview-label">${icon('play')}预览</span>${song.duration ? `<span class="poster-duration">${time(song.duration)}</span>` : ''}</button>
      <h3 title="${escape(song.title)}">${escape(song.title)}</h3><p class="video-uploader">${escape(song.uploader)}${song.provider === 'local' ? ' · <span class="local-status"></span>' : ''}</p>
      <div class="video-actions"><button data-add-video="${index}">${icon('plus')}<span>加入歌单</span></button><a href="${escape(song.url)}" target="_blank" rel="noopener noreferrer">${song.provider === 'local' ? '原站' : providerName(song.provider)} ↗</a></div>
    </article>`);
  }
  document.querySelectorAll('[data-result]').forEach(card => {
    const song = searchSongs[card.dataset.result];
    card.dataset.active = String(songs.some(item => item.url === song.url && item.provider === song.provider));
    if (song.provider === 'local') card.querySelector('.local-status').textContent = song.status === 'completed' ? '已准备' : song.status === 'running' ? '准备中' : song.status === 'queued' ? '等待准备' : '需准备';
  });
  filterVideos();
  updateVideoButtons();
}

function filterVideos() {
  const cards = [...document.querySelectorAll('[data-result]')];
  const visible = visibleResults(cards.filter(card => card.dataset.active === 'true').map(card => searchSongs[card.dataset.result]), videoSource, youtubePending);
  cards.forEach(card => {
    card.hidden = card.dataset.active !== 'true' || !visible.includes(searchSongs[card.dataset.result]);
  });
  document.querySelectorAll('[data-video-source]').forEach(button => {
    button.classList.toggle('selected', button.dataset.videoSource === videoSource);
    button.setAttribute('aria-pressed', String(button.dataset.videoSource === videoSource));
  });
}

function updateVideoButtons() {
  document.querySelectorAll('[data-add-video], #preview-add').forEach(button => {
    const song = button.id === 'preview-add' ? previewSong : searchSongs[button.dataset.addVideo];
    if (!song) return;
    const queued = state.entries.some(entry => entry.url === song.url);
    const adding = addingVideos.has(song.url);
    button.disabled = queued || adding;
    const label = button.querySelector('span') || button;
    label.textContent = queued ? '已在歌单 ✓' : adding ? '加入中…' : '加入歌单';
  });
}

function clearVideoSearch() {
  clearTimeout(searchTimer);
  videoSearch.cancel();
  searchQuery = '';
  searchSongs = [];
  $('video-results').hidden = true;
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
  videoSearch.search(query);
}

function editSearch() {
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
    toast('已加入歌单');
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
  if (song.provider === 'local' && song.status === 'completed' && song.job_id) {
    const video = document.createElement('video');
    video.src = `/v1/jobs/${song.job_id}/artifacts/mv`;
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
  $('quality-note').hidden = !song?.lyrics_need_review;
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
document.querySelectorAll('[data-video-source]').forEach(button => button.onclick = () => { videoSource = button.dataset.videoSource; filterVideos(); });
$('song-input').oncompositionstart = () => { composing = true; clearVideoSearch(); };
$('song-input').oncompositionend = () => { composing = false; editSearch(); };
$('clear-search').onclick = () => { $('song-input').value = ''; editSearch(); $('song-input').focus(); };
$('search-status').onclick = event => { if (event.target.closest('[data-search-retry]')) searchVideos(true); };
$('video-grid').onclick = event => {
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
    toast('已加入歌单');
  } catch (error) { toast(error.message); }
  finally { $('submit-song').disabled = false; }
};
$('song-grid').onclick = async event => {
  const button = event.target.closest('[data-song]');
  if (!button) return;
  const song = library.find(s => s.id === button.dataset.song);
  button.disabled = true;
  try { acceptState(await request('/v1/room/entries', song.job_id ? {job_id:song.job_id} : {url:song.url,title:song.title})); toast(`《${song.title}》已加入歌单`); }
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
  if (['INPUT','TEXTAREA','BUTTON'].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
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
