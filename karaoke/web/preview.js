import {Player} from './player.js';

// Preview shares the singing player's clock, buffering and independent tracks.
export function mountPreview(container, song, onError) {
  const video = document.createElement('video');
  const controls = document.createElement('div');
  controls.className = 'preview-controls';
  const button = document.createElement('button');
  button.textContent = '播放';
  const seek = document.createElement('input');
  Object.assign(seek, {type:'range', min:0, max:song.duration, step:.1, value:0});
  seek.setAttribute('aria-label', '预览进度');
  const time = document.createElement('time');
  controls.append(button, seek, time);
  container.replaceChildren(video, controls);
  const entry = {...song, id:song.job_id};
  let playing = false, position = 0, generation = 0;
  const update = value => {
    position = Math.max(0, value);
    seek.value = position;
    time.textContent = `${Math.floor(position / 60)}:${String(Math.floor(position % 60)).padStart(2, '0')}`;
  };
  const pause = () => {
    generation++;
    playing = false;
    player.stop();
    button.textContent = '播放';
    button.disabled = false;
  };
  const player = new Player(video, update, () => { pause(); update(0); });
  const start = async () => {
    const token = ++generation;
    button.disabled = true;
    try {
      await player.unlock();
      await player.load(entry, position);
      if (token !== generation) return;
      playing = true;
      button.textContent = '暂停';
      await player.apply({current_id:entry.id, position, playing:true, started_at:Date.now()/1000,
        levels:{instrumental:1, backing_vocals:1, lead_vocals:0}}, 0);
    } catch (error) {
      if (token === generation) { pause(); onError(error); }
    } finally {
      if (token === generation) button.disabled = false;
    }
  };
  player.onBuffering = value => { pause(); update(value); start(); };
  player.onError = error => { pause(); onError(error); };
  button.onclick = () => playing ? pause() : start();
  seek.onchange = () => {
    const resume = playing;
    const value = Number(seek.value);
    pause(); update(value);
    if (resume) start();
    else video.currentTime = value;
  };
  video.src = entry.artifacts.mv;
  video.preload = 'metadata';
  update(0);
  return () => { pause(); player.dispose(); };
}
