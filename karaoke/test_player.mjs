import test, {beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {Player} from './web/player.js';
import {isStaleRoomSnapshot} from './web/room-state.js';

test('late HTTP replies cannot undo a newer room command or same-revision progress update', () => {
  const current = {revision:4,server_time:100,entries:[{status:'completed'}]};
  assert.equal(isStaleRoomSnapshot({revision:3,server_time:101},current),true);
  assert.equal(isStaleRoomSnapshot({revision:4,server_time:99,entries:[{status:'running'}]},current),true);
  assert.equal(isStaleRoomSnapshot({revision:4,server_time:101},current),false);
  assert.equal(isStaleRoomSnapshot(current,current),false);
  assert.equal(isStaleRoomSnapshot({revision:5,server_time:99},current),false);
  assert.equal(isStaleRoomSnapshot({revision:0,server_time:100},{revision:0}),false);
});

let frame;
globalThis.requestAnimationFrame = callback => { frame = callback; };
class Context {
  sampleRate = 48000;
  currentTime = 10;
  state = 'running';
  destination = {};
  createGain() { return {gain:{value:0,setTargetAtTime(value){this.value=value;}},connect(target){this.target=target;},disconnect(){this.target=null;}}; }
  createBufferSource() {
    return {playbackRate:{value:1},connect(target){this.target=target;},start(time,offset,duration){this.started=[time,offset];this.length=duration;},stop(){this.stopped=true;},disconnect(){}};
  }
  resume() { return Promise.resolve(); }
  decodeAudioData(data) { return Promise.resolve({duration:20, tag:data}); }
}
globalThis.window = {AudioContext:Context};
const metadata = {sample_rate:48000, channels:2, frames:960000, chunk_frames:960000, padding_frames:256};
const fetchAudio = async url => ({ok:true,json:async()=>metadata,arrayBuffer:async()=>url});
globalThis.fetch = fetchAudio;
const song = {id:'one',job_id:'one',duration:20,artifacts:{mv:'video.mp4',instrumental_preview:'instrumental',backing_vocals_preview:'backing',lead_vocals_preview:'lead'}};
const state = {current_id:'one',position:3,started_at:100,playing:true,mix:'karaoke'};
function setup(onEnd=()=>{}) {
  let position=0, since=10, p;
  const video = Object.assign(new EventTarget(), {readyState:4,load(){},removeAttribute(){},
    pause(){position=this.currentTime;this.paused=true;},
    play(){since=p.context.currentTime;this.paused=false;return Promise.resolve();}});
  Object.defineProperty(video,'currentTime',{configurable:true,
    get:()=>position+(video.paused ? 0 : (p?.context?.currentTime ?? 10)-since),
    set:value=>{position=value;since=p?.context?.currentTime ?? 10;}});
  p=new Player(video,()=>{},onEnd);
  return p;
}
const clockOffset = () => 100-Date.now()/1000;
// The wall clock and simulated AudioContext advance only when the test asks.
beforeEach(t => t.mock.method(Date, 'now', () => 100_000));

test('a paused video loads its first frame before playback is requested', async () => {
  const p=setup(); p.video.preload='metadata';
  p.video.load=function () {
    this.readyState=1;
    if (this.preload === 'auto') queueMicrotask(() => {
      this.readyState=3; this.dispatchEvent(new Event('canplay'));
    });
  };
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  try {
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(loaded,true);
    assert.equal(p.video.paused,true);
    assert.equal(p.sources.length,0);
  } finally { p.clear(); await pending; }
});

test('media loading does not wait for an audio processor that needs a user gesture', async () => {
  const p=setup(); p.ensureContext(); p.context.state='suspended';
  let activateProcessor;
  const processor=new Promise(resolve=>{activateProcessor=resolve;});
  p.pitchFactory=()=>processor;
  p.context.resume=async()=>{p.context.state='running';};
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  try {
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(loaded,true);
    await p.apply(state,clockOffset());
    assert.equal(p.sources.length,0);
    let unlocked=false;
    const unlocking=p.unlock().then(()=>{unlocked=true;});
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(unlocked,false);
    activateProcessor({input:{},setPitch(){},stop(){},start(){}});
    await unlocking; await p.apply(state,clockOffset());
    assert.equal(p.sources.length,3);
  } finally { activateProcessor({input:{},setPitch(){},stop(){},start(){}}); await pending; p.clear(); }
});

test('decoded audio cannot start the clock before the video is ready', async () => {
  const p=setup(); await p.unlock(); p.video.readyState=0;
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  await new Promise(resolve=>setImmediate(resolve));
  await p.apply(state,clockOffset());
  assert.equal(loaded,false); assert.equal(p.sources.length,0); assert.equal(p.ready,false);
  p.video.readyState=1; p.video.dispatchEvent(new Event('loadeddata'));
  await new Promise(resolve=>setImmediate(resolve)); assert.equal(loaded,false);
  p.video.readyState=3; p.video.dispatchEvent(new Event('canplay'));
  await pending; await p.apply(state,clockOffset()); assert.equal(p.sources.length,3);
  p.clear();
});

test('a failed or superseded video never releases stale audio', async () => {
  const p=setup(); await p.unlock(); p.video.readyState=0;
  const failed=p.load(song);
  p.video.dispatchEvent(new Event('error'));
  await assert.rejects(failed,/视频加载失败/); assert.equal(p.ready,false);
  const old=p.load(song);
  const current=p.load({...song,id:'two'});
  await old; assert.equal(p.ready,false);
  p.video.readyState=3; p.video.dispatchEvent(new Event('canplay'));
  await current; assert.equal(p.entry.id,'two');
  p.clear();
});

test('native video startup must resolve before scheduling any audio; stale startup cannot restart a song', async () => {
  const p=setup(), ticks=[];
  p.onTick=value=>ticks.push(value);
  await p.unlock(); await p.load(song);
  const play=p.video.play.bind(p.video);
  let release;
  p.video.play=()=>new Promise(resolve=>{release=()=>{play();resolve();};});
  const starting=p.apply(state,clockOffset());
  await settled();
  p.context.currentTime+=3; frame();
  assert.equal(p.sources.length,0); assert.equal(ticks.at(-1),3);
  release(); await starting;
  assert.equal(p.sources.length,3);
  assert.ok(p.sources.every(source=>source.started[1]===3));
  assert.equal(p.video.paused,true); // Hold until the shared audible start.
  const stale=p.apply({...state,position:8},clockOffset());
  await settled();
  p.stop();
  await p.load({...song,id:'two',job_id:'two'});
  release(); await stale;
  assert.equal(p.sources.length,0); assert.equal(p.entry.id,'two');
  assert.equal(p.video.paused,true);
  p.clear();
});

test('seeking waits for the requested frame and does not retarget a slow decoder', async () => {
  const p=setup(); await p.unlock(); await p.load(song);
  let position=0, seeks=0, finished=false;
  Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>position,
    set:value=>{position=value;seeks++;p.video.seeking=true;}});
  const loading=p.load(song,12).then(()=>{finished=true;});
  await settled();
  for (let i=0;i<5;i++) p.video.dispatchEvent(new Event('canplay'));
  await settled();
  assert.equal(seeks,1); assert.equal(finished,false); assert.equal(p.sources.length,0);
  p.video.seeking=false; p.video.dispatchEvent(new Event('seeked'));
  await loading; await p.apply({...state,position:12},clockOffset());
  assert.equal(seeks,1); assert.equal(p.sources.length,3);
  p.clear();
});

test('video startup rejection or timeout freezes the room and exposes a retryable error', async t => {
  for (const failure of ['rejected','timeout']) {
    const p=setup(), waiting=[], errors=[];
    p.onBuffering=(...args)=>waiting.push(args); p.onError=error=>errors.push(error.message);
    await p.unlock(); await p.load(song);
    if (failure==='timeout') t.mock.timers.enable({apis:['setTimeout']});
    p.video.play=()=>failure==='timeout' ? new Promise(()=>{}) : Promise.reject(new Error('blocked'));
    const starting=p.apply(state,clockOffset());
    await settled();
    if (failure==='timeout') t.mock.timers.tick(15000);
    await starting;
    assert.equal(p.sources.length,0); assert.equal(p.anchor,null); assert.equal(p.unlocked,false);
    assert.deepEqual(waiting,[[3,100]]); assert.equal(errors.length,1);
    t.mock.timers.reset(); p.clear();
  }
});

test('readiness priming is shared across state updates and reused when the room starts', async () => {
  const p=setup(); await p.unlock(); await p.load(song);
  const play=p.video.play.bind(p.video); let calls=0, release;
  p.video.play=()=>{calls++;return new Promise(resolve=>{release=()=>{play();resolve();};});};
  const first=p.prepareVideo(3), repeated=p.prepareVideo(3);
  assert.equal(first,repeated);
  await settled(); assert.equal(calls,1); assert.equal(p.sources.length,0);
  release(); await first;
  await p.apply(state,clockOffset());
  assert.equal(calls,1); assert.equal(p.sources.length,3); assert.equal(p.video.paused,true);
  p.clear();
});

test('stems begin together; mode and volume changes do not restart or seek', async () => {
  const p=setup();await p.unlock();await p.load(song);await p.apply(state,clockOffset());
  assert.equal(p.sources.length,3);
  assert.deepEqual(p.sources.map(s=>s.started),[p.sources[0].started,p.sources[0].started,p.sources[0].started]);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
  const originalSources=[...p.sources], originalPosition=p.video.currentTime;
  await p.apply({...state,mix:'instrumental'},clockOffset());p.setVolume(.4);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,0,0]);assert.equal(p.master.gain.value,.4);
  await p.apply({...state,mix:'original'},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,1]);assert.deepEqual(p.sources,originalSources);
  await p.apply({...state,mix:'custom',levels:{instrumental:.8,backing_vocals:.35,lead_vocals:.2}},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,.35,.2]);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.equal(p.master.gain.value,.4);
  p.setLevels({backing_vocals:0});
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,0,.2]);
  await p.apply({...state,levels:{instrumental:0,backing_vocals:0,lead_vocals:0}},clockOffset());
  assert.ok(p.gains.every(g=>g.gain.value===0));assert.deepEqual(p.sources,originalSources);
  await p.apply({...state,started_at:98},clockOffset());
  assert.equal(p.video.currentTime,5);
  assert.ok(p.sources.every(s=>s.started[1]===5));
  p.clear();
});

test('opening pre-roll holds video and all stems on a future shared start; pause and seek cancel it', async () => {
  const p=setup(), ticks=[];
  p.onTick=value=>ticks.push(value);
  p.pitchFactory=async()=>({input:{},latency:.12,setPitch(){},start(){},stop(){}});
  await p.unlock(); await p.load(song);
  const start={...state,position:0,started_at:103.369};
  await p.apply(start,clockOffset());
  const sources=[...p.sources], audible=p.audibleWhen;
  assert.ok(Math.abs(audible-13.369)<.01);
  assert.ok(sources.every(s=>Math.abs(s.started[0]+.12-audible)<1e-9 && s.started[1]===0));
  frame(); assert.equal(p.video.paused,true); assert.ok(ticks.at(-1)<-3.3);
  await p.apply({...start,levels:{lead_vocals:.5}},clockOffset());
  assert.deepEqual(p.sources,sources); assert.equal(p.audibleWhen,audible);
  p.context.currentTime=audible-.01; frame(); assert.equal(p.video.paused,true);
  p.context.currentTime=audible; frame(); assert.equal(p.video.paused,false);
  assert.ok(Math.abs(ticks.at(-1))<1e-9); assert.equal(p.video.currentTime,0);
  await p.apply({...start,started_at:107.},clockOffset());
  const waiting=[...p.sources]; frame(); assert.equal(p.video.paused,true);
  await p.apply({...start,playing:false,started_at:null},clockOffset());
  assert.ok(waiting.every(s=>s.stopped)); assert.equal(p.anchor,null); assert.equal(ticks.at(-1),0);
  await p.apply({...state,position:5},clockOffset());
  assert.ok(p.sources.every(s=>Math.abs(s.started[1]-5)<.01));
  await p.load({...song,id:'two'});
  assert.equal(p.anchor,null); assert.equal(p.video.paused,true);
  p.clear();
});

test('seek and pause stop all previous sources; end notification fires once', async () => {
  const ended=[];const p=setup(id=>ended.push(id));await p.unlock();await p.load(song);
  await p.apply(state,clockOffset());const previous=[...p.sources];
  await p.apply({...state,position:12},clockOffset());
  assert.ok(previous.every(s=>s.stopped));assert.ok(p.sources.every(s=>Math.abs(s.started[1]-12)<.01));
  const seeked=[...p.sources];await p.apply({...state,playing:false,started_at:null,position:14},clockOffset());
  assert.ok(seeked.every(s=>s.stopped));assert.equal(p.sources.length,0);assert.equal(p.video.currentTime,14);
  p.video.seeking=true;
  await p.apply({...state,playing:false,started_at:null,position:15},clockOffset());
  assert.equal(p.video.currentTime,15); // A new explicit seek supersedes the old request.
  p.video.seeking=false;
  await p.apply({...state,position:19},clockOffset());p.context.currentTime+=2;
  frame();frame();assert.deepEqual(ended,['one']);assert.equal(p.sources.length,0);
});

test('a late download from the previous selection cannot replace current audio', async () => {
  const p=setup();await p.unlock();
  const originalFetch=globalThis.fetch;const pending=[];
  globalThis.fetch=url=>new Promise(resolve=>pending.push(()=>resolve({ok:true,json:async()=>metadata,arrayBuffer:async()=>url})));
  const oldLoad=p.load(song);
  globalThis.fetch=originalFetch;
  await p.load({...song,id:'two',job_id:'two'});
  pending.forEach(resolve=>resolve());await oldLoad;
  assert.equal(p.entry.id,'two');assert.equal(p.stream.at(0)[2].tag,'/v1/jobs/two/playback/lead_vocals/0?sample_rate=48000');
  p.clear();assert.equal(p.entry,null);assert.equal(p.ready,false);
});

test('a repaired asset reloads the same song; metadata updates keep the current media', async () => {
  const p=setup(); await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  try {
    const initialStream=p.stream, initialSources=[...p.sources];
    const renamed={...song,title:'Updated title',artifacts:{...song.artifacts,poster:'new-cover.jpg'}};
    assert.equal(p.matches(renamed),true);
    await p.load(renamed,3);
    assert.equal(p.stream,initialStream); assert.deepEqual(p.sources,initialSources);
    for (const artifact of ['mv','lead_vocals']) {
      const repaired={...p.entry,artifacts:{...p.entry.artifacts,[artifact]:`${artifact}?v=repaired`}};
      const previous=p.stream, playing=[...p.sources];
      assert.equal(p.matches(repaired),false);
      await p.load(repaired,3);
      assert.ok(previous.signal.aborted); assert.notEqual(p.stream,previous);
      assert.ok(playing.every(source=>source.stopped));
      assert.equal(p.video.src,repaired.artifacts.mv);
      assert.equal(p.matches(repaired),true);
      await p.apply({...state,playing:false,started_at:null},clockOffset());
      assert.equal(p.video.currentTime,3); assert.equal(p.video.paused,true);
      await p.apply(state,clockOffset());
      assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
    }
  } finally { p.clear(); }
});

test('late loading of an old file version cannot replace a repaired version of the same song', async () => {
  const p=setup(), pending=[], originalFetch=globalThis.fetch;
  await p.unlock();
  globalThis.fetch=url=>new Promise(resolve=>pending.push(()=>resolve({ok:true,json:async()=>metadata,arrayBuffer:async()=>url})));
  const old=p.load(song);
  try {
    const stale=p.stream;
    globalThis.fetch=originalFetch;
    const repaired={...song,artifacts:{...song.artifacts,mv:'video.mp4?v=repaired'}};
    await p.load(repaired,3);
    pending.forEach(resolve=>resolve()); await old;
    assert.ok(stale.signal.aborted);
    assert.equal(p.matches(song),false); assert.equal(p.matches(repaired),true);
    assert.equal(p.video.src,repaired.artifacts.mv); assert.equal(p.ready,true);
  } finally { globalThis.fetch=originalFetch; pending.forEach(resolve=>resolve()); p.clear(); }
});

test('live key changes affect the common mix without restarting; video follows audible time', async () => {
  const p=setup(), shifts=[], starts=[];
  p.pitchFactory=async () => ({input:{},latency:.12,setPitch(value){shifts.push(value);},start(when){starts.push(when);},stop(){}});
  await p.unlock();await p.load(song);await p.apply(state,clockOffset());
  const sources=[...p.sources], anchor=p.anchor, initial=p.video.currentTime;
  assert.ok(p.gains.every(g=>g.target===p.pitchChain.input));
  assert.ok(Math.abs(starts[0]-sources[0].started[0]-.12)<1e-9);
  await p.apply({...state,pitch_semitones:3},clockOffset());
  await p.apply({...state,pitch_semitones:-2,levels:{instrumental:.7,backing_vocals:.5,lead_vocals:.3}},clockOffset());
  assert.deepEqual(p.sources,sources);assert.equal(p.anchor,anchor);
  assert.ok(sources.every(s=>s.playbackRate.value===1));
  assert.deepEqual(shifts.slice(-2),[3,-2]);
  frame();assert.equal(p.video.currentTime,initial); // Wait through processing latency.
  p.context.currentTime=starts[0]; frame();
  p.context.currentTime=starts[0]+2;
  frame();assert.ok(Math.abs(p.video.currentTime-initial-2)<1e-6);
  await p.apply({...state,position:8,pitch_semitones:-2},clockOffset());
  assert.ok(sources.every(s=>s.stopped));assert.equal(p.pitch,-2);
  await p.apply({...state,position:8,pitch_semitones:0},clockOffset());
  assert.equal(p.pitch,0);
});

test('video, progress and song end follow device output rather than the ahead-of-time render clock', async t => {
  t.mock.method(performance,'now',()=>100_000);
  for (const timestamp of ['available','starting','unsupported']) {
    const ended=[], ticks=[], p=setup(id=>ended.push(id));
    p.onTick=position=>ticks.push(position);
    p.pitchFactory=async()=>({input:{},latency:.12,setPitch(){},start(){},stop(){}});
    await p.unlock(); await p.load(song);
    p.context.baseLatency=.02; p.context.outputLatency=.16;
    if (timestamp==='available') p.context.getOutputTimestamp=()=>({contextTime:p.context.currentTime-.20,performanceTime:99_980});
    if (timestamp==='starting') p.context.getOutputTimestamp=()=>({contextTime:0,performanceTime:0});
    try {
      await p.apply({...state,position:0},clockOffset());
      const sources=[...p.sources], outputStart=p.audibleWhen;
      p.context.currentTime=outputStart+.1; frame();
      assert.equal(p.video.paused,true,timestamp);
      p.context.currentTime=outputStart+.18; frame();
      assert.equal(p.video.paused,false,timestamp);
      assert.ok(Math.abs(ticks.at(-1))<1e-8,timestamp);
      p.context.currentTime=outputStart+2.18; p.video.currentTime=2; frame();
      assert.equal(p.video.playbackRate,1,timestamp);
      assert.ok(Math.abs(ticks.at(-1)-2)<1e-8,timestamp);
      assert.deepEqual(p.sources,sources);
      p.context.currentTime=outputStart+song.duration; frame();
      assert.deepEqual(ended,[],timestamp);
      p.context.currentTime+=.18; frame(); frame();
      assert.deepEqual(ended,['one'],timestamp);
    } finally { p.clear(); }
  }
});

test('a stalled video freezes the shared clock instead of chasing it with seeks', async () => {
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  p.context.currentTime=p.audibleWhen; frame();
  const sources=[...p.sources];
  let seeks=0, videoTime=p.video.currentTime;
  Object.defineProperty(p.video,'currentTime',{configurable:true,get:()=>videoTime,set:value=>{videoTime=value;seeks++;p.video.seeking=true;}});
  p.video.seeking=true;
  for (let i=0;i<120;i++) { p.context.currentTime+=1/60; frame(); }
  assert.equal(seeks,0);
  assert.equal(waiting.length,1); assert.equal(p.anchor,null);
  assert.ok(sources.every(s=>s.stopped));
  const loading=p.load(song,waiting[0][0]);
  await settled(); assert.equal(seeks,1);
  assert.equal(p.sources.length,0);
  p.video.seeking=false; p.video.dispatchEvent(new Event('seeked'));
  await loading;
  await p.apply({...state,position:videoTime,started_at:101},clockOffset());
  assert.equal(p.sources.length,3);
  p.clear();
});

test('small video drift uses playback rate without seeking or changing audio', async () => {
  const p=setup(); await p.unlock(); await p.load(song); await p.apply(state,clockOffset());
  p.context.currentTime+=2; p.video.currentTime=p.context.currentTime-p.anchor-.12;
  const before=p.video.currentTime, sources=[...p.sources];
  frame(); assert.equal(p.video.currentTime,before); assert.ok(p.video.playbackRate>1 && p.video.playbackRate<=1.1);
  p.video.currentTime=p.context.currentTime-p.anchor+.12; frame(); assert.ok(p.video.playbackRate<1);
  p.video.currentTime=p.context.currentTime-p.anchor; frame(); assert.equal(p.video.playbackRate,1);
  assert.deepEqual(p.sources,sources);
  p.clear(); assert.equal(p.video.playbackRate,1);
});

test('an unopened playback tab does not seek through another device\'s playing clock', async () => {
  const p=setup(); await p.load(song);
  let seeks=0, videoTime=0;
  Object.defineProperty(p.video,'currentTime',{get:()=>videoTime,set:value=>{videoTime=value;seeks++;}});
  await p.apply(state,clockOffset());
  for (let i=1;i<=20;i++) await p.apply(state,clockOffset()+i);
  assert.equal(seeks,1); assert.equal(videoTime,state.position); assert.equal(p.sources.length,0);
  await p.apply({...state,position:12},clockOffset()); assert.equal(videoTime,12);
  p.clear();
});

const settled = () => new Promise(resolve => setImmediate(resolve));
function streamingFetch(intercept = () => null) {
  const info = {...metadata, frames:48000*80, chunk_frames:48000*8};
  return async (url, options) => intercept(url.split('?')[0], options) || {
    ok:true, json:async()=>info, arrayBuffer:async()=>url,
  };
}

test('long songs retain two windows and schedule adjacent stems on exact common boundaries', async () => {
  globalThis.fetch=streamingFetch();
  const p=setup();
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,6);
    const first=p.sources.slice(0,3), next=p.sources.slice(3);
    assert.ok(first.every(s=>s.length===8 && s.started[1]===0));
    assert.ok(next.every(s=>Math.abs(s.started[0]-first[0].started[0]-8)<1e-9));
    assert.ok(next.every(s=>Math.abs(s.started[1]-256/48000)<1e-9));
    p.context.currentTime=p.audibleWhen; frame();
    for (let position=1;position<79;position++) {
      p.context.currentTime=p.anchor+position;
      for (const source of [...p.sources]) {
        if (source.started[0]+source.length<=p.context.currentTime) source.onended();
      }
      frame(); await settled();
      assert.notEqual(p.anchor,null);
      assert.ok(p.stream.windows.size<=2);
      assert.ok(p.sources.length<=6);
      assert.ok([...p.stream.windows.keys()].every(i=>i>=Math.floor(position/8)));
    }
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});

test('one delayed stem freezes all audio at the last common boundary and a seek can resume', async () => {
  let release;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? new Promise(resolve=>{release=()=>resolve({ok:true,arrayBuffer:async()=>url});}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,3);
    const previous=[...p.sources];
    p.context.currentTime=p.anchor+8.1; frame(); frame();
    assert.equal(p.anchor,null); assert.deepEqual(waiting,[[8,100]]);
    assert.ok(previous.every(s=>s.stopped));
    await p.load({...song,duration:80},60);
    await p.apply({...state,position:60},clockOffset());
    const resumed=p.sources.slice(0,3);
    assert.ok(resumed.every(s=>Math.abs(s.started[1]-(4+256/48000))<1e-8));
    release(); await settled();
    assert.deepEqual([...p.stream.windows.keys()],[7,8]);
    assert.ok(p.sources.every(s=>!s.buffer.tag.endsWith('/1')));
  } finally { release?.(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('a missing window drains already rendered audio before reporting buffering', async () => {
  let release;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? new Promise(resolve=>{release=()=>resolve({ok:true,arrayBuffer:async()=>url});}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    p.context.baseLatency=.02; p.context.outputLatency=.16;
    await p.apply({...state,position:0},clockOffset());
    const anchor=p.anchor;
    p.context.currentTime=p.audibleWhen+.18; frame();
    p.context.currentTime=anchor+8.1; frame();
    assert.deepEqual(waiting,[]);
    assert.notEqual(p.anchor,null);
    p.context.currentTime=anchor+8.19; frame(); frame();
    assert.deepEqual(waiting,[[8,100]]);
    assert.equal(p.anchor,null);
  } finally { release?.(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('a failed prefetch does not retry in a loop or interrupt the buffered window', async () => {
  let failures=0;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? (failures++, {ok:false}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    await p.apply({...state,position:0},clockOffset());
    for (let i=0;i<10;i++) { p.context.currentTime+=.1; frame(); }
    assert.equal(failures,1); assert.equal(waiting.length,0);
    p.context.currentTime=p.anchor+8.01; frame();
    assert.deepEqual(waiting,[[8,100]]);
    await assert.rejects(p.load({...song,duration:80},8),/音轨加载失败/);
    assert.equal(p.key,null);
    globalThis.fetch=streamingFetch();
    await p.load({...song,duration:80},8);
    assert.ok(p.hasAudio(8));
  } finally { p.clear(); globalThis.fetch=fetchAudio; }
});
