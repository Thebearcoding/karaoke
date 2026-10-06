import test from 'node:test';
import assert from 'node:assert/strict';
import {Player} from './web/player.js';

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
  const video = Object.assign(new EventTarget(), {currentTime:0,readyState:4,load(){},removeAttribute(){},pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}});
  return new Player(video,()=>{},onEnd);
}
const clockOffset = () => 100-Date.now()/1000;

test('a paused video loads its first frame before playback is requested', async () => {
  const p=setup(); p.video.preload='metadata';
  p.video.load=function () {
    this.readyState=1;
    if (this.preload === 'auto') queueMicrotask(() => {
      this.readyState=2; this.dispatchEvent(new Event('loadeddata'));
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
    p.apply(state,clockOffset());
    assert.equal(p.sources.length,0);
    let unlocked=false;
    const unlocking=p.unlock().then(()=>{unlocked=true;});
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(unlocked,false);
    activateProcessor({input:{},setPitch(){},stop(){},start(){}});
    await unlocking; p.apply(state,clockOffset());
    assert.equal(p.sources.length,3);
  } finally { activateProcessor({input:{},setPitch(){},stop(){},start(){}}); await pending; p.clear(); }
});

test('decoded audio cannot start the clock before the video is ready', async () => {
  const p=setup(); await p.unlock(); p.video.readyState=0;
  let loaded=false;
  const pending=p.load(song).then(()=>{loaded=true;});
  await new Promise(resolve=>setImmediate(resolve));
  p.apply(state,clockOffset());
  assert.equal(loaded,false); assert.equal(p.sources.length,0); assert.equal(p.ready,false);
  p.video.readyState=1; p.video.dispatchEvent(new Event('loadeddata'));
  await new Promise(resolve=>setImmediate(resolve)); assert.equal(loaded,false);
  p.video.readyState=2; p.video.dispatchEvent(new Event('canplay'));
  await pending; p.apply(state,clockOffset()); assert.equal(p.sources.length,3);
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
  p.video.readyState=2; p.video.dispatchEvent(new Event('canplay'));
  await current; assert.equal(p.entry.id,'two');
  p.clear();
});

test('stems begin together; mode and volume changes do not restart or seek', async () => {
  const p=setup();await p.unlock();await p.load(song);p.apply(state,clockOffset());
  assert.equal(p.sources.length,3);
  assert.deepEqual(p.sources.map(s=>s.started),[p.sources[0].started,p.sources[0].started,p.sources[0].started]);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,0]);
  const originalSources=[...p.sources], originalPosition=p.video.currentTime;
  p.apply({...state,mix:'instrumental'},clockOffset());p.setVolume(.4);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,0,0]);assert.equal(p.master.gain.value,.4);
  p.apply({...state,mix:'original'},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[1,1,1]);assert.deepEqual(p.sources,originalSources);
  p.apply({...state,mix:'custom',levels:{instrumental:.8,backing_vocals:.35,lead_vocals:.2}},clockOffset());
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,.35,.2]);
  assert.deepEqual(p.sources,originalSources);assert.equal(p.video.currentTime,originalPosition);
  assert.equal(p.master.gain.value,.4);
  p.setLevels({backing_vocals:0});
  assert.deepEqual(p.gains.map(g=>g.gain.value),[.8,0,.2]);
  p.apply({...state,levels:{instrumental:0,backing_vocals:0,lead_vocals:0}},clockOffset());
  assert.ok(p.gains.every(g=>g.gain.value===0));assert.deepEqual(p.sources,originalSources);
});

test('opening pre-roll holds video and all stems on a future shared start; pause and seek cancel it', async () => {
  const p=setup(), ticks=[];
  p.onTick=value=>ticks.push(value);
  p.pitchFactory=async()=>({input:{},latency:.12,setPitch(){},start(){},stop(){}});
  await p.unlock(); await p.load(song);
  const start={...state,position:0,started_at:103.369};
  p.apply(start,clockOffset());
  const sources=[...p.sources], audible=p.audibleWhen;
  assert.ok(Math.abs(audible-13.369)<.01);
  assert.ok(sources.every(s=>Math.abs(s.started[0]+.12-audible)<1e-9 && s.started[1]===0));
  frame(); assert.equal(p.video.paused,true); assert.ok(ticks.at(-1)<-3.3);
  p.apply({...start,levels:{lead_vocals:.5}},clockOffset());
  assert.deepEqual(p.sources,sources); assert.equal(p.audibleWhen,audible);
  p.context.currentTime=audible-.01; frame(); assert.equal(p.video.paused,true);
  p.context.currentTime=audible; frame(); assert.equal(p.video.paused,false);
  assert.ok(Math.abs(ticks.at(-1))<1e-9); assert.equal(p.video.currentTime,0);
  p.apply({...start,started_at:107.},clockOffset());
  const waiting=[...p.sources]; frame(); assert.equal(p.video.paused,true);
  p.apply({...start,playing:false,started_at:null},clockOffset());
  assert.ok(waiting.every(s=>s.stopped)); assert.equal(p.anchor,null); assert.equal(ticks.at(-1),0);
  p.apply({...state,position:5},clockOffset());
  assert.ok(p.sources.every(s=>Math.abs(s.started[1]-5)<.01));
  await p.load({...song,id:'two'});
  assert.equal(p.anchor,null); assert.equal(p.video.paused,true);
  p.clear();
});

test('seek and pause stop all previous sources; end notification fires once', async () => {
  const ended=[];const p=setup(id=>ended.push(id));await p.unlock();await p.load(song);
  p.apply(state,clockOffset());const previous=[...p.sources];
  p.apply({...state,position:12},clockOffset());
  assert.ok(previous.every(s=>s.stopped));assert.ok(p.sources.every(s=>Math.abs(s.started[1]-12)<.01));
  const seeked=[...p.sources];p.apply({...state,playing:false,started_at:null,position:14},clockOffset());
  assert.ok(seeked.every(s=>s.stopped));assert.equal(p.sources.length,0);assert.equal(p.video.currentTime,14);
  p.video.seeking=true;
  p.apply({...state,playing:false,started_at:null,position:15},clockOffset());
  assert.equal(p.video.currentTime,15); // A new explicit seek supersedes the old request.
  p.video.seeking=false;
  p.apply({...state,position:19},clockOffset());p.context.currentTime+=2;
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

test('live key changes affect the common mix without restarting; video follows audible time', async () => {
  const p=setup(), shifts=[], starts=[];
  p.pitchFactory=async () => ({input:{},latency:.12,setPitch(value){shifts.push(value);},start(when){starts.push(when);},stop(){}});
  await p.unlock();await p.load(song);p.apply(state,clockOffset());
  const sources=[...p.sources], anchor=p.anchor, initial=p.video.currentTime;
  assert.ok(p.gains.every(g=>g.target===p.pitchChain.input));
  assert.ok(Math.abs(starts[0]-sources[0].started[0]-.12)<1e-9);
  p.apply({...state,pitch_semitones:3},clockOffset());
  p.apply({...state,pitch_semitones:-2,levels:{instrumental:.7,backing_vocals:.5,lead_vocals:.3}},clockOffset());
  assert.deepEqual(p.sources,sources);assert.equal(p.anchor,anchor);
  assert.ok(sources.every(s=>s.playbackRate.value===1));
  assert.deepEqual(shifts.slice(-2),[3,-2]);
  frame();assert.equal(p.video.currentTime,initial); // Wait through processing latency.
  p.context.currentTime=starts[0]+2;
  frame();assert.ok(Math.abs(p.video.currentTime-initial-2)<1e-6);
  p.apply({...state,position:8,pitch_semitones:-2},clockOffset());
  assert.ok(sources.every(s=>s.stopped));assert.equal(p.pitch,-2);
  p.apply({...state,position:8,pitch_semitones:0},clockOffset());
  assert.equal(p.pitch,0);
});

test('a slow video decoder can finish seeking while audio keeps its common clock', async () => {
  const p=setup(); await p.unlock(); await p.load(song); p.apply(state,clockOffset());
  const sources=[...p.sources], anchor=p.anchor;
  let seeks=0, videoTime=p.video.currentTime;
  Object.defineProperty(p.video,'currentTime',{get:()=>videoTime,set:value=>{videoTime=value;seeks++;p.video.seeking=true;}});
  p.context.currentTime+=2; frame(); // One correction for a large gap.
  assert.equal(seeks,1);
  for (let i=0;i<120;i++) { p.context.currentTime+=1/60; frame(); }
  assert.equal(seeks,1); // No retargeting while the decoder is seeking.
  p.video.seeking=false; frame(); assert.equal(seeks,2);
  p.video.seeking=false; p.context.currentTime+=.2; frame();
  assert.equal(seeks,2); // Give the completed seek time to settle.
  assert.deepEqual(p.sources,sources); assert.equal(p.anchor,anchor);
  p.clear();
});

test('small video drift uses playback rate without seeking or changing audio', async () => {
  const p=setup(); await p.unlock(); await p.load(song); p.apply(state,clockOffset());
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
  p.apply(state,clockOffset());
  for (let i=1;i<=20;i++) p.apply(state,clockOffset()+i);
  assert.equal(seeks,1); assert.equal(videoTime,state.position); assert.equal(p.sources.length,0);
  p.apply({...state,position:12},clockOffset()); assert.equal(videoTime,12);
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
    p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,6);
    const first=p.sources.slice(0,3), next=p.sources.slice(3);
    assert.ok(first.every(s=>s.length===8 && s.started[1]===0));
    assert.ok(next.every(s=>Math.abs(s.started[0]-first[0].started[0]-8)<1e-9));
    assert.ok(next.every(s=>Math.abs(s.started[1]-256/48000)<1e-9));
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
    p.apply({...state,position:0},clockOffset());
    assert.equal(p.sources.length,3);
    const previous=[...p.sources];
    p.context.currentTime=p.anchor+8.1; frame(); frame();
    assert.equal(p.anchor,null); assert.deepEqual(waiting,[[8,100]]);
    assert.ok(previous.every(s=>s.stopped));
    await p.load({...song,duration:80},60);
    p.apply({...state,position:60},clockOffset());
    const resumed=p.sources.slice(0,3);
    assert.ok(resumed.every(s=>Math.abs(s.started[1]-(4+256/48000))<1e-8));
    release(); await settled();
    assert.deepEqual([...p.stream.windows.keys()],[7,8]);
    assert.ok(p.sources.every(s=>!s.buffer.tag.endsWith('/1')));
  } finally { release?.(); p.clear(); globalThis.fetch=fetchAudio; }
});

test('a failed prefetch does not retry in a loop or interrupt the buffered window', async () => {
  let failures=0;
  globalThis.fetch=streamingFetch(url=>url.endsWith('/lead_vocals/1') ? (failures++, {ok:false}) : null);
  const p=setup(), waiting=[];
  p.onBuffering=(...args)=>waiting.push(args);
  try {
    await p.unlock(); await p.load({...song,duration:80}); await settled();
    p.apply({...state,position:0},clockOffset());
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
