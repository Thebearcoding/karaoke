import test from 'node:test';
import assert from 'node:assert/strict';
import {Player} from './web/player.js';

let frame;
globalThis.requestAnimationFrame = callback => { frame = callback; };
class Context {
  currentTime = 10;
  state = 'running';
  destination = {};
  createGain() { return {gain:{value:0,setTargetAtTime(value){this.value=value;}},connect(){}}; }
  createBufferSource() {
    return {connect(target){this.target=target;},start(time,offset){this.started=[time,offset];},stop(){this.stopped=true;},disconnect(){}};
  }
  resume() { return Promise.resolve(); }
  decodeAudioData(data) { return Promise.resolve({duration:20, tag:data}); }
}
globalThis.window = {AudioContext:Context};
globalThis.fetch = async url => ({ok:true,arrayBuffer:async()=>url});
const song = {id:'one',duration:20,artifacts:{mv:'video.mp4',instrumental_preview:'instrumental',backing_vocals_preview:'backing',lead_vocals_preview:'lead'}};
const state = {current_id:'one',position:3,started_at:100,playing:true,mix:'karaoke'};
function setup(onEnd=()=>{}) {
  const video = {currentTime:0,readyState:4,load(){},removeAttribute(){},pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();}};
  return new Player(video,()=>{},onEnd);
}
const clockOffset = () => 100-Date.now()/1000;

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

test('seek and pause stop all previous sources; end notification fires once', async () => {
  const ended=[];const p=setup(id=>ended.push(id));await p.unlock();await p.load(song);
  p.apply(state,clockOffset());const previous=[...p.sources];
  p.apply({...state,position:12},clockOffset());
  assert.ok(previous.every(s=>s.stopped));assert.ok(p.sources.every(s=>Math.abs(s.started[1]-12)<.01));
  const seeked=[...p.sources];p.apply({...state,playing:false,started_at:null,position:14},clockOffset());
  assert.ok(seeked.every(s=>s.stopped));assert.equal(p.sources.length,0);assert.equal(p.video.currentTime,14);
  p.apply({...state,position:19},clockOffset());p.context.currentTime+=2;
  frame();frame();assert.deepEqual(ended,['one']);assert.equal(p.sources.length,0);
});

test('a late download from the previous selection cannot replace current audio', async () => {
  const p=setup();await p.unlock();
  const originalFetch=globalThis.fetch;const pending=[];
  globalThis.fetch=url=>new Promise(resolve=>pending.push(()=>resolve({ok:true,arrayBuffer:async()=>url})));
  const oldLoad=p.load(song);
  globalThis.fetch=originalFetch;
  await p.load({...song,id:'two',artifacts:{...song.artifacts,lead_vocals_preview:'new-lead'}});
  pending.forEach(resolve=>resolve());await oldLoad;
  assert.equal(p.entry.id,'two');assert.equal(p.buffers[2].tag,'new-lead');
  p.clear();assert.equal(p.entry,null);assert.equal(p.buffers,null);
});
