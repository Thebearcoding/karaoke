import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {pitchConfiguration} from './web/pitch-shift.js';

async function processor() {
  let Processor, ready;
  const loaded = new Promise(resolve => { ready = resolve; });
  const replies = new Map();
  const context = vm.createContext({
    sampleRate:48000, currentTime:0, WebAssembly, console, atob, setTimeout, clearTimeout, performance,
    AudioWorkletProcessor:class {
      constructor() { this.port = {postMessage:([id, value]) => {
        if (id === 'ready') ready();
        else replies.get(id)?.(value);
      }}; }
    },
    registerProcessor:(name, implementation) => { Processor = implementation; },
  });
  const code = await readFile(new URL('./web/vendor/signalsmith-stretch.mjs', import.meta.url), 'utf8');
  vm.runInContext(code.replace('export default _export;', ''), context);
  const node = new Processor({numberOfOutputs:1, outputChannelCount:[2]});
  await loaded;
  let counter = 0;
  const call = (method, ...args) => new Promise(resolve => {
    const id = counter++; replies.set(id, resolve);
    node.port.onmessage({data:[id, method, ...args]}); replies.delete(id);
  });
  await call('configure', pitchConfiguration);
  await call('schedule', {active:true, semitones:0});
  return {node, context, call};
}

function peakNear(samples, expected) {
  let best = {frequency:0, amplitude:0};
  for (let f = expected-4; f <= expected+4; f += .25) {
    let real=0, imaginary=0;
    for (let i=0; i<samples.length; i++) {
      const weight = .5-.5*Math.cos(2*Math.PI*i/(samples.length-1));
      const angle = 2*Math.PI*f*i/48000;
      real += samples[i]*weight*Math.cos(angle);
      imaginary += samples[i]*weight*Math.sin(angle);
    }
    const amplitude = 4*Math.hypot(real,imaginary)/samples.length;
    if (amplitude > best.amplitude) best={frequency:f, amplitude};
  }
  return best;
}

test('real WASM shifts a stereo three-tone mix dynamically without changing the time scale', async t => {
  const {node, context, call} = await processor();
  const latency = await call('latency');
  assert.ok(latency > 0 && latency < .25);
  let maxCents = 0;
  const tones=[[220,330,550],[261.6256,392,659.2551]];
  const output=[new Float32Array(480000),new Float32Array(480000)];
  let previous;
  for (let first=0; first<output[0].length; first+=128) {
    context.currentTime=first/48000;
    const shift=[0,3,-2,6,-6][Math.floor(first/96000)];
    if (shift !== previous) { await call('schedule', {semitones:shift}); previous=shift; }
    const input=tones.map(frequencies => Float32Array.from({length:128}, (_,i) =>
      frequencies.reduce((sum,f) => sum+.1*Math.sin(2*Math.PI*f*(first+i)/48000),0)));
    const block=[new Float32Array(128),new Float32Array(128)];
    node.process([input],[block],{});
    block.forEach((data,c) => output[c].set(data,first));
  }
  for (const [time,shift] of [[1,0],[3,3],[5,-2],[7,6],[9,-6]]) {
    for (let channel=0;channel<2;channel++) {
      const segment=output[channel].subarray(time*48000,(time+.4)*48000);
      assert.ok(segment.every(Number.isFinite));
      for (const tone of tones[channel]) {
        const expected=tone*2**(shift/12), peak=peakNear(segment,expected);
        const cents = Math.abs(1200*Math.log2(peak.frequency/expected));
        maxCents = Math.max(maxCents, cents);
        assert.ok(cents < 10, `${tone}Hz shifted ${shift}: ${cents.toFixed(2)} cents off`);
        assert.ok(peak.amplitude>.04 && peak.amplitude<.2, `Missing/clipped tone: ${JSON.stringify(peak)}`);
      }
    }
  }
  t.diagnostic(`Latency ${(latency*1000).toFixed(1)} ms; maximum pitch error ${maxCents.toFixed(2)} cents`);
});

test('reported processor delay matches the zero-shift audio timeline', async () => {
  const {node, context, call}=await processor();
  const delay=Math.round(await call('latency')*48000);
  const input=new Float32Array(48000), output=new Float32Array(96000);
  let seed=42;
  for (let i=0;i<input.length;i++) { seed=(Math.imul(seed,1664525)+1013904223)>>>0; input[i]=.2*(seed/2**32-.5); }
  for (let first=0;first<output.length;first+=128) {
    context.currentTime=first/48000;
    const block=new Float32Array(128), rendered=[new Float32Array(128),new Float32Array(128)];
    if (first<input.length) block.set(input.subarray(first,Math.min(first+128,input.length)));
    node.process([[block,block]],[rendered],{});output.set(rendered[0],first);
  }
  let error=0, energy=0;
  for (let i=10000;i<40000;i++) { error+=(input[i]-output[i+delay])**2; energy+=input[i]**2; }
  assert.ok(error/energy<1e-6, `Latency mismatch: relative error ${error/energy}`);
});

test('inactive pitch path skips DSP and resumes processing when enabled', async () => {
  const {node, call}=await processor();
  const original=node.wasmModule._process;
  let calls=0;
  node.wasmModule._process=(...args)=>{calls++;return original(...args);};
  const input=[new Float32Array(128).fill(.1),new Float32Array(128).fill(.2)];
  const output=input.map(a=>a.slice());
  await call('stop');
  for (let i=0;i<100;i++) node.process([input],[output],{});
  assert.equal(calls,0);
  assert.ok(output.every(channel=>channel.every(value=>value===0)));
  await call('configure',pitchConfiguration);
  await call('schedule',{active:true,semitones:3});
  node.process([input],[output],{});
  assert.equal(calls,1);
});
