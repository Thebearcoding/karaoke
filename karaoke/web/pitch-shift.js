import SignalsmithStretch from './vendor/signalsmith-stretch.mjs';

// A fixed delay keeps the unprocessed and pitch-shifted paths on one timeline.
export const pitchConfiguration = {blockMs:160, intervalMs:40, splitComputation:true};

export async function createPitchShift(context, destination) {
  if (!context.audioWorklet) throw new Error('实时升降调需要安全连接，请用 HTTPS 或本机 localhost 打开');
  const processor = await SignalsmithStretch(context);
  await processor.configure(pitchConfiguration);
  const latency = await processor.latency();
  const input = context.createGain(), dry = context.createGain(), wet = context.createGain();
  const delay = context.createDelay(1), gate = context.createGain();
  delay.delayTime.value = latency;
  dry.gain.value = 1; wet.gain.value = 0; gate.gain.value = 0;
  input.connect(delay).connect(dry).connect(gate);
  input.connect(processor).connect(wet).connect(gate);
  gate.connect(destination);
  let pitch = 0;
  let playing = false;
  let active = false;
  let idleTimer;
  const updateProcessor = () => {
    clearTimeout(idleTimer);
    if (playing && pitch) {
      if (!active) processor.configure(pitchConfiguration);
      processor.schedule({semitones:pitch, active:true});
      active = true;
    } else {
      // Let the wet-to-dry crossfade finish before suspending expensive DSP.
      idleTimer = setTimeout(() => { processor.stop(); active = false; }, (latency + .12) * 1000);
    }
  };
  return {
    input, latency,
    setPitch(value) {
      if (value === pitch) return;
      pitch = value;
      if (playing) updateProcessor();
      const when = context.currentTime + latency;
      for (const [node, level] of [[dry, value === 0 ? 1 : 0], [wet, value === 0 ? 0 : 1]]) {
        node.gain.cancelScheduledValues(context.currentTime);
        node.gain.setTargetAtTime(level, when, .015);
      }
    },
    start(when) {
      playing = true;
      updateProcessor();
      gate.gain.cancelScheduledValues(context.currentTime);
      gate.gain.setValueAtTime(0, context.currentTime);
      gate.gain.setValueAtTime(0, when);
      gate.gain.linearRampToValueAtTime(1, when+.005);
    },
    stop() {
      playing = false;
      active = false;
      clearTimeout(idleTimer);
      processor.stop();
      gate.gain.cancelScheduledValues(context.currentTime);
      gate.gain.setTargetAtTime(0, context.currentTime, .005);
      // Clear the previous song/seek's delayed audio before opening the gate.
      processor.configure(pitchConfiguration);
    },
  };
}
