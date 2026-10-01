import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSearch, previewURL} from './web/search.js';

function setup() {
  const requests = [], snapshots = [];
  const search = createVideoSearch(state => snapshots.push(structuredClone(state)), (url, {signal}) =>
    new Promise(resolve => requests.push({url, signal, finish: (results, ok = true) => resolve({ok, json:async () => ok ? {results} : {detail:'Provider failed'}})})));
  return {search, requests, snapshots};
}
const flush = () => new Promise(resolve => setImmediate(resolve));

test('a fast provider renders while the other is loading; errors retain usable results', async () => {
  const {search, requests, snapshots} = setup();
  const pending = search.search('Some song & singer');
  assert.equal(new URL(requests[0].url, 'http://localhost').searchParams.get('q'), 'Some song & singer');
  requests[0].finish([{title:'A real result'}]);
  await flush();
  assert.equal(snapshots.at(-1).youtube.results[0].title, 'A real result');
  assert.equal(snapshots.at(-1).bilibili.loading, true);
  requests[1].finish([], false);
  await pending;
  assert.equal(snapshots.at(-1).bilibili.error, 'Provider failed');
  assert.equal(snapshots.at(-1).youtube.results.length, 1);
});

test('new queries and clearing the input discard stale responses even if abort is ignored', async () => {
  const {search, requests, snapshots} = setup();
  const old = search.search('old');
  const latest = search.search('new');
  assert.ok(requests.slice(0,2).every(r => r.signal.aborted));
  const before = snapshots.length;
  requests[0].finish([{title:'stale'}]); requests[1].finish([]);
  await old;
  assert.equal(snapshots.length, before);
  requests[2].finish([{title:'current'}]); requests[3].finish([]);
  await latest;
  assert.equal(snapshots.at(-1).youtube.results[0].title, 'current');
  const cleared = search.search('cleared');
  search.cancel();
  const count = snapshots.length;
  requests[4].finish([]); requests[5].finish([]);
  await cleared;
  assert.equal(snapshots.length, count);
});

test('preview uses provider embeds, never arbitrary pages or autoplay', () => {
  const yt = previewURL({provider:'youtube', url:'https://www.youtube.com/watch?v=n5sjpCQhTDk'}, 'http://localhost:8746');
  assert.equal(new URL(yt).pathname, '/embed/n5sjpCQhTDk');
  assert.equal(new URL(yt).searchParams.get('autoplay'), '0');
  assert.equal(new URL(yt).searchParams.get('origin'), 'http://localhost:8746');
  const bili = previewURL({provider:'bilibili', url:'https://www.bilibili.com/video/BV1bK411W797'}, '');
  assert.equal(new URL(bili).searchParams.get('bvid'), 'BV1bK411W797');
  assert.equal(new URL(bili).searchParams.get('autoplay'), '0');
  assert.equal(previewURL({provider:'youtube', url:'https://www.youtube.com.attacker.test/watch?v=n5sjpCQhTDk'}, ''), '');
});
