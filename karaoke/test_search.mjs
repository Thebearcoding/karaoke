import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSearch, previewURL, searchLibrary, visibleResults} from './web/search.js';

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

test('local matches appear immediately and update while online providers are pending', async () => {
  let library = [{url:'local', title:'弯弯的月亮', artist:'刘欢', status:'running'}];
  const snapshots = [], requests = [];
  const search = createVideoSearch(value => snapshots.push(structuredClone(value)), () => new Promise(resolve => requests.push(resolve)), query => searchLibrary(library, query));
  const pending = search.search('刘欢 月亮');
  assert.equal(snapshots[0].local.results[0].title, '弯弯的月亮');
  assert.equal(snapshots[0].youtube.loading, true);
  library[0].status = 'completed';
  search.refreshLocal();
  assert.equal(snapshots.at(-1).local.results[0].status, 'completed');
  requests.forEach(resolve => resolve({ok:true,json:async () => ({results:[]})}));
  await pending;
  assert.equal(snapshots.at(-1).local.results.length, 1);
  search.cancel();
  const count = snapshots.length;
  search.refreshLocal();
  assert.equal(snapshots.length, count);
});

test('combined results prefer local, deduplicate sources, and keep source tabs usable', () => {
  const songs = [{url:'b',provider:'bilibili'}, {url:'a',provider:'youtube'},
    {url:'a',provider:'local'}, {url:'c',provider:'youtube'}, {url:'d',provider:'youtube'}];
  assert.deepEqual(visibleResults(songs,'all',false).map(s => [s.url,s.provider]),
    [['a','local'],['c','youtube'],['d','youtube']]);
  assert.deepEqual(visibleResults(songs,'local',true), [songs[2]]);
  assert.equal(visibleResults(songs,'youtube',false)[0],songs[1]);
  assert.deepEqual(visibleResults([songs[0]],'all',true),[]);
  assert.deepEqual(visibleResults([songs[0]],'bilibili',true),[songs[0]]);
});
