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

test('search calls only YouTube and preserves every returned result in order', async () => {
  const {search, requests, snapshots} = setup();
  const pending = search.search('Some song & singer');
  assert.equal(requests.length, 1);
  const query = new URL(requests[0].url, 'http://localhost').searchParams;
  assert.equal(query.get('q'), 'Some song & singer');
  assert.equal(query.get('provider'), 'youtube');
  const results = Array.from({length:35}, (_,i) => ({title:`Video ${i}`}));
  requests[0].finish(results);
  await pending;
  assert.deepEqual(snapshots.at(-1).youtube.results, results);
  assert.equal(snapshots.at(-1).youtube.loading, false);
  const failed = search.search('failure');
  requests[1].finish([], false);
  await failed;
  assert.equal(snapshots.at(-1).youtube.error, 'Provider failed');
});

test('new queries and clearing the input discard stale responses even if abort is ignored', async () => {
  const {search, requests, snapshots} = setup();
  const old = search.search('old');
  const latest = search.search('new');
  assert.ok(requests[0].signal.aborted);
  const before = snapshots.length;
  requests[0].finish([{title:'stale'}]);
  await old;
  assert.equal(snapshots.length, before);
  requests[1].finish([{title:'current'}]);
  await latest;
  assert.equal(snapshots.at(-1).youtube.results[0].title, 'current');
  const cleared = search.search('cleared');
  search.cancel();
  const count = snapshots.length;
  requests[2].finish([]);
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

test('YouTube view keeps provider order and has no result count or local deduplication filter', () => {
  const videos = Array.from({length:35}, (_,i) => ({url:`v${i}`,provider:'youtube'}));
  const local = {url:'v0',provider:'local'};
  const songs = [local,...videos];
  assert.deepEqual(visibleResults(songs,'youtube'), videos);
  assert.deepEqual(visibleResults(songs,'local'), [local]);
});
