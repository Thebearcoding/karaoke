import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSearch, previewURL, searchLibrary, visibleResults, playCount, orderResults} from './web/search.js';

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

test('song, artist and catalogue searches stay separate and discard stale mode responses', async () => {
  const {search, requests, snapshots} = setup();
  const song = search.search('Singer', 'songs');
  assert.equal(new URL(requests[0].url, 'http://localhost').searchParams.get('kind'), 'songs');
  const artist = search.search('Singer', 'artists');
  assert.ok(requests[0].signal.aborted);
  requests[1].finish([{title:'Singer',browse_id:'UC'+'a'.repeat(22)}]);
  await artist;
  const count = snapshots.length;
  requests[0].finish([{title:'stale song'}]);
  await song;
  assert.equal(snapshots.length, count);
  const catalogue = search.search('UC'+'a'.repeat(22), 'artist');
  assert.equal(requests[2].url, '/v1/music/artists/UC'+'a'.repeat(22));
  requests[2].finish(Array.from({length:150}, (_,i) => ({title:`Song ${i}`})));
  await catalogue;
  assert.equal(snapshots.at(-1).artist.results.length, 150);
  await search.search('Singer', 'local');
  assert.equal(requests.length, 3);
});

test('popularity uses reported counts, keeps unknowns last and never mutates provider order', () => {
  const songs = [{title:'A',views:null},{title:'B',views:'21K'},{title:'C',views:'1.2M'},
    {title:'D',views:'1,300',status:'completed'},{title:'E',views:'21K'}];
  assert.deepEqual(orderResults(songs,'relevance'), songs);
  assert.deepEqual(orderResults(songs,'popular').map(s=>s.title), ['C','B','E','D','A']);
  assert.deepEqual(orderResults(songs,'ready').map(s=>s.title), ['D','A','B','C','E']);
  assert.deepEqual(songs.map(s=>s.title), ['A','B','C','D','E']);
  for (const value of [null, undefined, '', 'unknown', -1, Infinity]) assert.equal(playCount(value), null);
  assert.equal(playCount('0'), 0);
  assert.equal(playCount('1.2B'), 1.2e9);
  assert.ok(previewURL({provider:'youtube_music',url:'https://www.youtube.com/watch?v=abcdefghijk'},'http://localhost').includes('/embed/abcdefghijk'));
});
