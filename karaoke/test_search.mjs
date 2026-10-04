import test from 'node:test';
import assert from 'node:assert/strict';
import {createVideoSearch, previewURL, searchLibrary, visibleResults, playCount, orderResults} from './web/search.js';

function setup() {
  const requests = [], snapshots = [];
  const search = createVideoSearch(state => snapshots.push(structuredClone(state)), (url, {signal}) =>
    new Promise(resolve => requests.push({url, signal, finish: (results, ok = true, next_cursor = null) => resolve({ok, json:async () => ok ? {results, next_cursor} : {detail:'Provider failed'}})})));
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
  const album = search.search('MPREexample', 'album');
  assert.equal(requests[3].url, '/v1/music/albums/MPREexample');
  requests[3].finish([{title:'An album track',url:'album-track'}]);
  await album;
  assert.equal(snapshots.at(-1).album.results[0].title, 'An album track');
  await search.search('Singer', 'local');
  assert.equal(requests.length, 4);
});

test('popularity uses reported counts, keeps unknowns last and never mutates provider order', () => {
  const songs = [{title:'A',views:null},{title:'B',views:'21K'},{title:'C',views:'1.2M'},
    {title:'D',views:'1,300',status:'completed'},{title:'E',views:'21K'}];
  assert.deepEqual(orderResults(songs,'relevance'), songs);
  assert.deepEqual(orderResults(songs,'popular').map(s=>s.title), ['C','B','E','D','A']);
  assert.deepEqual(orderResults(songs,'ready').map(s=>s.title), ['D','A','B','C','E']);
  assert.deepEqual(songs.map(s=>s.title), ['A','B','C','D','E']);
  for (const value of [null, undefined, '', 'unknown', -1, Infinity, '2001年', '3:21']) assert.equal(playCount(value), null);
  assert.equal(playCount('0'), 0);
  assert.equal(playCount('1.2B'), 1.2e9);
  assert.equal(playCount('播放次数：5163万'), 51630000);
  assert.equal(playCount('3443万次观看'), 34430000);
  assert.equal(playCount('52M plays'), 52000000);
  for (const [value, expected] of [['2591万',25910000], ['1.2亿',120000000], ['２.５萬',25000], ['3千',3000], ['1.2 億',120000000]]) {
    assert.equal(playCount(value), expected);
  }
  assert.deepEqual(orderResults([{views:'2.5M'}, {views:'2591万'}, {views:'1.2亿'}], 'popular').map(s=>s.views), ['1.2亿','2591万','2.5M']);
  assert.ok(previewURL({provider:'youtube_music',url:'https://www.youtube.com/watch?v=abcdefghijk'},'http://localhost').includes('/embed/abcdefghijk'));
});

test('default lowers explicit live versions without removing results or changing other sorts', () => {
  const titles = ['A (Live)', 'Studio', 'The One and Only', 'B LIVE at Wembley', 'C 现场版',
    'Alive', 'Live and Let Die', 'D【LIVE】', 'E ライブ', 'Another studio', 'F Live | Tour'];
  const songs = titles.map((title,i) => ({title,views:100-i,kind:i === 2 ? 'album' : 'song'}));
  assert.deepEqual(orderResults(songs,'relevance').map(s=>s.title), [
    'Studio','The One and Only','Alive','Live and Let Die','Another studio',
    'A (Live)','B LIVE at Wembley','C 现场版','D【LIVE】','E ライブ','F Live | Tour']);
  assert.deepEqual(orderResults(songs,'popular'), songs);
  assert.deepEqual(orderResults(songs,'album'), songs);
  assert.deepEqual(songs.map(s=>s.title), titles);
});


test('pages append once, retain distinct versions, preserve rows on failure and stop at the end', async () => {
  const {search, requests, snapshots} = setup();
  const first = search.search('Singer', 'songs');
  const rows = [{url:'studio',title:'Song'}, {url:'live',title:'Song (Live)'}];
  requests[0].finish(rows, true, 'page-2');
  await first;
  const more = search.loadMore();
  await search.loadMore();
  assert.equal(requests.length, 2);
  assert.equal(new URL(requests[1].url, 'http://localhost').searchParams.get('cursor'), 'page-2');
  assert.deepEqual(snapshots.at(-1).songs.results, rows);
  requests[1].finish([], false);
  await more;
  assert.deepEqual(snapshots.at(-1).songs.results, rows);
  assert.equal(snapshots.at(-1).songs.moreError, 'Provider failed');
  const retry = search.loadMore();
  assert.equal(requests[2].url, requests[1].url);
  requests[2].finish([rows[1], {url:'cover',title:'Song'}], true, 'page-3');
  await retry;
  assert.deepEqual(snapshots.at(-1).songs.results.map(row=>row.url), ['studio','live','cover']);
  assert.equal(snapshots.at(-1).songs.moreError, '');
  const last = search.loadMore();
  requests[3].finish([{url:'last',title:'Last song'}]);
  await last;
  await search.loadMore();
  assert.equal(requests.length, 4);
  assert.equal(snapshots.at(-1).songs.results.length, 4);
  assert.equal(snapshots.at(-1).songs.nextCursor, null);
});

test('changing query or mode cancels pending pagination and an empty page never loops automatically', async () => {
  const {search, requests, snapshots} = setup();
  const first = search.search('Singer', 'songs');
  requests[0].finish([{url:'first'}], true, 'page-2');
  await first;
  const more = search.loadMore();
  const artists = search.search('Singer', 'artists');
  assert.ok(requests[1].signal.aborted);
  const count = snapshots.length;
  requests[1].finish([{url:'stale song'}], true, 'page-3');
  await more;
  assert.equal(snapshots.length, count);
  requests[2].finish([{browse_id:'artist-one'}], true, 'artist-next');
  await artists;
  const empty = search.loadMore();
  requests[3].finish([], true, 'artist-next-again');
  await empty;
  assert.equal(snapshots.at(-1).artists.autoBlocked, true);
  assert.equal(snapshots.at(-1).artists.nextCursor, 'artist-next-again');
  assert.equal(requests.length, 4);
  const manual = search.loadMore();
  requests[4].finish([{browse_id:'artist-two'}]);
  await manual;
  assert.equal(snapshots.at(-1).artists.results.length, 2);
  await search.search('local', 'local');
  await search.loadMore();
  assert.equal(requests.length, 5);
});

test('artist pages keep their metadata and expired cursors leave existing songs available', async () => {
  const requests = [], snapshots = [];
  const search = createVideoSearch(value => snapshots.push(structuredClone(value)), (url) => new Promise(resolve => requests.push({url,resolve})));
  const initial = search.search('UC'+'a'.repeat(22), 'artist');
  requests[0].resolve({ok:true,json:async () => ({title:'Singer',results:[{url:'song'}],next_cursor:'next'})});
  await initial;
  const more = search.loadMore();
  assert.ok(requests[1].url.endsWith('?cursor=next'));
  requests[1].resolve({ok:false,status:410,json:async () => ({detail:'搜索已过期，请重新搜索'})});
  await more;
  assert.equal(snapshots.at(-1).artist.expired, true);
  assert.equal(snapshots.at(-1).artist.metadata.title, 'Singer');
  assert.equal(snapshots.at(-1).artist.results[0].url, 'song');
});
