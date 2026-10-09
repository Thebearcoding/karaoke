# Keep playback state current and exercise the real 100-song queue

## Purpose

Fix stale playback across iPad tabs, reclaim unused intermediate audio before
valuable media, then move the fixed100 validation cache into the user's live
service and exercise its normal queue through complete playable MV output.
The user explicitly authorized the data move and queue population. Preserve
the existing Blue Bird entry, history, settings, credentials, and source files.

## Progress

- [x] Inspect current code, production state, and the completed r54 evidence.
- [x] Reproduce stale-tab and audio-ready cleanup failures with behavior tests.
- [x] Fix the owning state/cache mechanisms and document the behavior.
- [ ] Run full regression, fresh-directory startup, commit, push, and CI.
- [ ] Inventory and migrate job directories/records with rollback evidence.
- [ ] Deploy, verify real desktop/mobile interactions and aesthetic consistency.
- [ ] Add the fixed100 to the live queue and measure the normal worker E2E path.
- [ ] Retain all failures, repair general causes, and report actual results.

## Findings and Decisions

- Production is `<production-checkout>`, port 8746; isolated validation is
  `<validation-checkout>`, port 18946. Both use
  `karaoke/service-data`. Validation is idle; production currently has no worker
  running and has one queued-list entry (Blue Bird, already prepared).
- `LocalPlayback.reconcile` deliberately recovers removed entries from library
  metadata. Persisted/tab-restored state consequently revives deleted entries.
  Clarification requested: coordinate tabs in one browser while retaining
  independent transport on other devices. Queue and preparation remain shared.
- `SongCache.compact` retains all stage outputs for `audio_ready`; `enforce`
  evicts video before removing surplus intermediates. Complete audio/subtitles
  must be recognized by validated artifacts, not a full-MV status alone.
- Preserve appearance and existing controls. Design tokens, spacing, typography,
  and layout stay as documented in `karaoke/docs/design/app.md`; inspect real
  desktop and mobile content for regressions after changing state behavior.
- The fixed100 source currently contains 53 complete audio/subtitle results and
  47 source-only entries: the previous LRU already evicted those 47 outputs.
  All 100 original audio sources survive. The new run must distinguish these
  cases and cannot describe all 100 as cache hits.
- Same-browser playback sharing is the stated assumption while preserving
  independent playback on other devices. Search/navigation remain per-tab.
- JavaScript: 76 tests passed. Empty-data startup passed. Full Python regression:
  1536 passed and 446 subtests passed with NUMBA_CACHE_DIR in the allowed user
  cache directory (the default path prevented two librosa imports).
- Real Chrome tabs using a separate room and actual Blue Bird / Once Again media
  verified ownership handoff, removal followed by queue-head selection, stale
  page reload, seek, lead volume, pitch, and pause. Removing a paused song also
  preserves pause intent; it does not silently enable continuous playback.
- Desktop and 390x844 screenshots were reviewed: existing grouping, alignment,
  button sizes and text truncation remain consistent, with no new UI controls.
  This is browser/mobile-viewport validation, not a physical iPad verification.

## Work and Invariants

`karaoke/web/room-state.js` owns current selection and saved transport. Expired
queue identities cannot return through library metadata or late browser events.
Tab coordination must prevent stale saves/callbacks from overriding its current
owner. Starting playback on another physical device must not interrupt a singer.

`karaoke/cache.py`, `karaoke/api.py`, and `karaoke/room.py` own retained artifacts,
queue restoration, and eviction. Complete audio/subtitles survive compaction and
resume by adding video only. Active writers/transfers remain protected. Under
pressure, discard unused intermediates first, then reconstructible videos, then
whole unprotected songs in access order. Do not change lyric/alignment algorithms.

Migrate actual job directories, not just filesystem paths: preserve IDs, DB docs,
cache identity, final artifact hashes, prepared inputs, and a move journal. Merge
records into the live database without replacing room/history/settings. Move
same-filesystem directories atomically while both writers are stopped. Validate
all final artifacts before removing intermediate files. Source-only evicted jobs
remain source-only and must be processed honestly by the real worker.

## Validation and Evidence

Extend existing `test_player.mjs`, `test_api.py`, and `test_room.py` for stale
restoration, expired events, tab ownership, cleanup ordering, transfer protection,
audio-only completion, and video-only resume. Run the repository's full Python
and JS commands and empty-data startup. Use actual browser tabs, mobile viewport,
and desktop viewport for queue/playback transitions, including deleting the
current entry and returning to a background tab. Record screenshots and limits.

Use production `/v1/room/entries` and normal workers for the fixed100, not the
isolated `run_probe` shortcut. Record queue submission, stage timing, completion,
cache reuse, source identity, and actual failures per song. Existing cached work
means this is a warm/mixed-cache E2E run, not a cold performance benchmark.

## Recovery and Idempotence

Before deployment/migration, save source hashes, service configuration references,
SQLite backups, exact source/destination IDs and paths, and the existing queue.
Never copy secrets into evidence or Git. An interrupted move resumes from its
journal; an ID collision must be compared before overwrite. Rollback stops only
owned workers, restores the original DB snapshot, and reverses journaled moves.
Queue population records entry IDs so rerunning it cannot duplicate submissions.

## Outcomes

Pending. The preceding algorithm checkpoint is `57fdaf6`, r54 is fixed100 with
zero hard failures, and its report SHA is
`297d6faddf0404337b5fcf3978863b437adb71ede61a99039ee700a746f068b2`.
