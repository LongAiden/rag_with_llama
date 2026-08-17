# Parse-stage OOM on a 702-page PDF — diagnosis and fix

## Context

`celery_worker_upload` was SIGKILLed by the cgroup at 2026-08-13 17:48:07 while
parsing `1d09bdaa-…_Building Machine Learning Systems with a Feature Store 6.pdf`
(702 pages, 13MB). This is a **cgroup OOM kill, not an application exception** —
`WorkerLostError: signal 9` is the parent observing a dead fork child, and there is
no traceback from inside the parse because there was no Python-level failure.

### What the log actually shows

Post-convert RSS, one sample per batch (`logs/celery_worker_upload.log:1333-1550`):

| batch | pages | rss | delta |
|---|---|---|---|
| 1 | 1-50 | 1587MB | — |
| 2 | 51-100 | 1944MB | +357 |
| 3 | 101-150 | 2265MB | +321 |
| 4 | 151-200 | 2504MB | +239 |
| 5 | 201-250 | 2818MB | +314 |
| 6 | 251-300 | 2736MB | −82 |
| 7 | 301-350 | 2759MB | +23 |
| 8 | 351-400 | **killed 17s into `convert()`** | — |

The limit is `celery_worker_upload: memory: 3.5G` (`docker-compose.yml`). The
previous NLTK run (504 pages) peaked at **3097MB = 86% of the limit**
(`ARCHITECTURE.md` §15.11). There was never headroom for a longer document; 702
pages was simply the first one to spend it.

### Why memory grows — two separate causes

`parse_pdf`'s docstring claims *"peak memory is O(batch size) rather than O(total
pages)"* (`gemini_docling_parser.py:924-927`). **That is false as measured.**

1. **Per-batch working set — proportional to batch size.** `_build_converter()`
   sets `generate_page_images = True` at `images_scale = 2.0` (144 DPI), so
   `doc.pages[n].image` holds a PIL render of *every* page in the batch from
   `convert()` until `del doc`. At ~1150×1500 RGB that is ~5.2MB/page →
   **~260MB per 50-page batch**, plus docling's own backend page buffers live
   during the convert itself. `_expand_and_crop` (`:629`) is the only consumer,
   and it only needs pages that carry a `PictureItem` — in batch 7 that was 10 of
   50 pages.

2. **A ratchet that batch size does not touch.** RSS steps up ~250-350MB per batch
   and never comes back down, then plateaus around 2.75GB. `del doc, page_items`
   (`:1025`) does run, and `tests/unit/test_pdf_parser_streaming.py`'s weakref
   check (`test_no_document_survives_the_parse`) proves no `DoclingDocument`
   survives — so this is **freed-but-not-returned-to-the-OS** memory, not a Python
   leak: glibc per-thread arenas (`DOCLING_NUM_THREADS=6`, default `MALLOC_ARENA_MAX`
   is 8× cores) plus torch's CPU caching allocator. The deceleration and plateau
   are the signature of arena reuse rather than unbounded retention.

3. **Pipelining adds a fixed slice.** §15.11 measured +211MB from holding one
   batch's built pages and in-flight VLM futures across the next `convert()`.
   Real but small next to (1) and (2), and not worth reverting — it bought −22%
   wall time.

The fatal event is therefore: a ~2.76GB floor + the batch-8 transient
(page renders + 6 threads of layout/TableFormer activations + the batch-7 crop
backlog) crossing 3584MB partway through the convert.

### Instrumentation gap

`peak_rss` in the `parse_pdf summary:` line is sampled only *after* `convert()`
returns, after `del doc`, and after the final emit (`:994, :1026, :1046`). The
peak that actually kills the process happens **inside** `convert()` and has never
been observed. Every memory number in `ARCHITECTURE.md` §15 is a post-convert
snapshot, i.e. a lower bound.

### Secondary finding — the document is stuck for ~9 hours

`task_acks_late=True` with Celery's default `task_reject_on_worker_lost=False`
means a SIGKILLed task is not requeued; it fails with `WorkerLostError`, and
because SIGKILL bypasses the `except` in `_run_stage`, nothing marks the row.
Doc `1d09bdaa` is still `stage='parsing'` with a live claim. Recovery needs
`INGESTION_CLAIM_TIMEOUT_MINUTES=180` to elapse **and** the next
`recover_and_dispatch_6h` tick (`celery_app.py:40-44`) — up to ~9 hours of
invisible stall.

### Intended outcome

A 702-page PDF parses inside the existing 3.5G limit, peak RSS becomes flat in
total page count rather than climbing per batch, the summary line reports a real
peak, and an OOM kill surfaces within minutes instead of hours.

Scope decisions from the user:

- **~700 pages is the ceiling** — take the allocator + batch-size route and keep
  the on-demand-crop rewrite as a gated fallback rather than doing it up front.
- **Raise `celery_worker_upload` to 4G** and set `DOCLING_PAGE_BATCH_SIZE=40`.

---

## Step 0 — Unstick the current document (user runs)

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "UPDATE documents SET stage='registered', claimed_at=NULL, claimed_by=NULL
   WHERE id='1d09bdaa-f450-4ff0-ad01-44f947c6292d';"
```

Leave `attempts` alone — it should carry the failure. Check it is below
`INGESTION_MAX_ATTEMPTS=2` first, and reset to 0 if not.

---

## Step 1 — Make the real peak visible

**`src/app/ingestion/processors/gemini_docling_parser.py`**

Reuse the existing `_rss_mb()` helper; do not add a dependency.

- Add a `_PeakRssSampler` — a daemon thread that calls `_rss_mb()` every 0.5s and
  keeps the max, started/stopped around the whole batch loop with a context
  manager. Fold its max into `peak_rss`. This is the only way to see the
  in-`convert()` peak.
- Log RSS at three points per batch instead of one: after `convert()` (exists),
  after `del doc, page_items` (currently sampled into `peak_rss` but not logged),
  and after `_emit_batch`. The delta across `del doc` is the number that tells us
  whether Step 2 worked.
- Extend the `parse_pdf summary:` line with `peak_rss` (sampled) alongside the
  existing post-convert value, e.g. `peak_rss=3400MB post_convert_rss=2759MB`.

Rewrite the `parse_pdf` docstring's memory claim (`:924-927`) — it currently
asserts an invariant the measurements contradict.

---

## Step 2 — Return memory to the OS between batches

**`src/app/ingestion/processors/gemini_docling_parser.py`**, immediately after
`del doc, page_items` (`:1025`):

- Module-level `_malloc_trim()` helper: resolve `libc.malloc_trim` once via
  `ctypes.CDLL("libc.so.6")`, cache the result (including the "not available"
  case), and no-op on macOS/musl. Must never raise — a missing symbol cannot
  fail a parse.
- Call `gc.collect()` then `_malloc_trim()`. Both are milliseconds against a
  ~75s convert; do not gate them behind a config flag.

**`docker-compose.yml`**, in the `x-common-env` anchor:

- `MALLOC_ARENA_MAX: "2"` — with `DOCLING_NUM_THREADS=6` glibc will otherwise
  open up to 8×NCPU arenas, each holding freed blocks the process never reuses.
  This is the single highest-leverage change for the ratchet.
- `MALLOC_TRIM_THRESHOLD_: "131072"` — makes the allocator trim on `free()`
  rather than only at the explicit `malloc_trim` call.

Both need `docker compose up -d --force-recreate celery_worker_upload
celery_worker_ingestion` (§15.5: a plain `restart` does not pick up env changes).

---

## Step 3 — Raise the ceiling and shrink the per-batch working set

**`docker-compose.yml`**

- `celery_worker_upload`: `memory: 3.5G → 4G`. Update the block comment above it —
  it currently reads "3.5G is the only value observed to hold", which this
  incident disproves.
- `x-common-env`: `DOCLING_PAGE_BATCH_SIZE: 50 → 40`.

Batch 40 cuts retained page images from ~260MB to ~208MB and shrinks the crop
backlog held across the next convert by the same ratio: **~52MB off the transient
peak**, and 18 converts instead of 15 for this book. Cost is a few percent of
`docling` wall time, visible in the existing `Converted pages … rate=Xs/page`
line, directly comparable to the 1.29-1.58s/page already logged.

Sizing note, because the two changes are not equal in weight: 4G buys ~512MB and
batch 40 buys ~52MB — against a ratchet that had already consumed ~1.2GB by batch
5. The run *may* now survive on headroom alone, since RSS looked like it was
plateauing near 2.75GB (batches 6-7: −82, +23), but that is a three-sample
plateau on a run that then died. **Step 2 is still the load-bearing fix**; treat
4G/40 as the margin that makes the next measurement safe to take, not as the
solution.

Overcommit to flag, not to act on: the Docker VM is 6.77G and the limits now sum
to 4G + 3G + 1G (postgres) + 768M (app) + 192M (beat). Limits are ceilings, not
reservations, so this only bites if `celery_worker_upload` and
`celery_worker_ingestion` parse large documents simultaneously — the case
`ARCHITECTURE.md` §15.2 already names as the one that exhausts the VM. If that
starts happening, drop `celery_worker_ingestion` to 2.5G rather than raising the
VM.

Change Step 3 **separately from Step 2** and re-measure between them, per §15.7's
warning about F12 (two simultaneous changes on two documents produced a 4×
improvement nobody could attribute). Order: Step 2 first — it is the one whose
effect the new per-batch RSS logging can actually attribute.

---

## Step 4 — Fail fast instead of stalling for 9 hours

**`src/app/worker/ingestion_tasks.py`**

Register a Celery `task_failure` signal handler for the ingestion tasks that, when
`einfo` is a `WorkerLostError`, marks the document `error` with
`error_stage='parse'` and a message naming the signal. This runs in `MainProcess`,
which survives the child's SIGKILL, so it is the only place the event is
observable.

Reuse `IngestionRepository.record_error()` — it already increments `attempts` and
sets `error_stage` for the retry-resumption path
(`src/app/infra/db/ingestion_repository.py`). Dispatch it through the persistent
event loop in `_run()`, **not** `asyncio.run()` (§4.5 / §9.2).

This makes the existing `INGESTION_MAX_ATTEMPTS=2` budget work as intended for
OOM kills, instead of the row waiting 180 minutes to look stale.

---

## Step 5 — Gate: is Step 2+3 enough?

Re-run the 702-page book and read the new sampled `peak_rss`.

- **Peak flat across batches and under ~3.2GB** (80% of the new 4G) → done.
  Update `ARCHITECTURE.md` §15.11 and the compose comment on
  `celery_worker_upload.memory` with the new measurement, and correct
  §15.1/§15.11's "O(batch size)" claim.
- **Peak still ratchets** → the remaining cause is docling retaining page renders
  we do not need, and the structural fix applies: set
  `generate_page_images = False` in `_build_converter()` and render crops on
  demand inside `_expand_and_crop` via `pypdfium2` (`page.render(scale=self._images_scale)`),
  cropping the same pixel box the method already computes. `_expand_and_crop`
  (`:629-652`) is the sole consumer of `page.image`, and the pdf-point→pixel
  math including the `ph - t` y-flip is already there, so the change is
  contained. Note `item.get_image(doc)` (`:724, :727, :763`) falls back to the
  page image when a `PictureItem` has none, so `generate_picture_images` must
  stay `True`. **Do not start this before the gate** — it is unnecessary work if
  the allocator was the whole story.

Fallback lever if a document larger than ~700 pages appears: drop
`celery_worker_ingestion` to 2.5G and give `celery_worker_upload` 4.5G, accepting
that two large parses cannot run concurrently. Not part of this plan.

---

## Verification

```bash
# 1. Rebuild env, force-recreate so the new env vars land
docker compose up -d --force-recreate celery_worker_upload celery_worker_ingestion

# 2. Unit tests — the streaming/memory contract
python -m pytest tests/unit/test_pdf_parser_streaming.py -v

# 3. Re-parse the 702-page book (Step 0 SQL, then re-dispatch via the UI or the
#    upload endpoint) and watch memory live in a second terminal
docker stats rag_celery_worker_upload --format "{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"

# 4. The numbers that decide it
grep -E "Converted pages|after del|parse_pdf summary" logs/celery_worker_upload.log | tail -60

# 5. Confirm no kill
docker inspect rag_celery_worker_upload --format '{{.State.OOMKilled}}'
```

Pass criteria:

- No `signal 9` in `logs/celery_worker_upload.log`; `OOMKilled` is `false`.
- Per-batch RSS is **flat**, not monotone — the step across `del doc` should now
  be roughly zero rather than +250-350MB.
- Sampled `peak_rss` < 3200MB (< 80% of the new 4G) for 702 pages.
- `rate=Xs/page` within ~10% of the 1.29-1.58s/page baseline.
- Output sanity: `data/parsed/1d09bdaa-…md` has 702 strictly-ascending `[PAGE:n]`
  markers and `vlm_failures=0`. Output determinism holds because
  `OLLAMA_VLM_TEMPERATURE=0.0` (§15.9), so a re-run of the same book is
  byte-comparable.

New unit tests to add:

- `_malloc_trim()` no-ops cleanly when `libc.so.6` or the symbol is absent
  (patch `ctypes.CDLL` to raise) — a parse must not die on a non-glibc host.
- The peak sampler reports a max above the value visible at any single
  post-convert sample (drive it with a fake `_rss_mb` sequence).
- The `WorkerLostError` handler calls `record_error` with `error_stage='parse'`
  and ignores non-`WorkerLostError` failures.

## Docs to update after measurement

- `ARCHITECTURE.md` §4.6.3 "Memory invariant" and §15.1/§15.11 — the O(batch size)
  claim, with the corrected model (per-batch working set + allocator floor).
- `ARCHITECTURE.md` §15.5 — `MALLOC_ARENA_MAX`, `MALLOC_TRIM_THRESHOLD_`, and the
  new `DOCLING_PAGE_BATCH_SIZE` default.
- `docker-compose.yml` — the `celery_worker_upload.memory` comment currently says
  3.5G "is the only value observed to hold"; that is now known to be
  document-size-dependent.
