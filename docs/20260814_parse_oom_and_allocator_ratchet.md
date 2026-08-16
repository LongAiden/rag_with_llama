# F24 — Peak RSS is O(total pages), not O(batch size) (2026-08-14)

> Follows §15.11 (VLM/convert pipelining), whose closing paragraph named a
> 3097 MB peak against a 3.5G limit as "the number to watch if a larger PDF
> appears". A larger PDF appeared. Plan:
> `docs/plans/20260814_parse_oom_702_page_pdf.md`.

## Problem

`celery_worker_upload` was SIGKILLed by the cgroup at 17:48:07 while parsing a
702-page, 13 MB book — batch 8 of 15, 17 seconds into `convert()`.

```
[17:48:04,599: INFO/ForkPoolWorker-4] VLM call #60: … elapsed=3.08s … done=stop
[17:48:07,303: ERROR/MainProcess] Process 'ForkPoolWorker-4' pid:748 exited with 'signal 9 (SIGKILL)'
[17:48:07,393: ERROR/MainProcess] Task handler raised error: WorkerLostError('Worker exited prematurely: signal 9 (SIGKILL) Job: 3.')
```

There is no traceback from inside the parse because nothing raised.
`WorkerLostError` is the parent process observing a dead fork child; the
billiard frames in the log are the *parent's* stack, not the failure site. Read
as an application error it is a dead end — the whole signal is in the RSS series
that precedes it.

### The measurement

Post-convert RSS, one sample per batch (`logs/celery_worker_upload.log:1333-1550`):

| batch | pages | rss | delta |
|---|---|---|---|
| 1 | 1-50 | 1587 MB | — |
| 2 | 51-100 | 1944 MB | +357 |
| 3 | 101-150 | 2265 MB | +321 |
| 4 | 151-200 | 2504 MB | +239 |
| 5 | 201-250 | 2818 MB | +314 |
| 6 | 251-300 | 2736 MB | −82 |
| 7 | 301-350 | 2759 MB | +23 |
| 8 | 351-400 | **killed** | — |

Convert rate was healthy throughout (1.29-1.58 s/page), VLM calls were healthy
(2-9 s, `done=stop`, 60 calls, 0 failures). Nothing was slow. Memory simply
walked into the ceiling.

### What this falsifies

`parse_pdf`'s docstring, §4.6.3 and §15.1 all claimed:

> peak memory is O(batch size) rather than O(total pages)

**That was false**, and the 504-page NLTK run which appeared to validate it was
actually sitting at 3097 MB against 3.5G — 86% utilised, ~487 MB of headroom.
The invariant had been "confirmed" by the only document large enough to test it
and small enough to survive.

## Root cause — two mechanisms, one of which batch size does not touch

### 1. Working set, proportional to batch size

`_build_converter()` sets `generate_page_images = True` at `images_scale = 2.0`
(144 DPI), so `doc.pages[n].image` holds a rendered PIL image for **every** page
in the batch, from `convert()` until `del doc`. At ~5.2 MB/page that is ~260 MB
per 50-page batch.

`_expand_and_crop` is the only consumer, and it only needs pages that carry a
`PictureItem` — **10 of 50** in batch 7. The other 40 renders are pure cost.

### 2. An allocator ratchet, independent of batch size

RSS steps up per batch and never comes back down. But:

- `del doc, page_items` does run, before the emit, every batch.
- `tests/unit/test_pdf_parser_streaming.py::test_no_document_survives_the_parse`
  holds only weakrefs and passes — no `DoclingDocument` is retained.

So nothing is leaking in Python. The blocks are freed to **glibc**, not to the
kernel: with `DOCLING_NUM_THREADS=6` and glibc's default `MALLOC_ARENA_MAX` of
8×NCPU, docling's threads each strand freed pages in their own arena, and torch's
CPU caching allocator holds more. The decelerating deltas and the plateau near
2.75 GB are the signature of arena reuse rather than unbounded retention — a
true leak would not flatten.

The fatal event is then: a ~2.76 GB floor, plus batch 8's transient (50 page
renders + six threads of layout/TableFormer activations + batch 7's crop
backlog), crossing 3584 MB partway through the convert.

### 3. The instrumentation could not have caught it

`peak_rss` was sampled only *after* `convert()` returned, after `del doc`, and
after the final emit. The peak that kills the process happens **inside**
`convert()`. Every memory figure in §15 is a snapshot taken at a local minimum,
i.e. a lower bound — including the 3097 MB that was treated as a measured peak.

### 4. And the document then hid for ~9 hours

`task_acks_late=True` with Celery's default `task_reject_on_worker_lost=False`
means a SIGKILLed task is not requeued: it fails with `WorkerLostError`, and
because SIGKILL bypasses the `except` in `_run_stage`, nothing marks the row.
The document kept `stage='parsing'` with a live claim. Recovery needs
`INGESTION_CLAIM_TIMEOUT_MINUTES=180` to elapse **and** the next 6-hourly
`recover_and_dispatch` tick — up to ~9 hours during which the document looks
like it is being worked on and its retry budget is never spent.

## Changes

### `src/app/ingestion/processors/gemini_docling_parser.py`

- **`_release_freed_memory()`** — `gc.collect()` then `libc.malloc_trim(0)`,
  called immediately after `del doc, page_items`. Ordering matters: trimming
  after the next `convert()` has already allocated leaves both batches resident
  at the same instant, which is the peak that kills the process.
- **`_resolve_malloc_trim()`** — resolves the symbol once and caches the result
  *including the failure*, because this runs once per batch and `CDLL` is not
  free. A missing `libc.so.6` (macOS, musl) is a silent no-op: those allocators
  have neither the per-thread arenas this counters nor the call itself.
- **`_PeakRssSampler`** — daemon thread sampling `_rss_mb()` every 0.5 s across
  the batch loop, so the in-`convert()` peak is finally observable. Daemon, so a
  crash in the parse cannot leave it holding the process open.
- **Per-batch RSS at three points** — after `convert()` (existed), after the
  release (`Released batch N-M: rss=… (freed …MB)`), and after the emit. The
  delta across the release is the number that says whether the trim worked.
- **Summary line** now carries `peak_rss=` (sampled) *and* `post_convert_rss=`.
  The old number is kept deliberately: every historical measurement in `docs/` is
  that number, and silently redefining it would break the series.
- **`_DOCLING_PAGE_BATCH_SIZE` 50 → 40**, and the docstring's memory claim
  rewritten to distinguish working set from peak RSS.

### `src/app/worker/ingestion_tasks.py`

- **`_record_worker_lost`** — a `task_failure` handler that marks the document
  `error` with the correct `error_stage` when, and only when, the exception is a
  `WorkerLostError`. It runs in `MainProcess`, which outlives the child and is
  the only place the kill is observable. Ordinary exceptions are ignored:
  `_run_stage` already recorded those, and recording twice would spend two
  attempts on one failure. Best-effort — a DB error inside the handler is logged,
  never raised, because the stale sweep is still the backstop.

### `docker-compose.yml`

- `MALLOC_ARENA_MAX: "2"` and `MALLOC_TRIM_THRESHOLD_: "131072"` in
  `x-common-env`. The higher-leverage half of the fix: `malloc_trim` cannot
  return what a stranded arena still owns.
- `DOCLING_PAGE_BATCH_SIZE` 50 → 40 (~260 MB → ~208 MB of page renders).
- `celery_worker_upload.memory` 3.5G → 4G, `celery_worker_ingestion.memory`
  3G → 4G. See "The recovery path was the more dangerous one" below.

### Config and tests

- `AppSettings.docling_page_batch_size` default 50 → 40; `.env.example` updated.
- `tests/unit/test_pdf_parser_streaming.py` — `TestMemoryRelease` (6) and
  `TestPeakRssSampler` (3): release called once per batch, ordered *before* the
  next convert, `malloc_trim(0)`, no-op without glibc, failure does not
  propagate, resolution cached, sampler sees a peak no between-batch sample
  would, thread stops on exit, summary carries both numbers.
- `tests/unit/test_ingestion_tasks.py` — `TestWorkerLostHandler` (9): stage
  mapping per task, `error_stage` propagation, ordinary exceptions ignored,
  document-less tasks ignored, DB failure contained, signal actually connected.

`tests/unit`: **605 passed / 9 failed**, the 9 being the documented pre-existing
baseline (§10.3) — unchanged by this work.

## Sizing — what each change is actually worth

| change | headroom bought |
|---|---|
| `celery_worker_upload` 3.5G → 4G | ~512 MB |
| `celery_worker_ingestion` 3G → 4G | ~1024 MB, on the path that had the least |
| batch 50 → 40 | ~52 MB |
| `_release_freed_memory()` + `MALLOC_ARENA_MAX` | the ~1.2 GB ratchet, if the diagnosis holds |

The first two are the margin that makes the next measurement safe to take. They
are not the fix, and it would be a mistake to read a surviving run as evidence
that they were — the run may simply have plateaued under a higher ceiling.

## Status: not yet measured

Everything above except the crash trace is **predicted**. The gate is a re-run of
the 702-page book:

- per-batch RSS **flat**, i.e. the freed delta at the release ≈ the batch's own
  allocation rather than ~0;
- sampled `peak_rss` < 3200 MB (< 80% of 4G);
- `rate=Xs/page` within ~10% of the 1.29-1.58 s/page baseline;
- 702 strictly-ascending `[PAGE:n]` markers, `vlm_failures=0`.

If it still ratchets, the remaining cause is docling retaining page renders the
parse does not need, and the structural fix applies: `generate_page_images=False`
with crops rendered on demand from `pypdfium2` inside `_expand_and_crop`. That
method is the sole consumer of `page.image` and already does the
pdf-point→pixel math including the y-flip, so the change is contained — but
`generate_picture_images` must stay `True`, because `item.get_image(doc)` falls
back to the page image when a `PictureItem` carries none.

## The recovery path was the more dangerous one

Found while answering "do I have to re-upload the document?", and worth
recording because it is not visible from the crash log at all.

Two paths dispatch the ingestion chain, and they do not go to the same worker:

| path | queue | worker | limit before |
|---|---|---|---|
| `POST /upload` (`document_routes.py:110`) | `upload` | `celery_worker_upload` | 3.5G |
| `recover_and_dispatch`, weekly scan (`_dispatch_pending`, `ingestion_tasks.py:310-327`) | `ingestion` | `celery_worker_ingestion` | **3G** |

`_dispatch_pending` hardcodes `INGESTION_QUEUE`. So the *recovery* path — the one
that picks up a document after exactly this kind of failure, unattended, on a
6-hourly timer — was handing large PDFs to the worker with **less** headroom
than the one that had just died on 3.5G. A retry of this book through recovery
would have OOM-killed faster than the original run, and `INGESTION_MAX_ATTEMPTS`
would have quietly spent both attempts and marked it `failed`.

`celery_worker_ingestion` is therefore also 4G. The two workers run the same
image and the same pipeline and receive the same documents; sizing them
differently only decides which path fails.

## Also worth knowing

- **The limits now overcommit the VM.** 4G + 4G (workers) + 1G (postgres) +
  768M (app) + 192M (beat) against a 6.77 GiB Docker VM. Limits are ceilings,
  not reservations, and the two workers are rarely both parsing, so this costs
  nothing until they are — the case §15.2 already names as the one that
  exhausts it. If it bites, scale `celery_worker_ingestion` to 0 for the
  duration of a large interactive upload; do not shrink its limit back, because
  that just restores the trap above.
- **This limit has now been raised twice by an OOM kill**, and both times the
  previous value had been recorded as a measured floor. Both measurements were
  honest and both were against a single document. The lesson is not "measure
  more" — it is that a peak sampled between batches was never the peak, which is
  what `_PeakRssSampler` exists to stop repeating.
