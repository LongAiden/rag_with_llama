# Overlap the VLM wait with docling's convert

**Date**: 2026-08-12
**Status**: planned, not implemented.
**Follows**: `docs/plans/20260812_parse_time_reduction.md` (this is its Step 5, promoted to
its own document).

## Context

A 504-page PDF still takes ~17 minutes through the parse stage. That number is the result of
five rounds of optimization, not a regression, and the latest run confirms where it now sits:

```
parse_pdf summary: pages=504 total=1035s docling=749s (72%) assembly=286s (28%)
                   vlm_wait=281s vlm_calls=86 vlm_failures=0 peak_rss=2886MB
```

The previous change (`DOCLING_NUM_THREADS` and worker `limits.cpus` 4 → 6, applied
2026-08-12) is now **measured**: docling fell ~820s → 749s, about 9%, sub-linear because
threads 5 and 6 land on the M1's efficiency cores. Keep it; it is a small real win.

What remains is structural: **docling and Ollama never run at the same time.** During the
749s of `convert()` the Ollama process is idle. During the 281s of VLM decode the container's
six threads are idle. `_process_page` blocks on its own futures before returning
([gemini_docling_parser.py:828-843](../../src/app/ingestion/processors/gemini_docling_parser.py#L828-L843)),
so raising `VLM_CONCURRENCY` cannot help — there is nothing to pipeline *across* pages.

**Intended outcome**: hide the VLM decode underneath the next batch's convert. Per batch the
VLM costs ~28s against a ~75s convert, so essentially all of it fits. Ceiling is the full
281s minus the final batch's tail: **1035s → ~770s, about 13 minutes.**

Not in scope: MPS / running the worker natively. That is the larger lever (docling is 72%)
but it is unmeasured, the container installs CPU-only torch
([requirements.txt:5-7](../../deploy/deployment/requirements.txt#L5-L7)), and there is no
local Python env to benchmark in. Revisit after this lands.

## Design

All changes in `src/app/ingestion/processors/gemini_docling_parser.py`.

### 1. Split `_process_page` into build and finalize halves

Today `_process_page` does three things: walk the items building `ordered` fragments, submit
VLM crops to the executor collecting `vlm_tasks`, then **join** those futures and return
markdown. Split the join off:

- **`_build_page(page_no, items, doc, executor) -> (ordered, vlm_tasks)`** — everything up to
  and including `executor.submit(...)`. The body is the existing loop verbatim, returning the
  two lists instead of falling through to the join.
- **`_finalize_page(page_no, ordered, vlm_tasks) -> str`** — the existing `if vlm_tasks:`
  block, the `ordered.sort(...)`, and the `f"[PAGE:{page_no}]\n\n{body}"` return.

**Keep `_process_page` as a thin wrapper** over the two:

```python
def _process_page(self, page_no, items, doc, executor=None) -> str:
    ordered, vlm_tasks = self._build_page(page_no, items, doc, executor)
    return self._finalize_page(page_no, ordered, vlm_tasks)
```

This preserves the synchronous `executor=None` path and keeps `test_vlm_table_routing.py`
(which calls `_process_page` and asserts on the returned markdown) working untouched.

`vlm_tasks` entries are `(col, y, x, future, kind, caption)`. `future` closes over a PIL crop
produced by `_expand_and_crop`
([:624-647](../../src/app/ingestion/processors/gemini_docling_parser.py#L624-L647)) or
`item.get_image(doc)` — both return **independent** images — and `caption` is already a plain
`str`. Nothing in the tuple references `doc`, which is what makes step 2 safe.

### 2. Rework the `parse_pdf` batch loop to carry one pending batch

Replace the current convert → assemble → release sequence
([:889-949](../../src/app/ingestion/processors/gemini_docling_parser.py#L889-L949)) with:

```
pending = None            # (batch_start, batch_end, [(page_no, ordered, vlm_tasks), ...])
for each batch N:
    convert(N) -> doc                  # batch N-1's VLM calls decode during this
    built = [ (page_no, *self._build_page(...)) for page_no in batch N ]
    del doc, page_items                # doc N released before any join
    if pending: emit(pending)          # joins N-1's futures — already complete
    pending = (N, built)
emit(pending)                          # final batch, unavoidable serial tail
```

`emit()` runs `_finalize_page` per page, then the existing per-page normalization
(`_normalize_tables_in_markdown` → `_clean_html` → `_fix_table_closing_tags`), appends to
`pages_md` and writes to `out_file`.

Notes on why this is correct:

- **Output order is preserved.** Batches are emitted 1, 2, 3, … just one batch later than
  today. Streaming to `output_path` lags by one batch; nothing reorders.
- **Memory stays O(batch).** Only one `DoclingDocument` is alive at a time — `doc` for batch
  N is deleted *before* batch N-1 is finalized. The extra retention is one batch's crops
  (~8 partial-page images) plus its `ordered` strings: single-digit MB against a 2886 MB peak.
  This is the invariant the OOM comment at
  [:947-949](../../src/app/ingestion/processors/gemini_docling_parser.py#L947-L949) protects
  and it is not weakened.
- **`VLM_CONCURRENCY=1` still holds.** Batch N's futures queue behind N-1's in the same
  single-worker executor, so Ollama is never asked to serve parallel calls — F15 measured
  that as 27% slower at 2 and 5× slower at 4.
- **Error isolation.** The current per-page `try/except` that logs and `continue`s must wrap
  the build and the finalize halves *separately*, so a failure in either skips only that page.

### 3. Make the win visible in the summary line

`_vlm_seconds` is the sum of call durations — total Ollama *work*, which pipelining does not
change. Once overlapped it is no longer the wall time paid, so `assembly ≈ vlm_wait` stops
holding and the existing line becomes misleading.

Add `self._vlm_blocked_seconds`, accumulated under `_vlm_stats_lock` around the
`future.result()` call in `_finalize_page`, and add it to the summary:

```
vlm_wait=281s vlm_blocked=25s vlm_calls=86 …
```

`vlm_blocked` is the metric that proves the change worked: it should collapse from ~281s to
roughly one batch's worth. Keep `vlm_wait` as-is so the number stays comparable with every
prior run in `docs/`.

## Files

| File | Change |
|---|---|
| `src/app/ingestion/processors/gemini_docling_parser.py` | `_build_page` / `_finalize_page` split, `_process_page` wrapper, `parse_pdf` pending-batch loop, `_vlm_blocked_seconds` |
| `tests/unit/test_pdf_parser_streaming.py` | Patches `_process_page`, which `parse_pdf` no longer calls — repoint the fake at `_build_page`/`_finalize_page` |
| `tests/unit/test_pdf_parser_streaming.py` (new cases) | Page order across batches; one `doc` alive at a time; futures resolved exactly once |
| `docs/plans/20260812_parse_time_reduction.md` | Mark Step 4 **measured** (keep 6 threads, ~9%); point Step 5 at this document |

`tests/unit/test_vlm_table_routing.py` and `tests/unit/test_f23_structure_preservation.py`
should need **no changes** — that is the point of keeping the `_process_page` wrapper, and it
doubles as the regression check on the split.

## Verification

```bash
uv run pytest tests/unit/test_pdf_parser_streaming.py \
              tests/unit/test_vlm_table_routing.py \
              tests/unit/test_f23_structure_preservation.py \
              tests/unit/test_ollama_vlm_call.py -q
```

Then re-ingest the **same** NLTK.pdf (F12's lesson: cross-document comparison is meaningless,
a table-heavy book differs 4× from a prose-heavy one) and compare against
`total=1035s docling=749s vlm_wait=281s peak_rss=2886MB`:

```bash
grep -E "parse_pdf summary|Converted pages" logs/celery_worker_upload.log
```

| metric | now | expected |
|---|---|---|
| `total` | 1035s | **~770s** |
| `docling` | 749s | ~749s (unchanged — this change does not speed up convert) |
| `vlm_wait` | 281s | ~281s (unchanged — same calls, same decode) |
| `vlm_blocked` | — (was 281s implicitly) | **~25s** |
| `vlm_calls` / `vlm_failures` | 86 / 0 | 86 / 0 |
| `peak_rss` | 2886 MB | ~2886 MB — **a rise here means a `doc` is being retained** |

Correctness, not just speed — the output must match in structure:

```bash
grep -c '^\[PAGE:' data/parsed/<id>_NLTK.md    # 504, and in ascending order
grep -c '<figure>' data/parsed/<id>_NLTK.md    # == vlm_calls
```

Diff the new `data/parsed/*.md` against the current one. Page ordering and figure placement
must match; only run-to-run VLM wording should differ.
