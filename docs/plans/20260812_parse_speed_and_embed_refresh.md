# Plan: Parse Speed (VLM Pipelining) + Embed Tab Refresh Button

**Date**: 2026-08-12
**Status**: **implemented and measured 2026-08-12**. `total` 1035s → **808s**
(−22%), `vlm_blocked=0s`, output byte-for-byte identical. Full results in
[`ARCHITECTURE.md` §15.11](../ARCHITECTURE.md). Part A Step 0 (separate baseline
run) was skipped by decision — `vlm_blocked` proved self-evidencing.

Three things below were wrong against the code and were **not** followed literally:

1. The attribute is `_vlm_seconds`, not `_vlm_wait_seconds`; `vlm_wait` is only a
   log-field name.
2. **Step 3 as written deadlocks.** Accumulating "under `_vlm_stats_lock` around the
   `future.result()` call" hangs: `_record_vlm_call` takes that same lock from the pool
   thread. The wait is timed outside the lock and accumulated after.
3. `peak_rss` is expected to rise ~100-150 MB, not stay flat — one batch's PIL crops
   now live across the next `convert()`. The A.5 table below is corrected accordingly.
**Supersedes**: nothing; complements [`20260812_parse_pipelining.md`](./20260812_parse_pipelining.md).
**Scope**: `gemini_docling_parser.py`, `home.html`, tests.

---

## Part A — Reduce ~17 min parse to ~13 min (VLM/convert pipelining)

### A.1 Problem

A 504-page PDF takes ~1035s through the parse stage. The two phases never overlap:

| phase | time | share |
|---|---|---|
| docling `convert()` | 749s | 72% |
| VLM wait (86 Ollama calls) | 281s | 27% |
| non-VLM assembly | ~5s | <1% |

During 749s of docling, Ollama is idle. During 281s of VLM decode, the container's 6 CPU threads are idle. The ceiling for overlap is 281s minus the final batch's tail.

### A.2 Approach

Implement the pipelining design from [`20260812_parse_pipelining.md`](./20260812_parse_pipelining.md) — this plan is the execution checklist.

**Core change**: split `_process_page` into `_build_page` (submit VLM futures) and `_finalize_page` (join futures), then carry one pending batch across the next `convert()` call. Pipeline depth = 1.

```
Current:  convert(N) → assemble(N) [VLM blocks] → release → convert(N+1) → …
Pipelined: convert(N) → build(N) → convert(N+1) → finalize(N) [VLM already done] → build(N+1) → …
```

**Why this is safe**:
- VLM futures hold independent PIL crops from `_expand_and_crop` (`:624-647`) — no reference to `doc`
- `del doc` before join preserves O(batch) memory invariant
- `_process_page` kept as thin wrapper → existing tests untouched
- `VLM_CONCURRENCY=1` preserved — futures queue serially in the same single-worker executor

### A.3 Pre-flight: Step 0 (measure out-of-band cost) — **RESOLVED: 2.9s**

Answered by the post-change run rather than a separate baseline: `Stage parse
completed … in 810.9s` against `total=808s` puts the out-of-band cost at **2.9s**,
20× under the 60s threshold below. **Relaxing `--max-tasks-per-child=1` is ruled
out**, not merely unmeasured — see A.6.

Original method, kept for reference:

```bash
docker compose logs -f celery_worker_upload | tee logs/parse_step0_$(date +%Y%m%d).log
# Upload the 504-page NLTK.pdf through the UI
grep -E "parse_pdf summary|Stage .* completed" logs/parse_step0_*.log
```

`Stage parse completed for <id> in Ns` minus `total=Xs` = torch re-import + converter construction. If >60s, relaxing `--max-tasks-per-child=1` on `celery_worker_upload` jumps the queue (Step 3 from the reduction plan).

### A.4 Implementation steps

All changes in `src/app/ingestion/processors/gemini_docling_parser.py` unless noted.

#### Step 1 — Split `_process_page`

Extract the existing method into two halves:

- **`_build_page(page_no, items, doc, executor) -> (ordered, vlm_tasks)`** — the item-walking loop verbatim, returning the two lists instead of falling through to the join.
- **`_finalize_page(page_no, ordered, vlm_tasks) -> str`** — the existing `if vlm_tasks:` block (`:828-843`), `ordered.sort(...)`, and the `f"[PAGE:{page_no}]\n\n{body}"` return.

Keep `_process_page` as a thin wrapper:

```python
def _process_page(self, page_no, items, doc, executor=None) -> str:
    ordered, vlm_tasks = self._build_page(page_no, items, doc, executor)
    return self._finalize_page(page_no, ordered, vlm_tasks)
```

This preserves the synchronous `executor=None` path and keeps `test_vlm_table_routing.py` working.

#### Step 2 — Rework the `parse_pdf` batch loop

Replace the current convert → assemble → release sequence (`:888-952`) with a pending-batch pattern:

```python
pending = None  # (batch_start, batch_end, [(page_no, ordered, vlm_tasks), ...])

for batch_start in range(1, total_pages + 1, self._page_batch_size):
    batch_end = min(batch_start + self._page_batch_size - 1, total_pages)

    # Convert batch N
    doc = converter.convert(pdf_path, page_range=(batch_start, batch_end)).document
    page_items = ...  # group by prov[0].page_no

    # Build batch N (submit VLM futures, don't join)
    built = []
    for page_no in range(batch_start, batch_end + 1):
        try:
            ordered, vlm_tasks = self._build_page(page_no, page_items.get(page_no, []), doc, executor)
            built.append((page_no, ordered, vlm_tasks))
        except Exception as exc:
            logger.error(f"[page {page_no}] build failed: {exc}", exc_info=True)

    # Release batch N's doc BEFORE finalizing batch N-1
    del doc, page_items

    # Finalize batch N-1 (VLM futures already resolved during N's convert)
    if pending:
        self._emit_batch(pending, pages_md, out_file, total_pages)

    pending = (batch_start, batch_end, built)

# Final batch — unavoidable serial tail
if pending:
    self._emit_batch(pending, pages_md, out_file, total_pages)
```

`_emit_batch` runs `_finalize_page` per page, then the existing per-page normalization (`_normalize_tables_in_markdown` → `_clean_html` → `_fix_table_closing_tags`), appends to `pages_mx` and writes to `out_file`.

**Invariants to preserve**:
- Output order: batches emitted 1, 2, 3, … — streaming to `out_file` lags by one batch
- Memory: only one `DoclingDocument` alive at a time — `del doc` before `_emit_batch`
- Error isolation: per-page `try/except` wraps build and finalize separately

#### Step 3 — Add `_vlm_blocked_seconds` metric

Add `self._vlm_blocked_seconds`, timing the `future.result()` call in `_finalize_page`
and accumulating **after** the wait — see correction 3 in the header; taking
`_vlm_stats_lock` across `.result()` deadlocks against `_record_vlm_call`. Add to the
summary line:

```
vlm_wait=281s vlm_blocked=25s vlm_calls=86 …
```

`vlm_blocked` is the metric that proves the change worked — should collapse from ~281s to ~25s (one batch's tail). Keep `vlm_wait` as-is for historical comparability.

#### Step 4 — Update tests

| Test file | Change |
|---|---|
| `tests/unit/test_pdf_parser_streaming.py` | Repoint mock from `_process_page` to `_build_page`/`_finalize_page`; add page-order and single-doc-alive assertions |
| `tests/unit/test_vlm_table_routing.py` | No changes needed (calls `_process_page` wrapper) |
| `tests/unit/test_f23_structure_preservation.py` | No changes needed (string-level tests) |
| `tests/unit/test_ollama_vlm_call.py` | No changes needed |

### A.5 Verification

```bash
# Unit tests
uv run pytest tests/unit/test_pdf_parser_streaming.py \
              tests/unit/test_vlm_table_routing.py \
              tests/unit/test_f23_structure_preservation.py \
              tests/unit/test_ollama_vlm_call.py -q

# Integration: re-ingest the SAME 504-page NLTK.pdf
docker compose logs -f celery_worker_upload | tee logs/parse_pipelined_$(date +%Y%m%d).log
grep -E "parse_pdf summary|Converted pages" logs/parse_pipelined_*.log
```

| metric | before | expected | **measured** | verdict |
|---|---|---|---|---|
| `total` | 1035s | ~770s | **808s** | ✅ −22% |
| `assembly` | 286s | — | **3s** | ✅ the actual win |
| `docling` | 749s | ~749s (unchanged) | **804s** | ❌ +55s — CPU contention, see below |
| `vlm_wait` | 281s | ~281s (unchanged) | **333s** | ❌ +52s — same cause |
| `vlm_blocked` | — (was 281s implicitly) | ~25s | **0s** | ✅ better than predicted |
| `vlm_calls` / `vlm_failures` | 86 / 0 | 86 / 0 | **86 / 0** | ✅ |
| `peak_rss` | 2886 MB | ~2900-3050 MB | **3097 MB** | ⚠️ slightly over, 86% of the 3.5G limit |

**The two misses share one cause.** The plan assumed `docling` and `vlm_wait`
were independent. They are not on this host: Ollama runs locally and shares the
CPU (§15.8), so once VLM decode overlaps `convert()` the two compete. Per-batch
timings show the signature — batch 1 is *faster* (nothing to overlap yet), the
4-page final batch is unchanged (no VLM calls), every genuinely overlapping batch
pays 4-9s. Net still −227s. On a host with remote Ollama the penalty should
vanish entirely.

Correctness came out stronger than this plan anticipated: the artifacts are
**byte-for-byte identical** (md5 `db1845905fcb30cfd10c568757fdae04`), because
`OLLAMA_VLM_TEMPERATURE=0.0` makes the VLM deterministic. The "only run-to-run
VLM wording should differ" caveat below was unnecessary.

Correctness checks:

```bash
grep -c '^\[PAGE:' data/parsed/<id>_NLTK.md    # 504, ascending order
grep -c '<figure>' data/parsed/<id>_NLTK.md    # == vlm_calls
```

Diff the new `data/parsed/*.md` against the current one. **Measured result: the
two are byte-identical** (`cmp -s` clean, same md5), so the weaker "page ordering
and figure placement must match, only VLM wording differs" bar was not needed —
with `OLLAMA_VLM_TEMPERATURE=0.0` a mismatch of any kind is a real regression.

### A.6 What this plan does NOT include

These are from [`20260812_parse_time_reduction.md`](./20260812_parse_time_reduction.md). **The
measurement re-prioritised them**: the parse is now 99% `docling`, so anything
that speeds up `convert()` is the only thing left that can move `total`, and
anything VLM-side is now off the critical path entirely.

- **Step 1 (MPS benchmark)** — run the probe script natively on macOS; if MPS ≥ 2× faster, proceed to Step 2 (native worker). Unmeasured. **Now the highest-value remaining item.**
- **Step 2 (native worker)** — architectural change, only if Step 1 shows real MPS win.
- ~~**Step 3 (relax `--max-tasks-per-child`)**~~ — **ruled out.** Step 0 measured
  the out-of-band cost at 2.9s against a 60s threshold (A.3).
- ~~**Step 6 (httpx.Client reuse)**~~ — **now pointless.** Its ~20s ceiling was VLM-side, and `vlm_blocked=0` means VLM time no longer reaches `total` at all. It would only reduce contention with docling, second-order at best.

---

## Part B — Refresh button in Embed tab

### B.1 Problem

The Embed tab's Domain dropdown (`#upload-domain`) has no refresh button. After creating a new domain from the Chat tab or via API, the user must reload the page to see it in the Embed form.

The Chat tab already has a `↻` button at `home.html:341` that calls `loadDomainList()`, which repopulates **both** dropdowns. The JS function already exists and handles the Embed dropdown.

### B.2 Approach

Mirror the existing Chat tab pattern. One HTML addition, zero JS/backend changes.

### B.3 Implementation

In `src/app/api/templates/home.html`, add a refresh button next to the `#upload-domain` `<select>` at line 387, inside the same `<label>`:

```html
<div class="form-group">
    <label>Domain
        <button type="button" onclick="loadDomainList()" title="Refresh domain list"
                style="margin-left:6px;padding:4px 8px;font-size:13px;cursor:pointer;float:right;">↻</button>
    </label>
    <select id="upload-domain" class="model-select" onchange="toggleNewDomain(this)">
        <option value="__new__">➕ New domain…</option>
    </select>
    ...
</div>
```

The `float:right` keeps the button aligned with the label text, matching the Chat tab's inline style.

### B.4 Files

| File | Change |
|---|---|
| `src/app/api/templates/home.html` | Add `↻` button next to Domain label in the Embed tab (~3 lines) |

No API, JS, or backend changes. `loadDomainList()` at `:746-775` already repopulates `#upload-domain`.

### B.5 Verification

1. Open the Embed tab — confirm the `↻` button appears next to "Domain"
2. Create a new domain via the Chat tab or `POST /domains`
3. Click `↻` in the Embed tab — confirm the new domain appears in the dropdown
4. Confirm the "➕ New domain…" option is still present at the bottom

---

## Execution order

1. **Part B** first — trivial, 2 minutes, no risk.
2. **Part A Step 0** — one baseline run, no code changes.
3. **Part A Steps 1-4** — pipelining implementation + tests.
4. **Part A Step 5** — integration verification on the same NLTK.pdf.
