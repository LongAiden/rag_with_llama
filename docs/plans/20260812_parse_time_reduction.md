# Plan — reducing the ~17 min parse of a 504-page PDF

**Date**: 2026-08-12
**Status**: plan. Nothing here is applied.
**Follows**: F21 (`docs/plans/20260811_tableformer_outlier_and_prompt_v2.md`), F22, F23
(`docs/20260812_structure_preservation.md`).

---

## Part 1 — Diagnosis: where the 17 minutes goes

### The number

`docs/plans/20260811_tableformer_outlier_and_prompt_v2.md:156`, 504-page NLTK book, Apple
M1 / 16 GB / local Ollama, through `celery_worker_upload`:

```
parse_pdf summary: pages=504 total=1045s docling=782s (75%) assembly=263s (25%)
                   vlm_wait=261s vlm_calls=79 vlm_failures=0 peak_rss=2000MB
```

**1045s = 17.4 min.** F22 (144 DPI page renders) puts it at **1083s** with peak RSS
**2965 MB** (`docs/20260812_structure_preservation.md:92`). F23 is string work only.

This is the *result* of five rounds of optimization, not a regression:

| run | total | docling | vlm_wait |
|---|---|---|---|
| F13 baseline (WSL2, remote Ollama) | ~60 min | — | 3593s by page 463 |
| F18 | 2194s | 1600s | 592s |
| F19 (`7631584`) | 1683s | 1333s | 348s |
| **F21 (`e084578`)** | **1045s** | **782s** | **261s** |
| F22 (144 DPI) | 1083s | — | — |

### Breakdown

| phase | time | share |
|---|---|---|
| docling `DocumentConverter.convert` | ~782s | **75%** |
| VLM wait (79 Ollama calls) | 261s | 24% |
| non-VLM assembly (all string work, 504 pages) | ~2s | 0.2% |

`assembly=263s` minus `vlm_wait=261s` = 2s. **Assembly is entirely VLM wait.** Markdown
normalization, caption binding, code fencing and HTML stripping across 504 pages together
cost about two seconds. There is nothing to optimize on that side.

**Ollama is not the bottleneck.** It is 24% of the time and within ~15% of its measured
floor on this hardware.

### Why docling is 782s

1. **The M1 GPU is idle for all 782s.**
   [gemini_docling_parser.py:510-512](src/app/ingestion/processors/gemini_docling_parser.py#L510-L512)
   sets `AcceleratorOptions(device=AcceleratorDevice.AUTO)`. Inside the Linux container AUTO
   resolves to `cpu` — Docker Desktop on Apple Silicon has no GPU/MPS passthrough. The layout
   model (`docling-layout-heron`, 171 MB) and TableFormer (212 MB) are torch models running
   pure CPU. This is the single largest structural fact behind the number.

2. **Four threads inside a 4.0-CPU quota.** `DOCLING_NUM_THREADS=4`
   ([docker-compose.yml:291](docker-compose.yml#L291)) against `cpus: "4.0"`
   ([:237](docker-compose.yml#L237)). The Docker VM now reports **NCPU=6** (verified
   2026-08-12 — F17's slider warning is resolved), on a host with 8 logical cores
   (4 performance + 4 efficiency). Both the thread count and the quota are below the VM.

3. **Average s/page hides a non-uniform profile.** Healthy batches run 1.3–1.6 s/page. The
   index (pages 451–500) is dense multi-column tables the layout model hands to TableFormer;
   that batch was 11.10 s/page — 42% of all docling time in one batch of fifty — until
   `DOCLING_TABLEFORMER_MODE=fast` took it to 2.42 s/page. Still ~1.6× the healthy rate.

4. **All 504 pages are rendered at 144 DPI; 79 of them need it.** `generate_page_images=True`
   with `images_scale=2.0` ([:513-515](src/app/ingestion/processors/gemini_docling_parser.py#L513-L515))
   renders every page to a full-page PIL image (~5.8 MB per Letter page, ~290 MB per 50-page
   batch). `_expand_and_crop` ([:624-647](src/app/ingestion/processors/gemini_docling_parser.py#L624-L647))
   reads `doc.pages[n].image.pil_image` — but only for pages carrying a crop. Measured
   F21→F22: **+38s and +965 MB peak RSS**. The 2.0 scale is deliberate (43 DPI made the VLM
   confabulate); the *render* is unconditional while its consumer is not. This is primarily a
   memory lever, not a time lever — 38s is 3.5% of the total.

5. **Nothing runs in parallel above the thread level.** Ten sequential `convert()` calls at
   [:890-895](src/app/ingestion/processors/gemini_docling_parser.py#L890-L895), each batch
   fully assembled and released before the next converts — deliberate, it is what keeps peak
   RSS O(batch); dropping it is what OOM-killed the worker. Celery adds none: sequential
   `.si()` chain on one queue, `-c 1`, prefork. API uploads pin the chain to the `upload`
   queue ([document_routes.py:110-111](src/app/api/routes/document_routes.py#L110-L111)), so
   `celery_worker_ingestion` sits idle for the full 17 minutes.

### Why the 261s of Ollama is near its floor

- **79 calls, not 500 and not per-chunk.** No summarizer, title extractor, QA extractor or
  contextual-retrieval step exists in ingestion — the chain is `parse → chunk → embed`
  ([ingestion_tasks.py:113-118](src/app/worker/ingestion_tasks.py#L113-L118)), chunking is
  regex + chonkie, embedding is local `all-MiniLM-L6-v2`. Ollama is called once per figure
  crop clearing the size gate. Tables go to TableFormer (`VLM_TABLES=false`). The entity/graph
  path that *would* be per-chunk is unwired, asserted by `tests/unit/test_graph_not_wired.py`.
- **3.3 s/call against a measured 3.87 s/call floor** at `VLM_CONCURRENCY=1`. Concurrency is 1
  because a local Ollama serializes on one GPU: 3.87s at 1, 4.93s at 2, 20.62s at 4 (F15).
- **Raising concurrency would not help anyway.** `_process_page` blocks on its own futures
  before returning ([:828-843](src/app/ingestion/processors/gemini_docling_parser.py#L828-L843)) —
  no pipelining across pages regardless of pool size.
- Thinking off, `temperature 0.0`, `num_predict 384`, mean output 109 tokens at ~35 tok/s.
  Latency is pure decode; prefill is ~206 tokens in 0.26s regardless of crop size.
- Call count already went 191 → 79 via the short-side gate; going further discards real content.

**The two phases never overlap.** During the 782s of docling, Ollama is idle. During the 261s
of VLM wait, the container's threads are idle.

### Costs outside the summary line, still unmeasured

`parse_pdf summary: total=` measures `parse_pdf` only. Two costs sit outside it:

- **`--max-tasks-per-child=1`** on both workers ([:256](docker-compose.yml#L256),
  [:381](docker-compose.yml#L381)) forks a fresh child per stage — torch + docling re-imported
  for parse, `SentenceTransformer` re-loaded for embed, once per document. Also defeats the
  per-process pipeline cache at
  [ingestion_tasks.py:84-101](src/app/worker/ingestion_tasks.py#L84-L101). Flagged F8 on
  2026-08-04, estimated "tens of seconds," never measured.
- **Converter construction** at [:876](src/app/ingestion/processors/gemini_docling_parser.py#L876)
  is untimed. The HF weight path once cost 12+ minutes before page 1; now mitigated by the
  `hf_cache` volume and `HF_HUB_DISABLE_XET=1`, but nothing records today's cost.

Both are isolated by one subtraction never recorded: `Stage parse completed for <id> in Ns`
([ingestion_tasks.py:156](src/app/worker/ingestion_tasks.py#L156)) minus `total=Xs`.

### Headroom, sized

| lever | size | confidence |
|---|---|---|
| GPU/MPS for docling | largest — potentially most of the 782s | unmeasured on this host |
| overlap VLM wait with the next batch's convert | up to 261s (24%) | arithmetic, not measured |
| thread/CPU quota 4 → 6 | modest | untested; bounded by 4 performance cores |
| out-of-band per-stage re-import | unknown, possibly tens of seconds | never measured |
| unconditional 144 DPI render of 425 figure-less pages | ~38s and ~965 MB RSS | measured F21→F22 |
| index-region TableFormer cost | 2.42 vs 1.47 s/page | needs per-region policy — design question |
| the VLM itself | ~none | 3.3 vs 3.87 s/call floor |

---

## Part 2 — What to do, in order

Each step is gated on the previous one's measurement. Do not batch them — every prior finding
in this repo that held up was a single-variable change.

### Step 0 — Capture a baseline you can actually diff against (no changes)

`logs/` is empty and bind-mounted; every number in every performance doc came from ephemeral
`docker compose logs`. There is no persisted timing history. Before touching anything:

```bash
docker compose up -d celery_worker_upload
docker compose logs -f celery_worker_upload | tee logs/parse_baseline_$(date +%Y%m%d).log
# ...upload the 504-page PDF through the UI...
grep -E "parse_pdf summary|Converted pages|Stage .* completed" logs/parse_baseline_*.log
```

**What you are looking for:** `Stage parse completed for <id> in Ns` minus `total=Xs` from the
summary line. That difference is the entire out-of-band cost — torch re-import plus converter
construction — and it has never been recorded. If it is 20s, ignore Step 3. If it is 120s,
Step 3 jumps the queue.

Cost: one run you were going to do anyway. Changes: none.

### Step 1 — Size the GPU prize before committing to it (30 min, no repo changes)

The largest lever requires moving the parse out of the Docker VM, which is a real
architectural change. Measure the payoff *first*, standalone, on the host:

```bash
# native macOS env, not the container
uv run python - <<'PY'
import time, torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

print("mps available:", torch.backends.mps.is_available())
for dev in (AcceleratorDevice.MPS, AcceleratorDevice.CPU):
    o = PdfPipelineOptions()
    o.do_ocr = False
    o.do_table_structure = True
    o.table_structure_options = TableStructureOptions(do_cell_matching=True, mode="fast")
    o.accelerator_options = AcceleratorOptions(num_threads=4, device=dev)
    o.generate_page_images = True
    o.generate_picture_images = True
    o.images_scale = 2.0
    c = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=o)})
    t = time.monotonic()
    c.convert("data/input/raw/<the-504-page>.pdf", page_range=(1, 50))
    print(dev, f"{time.monotonic()-t:.1f}s for 50 pages")
PY
```

Use the same 50-page range for both, and pick a *prose* range (not 451–500) so the index
outlier does not dominate. Compare against the container's `Converted pages` line for the same
range.

**Decision rule:**
- MPS ≥ 2× faster than CPU → Step 2 is worth the architectural cost. Proceed.
- MPS < 1.5× faster, or `mps.is_available()` is False for these models → **stop**; docling's
  torch ops may not have MPS kernels, and the 782s is simply what this document costs on this
  machine. Skip to Step 4.

Note `docling_core` was not installed in the local env as of F16 — `uv sync` first.

### Step 2 — (conditional on Step 1) Run the upload worker natively on the host

Only if Step 1 showed a real MPS win. Keep postgres/redis/app in Docker; run
`celery_worker_upload` as a native process so `AcceleratorDevice.AUTO` resolves to MPS:

- The worker already reads broker/DB from env, so this is env plumbing, not code:
  `CELERY_BROKER_URL=redis://localhost:6379/...` and the postgres host port.
- Ports for redis and postgres must be published to the host (check they are).
- `data/` paths must resolve identically — the worker writes `data/parsed/<id>.md`.
- Set `AcceleratorDevice` explicitly rather than trusting AUTO, so the choice is visible in
  logs and a fallback to CPU is loud rather than silent.
- Frees the 3.5G the upload worker holds in a 6.77 GiB VM as a side effect.

This is the step that costs real design work. Do not start it before Step 1's number exists.

### Step 3 — (conditional on Step 0) Relax `--max-tasks-per-child=1`

Only if Step 0 showed the out-of-band cost is material. On `celery_worker_upload` only:
`--max-tasks-per-child=1` → `4`, or drop it. That flag exists to bound RSS growth across
documents, so watch `peak_rss` in the summary line across four consecutive documents before
keeping the change. Revert on any upward trend — an OOM-killed worker costs more than 60s.

### Step 4 — Cheap config experiment: threads 4 → 6 — **APPLIED + MEASURED 2026-08-12: keep**

The VM has 6 CPUs; the workers were capped at 4.0 and docling was told to use 4 threads.
Both raised to 6 on `celery_worker_upload` and `celery_worker_ingestion` (same image, same
pipeline), in the same pass that deduplicated the compose env blocks into YAML anchors:

- `celery_worker_upload` / `celery_worker_ingestion` `limits.cpus` `"4.0"` → `"6.0"`
- `x-common-env` `DOCLING_NUM_THREADS` `4` → `6` (shared by app, both workers, beat)

Expect a *sub*-linear gain: threads 5 and 6 land on efficiency cores, which are materially
slower, and Ollama shares the host. Measure `Converted pages … rate=Xs/page` on the same
document and page range — F12's lesson is that document complexity alone produces a 4×
difference, so cross-document comparisons are meaningless.

**Measured** on the same 504-page NLTK.pdf, F23 code:

```
parse_pdf summary: pages=504 total=1035s docling=749s (72%) assembly=286s (28%)
                   vlm_wait=281s vlm_calls=86 vlm_failures=0 peak_rss=2886MB
```

| metric | F21 | F22 (144 DPI) | this run |
|---|---|---|---|
| total | 1045s | 1083s | **1035s** |
| docling | 782s | ~820s (est: F21 + the +38s 144 DPI delta) | **749s** |
| vlm_wait / calls | 261s / 79 | — | 281s / 86 |
| peak RSS | 2000 MB | 2965 MB | 2886 MB |

**Verdict: keep the 6.** Docling fell ~820s → 749s, about **9%** — a real gain but
sub-linear, exactly as predicted: threads 5 and 6 land on efficiency cores. Peak RSS did not
rise (2886 vs F22's 2965), so the headroom warning did not materialise. Per-VLM-call cost is
unchanged at 3.27s; `vlm_wait` rose only because F23's figure wrapping produced 7 more calls.

Total is still ~17 minutes, and the shape is unchanged — docling 72%, VLM 27%, no overlap.
Step 5 is now the live work, promoted to `docs/plans/20260812_parse_pipelining.md`.

### Step 5 — Overlap VLM wait with the next batch's convert (code) — **promoted to its own plan**

> Designed in full at `docs/plans/20260812_parse_pipelining.md` (2026-08-12), against the
> measured 281s / 86 calls rather than the 261s / 79 sketched below. Read that instead.

The ceiling is the full 261s (24%) and it is the largest win available *without* leaving
Docker. Currently: convert batch N → assemble batch N (blocking on 79 VLM futures across the
run) → convert batch N+1. The two resources alternate; neither is ever saturated.

The change: submit batch N's VLM tasks, start batch N+1's `convert()`, and resolve batch N's
futures afterwards — a pipeline of depth 1.

**Why this is tractable:** `_expand_and_crop` materializes an independent PIL crop *before*
the future is submitted, so futures hold no reference to `doc`. `del doc` after each batch
stays valid, and the O(batch) RSS property is preserved. What must be held across one extra
convert is one batch's ordered page items plus its crops — small next to the 290 MB of page
images already resident.

**Risks to design for:**
- Output ordering. `parse_pdf` writes pages streaming to `out_file`; a depth-1 pipeline means
  batch N is written *during* batch N+1's convert. The write must stay strictly ordered.
- Peak RSS. Two batches partly alive at once. Measure `peak_rss` on the summary line; this is
  the failure mode that OOM-killed the worker before.
- `_process_page` currently joins its own futures ([:828-843](src/app/ingestion/processors/gemini_docling_parser.py#L828-L843)).
  Deferring that join is the core of the change and touches the F23 wrapping logic
  (`kind`/`caption` tuple) — `tests/unit/test_f23_structure_preservation.py` and
  `test_pdf_parser_streaming.py` must both stay green.

Gains nothing if Step 2 lands and docling drops to ~250s — at that point VLM wait becomes the
majority and this becomes the *first* priority instead of the fifth. Re-rank after Step 2.

### Step 6 — Leave the VLM alone

`VLM_CONCURRENCY`, `OLLAMA_VLM_NUM_PREDICT`, `VLM_MIN_IMAGE_SHORT_PT`, `OLLAMA_VLM_THINK` are
all at measured optima. 3.3 s/call against a 3.87 s/call floor. The only remaining VLM item is
reusing an `httpx.Client` across calls — connect + prefill is 0.26s of a 3.3s call, so the
ceiling is ~20s across the run. Not worth doing before Steps 1–5.

Do **not** raise concurrency while Ollama is local. Do **not** set `VLM_TABLES=true` without a
substantially larger vision model (F16: the 0.8B model hallucinated table content).

### Not recommended

- **Gating `generate_page_images` per page.** You cannot know which pages carry figures until
  after `convert()`, so gating means either a two-pass convert (worse) or a separate page-image
  render outside docling (a new dependency and a new coordinate-mapping bug surface). The prize
  is 38s of 1083. Revisit only as a *memory* fix if RSS becomes binding again.
- **Lowering `images_scale` below 2.0.** F22 chose it deliberately; 43 DPI made the VLM
  confabulate. 1.5 (108 DPI) is an untested middle ground — a quality experiment, not a
  performance one.
- **Per-region TableFormer policy** for the index. A real 0.95 s/page × 50 pages, but it needs
  a page-classification heuristic that will misfire on other documents.

---

## Unrelated, noted while reading

`e084578` removed the `profiles:` key from `langfuse`, so it now starts on a plain
`docker compose up -d` — `docker compose config --services` lists 7. F17 had explicitly gated
it behind the `observability` profile because it holds a share of a 6.77 GiB VM that the
performance docs argue is the binding constraint. Not a parse cost (the parse is CPU-bound,
not memory-bound), but it silently reverses a deliberate earlier fix. Restoring
`profiles: [observability]` is a one-line change, independent of everything above.

## Verification commands

```bash
docker compose logs -f celery_worker_upload | tee logs/parse_$(date +%Y%m%d_%H%M).log
grep "parse_pdf summary"   logs/parse_*.log   # total / docling% / assembly% / vlm_wait / peak_rss
grep "Converted pages"     logs/parse_*.log   # per-batch s/page — confirms the index outlier
grep "Stage .* completed"  logs/parse_*.log   # subtract total= to size the out-of-band cost
grep -c "VLM call #"       logs/parse_*.log   # expect ~79, not ~500
docker info --format '{{.NCPU}}'              # 6 as of 2026-08-12
```

Always compare the **same document over the same page range**. Every misleading number in this
repo's history came from comparing across documents.
