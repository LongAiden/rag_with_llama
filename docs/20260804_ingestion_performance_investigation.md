# Ingestion Performance Investigation — Why Docker/Celery Parsing Is Slower

**Date**: 2026-08-04
**Status**: diagnosis complete; instrumentation and the unconditional fixes landed 2026-08-05. No
baseline measured yet, no tuning fix applied.

> ## Correction — 2026-08-05
>
> **Everything this document described as "✅ DONE" or "Fix applied" was written on the WSL2 box and
> never reached this repository.** Only the two `.md` files crossed over; the working tree was
> otherwise clean. Verified before any work started: no `_rss_mb`, no `elapsed=`, no `_vlm_seconds`,
> no stats lock, zero matches for `TORCHDYNAMO_DISABLE` in `docker-compose.yml`, zero matches for
> `keep_alive` in `src/`, `num_threads=2` still hardcoded, upload worker still at `cpus: "2.0"`, and
> neither the `_run_stage` timing nor the `logfire.span` existed. So the repo was at the
> pre-instrumentation baseline and the F11 crash and F13 model-reload cost were both still live here.
>
> **Landed 2026-08-05** — Step 1 instrumentation (now in the base class, so it covers *both*
> backends rather than only the Ollama path), Fix C, Fix E, Fix F, the F11 `TORCHDYNAMO_DISABLE`
> fix, the F13 `keep_alive` fix, and the F10 error-detail logging.
>
> **Also landed, not in the original plan**: `parse_pdf` was hoisted into `GeminiDoclingParser` and
> `OllamaPDFParser.parse_pdf` deleted. The two copies were ~95% identical and both needed the same
> Fix C restructure — duplicating the fix is what let them drift in the first place. Dead code went
> in the same pass: `_parse_page_by_page` (never called) and the first `_fix_markdown_headings`
> (shadowed by a second definition of the same name at `:208`).
>
> **Deliberately not landed**: Fix A, Fix B, Fix D and Fix G, which are all tuning. Instead
> `DOCLING_NUM_THREADS`, `DOCLING_PAGE_BATCH_SIZE` and `VLM_CONCURRENCY` are now settings whose
> defaults reproduce the previously hardcoded values exactly. Behaviour is unchanged, the baseline
> below stays uncontaminated, and each deferred fix becomes a one-variable `.env` experiment rather
> than a code edit. F12 is why: two simultaneous changes measured on two different documents
> produced a 4× improvement that cannot be attributed to either.
>
> F12's own numbers are **not** in this repo — `num_threads` is back to its documented default of 2
> and the upload worker to `cpus: "2.0"`. The controlled comparison F12 asks for (Algorithms By Jeff
> Erickson.pdf, pages 1-50, at `DOCLING_NUM_THREADS=4` against the 5.72s/page on record) is now an
> env-var change away.

## Context

A ~500-page PDF takes ~1 hour to get through the parse stage (docling + Ollama VLM) when run
through the Celery workers in Docker, and is noticeably faster when the same parsing code is run
directly with all cores available. Ollama runs on a **separate machine**, so VLM calls are pure
network I/O for the worker and do not compete for local CPU.

**Host / VM sizing (confirmed from WSL2 settings, 2026-08-04):**

| | |
|---|---|
| WSL2 VM logical processors | **8** |
| WSL2 VM memory | **8101 MB** |
| WSL2 VM swap | 2048 MB |
| Celery worker containers | 2 (`celery_worker_upload`, `celery_worker_ingestion`), `-c 1` each |

Docker Desktop runs its containers inside that WSL2 VM, so "8 logical processors" is the CPU budget
every container shares — not the 4 physical cores of the host. This is the number the `cpus:` limits
in `docker-compose.yml` are carved out of.

The question was whether Celery is the bottleneck. It is not — but the container configuration the
workers run in is. Below is the diagnosis, followed by a measure-then-fix plan (the chosen
approach), because **nothing in the parse path currently records a duration**, so any fix applied
today would be unverifiable.

---

## Findings

### Answer: Celery is not the cause. The CPU quota and a hardcoded thread count are.

**F1 — The worker is capped at 2 of the VM's 8 logical processors, i.e. 25%.**
[docker-compose.yml:218-225](docker-compose.yml#L218-L225) sets `cpus: "2.0"` on
`celery_worker_ingestion`, and [:162-169](docker-compose.yml#L162-L169) sets the same on
`celery_worker_upload`. Compose V2 applies `deploy.resources.limits` (confirmed by the fact that the
memory limits in the same block produced real OOM kills, per
[docs/ARCHITECTURE.md:487-495](docs/ARCHITECTURE.md#L487-L495)). Running the parser directly gets
all 8. The docling layout + TableFormer pass is CPU-bound and scales close to linearly with threads,
so this is the primary answer to the question — and it is a wider gap than a "4 core" reading
suggests. Hyperthreading means the real speedup from lifting the cap is bounded by the 4 physical
cores, so expect something in the 2-2.5× range rather than a full 4×.

**F1b — Both workers are capped identically, and API uploads run on the *upload* worker.**
[document_routes.py:89](src/app/api/routes/document_routes.py#L89) builds the chain with
`queue=UPLOAD_QUEUE`, so a PDF uploaded through the web UI is parsed by `celery_worker_upload`, not
`celery_worker_ingestion` — only the weekly scan and the 6-hourly recovery use the ingestion queue
([ingestion_tasks.py:311-314](src/app/worker/ingestion_tasks.py#L311-L314)). The finding is
unaffected because both containers carry the same `cpus: "2.0"`, but any measurement or fix must
target the container that actually ran the job.

**F2 — Even with more CPU, docling would not use it.**
[gemini_docling_parser.py:305-307](src/app/ingestion/processors/gemini_docling_parser.py#L305-L307)
hardcodes `AcceleratorOptions(num_threads=2)`. `OllamaPDFParser` inherits `_build_converter()`
unchanged. There is no env var for it. Raising the CPU quota without also raising this changes
nothing.

**F3 — The docling pass is the largest fixed cost and it is invisible.**
[ollama_pdf_parser.py:135-141](src/app/ingestion/processors/ollama_pdf_parser.py#L135-L141) runs
10 sequential `converter.convert()` calls (`_DOCLING_PAGE_BATCH_SIZE = 50`, 500 pages). OCR is off
but `do_table_structure=True` with `do_cell_matching=True` runs TableFormer on *every* detected
table, and docling's default `TableFormerMode` is the slow ACCURATE variant. The only log lines are
`"Converting pages 1-50..."` with no elapsed time.

**F4 — Peak memory grows with total page count, not batch size.**
`batch_docs` ([ollama_pdf_parser.py:133, 141](src/app/ingestion/processors/ollama_pdf_parser.py#L133))
retains all 10 `DoclingDocument`s — each holding rendered page images
(`generate_page_images=True`, `generate_picture_images=True`) — for the entire run. The batching
bounds docling's *working* memory but not its *retained* memory. The VM has **8101 MB** and the
compose limits already sum to roughly **7.5 GB** (postgres 1G + app 1G + upload worker 2.5G +
ingestion worker 2.5G + redis 256M + beat 256M), before the WSL2 kernel and the Docker daemon take
their share — and `langfuse` adds another 1G when it starts, which it does by default (it is *not*
profile-gated, unlike `pgadmin` and `test`). With only 2048 MB of swap behind that, this pushes the
VM toward swap, which
would show up as a large, non-linear slowdown. This is the most likely second-order cause of
"much slower in Docker".

**F5 — VLM calls are effectively serial, page by page.**
`parse_pdf` creates one `ThreadPoolExecutor(max_workers=_VLM_CONCURRENCY=2)`
([ollama_pdf_parser.py:162](src/app/ingestion/processors/ollama_pdf_parser.py#L162)), but
`_process_page` blocks on all of its own futures before returning
([gemini_docling_parser.py:549-558](src/app/ingestion/processors/gemini_docling_parser.py#L549-L558)).
So there is no pipelining across pages, and a page with a single figure runs fully serial. Since
Ollama is on a separate machine, these are pure network waits — the worker sits idle during them
and the concurrency limit of 2 is far below what a remote server could absorb.

**F6 — The claim timeout is shorter than the parse.**
`INGESTION_CLAIM_TIMEOUT_MINUTES` defaults to 30
([app_config.py:101](src/app/config/app_config.py#L101)) but the parse takes ~60. When the 6-hourly
`recover_and_dispatch` beat tick lands mid-parse,
[`reset_stale_claims`](src/app/infra/db/ingestion_repository.py#L271-L306) increments `attempts`,
resets `stage` to `registered`, and re-dispatches — while the original parse is still running. Worst
case two docling conversions of the same document run concurrently on the same 2-core quota, which
would roughly halve throughput and corrupt the retry budget.

**F7 — The parse blocks the worker's event loop for the full hour.**
[pipeline.py:138](src/app/ingestion/embedding/pipeline.py#L138) calls the synchronous
`parser.parse_pdf(...)` directly inside an `async def`, with no `asyncio.to_thread`. With `-c 1`
nothing else runs in that worker, so this does not slow the parse itself — but it does mean the
asyncpg pool cannot service anything, so no claim heartbeat or progress update is possible during
the run.

**F8 — `--max-tasks-per-child=1` re-imports torch/docling per stage.**
[docker-compose.yml:226](docker-compose.yml#L226). Combined with `-c 1`, every stage forks a fresh
child, so docling + torch (and `SentenceTransformer` for embed) are loaded from scratch three times
per document. Real but small — likely tens of seconds, not the hour. Worth measuring before
touching, since it exists to bound memory.

**F9 — No thread caps anywhere.** `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`torch.set_num_threads` appear nowhere in the repo. Torch and OpenBLAS size their pools from the
*host* core count (4), not the cgroup quota (2), so the embed stage and any numpy/PIL work
oversubscribe 2× inside the container. Docling's own predictors are capped by `num_threads=2`, so
the parse pass is mostly protected; the embed stage is not.

**F10 — Worth verifying while measuring: is `qwen3.5:0.8b` a vision model?**
It is the default `OLLAMA_VLM_MODEL` ([docker-compose.yml:236](docker-compose.yml#L236)). If it does
not accept images, every call raises and
[ollama_pdf_parser.py:117-119](src/app/ingestion/processors/ollama_pdf_parser.py#L117-L119)
silently swallows it and returns the literal string `"[IMAGE]"`. That would be fast but would mean
every figure and complex table is being dropped from the parsed output.

**F11 — `InvalidCxxCompiler` crash on Windows/WSL2, discovered 2026-08-04.**
Mid-parse, the worker crashed with:
```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
InvalidCxxCompiler: No working C++ compiler found in torch._inductor.config.cpp.cxx: (None, 'g++')
```
Nothing in this repo calls `torch.compile` (`grep -rn "torch.compile\|dynamo" src/` is empty) — this
is a downstream library, most likely torchvision's NMS op falling back to a dynamo-compiled path
during docling's layout/table postprocessing. It fails because the **runtime image deliberately ships
without a compiler**:
[Dockerfile:51-57](deploy/deployment/Dockerfile#L51-L57) is commented `# Runtime dependencies only
(no compilers)` and installs only `curl`, `libpq-dev`, `libgl1`, `libglib2.0-0`. `g++`/`gcc` exist
only in `Dockerfile.base`, which builds the venv but is not part of the runtime image — `COPY
--from=builder /opt/venv` brings the installed Python packages, not the apt packages used to build
them.

**Fix applied**: `TORCHDYNAMO_DISABLE: "1"` added to all four app-image services in
`docker-compose.yml` (`app` [:117](docker-compose.yml#L117), `celery_worker_upload`
[:193](docker-compose.yml#L193), `celery_worker_ingestion` [:254](docker-compose.yml#L254),
`celery_beat` [:310](docker-compose.yml#L310)). This is the documented PyTorch env var for fully
bypassing Dynamo: any `torch.compile`-wrapped call becomes a pass-through to normal eager execution.
No rebuild needed — it's an env var, but compose won't pick it up on a plain `restart`; use
`docker compose up -d --force-recreate <service>`.

**Why disabling it is safe, not just a workaround**: `torch.compile` is an opt-in speed optimization
layered on top of PyTorch's eager mode. Eager execution is the default, well-tested code path used
almost everywhere; nothing about correctness depends on the compiled path succeeding. Disabling Dynamo
does not change what the layout/table model computes, only whether that one op gets a JIT-compiled
kernel — so the fix trades a small, hard-to-quantify inference speedup for eliminating a hard crash.

**Why the Mac run didn't hit this even before investigating further**: the Mac test was run *after*
this `docker-compose.yml` fix was already in the repo, so `TORCHDYNAMO_DISABLE=1` was present from the
start and Dynamo was never invoked — no compile attempt, so whether the Mac's runtime container has a
compiler was never tested. It's tempting to assume Apple Silicon dev machines are naturally immune
because macOS ships Xcode Command Line Tools (`clang`) by default, but that's irrelevant here: the
container is still built from the same `Dockerfile`, which strips compilers from the runtime stage
regardless of host OS or architecture (Docker Desktop on Mac runs a native arm64 Linux VM with the
same Debian-slim base). Absent the fix, an arm64 build of this image would very likely hit the same
`InvalidCxxCompiler` error — the crash is a property of the image, not the host. This has not been
verified by actually reverting the fix and re-testing; it follows from the Dockerfile being identical
across platforms, not from a direct A/B test.

**F12 — Manual thread/CPU tuning applied 2026-08-04; speedup not yet isolated from a confound.**
Two changes were made directly by hand while investigating the 5.72s/page docling rate observed in
F2/F3 (measured on *Algorithms By Jeff Erickson.pdf*, pages 1-50, at the original `num_threads=2`):

- `celery_worker_upload` CPU limit raised `2.0` → `6.0`
  ([docker-compose.yml:170](docker-compose.yml#L170)).
- Docling's own thread count raised `num_threads=2` → `4`
  ([gemini_docling_parser.py:305-307](src/app/ingestion/processors/gemini_docling_parser.py#L305-L307),
  inherited unchanged by `OllamaPDFParser`).

`4` rather than `6` was chosen deliberately: the WSL2 VM reports **8 logical processors** against a
Docker Desktop config originally described as **4 CPU cores** — classic hyperthreading, 4 physical
cores presenting as 8 logical ones. Docling's layout/TableFormer inference is compute- and
memory-bandwidth-bound, the kind of workload that gains little from hyperthread siblings sharing a
physical core's execution units and cache. `num_threads=4` targets the physical core count rather
than the logical one, and deliberately leaves ~2 logical cores of slack inside the container's
`cpus: "6.0"` quota for the orchestrating Python thread, the VLM executor, and Celery/asyncpg
bookkeeping — Docker enforces `--cpus` as a CFS bandwidth quota over a rolling window, so saturating
every logical core with docling's own threads risks throttling everything else in the container
alongside it, not just failing to help.

Only the CPU-limit change (`2.0` → `6.0`) is uninteresting on its own: raising the container's quota
without also raising docling's internal `num_threads` does nothing, since docling never asks for more
than the value baked into `AcceleratorOptions` regardless of what the container is allowed (this is
exactly F2 — the quota was never the constraint). The `num_threads` change is the one that could
plausibly move the needle.

**Evidence so far does not isolate the effect.** After both changes, a batch logged:
```
Converted pages 51-100: elapsed=71.8s rate=1.44s/page rss=1821MB
```
— a large drop from the 5.72s/page baseline. But this is **not a controlled comparison**: the
baseline was pages 1-50 of *Algorithms By Jeff Erickson.pdf* (a dense, table- and diagram-heavy
technical book), and the new number is pages 51-100 of a *different document*,
*software_design.pdf*. `do_table_structure=True` runs TableFormer on every detected table regardless
of complexity, so a table-heavy book pays a per-page cost a prose-heavy one doesn't, independent of
thread count entirely. The ~4× drop could be mostly threading, mostly document complexity, or some
mix — there is no way to tell from these two numbers.

**To actually attribute the change**: re-run *Algorithms By Jeff Erickson.pdf*, pages 1-50
specifically, at `num_threads=4`, and compare directly against the 5.72s/page figure already on
record for that exact document and page range. That is the only controlled comparison available
without capturing new baseline data on a different book first.

**F13 — VLM calls cost 27-33s each on a 0.8B model; VLM wait, not docling, dominates the hour.**
A production run logged VLM call #108 (a 328×314px, 47.1KB image, `qwen3.5:0.8b`) at
`elapsed=27.54s`, with cumulative `vlm_wait=3593s` (~60 min) by page 463/514. That is, on its own,
roughly the entire "~1 hour" this investigation started from — larger than the docling conversion
cost flagged in F3. 3593s / 108 calls ≈ 33s/call average, consistent with the single sample. A
sub-1B model captioning a tiny image should return in low single-digit seconds even on CPU, so
27-33s indicates something is wrong, not that VLM is inherently this expensive.

A fresh read of `_call_vlm`
([ollama_pdf_parser.py:100-161](src/app/ingestion/processors/ollama_pdf_parser.py#L100-L161))
found the request payload sent to Ollama's `/api/generate` was `{"model", "prompt", "images",
"stream": False}` only — **no `keep_alive` field**. Ollama's documented default when `keep_alive`
is absent is to unload the model 5 minutes after its last use. Combined with F5 (calls
effectively serialized per page, so gaps between VLM-triggering pages are easy to exceed) and a
second model (`deepseek-r1:1.5b`, `OLLAMA_MODEL` default in
[docker-compose.yml:96](docker-compose.yml#L96)) sharing the same Ollama instance and potentially
evicting `qwen3.5:0.8b` between calls, repeated cold-load is the strongest candidate explanation:
each 27-33s call would be paying a full model-load cost rather than inference time.

Two smaller contributors were checked and ruled out as primary causes:
- Each call is a module-level `httpx.post(...)` — a new client/TCP connection per call, no
  pooling. A fresh connection to `host.docker.internal` costs single-digit ms, not tens of
  seconds, so this is real but marginal.
- No retry logic exists anywhere in the VLM call path (grepped for `retry|Retry|tenacity` in both
  parser files, zero matches) — the 27-33s figures are single-attempt costs, not doubled by
  retries.

This also updates **F10**: its "fast failure" hypothesis (non-vision model → fast 400 → silent
`[IMAGE]` fallback) is ruled out by the observed latency. A 400 response would fail fast, not in
27-33s, so if `qwen3.5:0.8b` were rejecting images outright this finding's timing wouldn't look
like this — F10 remains open but the fast-failure framing no longer fits the evidence.

> **Superseded by F14** in `docs/20260805_vlm_thinking_and_table_routing.md`. The cold-load
> hypothesis below is not the main cause: `qwen3.5:0.8b` is a *reasoning* model and thinking
> was on, so each call generated thousands of discarded reasoning tokens. That is what the
> 27-33s was, and it fits the `eval_count` in the thousands that a cold load would not
> explain. `keep_alive` remains correct and stays.

**Fix applied (2026-08-04, not yet measured):** added `"keep_alive": "30m"` to the request payload
in `_call_vlm`, keeping the model resident well past any gap between VLM-triggering pages.
Deliberately the only change made — connection pooling (module-level `httpx.post` →  a reused
`httpx.Client`) and any change to `_VLM_CONCURRENCY`/per-page serialization (F5) are left as
follow-ups, not bundled in, so the effect of `keep_alive` alone can be measured in isolation. Same
"controlled re-test" discipline as F12: re-run and compare per-call `elapsed=` values (expect a
slow first call, then fast ones) and cumulative `vlm_wait` against the 3593s figure above before
concluding anything.

**F14-F17 — measured on the Mac dev host, 2026-08-05.** VLM reasoning tokens, the silent
empty-table responses, VLM concurrency against a local Ollama, table routing, and the
Docker Desktop CPU slider that had not been applied. Written up separately in
**`docs/20260805_vlm_thinking_and_table_routing.md`**, because they were measured on a
different machine under different assumptions than F1-F13 (Ollama local rather than
remote) and several of them revise conclusions here.

### Noted, not proposed for change

- `_RateLimiter` mutates a deque from two pool threads without a lock. Only affects the Gemini path
  and its 10 RPM pacing. **The `_vlm_calls` half is fixed** — the counters moved behind
  `_vlm_stats_lock` in `_record_vlm_call`, because the timing totals the instrumentation reports
  would otherwise silently under-report and defeat the measurement.
- ~~`_fix_markdown_headings` is defined twice~~ — **fixed 2026-08-05**, the shadowed first
  definition was deleted.
- ~~`_parse_page_by_page` is dead code~~ — **fixed 2026-08-05**, deleted. The streaming `parse_pdf`
  supersedes its purpose.
- `container_name: rag_celery_worker_ingestion` makes the `--scale celery_worker_ingestion=2`
  suggested in the nearby comment fail. Still open.

---

## Plan

### Step 1 — Instrument the parse path ✅ DONE

Goal: split the hour into `docling convert` / `VLM wait` / `page assembly` / `everything else`, with
per-batch and per-page granularity, plus RSS at batch boundaries to confirm or rule out F4.

All changes are additive logging — no behavioural change to parsing itself.

**Landed:**

1. **`src/app/ingestion/processors/ollama_pdf_parser.py`**
   - Module-level `_rss_mb()` reads `VmRSS` from `/proc/self/status`, returning `0.0` where
     unavailable. Dependency-free — no `psutil` added; the fallback only hits on a Windows dev box.
   - `_call_vlm` times the whole call and accumulates `self._vlm_seconds`. The per-call
     `logger.info` now carries `elapsed=Xs`, and the counter increment moved inside a lock.
   - The failure path logs elapsed time, the exception type, and — on `httpx.HTTPStatusError` —
     the HTTP status and first 300 bytes of the response body. This is what makes **F10**
     answerable: a non-vision model answers 400 with an explanatory message, and the `[IMAGE]`
     fallback was otherwise completely silent.
   - Batch loop logs `elapsed=Xs rate=Ys/page rss=ZMB` per 50-page batch.
   - Assembly loop logs per-page elapsed, plus a cumulative `vlm_wait` for any page over 1s.
   - New `parse_pdf summary:` line reports `pages / total / docling (%) / assembly (%) /
     vlm_wait / vlm_calls / vlm_failures / peak_rss`. A separate `WARNING` fires if any VLM call
     failed, naming the model.
   - Added `self._vlm_stats_lock`: `_call_vlm` runs on `_VLM_CONCURRENCY` pool threads, so the
     counters need a lock or the totals silently under-report — which would defeat the measurement.
     This closes the `_vlm_calls` half of the thread-safety note above, for the Ollama path only.

2. **`src/app/ingestion/embedding/pipeline.py`** — the `parse_pdf` call is wrapped in a
   `logfire.span("parse_pdf", document_id=..., filename=..., backend=..., file_size=...)`.
   `logfire` was already imported and used a few lines below; this is the first span in the
   ingestion path (spans previously existed only in `retrieval/`).

3. **`src/app/worker/ingestion_tasks.py`** — `_run_stage` times `await work(...)` and logs
   `Stage <name> completed for <doc_id> in <N>s`, and the elapsed time on the failure path too.
   Gives per-stage totals for parse / chunk / embed straight from the worker log, without needing
   the Celery result backend.

No tuning fix was applied in this step, by design — the point is a clean baseline.

> **Not yet verified by execution.** No Python or Docker was available on the machine where this
> was written, so the three files were syntax-reviewed by reading, not compiled or run. First
> action in Step 2 is to confirm the worker starts and the new log lines appear.

### Step 2 — Take the baseline measurement

```bash
docker compose up -d --build celery_worker_ingestion
# confirm the instrumentation loaded before committing to an hour-long run:
docker compose logs celery_worker_ingestion | grep -i "ready\|Traceback"

# upload / re-register the 500-page PDF, then:
docker compose logs -f celery_worker_ingestion | tee baseline.log
docker stats rag_celery_worker_ingestion   # in a second terminal
```

Pull the numbers out with:

```bash
grep "parse_pdf summary" baseline.log          # the one line that answers the question
grep "Converted pages"   baseline.log          # per-batch docling rate and RSS growth
grep "Stage .* completed" baseline.log         # parse / chunk / embed totals
grep -c "VLM call failed" baseline.log         # F10 — should be 0
```

Record in the table at Step 4: docling total, VLM wait total, VLM call count, peak RSS, and whether
any `VLM call failed` warnings appear.

Two things the baseline settles that the static reading could not:

- **F4** — whether `rss=` in the per-batch lines climbs across all 10 batches (retention scales with
  page count) or stays flat (batching already bounds it, and F4 is wrong).
- **F1/F2** — whether `docker stats` CPU% sits pinned just under 200%, which is the direct evidence
  that the 2.0 quota, not Celery, is the constraint.

### Step 3 — Apply fixes, gated on what Step 2 shows

| If the baseline shows | Apply |
|---|---|
| docling total dominates (expected) | **Fix A + B** below — the 2× CPU lever |
| peak RSS approaches 2.5 G, or `docker stats` shows sustained pressure | **Fix C** — bound retained memory |
| VLM wait total is a large share | **Fix D** — pipeline VLM across pages |
| any run at all | **Fix E + F** — unconditional, they are correctness issues |

**Fix A — give the worker the machine.** In [docker-compose.yml](docker-compose.yml), raise
`celery_worker_ingestion` `cpus` from `"2.0"` to `"4.0"`, and drop `postgres`
([:16](docker-compose.yml#L16)) and `app` ([:79](docker-compose.yml#L79)) to `"1.0"` so the limits
stop summing to 8.0 on a 4-core box.

**Fix B — make docling's thread count configurable and raise it.** Add
`docling_num_threads: int = Field(default=2, validation_alias='DOCLING_NUM_THREADS')` to
`AppSettings` in [app_config.py](src/app/config/app_config.py) (alongside the existing ingestion
settings around line 100). Thread it through `create_pdf_parser`
([pdf_parser_factory.py:21-26](src/app/ingestion/processors/pdf_parser_factory.py#L21-L26)) into
the parser constructors and into `_build_converter()`
([gemini_docling_parser.py:305-307](src/app/ingestion/processors/gemini_docling_parser.py#L305-L307)),
replacing the hardcoded `2`. Set `DOCLING_NUM_THREADS=4` in compose for the ingestion worker.
Note the factory currently passes only `ollama_base_url` and `vlm_model` — this is also the right
place to make `images_scale` and `min_image_px` configurable, but do that only if Step 2 justifies
it.

**Fix C — release each batch document after use.** Restructure `parse_pdf`
([ollama_pdf_parser.py:121-196](src/app/ingestion/processors/ollama_pdf_parser.py#L121-L196)) so
each 50-page batch is converted → its pages fully assembled (including VLM) → the `DoclingDocument`
dropped, instead of accumulating all batches in `batch_docs` first. `_process_page` needs the `doc`
only for `item.get_image(doc)`, so moving assembly inside the batch loop is sufficient. This turns
peak memory from O(total pages) into O(batch size). Mirror the same change in
`gemini_docling_parser.parse_pdf` ([:566-648](src/app/ingestion/processors/gemini_docling_parser.py#L566-L648))
so the two backends do not diverge. Also make `_DOCLING_PAGE_BATCH_SIZE` env-configurable in the
same pass.

**Fix D — pipeline VLM calls across pages.** Only if VLM wait is significant. The change is to stop
`_process_page` from joining its own futures
([gemini_docling_parser.py:549-558](src/app/ingestion/processors/gemini_docling_parser.py#L549-L558)):
have it return the ordered list with unresolved futures, and resolve them in `parse_pdf` after
several pages have been submitted. Raise `_VLM_CONCURRENCY` (make it env-configurable) — with
Ollama remote these are network waits, so 4-8 is reasonable. Also switch `_call_vlm` to a shared
`httpx.Client` instead of a fresh `httpx.post` per call
([ollama_pdf_parser.py:101](src/app/ingestion/processors/ollama_pdf_parser.py#L101)) to stop paying
TCP + handshake per image. **Caveat:** this interacts with Fix C — if assembly moves inside the
batch loop, pipelining is bounded by the batch. Apply C first, then D.

**Fix E — raise the claim timeout above the worst-case parse.** Set
`INGESTION_CLAIM_TIMEOUT_MINUTES=180` in compose for both workers (default is 30 at
[app_config.py:101](src/app/config/app_config.py#L101)). This is the F6 fix and is required
regardless of the tuning outcome — today a 60-minute parse can be declared stale and duplicated
mid-run.

**Fix F — offload the parse off the event loop.** Change
[pipeline.py:138](src/app/ingestion/embedding/pipeline.py#L138) to
`parsed_text = await asyncio.to_thread(parser.parse_pdf, str(file_path), None)`. Matches the pattern
already used for embedding at [pipeline.py:312-314](src/app/ingestion/embedding/pipeline.py#L312-L314)
and frees the asyncpg pool during the run.

**Fix G (embed stage, optional).** If Step 2 shows the embed stage is also slow, add
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` matching the CPU quota to the worker env in compose, so torch
stops sizing its pool from the host core count (F9).

### Step 4 — Land this document in the repo and update the architecture doc

Copy this file to `docs/20260804_ingestion_performance_investigation.md` and fill in two tables that
cannot be written yet:

| Phase | Baseline (Step 2) | After fixes (Step 3) |
|---|---|---|
| docling convert (10 × 50 pages) | | |
| VLM wait (total, remote Ollama) | | |
| page assembly (non-VLM) | | |
| parse stage wall clock | | |
| peak RSS | | |
| VLM calls / failures | | |

Then update `docs/ARCHITECTURE.md`:

- Add a §15.7 pointing at the new document. §15.4's "Expected Performance Improvements" table has
  **no wall-clock figure** for the 504-page document today — only "OOM kill at 2G → completes at
  ~1.15G". That is exactly the gap this closes.
- Correct §15.1, which states the page-batching made memory "flat regardless of document size".
  Finding F4 shows that is not true: `batch_docs` retains every batch document for the whole run, so
  peak memory still scales with total page count. Fix C is what actually makes it flat.
- Remove from §10.5 "Hardcoded values that should be configurable" whatever Fix B and Fix C made
  configurable (`num_threads`, `_DOCLING_PAGE_BATCH_SIZE`, and `_VLM_CONCURRENCY` if Fix D lands).
- §15.5 claims "No new environment variables required" — update it for `DOCLING_NUM_THREADS` and
  `INGESTION_CLAIM_TIMEOUT_MINUTES`, and add both to `.env.example`.

---

## Verification

1. **Instrumentation is correct**: re-run the 500-page PDF and confirm the summary line's
   `docling_total + assembly_total` accounts for ~the wall clock reported by `_run_stage`. A large
   unexplained remainder means the instrumentation missed a phase.
2. **Fix A + B took effect**: `docker stats rag_celery_worker_ingestion` should show CPU% climbing
   above 200% during the docling phase (it is pinned below 200% today). Expect the docling total to
   roughly halve.
3. **Fix C took effect**: peak RSS reported in the summary line should be flat across batches
   instead of climbing, and independent of total page count. Confirm no OOM:
   `docker inspect rag_celery_worker_ingestion --format '{{.State.OOMKilled}}'`.
4. **Fix E took effect**: query the status DB after a full run and confirm `attempts` is still 0 and
   the document reached `embedded` in one pass:
   `SELECT stage, attempts, last_error FROM documents WHERE id = '<doc_id>';`
5. **No regression in output quality**: diff the new `data/parsed/<doc_id>_<name>.md` against the
   pre-change one. Fix D reorders VLM resolution, so verify figure/table placement is unchanged and
   that the count of `[IMAGE]` fallbacks did not increase.
6. **Tests**: `python -m pytest tests/unit -v --ignore=tests/unit/test_pdf_to_markdown.py`. Note
   [docs/ARCHITECTURE.md:350-357](docs/ARCHITECTURE.md#L350-L357) documents 12 pre-existing
   failures — the count must not grow.
