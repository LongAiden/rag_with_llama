# Plan — cutting retrieval latency for vector search + cross-encoder rerank (CPU/Docker)

**Date**: 2026-08-18
**Status**: plan. Nothing here is applied.
**Revision**: v2 — reordered after a code audit. v1 led with the two levers that cost retrieval
quality (smaller model, fewer candidates) and missed the free one (thread caps). See
[§ Changelog](#changelog).
**Follows**: `docs/ARCHITECTURE.md` §5 (Query Pipeline), §10.2 (Performance limitations).
**Scope**: the synchronous `/query` path on the `app` container only. Ingestion/parse is out of
scope (see `docs/plans/20260812_parse_time_reduction.md` for that).

---

## Part 1 — Diagnosis: where the ~1.2 s before the LLM call goes

### The trace

Vector-only search (`search_mode="vector"`), CPU inside Docker, `rag_app`:

```
17:31:40.957 document_search
17:31:40.960   embedding_generation_for_search
17:31:40.961     Generating embeddings for search query
17:31:41.189     Vector search completed
17:31:41.190   cross_encoder_reranking
17:31:41.191     reranking
17:31:41.191       Computing rerank scores
17:31:42.089       Reranking completed
17:31:42.166     Cross-encoder reranking completed
17:31:42.167   context_building
17:31:42.168     Context built
17:31:42.173   LLM request
```

### Breakdown

| stage | start | end | duration | share of retrieval |
|---|---|---|---|---|
| embedding generation + pgvector search | 40.960 | 41.189 | **~229 ms** | 19% |
| cross-encoder reranking | 41.190 | 42.166 | **~976 ms** | **80%** |
| context building | 42.167 | 42.168 | ~1 ms | <1% |

Retrieval (everything before `LLM request`) is **~1.2 s**, and the cross-encoder is four times
the combined embedding + vector search. This is the single lever worth pulling.

### Why reranking is ~976 ms

**The headline number is 45 ms/pair, and that is 3-4× slower than this model should be.**
`cross-encoder/ms-marco-MiniLM-L-6-v2` is 6 layers / 22.7M params; at 512 tokens on one modern
core it costs ~12-15 ms/pair. The 45 ms is the finding, not the baseline — something is wasting
2/3 of the time before any model swap is considered. Ranked by contribution:

1. **Torch thread pool is unbounded against a 1.0-CPU quota.** `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `torch.set_num_threads` appear **nowhere in the
   repo** — already logged as F9 in
   [20260804_ingestion_performance_investigation.md:146](../20260804_ingestion_performance_investigation.md#L146)
   for the workers, never applied to `app`. Torch sizes its intra-op pool from `os.cpu_count()`,
   which inside the container reports the Docker VM's **6** CPUs, while
   [docker-compose.yml:255](../../docker-compose.yml#L255) caps `app` at `cpus: "1.0"`. Docker
   enforces that as a CFS bandwidth quota, so six threads burn the 100 ms period's quota in ~16 ms
   of wall time and then sit throttled — while still paying a 6-way sync barrier on every GEMM.
   This is the leading explanation for the 45 ms/pair and it costs nothing to fix.

2. **Sequence length is uncapped.** `CrossEncoder(model_name)`
   ([reranking.py:54](../../src/app/retrieval/reranking.py#L54)) takes the model's own
   `max_length` (512). `DEFAULT_CHUNK_SIZE=512`
   ([app_config.py:105](../../src/app/config/app_config.py#L105)) means pairs run at or near that
   ceiling, and attention cost is quadratic in sequence length. Nothing truncates.

3. **`HYBRID_LIMIT = 20` is hardcoded**
   ([search.py:76](../../src/app/retrieval/search.py#L76)). Vector search returns 20 candidates,
   and `Reranker.rerank` scores **all 20** query↔chunk pairs before slicing to `rerank_top_k=5`
   ([reranking.py:98](../../src/app/retrieval/reranking.py#L98), [:136-137](../../src/app/retrieval/reranking.py#L136-L137)).
   `top_k` only filters *after* `self.model.predict(pairs)` — it does not bound the work. Real,
   but halving it costs recall, so it is not the first lever.

4. **The reranker is lazy-loaded on the first query**
   ([utils.py:58-81](../../src/app/retrieval/utils.py#L58-L81)). `get_reranker` runs in
   `asyncio.to_thread` and the `CrossEncoder` constructor downloads/loads weights — multi-second
   cold start. This trace is warm (no load gap before `Computing rerank scores`), so the hit is
   paid by the first user after every container start. Does not affect the steady-state number
   above, but it is the worst single query anyone experiences.

### Why the rest is not the bottleneck

- **Embedding generation** is a single `all-MiniLM-L6-v2` encode of one short query
  ([generator.py:40](../../src/app/ingestion/embedding/generator.py#L40), called via
  [pipeline.py:477-479](../../src/app/ingestion/embedding/pipeline.py#L477-L479) in
  `asyncio.to_thread`). ~50-80 ms of the 229 ms. It shares the same throttled thread pool, so
  Step 0 should shave this a little too — treat any gain as a bonus, not a target.
- **pgvector HNSW search** over 20 nearest neighbours with cosine ops
  ([vector_store.py:198-233](../../src/app/ingestion/embedding/vector_store.py#L198-L233)) is the
  remainder, ~150 ms. Note the threshold is a **post-filter** in the `WHERE` clause
  ([:227](../../src/app/ingestion/embedding/vector_store.py#L227)), so the scan cost is driven by
  `ef_search`, not by `limit` — lowering `limit` will *not* speed this stage up. Not addressable
  here.
- **BM25 / RRF** are skipped because `search_mode="vector"` (see
  [search.py:93-106](../../src/app/retrieval/search.py#L93-L106)). When hybrid *is* used, BM25
  rebuilds the index every query
  ([vector_store.py:269-337](../../src/app/ingestion/embedding/vector_store.py#L269-L337)) — known
  O(n) landmine, flagged in ARCHITECTURE.md §10.2, **not in scope for this plan**.
- **Sibling expansion** ([search.py:180-200](../../src/app/retrieval/search.py#L180-L200)) only
  fires for structural queries (`how many`, `list all`, …); the trace query does not match
  `_STRUCTURAL_RE`.

### Headroom, sized

| lever | size | quality cost | confidence |
|---|---|---|---|
| cap torch threads to the CPU quota | ~2-3× on rerank | **none** | strong — 45 ms/pair vs 12-15 ms expected |
| `max_length=256` on the CrossEncoder | ~1.5-2× | small (sees first 256 tok of each chunk) | arithmetic from quadratic attention |
| preload reranker at startup | removes first-query cold start only | none | exact |
| score 10 pairs instead of 20 | ~2× fewer `predict` rows | **real** — halves rerank's rescue range | exact |
| smaller cross-encoder (TinyBERT-L-2) | ~2-3× faster inference | **real** — MAP ~0.79 vs ~0.82 | arithmetic from layer count |
| raise `app` `limits.cpus` 1.0 → 2.0 | ~linear on rerank | none | competes with embedding |
| ONNX/quantized reranker | further 1.5-2× | none | unmeasured on these models |

The ordering below follows this table: **free levers first, quality-costing levers last and one at
a time.** ONNX stays Phase 2.

---

## Part 2 — What to do, in order

Phases 0-2 are ~15 lines and cost no retrieval quality. **Measure after Phase 0 before writing
any of Phase 3-4** — if rerank lands under 350 ms on thread caps alone, Phases 3-4 are dead code
you don't have to defend later.

### Step 0 — Cap the torch thread pool (env only, no code)

**`docker-compose.yml`** — in the `app` service `environment:` block (not `x-common-env`; the
workers want a different value, see the note):

```yaml
      # Torch sizes its intra-op pool from os.cpu_count(), which reports the
      # Docker VM's 6 CPUs, while limits.cpus below caps this container at 1.0.
      # Six threads against a 1.0 CFS quota burn the period in ~16ms and then
      # sit throttled, paying a 6-way barrier per GEMM. Measured symptom:
      # 45ms/pair on a 6-layer MiniLM cross-encoder that should cost 12-15ms.
      # Keep this equal to limits.cpus.
      OMP_NUM_THREADS: ${APP_OMP_NUM_THREADS:-1}
      MKL_NUM_THREADS: ${APP_OMP_NUM_THREADS:-1}
      OPENBLAS_NUM_THREADS: ${APP_OMP_NUM_THREADS:-1}
```

Do **not** put these in `x-common-env`. The parse workers are deliberately allowed 6 threads
(`DOCLING_NUM_THREADS: 6` at [docker-compose.yml:50](../../docker-compose.yml#L50), matching
`cpus: "6.0"`); pinning them to 1 would wreck ingestion.

Env vars are read by OpenMP at library load, so they must be set before the process starts —
that is why this is compose, not Python. Nothing in the app reads them, so there is no config
field and no code change.

**Measure here.** Re-run the trace query and record `cross_encoder_reranking`. This is a
single-variable change; if it does not move, the throttling hypothesis is wrong and Steps 3-4
become the primary levers instead of the fallback.

### Step 1 — Bound the cross-encoder sequence length (config + 1 line)

**`src/app/config/app_config.py`** — next to `rerank_model` at
[:81](../../src/app/config/app_config.py#L81):

```python
# Cross-encoder input truncation. Chunks target DEFAULT_CHUNK_SIZE=512 tokens
# and the model's own max_length is 512, so pairs run at the ceiling where
# attention cost is quadratic. 256 roughly halves per-pair cost while the
# reranker still SEES every candidate — unlike cutting vector_search_limit,
# which removes candidates from consideration entirely.
rerank_max_length: int = Field(default=256, validation_alias='RERANK_MAX_LENGTH')
```

**`src/app/retrieval/reranking.py`** — thread it through the constructor at
[:40](../../src/app/retrieval/reranking.py#L40) and [:54](../../src/app/retrieval/reranking.py#L54):

```python
    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        max_length: int = 256,
    ):
        ...
        self.model = CrossEncoder(model_name, max_length=max_length)
```

**`src/app/retrieval/utils.py`** — pass it at [:79](../../src/app/retrieval/utils.py#L79):

```python
            config.reranker = Reranker(
                model_name=rerank_model,
                max_length=config.settings.rerank_max_length,
            )
```

Measure again. Set `RERANK_MAX_LENGTH=512` to revert without a rebuild.

### Step 2 — Preload the reranker at app startup, safely

Three sub-parts. **All three are required** — preloading without the cache fix makes every
container recreate download the model during startup, and without the try/except a failed
download becomes a boot loop.

#### 2a. Bake the model into the image

[Dockerfile:78-85](../../deploy/deployment/Dockerfile#L78-L85) already pre-downloads the
embedder. `app` **does not mount `hf_cache`** — only the two celery workers do
([docker-compose.yml:332](../../docker-compose.yml#L332),
[:389](../../docker-compose.yml#L389)) — so for the `app` container the cross-encoder would land
in the writable layer and be re-fetched on every `--force-recreate`, exactly the failure mode the
`hf_cache` comment at [:577-580](../../docker-compose.yml#L577-L580) documents for docling.

Extend the existing pre-warm step rather than adding a volume:

```dockerfile
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config.app_config import DEFAULT_EMBEDDING_MODEL, AppSettings
SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
CrossEncoder(AppSettings().rerank_model)
PY
```

Baking beats mounting here: it is immutable across recreates and works with no network. The one
gap is a runtime `RERANK_MODEL` override to something not baked — that still downloads on first
use. If you expect to A/B models at runtime, also add `- hf_cache:/home/appuser/.cache/huggingface`
to the `app` service volumes; otherwise skip it.

#### 2b. Eager wrapper

**`src/app/retrieval/utils.py`** — `get_reranker` already does thread-safe lazy init
([:55-81](../../src/app/retrieval/utils.py#L55-L81)). Add:

```python
def preload_reranker(config) -> None:
    """Eagerly construct the cross-encoder. Call from app startup in a thread."""
    get_reranker(config)
```

No lock needed — `get_reranker` already serialises first construction. The
`preload_reranker` setting is checked by the caller, not here, so the function stays a plain
"do it now" primitive.

#### 2c. Lifespan that cannot kill the container

**`src/app/api/app.py`** — the app has no lifespan today
([app.py:33](../../src/app/api/app.py#L33) is a plain `FastAPI(...)`).

```python
from contextlib import asynccontextmanager
import asyncio
import logfire
from app.retrieval.utils import preload_reranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.settings.preload_reranker:
        try:
            await asyncio.to_thread(preload_reranker, config)
        except Exception as e:
            # Never fail startup on this. Reranker.__init__ re-raises
            # (reranking.py:60), and search.py:142-144 already degrades to
            # vector-only scores on a reranker failure — an unguarded raise here
            # would convert that soft degradation into an unhealthy container
            # and a restart: unless-stopped loop.
            logfire.error("Reranker preload failed; falling back to lazy load",
                          error=str(e))
    yield

app = FastAPI(title="pgvector RAG API", version="1.0.0", lifespan=lifespan)
```

`config` is already imported at [app.py:22](../../src/app/api/app.py#L22). Loading a CrossEncoder
is blocking CPU work, so it must go through `asyncio.to_thread` or startup stalls the event loop
and the health check will not answer until the model is resident. Baked-in weights make this a
~1-3 s disk load, comfortably inside the 40 s `start_period`
([docker-compose.yml](../../docker-compose.yml)).

Memory: the cross-encoder is ~90 MB fp32 (MiniLM-L-6-v2, 22.7M params) on top of the documented
392M baseline ([docker-compose.yml:256-259](../../docker-compose.yml#L256-L259)). Peak RSS does
not change — preloading moves the allocation earlier, it does not add one — but the floor rises.
Watch `docker stats rag_app`; bump the 768M limit to 896M if it crowds.

### Step 3 — Make the retrieval knobs configurable, defaults unchanged

This step is **observability, not speed**. Ship it with the current values so Step 4 has a
one-line revert.

**`src/app/config/app_config.py`**:

```python
# Candidates pgvector returns before reranking. Was hardcoded HYBRID_LIMIT=20
# in retrieval/search.py. Kept at 20: lowering it is a recall tradeoff, not a
# free win — see Step 4.
vector_search_limit: int = Field(default=20, validation_alias='VECTOR_SEARCH_LIMIT')
# Final top-k after cross-encoder reranking. Was three literals in query_routes.py.
rerank_top_k: int = Field(default=5, validation_alias='RERANK_TOP_K')
# Eagerly load the cross-encoder on FastAPI startup so the first /query does
# not pay the model-load hit.
preload_reranker: bool = Field(default=True, validation_alias='PRELOAD_RERANKER')
```

`rerank_model` default stays `cross-encoder/ms-marco-MiniLM-L-6-v2` — see Step 4.

**`src/app/retrieval/search.py`** — replace the hardcoded
[:76](../../src/app/retrieval/search.py#L76) `HYBRID_LIMIT = 20`:

```python
        # Candidate depth for the reranker. Guarded because rerank_top_k slices
        # AFTER predict(): a depth below top_k would silently return fewer than
        # the caller asked for.
        candidate_depth = max(config.settings.vector_search_limit, rerank_top_k)
        logfire.info("Retrieval knobs",
                     candidate_depth=candidate_depth,
                     rerank_top_k=rerank_top_k,
                     rerank_model=config.settings.rerank_model,
                     rerank_max_length=config.settings.rerank_max_length)
```

and use `candidate_depth` at both [:83](../../src/app/retrieval/search.py#L83) (vector) and
[:97](../../src/app/retrieval/search.py#L97) (BM25), so the two lists fed into RRF keep equal
depth. Fixing BM25's O(n) rebuild stays out of scope — noted in Risks.

**Dead-parameter cleanup.** With `enable_reranking=True` the `limit` argument is *entirely
ignored*: the final count comes from `rerank_top_k`
([search.py:117-137](../../src/app/retrieval/search.py#L117-L137)) and `limit` only applies on the
`else` branch at [:147](../../src/app/retrieval/search.py#L147). Adding a third depth knob on top
of two that already conflict makes this worse, so either document `limit` as
"non-reranked path only" in the docstring at [:52](../../src/app/retrieval/search.py#L52) or drop
it from the reranked path explicitly. Do not leave it ambiguous.

**`src/app/api/routes/query_routes.py`** — `rerank_top_k` is hardcoded in **three** places, not
one: [:84](../../src/app/api/routes/query_routes.py#L84) (signature default),
[:149](../../src/app/api/routes/query_routes.py#L149) (`request.rerank_top_k or 5`), and
[:196](../../src/app/api/routes/query_routes.py#L196) (upload path). Point all three at
`config.settings.rerank_top_k`.

**`docker-compose.yml`** — add to `x-common-env` near [:50](../../docker-compose.yml#L50). Only
`app` reads them; the anchor is shared and the defaults are inert for workers:

```yaml
  RERANK_MODEL: ${RERANK_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}
  RERANK_MAX_LENGTH: ${RERANK_MAX_LENGTH:-256}
  VECTOR_SEARCH_LIMIT: ${VECTOR_SEARCH_LIMIT:-20}
  RERANK_TOP_K: ${RERANK_TOP_K:-5}
  PRELOAD_RERANKER: ${PRELOAD_RERANKER:-true}
```

Note: `RERANK_MODEL` is currently **not** passed through `x-common-env`, so the
`RERANK_MODEL=` line already in [.env.example:72](../../.env.example#L72) never reaches the
container — anyone who set it has been silently running the pydantic default. Adding it here
fixes a live bug; call it out in the commit message.

**`.env.example`** — under the existing "Retrieval & Embedding" section at
[:68-72](../../.env.example#L68-L72), add the four new lines with the same commenting style as the
PDF-parse block.

### Step 4 — Quality-costing levers, only if still needed, one at a time

Gate: run this **only if `cross_encoder_reranking` is still > 350 ms after Steps 0-2**. Apply as
two separate commits with a measurement between them, so a quality regression is attributable.

**4a. `VECTOR_SEARCH_LIMIT=10`** — env-only after Step 3. Note this does *not* speed up pgvector
(threshold is a post-filter, `ef_search` drives the scan); the entire win is halving `predict`
rows. Cost: the reranker's job is rescuing chunks buried at rank 8-20 in vector order, and this
halves that range.

**4b. `RERANK_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2`** — env-only after Step 3, but the
model must be baked (Step 2a bakes whatever `AppSettings().rerank_model` resolves to at build
time, so switching the default requires a rebuild; an env-only switch downloads on first use).
2 layers vs 6, ~17 MB vs ~90 MB, MAP ~0.79 vs ~0.82 on MS-MARCO.

Doing 4a and 4b together degrades both the candidate pool and the scorer at once, which is how you
end up unable to explain a quality regression three weeks later — the same lesson as
`docs/plans/20260812_parse_time_reduction.md` Step 4.

---

## Files

| Phase | File | Change |
|---|---|---|
| 0 | `docker-compose.yml` | `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` on the **`app` service only** |
| 1 | `src/app/config/app_config.py` | Add `rerank_max_length` |
| 1 | `src/app/retrieval/reranking.py` | `Reranker.__init__` takes `max_length`, passes to `CrossEncoder` |
| 1 | `src/app/retrieval/utils.py` | Pass `max_length` from settings |
| 2 | `deploy/deployment/Dockerfile` | Pre-warm `CrossEncoder` alongside the embedder |
| 2 | `src/app/retrieval/utils.py` | Add `preload_reranker(config)` eager wrapper |
| 2 | `src/app/api/app.py` | `lifespan` calling `preload_reranker` in `asyncio.to_thread`, **wrapped in try/except** |
| 3 | `src/app/config/app_config.py` | Add `vector_search_limit=20`, `rerank_top_k=5`, `preload_reranker=True` |
| 3 | `src/app/retrieval/search.py` | `HYBRID_LIMIT` → `candidate_depth = max(vector_search_limit, rerank_top_k)`; log knobs; resolve the dead `limit` param |
| 3 | `src/app/api/routes/query_routes.py` | Three `rerank_top_k` literals → `config.settings.rerank_top_k` |
| 3 | `docker-compose.yml` | Pass the five retrieval vars through `x-common-env` (fixes `RERANK_MODEL` never reaching the container) |
| 3 | `.env.example` | Document the knobs |
| 4 | env only | `VECTOR_SEARCH_LIMIT=10`, then `RERANK_MODEL=…TinyBERT-L-2-v2` (+ rebuild) |

No DB migration. No new Python dependency. No change to `vector_store.py` schema or indexes.

## Expected effect

| metric | before | after Phase 0 | after Phase 1 | after Phase 4 (if needed) |
|---|---|---|---|---|
| `cross_encoder_reranking` | ~976 ms | ~350-500 ms | ~200-300 ms | ~100-150 ms |
| ms/pair | ~45 | ~15-25 | ~10-15 | ~5-8 |
| pairs scored | 20 | 20 | 20 | 10 |
| retrieval quality | baseline | **unchanged** | ~unchanged | measurably lower |
| first-query cold start | multi-second | multi-second | multi-second | none (from Phase 2) |
| `embedding_generation_for_search` + vector search | ~229 ms | ~180-229 ms | unchanged | unchanged |

Target: retrieval before LLM drops from **~1.2 s** to **~0.5 s with no quality cost** (Phases
0-2). Phase 4 buys another ~150 ms and is probably not worth it.

## Tradeoffs

- **`OMP_NUM_THREADS=1` is right only while `limits.cpus` is `1.0`.** They must move together. If
  the `app` quota is later raised to 2.0, raise `APP_OMP_NUM_THREADS` to 2 or the extra core sits
  idle. The `${APP_OMP_NUM_THREADS:-1}` indirection exists to make that a one-line `.env` change.
- **`RERANK_MAX_LENGTH=256` truncates long chunks.** The cross-encoder scores the first 256
  tokens of a 512-token chunk. For chunks whose relevance signal sits in the tail this shifts
  ranking. Milder than dropping candidates — every candidate is still scored — but not free.
  Revert with `RERANK_MAX_LENGTH=512`.
- **Startup time increases by the reranker load** (~1-3 s from baked weights). Acceptable: the
  alternative is the first user paying it inline. The 40 s healthcheck `start_period` absorbs it.
- **Phase 4a gives the reranker fewer candidates.** If the right chunk is at rank 11-20 in pure
  vector order, reranking cannot rescue it. On small corpora (<10k chunks) unlikely to matter; on
  large corpora, don't.
- **Phase 4b's TinyBERT-L-2 ranks worse** (MAP ~0.79 vs ~0.82). On a 10-20 chunk candidate set the
  practical difference is usually small, but it is a real regression and it is the last thing to
  try, not the first.

## Risks

- **Thread caps could regress something else in `app`.** The app process also runs the query-time
  embedder. Pinning to 1 thread is correct under a 1.0-CPU quota — you cannot use more than one
  core's worth regardless — but confirm `embedding_generation_for_search` does not *rise* in the
  Step 0 measurement. If it does, the quota is not actually binding and the diagnosis needs
  revisiting.
- **Baking the cross-encoder grows the image** by ~90 MB. Both worker services share the same
  image ([docker-compose.yml](../../docker-compose.yml) `image: rag_with_llama:latest`) and don't
  need the reranker, so they carry the weight for nothing. Acceptable; splitting the image is a
  bigger change than this plan.
- **A runtime `RERANK_MODEL` override bypasses the bake** and downloads on first use — during
  lifespan, with no `hf_cache` volume on `app`. The try/except in 2c keeps that from bricking
  startup, but the first query then pays a lazy load. Add the volume if you plan to A/B models.
- **`app` memory limit 768M.** Preloading raises the floor by ~90 MB before the first request.
  Peak does not change. Watch `docker stats rag_app` for OOM-kill; bump to 896M if needed — the
  comment at [:256-259](../../docker-compose.yml#L256-L259) documents the 392M baseline.
- **BM25 is still O(n) per query.** Not addressed here. If `search_mode="hybrid"` is enabled
  later, the BM25 rebuild will dominate and this plan's gains will be masked. Track separately.

## Not recommended (in this plan)

- **ONNX / quantized reranker.** Real further win (1.5-2×), but adds `optimum` + `onnxruntime`
  dependencies and a model-conversion step. Phase 2, gated on Steps 0-3 not being enough — and
  after thread caps, much of what ONNX would have bought is already collected.
- **Raising `app` `limits.cpus` 1.0 → 2.0.** Genuinely helps, but do it *after* Step 0, not
  instead of it: raising the quota without capping threads leaves 6 threads on 2 cores and keeps
  most of the thrash. If you do raise it, raise `APP_OMP_NUM_THREADS` to match. The Docker VM has
  6 cores shared with parse workers.
- **Replacing the cross-encoder with a bi-encoder re-rank or ColBERT.** Order-of-magnitude
  faster but a different retrieval architecture; out of scope for a tuning plan.

## Verification

Run after **each** phase, not once at the end — the whole point of the ordering is attribution.

```bash
# 1. Rebuild and start the app service
docker compose up -d --build app

# 2. Confirm the threads actually took (Phase 0)
docker compose exec app python -c "import torch; print('torch threads:', torch.get_num_threads())"
# expect: 1

# 3. Confirm the reranker loaded at startup (Phase 2) — this line must appear
#    BEFORE any /query is sent
docker compose logs app | grep -E "Reranker initialized|Application startup complete"

# 4. Send the same query as the trace and capture the new spans.
#    NOTE: 'app' is the compose service; 'rag_app' is the container_name and is
#    NOT a valid argument to `docker compose logs`.
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"<same query>","limit":5,"table_name":"test"}'

docker compose logs app | grep -E "document_search|Retrieval knobs|cross_encoder|Vector search completed|Reranking completed|Context built|LLM request"
```

Pass criteria, per phase:

| after | span | target |
|---|---|---|
| Phase 0 | `cross_encoder_reranking` | **< 500 ms**, ms/pair < 25 |
| Phase 1 | `cross_encoder_reranking` | **< 350 ms** |
| Phase 2 | first-query model-load gap | **0** (loaded at startup) |
| all | `embedding_generation_for_search` + vector search | ~229 ms or better, never worse |

Correctness — retrieval quality must not move in Phases 0-3:

```bash
# Phases 0 and 2 change no scoring input: the top-5 chunk_ids and rerank_scores
# must be IDENTICAL to the pre-change run. Any difference means something other
# than threading changed and should be investigated before continuing.
#
# Phase 1 (max_length) may reorder chunks longer than 256 tokens. Compare top-5
# sets; a swap within the set is expected, a chunk dropping out is a signal.
#
# Phase 4 is expected to reorder. Compare against the Phase 3 baseline, not the
# original, and decide whether the ~150ms is worth it.
```

---

## Changelog

**v2 (this revision)** — reordered and corrected after auditing the plan against the code:

- **Added Step 0 (thread caps)**, absent from v1 and the likely root cause. The 45 ms/pair figure
  v1 used as a *baseline* is 3-4× the expected cost for a 6-layer MiniLM and is itself the
  finding.
- **Added Step 1 (`max_length`)**, absent from v1 — a ~2× win that keeps every candidate scored.
- **Demoted the model swap and `VECTOR_SEARCH_LIMIT=10`** from Steps 1-2 to a gated, split
  Step 4. Both cost retrieval quality; v1 applied them together and first, making a regression
  unattributable.
- **v1 default `vector_search_limit=10` → 20** and `rerank_model` left at MiniLM-L-6, so Step 3
  ships as a pure no-op refactor.
- **Fixed: `app` does not mount `hf_cache`.** v1's Risks claimed the volume caches the download
  across restarts; only the celery workers mount it. Baking into the Dockerfile is now a required
  sub-step of preloading rather than a contingency.
- **Fixed: lifespan had no error handling.** `Reranker.__init__` re-raises, so v1's version turned
  a soft degradation (`search.py:142-144` already falls back to vector scores) into a boot loop.
- **Fixed model sizes.** v1 said "TinyBERT-L-2 (~67 MB) is smaller than MiniLM-L-6 (~22 MB)",
  which is self-contradictory. Actual: TinyBERT-L-2-v2 ~4.4M params (~17 MB), MiniLM-L-6-v2
  ~22.7M params (~90 MB).
- **Fixed `rerank_top_k` scope.** v1 called `query_routes.py` an "audit only — may already be
  literal 5"; there are three literal sites.
- **Fixed the verification command.** v1 used `docker compose logs rag_app`; `rag_app` is the
  `container_name`, not the service, so that command errors.
- **Added** the `candidate_depth >= rerank_top_k` guard and a note that `limit` is a dead
  parameter on the reranked path.
- **Noted** that lowering `limit` does not speed up pgvector — the threshold is a post-filter and
  `ef_search` drives the scan.
