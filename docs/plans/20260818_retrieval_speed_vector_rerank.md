# Plan — cutting retrieval latency for vector search + cross-encoder rerank (CPU/Docker)

**Date**: 2026-08-18
**Status**: plan. Nothing here is applied.
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

1. **`HYBRID_LIMIT = 20` is hardcoded**
   ([search.py:76](../../src/app/retrieval/search.py#L76)). Vector search returns 20 candidates,
   and `Reranker.rerank` scores **all 20** query↔chunk pairs before slicing to `rerank_top_k=5`
   ([reranking.py:98](../../src/app/retrieval/reranking.py#L98), [:136-137](../../src/app/retrieval/reranking.py#L136-L137)).
   `top_k` only filters *after* `self.model.predict(pairs)` — it does not bound the work.

2. **The model is `cross-encoder/ms-marco-MiniLM-L-6-v2`** (6 layers), configured at
   [app_config.py:81](../../src/app/config/app_config.py#L81). Cross-encoders run full
   query↔chunk transformer inference per pair, ~45 ms/pair on CPU in the Docker VM → 20 pairs ≈
   900 ms. This matches the trace.

3. **The reranker is lazy-loaded on the first query**
   ([utils.py:58-81](../../src/app/retrieval/utils.py#L58-L81)). `get_reranker` runs in
   `asyncio.to_thread` and the `CrossEncoder` constructor downloads/loads weights — multi-second
   cold start. This trace is warm (no load gap before `Computing rerank scores`), so the hit is
   paid by the first user after every container start.

4. **CPU-only torch inside Docker.**
   [requirements.txt](../../deploy/deployment/requirements.txt) installs CPU-only torch; the `app`
   service is capped at `cpus: "1.0"` and `memory: 768M`
   ([docker-compose.yml:254-259](../../docker-compose.yml#L254-L259)). No GPU/MPS passthrough on
   Apple Silicon Docker Desktop. Raising the CPU quota helps linearly but competes with the
   embedding model for the same core.

### Why the rest is not the bottleneck

- **Embedding generation** is a single `all-MiniLM-L6-v2` encode of one short query
  ([generator.py:40](../../src/app/ingestion/embedding/generator.py#L40), called via
  [pipeline.py:477-479](../../src/app/ingestion/embedding/pipeline.py#L477-L479) in
  `asyncio.to_thread`). ~50-80 ms of the 229 ms.
- **pgvector HNSW search** over 20 nearest neighbours with cosine ops
  ([vector_store.py:110-113](../../src/app/ingestion/embedding/vector_store.py#L110-L113)) is the
  remainder, ~150 ms, and only grows if the chunk table grows large. Not addressable here.
- **BM25 / RRF** are skipped because `search_mode="vector"` (see
  [search.py:93-106](../../src/app/retrieval/search.py#L93-L106)). When hybrid *is* used, BM25
  rebuilds the index every query
  ([vector_store.py:269-337](../../src/app/ingestion/embedding/vector_store.py#L269-L337)) — known
  O(n) landmine, flagged in ARCHITECTURE.md §10.2, **not in scope for this plan**.
- **Sibling expansion** ([search.py:180-200](../../src/app/retrieval/search.py#L180-L200)) only
  fires for structural queries (`how many`, `list all`, …); the trace query does not match
  `_STRUCTURAL_RE`.

### Headroom, sized

| lever | size | confidence |
|---|---|---|
| smaller cross-encoder (TinyBERT-L-2) | ~2-3× faster inference | arithmetic from layer count |
| score 10 pairs instead of 20 | ~2× fewer `predict` calls | exact |
| preload reranker at startup | removes first-query cold start only | exact |
| raise `app` `limits.cpus` 1.0 → 2.0 | ~linear on rerank | competes with embedding |
| ONNX/quantized reranker | further 1.5-2× | unmeasured on these models |

The first three are config + ~20 lines, no new dependency. ONNX is Phase 2.

---

## Part 2 — What to do, in order

### Step 1 — Make the retrieval knobs configurable (config)

**`src/app/config/app_config.py`** — add to `AppSettings` next to
[:81](../../src/app/config/app_config.py#L81):

```python
# How many candidates pgvector returns before reranking. Was hardcoded
# HYBRID_LIMIT=20 in retrieval/search.py; 10 cuts cross-encoder work in half.
vector_search_limit: int = Field(default=10, validation_alias='VECTOR_SEARCH_LIMIT')
# Final top-k after cross-encoder reranking.
rerank_top_k: int = Field(default=5, validation_alias='RERANK_TOP_K')
# Eagerly load the cross-encoder on FastAPI startup so the first /query does
# not pay the multi-second model-load hit.
preload_reranker: bool = Field(default=True, validation_alias='PRELOAD_RERANKER')
```

Change the `rerank_model` default to the faster model:

```python
rerank_model: str = Field(
    default='cross-encoder/ms-marco-TinyBERT-L-2-v2',
    validation_alias='RERANK_MODEL',
)
```

TinyBERT-L-2 is 2 layers vs MiniLM-L-6's 6 — same tokenizer family, ~2-3× faster on CPU, slightly
lower ranking quality. Revertible to MiniLM-L-6 via `RERANK_MODEL` env without code change.

### Step 2 — Use the config in the search path (search)

**`src/app/retrieval/search.py`** — replace the hardcoded
[:76](../../src/app/retrieval/search.py#L76) `HYBRID_LIMIT = 20` with a read from config, and
thread `rerank_top_k` from settings when the caller did not override it.

```python
hybrid_limit = config.settings.vector_search_limit
...
vector_results = await pipeline.search_documents(
    query=query,
    limit=hybrid_limit,
    threshold=threshold,
    document_ids=document_ids,
)
```

For BM25 (only used when `search_mode="hybrid"`), pass the same `hybrid_limit` to
`pipeline.vector_store.search_bm25` at [:95-99](../../src/app/retrieval/search.py#L95-L99) so the
two lists fed into RRF have equal depth. (Out of scope to fix the O(n) rebuild — noted in
Risks.)

For the `rerank_top_k` parameter at [:44](../../src/app/retrieval/search.py#L44): keep the function
argument as an override, but default it to `config.settings.rerank_top_k` at the call site in
`query_routes.py` rather than the literal `5`. Add a `logfire.info` line recording
`hybrid_limit`, `rerank_top_k`, `rerank_model` once per query so future traces are self-describing.

### Step 3 — Preload the reranker at app startup (startup)

**`src/app/retrieval/utils.py`** — `get_reranker` already does thread-safe lazy init
([:55-81](../../src/app/retrieval/utils.py#L55-L81)). Add a thin eager wrapper:

```python
def preload_reranker(config) -> None:
    """Eagerly construct the cross-encoder. Call from app startup in a thread."""
    if config.settings.preload_reranker:
        get_reranker(config)
```

No lock needed here — `get_reranker` already serialises first construction.

**`src/app/api/app.py`** — add a FastAPI `lifespan` handler. Today the app has no lifespan
([app.py:32-34](../../src/app/api/app.py#L32-L34)); `app = FastAPI(...)` is plain.

```python
from contextlib import asynccontextmanager
import asyncio
from app.api.dependencies import config
from app.retrieval.utils import preload_reranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.settings.preload_reranker:
        await asyncio.to_thread(preload_reranker, config)
    yield

app = FastAPI(title="pgvector RAG API", version="1.0.0", lifespan=lifespan)
```

Loading a CrossEncoder is blocking CPU work — it must go through `asyncio.to_thread` or startup
will stall the event loop and the health check will not answer until the model is resident. The
`app` container's 768M memory limit
([docker-compose.yml:259](../../docker-compose.yml#L259)) already covers the baked-in MiniLM
embedder plus torch; TinyBERT-L-2 (~67 MB) is smaller than MiniLM-L-6 (~22 MB cross-encoder +
shared tokenizer), so no memory bump is needed. Verify RSS on the first query with `docker stats
rag_app`.

### Step 4 — Plumb the env vars through compose (env)

**`docker-compose.yml`** — add to the `x-common-env` anchor near
[:50](../../docker-compose.yml#L50) (alongside the other tunables). Only the `app` service reads
them, but the anchor is shared and the defaults are inert for workers:

```yaml
RERANK_MODEL: ${RERANK_MODEL:-cross-encoder/ms-marco-TinyBERT-L-2-v2}
VECTOR_SEARCH_LIMIT: ${VECTOR_SEARCH_LIMIT:-10}
RERANK_TOP_K: ${RERANK_TOP_K:-5}
PRELOAD_RERANKER: ${PRELOAD_RERANKER:-true}
```

Note: `RERANK_MODEL` is currently *not* passed through `x-common-env` — the app relies on the
pydantic default. Adding it here makes the override actually reach the container.

**`.env.example`** — update [:72](../../.env.example#L72) and add the three new lines under the
"Retrieval & Embedding" section:

```env
# Smaller cross-encoder (2 layers) than MiniLM-L-6 (6 layers). ~2-3× faster on
# CPU at a small quality cost; revert to cross-encoder/ms-marco-MiniLM-L-6-v2
# if ranking quality drops.
RERANK_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
# Candidates pgvector returns before reranking. 10 halves cross-encoder work
# vs the previous hardcoded 20; raise if vector recall is weak on large corpora.
VECTOR_SEARCH_LIMIT=10
# Final top-k after cross-encoder reranking.
RERANK_TOP_K=5
# Load the cross-encoder on app startup so the first /query is not penalised.
PRELOAD_RERANKER=true
```

---

## Files

| File | Change |
|---|---|
| `src/app/config/app_config.py` | Add `vector_search_limit`, `rerank_top_k`, `preload_reranker` fields; change `rerank_model` default to TinyBERT-L-2 |
| `src/app/retrieval/search.py` | Replace `HYBRID_LIMIT = 20` with `config.settings.vector_search_limit`; log the knobs; thread `rerank_top_k` from settings |
| `src/app/retrieval/utils.py` | Add `preload_reranker(config)` eager wrapper |
| `src/app/api/app.py` | Add `lifespan` that calls `preload_reranker` in `asyncio.to_thread` |
| `docker-compose.yml` | Pass `RERANK_MODEL`, `VECTOR_SEARCH_LIMIT`, `RERANK_TOP_K`, `PRELOAD_RERANKER` through `x-common-env` |
| `.env.example` | Document the four knobs; update `RERANK_MODEL` default |
| `src/app/api/routes/query_routes.py` | Default `rerank_top_k` to `config.settings.rerank_top_k` at the call site (audit only — may already be literal `5`) |

No DB migration. No new Python dependency. No change to `vector_store.py` schema or indexes.

## Expected effect

| metric | before | expected |
|---|---|---|
| `cross_encoder_reranking` span | ~976 ms | **~200-300 ms** |
| pairs scored per query | 20 | 10 |
| reranker model | MiniLM-L-6 (6 layers) | TinyBERT-L-2 (2 layers) |
| first-query cold start | multi-second model load | none (loaded at startup) |
| `embedding_generation_for_search` span | ~229 ms | ~229 ms (unchanged) |

Combined: retrieval before LLM drops from **~1.2 s** to **~0.5 s**.

## Tradeoffs

- **TinyBERT-L-2 ranks slightly worse than MiniLM-L-6** on MS-MARCO benchmarks (MAP ~0.79 vs
  ~0.82). On a RAG workload the practical difference is usually small because the candidate set
  is only 10-20 chunks, but if answer quality regresses, set `RERANK_MODEL` back to MiniLM-L-6
  and keep the `VECTOR_SEARCH_LIMIT=10` win — that alone halves rerank time.
- **`VECTOR_SEARCH_LIMIT=10` gives the reranker fewer candidates.** If the right chunk is at
  rank 11-20 in pure vector order, reranking cannot rescue it. On small corpora (<10k chunks)
  this is unlikely to matter; on large corpora, raise to 15 and re-measure.
- **Startup time increases by the reranker load** (1-3 s, downloaded weights cached in the
  `hf_cache` volume). Acceptable: the alternative is the first user paying it inline.

## Risks

- **Model download on first container start.** TinyBERT-L-2 will be fetched from HuggingFace the
  first time. The existing `hf_cache` volume
  ([docker-compose.yml](../../docker-compose.yml)) caches it across restarts. If the container has
  no network, bake the model into the image alongside `all-MiniLM-L6-v2` (the Dockerfile already
  pre-warms the embedder — extend that step).
- **`app` memory limit 768M.** Loading the cross-encoder at startup adds resident memory *before*
  the first request rather than during it; peak RSS does not change but the floor rises. Watch
  `docker stats rag_app` for OOM-kill. If it approaches 768M, bump to 896M — the comment at
  [:256-259](../../docker-compose.yml#L256-L259) already documents the 392M baseline.
- **BM25 is still O(n) per query.** Not addressed here. If `search_mode="hybrid"` is enabled
  later, the BM25 rebuild will dominate and this plan's gains will be masked. Track separately.
- **`RERANK_MODEL` was not previously passed through compose.** Anyone relying on a `.env`
  override today is silently getting the default. Step 4 fixes this as a side effect — call it
  out in the commit message.

## Not recommended (in this plan)

- **ONNX / quantized reranker.** Real further win (1.5-2×), but adds `optimum` + `onnxruntime`
  dependencies and a model-conversion step. Phase 2, gated on Step 1-4 not being enough.
- **Raising `app` `limits.cpus` 1.0 → 2.0.** Helps rerank linearly but the embedder and reranker
  contend for the same core on a warm request, and the Docker VM only has 6 cores shared with
  parse workers. Try the model + limit-10 win first.
- **Replacing the cross-encoder with a bi-encoder re-rank or ColBERT.** Order-of-magnitude
  faster but a different retrieval architecture; out of scope for a tuning plan.

## Verification

```bash
# 1. Rebuild and start the app service
docker compose up -d --build app

# 2. Confirm the reranker loaded at startup (should see "Reranker initialized"
#    in the app log BEFORE any /query is sent)
docker compose logs app | grep -E "Reranker initialized|Application startup complete"

# 3. Send the same query as the trace and capture the new spans
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"<same query>","limit":5,"table_name":"test"}'

docker compose logs rag_app | grep -E "document_search|cross_encoder|Vector search completed|Reranking completed|Context built|LLM request"
```

Compare the new span deltas against the baseline trace above. Pass criteria:

| span | baseline | target |
|---|---|---|
| `embedding_generation_for_search` + vector search | ~229 ms | ~229 ms (±50) |
| `cross_encoder_reranking` | ~976 ms | **< 350 ms** |
| first-query model-load gap | multi-second | **0** (loaded at startup) |

Correctness — retrieval quality must not collapse:

```bash
# Same query, compare top-5 chunk_ids and similarity scores against the
# pre-change run. Order may shift (that is the point of a different reranker)
# but the same chunks should mostly still appear in the top 5.
```

If `cross_encoder_reranking` is still > 500 ms, set `RERANK_MODEL` back to MiniLM-L-6 and
isolate whether the win came from the model or from `VECTOR_SEARCH_LIMIT=10` — single-variable,
same lesson as `docs/plans/20260812_parse_time_reduction.md` Step 4.
