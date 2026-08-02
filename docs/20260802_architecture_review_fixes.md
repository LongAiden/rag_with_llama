# 20260802 Architecture Review — Findings and Fixes

This document summarizes a design/architecture review of the post-refactor repo and the
fixes applied. The review was static: every non-archived module under `api/`,
`ingestion/`, `retrieval/`, `infra/`, `worker/`, `config/`, and `migrations/` was read
and each finding traced to a concrete failing path.

Two findings from an earlier review were confirmed already fixed and are not repeated
here: the `sources_html | safe` XSS in the search results template, and the wrong patch
path in `tests/unit/test_embedding_generator.py`.

---

## P0 — Every successful query returned HTTP 500

**Problem**: `retrieval/search.py` called
`generate_llm_response(query, context, results, config.agent, model=model)`. `config` is
the `AppConfig` instance built in `api/app.py`, and `AppConfig.__init__` never defines an
`agent` attribute — the name was referenced at exactly one line in the codebase and
assigned nowhere.

Every query that retrieved at least one chunk raised
`AttributeError: 'AppConfig' object has no attribute 'agent'`. The no-results branch
returns early *before* the LLM call, so searches that matched nothing still succeeded.
That asymmetry made the failure look intermittent rather than total.

It was invisible because both callers swallow the exception — `/query` turns it into a
generic `500 Query failed: ...` and `/query-form` renders `search_error.html`.

The argument was dead on arrival anyway: `generate_llm_response` accepted `agent` and
never used it.

**Solution**:
- Dropped the argument from the call and the `agent` parameter from the signature.
- Removed the same unused `_agent` parameter from `OllamaBackend.generate` and
  `GeminiBackend.generate`, plus the unused `GeminiBackend._agent` attribute.
- Added `tests/unit/test_document_search.py`, the first coverage of
  `perform_document_search`. The fixture uses a `SimpleNamespace` built from
  `AppConfig`'s real attribute set rather than a `MagicMock`, so any stray
  `config.<missing>` raises instead of silently auto-creating an attribute.

**Verification**: temporarily reintroducing `config.agent` fails 5 of the 6 new tests;
the no-results test passes in both states, matching the diagnosis exactly.

**Files Changed**: `retrieval/search.py`, `retrieval/llm_operations.py`,
`tests/unit/test_document_search.py`

---

## P1 — Resource lifetime

### 1. Connection leak on every database error

**Problem**: The previous refactor added `VectorStore.connection()`, a correct
`@asynccontextmanager` — but no method in the class used it. Six methods
(`add_chunks`, `search_similar_chunks`, `search_bm25`, `get_chunks_by_section`,
`delete_document_chunks`, `get_collection_stats`) acquired a connection manually and
released it as the *last statement*, with no `try/finally`. Every `except` block
re-raised without releasing.

`ConnectionPoolManager` caps the pool at `max_size=10`. Ten failed queries — one bad
table name, one malformed embedding, one transient network blip — permanently exhaust
the pool, after which the API hangs forever on `pool.acquire()` with no timeout. Only a
restart recovers it.

**Solution**: all six methods now use `async with self.connection() as conn:`, and
`connection()` itself was rewritten to use `async with pool.acquire()` rather than a
manual acquire/release pair. That form also returns the connection on task
cancellation, not just on exceptions.

**Files Changed**: `ingestion/embedding/vector_store.py`

### 2. Two subsystems bypassed the pool entirely

**Problem**: `infra/telemetry/llm_logger.py` opened a raw `asyncpg.connect()` for every
logged interaction (one per query), and `api/routes/observability_routes.py` did the
same on every request to all three of its endpoints. Each call paid a full
TCP + TLS + auth handshake, and neither got the json/jsonb codecs that
`ConnectionPoolManager` registers via its `init` hook.

**Solution**: both now borrow from `ConnectionPoolManager`. The observability router
also replaced its module-global monkey-patch (`_obs_routes._connection_string = ...`
assigned from `app.py`) with an explicit `set_connection_string()` call, and dropped the
unused `Depends` import and the `get_connection_string` dependency that was never wired
to anything.

**Files Changed**: `infra/telemetry/llm_logger.py`,
`api/routes/observability_routes.py`, `api/app.py`

### 3. Fire-and-forget task could be garbage collected mid-write

**Problem**: `asyncio.create_task(log_interaction(...))` kept no reference to the task.
asyncio holds only a weak reference to a running task, so it could be collected before
the insert completed, silently dropping interaction records.

**Solution**: tasks are held in a module-level `_BACKGROUND_TASKS` set and discarded via
`add_done_callback`.

**Files Changed**: `retrieval/search.py`

---

## P2 — The API was effectively single-threaded

**Problem**: every CPU-bound and blocking call in the query path ran directly on the
event loop:

| Call | Location |
|---|---|
| `model.generate_content(prompt)` — **synchronous** Gemini SDK call | `retrieval/llm_operations.py` |
| `SentenceTransformer(...)` — loads model from disk | `ingestion/embedding/generator.py` |
| `CrossEncoder(...)` — loads model from disk | `retrieval/reranking.py` |
| `model.encode(...)` — inference | `ingestion/embedding/generator.py` |
| `reranker.rerank(...)` — inference | `retrieval/search.py` |
| `BM25Okapi(corpus)` + `get_scores` | `ingestion/embedding/vector_store.py` |

`generate_content` was the worst: it blocks the whole process for the full LLM latency,
so concurrent requests serialized completely. `OllamaBackend` already did this correctly
with `httpx.AsyncClient`; `GeminiBackend` did not.

**Solution**: each is wrapped in `await asyncio.to_thread(...)`. `get_reranker` also
gained a `threading.Lock` so two concurrent first-calls do not each load their own copy
of the cross-encoder.

**Files Changed**: `retrieval/llm_operations.py`, `retrieval/search.py`,
`retrieval/utils.py`, `ingestion/embedding/pipeline.py`,
`ingestion/embedding/vector_store.py`

### The single-slot pipeline cache made this reachable per request

**Problem**: `get_pipeline` in `api/app.py` was a one-entry cache keyed on table name,
stored on shared global state:

```python
if config.pipeline is None or config.pipeline.vector_store.table_name != table_name:
    config.pipeline = ChunkEmbeddingPipeline(...)   # loads SentenceTransformer
```

Because `table_name` is caller-supplied, alternating it between two values on successive
`/query` calls forced a full blocking model load on *every* request.
`worker/ingestion_tasks.py` already solved this correctly with a `_PIPELINES` dict.

**Solution**: replaced with a per-table dict behind an `asyncio.Lock` (double-checked),
with construction moved to a worker thread. Added `forget_pipeline(table_name)` so a
deleted table's pipeline is evicted rather than left cached.

**Files Changed**: `api/dependencies.py` (new), `api/app.py`

---

## P2 — Document lifecycle

### 4. `DELETE /table/{name}` orphaned status rows

**Problem**: the endpoint truncated and dropped the chunk table but never touched
`documents`. Those rows kept `stage='embedded'` and a `target_table_name` pointing at a
table that no longer existed.

Combined with the filename unique constraint (below), re-uploading the same file was
answered `status="duplicate"` and refused. The vectors were gone and the file could not
be re-added without manual SQL.

**Solution**: `delete_table` now also deletes the `documents` rows for that
`target_table_name` via a new `IngestionRepository.delete_documents_for_table` (the
artifact tables clean up through their existing `ON DELETE CASCADE`), and evicts the
cached pipeline. The redundant `TRUNCATE` before `DROP` was removed — it only bought an
extra exclusive lock.

**Files Changed**: `api/routes/table_routes.py`, `infra/db/ingestion_repository.py`

### 5. Filename de-duplication removed

**Problem**: `documents.file_name` was `NOT NULL UNIQUE` (migration 003) and
`register_document` deduped on it with `ON CONFLICT (file_name) DO NOTHING`. The
filename was therefore the de-duplication key for the entire system, which meant the
same file could never be ingested into a second chunk table, and could never be
re-ingested after its table was dropped.

**Solution**: de-duplication by filename was removed entirely — uploads are now always
registered as new documents.

- `migrations/005_drop_filename_dedupe.sql` drops the unique constraint and replaces it
  with a plain index (filename lookups are still used for reporting).
- `register_document` has no `ON CONFLICT` clause and always returns the row it created.
- The `status="duplicate"` branch was removed from the upload route.
- `is_file_registered` was deleted. The directory sweep now dedupes on
  `raw_storage_path` only — that check exists to avoid reprocessing the same file on
  disk on every sweep, which is a different concern from content de-duplication.

> **Action required**: migrations are mounted at `/docker-entrypoint-initdb.d`, which
> Postgres only runs on an empty data directory. On an existing volume apply this by hand:
> ```bash
> docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
>   < migrations/005_drop_filename_dedupe.sql
> ```

**Files Changed**: `migrations/005_drop_filename_dedupe.sql` (new),
`infra/db/ingestion_repository.py`, `api/routes/document_routes.py`,
`worker/ingestion_tasks.py`

### 6. Full page text stored twice per chunk

**Problem**: chunk metadata contained both `page_content` and `full_content` holding the
*identical* value. A page split into 10 chunks stored 20 copies of that page's text in
JSONB, and both keys rode along in every API response via `RAGSource.metadata`. Only
`page_content` was ever read.

**Solution**: `full_content` dropped from chunk metadata.

**Files Changed**: `ingestion/embedding/pipeline.py`

### 7. Chunk post-processing was O(n²) in document length

**Problem**: for every chunk, the code sliced `parsed_text[:chunk.start_index]` and
re-ran two regexes over that prefix — once for page markers, once for heading hierarchy.
On a large PDF with thousands of chunks this is quadratic over a multi-megabyte string.

**Solution**: replaced with a `_MarkdownStructure` class that scans the document once and
answers per-chunk queries with a binary search (`bisect_right`).

**Verification**: differential-tested against the old implementation at every character
offset of a representative document — **0 differences** in page number and 0 in section
path at any offset outside heading text. The only divergences are positions strictly
*inside* a heading line, where the old code returned a truncated fragment such as
`[Cha]`; the new code is correct there. Measured ~250x faster on the same input.

**Files Changed**: `ingestion/embedding/pipeline.py`

---

## P3 — Consistency

### 8. Table name validated on some routes but not others

`validate_table_name` was called in the document and table routes but not in either
query route. Injection was still blocked downstream by `quote_ident`, so this was **not**
an exploitable SQL injection — but it surfaced as a 500 instead of a 400, and only after
`get_pipeline` had already constructed a pipeline and loaded a model.

Separately, `vector_store.py` interpolated the **raw** table name into `CREATE INDEX`
names rather than the quoted form. That was unreachable only because the `CREATE TABLE`
above it evaluated `safe_table_name` first and raised; any reordering would have turned
it into real injection.

**Solution**: both query routes validate before `get_pipeline`, and the index names are
now derived from an explicitly validated table name.

**Files Changed**: `api/routes/query_routes.py`, `ingestion/embedding/vector_store.py`

### 9. Two configuration systems that disagreed

**Problem**: `AppSettings` (pydantic-settings, which reads `.env`) declared
`input_raw_dir`, `ingestion_max_attempts`, `ingestion_claim_timeout_minutes`,
`default_chunk_size`, `default_parse_backend` and `ollama_base_url` — but the code that
needed them called bare `os.getenv` at import time instead. `os.getenv` does not read
`.env`, so **anything set only in `.env` silently did not apply** to the worker or those
modules.

The defaults had also drifted: `OLLAMA_BASE_URL` defaulted to `http://localhost:11434`
in `app_config.py` but `http://host.docker.internal:11434` in `llm_operations.py`. The
latter matches `docker-compose.yml` and `.env.example`; `localhost` would point at the
container itself.

**Solution**: all of the above now read through `AppSettings`. `rerank_model`,
`celery_upload_queue` and `celery_ingestion_queue` were added to `AppSettings`, and
`table_name` gained its `DEFAULT_TABLE_NAME` alias. The `ollama_base_url` default was
reconciled to `host.docker.internal`.

**Files Changed**: `config/app_config.py`, `worker/ingestion_tasks.py`,
`api/routes/document_routes.py`, `retrieval/utils.py`, `retrieval/llm_operations.py`

### 10. Every route was defined twice, and only one router was mounted

**Problem**: `api/app.py` mounted only `observability_routes.router`. The other four
route modules decorated their handlers with `@router.get(...)` / `@router.post(...)` —
and those routers were never included. Instead `app.py` re-declared all 13 routes as
wrapper functions that passed `config` and `get_pipeline` positionally.

Consequences:
- Adding a route to a route module the obvious way produced a 404.
- Handler signatures carried `config=None, get_pipeline=None`, which FastAPI would have
  interpreted as query parameters if a router were ever mounted.
- Two DI styles coexisted: wrapper injection for four modules, module-global
  monkey-patching for the fifth.

**Solution**: introduced `api/dependencies.py` holding the shared `config`, the pipeline
cache, and `Depends()` providers. It lives outside `app.py` so route modules can import
it without a circular import. All five routers are now mounted directly and `app.py` is
wiring only — 264 lines down to 53.

**Verification**: the route table was captured before and after — **all 19 routes
identical** (path and method), and the generated OpenAPI schema confirms no
`config` / `get_pipeline` / `forget_pipeline` parameter leaked into any endpoint's
public signature.

**Files Changed**: `api/dependencies.py` (new), `api/app.py`, `api/__init__.py`,
`api/routes/document_routes.py`, `api/routes/query_routes.py`,
`api/routes/table_routes.py`, `api/routes/admin_routes.py`

### 11. Dead code with tests that gave a false coverage signal

Removed, along with the tests that were their only callers:

- `retrieval/utils.py::rerank_bm25` — the live path uses
  `vector_store.search_bm25` + `merge_with_rrf`.
- `api/validators.py::celery_enabled` / `celery_upload_enabled` /
  `entity_extraction_enabled` — all three env flags were unread by production code.
- `config/app_config.py::graph_pool` — never read; the graph feature is archived.

**Files Changed**: `retrieval/utils.py`, `api/validators.py`, `config/app_config.py`,
`tests/unit/test_retrieval_utils.py`, `tests/unit/test_validators_extra.py` (deleted)

### 12. Smaller items

- `require_access_password` used `!=`, a non-constant-time comparison. Now
  `secrets.compare_digest`.
- `CREATE EXTENSION IF NOT EXISTS vector` ran on **every** connection acquire — a round
  trip per query, requiring elevated privileges each time. Moved into
  `_initialize_database`, which runs once.
- `TRUNCATE` immediately before `DROP` in `delete_table` removed.

---

## Test status

| | Before | After |
|---|---|---|
| Passing | 293 | 290 |
| Failing | 22 (+19 collection errors) | 12 |

The 12 remaining failures are **all pre-existing** and unrelated to this work:

- **5 in `test_chunker_factory.py`** — file untouched by these changes.
- **6 in `test_delete_table_security.py`** — a `MagicMock` recursing inside FastAPI's
  `jsonable_encoder` during response serialization. Confirmed pre-existing by reverting
  only the changed files and re-running: identical 6 failed / 8 passed.
- **1 in `test_app_config.py`** — asserts the Ollama default is `deepseek-r1:8b` while
  `config/app_config.py` defaults to `deepseek-r1:1.5b`. The test is also
  environment-dependent: `patch.dict(os.environ, {}, clear=True)` does not stop
  pydantic-settings from reading a `.env` file.

Three tests were updated because they asserted behavior deliberately changed here:
the register-duplicate test (dedupe removed), `test_is_file_registered` (method removed),
and `test_add_chunks_executes_insert` — which asserted exactly one acquire, whereas
`add_chunks` legitimately borrows twice on a cold store. It now asserts acquires and
releases *balance*, which is the invariant that actually catches a leak.

The pre-existing failures were left in place at the user's direction and remain
outstanding.

---

## Net change

24 files, +608 / −908 lines.
