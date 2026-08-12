# Project Architecture — LLM Onboarding Document

**Read this first.** This document gives any LLM or human contributor the context needed to understand the repository before diving into specific modules, tests, or docs. It explains the system's purpose, the high-level data flow, the directory layout, the critical design decisions that shape the code, and the known rough edges.

For build / run instructions, see [`README.md`](../README.md). For detailed operational guides, see the `docs/` folder. For design decisions behind specific refactors, see [`docs/20260802_project_refactoring.md`](./20260802_project_refactoring.md), [`docs/20260802_architecture_review_fixes.md`](./20260802_architecture_review_fixes.md), and [`docs/20260802_ingestion_pipeline_fixes.md`](./20260802_ingestion_pipeline_fixes.md).

---

## 1. What This Project Is

A **Retrieval-Augmented Generation (RAG)** pipeline served by a **FastAPI** web UI and API. It ingests documents (PDF, DOCX, TXT), stores them as vector embeddings in **PostgreSQL + pgvector**, then answers questions using a hybrid of vector similarity search, BM25 lexical search, optional cross-encoder reranking, and a final LLM answer.

LLM backends are pluggable:

- **Gemini** (`google-generativeai` SDK) — cloud, requires `GOOGLE_API_KEY`.
- **Ollama** (`/api/generate` HTTP endpoint) — local, no API key required.

The default embedding model is **all-MiniLM-L6-v2** (384-dim). The default reranker is **cross-encoder/ms-marco-MiniLM-L-6-v2**.

### Core data flow

```
Raw file  →  documents status DB  →  parse  →  chunk  →  embed  →  pgvector  →  query + rerank  →  LLM answer
data/input/raw/   one row per file         chunker    embedding    chunk table   BM25 / RRF / cross-encoder   Gemini / Ollama
                (stage-based, claim/retry)     │          │
                                        data/parsed/  data/chunks/<doc>/
                                        (inspectable artifacts, gitignored)
```

---

## 2. Directory Layout

All application code lives under `src/app/` and imports as `app.*` — e.g.
`from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline`. `pytest.ini`
puts `src/` on the path locally; the Docker images set `PYTHONPATH=/app/src`.

| Directory | Purpose |
|---|---|
| `src/app/api/` | FastAPI app, routes, request validation, dependencies, HTML templates |
| `src/app/config/` | Centralized configuration (`AppSettings` via pydantic-settings, `AppConfig`) |
| `src/app/infra/` | Shared infrastructure: asyncpg connection pool, repositories, telemetry |
| `src/app/ingestion/` | Document ingestion: processors, chunkers, embedding, text cleaning, artifacts |
| `src/app/graph/` | Knowledge graph feature — present but **not wired into the app** |
| `src/app/models/` | Pydantic request/response schemas |
| `src/app/retrieval/` | Search, reranking, and LLM answer generation |
| `src/app/worker/` | Celery app and stage-based ingestion tasks |
| `data/input/raw/` | Original uploaded / scanned files (gitignored) |
| `data/parsed/` | Markdown written by the parse stage (gitignored, regenerable) |
| `data/chunks/` | One folder per document written by the chunk stage (gitignored) |
| `data/temp_uploads/` | Temporary upload staging area (gitignored, regenerable) |
| `deploy/migrations/` | SQL schema files applied by Postgres on first volume creation |
| `deploy/deployment/` | Dockerfiles, `requirements.txt`, Makefile |
| `tests/` | `unit/` (no DB) and `integration/` (requires Postgres); `htmlcov/` for coverage reports |
| `docs/` | Architecture decisions, design notes, and runbooks |
| `experiments/` | Scratch notebooks and scripts, reference only |

A more detailed file map is in the [`README.md` project structure section](../README.md#project-structure).

---

## 3. High-Level Architecture

### 3.1 Three main pipelines

1. **Ingestion pipeline** (async, background, Celery): parse → chunk → embed.
2. **Query pipeline** (API-synchronous): vector search + BM25 + RRF + rerank + LLM.
3. **Observability pipeline** (fire-and-forget): log every query/answer to `llm_interactions`.

### 3.2 Services and queues

| Service | Role | Queue / port |
|---|---|---|
| `app` | FastAPI web UI + API | `8000` |
| `postgres` | PostgreSQL + pgvector | `5432` |
| `redis` | Celery broker/backend | `6379` |
| `celery_worker_upload` | Single worker for interactive uploads | `upload` |
| `celery_worker_ingestion` | Scalable workers for batch ingestion | `ingestion` |
| `celery_beat` | Scheduler (weekly scan, 6h recovery) | — |
| `langfuse` (optional) | LLM observability UI | `3000` |
| `pgadmin` (optional) | DB admin UI | `5050` |

### 3.3 Why two Celery queues?

API uploads go to the `upload` queue (single worker) so a large weekly batch scan on the `ingestion` queue cannot starve interactive uploads. Batch work and recovery use the `ingestion` queue, which can be scaled to multiple workers.

---

## 4. Ingestion Pipeline — Stage-Based with Status DB

### 4.1 Status database (`documents`)

Every input file gets exactly one row in `documents`.

| Column | Meaning |
|---|---|
| `id` | UUID string used as `document_id` for chunks |
| `file_name` | Display filename (no longer unique, see migration 005) |
| `doc_name` | Human-readable document name from upload; defaults to the filename stem — added in migration 006 |
| `domain` | Domain this document belongs to; FK to `domains(name)` — added in migration 006 |
| `raw_storage_path` | Absolute path to the file in `data/input/raw/` |
| `stage` | `registered` → `parsing` → `parsed` → `chunking` → `chunked` → `embedding` → `embedded` / `error` / `failed` |
| `attempts` | Number of failures so far |
| `claimed_at` / `claimed_by` | Worker lease for coordination |
| `target_table_name` | Which chunk table to write to |
| `chunk_size` | Chunk size for this document |
| `parse_backend` | `ollama` or `gemini-docling` |
| `file_type` | Lowercased extension (`pdf`, `docx`, `txt`) — added in migration 004 |
| `error_stage` | Stage that failed, used for retry resumption |
| `parsed_id` / `chunked_id` | FKs to intermediate artifact tables |

### 4.2 Intermediate artifact tables

- `document_parsed` — parsed text, parser metadata, page mapping.
- `document_chunked` — serialized chunk objects before embedding.

Both are unique on `document_id` (migration 004) so retries overwrite rather than append.

### 4.2c Domains (`domains`)

A domain is a named bucket of documents backed **1:1 by one pgvector chunk table**.
`domains` makes explicit what a chunk table name has always implied.

| Column | Meaning |
|---|---|
| `name` | Slug and API key. CHECK-constrained to a safe SQL identifier because it is also the table name |
| `display_name` | Human-readable name shown in the UI |
| `description` | Optional free text |
| `table_name` | Physical chunk table. Equals `name` today; separate so a rename need not move the table |

One domain = one table = many documents = many chunks. `doc_name` is **denormalized
onto every chunk row** so the search `SELECT` returns it without joining `documents`;
`documents.doc_name` is authoritative and a rename must update both or they drift.
`document_id` stays the filter key — two uploads of the same book share a `doc_name`
but not an id. See `docs/plans/20260812_domains_and_doc_name.md`.

### 4.2b On-disk artifacts (`data/`)

Postgres is the source of truth, but JSONB is a poor way to eyeball a bad chunk
boundary, so the parse and chunk stages also dump their output to disk:

```
data/parsed/<document_id>_<name>.md        markdown from the parse stage
data/chunks/<document_id>_<name>/
    0000.md, 0001.md, ...                  one file per chunk, text only
    index.json                             per-chunk page/section/token metadata
```

Written by `src/app/ingestion/artifacts.py`, called from
`ChunkEmbeddingPipeline.parse_file` and `chunk_parsed_document`. Rules that make
this safe to leave on:

- `data/` is gitignored and fully regenerable — deleting it loses nothing.
- Writes never raise. An `OSError` is logged and skipped, because a debugging aid
  must not fail an ingestion that otherwise succeeded.
- `document_id` is allowlisted (`[A-Za-z0-9_-]{1,64}`) before it becomes a path
  component, and filenames are stripped to their basename and sanitised.
- A re-chunk clears the document's folder first, so a retry producing fewer
  chunks cannot leave a stale tail behind that reads as real output.
- `index.json` deliberately omits `full_content` — it holds the chunk's entire
  source page, so every chunk on a page would repeat that page into the index.

Set `PERSIST_INGESTION_ARTIFACTS=false` to disable, or point `PARSED_DIR` /
`CHUNKS_DIR` elsewhere.

### 4.3 Stage flow

```
registered → [parse] → parsed → [chunk] → chunked → [embed] → embedded
  ↑                    │                    │                    │
  └──── retry on error ┘                    │                    │
       resume from last completed artifact ─┴────────────────────┘
```

### 4.4 Key files

- `src/app/worker/ingestion_tasks.py` — Celery task definitions and the stage runner (`_run_stage`).
- `src/app/infra/db/ingestion_repository.py` — atomic DB operations (`claim_document`, `transition_to_*`, `record_error`, etc.).
- `src/app/ingestion/embedding/pipeline.py` — `ChunkEmbeddingPipeline` with static `parse_file` and `chunk_parsed_document`.
- `src/app/ingestion/embedding/vector_store.py` — `VectorStore` (pgvector CRUD + BM25).
- `src/app/ingestion/artifacts.py` — on-disk parse/chunk dumps under `data/`.
- `src/app/ingestion/processors/` — per-file-type parsers and PDF backend factory.
- `src/app/ingestion/chunking/chunker_factory.py` — chonkie-based chunker factory.

### 4.5 Worker persistence caveats

Two things are load-bearing in the worker:

1. **Immutable Celery signatures** (`.si()`): a plain `.s()` chain would pass each task's return value as the next task's first argument, but the task signature is `(doc_id,)`.
2. **One persistent event loop per worker process**: `asyncio.run()` per task closes the loop while the module-level `ConnectionPoolManager` cache still holds connections bound to it, causing the next task to fail with `RuntimeError: Event loop is closed`.

See `src/app/worker/ingestion_tasks.py:_run()` and `build_ingestion_chain()`.

---

## 5. Query Pipeline

### 5.1 End-to-end flow

1. `POST /query` (or `POST /query-form`) receives `query`, `table_name`, `limit`, `threshold`, `model`, etc.
2. `src/app/api/routes/query_routes.py` validates the table name, then calls `get_pipeline(table_name)` to retrieve or create a `ChunkEmbeddingPipeline`.
3. `src/app/retrieval/search.py:perform_document_search()` does the work:
   - Generate query embedding (SentenceTransformer, in worker thread).
   - Vector search (`pgvector` cosine similarity, `HYBRID_LIMIT = 20`).
   - BM25 lexical search over the same table.
   - RRF merge of the two result lists.
   - Optional cross-encoder reranking (top `rerank_top_k`, default 5).
   - **Sibling expansion** for structural queries (e.g., "how many", "list all") — fetches all chunks sharing the same `section_path` to reconstruct split sections.
   - Build context with page numbers and optional full page context.
   - Call the LLM backend.
   - Fire-and-forget log the interaction to `llm_interactions`.
4. Return `RAGResponse` (JSON) or HTML rendered results.

### 5.2 Key files

- `src/app/retrieval/search.py` — orchestration and context building.
- `src/app/retrieval/llm_operations.py` — `GeminiBackend` and `OllamaBackend`.
- `src/app/retrieval/reranking.py` — `Reranker` (cross-encoder) and `HybridScorer`.
- `src/app/retrieval/utils.py` — `merge_with_rrf()` and `get_reranker()`.
- `src/app/api/dependencies.py` — per-table pipeline cache and dependency providers.

### 5.3 Pipeline caching

`ChunkEmbeddingPipeline` loads a `SentenceTransformer` model, which is slow and CPU-intensive. Two caches avoid reloading per request:

- **API process**: `src/app/api/dependencies.py` keeps a per-table `_PIPELINES` dict under an `asyncio.Lock`.
- **Worker processes**: `src/app/worker/ingestion_tasks.py` keeps `_PIPELINES` per table per process.

When a table is deleted (`DELETE /table/{name}`), `forget_pipeline()` evicts the cache entry.

### 5.4 LLM backend selection

`src/app/retrieval/llm_operations.py:_get_backend(model)` returns `GeminiBackend` for model names starting with `gemini-`, otherwise `OllamaBackend`. The Gemini SDK call is synchronous and is wrapped in `asyncio.to_thread()` to avoid blocking the event loop.

---

## 6. Database Layer

### 6.1 Connection pool

`src/app/infra/db/pool.py:ConnectionPoolManager` is a singleton keyed by connection string. It creates one `asyncpg` pool per unique connection string and registers json/jsonb codecs on every connection so Python `dict` and `list` values round-trip natively.

### 6.2 Repositories

| Repository | File | Responsibility |
|---|---|---|
| `IngestionRepository` | `src/app/infra/db/ingestion_repository.py` | `documents`, `document_parsed`, `document_chunked` CRUD, claims, retries, status |
| `TableRepository` | `src/app/infra/db/table_repository.py` | List/count chunk tables, drop tables, per-table stats |
| `DomainRepository` | `src/app/infra/db/domain_repository.py` | `domains` registry CRUD, documents per domain, name→id resolution. `list_domains(reconcile=True)` also registers chunk tables that have no registry row |
| `VectorStore` | `src/app/ingestion/embedding/vector_store.py` | Per-table chunk CRUD, similarity search, BM25, delete by document |

### 6.3 Safe identifiers

`src/app/infra/db/identifiers.py` validates and quotes table names. Only `[a-zA-Z_][a-zA-Z0-9_]{0,62}` is allowed. SQL identifiers are double-quoted before interpolation. All user-supplied table names must pass `validate_table_name()` before they reach `VectorStore` or `TableRepository`.

`validate_table_name()` also rejects the reserved application tables (`domains`,
`documents`, `document_parsed`, `document_chunked`, `llm_interactions`, and the graph
tables). Before migration 006 that `_SYSTEM_TABLES` frozenset was declared but never
read, so an upload with `table_name=documents` reached `VectorStore._initialize_database()`,
whose `CREATE TABLE IF NOT EXISTS` silently matched the status table and then failed
on INSERT with a column error. Domain names go through the same check.

### 6.4 Migrations

SQL files in `deploy/migrations/` are mounted to `/docker-entrypoint-initdb.d` and run by Postgres **only when the data volume is empty**. Existing volumes require manual application. `deploy/migrations/optional/` is a subdirectory, which initdb skips — it holds the graph schema, which is not applied (see 9.3).

| Migration | Purpose |
|---|---|
| `002_create_llm_interactions.sql` | Query/answer logging table |
| `003_create_ingestion_status.sql` | `documents`, `document_parsed`, `document_chunked` |
| `004_ingestion_fixes.sql` | Adds `file_type`, `error_stage`, unique artifact indexes |
| `005_drop_filename_dedupe.sql` | Removes `file_name` UNIQUE constraint; uploads always create new rows |
| `006_domains_and_doc_name.sql` | `domains` registry; `documents.doc_name`/`domain`; `doc_name` + index on every existing chunk table; backfills one domain per chunk table |

`006` deliberately does **not** backfill `doc_name` onto existing chunk *rows* — that
is an `UPDATE ... FROM documents` per table which rewrites every row. Pre-006 chunks
return `doc_name: null` and the UI falls back to the id prefix, which is the pre-006
behaviour. The opt-in command is in the migration's trailing comment. Chunk tables
created before 006 also self-heal on first use: `VectorStore._initialize_database()`
runs `ADD COLUMN IF NOT EXISTS doc_name`.

---

## 7. Configuration

Configuration is centralized in `src/app/config/app_config.py`.

- `AppSettings` — pydantic-settings class that reads from `.env`.
- `DatabaseConfig` — Postgres connection parameters.
- `AppConfig` — global config object, lazy-loads pipeline and reranker, configures Logfire.

`src/app/api/dependencies.py` creates a module-level `config = AppConfig()` and exposes `get_config()`, `get_pipeline_factory()`, `get_forget_pipeline()` for FastAPI `Depends`.

Important rule: **all code must read environment variables through `AppSettings`**, not bare `os.getenv`, because pydantic-settings reads `.env` but `os.getenv` does not. Several bugs were fixed where `.env`-only values were silently ignored by the worker.

Key environment variables (see [`.env.example`](../.env.example) for the full list):

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (optional if only Ollama is used) |
| `POSTGRES_*` | Database credentials |
| `OLLAMA_BASE_URL` | Default `http://host.docker.internal:11434` (Docker uses host network) |
| `OLLAMA_MODEL` | Text model for Q&A |
| `OLLAMA_VLM_MODEL` | VLM model for PDF parsing (Ollama backend) |
| `GEMINI_MODEL` | Gemini model for Q&A / parsing |
| `PDF_PARSER_BACKEND` | `ollama` or `gemini-docling` |
| `CHUNKER_TYPE` | `markdown` (default), `recursive`, `token`, `semantic` |
| `INPUT_RAW_DIR` | Where raw files are stored (default `data/input/raw`) |
| `PARSED_DIR` | Where parse-stage markdown is dumped (default `data/parsed`) |
| `CHUNKS_DIR` | Where chunk-stage folders are dumped (default `data/chunks`) |
| `PERSIST_INGESTION_ARTIFACTS` | Set false to skip both dumps (default true) |
| `INGESTION_MAX_ATTEMPTS` | Max retries before `failed` |
| `INGESTION_CLAIM_TIMEOUT_MINUTES` | Stale claim timeout |
| `DEFAULT_CHUNK_SIZE` | Default chunk size |
| `DEFAULT_PARSE_BACKEND` | Default parser for uploads/scan |
| `APP_ACCESS_PASSWORD` | Optional password for web UI |
| `LOGFIRE_WRITE_TOKEN` | Optional Logfire monitoring token |
| `LANGFUSE_*` | Optional Langfuse observability |

---

## 8. API Routing

`src/app/api/app.py` is intentionally minimal: it mounts the routers and wires the observability connection string. Route definitions live in `src/app/api/routes/`.

| Router | File | Endpoints |
|---|---|---|
| `query_routes` | `src/app/api/routes/query_routes.py` | `GET /`, `POST /query`, `POST /query-form` |
| `document_routes` | `src/app/api/routes/document_routes.py` | `POST /upload`, `GET /documents/{id}/status`, `DELETE /documents/{id}`, `GET /supported-types` |
| `table_routes` | `src/app/api/routes/table_routes.py` | `GET /tables`, `GET /tables/count`, `DELETE /table/{name}` |
| `domain_routes` | `src/app/api/routes/domain_routes.py` | `GET /domains`, `POST /domains`, `GET /domains/{name}`, `GET /domains/{name}/documents`, `DELETE /domains/{name}` |
| `admin_routes` | `src/app/api/routes/admin_routes.py` | `GET /stats`, `GET /health` |
| `observability_routes` | `src/app/api/routes/observability_routes.py` | `GET /observability/stats`, `GET /observability/history`, `GET /observability/metrics` |

`DELETE /table/{name}` and `DELETE /domains/{name}` share
`src/app/api/routes/table_deletion.py:drop_chunk_table()` — dropping a chunk table is
three coupled steps (DROP, delete the `documents` rows pointing at it, evict the cached
pipeline), and only the first is obvious. `/tables` stays for backward compatibility;
`/domains` is additive.

Previously, routes were declared twice and only the wrapper versions were mounted. The refactor put all routing into the route modules and mounted them directly. New routes should be added with the standard FastAPI router decorators.

---

## 9. Important Design Patterns

### 9.1 Patterns used

- **Factory Method**: `processor_factory.py` selects processor by file extension; `pdf_parser_factory.py` selects PDF backend; `chunker_factory.py` selects chunker strategy.
- **Abstract Method**: `src/app/ingestion/processors/base_processor.py` defines the processor contract.
- **Repository**: `IngestionRepository`, `TableRepository`, `VectorStore` encapsulate SQL.
- **Lazy initialization**: pipeline and reranker are loaded on first use.
- **Singleton / process-scoped cache**: `ConnectionPoolManager`, `_PIPELINES` caches.
- **Dependency injection**: FastAPI `Depends` in `src/app/api/dependencies.py`.

### 9.2 Patterns to preserve

- **Never** use bare `os.getenv` for `.env` variables; use `AppSettings`.
- **Never** use `asyncio.run()` inside a Celery worker task; use the persistent loop in `_run()`.
- **Always** use the shared `ConnectionPoolManager` for DB access; do not open raw `asyncpg.connect()` per request.
- **Always** use `async with self.connection()` in `VectorStore` so connections are released on exceptions and cancellation.
- **Always** use immutable Celery signatures (`.si()`) for the ingestion chain.
- **Always** validate table names before they reach SQL construction.
- **Always** keep background tasks referenced until completion (see `_BACKGROUND_TASKS` in `src/app/retrieval/search.py`).

### 9.3 Present but unwired features

- **Knowledge graph**: the full implementation lives in the active tree — `src/app/graph/` (extractors, providers, graph service), `src/app/config/graph_config.py`, `src/app/models/graph_models.py`, `src/app/ingestion/extraction/`, and `src/app/api/routes/graph_routes.py`. It is **not reachable and not configured**:

  - `graph_routes.router` is never mounted in `api/app.py`, so no `/graph` endpoint exists.
  - No live module imports any graph module. Starting the API and the worker loads 53 `app.*` modules, none of them graph.
  - Its schema is parked in `deploy/migrations/optional/`, a subdirectory Postgres' initdb ignores, so a fresh volume comes up with no `entities` / `relationships` tables.
  - Its environment variables were removed from `docker-compose.yml` and `.env.example`; nothing outside `graph_config.py` reads them.

  `tests/unit/test_graph_not_wired.py` asserts all four properties, so re-integration cannot happen silently. To enable the feature: mount the router, apply `deploy/migrations/optional/001_create_graph_tables.sql`, restore the settings it needs, and delete that test file in the same commit.

  The `entities`/`relationships` names still appear in the safe-table denylists in `infra/db/identifiers.py` and `infra/db/table_repository.py`. Those are protective — they stop a user creating or dropping a chunk table under those names — and are kept deliberately.
- **Celery `rag` queue**: removed; default queue is now `ingestion`.

---

## 10. Known Limitations and Open Issues

### 10.1 Retrieval limitations

- **Structural queries** ("how many points in this section?") can fail because the chunker may split a section across chunk boundaries. Sibling expansion mitigates this but is not perfect.
- **Document-level headers** may be separated from their content during chunking and score below the similarity threshold.

### 10.2 Performance limitations

- `VectorStore.search_bm25()` rebuilds the BM25 index on every query by loading the entire chunk table. This is fine for POC scale but is O(n) per query.
- The default embedding dimension is hardcoded to 384 in the `CREATE TABLE` SQL (matches `all-MiniLM-L6-v2`). Changing models requires a matching migration.

### 10.3 Pre-existing test failures

Running `tests/unit` on `refactor/document-ingestion` gives **9 failures / 589
passed** (re-counted 2026-08-12), all pre-existing and unrelated to current work:

- `tests/unit/test_chunker_factory.py` (5, chonkie API drift — the local chonkie is newer than the pinned one)
- `tests/unit/test_pdf_parser_factory.py` (2, still asserts `min_image_short_px`, which the parser refactor renamed to `min_image_short_pt` and joined with `images_scale`/`tableformer_mode`)
- `tests/unit/test_delete_table_security.py` (1, `test_path_traversal_blocked` — MagicMock/jsonable_encoder issue)
- `tests/unit/test_llm_provider.py` (1, `test_generate_content_retries_on_error` — this file was archived with the graph feature and never ran until it was restored to `tests/unit/`)

This section previously claimed 12 on master and 14 on this branch. The
`test_delete_table_security.py` count in particular has drifted — 5 of its 6
documented failures now pass. Nobody recorded which change fixed them, so treat
the list above as the current baseline rather than a history.

### 10.4 UI / template debt

- `src/app/api/renderer.py` / `src/app/api/templates/` use Jinja2 for HTML pages.
- `src/app/api/routes/observability_routes.py` uses inline HTML strings (legacy) and should be migrated to templates.

### 10.5 Hardcoded values that should be configurable

- `chunk_overlap=50` in `base_processor.py` and `pipeline.py`
- `vector(384)` in `vector_store.py`
- `batch_size=32` (embed) and `100` (insert) in `vector_store.py`
- `chars_per_page=2500` in `docx_processor.py`
- `h1/h2/h3_min_height` in `gemini_docling_parser.py`
- `images_scale` and `min_image_px` in `gemini_docling_parser.py` — deliberately still
  constructor-only: they change output quality, not throughput, so they are not part
  of the tuning surface (§15.5)
- `max_file_size_mb=50` in `src/app/models/schemas.py`

Resolved: `num_threads`, `_DOCLING_PAGE_BATCH_SIZE` and `_VLM_CONCURRENCY` are now
`DOCLING_NUM_THREADS`, `DOCLING_PAGE_BATCH_SIZE` and `VLM_CONCURRENCY` (§15.5).

See `docs/20260802_project_refactoring.md` "Future Work" for the full list.

---

## 11. How to Navigate the Codebase

### 11.1 For a new feature

1. Read this file.
2. Decide whether the feature touches ingestion, query, or admin/observability.
3. Look at the relevant route module in `src/app/api/routes/`.
4. Trace into the domain layer: `src/app/ingestion/`, `src/app/retrieval/`, or `src/app/infra/db/`.
5. If it runs in background, add tasks in `src/app/worker/ingestion_tasks.py` and update `celery_app.py` schedules/queues.
6. Add tests in `tests/unit/` (fast) or `tests/integration/` (requires Postgres).
7. Update this document if the architecture changes.

### 11.2 For debugging

- Check `documents` table for file status, `last_error`, `claimed_at`, `claimed_by`.
- Check `docker compose logs celery_worker_upload celery_worker_ingestion` for worker errors.
- Check `llm_interactions` for query history and token/latency stats.
- Use `/health` and `/stats` endpoints or the UI for quick status.

### 11.3 For running tests

```bash
# All tests in Docker
make -f deploy/deployment/Makefile test-docker

# Unit tests only (no DB)
python -m pytest tests/unit -v --ignore=tests/unit/test_pdf_to_markdown.py

# Integration tests (requires Postgres)
python -m pytest tests/integration -v
```

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Status DB** | The `documents` table and its artifact tables (`document_parsed`, `document_chunked`). |
| **Chunk table** | A user-named pgvector table (default `document_chunks`) holding chunk rows. |
| **Domain** | A named bucket of documents, backed 1:1 by one chunk table and registered in `domains`. The user-facing name for what was previously just "the table you uploaded into". |
| **`doc_name`** | Human-readable document name. Authoritative on `documents`, denormalized onto every chunk row so search results are attributable without a join. A label, not a key — filters use `document_id`. |
| **Pipeline** | A `ChunkEmbeddingPipeline` instance: embedding generator + vector store for a specific table. |
| **VLM** | Vision-language model used for PDF parsing when the `ollama` backend is selected. |
| **RRF** | Reciprocal Rank Fusion, merges vector and BM25 rankings. |
| **Claim** | A row-level worker lease in the `documents` table to prevent two workers from processing the same document. |
| **Sibling expansion** | Fetching all chunks in the same section for structural queries. |

---

## 13. Recent Architectural Changes (as of 2026-08-02)

The repo underwent a major refactor and architecture review. Key outcomes:

- Introduced stage-based ingestion with status DB and intermediate artifact tables.
- Split Celery into `upload` and `ingestion` queues.
- Removed filename-based de-duplication; uploads always create new documents.
- Fixed connection-pool leaks, single-threaded event-loop blocking, and event-loop reuse in workers.
- Replaced module-level caches with per-process caches and locks.
- Routed all configuration through `AppSettings`.
- Mounted route modules directly instead of duplicating routes in `api/app.py`.
- Added sibling expansion for structural queries.
- Removed dead code and the unused Pydantic AI agent.
- Archived the knowledge graph feature.

See `docs/20260802_project_refactoring.md`, `docs/20260802_architecture_review_fixes.md`, and `docs/20260802_ingestion_pipeline_fixes.md` for the full engineering log.

---

---

## 14. Layout Move (2026-08-03)

- All application packages moved under `src/app/`; every import is now `app.*`. No
  compatibility shims were left at the old paths.
- Celery task names moved with them (`app.worker.ingestion_tasks.*`), and
  `celery_app.include` / the beat schedule were updated to match.
- The knowledge graph feature came out of `.archive/` into the active tree,
  still unmounted, with its migration moved to `migrations/optional/` and its
  environment variables removed from compose and `.env.example` (see 9.3).
- `deployment/` and `migrations/` moved under `deploy/`.
- `experiment/` renamed to `experiments/`.
- Added `data/parsed/` and `data/chunks/` artifacts (see 4.2b).
- `Dockerfile.test` now installs `requirements.txt`. It never did, so the whole
  `make test-docker` path failed at `import dotenv` before this.

---

## 15. Performance and Stability Improvements (2026-08-04)

### 15.1 Parser Memory Optimization

> **Corrected 2026-08-05.** The claim below that memory became "flat regardless of
> document size" was **not true as originally implemented**. Page-batching bounded
> docling's *working* memory but not its *retained* memory: `parse_pdf` converted
> every batch up front into a `batch_docs` dict and never released any of them, so
> all 10 `DoclingDocument`s of a 500-page PDF — and the rendered page images they
> carry — were alive simultaneously and peak memory still scaled with total pages.
> This is finding F4 in `docs/20260804_ingestion_performance_investigation.md`, and
> it is why the worker kept being OOM-killed after the batching landed.
>
> `parse_pdf` now **streams**: each batch is converted, assembled (including VLM)
> and released before the next `convert()` runs, which is what actually makes peak
> memory O(batch size). See §15.7.

Both PDF parsers (`GeminiDoclingParser` and `OllamaPDFParser`) now use **page-batched conversion** to prevent OOM kills on large documents:

- **Batch size**: 50 pages per `convert()` call (configurable via `DOCLING_PAGE_BATCH_SIZE`)
- **Memory reduction**: Peak memory usage drops from ~1.9 GB (504-page document) to ~1.15 GB, flat regardless of document size *(only true since the streaming rewrite — see the note above)*
- **Thread reduction**: `num_threads` reduced from 4 to 2 to match container CPU limits
- **Image scale**: Reduced from 1.0 to 0.75 (Gemini) and 0.75 to 0.6 (Ollama) to cut page-image memory by 44%. **Reverted and raised to 2.0 by F22** (`VLM_IMAGES_SCALE`): 0.6 renders at 43 DPI, at which the VLM cannot read a figure and confabulates instead — see `docs/20260811_tableformer_outlier_and_prompt_v2.md`. The memory saving was real but it was being paid for in fabricated descriptions entering the chunk index.
- **Minimum image size**: `min_image_px` increased from 0 to 150 to skip decorative icons and reduce VLM calls

**Key changes in `src/app/ingestion/processors/gemini_docling_parser.py`:**
- `_build_converter()`: `num_threads=2`, `images_scale` from `VLM_IMAGES_SCALE` (default 2.0)
- `parse_pdf()`: Converts in 50-page batches instead of whole document
- `_is_complex_table()`: Changed from OR to AND logic (both row AND column thresholds must be exceeded) to prevent narrow tall tables (e.g., Table of Contents) from triggering VLM calls
- `_process_page()`: Now accepts optional `ThreadPoolExecutor` for concurrent VLM dispatch

**Key changes in `src/app/ingestion/processors/ollama_pdf_parser.py`:**
- `images_scale` reduced to 0.6
- Removed redundant `_is_complex_table()` override (now in base class)
- `parse_pdf()`: Uses same page-batched conversion as Gemini parser

### 15.2 Worker Stability

**Docker Compose memory limits (`docker-compose.yml`):**

| Service | Old Limit | New Limit | Rationale |
|---|---|---|---|
| `celery_worker_ingestion` | 2.0G | **2.5G** | 2.2× headroom over projected 1.15 GB peak |
| `celery_worker_upload` | 3.0G | **2.5G** | Peaked at 95 MiB; 3G was unreachable |
| `postgres` | 1.5G | **1.0G** | Peaked at 44 MiB; 1.5G was unreachable |
| `celery_beat` | *(unbounded)* | **256M** | Now bounded |

**Task recycling**: Both workers now use `--max-tasks-per-child=1` to replace the fork child after each task, ensuring docling/torch memory returns to the OS and preventing memory stacking between parse and embed stages.

**Retry budget fix (`src/app/infra/db/ingestion_repository.py`):**
- `reset_stale_claims()` now increments `attempts` and moves documents to `failed` when `attempts >= max_attempts`
- Prevents infinite re-dispatch loops when workers are killed (OOM, container restart, etc.)
- Previously, SIGKILL bypassed the `except` block in `_run_stage()`, so `attempts` was never incremented and documents were re-dispatched forever

**Key changes in `src/app/worker/ingestion_tasks.py`:**
- `_recover_and_dispatch()` and `_register_and_dispatch()` now pass `MAX_ATTEMPTS` to `reset_stale_claims()`

### 15.3 Query Pipeline: Two Search Modes

The query pipeline now supports two selectable modes:

1. **Vector-only (default)**: Vector similarity search + cross-encoder reranking
   - Faster, suitable for most queries
   - Skips BM25 and RRF merge
   - `search_method` reported as `"vector"` or `"vector_crossencoder"`

2. **Hybrid**: Vector search + BM25 lexical search + RRF merge + cross-encoder reranking
   - Slower but more comprehensive
   - Useful for queries with specific keywords or exact phrases
   - `search_method` reported as `"hybrid_bm25_vector"` or `"hybrid_bm25_vector_crossencoder"`

**Key changes:**

- `src/app/models/schemas.py`: `QueryRequest` now includes `search_mode: Literal["vector", "hybrid"] = "vector"`
- `src/app/retrieval/search.py`: `perform_document_search()` accepts `search_mode` parameter and conditionally skips BM25/RRF
- `src/app/api/routes/query_routes.py`: Both `/query` and `/query-form` endpoints accept `search_mode`
- `src/app/api/templates/home.html`: Added search mode selector in settings panel (defaults to "Vector Only")

**UI changes:**
- Settings panel now includes a "Search Mode" dropdown with two options
- `sendChat()` JavaScript function includes `search_mode` in the request body

### 15.4 Expected Performance Improvements

| Metric | Before | After | Improvement |
|---|---|---|---|
| 504-page PDF ingestion | OOM kill at 2G | Completes at ~1.15G | Survives |
| Worker memory (idle) | 1.3G | ~300M | -77% |
| Query latency (vector mode) | 2-5s (BM25 scan) | 0.5-1.5s | -60-70% |
| VLM calls (decorative icons) | Every image | Only images ≥150px | -80-90% |
| Infinite retry loops | Possible | Prevented | Fixed |

### 15.5 Configuration

- `INGESTION_MAX_ATTEMPTS`: Default 2 (used by `reset_stale_claims()`)
- `INGESTION_CLAIM_TIMEOUT_MINUTES`: **180** in `docker-compose.yml` (`AppSettings` default is still 30). Must exceed the worst-case parse: a 500-page PDF runs ~60 minutes, and at 30 the 6-hourly recovery sweep declared live parses stale, re-dispatched a duplicate conversion, and consumed the retry budget of a document that never failed.
- `DOCLING_NUM_THREADS`: Default **2**. Docling's `AcceleratorOptions` thread count. Raising it does nothing unless the container's `cpus:` limit rises too, and vice versa.
- `DOCLING_PAGE_BATCH_SIZE`: Default **50**. Pages per `convert()` call; sets peak parse memory now that batches are released as they finish.
- `VLM_CONCURRENCY`: Default **1**. Parallel VLM calls *within a single parse*. With Ollama on a separate host these are network waits, so higher values can pay off; against a local Ollama they serialize on one GPU (§15.8). Note it does not bound concurrency **across** workers — `celery_worker_upload` and `celery_worker_ingestion` are separate processes and can each be parsing a document.
- `OLLAMA_VLM_TEMPERATURE` / `OLLAMA_VLM_NUM_PREDICT`: Default **0.0** / **384**. VLM latency is pure decode at ~35 tok/s, so elapsed *is* the output length, and Ollama's defaults leave it unbounded (§15.9).
- `VLM_MIN_IMAGE_SHORT_PX`: Default **64**. Skips pictures whose short side is under this. The older `min_image_px=150` rule needs *both* dimensions small, so it only caught square icons and let every thin strip through.
- `TORCHDYNAMO_DISABLE`: **1** on all four app-image services. The runtime image ships without a C++ compiler by design, so a downstream `torch.compile` path (torchvision NMS inside docling's layout pass) crashes with `InvalidCxxCompiler`. Nothing in this repo calls `torch.compile`, so forcing eager execution removes the crash at no cost. An env change needs `docker compose up -d --force-recreate <service>` — a plain `restart` will not pick it up.

The three parse-tuning defaults reproduce the values that were previously hardcoded, so exposing them changed no behaviour. They exist so the CPU/thread and VLM-concurrency experiments are `.env` changes rather than code edits.

### 15.6 Verification

After rebuilding containers:

```bash
# Reset stranded documents
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "UPDATE documents SET stage='registered', claimed_at=NULL, claimed_by=NULL, attempts=0 WHERE stage='parsing';"

# Check worker memory usage
docker stats --format "{{.Name}}\t{{.MemUsage}}"

# Verify no OOM kills
docker inspect rag_celery_worker_ingestion --format '{{.State.OOMKilled}}'
```

Test with the 504-page `NLTK.pdf` on both backends. Both should complete without OOM.

### 15.7 Streaming parse, instrumentation and the F-series fixes (2026-08-05)

Full diagnosis: **`docs/20260804_ingestion_performance_investigation.md`** (findings
F1–F13) and **`docs/20260804_stats_endpoint_keyerror_fix.md`**.

What landed:

- **Streaming `parse_pdf` (F4)** — one implementation, in `GeminiDoclingParser`. Each
  page batch is converted → grouped by absolute `prov[0].page_no` → assembled →
  released before the next `convert()`. `OllamaPDFParser.parse_pdf` was deleted;
  the subclass is now just `_call_vlm`, `get_backend_name` and its tuned defaults,
  so a fix can no longer be applied to one backend and not the other.
- **Instrumentation** — `_rss_mb()` (dependency-free, reads `/proc/self/status`),
  per-batch `elapsed / rate / rss`, per-page elapsed with cumulative `vlm_wait`, and
  a single `parse_pdf summary:` line carrying docling %, assembly %, VLM wait, call
  and failure counts, and peak RSS. VLM counters are guarded by a lock because they
  are written from `VLM_CONCURRENCY` pool threads. `_run_stage` times each stage;
  the parse is wrapped in a `logfire.span`.
- **`keep_alive: "30m"` on Ollama calls (F13)** — without it Ollama unloads the model
  5 minutes after last use, so gaps between figure-bearing pages made each call pay a
  cold model load. This is what the observed 27–33s per call on a 0.8B model was, and
  cumulative `vlm_wait` had reached 3593s by page 463 of 514.
- **VLM failures are attributable (F10)** — the failure path now logs elapsed time,
  exception type, and the HTTP status and body. The `[IMAGE]` fallback was previously
  silent, so a non-vision model would drop every figure with no visible error.
- **`asyncio.to_thread` for the parse (F7)** — the synchronous hour-long `parse_pdf`
  no longer blocks the worker's event loop and its asyncpg pool.
- **Dead code** — removed `_parse_page_by_page` (never called) and the first
  `_fix_markdown_headings`, which was shadowed by a second definition of the same name.

Deliberately **not** applied at that point, pending measurement: raising the container
CPU quota (F1/Fix A), raising `DOCLING_NUM_THREADS` (F2/Fix B), cross-page VLM
pipelining and `httpx.Client` connection pooling (F5/Fix D), and `OMP_NUM_THREADS`
for the embed stage (F9/Fix G). The tuning knobs were exposed at their previous
hardcoded values precisely so the baseline stayed uncontaminated and each became a
one-variable experiment — F12 is the cautionary example, where two simultaneous
changes measured on two different documents produced a 4× improvement nobody can
attribute. §15.8 is what those knobs were then set to, and why.

### 15.8 VLM reasoning, table routing and host resources (2026-08-05)

Full write-up: **`docs/20260805_vlm_thinking_and_table_routing.md`** (F14-F17),
measured on the Mac dev host, where — unlike the WSL2 investigation — **Ollama runs
locally** and shares the machine.

The headline is that the assemble cost was never docling and never the network:

- **`OLLAMA_VLM_THINK=false` (F14).** `qwen3.5:0.8b` is a reasoning model and Ollama
  defaults thinking on. Measured 85-87s per call versus 2-6s with it off, and the
  reasoning is discarded unread — `_call_vlm` reads only the `response` field, while
  Ollama returns reasoning in a separate `thinking` key. The request field is the API
  equivalent of `/set nothink`; appending `/no_think` to the prompt is a Qwen3
  convention with no effect on qwen3.5. This supersedes F13's cold-load explanation.
- **Blank responses are now failures (F14b).** With thinking on, every table call
  returned an empty body after ~3600 reasoning tokens — 3 of 3. `_process_page` wrapped
  that into an empty `<table>`, so tables vanished from the output silently. They now
  count in `vlm_failures` and fall back to `[IMAGE]`.
- **`VLM_TABLES=false` (F16).** A 0.8B VLM cannot do table OCR: on `bert.pdf` the
  13-column GLUE table came back with headers `I, II, III, IV…` and a small table came
  back with invented rows. The old rule sent a table to the VLM only when
  `_is_complex_table()` was true (>8 rows **and** >6 cols) — i.e. the hardest tables to
  the weakest extractor. All tables now go to docling's TableFormer, which the pipeline
  already used for every simple table. Figures still go to the VLM, where it does well
  (2.78s and an accurate description of the BERT architecture diagram).
- **`VLM_CONCURRENCY=1` (F15).** Against a local Ollama serializing on one GPU:
  3.87s/call at 1, 4.93s at 2, 20.62s at 4.
- **Host resources (F17).** The Docker Desktop CPU slider showed 6 but `docker info`
  reported 4 — that slider needs *Apply & Restart*, and no thread tuning means anything
  until it reads 6. CPU ceilings summed to 9.0 against 4 real CPUs, and `langfuse` was
  not profile-gated despite its own comment, so it started by default and held 1G of a
  6.77 GiB VM. Workers are now `cpus: "4.0"`, postgres and app `"1.0"`, `langfuse` is
  gated behind `profiles: [observability]`, and `DOCLING_NUM_THREADS` is 4.

Still unverified: docling's TableFormer output has not been compared side by side
against the VLM's, and the parse-side CPU changes have no measurement on this machine
yet. Both are listed in that document's "Still to verify".

### 15.9 Bounding VLM output length (2026-08-05, F18)

Full write-up: **`docs/20260805_vlm_output_length_and_image_gate.md`** (F18, F18b, F18c).
With thinking off, the first full run still showed
12–17s per call on 218×54px crops. Measured over that run's 191 calls: 2245s total,
mean 11.75s, range 1.04–93.3s, and **no correlation with image size**.

- **Latency is decode, not prefill or I/O.** Prefill is ~206 tokens in 0.26s whatever the
  crop — Qwen-VL pads small images to a fixed tile budget — and then `eval_duration ≈
  elapsed` at ~35 tok/s. Elapsed *is* the number of output tokens.
- **`OLLAMA_VLM_TEMPERATURE=0.0`, `OLLAMA_VLM_NUM_PREDICT=384`.** `_call_vlm` sent no
  `options`, so Ollama applied `temperature 0.8` / `num_predict -1`. On a 0.8B model
  given a near-empty crop that wanders into invented content: the same 218×54px equation
  strip returned 22 tokens in 1.55s and 342 tokens of fabricated flowchart in 10.94s on
  byte-identical consecutive requests. The 93.3s worst case is ~3200 tokens, i.e. the
  4096 context. Temperature is the lever; `num_predict` only caps the tail.
- **`VLM_MIN_IMAGE_SHORT_PX=64`.** The size gate skipped a picture only when *both*
  dimensions were under 150px, so full-column 40–60px-tall strips — rules, equation
  lines, header bands — all went to the VLM: 113 of the 191 calls, 60% of the VLM
  budget, producing hallucinated descriptions that then get embedded.
- **The `VLM call #` line now carries token counts**, derived `tok/s` and `done_reason`,
  and warns on `done=length`. F14 and F18 are both "too many output tokens" and the log
  carried only latency, which is why telling them apart needed an out-of-band experiment
  twice.

F18 is landed but **not unit-tested** — tests for the `options` payload, the
`done=length` warning and the short-side gate are owed, and the ~10× improvement is
predicted from the probe, not yet measured on a full document.

### 15.10 Structure preservation (2026-08-12, F23)

Full write-up: **`docs/20260812_structure_preservation.md`**. Plan:
`docs/plans/20260812_parser_structure_preservation.md`.

`_process_page` threw away most of docling's structural classification. `CodeItem`
and `ListItem` both subclass `TextItem`, so the `isinstance(item, TextItem)` branch
caught them first and emitted bare `item.text` — `item.label`, `item.marker`,
`item.code_language` and `item.captions` were never read. The VLM's `<figure>`
wrapper was never enforced in code. Measured on the 504-page NLTK artifact: 631
`>>>` lines with 0 fenced blocks, 0 `<figure>` wrappers surviving out of 79 VLM
calls, 0 `<figure_caption>` bound to parent, 3 list markers preserved out of
thousands.

What landed:

- **Label dispatch** — the `TextItem` branch dispatches on `getattr(item.label,
  "value", "")` rather than adding more `isinstance` checks. Immune to future
  `XItem(TextItem)` subclasses.
- **`_format_code(item)`** — fences code with `` ```lang ``, restores doctest
  line breaks by splitting on whitespace before `>>>` / `...` prompts, strips
  trailing bare prompts.
- **`_format_list_item(item)`** — uses `item.marker` when present, falls back to
  `1.` for enumerated, `-` otherwise.
- **`_wrap_figure(md, caption)`** — strips any `<figure>` tags the VLM did or did
  not emit, re-adds them deterministically with `<figure_caption>` inside. Applied
  on both sync and async VLM paths.
- **Caption binding** — pre-pass collects caption text via `caption_text(doc)`;
  loose `TextItem`s matching that text are skipped so captions appear only inside
  their parent block.
- **VLM task tuple** — carries `(col, y, x, future, kind, caption)`. The fragile
  `_item_sort_key` re-lookup that decided table-vs-figure wrapping by re-scanning
  all items is deleted.
- **Fence-aware post-passes** — `_fix_markdown_headings`,
  `_normalize_tables_in_markdown`, and `_strip_stray_headers` all track an
  `in_fence` flag. Content inside fenced code blocks passes through unchanged.

27 unit tests in `tests/unit/test_f23_structure_preservation.py`. All string work —
no expected change to parse time or peak RSS.

### 15.11 VLM/convert pipelining (2026-08-12)

Plan: `docs/plans/20260812_parse_speed_and_embed_refresh.md` (Part A); design:
`docs/plans/20260812_parse_pipelining.md`.

The parse stage's two phases never overlapped. On the 504-page NLTK.pdf: docling
`convert()` 749s (72%), VLM wait 281s (27%), non-VLM assembly ~5s. During the
749s of convert, Ollama is idle; during the 281s of VLM decode, the container's
threads are idle.

`_process_page` is now split, and the join is deferred by one batch:

```
before:  convert(N) → assemble(N) [VLM blocks] → release → convert(N+1) → …
after:   convert(N) → build(N) → release → convert(N+1) → emit(N) → build(N+1) → …
```

- **`_build_page`** — the item walk. Submits VLM futures and returns
  `(ordered, vlm_tasks)` without joining. Everything that reads `doc` happens
  here: crops are cut eagerly into independent PIL images and `ordered` holds
  only strings, so the returned payload carries no `DoclingDocument` reference.
- **`_finalize_page`** — joins the futures, wraps figures/tables, sorts, returns
  the `[PAGE:n]` block.
- **`_process_page`** — kept as a thin wrapper over both, so the synchronous
  `executor=None` path and its callers are unchanged.
- **`_emit_batch`** — runs `_finalize_page` plus the existing per-page
  normalization for one built batch, appends to `pages_md` and streams to
  `out_file`.

Invariants: `del doc, page_items` still happens before the emit, so only one
`DoclingDocument` is ever alive; pages come out in order, with `out_file`
streaming one batch behind; build and finalize have separate per-page
`try/except` so one bad page cannot take the batch down. The final batch's emit
stays *inside* the `with ThreadPoolExecutor` block — outside it, `shutdown(wait=True)`
would do the same waiting first with the log order scrambled.

**Depth stays 1 deliberately.** Per batch (50 pages, 11 batches) convert is ~68s
against ~26s of VLM, so one batch of lag already absorbs all of it; depth 2 would
buy nothing and hold a second batch of crops in RAM. The pipeline is
docling-bound and stays that way. The irreducible serial cost is the last batch's
VLM tail, which has no convert left to hide under.

**New metric `vlm_blocked`** in the `parse_pdf summary:` line. `vlm_wait`
(`_vlm_seconds`) is unchanged — total time inside VLM calls on the pool threads,
kept for historical comparability. `_vlm_blocked_seconds` is how much of that the
assembly loop actually waited for, and is the number that proves the overlap:
it collapsed from an implicit 281s to a **measured 0s**.

The wait is timed **outside** `_vlm_stats_lock`. `_record_vlm_call` takes that
same lock from the pool thread, so holding it across `future.result()` deadlocks
two ways — the main thread waiting on a future whose thread waits on the main
thread, and then a non-reentrant re-acquire. `TestFinalizePageJoin` pins this
with daemon threads, a bare `Future` and timed waits, so the regression fails in
5s instead of wedging the suite.

#### Measured (2026-08-12, 504-page NLTK.pdf)

Both runs are in `logs/celery_worker_upload.log` — baseline at 03:44:20, pipelined
at 17:19:25, same document, same knobs (`DOCLING_PAGE_BATCH_SIZE=50`,
`VLM_CONCURRENCY=1`).

| metric | baseline | pipelined | delta |
|---|---|---|---|
| `total` | 1035s | **808s** | **−227s (−22%)** |
| `assembly` | 286s (28%) | **3s (0%)** | −283s |
| `docling` | 749s (72%) | 804s (99%) | **+55s (+7%)** |
| `vlm_wait` | 281s | 333s | +52s |
| `vlm_blocked` | — (implicitly 281s) | **0s** | full overlap |
| `vlm_calls` / `vlm_failures` | 86 / 0 | 86 / 0 | unchanged |
| `peak_rss` | 2886 MB | 3097 MB | +211 MB |

**Output is byte-for-byte identical** — both artifacts md5
`db1845905fcb30cfd10c568757fdae04`, 504 pages strictly ascending, 86 `<figure>`,
67 `<table>`. That check is only this strong because `OLLAMA_VLM_TEMPERATURE=0.0`
(§15.9) makes the VLM deterministic; without it only page ordering and figure
placement would be comparable.

**The saving is real but it came from a different place than predicted, and cost
more than predicted.** `assembly` collapsing 286s → 3s is the whole win —
`vlm_blocked=0` means every VLM call had resolved before its emit, so not even
the final batch's tail was paid. But `docling` rose 55s, which the plan expected
to stay flat. Per-batch convert times show a uniform tax rather than an outlier:
batch 1 is *faster* (no prior batch's VLM work to overlap yet), the 4-page final
batch is unchanged (no VLM calls), and every batch that genuinely overlaps pays
4-9s. That shape is CPU contention. Ollama runs locally on this Mac and shares
the machine (§15.8), so VLM decode now competes with docling instead of following
it; `vlm_wait` rising 281→333s is the same effect seen from the other side.

Net: trade 283s of serial VLM for 55s of slower convert. **On a host where Ollama
is remote, the docling penalty should not appear at all** — the overlap would be
pure gain. Conversely, raising `VLM_CONCURRENCY` on this host would deepen the
contention, not relieve it.

The parse is now **99% docling**. Every remaining second is in `convert()`, so
further speedup has to come from docling itself (threads, accelerator, image
scale) — no amount of VLM tuning will move `total` again.

**Memory.** `peak_rss` 3097 MB against `celery_worker_upload`'s 3.5G limit is
**86% utilised**, ~487 MB of headroom, down from ~700 MB at baseline. The +211 MB
is the crop backlog: one batch's PIL crops now live across the next `convert()`.
The run completed with no OOM and NLTK.pdf is the heaviest document tested, so
this is recorded rather than acted on — but it is the number to watch if a larger
PDF appears. A rise of ~1 GB would mean a `DoclingDocument` is being retained
instead; `tests/unit/test_pdf_parser_streaming.py`'s weakref check guards that.

**Out-of-band cost is 2.9s** (`Stage parse completed … in 810.9s` against
`total=808s`) — torch re-import plus converter construction. The parse-time
reduction plan gated "relax `--max-tasks-per-child=1`" on this exceeding 60s. At
20× under, that option is ruled out rather than merely unmeasured.

Also in this change: the Embed tab's Domain select gained the `↻` refresh button
the Chat tab already had. `loadDomainList()` already repopulated both dropdowns,
so it is markup only.

---

**Last updated**: 2026-08-12
