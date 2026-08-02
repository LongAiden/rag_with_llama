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
input/raw/        one row per file         chunker    embedding    chunk table   BM25 / RRF / cross-encoder   Gemini / Ollama
                (stage-based, claim/retry)
```

---

## 2. Directory Layout

| Directory | Purpose |
|---|---|
| `api/` | FastAPI app, routes, request validation, dependencies, HTML templates |
| `config/` | Centralized configuration (`AppSettings` via pydantic-settings, `AppConfig`) |
| `infra/` | Shared infrastructure: asyncpg connection pool, repositories, telemetry |
| `ingestion/` | Document ingestion: processors, chunkers, embedding, text cleaning |
| `models/` | Pydantic request/response schemas |
| `retrieval/` | Search, reranking, and LLM answer generation |
| `worker/` | Celery app and stage-based ingestion tasks |
| `migrations/` | SQL schema files applied by Postgres on first volume creation |
| `deployment/` | Dockerfiles, `requirements.txt`, Makefile |
| `tests/` | `unit/` (no DB) and `integration/` (requires Postgres) |
| `docs/` | Architecture decisions, design notes, and runbooks |
| `.archive/` | Disabled features (knowledge graph) kept out of the active tree |

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
| `raw_storage_path` | Absolute path to the file in `input/raw/` |
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

### 4.3 Stage flow

```
registered → [parse] → parsed → [chunk] → chunked → [embed] → embedded
  ↑                    │                    │                    │
  └──── retry on error ┘                    │                    │
       resume from last completed artifact ─┴────────────────────┘
```

### 4.4 Key files

- `worker/ingestion_tasks.py` — Celery task definitions and the stage runner (`_run_stage`).
- `infra/db/ingestion_repository.py` — atomic DB operations (`claim_document`, `transition_to_*`, `record_error`, etc.).
- `ingestion/embedding/pipeline.py` — `ChunkEmbeddingPipeline` with static `parse_file` and `chunk_parsed_document`.
- `ingestion/embedding/vector_store.py` — `VectorStore` (pgvector CRUD + BM25).
- `ingestion/processors/` — per-file-type parsers and PDF backend factory.
- `ingestion/chunking/chunker_factory.py` — chonkie-based chunker factory.

### 4.5 Worker persistence caveats

Two things are load-bearing in the worker:

1. **Immutable Celery signatures** (`.si()`): a plain `.s()` chain would pass each task's return value as the next task's first argument, but the task signature is `(doc_id,)`.
2. **One persistent event loop per worker process**: `asyncio.run()` per task closes the loop while the module-level `ConnectionPoolManager` cache still holds connections bound to it, causing the next task to fail with `RuntimeError: Event loop is closed`.

See `worker/ingestion_tasks.py:_run()` and `build_ingestion_chain()`.

---

## 5. Query Pipeline

### 5.1 End-to-end flow

1. `POST /query` (or `POST /query-form`) receives `query`, `table_name`, `limit`, `threshold`, `model`, etc.
2. `api/routes/query_routes.py` validates the table name, then calls `get_pipeline(table_name)` to retrieve or create a `ChunkEmbeddingPipeline`.
3. `retrieval/search.py:perform_document_search()` does the work:
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

- `retrieval/search.py` — orchestration and context building.
- `retrieval/llm_operations.py` — `GeminiBackend` and `OllamaBackend`.
- `retrieval/reranking.py` — `Reranker` (cross-encoder) and `HybridScorer`.
- `retrieval/utils.py` — `merge_with_rrf()` and `get_reranker()`.
- `api/dependencies.py` — per-table pipeline cache and dependency providers.

### 5.3 Pipeline caching

`ChunkEmbeddingPipeline` loads a `SentenceTransformer` model, which is slow and CPU-intensive. Two caches avoid reloading per request:

- **API process**: `api/dependencies.py` keeps a per-table `_PIPELINES` dict under an `asyncio.Lock`.
- **Worker processes**: `worker/ingestion_tasks.py` keeps `_PIPELINES` per table per process.

When a table is deleted (`DELETE /table/{name}`), `forget_pipeline()` evicts the cache entry.

### 5.4 LLM backend selection

`retrieval/llm_operations.py:_get_backend(model)` returns `GeminiBackend` for model names starting with `gemini-`, otherwise `OllamaBackend`. The Gemini SDK call is synchronous and is wrapped in `asyncio.to_thread()` to avoid blocking the event loop.

---

## 6. Database Layer

### 6.1 Connection pool

`infra/db/pool.py:ConnectionPoolManager` is a singleton keyed by connection string. It creates one `asyncpg` pool per unique connection string and registers json/jsonb codecs on every connection so Python `dict` and `list` values round-trip natively.

### 6.2 Repositories

| Repository | File | Responsibility |
|---|---|---|
| `IngestionRepository` | `infra/db/ingestion_repository.py` | `documents`, `document_parsed`, `document_chunked` CRUD, claims, retries, status |
| `TableRepository` | `infra/db/table_repository.py` | List/count chunk tables, drop tables, per-table stats |
| `VectorStore` | `ingestion/embedding/vector_store.py` | Per-table chunk CRUD, similarity search, BM25, delete by document |

### 6.3 Safe identifiers

`infra/db/identifiers.py` validates and quotes table names. Only `[a-zA-Z_][a-zA-Z0-9_]{0,62}` is allowed. SQL identifiers are double-quoted before interpolation. All user-supplied table names must pass `validate_table_name()` before they reach `VectorStore` or `TableRepository`.

### 6.4 Migrations

SQL files in `migrations/` are mounted to `/docker-entrypoint-initdb.d` and run by Postgres **only when the data volume is empty**. Existing volumes require manual application.

| Migration | Purpose |
|---|---|
| `001_create_graph_tables.sql` | Graph tables (disabled); PageRank function commented out |
| `002_create_llm_interactions.sql` | Query/answer logging table |
| `003_create_ingestion_status.sql` | `documents`, `document_parsed`, `document_chunked` |
| `004_ingestion_fixes.sql` | Adds `file_type`, `error_stage`, unique artifact indexes |
| `005_drop_filename_dedupe.sql` | Removes `file_name` UNIQUE constraint; uploads always create new rows |

---

## 7. Configuration

Configuration is centralized in `config/app_config.py`.

- `AppSettings` — pydantic-settings class that reads from `.env`.
- `DatabaseConfig` — Postgres connection parameters.
- `AppConfig` — global config object, lazy-loads pipeline and reranker, configures Logfire.

`api/dependencies.py` creates a module-level `config = AppConfig()` and exposes `get_config()`, `get_pipeline_factory()`, `get_forget_pipeline()` for FastAPI `Depends`.

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
| `INPUT_RAW_DIR` | Where raw files are stored |
| `INGESTION_MAX_ATTEMPTS` | Max retries before `failed` |
| `INGESTION_CLAIM_TIMEOUT_MINUTES` | Stale claim timeout |
| `DEFAULT_CHUNK_SIZE` | Default chunk size |
| `DEFAULT_PARSE_BACKEND` | Default parser for uploads/scan |
| `APP_ACCESS_PASSWORD` | Optional password for web UI |
| `LOGFIRE_WRITE_TOKEN` | Optional Logfire monitoring token |
| `LANGFUSE_*` | Optional Langfuse observability |

---

## 8. API Routing

`api/app.py` is intentionally minimal: it mounts the routers and wires the observability connection string. Route definitions live in `api/routes/`.

| Router | File | Endpoints |
|---|---|---|
| `query_routes` | `api/routes/query_routes.py` | `GET /`, `POST /query`, `POST /query-form` |
| `document_routes` | `api/routes/document_routes.py` | `POST /upload`, `GET /documents/{id}/status`, `DELETE /documents/{id}`, `GET /supported-types` |
| `table_routes` | `api/routes/table_routes.py` | `GET /tables`, `GET /tables/count`, `DELETE /table/{name}` |
| `admin_routes` | `api/routes/admin_routes.py` | `GET /stats`, `GET /health` |
| `observability_routes` | `api/routes/observability_routes.py` | `GET /observability/stats`, `GET /observability/history`, `GET /observability/metrics` |

Previously, routes were declared twice and only the wrapper versions were mounted. The refactor put all routing into the route modules and mounted them directly. New routes should be added with the standard FastAPI router decorators.

---

## 9. Important Design Patterns

### 9.1 Patterns used

- **Factory Method**: `processor_factory.py` selects processor by file extension; `pdf_parser_factory.py` selects PDF backend; `chunker_factory.py` selects chunker strategy.
- **Abstract Method**: `ingestion/processors/base_processor.py` defines the processor contract.
- **Repository**: `IngestionRepository`, `TableRepository`, `VectorStore` encapsulate SQL.
- **Lazy initialization**: pipeline and reranker are loaded on first use.
- **Singleton / process-scoped cache**: `ConnectionPoolManager`, `_PIPELINES` caches.
- **Dependency injection**: FastAPI `Depends` in `api/dependencies.py`.

### 9.2 Patterns to preserve

- **Never** use bare `os.getenv` for `.env` variables; use `AppSettings`.
- **Never** use `asyncio.run()` inside a Celery worker task; use the persistent loop in `_run()`.
- **Always** use the shared `ConnectionPoolManager` for DB access; do not open raw `asyncpg.connect()` per request.
- **Always** use `async with self.connection()` in `VectorStore` so connections are released on exceptions and cancellation.
- **Always** use immutable Celery signatures (`.si()`) for the ingestion chain.
- **Always** validate table names before they reach SQL construction.
- **Always** keep background tasks referenced until completion (see `_BACKGROUND_TASKS` in `retrieval/search.py`).

### 9.3 Disabled / archived features

- **Knowledge graph**: full implementation exists in `.archive/graph_feature/` but is not wired into the active app. The `entities`/`relationships` system tables are referenced in safe-table guards but unused. The migration `001_create_graph_tables.sql` has a commented-out PageRank function.
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

At the time of the last refactor there were ~12 failures in the suite, all pre-existing and unrelated to the recent fixes:

- `tests/unit/test_chunker_factory.py` (5)
- `tests/unit/test_delete_table_security.py` (6, MagicMock/jsonable_encoder issue)
- `tests/unit/test_app_config.py` (1, default model mismatch / environment-dependent)

Additionally, `tests/unit/test_pdf_to_markdown.py` does not import because it targets a module deleted in an earlier refactor.

### 10.4 UI / template debt

- `api/renderer.py` / `api/templates/` use Jinja2 for HTML pages.
- `api/routes/observability_routes.py` uses inline HTML strings (legacy) and should be migrated to templates.

### 10.5 Hardcoded values that should be configurable

- `chunk_overlap=50` in `base_processor.py` and `pipeline.py`
- `vector(384)` in `vector_store.py`
- `batch_size=32` (embed) and `100` (insert) in `vector_store.py`
- `chars_per_page=2500` in `docx_processor.py`
- `h1/h2/h3_min_height` in `gemini_docling_parser.py`
- `max_file_size_mb=50` in `models/schemas.py`

See `docs/20260802_project_refactoring.md` "Future Work" for the full list.

---

## 11. How to Navigate the Codebase

### 11.1 For a new feature

1. Read this file.
2. Decide whether the feature touches ingestion, query, or admin/observability.
3. Look at the relevant route module in `api/routes/`.
4. Trace into the domain layer: `ingestion/`, `retrieval/`, or `infra/db/`.
5. If it runs in background, add tasks in `worker/ingestion_tasks.py` and update `celery_app.py` schedules/queues.
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
make -f deployment/Makefile test-docker

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

**Last updated**: 2026-08-02
