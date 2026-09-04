# Changelogs

Consolidated design plans, performance investigations, and refactoring records.
Newest first. Original files lived in `docs/plans/` and `docs/archive/`.

---

# Part I — Plans

---

## 2026-08-18 — Retrieval Speed: Vector Search + Cross-Encoder Rerank

**Status**: Phases 0-3 implemented. Phase 4 gated on measurement.
**Scope**: Synchronous `/query` path on `app` container.

### Diagnosis

Retrieval before LLM call: ~1.2s. Cross-encoder reranking: ~976ms (80%). Root cause: torch thread pool unbounded against 1.0-CPU quota (45ms/pair vs expected 12-15ms), uncapped sequence length (512 tokens), hardcoded HYBRID_LIMIT=20 scoring all pairs before slicing to top_k=5.

### Solution (4 phases, free levers first)

- **Phase 0** (env only): Cap `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to 1 on `app` service only (not workers). Target: <500ms rerank.
- **Phase 1** (config + 1 line): `RERANK_MAX_LENGTH=256` truncation via `rerank_max_length` in AppSettings, threaded through `Reranker.__init__` → `CrossEncoder(max_length=)`. Target: <350ms.
- **Phase 2** (preload): Bake CrossEncoder in Dockerfile alongside embedder. Add `preload_reranker()` eager wrapper in `utils.py`. Wire into FastAPI `lifespan` with try/except (never fail startup).
- **Phase 3** (observability): Make `vector_search_limit=20`, `rerank_top_k=5`, `preload_reranker=True` configurable in AppSettings. Replace hardcoded `HYBRID_LIMIT` with `candidate_depth = max(vector_search_limit, rerank_top_k)`. Fix 3 hardcoded `rerank_top_k` literals in `query_routes.py`. Fix live bug: `RERANK_MODEL` never reached container via `x-common-env`.
- **Phase 4** (gated, quality-costing): `VECTOR_SEARCH_LIMIT=10` then `RERANK_MODEL=…TinyBERT-L-2-v2` — one at a time, with measurement between.

### Key files
- `docker-compose.yml` — thread caps on app, retrieval env vars
- `src/app/config/app_config.py` — `rerank_max_length`, `vector_search_limit`, `rerank_top_k`, `preload_reranker`
- `src/app/retrieval/reranking.py` — `max_length` parameter
- `src/app/retrieval/utils.py` — `preload_reranker()` wrapper
- `src/app/api/app.py` — lifespan with error handling
- `src/app/retrieval/search.py` — `candidate_depth`, knob logging
- `src/app/api/routes/query_routes.py` — 3× `rerank_top_k` literals fixed

### v2 corrections over v1
- Added Step 0 (thread caps) — the likely root cause v1 missed
- Added Step 1 (max_length) — ~2× win keeping every candidate scored
- Demoted model swap and limit reduction to gated Step 4
- Fixed: `app` does not mount `hf_cache` — baking into Dockerfile is required
- Fixed: lifespan had no error handling — would cause boot loop
- Fixed model sizes: TinyBERT-L-2-v2 ~17MB, MiniLM-L-6-v2 ~90MB (v1 had them reversed)

---

## 2026-08-16 — Math Symbol Preservation

**Status**: Planned, not implemented.
**Scope**: Prompts, text cleaning pipeline, config.

### Problem
`∇f(x) ≤ ε for all x ∈ X` reaches the LLM as `[math] f(x) <= epsilon for all x [math] X`. Two causes:
1. `MathNotationNormalizer` destroys symbols at ingest time (replaces `∀ ∃ ∈ ∇` with `" [math] "`, deletes Math Alphanumeric Symbols, converts Greek to words)
2. RAG prompt says "Do not copy sentences verbatim" — formulas must be copied verbatim

### Solution: Unicode passthrough

**Prompt changes:**
- `VLM_IMAGE_PROMPT` (Gemini): Add rule to transcribe math expressions exactly, preserving every symbol
- `VLM_TABLE_PROMPT` (Gemini): Add symbol preservation rule + pipe escape rule
- `OLLAMA_IMAGE_PROMPT`: **No change** (0.8B model ignores added rules, costs output budget)
- `RAG_PROMPT_TEMPLATE`: Rename from `OLLAMA_RAG_PROMPT_TEMPLATE`, make single shared template for both backends. Replace blanket "don't copy" with narrowed rule + explicit formula carve-out. Delete duplicated inline Gemini prompt in `llm_operations.py`.

**Cleaner changes (`cleaners.py`):**
1. Drop `MathNotationNormalizer` from default chain, gate with `PRESERVE_MATH_NOTATION` env var (default true)
2. `UnicodeNormalizer('NFKC')` → `NFC` — NFKC turns `²` into `2` and `½` into `1⁄2`
3. Scope `TableStructurePreserver` pipe rewrite to table-row lines only (`^\s*\|.*\|\s*$`)
4. Trim `SpecialSymbolNormalizer` — remove currency (`€→EUR`) and fraction entries

### Key files
- `src/app/ingestion/processors/prompts.py` — VLM prompts + shared RAG template
- `src/app/retrieval/llm_operations.py` — delete duplicated Gemini prompt
- `src/app/ingestion/text_cleaning/cleaners.py` — default chain, NFC, pipe scoping
- `src/app/config/app_config.py` — `preserve_math_notation` setting
- `src/app/ingestion/embedding/pipeline.py` — pass setting to TextCleaningPipeline

---

## 2026-08-14 — Parse-Stage OOM on 702-Page PDF

**Status**: Implemented (Steps 1-4). 702-page PDF now parses inside 4G limit with flat RSS.
**Scope**: `gemini_docling_parser.py`, `ingestion_tasks.py`, `docker-compose.yml`.

### Root cause
`celery_worker_upload` SIGKILLed at 3.5G limit while parsing 702-page PDF. Two causes of memory growth:
1. **Per-batch working set**: 144 DPI page renders (~260MB per 50-page batch) held during `convert()`
2. **Allocator ratchet**: RSS steps up ~250-350MB per batch and never returns — glibc per-thread arenas + torch CPU caching allocator. Not a Python leak (weakref test proves no DoclingDocument survives).

Secondary finding: SIGKILLed task stuck for ~9 hours — `task_acks_late=True` + `task_reject_on_worker_lost=False` means no requeue, and `WorkerLostError` bypasses `_run_stage` except block.

### Solution (5 steps)
- **Step 0**: Unstick the document via SQL
- **Step 1**: `_PeakRssSampler` daemon thread sampling RSS every 0.5s during batch loop. Log RSS at 3 points per batch. Extend summary line with sampled `peak_rss`.
- **Step 2**: `_malloc_trim()` helper (ctypes, no-op on macOS). `gc.collect()` + `malloc_trim(0)` after `del doc`. `MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_=131072` in compose env.
- **Step 3**: Raise `celery_worker_upload` to 4G. `DOCLING_PAGE_BATCH_SIZE` 50→40 (~52MB off transient peak).
- **Step 4**: Celery `task_failure` signal handler for `WorkerLostError` → `record_error(error_stage='parse')`. Surfaces OOM kills in minutes instead of hours.
- **Step 5** (gated): If RSS still ratchets, set `generate_page_images=False` and render crops on demand via `pypdfium2`.

### Key files
- `src/app/ingestion/processors/gemini_docling_parser.py` — peak sampler, malloc_trim, RSS logging
- `src/app/worker/ingestion_tasks.py` — WorkerLostError signal handler
- `docker-compose.yml` — 4G limit, batch size 40, MALLOC_ARENA_MAX=2

---

## 2026-08-12 — Parser Structure Preservation (F23)

**Status**: Implemented. 504-page NLTK.pdf: 0→600+ fenced blocks, 0→86 figure wrappers, captions bound.
**Scope**: `gemini_docling_parser.py` assembler only.

### Problem
Docling classifies pages correctly but `_process_page` throws classification away:
- 631 `>>>` lines, 0 fenced code blocks (CodeItem subclasses TextItem, caught by `isinstance` first)
- 79 `<figure_type>` tags, 0 `<figure>` wrappers (0.8B model ignores wrapper rule)
- Captions float as loose paragraphs beside figures/tables
- 3 bullet lines in 12,378 lines (ListItem also subclasses TextItem)

### Solution
1. **Dispatch on `item.label`** not `isinstance` — immune to future `XItem(TextItem)` MRO issues
2. **`_format_code`** — fence blocks, restore doctest line breaks via `>>>` / `...` prompt splitting
3. **`_format_list_item`** — use `item.marker` from docling
4. **Bind captions** — emit inside figure/table block, suppress duplicate via `self_ref` set
5. **`_wrap_figure`** — deterministic `<figure>` wrapping in code (both VLM paths)
6. **Fence-aware post-passes** — add ``` toggle to `_fix_markdown_headings` and `_normalize_tables_in_markdown` to prevent silent corruption of fenced code

### Key files
- `src/app/ingestion/processors/gemini_docling_parser.py` — label dispatch, formatters, figure wrapping, fence-aware post-passes

---

## 2026-08-12 — Parse Time Reduction Plan

**Status**: Steps 4-5 applied. Steps 1-3 remain unmeasured/gated.
**Scope**: Diagnostic plan for reducing ~17min parse of 504-page PDF.

### Diagnosis
1045s total: docling 782s (75%), VLM wait 261s (24%), assembly 2s (0.2%).

**Why docling is 782s:**
1. M1 GPU idle — Docker has no MPS passthrough, layout/TableFormer run on CPU
2. 4 threads in 4.0-CPU quota (later raised to 6)
3. Index pages (451-500) are dense tables: 11.10 s/page vs 1.47 s/page healthy
4. All 504 pages rendered at 144 DPI; only 79 need it (+38s, +965MB RSS)
5. No parallelism above thread level

**Why VLM is near floor:** 79 calls at 3.3s/call vs 3.87s measured floor. Concurrency=1 because local Ollama serializes on one GPU.

### Steps (gated, one variable at a time)
- **Step 0**: Capture baseline (never done — out-of-band cost measured at 2.9s post-implementation)
- **Step 1**: Size MPS prize — benchmark native macOS vs Docker CPU (unmeasured)
- **Step 2**: Conditional native worker (if Step 1 shows MPS ≥2×)
- **Step 3**: Relax `--max-tasks-per-child=1` (ruled out — 2.9s out-of-band cost)
- **Step 4**: Threads 4→6 — **Applied, measured: docling 820s→749s (~9%), keep**
- **Step 5**: VLM/convert pipelining — promoted to `20260812_parse_pipelining.md`
- **Step 6**: Leave VLM alone (at measured optima)

---

## 2026-08-12 — Parse Speed (VLM Pipelining) + Embed Tab Refresh

**Status**: Implemented and measured. `total` 1035s → **808s** (−22%), `vlm_blocked=0s`, output byte-identical.
**Scope**: `gemini_docling_parser.py`, `home.html`.

### Part A — VLM/convert pipelining
**Core change**: Split `_process_page` into `_build_page` (submit VLM futures) and `_finalize_page` (join futures). Carry one pending batch across next `convert()`. Pipeline depth=1.

```
Current:   convert(N) → assemble(N) [VLM blocks] → release → convert(N+1)
Pipelined: convert(N) → build(N) → convert(N+1) → finalize(N) [VLM done] → build(N+1)
```

**Safety**: VLM futures hold independent PIL crops (no ref to `doc`), `del doc` before join preserves O(batch) memory, `_process_page` kept as thin wrapper for test compatibility.

**Measured results:**
| metric | before | after |
|---|---|---|
| total | 1035s | **808s** (−22%) |
| assembly | 286s | **3s** |
| vlm_blocked | (281s implicit) | **0s** |
| peak_rss | 2886MB | 3097MB |

Two misses (docling +55s, vlm_wait +52s) share one cause: local Ollama shares CPU with docling during overlap. Net still −227s.

**Correctness**: Output byte-for-byte identical (md5 match) — `OLLAMA_VLM_TEMPERATURE=0.0` makes VLM deterministic.

### Part B — Embed tab refresh button
Added `↻` button next to Domain dropdown in Embed tab, mirroring existing Chat tab pattern. One HTML addition, zero JS/backend changes.

### Ruled out
- ~~Step 3 (relax --max-tasks-per-child)~~ — out-of-band cost 2.9s, not worth it
- ~~Step 6 (httpx.Client reuse)~~ — vlm_blocked=0 means VLM time no longer reaches total

---

## 2026-08-12 — Parse Pipelining Design

**Status**: Design document. Implemented in `20260812_parse_speed_and_embed_refresh.md`.
**Scope**: `gemini_docling_parser.py`.

### Design
1. **Split `_process_page`** into `_build_page` (submit VLM futures, return `(ordered, vlm_tasks)`) and `_finalize_page` (join futures, return markdown). Keep `_process_page` as thin wrapper.
2. **Pending-batch loop**: convert(N) → build(N) → del doc → emit(N-1) → pending=N. Final batch is unavoidable serial tail.
3. **`vlm_blocked` metric**: Time `future.result()` in `_finalize_page`, accumulate after (not under lock — deadlocks against `_record_vlm_call`). Add to summary line.

**Invariants**: Output order preserved (lag by one batch), memory O(batch) (one DoclingDocument alive), VLM_CONCURRENCY=1 (serial queue), error isolation (separate try/except for build and finalize).

---

## 2026-08-12 — Domains and doc_name

**Status**: Implemented. Migration 006 needs manual application on existing volumes.
**Supersedes**: 2026-08-11 doc_name plan.
**Scope**: DB schema, ingestion, query, API, UI.

### Data model
- **`domains`** — registry table. One row per domain, 1:1 with a physical chunk table. Columns: `name` (PK, CHECK-constrained to safe SQL identifier), `display_name`, `description`, `table_name`.
- **`documents`** — gains `doc_name` (human-readable, defaults to filename stem) and `domain` (FK to domains.name).
- **Chunk tables** — gain `doc_name TEXT` column + index. Denormalized so search SELECT needs no join.

**One domain = one table = many documents = many chunks.**

### Key decisions
1. `doc_name` denormalized onto every chunk row (no join per query)
2. `document_id` stays the filter key; `doc_name` is a label (duplicate uploads share doc_name but not id)
3. `domains` added to `_SYSTEM_TABLES` denylist — latent bug where `table_name=documents` could collide with status table
4. Chunk-row backfill is opt-in (not in migration) — `UPDATE ... FROM documents` per table rewrites every row

### New endpoints
- `GET /domains` — list with document counts
- `POST /domains` — create
- `GET /domains/{name}` — one domain
- `GET /domains/{name}/documents` — list documents with stage
- `DELETE /domains/{name}` — drop table + status rows + registry row

### Key files
- `deploy/migrations/006_domains_and_doc_name.sql` — registry, columns, backfill, FK
- `src/app/infra/db/domain_repository.py` — DomainRepository
- `src/app/api/routes/domain_routes.py` — 5 endpoints
- `src/app/infra/db/identifiers.py` — `_SYSTEM_TABLES` made load-bearing
- `src/app/ingestion/embedding/vector_store.py` — doc_name in CREATE/ALTER/INSERT/search
- `src/app/retrieval/search.py` — preserve doc_name through rerank via `original_by_id`
- `src/app/api/templates/home.html` — domain picker, document filter, source rendering

---

## 2026-08-11 — TableFormer Outlier + Prompt v2 (F21)

**Status**: Complete. Total 1683s → **1045s**, index batch 555s → **121s**, filler 15→3.
**Scope**: Docling TableFormer config, Ollama VLM image prompt.

### Part A — TableFormer outlier
Pages 451-500 (NLTK index) took 555s — 42% of all docling time. Dense multi-column tables decoded by TableFormer ACCURATE mode at 11.10 s/page.

**Fix**: `DOCLING_TABLEFORMER_MODE=fast`. Measured: 540.8s → 129.7s (−76%), `tables=15` identical. Cell matching stays on (3.5% cost, effectively free).

Made configurable: `_DEFAULT_TABLEFORMER_MODE` constant, `tableformer_mode` constructor arg, `DOCLING_TABLEFORMER_MODE` env var on all 4 services. `_resolve_tableformer_mode()` maps string to docling enum with fallback.

### Part B — Prompt v2 → v3
Three prompt iterations measured:
- **v1** (F19): mean 160 tok, filler 15/79. Anti-filler rule left checklist in place → model echoed it.
- **v2**: Regressed — rule narrating answer shape was recited back as output.
- **v3**: mean 91 tok, filler 1/17 (6%). Deleted checklist, stated bound without describing answer shape, added grounding rule.

**Code fix**: `_strip_html_wrappers()` added to sanitizer chain — deterministic cleanup of format defects three prompt rounds couldn't fix at 0.8B. Unwraps `<p>`/`<div>`/`<span>`, preserves `<figure>` and `<table>`.

### Final result
| metric | F19 | F21 |
|---|---|---|
| total | 1683s | **1045s** (−38%) |
| pages 451-500 | 555s | **121s** (−78%) |
| mean output tok | 160 | **109** |
| filler clauses | 15 | **3** |
| tables | 67 | **67** (not lossy) |
| peak_rss | 2524MB | **2000MB** |

---

## 2026-08-11 — doc_name Column Plan (Superseded)

**Status**: Superseded by `20260812_domains_and_doc_name.md`.
**Scope**: Original plan for adding `doc_name` to chunk tables.

### What was right
- `doc_name` column on chunk tables + documents table
- Denormalization (no join per query)
- Migration via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
- Lazy schema evolution in `VectorStore._initialize_database()`

### What was corrected in the 08-12 revision
1. §3.7.2 assumed a search form in `home.html` — the chat panel is JS (`sendChat()` POSTs JSON)
2. `doc_name` string filtering replaced by `document_ids` (exact, already plumbed)
3. `_SYSTEM_TABLES` needed to become load-bearing (latent collision bug)
4. Chunk-row backfill made opt-in (not in migration)
5. Added `domains` registry (not in original plan)

---

## 2026-06-19 — UI Blue Accent Design

**Status**: Implemented.
**Scope**: CSS styling across HTML templates.

### Design tokens
- Primary blue: `#3b82f6`
- Secondary blue: `#0ea5e9`
- Chat area tint: `rgba(59, 130, 246, 0.12)`

### Changes per page
- **Main page**: Blue gradient top borders on `.hero-card` (3px) and `.section` (2px), blue left border on `h2`, blue tint on `#chat-messages`, blue active tab button
- **Health page**: Blue top borders on `.header`, `.component-card`, `.metrics-section`
- **Stats page**: Blue top borders on `.header`, `.stat-card`, `.config-section`, secondary blue for `.config-value`

---

---

# Part II — Archive (2026-06-19)

---

## 2026-06-19 — Refactoring Summary

**24 files modified, ~220 lines removed, 0 breaking changes.**

### Completed items
1. **SQL injection mitigation** — `TableRepository` with `validate_table_name()` + `quote_ident()`. Replaced all unsafe f-string SQL interpolation.
2. **Duplicated table query extraction** — Common `list_chunk_tables()` query consolidated from 3 places.
3. **Connection pooling** — `ConnectionPoolManager` with per-connection-string pools (min=2, max=10). Replaced per-operation `asyncpg.connect()`.
4. **Entity extraction caching** — `EntityCache` with SHA-256 content hashing and configurable TTL.
5. **Ollama as default graph LLM** — Local, fast, free. Gemini available as option.
6. **Fixed 19 `sys.path.insert` hacks** — `pythonpath = .` in `pytest.ini`, missing Dockerfile COPY commands added.
7. **Flattened TextCleanerFactory** — 4 factory methods → direct `TextCleaningPipeline()` (only `create_default_cleaner` was used).

### Kept intentionally
- `PDFParserBase` + factory (2 backends, clean polymorphism)
- `DocumentProcessor` ABC + registry (3 processors, used by `/supported-types`)
- `chunker_factory` (shared by PDF and non-PDF paths, caching + adaptive sizing)
- Celery worker (queue persistence, resource isolation, scalability)
- Graph processing module (~3,246 LOC, disabled but ready)

---

## 2026-06-19 — High Priority Refactoring Items

All completed as part of the refactoring summary above.

1. **SQL injection risk mitigation** — `TableRepository` with `validate_table_name()` and `quote_ident()`
2. **Duplicated table query logic extraction** — `list_chunk_tables()` consolidated
3. **Connection pooling** — `ConnectionPoolManager` singleton per connection string
4. **Entity extraction caching** — SHA-256 content hash, configurable TTL
5. **Graph processing: Ollama as default LLM provider** — local, fast, no rate limits

---

## 2026-06-19 — Medium Priority Refactoring Items

Recommended but not yet implemented:

1. **Break up large functions** — `ingest_document()` (~250 lines), `upload_and_process()` (~150 lines), `extract_from_chunks()` (~140 lines). Apply SRP, extract focused helpers.
2. **Proper dependency injection** — Use FastAPI `Depends()` system instead of global state and manual parameter passing.
3. **Rate limiting** — `slowapi` for API endpoints (10/min upload, 30/min query, 60/min stats).
4. **Centralize configuration** — Unified Pydantic settings hierarchy (`DatabaseSettings`, `LLMSettings`, `AppSettings`).
5. **Error handling consistency** — Centralized `AppError` exception with public/private messages.
6. **Context managers for DB connections** — `PoolConnection` async context manager.
7. **Standardize naming conventions** — snake_case in Python, camelCase in JSON responses via Pydantic aliases.

---

## 2026-06-19 — Low Priority Refactoring Items

Recommended but not yet implemented:

1. **Comprehensive tests** — Unit + integration test coverage, mock external services
2. **Type hints** — Return types, specific types over `Dict[str, Any]`
3. **Dead code removal** — Disabled graph routes, deprecated functions, unused imports (`autoflake`, `vulture`)
4. **Standardize logging** — Remove `print()`, use `logfire` consistently, structured logging
5. **API documentation** — OpenAPI/Swagger descriptions, request/response examples
6. **Optimize database queries** — Fix N+1 patterns, add indexes, `EXPLAIN ANALYZE`
7. **Performance monitoring** — Request timing middleware, query profiling
8. **Graceful shutdown** — Close connection pools, stop background tasks, clean resource cleanup
9. **Configuration validation** — Fail fast on missing env vars at startup
10. **Architecture Decision Records** — Document why patterns were chosen, trade-offs, constraints

---

## 2026-06-19 — Chunking Strategies

**Status**: Implemented

**Scope**: Document chunking strategies for the RAG pipeline.

**Details**:
- 4 chunker types: **MarkdownChunker** (default, structure-aware), **RecursiveChunker**, **TokenChunker**, **SemanticChunker**
- MarkdownChunker splits on headings, code blocks, paragraphs, lists, line breaks
- Configurable via `CHUNKER_TYPE` env var or `chunker_type` API parameter
- Performance benchmarks (15MB PDF): Token 5s, Recursive/Markdown 20s, Semantic 30min
- SemanticChunker auto-falls back to Recursive above 100KB

---

## 2026-06-19 — Docker Setup

**Status**: Implemented

**Scope**: Multi-service Docker Compose configuration.

**Details**:
- Services: postgres (pgvector), redis, app (FastAPI), celery_worker
- Optional profiles: langfuse (observability), pgadmin (dev), test (test profile)
- Critical fixes: Dockerfile.postgres COPY path, env var consistency
- Security: removed hardcoded credentials, required vars with `${VAR:?message}`
- Redis healthcheck, Dockerfile.test completeness

---

## 2026-06-19 — Project Architecture Summary

**Status**: Documented

**Scope**: Full architectural overview of the RAG system.

**Details**:
- **Ingestion**: PDF → Markdown (PyMuPDF / Docling+Ollama / Docling+Gemini) → Chunk → Embed → pgvector
- **Query**: vector search → BM25 rerank → LLM (Gemini 2.5 Flash or DeepSeek-R1 8B via Ollama)
- **Design patterns**: Abstract Method + Factory (processors), Factory (chunkers), Lazy Init (AppConfig)
- **Disabled**: knowledge graph, graph enrichment, graph API routes
- Module responsibilities table with key entry points documented

---

## 2026-06-19 — Testing Guide

**Status**: Implemented

**Scope**: Test structure and execution guide.

**Details**:
- Unit tests (no DB) and integration tests (PostgreSQL + pgvector required)
- Test structure: `unit/` (PDF, chunking, embedding, config), `integration/` (DB, pipeline, retrieval)
- Run: `pytest tests/unit -v`, `make test-docker`, coverage with `--cov`
- AAA pattern (Arrange-Act-Assert), pytest fixtures for PDF tests
- Integration tests require `GOOGLE_API_KEY` for LLM tests

---

## 2026-06-26 — Chunk Context Enrichment

**Status**: Implemented

**Scope**: Enriching chunks with structural and page-level context for better retrieval quality.

**Details**:
- Two enrichments: **section/sub-section prefix** (heading hierarchy) and **full page content**
- Section prefix: walks backwards through markdown to find H1→H2→H3 hierarchy, stored as `section_path`
- Sibling expansion: structural queries (`how many`, `list all`) retrieve all chunks in same section
- Full page content: extracted from `[Page N]` markers, deduplicated at query time per `(doc_id, page_number)`
- Final prompt format: `[Source N (Page P)] [Matched chunk] [Full page context]`
- **Pros**: better answer quality, accurate citations, reduced fragmentation
- **Cons**: larger prompts, metadata bloat, diminishing returns for small chunks

---

## 2026-08-02 — Architecture Review Fixes

**Status**: Implemented  
**Scope**: 24 files, +608/−908 lines. Critical bug fixes and architectural improvements.

**Key fixes**:
- **P0**: Every successful query returned HTTP 500 — `config.agent` attribute never defined. Dropped argument, added `test_document_search.py`
- **P1.1**: Connection leak on every DB error — 6 methods acquired connections without try/finally. All now use `async with self.connection()`
- **P1.2**: Two subsystems bypassed pool — `llm_logger.py` and `observability_routes.py` used raw `asyncpg.connect()`. Now borrow from pool
- **P1.3**: Fire-and-forget task GC'd — `asyncio.create_task` with no reference. Held in `_BACKGROUND_TASKS` set
- **P2**: API single-threaded — synchronous Gemini SDK call, model loading, inference all on event loop. Wrapped in `asyncio.to_thread()`
- **P2**: Pipeline cache keyed on caller-supplied table name — forced model reload on alternating queries. Replaced with per-table dict behind `asyncio.Lock`
- **P2.4**: `DELETE /table/{name}` orphaned status rows — now also deletes documents rows
- **P2.5**: Filename de-duplication removed — `documents.file_name` UNIQUE constraint dropped, uploads always registered as new
- **P2.6**: Full page text stored twice per chunk — `full_content` dropped from metadata
- **P2.7**: Chunk post-processing O(n²) — replaced with `_MarkdownStructure` class using `bisect_right`, ~250× faster
- **P3**: Table name validation missing on query routes, two config systems that disagreed, every route defined twice

---

## 2026-08-02 — Ingestion Pipeline Fixes

**Status**: Implemented  
**Scope**: Critical fixes to make the ingestion pipeline functional. Schema migration: `migrations/004_ingestion_fixes.sql`

**Headline**: Before fixes, upload could not reach pgvector. Three independent hard failures + one silent quality regression.

**Key fixes**:
- **B1**: JSONB values never encoded/decoded — no init hook on pool. Fixed with `set_type_codec` for json/jsonb
- **B2**: Celery chains used mutable signatures — `.s()` prepends return value. Fixed with `.si()` immutable signatures
- **B3**: `claim_next_document()` corrupted state — claimed oldest row regardless of id. Fixed with single guarded `UPDATE WHERE id=$1 AND stage=$2`
- **B4**: `file_type` never available — no column on documents. Every PDF chunked generically (no page resolution, no section prefix). Added column + persistence
- **B5**: `asyncio.run()` per task killed connection pool — creates and closes event loop. Fixed with one persistent loop per worker
- **B6**: Filename conflict dispatched id with no row — `ON CONFLICT DO NOTHING` returned existing row but route used minted UUID. Fixed to use returned row
- **B7**: Scan re-ingested every file — checked prefixed name instead of stored path. Fixed with `is_path_registered(raw_storage_path)`
- **B8**: Path traversal on upload filename — `Path(file.filename).name` strips directory component

**Refactoring**: collapsed claim/try/error blocks, parse/chunk no longer load SentenceTransformer, UNIQUE on artifact tables, `error_stage` for resume

---

## 2026-08-02 — Ingestion Workflow

**Status**: Documented

**Scope**: Stage-based ingestion pipeline design and configuration.

**Details**:
- Stage-based pipeline: **Parse → Chunk → Embed**, tracked by `documents` status table
- Intermediate artifacts: `document_parsed` (parsed text), `document_chunked` (chunk objects)
- Celery queues: `upload` (API uploads, 1 worker), `ingestion` (weekly batch, 2 workers)
- Status stages: `registered → parsing → parsed → chunking → chunked → embedding → embedded → error → failed`
- Schedules: weekly scan (Sunday 00:00), 6-hourly stale claim reset
- Reprocessing: failed docs resume from last successful stage
- Env vars: `INPUT_RAW_DIR`, `INGESTION_MAX_ATTEMPTS=2`, `INGESTION_CLAIM_TIMEOUT_MINUTES=30`

---

## 2026-08-02 — Project Refactoring

**Status**: Implemented

**Scope**: Codebase cleanup and structural improvements.

**Key changes**:
- **P0.1**: Celery queue routing mismatch — `task_routes` sent to `ingestion` but upload dispatched to `upload`. Fixed routing
- **P0.2**: Fork-unsafe globals — module-level `_config`, `_repo`, `_pipeline_cache`. Removed, created per-invocation helpers
- **P1**: Deleted obsolete files: `worker/tasks.py`, `ingestion/chunking/legacy/`, `experiment/`, `input/markdown/`
- **P1**: Removed unused Pydantic AI Agent from AppConfig
- **P1**: Replaced `print()` with `logging` module
- Config fixes: `OLLAMA_VLM_MODEL` standardized, requirements.txt duplicates removed, stale env vars cleaned
- Dockerfile improvements: test uses `rag-base`, postgres duplicate COPY removed
- Migration fix: non-existent `pgr_pageRank` function commented out

---

## 2026-08-04 — Ingestion Performance Investigation

**Status**: Investigated, fixes gated

**Scope**: Performance analysis of ~500-page PDF ingestion (~1 hour in Docker/Celery).

**Findings**:
- **F1**: Worker capped at 2 of 8 VM CPUs (25%). `cpus: "2.0"` in compose
- **F2**: Docling hardcoded `num_threads=2` — raising CPU quota alone changes nothing
- **F3**: Docling is largest fixed cost, invisible (no elapsed time logging)
- **F4**: Peak memory grows with total pages, not batch size — `batch_docs` retains all documents
- **F5**: VLM calls serial per page, no pipelining across pages
- **F6**: Claim timeout (30min) shorter than parse (~60min) — stale claims reset mid-parse
- **F7**: Parse blocks event loop — no heartbeat possible during run
- **F8**: `--max-tasks-per-child=1` re-imports torch/docling per stage
- **F9**: No thread caps — torch/OpenBLAS oversubscribe 2× inside container
- **F10**: Is `qwen3.5:0.8b` a vision model? (later confirmed yes)
- **F11**: `InvalidCxxCompiler` crash — runtime image has no compiler, torch dynamo tries to JIT. Fixed with `TORCHDYNAMO_DISABLE=1`
- **F12**: Manual thread tuning (2→4 threads, 2→6 CPUs) — 4× improvement but confounded by different document
- **F13**: VLM calls 27-33s each — `keep_alive` missing, model unloaded between calls. Fixed with `keep_alive: "30m"` (later superseded by F14)

**Plan**: instrument → baseline → gated fixes (A-G)

---

## 2026-08-04 — Stats Endpoint KeyError Fix

**Status**: Fixed

**Scope**: `GET /stats` endpoint returning "Error: 'docs'" due to column name mismatch.

**Problem**: SQL aliases `COUNT(DISTINCT document_id) as documents` but code read `result['docs']`

**Fix**: Two-line fix — `result['docs']` → `result['documents']` in `admin_routes.py:55,59`

**Tests**: Regression tests in `test_admin_stats.py` with static key check

---

## 2026-08-05 — VLM Output Length and Image Gate

**Status**: Implemented

**Scope**: VLM output stabilization and image size filtering.

**Key fixes**:
- **F18**: Latency is pure decode at ~35 tok/s, output length unbounded. `num_predict -1` + `temperature 0.8` = wandering output (22 tok one run, 342 the next on byte-identical input). Fixed with `temperature: 0.0`, `num_predict: 384`
- **F18b**: Size gate missed strips — `width < 150 AND height < 150` only caught square icons. 113 of 191 calls (60% of VLM budget) were sub-64px strips. Fixed with `VLM_MIN_IMAGE_SHORT_PX=64` (later superseded by `VLM_MIN_IMAGE_SHORT_PT=107pt`)
- **F18c**: Log couldn't answer its own question — added `in=`/`out=` tokens, `tok/s`, `done_reason` to VLM call log line. Warning on `done=length`

**Measured result**: `vlm_wait` 2245s → 592s on full 504-page run (after prompt v3: 261s)

**Key files**: `ollama_pdf_parser.py`, `gemini_docling_parser.py`, `app_config.py`, `pdf_parser_factory.py`

---

## 2026-08-05 — VLM Thinking and Table Routing

**Status**: Implemented

**Scope**: VLM reasoning model behavior fixes and resource optimization.

**Key fixes**:
- **F14**: `qwen3.5:0.8b` is a reasoning model with thinking on by default. 85s → 2.3s per call with `"think": false`. Reasoning discarded unread. `/no_think` prompt suffix doesn't work on qwen3.5
- **F14b**: Thinking returned empty response for tables — 3600 reasoning tokens, 0 answer tokens. Fixed: blank response counted as failure, falls back to `[IMAGE]`
- **F15**: VLM concurrency makes it worse locally — concurrency 2 is 27% slower, 4 is 5× slower against local Ollama. Default changed to 1
- **F16**: 0.8B VLM cannot read tables — garbage output, hallucinated rows. Pipeline already had TableFormer for simple tables. Fixed: VLM table branch gated on `VLM_TABLES=false`
- **F17**: Docker Desktop CPU slider not applied — showed 6 but `docker info` read 4. CPU ceilings summed to 9.0 against 4 real CPUs. `langfuse` not profile-gated (held 1G). Fixed: workers→4.0, postgres/app→1.0, langfuse profile-gated

**Key files**: `ollama_pdf_parser.py`, `gemini_docling_parser.py`, `app_config.py`, `docker-compose.yml`

---

## 2026-08-12 — Docker Logs and Auto-Migration

**Status**: Implemented

**Scope**: Persistent logging and automatic database migration on startup.

**Details**:
- **Persistent logs**: All services write to `./logs/<service>.log` via `tee -a`. `trap 'kill 0' EXIT` for clean shutdown. json-file driver with rotation
- **Auto-migration**: New `migrate` service runs before app services. Creates `schema_migrations` table, iterates `/migrations/[0-9]*.sql`, applies unapplied ones. Idempotent
- Service dependencies: app/workers depend on `migrate: service_completed_successfully`
- `schema_migrations` added to `_SYSTEM_TABLES` denylist

**Key files**: `docker-compose.yml`, `.gitignore`, `identifiers.py`

---

## 2026-08-20 — Logfire Config Crash Blocks Ingestion

**Status**: Fixed

**Scope**: `logfire.configure()` crash in `AppConfig.__init__` killing chunk stage.

**Problem**: Every stage enters through `_run_stage` → `_get_config()` → `AppConfig()` → `_configure_logfire()`. No guard meant config failure aborted task before stage work ran. Workers didn't receive `LOGFIRE_WRITE_TOKEN` (not in `x-common-env`), fell through to no-token branch which raises when no local auth present.

**Fixes**:
- **Fix 1**: `_configure_logfire` wraps both branches in try/except, degrades gracefully
- **Fix 2**: `LOGFIRE_WRITE_TOKEN`/`LOGFIRE_READ_TOKEN` moved into `x-common-env` for single source of truth

**Key files**: `app_config.py`, `docker-compose.yml`
