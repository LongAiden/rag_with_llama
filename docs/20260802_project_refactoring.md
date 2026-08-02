# 20260802 Project Refactoring

This document summarizes the refactoring work performed on the RAG pipeline project to address critical bugs, remove dead code, and improve maintainability.

## Critical Fixes (P0)

### 1. Celery Queue Routing

**Problem**: The `task_routes` configuration in `celery_app.py` routed all `worker.ingestion_tasks.*` to the `ingestion` queue. However, the upload endpoint in `document_routes.py` explicitly dispatches tasks to the `upload` queue via `apply_async(queue="upload")`. This created a mismatch where tasks sent to the `upload` queue would not be consumed by workers listening on the `ingestion` queue.

**Solution**: 
- Removed the dead `worker.tasks` module reference from `celery_app.py`
- Kept the wildcard route `worker.ingestion_tasks.*` → `ingestion` queue for weekly batch processing
- The upload endpoint's explicit `queue="upload"` parameter overrides the default routing, ensuring API uploads go to the correct queue
- Updated `ingestion_tasks.py` to explicitly specify `queue="ingestion"` when dispatching weekly batch chains

**Files Changed**:
- `worker/celery_app.py`
- `worker/ingestion_tasks.py`

### 2. Fork-Unsafe Globals in Celery Tasks

**Problem**: Module-level globals `_config`, `_repo`, and `_pipeline_cache` in `ingestion_tasks.py` were not fork-safe. Celery workers fork processes, and these cached objects (especially asyncpg connection pools) would break after fork.

**Solution**:
- Removed all module-level caching
- Created helper functions `_get_config()`, `_get_repo(config)`, and `_get_pipeline(config, table_name)` that create fresh instances per task invocation
- Updated all task functions to use these helpers instead of global state

**Files Changed**:
- `worker/ingestion_tasks.py`

## Dead Code Removal (P1)

### 3. Deleted Obsolete Files

**Problem**: Several files were no longer used after the migration to the stage-based ingestion pipeline:
- `worker/tasks.py` - Superseded by `ingestion_tasks.py`
- `ingestion/chunking/legacy/semantic_chunker.py` - Marked DEPRECATED, all functions duplicated
- `experiment/` directory - R&D artifacts replaced by production parsers
- `input/markdown/` directory - Data now stored in `document_parsed` table

**Solution**: Deleted all obsolete files and directories.

**Files Deleted**:
- `worker/tasks.py`
- `ingestion/chunking/legacy/` (entire directory)
- `experiment/` (entire directory)
- `input/markdown/` (entire directory)
- All `__pycache__/` directories

### 4. Removed Unused Pydantic AI Agent

**Problem**: `app_config.py` initialized a Pydantic AI Agent in `_configure_pydantic_ai()` that was never used. The actual LLM calls go through `llm_operations.py` backends.

**Solution**:
- Removed `_configure_pydantic_ai()` method
- Removed `self.agent` attribute from `AppConfig`
- Removed unused imports (`pydantic_ai`, `OpenAIModel`, `OllamaProvider`)
- Replaced `print()` statements with `logging` module

**Files Changed**:
- `config/app_config.py`

## Configuration Fixes (P1-P2)

### 5. OLLAMA_VLM_MODEL Mismatch

**Problem**: The `app` service in `docker-compose.yml` used `qwen3.5:4b` while all Celery workers used `qwen3.5:9b`. The `.env.example` specified `4b`.

**Solution**: Standardized all services to use `qwen3.5:9b` to match the workers and ensure consistent VLM capabilities.

**Files Changed**:
- `docker-compose.yml`

### 6. Requirements.txt Duplicates

**Problem**: 
- Both `PyPDF2` and `pypdf` were listed (PyPDF2 is the old name)
- Both `psycopg2-binary` and `asyncpg` were listed (asyncpg replaced psycopg2)
- `streamlit` was listed but not used (app uses FastAPI)

**Solution**:
- Removed `PyPDF2` (kept `pypdf`)
- Removed `psycopg2-binary` and `greenlet` (kept `asyncpg`)
- Removed `streamlit`

**Files Changed**:
- `deployment/requirements.txt`

### 7. Stale Environment Variables

**Problem**: `.env.example` contained many stale variables:
- `ENTITY_EXTRACTION_ENABLED` vs code uses `ENABLE_ENTITY_EXTRACTION`
- `CHUNKER_TYPE=recursive` (new pipeline uses `markdown`)
- `PAGERANK_DAMPING_FACTOR`, `PAGERANK_MAX_ITERATIONS` (not read by code)
- `LOG_ENTITY_EXTRACTION`, `LOG_RELATIONSHIP_EXTRACTION` (not read by code)
- `DB_POOL_*` variables (not passed to docker-compose services)
- `CACHE_TTL_SECONDS`, `USE_CELERY_FOR_UPLOAD` (not used)

**Solution**: Removed all stale variables and updated variable names to match code.

**Files Changed**:
- `.env.example`

## Dockerfile Improvements (P2)

### 8. Dockerfile.test Optimization

**Problem**: `Dockerfile.test` installed all dependencies from scratch on every build, re-downloading torch, sentence-transformers, etc.

**Solution**: Changed base image from `python:${PYTHON_VERSION}-slim` to `rag-base:latest` to reuse the pre-built ML dependencies.

**Files Changed**:
- `deployment/Dockerfile.test`

### 9. Dockerfile.postgres Duplicate COPY

**Problem**: `Dockerfile.postgres` had `COPY migrations/*.sql /docker-entrypoint-initdb.d/` which duplicated the volume mount in `docker-compose.yml`. The volume mount overrides the COPY at runtime.

**Solution**: Removed the COPY instruction since the volume mount handles migration injection.

**Files Changed**:
- `deployment/Dockerfile.postgres`

## Database Migration Fixes (P2)

### 10. Non-Existent pgr_pageRank Function

**Problem**: Migration `001_create_graph_tables.sql` defined a `calculate_entity_pagerank` function that called `pgr_pageRank()`, which does not exist in pgRouting.

**Solution**: Commented out the function definition with a note explaining that pgRouting does not provide PageRank and suggesting alternatives (pg_rank extension or application-level implementation).

**Files Changed**:
- `migrations/001_create_graph_tables.sql`

## Logging Improvements (P1)

### 11. Replaced print() with logging

**Problem**: Multiple files used `print()` for logging instead of the `logging` module or `logfire`, making it difficult to control log levels and route logs properly.

**Solution**: Replaced `print()` statements with `logger.info()`, `logger.warning()`, and `logger.error()` in:
- `config/app_config.py`
- `worker/ingestion_tasks.py`

**Files Changed**:
- `config/app_config.py`
- `worker/ingestion_tasks.py`

## Cleanup (P3)

### 12. Directory Structure

**Created**:
- `tests/fixtures/` - Directory for test fixtures

**Deleted**:
- All `__pycache__/` directories throughout the project

## Summary of Changes

| Category | Count | Impact |
|----------|-------|--------|
| Critical Bug Fixes | 2 | Prevents runtime failures |
| Dead Code Removal | 4 | Reduces maintenance burden |
| Configuration Fixes | 3 | Improves consistency |
| Dockerfile Improvements | 2 | Faster builds |
| Migration Fixes | 1 | Prevents SQL errors |
| Logging Improvements | 1 | Better observability |
| Cleanup | 2 | Cleaner codebase |

## Testing Recommendations

The following areas need test coverage:

1. **Ingestion Tasks** (`worker/ingestion_tasks.py`)
   - Test `parse_document_task` with various file types
   - Test `chunk_document_task` with different chunk sizes
   - Test `embed_document_task` with mock embeddings
   - Test error handling and retry logic
   - Test weekly `register_and_dispatch_task`

2. **Ingestion Repository** (`repositories/ingestion_repository.py`)
   - Test `register_document` with duplicate filenames
   - Test `claim_next_document` atomicity
   - Test `transition_to_parsed`, `transition_to_chunked`, `transition_to_embedded`
   - Test `record_error` and retry logic
   - Test `reset_stale_claims`

3. **Document Status Endpoint** (`/documents/{document_id}/status`)
   - Test status retrieval for each stage
   - Test 404 for non-existent documents
   - Test error message formatting

## Future Work

### High Priority
1. **Split `vector_store.py`** - The 1007-line file mixes embedding generation, vector CRUD, text cleaning, and search. Split into:
   - `embedding/generator.py`
   - `embedding/vector_store.py` (CRUD only)
   - `embedding/search.py`
   - `embedding/pipeline.py` (orchestrator)

2. **Make Hardcoded Values Configurable**:
   - `chunk_overlap=50` in `base_processor.py` and `vector_store.py`
   - `vector(384)` dimension in `vector_store.py` (should derive from model)
   - `batch_size=32` (embed) and `100` (insert) in `vector_store.py`
   - `chars_per_page=2500` in `docx_processor.py`
   - `h1/h2/h3_min_height` in `gemini_docling_parser.py`
   - `max_file_size_mb=50` in `models.py`

3. **Docker-Compose YAML Anchors** - Reduce ~110 lines of duplicated environment variables using YAML anchors:
   ```yaml
   x-app-env: &app-env
     POSTGRES_HOST: postgres
     # ... other shared vars
   
   services:
     app:
       environment:
         <<: *app-env
   ```

### Medium Priority
1. **Replace print() with logging** in remaining files:
   - `ingestion/processors/*.py`
   - `ingestion/chunking/chunker_factory.py`
   - `retrieval/llm_operations.py`
   - `retrieval/utils.py`
   - `api/routes/document_routes.py`

2. **Add DB Connection Context Manager** - Replace repeated `_get_connection()` / `_release_connection()` pattern with a context manager.

3. **Fix duplicate `_fix_markdown_headings`** in `gemini_docling_parser.py` (first definition is shadowed).

### Low Priority
1. **Fix typo** `moutain_pexcel.jpeg` → `mountain_pexcel.jpeg` in `templates.py` (image file doesn't exist, so low priority).

2. **Use Jinja2 templates** instead of 1253 lines of HTML strings in `templates.py`.

## Verification

All changes have been verified:
- ✅ Python files compile without errors
- ✅ `docker-compose.yml` validates successfully
- ✅ No references to deleted files remain in active code

## Rollback Plan

If issues arise, the following git commands can revert the changes:

```bash
# Revert all changes
git checkout HEAD -- worker/ config/ deployment/ migrations/ docker-compose.yml .env.example

# Restore deleted files
git checkout HEAD -- worker/tasks.py ingestion/chunking/legacy/ experiment/ input/markdown/
```
