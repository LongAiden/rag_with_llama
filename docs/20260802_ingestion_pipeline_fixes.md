# 20260802 Ingestion Pipeline Fixes

Review and repair of the stage-based ingestion pipeline (`Doc → Parsing → Chunking → Embedding → Vector Store`) introduced on `refactor/document-ingestion`, with the status DB (`documents`, `document_parsed`, `document_chunked`) as source of truth.

**Headline finding: before these fixes an upload could not reach pgvector.** Three independent hard failures sat in front of the first embedding, plus one silent quality regression that would have survived all of them.

See also [20260802_ingestion_workflow.md](20260802_ingestion_workflow.md) and [20260802_project_refactoring.md](20260802_project_refactoring.md).

---

## Blockers

### B1. JSONB values were never encoded or decoded

`ConnectionPoolManager.get_pool()` created the pool with no `init` hook, so asyncpg mapped `json`/`jsonb` to `str`.

- `register_document(metadata={...})` passed a `dict` into a JSONB parameter → `DataError` on **every** upload, before Celery was ever reached.
- `transition_to_chunked(chunks=[...])` had the same problem with a list.
- On read, `document_chunked.chunks` came back as raw text, so `[SimpleNamespace(**c) for c in chunked["chunks"]]` and `dict(doc["metadata"])` both failed.

`VectorStore` had been hand-rolling `json.dumps` / `json.loads` around this for its own metadata column; the ingestion repository had not.

**Fix** — register the codec once, in the pool:

```python
async def _init_connection(conn):
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(
            type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
```

Coupled change: `VectorStore.add_chunks` now passes the metadata dict directly (encoding it first would double-encode), and the three `json.loads` fallbacks in the read paths were removed as dead code.

**Files**: `repositories/connection_pool.py`, `ingestion/embedding/vector_store.py`

### B2. Celery chains used mutable signatures

```python
chain(parse_document_task.s(doc_id), chunk_document_task.s(doc_id), ...)
```

A chain prepends each task's return value to the next task's arguments, so `chunk_document_task` was invoked as `chunk_document_task({"status": "parsed", ...}, doc_id)` → `TypeError`. No upload ever got past parse.

**Fix** — immutable signatures (`.si`), built in one shared place so the two dispatch sites cannot drift:

```python
def build_ingestion_chain(doc_id, from_stage="registered", queue=INGESTION_QUEUE):
    steps = {
        "registered": (parse_document_task, chunk_document_task, embed_document_task),
        "parsed":     (chunk_document_task, embed_document_task),
        "chunked":    (embed_document_task,),
    }.get(from_stage)
    if not steps:
        return None
    return chain(*[task.si(doc_id).set(queue=queue) for task in steps])
```

The queue is set per signature rather than relying on chain-level option propagation, so every stage provably lands on a worker that consumes it.

**Files**: `worker/ingestion_tasks.py`, `api/routes/document_routes.py`

### B3. Claiming "the next document in stage" corrupted state

`_parse_document(doc_id)` called `claim_next_document()`, which claimed whichever row was oldest in `registered`, **moved it to `parsing`**, then compared ids and returned `{"status": "skipped"}`.

With two uploads in flight, document B's task claimed document A, abandoned it still claimed and stuck in `parsing` for the full 30-minute timeout, and B never ran at all. `skipped` was also returned as success, so the chain continued and each later stage repeated the damage.

**Fix** — `claim_document(doc_id, current_stage, processing_stage, worker_id, timeout_minutes)`: a single guarded `UPDATE ... WHERE id = $1 AND stage = $2 AND (claimed_at IS NULL OR claimed_at < NOW() - $5) RETURNING *`. It touches exactly one row or none. `claim_next_document` was removed.

**Files**: `repositories/ingestion_repository.py`, `worker/ingestion_tasks.py`

### B4. `file_type` was never available — the silent one

`documents` had no `file_type` column, so `doc.get("file_type", "")` was **always** `""`, and `chunk_parsed_document` never entered its `if file_type == 'pdf'` branch.

Every PDF was chunked by the generic path: no `[Page N]` resolution, no section-hierarchy prefix, no `full_content`. Retrieval quality degraded with **zero error anywhere** — this would have survived every other fix on this list.

A related leak: `page_mapping` (used for non-PDF page numbers) was produced by parse and then dropped, so DOCX/TXT chunks all reported page 1.

**Fix** — added `documents.file_type`, populated at the parsed transition; `page_mapping` is persisted in the parsed artifact's metadata and passed into chunking. Chunk and embed read the real value, falling back to the artifact metadata when the column is NULL.

**Files**: `migrations/004_ingestion_fixes.sql`, `repositories/ingestion_repository.py`, `worker/ingestion_tasks.py`

### B5. `asyncio.run()` per task killed the connection pool

Each task called `asyncio.run()`, which creates **and closes** an event loop. `ConnectionPoolManager._instances` is a class attribute that survives in the worker process, so the second task in a process received a pool whose connections were bound to a closed loop → `RuntimeError: Event loop is closed`.

**Fix** — one persistent loop per worker process, deliberately never closed:

```python
def _run(coro):
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP.run_until_complete(coro)
```

**Files**: `worker/ingestion_tasks.py`

> **Note on the earlier fork-safety decision.** [20260802_project_refactoring.md](20260802_project_refactoring.md) removed module-level caches as "fork-unsafe". That reasoning applies to state populated in the **parent** before `fork()`. Both the loop above and the pipeline cache below are populated lazily inside task execution, which always runs in the child — so each worker process builds its own. The pools stay per-process and fork-safe.

### B6. A filename conflict dispatched an id that had no row

`register_document` uses `ON CONFLICT (file_name) DO NOTHING` and returns the **existing** row, but the upload route discarded that return value and chained on the UUID it had just minted. That id existed in no table, so all three stages skipped it, `/documents/{id}/status` returned 404, and the raw file was orphaned on disk.

**Fix** — the route uses the returned row. On a conflict it removes the file it just wrote and responds `status="duplicate"` with the **existing** document id and stage, pointing at `DELETE /documents/{id}` to re-ingest.

**Files**: `api/routes/document_routes.py`

### B7. The scan re-ingested every uploaded file

Uploads are written to disk as `<uuid>_<name>` but registered under `<name>`. The scan checked `is_file_registered(entry.name)` — i.e. the prefixed name — which never matched, so it registered a **second document for the same bytes** and re-parsed, re-chunked and re-embedded it. Duplicate chunks accumulated in the vector store on every run.

**Fix** — `is_path_registered(raw_storage_path)`; the scan keys on the stored path, with the filename check kept as a secondary guard.

**Files**: `repositories/ingestion_repository.py`, `worker/ingestion_tasks.py`

### B8. Path traversal on the upload filename

`INPUT_RAW_DIR / f"{document_id}_{file.filename}"` — FastAPI does not sanitise `UploadFile.filename`, so a name like `../../../app/worker/celery_app.py` escaped `data/input/raw/`.

**Fix** — `Path(file.filename).name` strips any directory component before the path is built. `validate_table_name()` is now also applied on `/upload`; only the delete route had been checking it, though the upload value reaches DDL through the vector store.

**Files**: `api/routes/document_routes.py`

---

## Refactoring

| # | Change |
|---|---|
| R1 | Three near-identical claim → try → `record_error` blocks collapsed into one `_run_stage(doc_id, stage_name, work)` helper. The error path now exists once. |
| R2 | Parse and chunk no longer construct a `ChunkEmbeddingPipeline`. `parse_file` and `chunk_parsed_document` are now `@staticmethod` (they never used `self`), so only the embed stage loads a SentenceTransformer — and it is cached per table per worker process instead of reloaded three times per document. |
| R3 | `UNIQUE (document_id)` on both artifact tables plus `ON CONFLICT DO UPDATE`. Previously a retry appended a second row and `get_parsed()` — a `fetchrow` with no `ORDER BY` — picked one non-deterministically. |
| R4 | `record_error` stores `error_stage`, and `reset_error_documents` resumes from the last completed artifact. An embedding failure no longer pays for re-parsing a PDF through a VLM backend. |
| R5 | The UI claimed *"uploaded and processed successfully"* the instant `/upload` returned, while the response was `status="queued"`. It now says queued and polls `/documents/{id}/status` every 3s, reporting each stage and ending on `embedded` or `failed`. |
| R6 | **New**: `DELETE /documents/{document_id}` — clears the status row (cascading to both artifacts), the vector chunks, and the raw file. Flags `delete_chunks` / `delete_raw_file` default to true. Raw files outside `INPUT_RAW_DIR` are never unlinked. |
| R7 | The 6-hourly sweep reset stale stages but **dispatched nothing**, so a released document idled until Sunday. Split into `recover_and_dispatch` (every 6h: sweep, retry, re-queue) and `register_and_dispatch` (weekly: the above plus the directory scan). |
| R8 | Removed the `celery_worker_rag` service — it consumed a `rag` queue nothing published to, a 1 GB idle container. `task_default_queue` now points at `ingestion`, a queue that is actually consumed. |
| R9 | Dead/incorrect code: unreachable `stage == "error"` branch; `doc.get("chunk_size", 512)` returning `None` for a NULL column (`.get`'s default only applies to *missing* keys); unused imports; `upload_timestamp` built from `uuid.uuid1().time` (a UUID clock tick, not a timestamp) replaced with an ISO-8601 UTC timestamp. |

---

## Schema changes — `migrations/004_ingestion_fixes.sql`

```sql
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type   TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_stage TEXT;

CREATE UNIQUE INDEX ... ON document_parsed(document_id);
CREATE UNIQUE INDEX ... ON document_chunked(document_id);
CREATE INDEX ... ON documents(raw_storage_path);
```

Existing duplicate artifact rows are de-duplicated (newest kept) before the unique indexes are created.

> ⚠️ Migrations are mounted at `/docker-entrypoint-initdb.d`, which Postgres runs **only on an empty data directory**. On an existing volume, apply by hand:
> ```bash
> docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
>   < migrations/004_ingestion_fixes.sql
> ```
> Or start clean with `docker compose down -v`.

---

## Tests

The pre-existing suite mocked both `_get_repo` and `_get_pipeline`, so B1, B2, B3 and B5 were all invisible to it — and one test asserted a `file_type` key on a `documents` row that had no such column, actively encoding B4.

Added regression guards, each named for the bug it pins:

- `claim_document` is scoped to one id and one stage, and never emits a `LIMIT`
- chains are built from immutable signatures with `args == (doc_id,)` and a queue on every task
- chains resume correctly from `parsed` / `chunked`, and are `None` for terminal stages
- a PDF reaches the chunker as `"pdf"`, including the fallback from artifact metadata
- a NULL `chunk_size` column does not propagate as `None`
- the scan skips files already registered by path
- recovery re-dispatches what it resets, and does not run the directory scan
- `_run` reuses a single, still-open event loop across calls
- json/jsonb codecs registered with a round-trip over the real artifact shapes
- delete endpoint: cascade, 404, auth, flags, and refusal to unlink outside `INPUT_RAW_DIR`

**Result**: 436 passed, 25 failed — the same 25 that failed before this work (`test_app_config`, `test_chunker_factory`, `test_delete_table_security`, `test_llm_provider`, `test_vector_store`), all pre-existing and unrelated. `tests/unit/test_pdf_to_markdown.py` does not import at all: it targets `ingestion.processors.pdf_to_markdown`, deleted in an earlier refactor. Both are still open.

---

## Verification

```bash
# 1. Unit
python -m pytest tests/unit -q --ignore=tests/unit/test_pdf_to_markdown.py

# 2. Schema (fresh volume)
docker compose down -v && docker compose up -d postgres
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\d documents'

# 3. Upload a multi-page PDF from the UI at localhost:8000
docker compose up -d
docker compose logs -f celery_worker_upload
# expect: registered → parsing → parsed → chunking → chunked → embedding → embedded
```

**4. B4 regression check — the silent one.** After ingest, confirm the PDF metadata is populated rather than defaulted:

```sql
SELECT metadata->>'page_number', metadata->>'section_path'
FROM document_chunks WHERE document_id = '<id>' LIMIT 10;
```

`page_number` must vary across pages and `section_path` must be non-empty. Before the fix, every row was page `1` with an empty path.

**5. B5** — upload two documents back-to-back through the same worker; the second must not raise "Event loop is closed".

**6. B3** — upload two documents concurrently; neither may be parked in `parsing` with a non-null `claimed_at`, and both must reach `embedded`.

**7. Dedupe + delete** — re-upload the same filename and expect `status="duplicate"` with the original id. Then `DELETE /documents/{id}` and confirm the status row, both artifacts, the chunks and the raw file are gone, and that re-uploading ingests fresh.

**8. B7** — with an already-ingested file in `data/input/raw/`:

```bash
docker compose exec celery_worker_ingestion \
  celery -A worker.celery_app call worker.ingestion_tasks.register_and_dispatch
```

must report `registered: 0`.

---

## Known gaps (not addressed)

- **Dedupe is by filename**, retained deliberately for the POC. A corrected version of `report.pdf` cannot be re-ingested without deleting the original first — hence `DELETE /documents/{id}`. The unused `documents.content_hash` column is where hash-based dedupe would go.
- **`search_bm25` loads the entire chunk table into memory** and rebuilds the BM25 index on every query. Fine at POC scale, O(n) per query beyond it.
- **Graph modules** (`graph_processing/`) were out of scope and remain disabled. They use their own `asyncpg.create_pool` without the new codec, so their `json.dumps` calls are unaffected.
- **The 25 pre-existing unit failures** and the non-importing `test_pdf_to_markdown.py` are untouched.
