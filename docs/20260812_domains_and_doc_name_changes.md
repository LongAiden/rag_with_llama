# Changes: Domain Registry + `doc_name` — Implementation Record

**Date**: 2026-08-12
**Branch**: `refactor/document-ingestion`
**Plan**: [`plans/20260812_domains_and_doc_name.md`](./plans/20260812_domains_and_doc_name.md)
**Status**: Code complete. Migration 006 not yet applied — see [§7](#7-what-you-need-to-run).

---

## 1. What Changed, in One Paragraph

A chunk table (`math`, `history`) was always an implicit topic bucket, and the only
thing tying a retrieved chunk to its source was an opaque UUID. This makes the bucket
explicit as a **domain** with a registry table, gives every document a **human-readable
name**, and carries that name all the way to the top-k results and the chat UI.

```
Upload  "Linear_Algebra_v3.pdf",  doc_name="Linear Algebra",  domain="math"
  → ensure_domain('math')          domains:   name=math, table_name=math
  → register_document(...)         documents: doc_name='Linear Algebra', domain='math',
                                              target_table_name='math'
  → parse → chunk → embed
  → INSERT INTO math (..., doc_name) VALUES (..., 'Linear Algebra')   × 847 rows

Query   "What is a vector?",  domain="math",  document_ids=["abc-123"] (optional)
  → resolve domain → table_name='math'
  → SELECT id, text, metadata, document_id, doc_name, similarity
  → (+ BM25 / RRF in hybrid mode, both carrying doc_name)
  → rerank → doc_name restored from original_by_id
  → UI: "Source 1 — 87.2% similarity | 📄 Linear Algebra · Page 12"
```

One domain = one table = many documents = many chunks.

---

## 2. Three Decisions Worth Knowing

**`doc_name` is denormalized onto every chunk row.** Deliberate: the search `SELECT`
returns the name with no per-query join to `documents`. The price is drift —
`documents.doc_name` is authoritative, the chunk copy is a snapshot, and a rename
endpoint (out of scope) must update both. The column COMMENT in the migration points
at the plan so the next person hits this in the schema, not in production.

**`document_id` stays the filter key; `doc_name` is a label.** Upload the same book
twice and you get two `documents` rows with different UUIDs and the *same* `doc_name`.
So the UI lists documents and sends `document_ids`; the `doc_name` query filter
resolves to ids server-side and returns all matches rather than silently picking one.

**`documents.domain` is user-facing; `target_table_name` is derived** from
`domains.table_name` at upload time. The worker still reads `target_table_name` and
needed no change to how it picks a table.

---

## 3. Files Changed

### New (5)

| File | What it is |
|---|---|
| `deploy/migrations/006_domains_and_doc_name.sql` | Registry table, `documents.doc_name`/`domain`, per-chunk-table `doc_name` + index, backfills, FK |
| `src/app/infra/db/domain_repository.py` | `DomainRepository` — registry CRUD, documents-per-domain, name→id resolution |
| `src/app/api/routes/domain_routes.py` | 5 endpoints under `/domains` |
| `src/app/api/routes/table_deletion.py` | `drop_chunk_table()` — the drop sequence shared by `DELETE /table` and `DELETE /domains` |
| `tests/unit/test_domains_and_doc_name.py` | 30 unit tests, no DB |
| `tests/integration/test_domains_integration.py` | Registry, membership, chunk-table, and reconciliation tests (needs Postgres) |

### Modified (16)

| File | Change |
|---|---|
| `src/app/infra/db/identifiers.py` | Made `_SYSTEM_TABLES` load-bearing; added the application tables (§4) |
| `src/app/infra/db/__init__.py` | Export `DomainRepository` |
| `src/app/infra/db/ingestion_repository.py` | `register_document(doc_name=, domain=)`; filename-stem default |
| `src/app/ingestion/embedding/chunk.py` | `doc_name: Optional[str] = None` |
| `src/app/ingestion/embedding/vector_store.py` | `doc_name` in CREATE + idempotent ALTER + index; 6-tuple INSERT with upsert; `doc_name` in all three read paths |
| `src/app/ingestion/embedding/pipeline.py` | `doc_name` through `embed_chunks` and `ingest_document`, into `Chunk` and chunk metadata |
| `src/app/worker/ingestion_tasks.py` | `_embed_document` reads `doc_name`/`domain`; the directory scan ensures its domain exists and names scanned files |
| `src/app/models/schemas.py` | `DomainInfo`, `DomainDocument`, `CreateDomainRequest`; `QueryRequest.domain`/`doc_name`; `RAGSource.doc_name`; `UploadResponse.doc_name`/`domain` |
| `src/app/retrieval/search.py` | `doc_name` restored through rerank; set on `RAGSource`; source blocks in the LLM prompt now named |
| `src/app/api/routes/document_routes.py` | `domain`/`doc_name` form fields, `ensure_domain`, both echoed in the response and the status endpoint |
| `src/app/api/routes/query_routes.py` | `_resolve_search_target()`; `domain` → table, `doc_name` → ids; `doc_name` in the HTML source dict |
| `src/app/api/routes/table_routes.py` | `DELETE /table/{name}` now calls the shared helper |
| `src/app/api/app.py` | Mount `domain_routes` |
| `src/app/api/templates/home.html` | Upload name field + domain picker; chat domain/document selects; named sources; `loadDomainList`/`loadDomainDocuments` |
| `src/app/api/templates/search_results.html` | Show `doc_name` |
| `docs/ARCHITECTURE.md` | §4.1 columns, §4.2c domains, §6.2 repositories, §6.3 reserved names, §6.4 migration 006, §8 routers + shared helper, §10.3 baseline, §12 glossary |
| `tests/unit/test_vector_store.py`, `tests/unit/test_ingestion_tasks.py` | Fixtures updated for the new column and the domain lookup |

---

## 4. A Latent Bug Fixed Along the Way

`_SYSTEM_TABLES` in `identifiers.py` was **declared and never read**. Uploading with
`table_name=documents` therefore reached `VectorStore._initialize_database()`, whose
`CREATE TABLE IF NOT EXISTS "documents"` silently matched the *status* table and then
failed on INSERT with a column error. `validate_table_name()` now consults the list,
which grew to include `domains`, `documents`, `document_parsed`, `document_chunked`,
and `llm_interactions` alongside the graph tables. Domain names go through the same
check, since a domain name *is* a table name.

This is independent of the feature and worth having regardless.

---

## 5. The Fragile Point

The rerank block in [`search.py`](../src/app/retrieval/search.py) rebuilds result dicts
from `RerankedResult`, which declares only `chunk_id, text, document_id, metadata,
similarity, rerank_score, original_rank, new_rank`. **Any field not explicitly restored
from `original_by_id` is dropped silently** — that is why `bm25_score` and `rrf_score`
were already being restored there. `doc_name` joins them:

```python
'doc_name': original_by_id[r.chunk_id].get('doc_name'),
```

`tests/unit/test_domains_and_doc_name.py::TestRerankPreservesDocName` asserts both that
`RerankedResult` carries no `doc_name` *and* that the name survives a rerank that
reorders results. That pair is what catches the regression if someone edits the block.

---

## 6. API Surface

### New endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/domains` | `{"domains": [DomainInfo], "total": n}` — also registers chunk tables with no registry row |
| `POST` | `/domains` | Create (idempotent); body `{name, display_name?, description?}`; password-protected |
| `GET` | `/domains/{name}` | One `DomainInfo` |
| `GET` | `/domains/{name}/documents` | Documents with `doc_name`, `stage`, `chunk_count` — including ones still ingesting |
| `DELETE` | `/domains/{name}` | Chunk table + status rows + registry row; password-protected |

### Changed endpoints

- **`POST /upload`** — new `domain` and `doc_name` form fields. `domain` wins over
  `table_name`; an unknown domain is created. `doc_name` defaults to the filename stem.
  Response echoes both.
- **`POST /query`** — new `domain` and `doc_name` body fields. `domain` resolves to the
  chunk table; `doc_name` resolves to `document_ids`. Every `RAGSource` now carries
  `doc_name`.
- **`POST /query-form`** — same two fields; `document_ids` is no longer hardcoded to
  `None`.
- **`GET /documents/{id}/status`** — now returns `doc_name` and `domain`.

### Unchanged

`GET /tables`, `GET /tables/count`, `DELETE /table/{name}` all behave exactly as before.
`POST /query` with only `table_name` takes the old path with no registry lookup.
`RAGSource.doc_name` is an added optional field.

---

## 7. What You Need to Run

Migrations are mounted at `/docker-entrypoint-initdb.d`, which Postgres runs **only on
an empty data directory**. On an existing volume, apply 006 by hand:

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < deploy/migrations/006_domains_and_doc_name.sql

docker compose restart app celery_worker_upload celery_worker_ingestion
```

Then check it took:

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT name, display_name, table_name FROM domains;"
```

### Optional: backfill `doc_name` onto existing chunk rows

Deliberately **not** in the migration — it is an `UPDATE ... FROM documents` per table
that rewrites every row. Until you run it, pre-006 chunks return `doc_name: null` and
the UI falls back to the truncated UUID, which is exactly today's behaviour. Per domain:

```sql
UPDATE math m
SET doc_name = d.doc_name
FROM documents d
WHERE m.document_id = d.id AND m.doc_name IS NULL;
```

Chunk tables created before 006 also self-heal on first use —
`VectorStore._initialize_database()` runs `ADD COLUMN IF NOT EXISTS doc_name`.

---

## 8. Test Status

```
tests/unit: 547 passed, 14 failed
```

All 14 failures are pre-existing on this branch and unrelated:

- `test_chunker_factory.py` (5) — chonkie API drift
- `test_delete_table_security.py` (6) — MagicMock/jsonable_encoder; verified same count
  and same failure mode before and after the `drop_chunk_table` extraction
- `test_llm_provider.py` (1) — archived file restored to `tests/unit/`
- `test_pdf_parser_factory.py` (2) — the parser refactor renamed `min_image_short_px`
  to `min_image_short_pt`; the test still asserts the old kwarg

Three tests broke on the new behaviour and were updated rather than worked around:
`test_vector_store` (mock row needed `doc_name`) and two `test_ingestion_tasks` scan
tests (the scan now ensures its domain exists).

`tests/integration/test_domains_integration.py` skips itself if migration 006 has not
been applied, so it is safe to run before you migrate. Run after:

```bash
uv run pytest tests/integration/test_domains_integration.py -v
```

---

## 9. Out of Scope

- Renaming a document or domain (needs the coordinated two-table update from §2)
- Cross-domain search — one query spanning several chunk tables
- `doc_name` autocomplete / fuzzy matching
- Per-domain embedding models or chunk sizes
- Migrating `/tables` consumers off the endpoint
