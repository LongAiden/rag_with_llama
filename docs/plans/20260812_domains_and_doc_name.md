# Plan: Domain Registry + `doc_name` on Chunks and Query Results

**Date**: 2026-08-12
**Status**: Implemented (2026-08-12) — migration 006 still needs to be applied by hand on existing volumes, see §7.1
**Supersedes**: [`20260811_doc_name_column.md`](./20260811_doc_name_column.md)
**Scope**: DB schema, ingestion pipeline, query pipeline, API routes, response models, UI

---

## 1. The Model, Concretely

Upload `Linear_Algebra_v3.pdf` with `doc_name="Linear Algebra"` into domain `math`.

**`domains`** — the registry. One row per domain, 1:1 with a physical chunk table:

| name | display_name | description | table_name |
|---|---|---|---|
| `math` | Mathematics | Linear algebra, calculus, analysis | `math` |

**`documents`** — one row per uploaded book, unchanged in role (ingestion status DB):

| id | file_name | doc_name | domain | target_table_name | stage | chunk_count |
|---|---|---|---|---|---|---|
| `abc-123` | `Linear_Algebra_v3.pdf` | Linear Algebra | `math` | `math` | embedded | 847 |
| `def-456` | `Calculus.pdf` | Calculus | `math` | `math` | embedded | 612 |

**`math`** — the pgvector chunk table. 1459 rows, every chunk carrying its book's id *and* name:

| id | document_id | doc_name | text | embedding | metadata |
|---|---|---|---|---|---|
| `chunk-001` | `abc-123` | Linear Algebra | "Vectors are…" | `[0.1, …]` | `{page_number: 12, …}` |
| `chunk-002` | `abc-123` | Linear Algebra | "A matrix is…" | `[0.3, …]` | `{page_number: 12, …}` |
| `chunk-848` | `def-456` | Calculus | "A derivative…" | `[0.7, …]` | `{page_number: 3, …}` |

So: **one domain = one table = many documents = many chunks.** Nothing about how
chunks are stored or searched changes; `VectorStore` still receives a validated
table name and is unaware domains exist.

### Three consequences worth stating up front

1. **`doc_name` is denormalized onto every chunk row.** Deliberate: the search
   `SELECT` returns the name with no join to `documents` per query. Cost is ~15
   bytes × 1459 rows here. The price is §7.3 — a rename must update both tables or
   they drift.
2. **`document_id` stays the filter key; `doc_name` is a label.** Upload
   `Linear_Algebra.pdf` twice and you get two `documents` rows with different UUIDs
   and the *same* `doc_name`; filtering on `doc_name = 'Linear Algebra'` would hit
   both editions. The UI therefore lists documents and sends `document_ids`.
3. **`documents.domain` is the user-facing key; `target_table_name` is derived**
   from `domains.table_name` at upload time. The worker
   ([ingestion_tasks.py:283](../../src/app/worker/ingestion_tasks.py#L283)) keeps
   reading `target_table_name` and needs no change to how it picks a table.

---

## 2. What Changed vs. the 2026-08-11 Plan

That plan is directionally right and **none of it is implemented** —
`grep -rn doc_name src/ deploy/` returns nothing, migrations stop at `005`. This
revision keeps its `doc_name` core and corrects four things, then adds domains.

| # | 08-11 plan said | Reality / this plan |
|---|---|---|
| 1 | §3.7.2: add a `doc_name` input to the "Search Form" in `home.html` | **There is no search form.** The chat panel is JS — `sendChat()` at [home.html:592](../../src/app/api/templates/home.html#L592) POSTs JSON to `/query`. `/query-form` + `search_results.html` are the legacy path. Both get touched; the JS path is the one users see. |
| 2 | Filter with `doc_name = $n` string equality | **Fragile**, per §1.2 above — that plan's own §8.1 allows duplicate `doc_name`, so the filter is ambiguous by construction and a typo returns zero rows silently. `document_ids` is exact and already plumbed end to end. |
| 3 | Migration `006` loops chunk tables via `information_schema` | Kept, matching the introspection rule in [table_repository.py:12](../../src/app/infra/db/table_repository.py#L12). |
| 4 | — | **New: the `domains` registry**, plus reserving `domains` as a table name (§3.2 — a latent bug found on the way). |

Everything else from 08-11 carries over unless noted.

---

## 3. Database Changes

### 3.1 Migration `deploy/migrations/006_domains_and_doc_name.sql`

```sql
-- ============================================
-- Migration 006: domain registry + doc_name
--
-- A chunk table (math, history, technical) has always been an implicit topic
-- bucket. This makes it explicit: `domains` is the registry, `documents.domain`
-- is the membership FK, and `doc_name` is the human-readable document label,
-- denormalized onto chunk rows so the search SELECT needs no join.
--
-- NOTE: migrations are mounted at /docker-entrypoint-initdb.d, which Postgres
-- only runs on an empty data directory. On an existing volume apply by hand:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     < deploy/migrations/006_domains_and_doc_name.sql
-- ============================================

CREATE TABLE IF NOT EXISTS domains (
    name         TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description  TEXT,
    table_name   TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT domains_name_is_identifier
        CHECK (name ~ '^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')
);

COMMENT ON TABLE  domains IS 'Registry of document domains. One domain maps 1:1 to one pgvector chunk table.';
COMMENT ON COLUMN domains.name IS 'Slug and API key. Constrained to a safe SQL identifier because it is also the chunk table name.';
COMMENT ON COLUMN domains.table_name IS 'Physical chunk table. Equals name today; kept separate so a domain can be renamed without a table rename.';

-- documents: membership + human-readable name
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain   TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_domain   ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_doc_name ON documents(doc_name);

COMMENT ON COLUMN documents.doc_name IS 'Human-readable document name from upload; defaults to the filename stem. Denormalized onto chunk rows — see migration 006 and docs/plans/20260812_domains_and_doc_name.md §7.3 before renaming.';
COMMENT ON COLUMN documents.domain   IS 'Domain this document belongs to; FK to domains(name). NULL for pre-006 rows until backfilled.';

-- Backfill a domain row for every chunk table that already exists, so nothing
-- disappears from the UI after the migration.
DO $$
DECLARE t TEXT;
BEGIN
    FOR t IN
        SELECT DISTINCT t1.table_name
        FROM information_schema.columns t1
        WHERE t1.table_schema = 'public'
          AND t1.column_name = 'document_id'
          AND EXISTS (
              SELECT 1 FROM information_schema.columns t2
              WHERE t2.table_name = t1.table_name
                AND t2.table_schema = 'public'
                AND t2.column_name = 'embedding'
          )
          AND t1.table_name NOT IN
              ('entities', 'relationships', 'entity_nodes', 'entity_edges')
    LOOP
        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS doc_name TEXT', t);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (doc_name)',
                       t || '_doc_name_idx', t);

        INSERT INTO domains (name, display_name, table_name)
        VALUES (t, initcap(replace(t, '_', ' ')), t)
        ON CONFLICT (name) DO NOTHING;
    END LOOP;
END $$;

-- Backfill existing documents: doc_name from the filename stem, domain from the
-- table they were ingested into.
UPDATE documents
SET doc_name = regexp_replace(file_name, '\.[^.]+$', '')
WHERE doc_name IS NULL;

UPDATE documents
SET domain = target_table_name
WHERE domain IS NULL
  AND target_table_name IN (SELECT name FROM domains);

-- FK added after the backfill so it cannot reject the migration's own rows.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_domain_fkey;
ALTER TABLE documents
    ADD CONSTRAINT documents_domain_fkey
    FOREIGN KEY (domain) REFERENCES domains(name) ON DELETE SET NULL;
```

**Chunk-row backfill is *not* in the migration.** Setting `doc_name` on existing
chunk rows needs a per-table `UPDATE ... FROM documents` that rewrites every row.
§7.2 makes it an opt-in command.

### 3.2 Reserve `domains` as a table name — latent bug

`validate_table_name` ([identifiers.py:19](../../src/app/infra/db/identifiers.py#L19))
accepts `domains`. Today a user can upload with `table_name=documents`, and
`VectorStore._initialize_database()` runs `CREATE TABLE IF NOT EXISTS "documents"` —
which silently matches the *status* table, then fails on INSERT with a column
error. Adding the registry makes one more name collidable.

Fix: add `domains`, `documents`, `document_parsed`, `document_chunked`,
`llm_interactions` to `_SYSTEM_TABLES` and make `validate_table_name` actually
consult it. That frozenset is currently declared and never read — in
`identifiers.py` *and* in `table_repository.py`, where the same list is
re-hardcoded inside `CHUNK_TABLES_QUERY`. This is independent of the feature and
worth doing regardless.

### 3.3 Lazy schema evolution in `VectorStore`

Add `doc_name TEXT` to the `CREATE TABLE` at
[vector_store.py:88](../../src/app/ingestion/embedding/vector_store.py#L88), plus an
idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS doc_name TEXT` and a
`{table}_doc_name_idx` index, so a table created before 006 self-heals on first
use instead of waiting for a manual migration run.

---

## 4. Backend Changes

### 4.1 New: `src/app/infra/db/domain_repository.py`

```python
class DomainRepository:
    """CRUD over the `domains` registry. Owns nothing about chunk storage."""

    async def list_domains(self) -> List[Dict[str, Any]]: ...
        # SELECT d.*, COUNT(doc.id) AS document_count
        #   FROM domains d LEFT JOIN documents doc ON doc.domain = d.name
        #  GROUP BY d.name ORDER BY d.display_name

    async def get_domain(self, name: str) -> Optional[Dict[str, Any]]: ...

    async def create_domain(self, name, display_name=None, description=None) -> Dict: ...
        # validate_table_name(name) first — name IS the table name.
        # display_name defaults to name.replace('_', ' ').title()
        # INSERT ... ON CONFLICT (name) DO NOTHING, then re-select.

    async def ensure_domain(self, name: str) -> Dict: ...
        # get_domain or create_domain. Used by /upload so uploading into a new
        # domain works without a separate create call — matching today's implicit
        # table creation.

    async def list_documents(self, name: str) -> List[Dict[str, Any]]: ...
        # SELECT id, doc_name, file_name, stage, chunk_count, created_at
        #   FROM documents WHERE domain = $1 ORDER BY doc_name

    async def delete_domain(self, name: str) -> None: ...
```

`list_documents` reads `documents`, not the chunk table, so a book still mid-ingest
(`stage='parsing'`) is visible with its stage — the UI greys it out rather than
pretending it doesn't exist.

### 4.2 `Chunk` — [chunk.py](../../src/app/ingestion/embedding/chunk.py)

```python
doc_name: Optional[str] = None
```

### 4.3 `VectorStore` — [vector_store.py](../../src/app/ingestion/embedding/vector_store.py)

**No `doc_name` filter parameter** (departure from 08-11 §3.5.2). The existing
`document_ids` filter stays as-is and `doc_name` is only *added to the SELECT list
and the returned dicts*. The 08-11 approach layered manual `param_idx` arithmetic
onto the already-branched `$3`/`$4` construction at
[vector_store.py:217-230](../../src/app/ingestion/embedding/vector_store.py#L217-L230) —
positional-parameter bookkeeping that breaks quietly.

- `add_chunks` → 6-tuple; `INSERT ... (id, document_id, text, embedding, metadata, doc_name)`; `doc_name = EXCLUDED.doc_name` in the upsert.
- `search_similar_chunks` → `doc_name` in SELECT and each result dict.
- `search_bm25` → same.
- `get_chunks_by_section` → same, so sibling-expansion chunks reach the context builder attributed.

### 4.4 `ChunkEmbeddingPipeline` — [pipeline.py](../../src/app/ingestion/embedding/pipeline.py)

- `embed_chunks(..., doc_name: Optional[str] = None)` → into each `Chunk(...)` at
  [pipeline.py:370](../../src/app/ingestion/embedding/pipeline.py#L370), and into
  `chunk_metadata` (cheap, and keeps the `data/chunks/*/index.json` dumps
  self-describing).
- `ingest_document(..., doc_name=None)` → forwards. Defaulted, so existing call
  sites are untouched.

### 4.5 Worker — [ingestion_tasks.py:260](../../src/app/worker/ingestion_tasks.py#L260)

```python
doc_name = doc.get("doc_name") or Path(doc["file_name"]).stem
metadata.update({..., "doc_name": doc_name, "domain": doc.get("domain")})
await pipeline.embed_chunks(..., doc_name=doc_name)
```

The filename-stem fallback covers pre-006 rows whose backfill hasn't run.

### 4.6 `IngestionRepository.register_document` — [ingestion_repository.py:35](../../src/app/infra/db/ingestion_repository.py#L35)

Add `doc_name: Optional[str] = None`, `domain: Optional[str] = None`; include both
in the INSERT. `get_document_status` already does `SELECT *`, so both appear once
added to the response dict at
[document_routes.py:143](../../src/app/api/routes/document_routes.py#L143).

### 4.7 Schemas — [schemas.py](../../src/app/models/schemas.py)

```python
class DomainInfo(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    table_name: str
    document_count: int = 0

class DomainDocument(BaseModel):
    document_id: str
    doc_name: Optional[str] = None
    file_name: str
    stage: str
    chunk_count: Optional[int] = None

class QueryRequest(BaseModel):
    ...
    domain: Optional[str] = None      # NEW — preferred over table_name
    doc_name: Optional[str] = None    # NEW — optional exact-match convenience filter
    # table_name stays for backward compatibility

class RAGSource(BaseModel):
    ...
    doc_name: Optional[str] = None    # NEW

class UploadResponse(BaseModel):
    ...
    doc_name: Optional[str] = None    # NEW
    domain: Optional[str] = None      # NEW
```

Resolution rule: if `domain` is set, look up `table_name` from the registry and
ignore any supplied `table_name`; otherwise use `table_name` exactly as today.

### 4.8 `perform_document_search` — [search.py](../../src/app/retrieval/search.py)

The load-bearing part is reranking. The block at
[search.py:123-132](../../src/app/retrieval/search.py#L123-L132) rebuilds result
dicts from `RerankResult` attributes, so **any field the reranker doesn't carry is
silently dropped**. It already keeps `original_by_id` for exactly this reason —
`doc_name` joins `bm25_score`/`rrf_score` there:

```python
'doc_name': original_by_id[r.chunk_id].get('doc_name'),
```

Then `doc_name=r.get('doc_name')` on the `RAGSource(...)` at
[search.py:265](../../src/app/retrieval/search.py#L265).

Also label the context blocks so the LLM can attribute in prose ("According to
*Linear Algebra*, …") rather than only the UI showing provenance:

```python
f"[Source {i+1} — {doc_name}{page_info}]: {chunk_text}"
```

### 4.9 Routes

**New `src/app/api/routes/domain_routes.py`**, mounted in `api/app.py`:

| Method | Path | Returns |
|---|---|---|
| `GET` | `/domains` | `{"domains": [DomainInfo], "total": n}` |
| `POST` | `/domains` | Create; body `{name, display_name?, description?}`; password-protected |
| `GET` | `/domains/{name}` | One `DomainInfo` |
| `GET` | `/domains/{name}/documents` | `{"domain": ..., "documents": [DomainDocument]}` |
| `DELETE` | `/domains/{name}` | Drops chunk table + status rows + registry row; password-protected |

`DELETE /domains/{name}` must reuse the body of `DELETE /table/{table_name}`
([table_routes.py:58](../../src/app/api/routes/table_routes.py#L58)) — extract the
drop → status-cleanup → pipeline-evict sequence into a shared helper rather than
duplicating it. `/tables` stays for backward compatibility.

**`POST /upload`** ([document_routes.py:24](../../src/app/api/routes/document_routes.py#L24)):

```python
doc_name: str = Form(""),
domain:   str = Form(""),
```

- `domain = domain.strip() or table_name` → `validate_table_name(domain)` →
  `ensure_domain(domain)` → `table_name = row["table_name"]`
- `doc_name = doc_name.strip() or Path(safe_filename).stem`
- Both into `register_document`; both echoed in `UploadResponse`.

**`POST /query`** ([query_routes.py:72](../../src/app/api/routes/query_routes.py#L72)):
resolve `domain` → `table_name` before `validate_table_name`; thread
`document_ids` and `doc_name` through `_execute_traced_search`.

**`POST /query-form`** ([query_routes.py:111](../../src/app/api/routes/query_routes.py#L111)):
add `domain`/`doc_name` form fields and `"doc_name": source.doc_name or "Unknown"`
in the sources dict at
[query_routes.py:147-159](../../src/app/api/routes/query_routes.py#L147-L159).
Note this endpoint currently hardcodes `document_ids=None`.

---

## 5. UI Changes

### 5.1 Upload tab — [home.html:359-386](../../src/app/api/templates/home.html#L359-L386)

Replace the free-text `Table Name` input with a domain picker, and add the name field:

```html
<div class="form-group">
    <label>Document Name</label>
    <input type="text" name="doc_name" placeholder="e.g. Linear Algebra (defaults to the filename)">
</div>
<div class="form-group">
    <label>Domain</label>
    <select name="domain" id="upload-domain" class="model-select" onchange="toggleNewDomain(this)">
        <!-- populated from GET /domains -->
        <option value="__new__">➕ New domain…</option>
    </select>
    <input type="text" id="upload-new-domain" name="new_domain" style="display:none"
           placeholder="lowercase_with_underscores">
</div>
```

`__new__` reveals the text input; the submit handler sends whichever is active as
`domain`. Keep a hidden `table_name` mirroring it so older form posts still work.

### 5.2 Chat settings — [home.html:336-341](../../src/app/api/templates/home.html#L336-L341)

Replace `Table:` with `Domain:` (same control, fed by `/domains`, showing
`display_name` with `name` as the value) and add a document filter below it:

```html
<label>Domain:
    <select id="chat-domain" class="model-select" onchange="loadDomainDocuments()"></select>
    <button type="button" onclick="loadDomainList()" title="Refresh">↻</button>
</label>
<label>Document:
    <select id="chat-document" class="model-select">
        <option value="">All documents in domain</option>
    </select>
</label>
```

- `loadDomainList()` replaces `loadTableList()` ([home.html:698](../../src/app/api/templates/home.html#L698)), hitting `/domains`.
- `loadDomainDocuments()` fetches `/domains/{name}/documents` and fills the second
  select with `doc_name` (value = `document_id`). Documents with `stage != 'embedded'`
  render disabled, labelled `— (parsing)`.
- `sendChat()` ([home.html:592](../../src/app/api/templates/home.html#L592)) sends
  `domain`, plus `document_ids: [id]` when one is selected.

### 5.3 Source rendering in chat — [home.html:638-649](../../src/app/api/templates/home.html#L638-L649)

Currently `Doc: ${escapeHtml(docId)}` with an 8-char UUID prefix. Lead with the
name, keep the id on hover:

```js
const docLabel = s.doc_name
    ? escapeHtml(s.doc_name)
    : (s.document_id ? escapeHtml(s.document_id.substring(0, 8)) + '…' : 'N/A');
const docTitle = s.document_id ? ` title="${escapeHtml(s.document_id)}"` : '';
return `<div class="source-item">
    <div class="source-meta">Source ${i + 1} &mdash; ${sim}% similarity${bm25}${rrf}${rerank}
        &nbsp;|&nbsp; <span class="source-doc"${docTitle}>📄 ${docLabel}</span>${page}</div>
    <div class="source-text">${escapeHtml(s.text)}</div>
</div>`;
```

Add `.source-doc { font-weight: 600; }` so the name is what the eye lands on.

### 5.4 Legacy results template — [search_results.html:107](../../src/app/api/templates/search_results.html#L107)

```html
<em>📄 {{ source.doc_name }} | Document: {{ source.document_id }}… | Page: {{ source.page_number }}</em>
```

### 5.5 Upload status polling — [home.html:465](../../src/app/api/templates/home.html#L465)

`pollDocumentStatus` shows the filename; switch to `doc_name` (falling back to
filename) once `/documents/{id}/status` returns it.

---

## 6. End-to-End Flow

```
Upload  "Linear_Algebra_v3.pdf",  doc_name="Linear Algebra",  domain="math"
  → ensure_domain('math')                domains:   name=math, table_name=math
  → register_document(...)               documents: doc_name='Linear Algebra',
                                                    domain='math',
                                                    target_table_name='math'
  → parse → chunk → embed
  → _embed_document reads doc["doc_name"]
  → embed_chunks(doc_name=...) → Chunk(doc_name=...)
  → INSERT INTO math (..., doc_name) VALUES (..., 'Linear Algebra')   × 847 rows

Query   "What is a vector?",  domain="math",  document_ids=["abc-123"] (optional)
  → resolve domain → table_name='math'
  → search_similar_chunks → SELECT id, text, metadata, document_id, doc_name, similarity
  → (+ BM25 / RRF in hybrid mode, both carrying doc_name)
  → rerank → doc_name restored from original_by_id
  → top-k RAGSource[], each with doc_name
  → UI: "Source 1 — 87.2% similarity | 📄 Linear Algebra · Page 12"
```

---

## 7. Migration & Backward Compatibility

### 7.1 Applying on an existing volume

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < deploy/migrations/006_domains_and_doc_name.sql
docker compose restart app celery_worker_upload celery_worker_ingestion
```

### 7.2 Optional chunk-row backfill (per domain, opt-in)

```sql
UPDATE math m
SET doc_name = d.doc_name
FROM documents d
WHERE m.document_id = d.id AND m.doc_name IS NULL;
```

Rewrites every row, so it stays out of the migration. Until it runs, pre-006
chunks return `doc_name: null` and the UI falls back to the truncated UUID — which
is today's behaviour, so nothing regresses.

### 7.3 Renaming a document

`documents.doc_name` is authoritative; chunk `doc_name` is a snapshot. A rename
endpoint (out of scope) must `UPDATE documents` **and** the chunk table or the two
drift. The column COMMENT in §3.1 points here so the next person hits it in the
schema, not in production.

### 7.4 Contract compatibility

| Existing behaviour | After |
|---|---|
| `POST /upload` without `doc_name`/`domain` | `doc_name` = filename stem, `domain` = `table_name` |
| `POST /query` with `table_name`, no `domain` | Unchanged path, no registry lookup |
| `RAGSource` consumers | `doc_name` is an added optional field |
| `GET /tables`, `DELETE /table/{name}` | Unchanged; `/domains` is additive |
| Chunk tables created before 006 | Self-heal via §3.3 `ADD COLUMN IF NOT EXISTS` |

---

## 8. Edge Cases

| Case | Handling |
|---|---|
| Domain name that isn't a valid SQL identifier | Rejected by `validate_table_name` and the `domains_name_is_identifier` CHECK |
| Domain named `domains` / `documents` / `llm_interactions` | Rejected by the §3.2 denylist |
| Upload into a domain that doesn't exist | `ensure_domain` creates it — matches today's implicit table creation |
| Same book uploaded twice into one domain | Two `documents` rows, two UUIDs, same `doc_name`, both sets of chunks in the table. Allowed — this is why `document_ids` is the filter |
| `DELETE /domains/{name}` with documents present | Drops table + status rows + registry row, same as `DELETE /table/{name}` today |
| Chunk table exists with no registry row | Backfilled by 006; `list_domains` should also reconcile on read so a hand-created table still appears |
| Registry row whose table was dropped out of band | `/domains/{name}/documents` still works (reads `documents`); the query path fails on the missing table, as today |
| Document mid-ingest | Listed with its stage; disabled in the UI picker |
| `doc_name` NULL in a response | `null` in JSON, UUID prefix in the chat UI, "Unknown" in the legacy HTML |

---

## 9. File-by-File Checklist

| # | File | Change |
|---|---|---|
| 1 | `deploy/migrations/006_domains_and_doc_name.sql` | **New** — registry, columns, indexes, backfill, FK |
| 2 | `src/app/infra/db/domain_repository.py` | **New** — `DomainRepository` |
| 3 | `src/app/infra/db/__init__.py` | Export `DomainRepository` |
| 4 | `src/app/infra/db/identifiers.py` | Make `_SYSTEM_TABLES` load-bearing; add reserved names (§3.2) |
| 5 | `src/app/api/routes/domain_routes.py` | **New** — 5 endpoints |
| 6 | `src/app/api/app.py` | Mount `domain_routes.router` |
| 7 | `src/app/ingestion/embedding/chunk.py` | `doc_name` field |
| 8 | `src/app/ingestion/embedding/vector_store.py` | CREATE/ALTER/index; INSERT; `doc_name` in 3 query methods |
| 9 | `src/app/ingestion/embedding/pipeline.py` | `doc_name` through `embed_chunks` / `ingest_document` |
| 10 | `src/app/worker/ingestion_tasks.py` | Read `doc_name`/`domain` from the row; pass to `embed_chunks` |
| 11 | `src/app/infra/db/ingestion_repository.py` | `doc_name` + `domain` in `register_document` |
| 12 | `src/app/api/routes/document_routes.py` | Form fields, `ensure_domain`, status response |
| 13 | `src/app/api/routes/table_routes.py` | Extract the drop sequence into a helper shared with `DELETE /domains/{name}` |
| 14 | `src/app/models/schemas.py` | `DomainInfo`, `DomainDocument`, `QueryRequest`, `RAGSource`, `UploadResponse` |
| 15 | `src/app/retrieval/search.py` | Preserve `doc_name` through rerank; set on `RAGSource`; context label |
| 16 | `src/app/api/routes/query_routes.py` | Resolve `domain`; thread `document_ids`/`doc_name`; HTML source dict |
| 17 | `src/app/api/templates/home.html` | Upload fields; chat domain/document selects; source rendering; `loadDomainList` |
| 18 | `src/app/api/templates/search_results.html` | Show `doc_name` |
| 19 | `docs/ARCHITECTURE.md` | §4.1 columns, §6.2 repositories, §6.4 migrations, §8 routers, §12 glossary entry for *domain* |

**Estimate**: ~450 lines across 19 files (2 new modules, 1 new migration).

---

## 10. Testing

**Unit** — `tests/unit/test_domains_and_doc_name.py`, no DB:

- `Chunk(doc_name=...)` set, and defaulting to `None`.
- `RAGSource(doc_name=...)`; `RAGSource` without it still validates.
- `QueryRequest` accepts `domain`, `doc_name`, and neither.
- `validate_table_name('domains')` raises — regression guard for §3.2.
- Domain slug normalization; `My Domain!` rejected.
- **Rerank preserves `doc_name`** — fake reranker returning `RerankResult`s that
  carry no `doc_name`; assert the rebuilt dicts still have it. This is the test
  that will actually catch a regression: it's the same bug shape that already
  forced `original_by_id` to exist for `bm25_score` and `rrf_score`.

**Integration** — `tests/integration/test_domains_integration.py`, needs Postgres:

- `POST /domains` → `GET /domains` shows it, `document_count: 0`.
- Upload with `doc_name` + `domain` → chunk rows carry `doc_name`; `/documents/{id}/status` returns both.
- Upload with neither → `doc_name` = filename stem, domain = table name.
- Two books into one domain → `/domains/{name}/documents` lists both; chunk table holds both sets.
- `POST /query` with `domain` → every source has the correct `doc_name`.
- `POST /query` with `document_ids` → all sources come from that document.
- `POST /query` with legacy `table_name` only → behaviour unchanged.
- `DELETE /domains/{name}` → table, `documents` rows, and registry row all gone.
- Migration over a pre-006 table with rows → column added, domain row created, existing chunks still readable with `doc_name IS NULL`.

[ARCHITECTURE.md §10.3](../ARCHITECTURE.md#103-pre-existing-test-failures) records 12
pre-existing unit failures. Baseline before and after; the count must be unchanged.

---

## 11. Approval Criteria

- [ ] Migration runs clean on a fresh volume and on an existing one with data
- [ ] Existing chunk tables appear as domains after migration; none disappear
- [ ] Upload with `doc_name` + `domain` stores both in `documents` and `doc_name` on every chunk row
- [ ] Upload with neither still works (filename stem, domain = table name)
- [ ] `GET /domains` lists domains with document counts
- [ ] `GET /domains/{name}/documents` lists documents with `doc_name` and stage
- [ ] **Top-k results carry `doc_name` in `RAGSource` — in vector mode, hybrid mode, and after reranking**
- [ ] Chat UI shows the document name on every source instead of a UUID prefix
- [ ] Chat UI can scope a query to one document within a domain
- [ ] Upload form has a document-name field and a domain picker with "new domain"
- [ ] `POST /query` with only `table_name` behaves exactly as before
- [ ] `tests/unit` failure count unchanged from baseline; new tests pass

---

## 12. Out of Scope

- Renaming a document or domain (§7.3 — needs a coordinated two-table update)
- Cross-domain search (one query spanning several chunk tables)
- `doc_name` autocomplete / fuzzy matching
- Per-domain embedding models or chunk sizes
- Migrating `/tables` consumers off the endpoint — `/domains` is additive for now

---

**End of Plan**
