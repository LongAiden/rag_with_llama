# Plan: `doc_name` Column for Chunk Tables

**Date**: 2026-08-11  
**Status**: Proposed  
**Scope**: Ingestion pipeline, query pipeline, database schema, API routes, response models

---

## 1. Problem Statement

Currently, chunk tables (e.g., `math`, `history`, `technical`) store chunks from multiple documents, but there is no efficient way to:
- Identify which document a chunk belongs to (beyond the opaque `document_id` UUID)
- Filter search results to a specific document within a table
- Display human-readable document names in query results
- **Return the document name alongside retrieved chunks in API responses**

The `documents` status table tracks ingestion metadata, but the chunk tables themselves lack a direct, queryable reference to the document's human-readable name.

---

## 2. Current Architecture

### 2.1 Chunk Table Schema (per table, e.g., `math`)

```sql
CREATE TABLE math (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

- `document_id`: UUID linking to `documents.id`
- `metadata`: JSONB containing `filename`, `page_number`, `section_path`, etc.
- No dedicated column for document name

### 2.2 Documents Status Table

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    raw_storage_path TEXT NOT NULL,
    target_table_name TEXT,
    metadata JSONB DEFAULT '{}',
    ...
);
```

- `file_name`: The uploaded filename (e.g., `Linear_Algebra.pdf`)
- `metadata`: JSONB with upload metadata
- No `doc_name` field

### 2.3 Upload Flow

```
POST /upload
  file: UploadFile
  table_name: str (e.g., "math")
  chunk_size: int
  parse_backend: str
  → No doc_name parameter
```

### 2.4 Query Flow

```
POST /query
  query: str
  table_name: str
  document_ids: Optional[List[str]]
  → No doc_name filter
  → Response sources don't include doc_name
```

### 2.5 Response Model

```python
class RAGSource(BaseModel):
    chunk_id: str
    text: str
    similarity: float
    document_id: str
    page_number: Optional[int]
    metadata: Dict[str, Any]
    # No doc_name field
```

---

## 3. Proposed Changes

### 3.1 Add `doc_name` Column to Chunk Tables

Each chunk table gains a `doc_name TEXT` column:

```sql
ALTER TABLE math ADD COLUMN IF NOT EXISTS doc_name TEXT;
CREATE INDEX IF NOT EXISTS math_doc_name_idx ON math (doc_name);
```

- **Type**: `TEXT` (nullable for backward compatibility)
- **Index**: B-tree for fast equality filtering
- **Default**: `NULL` (existing chunks remain unchanged)

### 3.2 Add `doc_name` Column to Documents Table

```sql
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_name TEXT;
```

- Stores the user-provided document name at upload time
- Defaults to `file_name` if not provided (backward compatibility)

### 3.3 Upload Endpoint: Accept `doc_name`

```python
@router.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    table_name: str = Form("document_chunks"),
    doc_name: str = Form(""),  # NEW
    ...
):
    doc_name = doc_name.strip() or Path(file.filename).stem
    await repo.register_document(
        doc_id=document_id,
        file_name=safe_filename,
        doc_name=doc_name,  # NEW
        ...
    )
```

- `doc_name` is optional; defaults to filename without extension
- Stored in `documents.doc_name` and propagated to chunk tables

### 3.4 Ingestion Pipeline: Propagate `doc_name`

#### 3.4.1 `Chunk` Dataclass

```python
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict] = None
    doc_name: Optional[str] = None  # NEW
```

#### 3.4.2 `VectorStore.add_chunks()`

```python
async def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
    chunk_data = []
    for chunk in chunks:
        chunk_data.append((
            chunk.id,
            chunk.document_id,
            chunk.text,
            embedding_str,
            chunk.metadata if chunk.metadata else {},
            chunk.doc_name,  # NEW
        ))

    insert_sql = f"""
    INSERT INTO {self.safe_table_name} 
        (id, document_id, text, embedding, metadata, doc_name)
    VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6)
    ON CONFLICT (id) DO UPDATE SET
        document_id = EXCLUDED.document_id,
        text        = EXCLUDED.text,
        embedding   = EXCLUDED.embedding,
        metadata    = EXCLUDED.metadata,
        doc_name    = EXCLUDED.doc_name;
    """
```

#### 3.4.3 `ChunkEmbeddingPipeline.embed_chunks()`

```python
async def embed_chunks(
    self,
    chunks: List[Any],
    document_id: str,
    doc_name: str,  # NEW
    ...
):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_objects.append(Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            text=chunk.text,
            embedding=embedding,
            metadata=chunk_metadata,
            doc_name=doc_name,  # NEW
        ))
```

#### 3.4.4 Worker: `_embed_document()`

```python
async def _embed_document(doc_id: str) -> Dict[str, Any]:
    async def work(repo, config, doc):
        doc_name = doc.get("doc_name") or doc["file_name"]  # NEW
        await pipeline.embed_chunks(
            chunks=chunks,
            document_id=doc_id,
            doc_name=doc_name,  # NEW
            ...
        )
```

### 3.5 Query Pipeline: Filter by `doc_name` and Return in Results

#### 3.5.1 `QueryRequest` Schema

```python
class QueryRequest(BaseModel):
    query: str
    table_name: str
    document_ids: Optional[List[str]] = None
    doc_name: Optional[str] = None  # NEW
    ...
```

#### 3.5.2 `VectorStore.search_similar_chunks()`

```python
async def search_similar_chunks(
    self,
    query_embedding: List[float],
    limit: int = 5,
    threshold: float = 0.7,
    document_ids: Optional[List[str]] = None,
    doc_name: Optional[str] = None,  # NEW
) -> List[Dict]:
    base_query = f"""
        SELECT id, text, metadata, document_id, doc_name,
               (1 - (embedding <=> $1::vector)) as similarity
        FROM {self.safe_table_name}
        WHERE (1 - (embedding <=> $1::vector)) >= $2
    """
    params = [query_embedding_str, threshold]

    if document_ids:
        base_query += " AND document_id = ANY($3)"
        params.append(document_ids)

    if doc_name:  # NEW
        param_idx = len(params) + 1
        base_query += f" AND doc_name = ${param_idx}"
        params.append(doc_name)

    base_query += f" ORDER BY embedding <=> $1 LIMIT ${len(params) + 1}"
    params.append(limit)

    # Include doc_name in returned dict
    return [
        {
            'chunk_id': row['id'],
            'text': row['text'],
            'metadata': row['metadata'],
            'document_id': row['document_id'],
            'doc_name': row['doc_name'],  # NEW
            'similarity': float(row['similarity'])
        }
        for row in rows
    ]
```

#### 3.5.3 `VectorStore.search_bm25()`

```python
async def search_bm25(
    self,
    query: str,
    limit: int = 20,
    document_ids: Optional[List[str]] = None,
    doc_name: Optional[str] = None,  # NEW
) -> List[Dict]:
    base_query = f"""
        SELECT id, text, metadata, document_id, doc_name
        FROM {self.safe_table_name}
    """
    conditions = []
    params = []

    if document_ids:
        conditions.append(f"document_id = ANY(${len(params) + 1})")
        params.append(document_ids)

    if doc_name:  # NEW
        conditions.append(f"doc_name = ${len(params) + 1}")
        params.append(doc_name)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    # Include doc_name in returned dict
    results.append({
        'chunk_id': row['id'],
        'text': row['text'],
        'metadata': row['metadata'],
        'document_id': row['document_id'],
        'doc_name': row['doc_name'],  # NEW
        'bm25_score': float(bm25_scores[idx]),
    })
```

#### 3.5.4 `RAGSource` Schema: Add `doc_name` Field

```python
class RAGSource(BaseModel):
    """Information about a source used in RAG response."""
    chunk_id: str = Field(description="Unique identifier for the source chunk")
    text: str = Field(description="Text content of the chunk")
    similarity: float = Field(ge=0, le=1, description="Similarity score to query")
    document_id: str = Field(description="Document this chunk comes from")
    doc_name: Optional[str] = Field(  # NEW
        None, description="Human-readable document name"
    )
    page_number: Optional[int] = Field(
        None, description="Page number where this chunk appears"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rerank_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    graph_entities: List[Dict[str, Any]] = Field(default_factory=list)
```

#### 3.5.5 `perform_document_search()`: Pass `doc_name` and Include in Response

```python
async def perform_document_search(
    query: str,
    ...
    doc_name: Optional[str] = None,  # NEW
) -> RAGResponse:
    # Pass doc_name to vector search
    vector_results = await pipeline.search_documents(
        query=query,
        ...
        doc_name=doc_name,  # NEW
    )

    # Pass doc_name to BM25 search
    if search_mode == "hybrid":
        bm25_results = await pipeline.vector_store.search_bm25(
            query=query,
            ...
            doc_name=doc_name,  # NEW
        )

    # Preserve doc_name through reranking
    if enable_reranking and merged_results:
        original_by_id = {r['chunk_id']: r for r in merged_results}
        reranked = await asyncio.to_thread(reranker.rerank, ...)
        merged_results = [{
            'chunk_id': r.chunk_id,
            'text': r.text,
            'document_id': r.document_id,
            'doc_name': original_by_id[r.chunk_id].get('doc_name'),  # NEW
            'metadata': r.metadata,
            'similarity': r.similarity,
            ...
        } for r in reranked]

    # Include doc_name in RAGSource
    return RAGResponse(
        query=query,
        answer=llm_response.answer,
        sources=[
            RAGSource(
                chunk_id=r['chunk_id'],
                text=r['text'],
                similarity=round(r['similarity'], 3),
                document_id=r['document_id'],
                doc_name=r.get('doc_name'),  # NEW
                page_number=r.get('metadata', {}).get('page_number'),
                metadata=r.get('metadata', {}),
                ...
            ) for r in results
        ],
        ...
    )
```

### 3.6 API Routes: Accept `doc_name` and Return in Responses

#### 3.6.1 `POST /query`

```python
@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: QueryRequest,
    ...
):
    result = await _execute_traced_search(
        ...
        doc_name=request.doc_name,  # NEW
    )
    # RAGResponse automatically includes doc_name in sources
    return result
```

**Example JSON Response:**

```json
{
  "query": "What is a vector?",
  "answer": "A vector is a mathematical object...",
  "sources": [
    {
      "chunk_id": "abc-123",
      "text": "Vectors are mathematical objects...",
      "similarity": 0.87,
      "document_id": "def-456",
      "doc_name": "Linear Algebra",
      "page_number": 12,
      "metadata": { ... }
    }
  ],
  "search_stats": { ... }
}
```

#### 3.6.2 `POST /query-form`

```python
@router.post("/query-form", response_class=HTMLResponse)
async def query_documents_form(
    query: str = Form(...),
    doc_name: str = Form(""),  # NEW
    ...
):
    result = await _execute_traced_search(
        ...
        doc_name=doc_name.strip() or None,  # NEW
    )

    # Include doc_name in HTML template context
    sources = [
        {
            "index": i + 1,
            "similarity": source.similarity,
            "document_id": source.document_id[:8] if source.document_id else "Unknown",
            "doc_name": source.doc_name or "Unknown",  # NEW
            "page_number": source.page_number or 'N/A',
            "text": source.text,
            ...
        }
        for i, source in enumerate(result.sources)
    ]

    return render("search_results.html", sources=sources, ...)
```

### 3.7 UI: Add `doc_name` Fields and Display

#### 3.7.1 Upload Form (`home.html`)

```html
<div class="form-group">
  <label for="doc_name">Document Name (optional):</label>
  <input type="text" id="doc_name" name="doc_name" 
         placeholder="e.g., Linear Algebra">
</div>
```

#### 3.7.2 Search Form (`home.html`)

```html
<div class="form-group">
  <label for="doc_name">Filter by Document (optional):</label>
  <input type="text" id="doc_name" name="doc_name" 
         placeholder="e.g., Linear Algebra">
</div>
```

#### 3.7.3 Search Results Template (`search_results.html`)

```html
<div class="source-card">
  <div class="source-header">
    <span class="source-index">Source {{ source.index }}</span>
    <span class="source-doc-name">{{ source.doc_name }}</span>
    <span class="source-page">Page {{ source.page_number }}</span>
  </div>
  <div class="source-text">{{ source.text }}</div>
  <div class="source-meta">
    Similarity: {{ source.similarity }}
  </div>
</div>
```

---

## 4. Migration Strategy

### 4.1 New Migration File

**File**: `deploy/migrations/006_add_doc_name.sql`

```sql
-- Migration 006: Add doc_name column for human-readable document identification

ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_name TEXT;

COMMENT ON COLUMN documents.doc_name IS 
  'Human-readable document name provided by user at upload time; defaults to filename if not specified';

-- Add doc_name to all existing chunk tables
DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN
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
          AND t1.table_name NOT IN ('entities', 'relationships', 'entity_nodes', 'entity_edges')
    LOOP
        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS doc_name TEXT', table_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (doc_name)', 
                       table_name || '_doc_name_idx', table_name);
    END LOOP;
END $$;
```

### 4.2 Backfill Existing Data (Optional)

For existing chunks, `doc_name` will be `NULL`. Optionally backfill from `documents.file_name`:

```sql
UPDATE math m
SET doc_name = d.file_name
FROM documents d
WHERE m.document_id = d.id
  AND m.doc_name IS NULL;
```

**Note**: This is optional and can be done manually per table if needed.

### 4.3 Lazy Schema Evolution

`VectorStore._initialize_database()` must handle existing tables:

```python
async def _initialize_database(self):
    async with self.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.safe_table_name} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector(384),
            metadata JSONB,
            entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
            doc_name TEXT,  -- NEW
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Backward compatibility: add column if missing
        await conn.execute(f"""
        ALTER TABLE {self.safe_table_name} 
        ADD COLUMN IF NOT EXISTS doc_name TEXT;
        """)
        
        # Create indexes
        await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {embedding_idx}
        ON {self.safe_table_name} USING hnsw (embedding vector_cosine_ops);
        """)
        
        await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {document_id_idx}
        ON {self.safe_table_name} (document_id);
        """)
        
        # NEW: doc_name index
        doc_name_idx = f'"{self.table_name}_doc_name_idx"'
        await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {doc_name_idx}
        ON {self.safe_table_name} (doc_name);
        """)
```

---

## 5. Data Flow

### 5.1 Upload Flow

```
User uploads "Linear_Algebra.pdf" with doc_name="Linear Algebra"
  ↓
POST /upload
  file: Linear_Algebra.pdf
  table_name: "math"
  doc_name: "Linear Algebra"
  ↓
IngestionRepository.register_document(
  doc_id="abc-123",
  file_name="Linear_Algebra.pdf",
  doc_name="Linear Algebra",
  target_table_name="math"
)
  ↓
documents table:
  id: "abc-123"
  file_name: "Linear_Algebra.pdf"
  doc_name: "Linear Algebra"
  target_table_name: "math"
  ↓
Celery task chain: parse → chunk → embed
  ↓
_embed_document(doc_id="abc-123")
  reads doc["doc_name"] = "Linear Algebra"
  ↓
ChunkEmbeddingPipeline.embed_chunks(
  doc_name="Linear Algebra"
)
  ↓
Chunk objects created with doc_name="Linear Algebra"
  ↓
VectorStore.add_chunks()
  INSERT INTO math (..., doc_name) VALUES (..., 'Linear Algebra')
  ↓
math table:
  id: "chunk-456"
  document_id: "abc-123"
  text: "Vectors are..."
  embedding: [0.1, 0.2, ...]
  doc_name: "Linear Algebra"
```

### 5.2 Query Flow

```
User queries: "What is a vector?" with doc_name="Linear Algebra"
  ↓
POST /query
  query: "What is a vector?"
  table_name: "math"
  doc_name: "Linear Algebra"
  ↓
perform_document_search(
  doc_name="Linear Algebra"
)
  ↓
VectorStore.search_similar_chunks(
  doc_name="Linear Algebra"
)
  ↓
SQL:
  SELECT id, text, metadata, document_id, doc_name,
         (1 - (embedding <=> $1::vector)) as similarity
  FROM math
  WHERE (1 - (embedding <=> $1::vector)) >= $2
    AND doc_name = $3
  ORDER BY embedding <=> $1
  LIMIT $4
  ↓
Result dicts include doc_name:
  {
    'chunk_id': 'chunk-456',
    'text': 'Vectors are...',
    'document_id': 'abc-123',
    'doc_name': 'Linear Algebra',
    'similarity': 0.87
  }
  ↓
RAGSource built with doc_name:
  RAGSource(
    chunk_id='chunk-456',
    text='Vectors are...',
    document_id='abc-123',
    doc_name='Linear Algebra',
    similarity=0.87
  )
  ↓
RAGResponse serialized to JSON:
  {
    "sources": [
      {
        "chunk_id": "chunk-456",
        "text": "Vectors are...",
        "document_id": "abc-123",
        "doc_name": "Linear Algebra",
        "similarity": 0.87
      }
    ]
  }
```

---

## 6. Backward Compatibility

### 6.1 Existing Chunk Tables

- Migration adds `doc_name TEXT` column (nullable)
- Existing chunks have `doc_name = NULL`
- Queries without `doc_name` filter work as before (search all chunks)
- Queries with `doc_name` filter only match new chunks

### 6.2 Existing Uploads

- Old uploads without `doc_name` default to `file_name` (without extension)
- `IngestionRepository.register_document()` sets `doc_name = doc_name or file_name`

### 6.3 Existing Queries

- `QueryRequest.doc_name` is optional (`Optional[str] = None`)
- Omitting `doc_name` searches all documents in the table (current behavior)
- `RAGSource.doc_name` is optional; existing responses without it remain valid
- No breaking changes to existing API contracts

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File**: `tests/unit/test_doc_name_column.py`

```python
def test_chunk_dataclass_accepts_doc_name():
    chunk = Chunk(
        id="1",
        document_id="doc-1",
        text="test",
        embedding=[0.1],
        doc_name="Linear Algebra"
    )
    assert chunk.doc_name == "Linear Algebra"

def test_chunk_dataclass_doc_name_optional():
    chunk = Chunk(id="1", document_id="doc-1", text="test", embedding=[0.1])
    assert chunk.doc_name is None

def test_rag_source_includes_doc_name():
    source = RAGSource(
        chunk_id="1",
        text="test",
        similarity=0.9,
        document_id="doc-1",
        doc_name="Linear Algebra"
    )
    assert source.doc_name == "Linear Algebra"
```

### 7.2 Integration Tests

**File**: `tests/integration/test_doc_name_integration.py`

```python
async def test_upload_with_doc_name():
    response = await client.post("/upload", data={
        "table_name": "math",
        "doc_name": "Linear Algebra",
        ...
    })
    assert response.status_code == 200
    doc_id = response.json()["document_id"]
    
    # Wait for ingestion
    status = await get_document_status(doc_id)
    assert status["stage"] == "embedded"
    assert status["doc_name"] == "Linear Algebra"

async def test_query_filter_by_doc_name():
    # Upload two documents to same table
    await upload_doc("math", "Linear Algebra", "Vectors are...")
    await upload_doc("math", "Calculus", "Derivatives are...")
    
    # Query with filter
    response = await client.post("/query", json={
        "query": "What is a vector?",
        "table_name": "math",
        "doc_name": "Linear Algebra"
    })
    
    sources = response.json()["sources"]
    assert all(s["doc_name"] == "Linear Algebra" for s in sources)

async def test_query_without_doc_name_searches_all():
    response = await client.post("/query", json={
        "query": "What is mathematics?",
        "table_name": "math"
    })
    
    sources = response.json()["sources"]
    doc_names = {s["doc_name"] for s in sources}
    assert "Linear Algebra" in doc_names
    assert "Calculus" in doc_names

async def test_response_includes_doc_name():
    await upload_doc("math", "Linear Algebra", "Vectors are...")
    
    response = await client.post("/query", json={
        "query": "What is a vector?",
        "table_name": "math"
    })
    
    sources = response.json()["sources"]
    assert len(sources) > 0
    assert sources[0]["doc_name"] == "Linear Algebra"
```

### 7.3 Migration Test

```python
async def test_migration_adds_doc_name_column():
    # Create table without doc_name
    await conn.execute("""
        CREATE TABLE test_table (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector(384)
        )
    """)
    
    # Run migration
    await apply_migration("006_add_doc_name.sql")
    
    # Verify column exists
    result = await conn.fetchrow("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'test_table' AND column_name = 'doc_name'
    """)
    assert result is not None
```

---

## 8. Edge Cases and Risks

### 8.1 Edge Cases

| Case | Handling |
|------|----------|
| Empty `doc_name` in upload form | Default to `file_name` without extension |
| `doc_name` with special characters | Stored as-is (TEXT column); no sanitization needed |
| Duplicate `doc_name` in same table | Allowed; user can upload multiple versions of "Linear Algebra" |
| Query with non-existent `doc_name` | Returns empty results (no error) |
| Existing chunks with `doc_name = NULL` | Queries without filter include them; queries with filter exclude them |
| `doc_name` in response when NULL | Returns `null` in JSON, "Unknown" in HTML |

### 8.2 Risks

| Risk | Mitigation |
|------|------------|
| Migration fails on large tables | `ALTER TABLE ADD COLUMN` is metadata-only (fast) |
| Index creation blocks writes | `CREATE INDEX CONCURRENTLY` not needed for POC scale |
| Backfill query locks tables | Optional; can be done offline or skipped |
| `doc_name` typos in queries | User responsibility; no autocomplete in POC |
| Response size increase | `doc_name` is small (~50 bytes per source); negligible |

### 8.3 Performance Impact

- **Index size**: One B-tree index per chunk table on `doc_name` (minimal)
- **Query overhead**: Additional `WHERE` clause with index lookup (negligible)
- **Storage**: One TEXT column per chunk row (~50 bytes per row)
- **Response size**: One additional field per source (~50 bytes); negligible

---

## 9. File-by-File Implementation Checklist

| # | File | Change | Lines Affected |
|---|------|--------|----------------|
| 1 | `deploy/migrations/006_add_doc_name.sql` | **New file** | ~30 lines |
| 2 | `src/app/ingestion/embedding/chunk.py` | Add `doc_name` field | ~2 lines |
| 3 | `src/app/ingestion/embedding/vector_store.py` | Add column, ALTER, INSERT, search filters, return `doc_name` | ~50 lines |
| 4 | `src/app/ingestion/embedding/pipeline.py` | Accept `doc_name` in `embed_chunks()` and `ingest_document()` | ~10 lines |
| 5 | `src/app/worker/ingestion_tasks.py` | Read `doc_name` from document row | ~3 lines |
| 6 | `src/app/infra/db/ingestion_repository.py` | Accept `doc_name` in `register_document()` | ~5 lines |
| 7 | `src/app/api/routes/document_routes.py` | Add `doc_name` Form field | ~5 lines |
| 8 | `src/app/models/schemas.py` | Add `doc_name` to `QueryRequest` and `RAGSource` | ~5 lines |
| 9 | `src/app/retrieval/search.py` | Accept, pass, and return `doc_name` | ~15 lines |
| 10 | `src/app/api/routes/query_routes.py` | Accept `doc_name` in both endpoints, include in HTML | ~10 lines |
| 11 | `src/app/api/templates/home.html` | Add `doc_name` inputs | ~10 lines |
| 12 | `src/app/api/templates/search_results.html` | Display `doc_name` in results | ~5 lines |

**Total**: ~155 lines of changes across 12 files

---

## 10. Future Enhancements (Out of Scope)

- **Autocomplete**: Suggest `doc_name` values in search form (requires new endpoint)
- **Document listing**: `GET /tables/{name}/documents` to list all `doc_name` values
- **Bulk operations**: Delete all chunks for a `doc_name` (not just `document_id`)
- **Metadata enrichment**: Store `doc_name` in chunk `metadata` JSONB for redundancy
- **Multi-language**: Support `doc_name` in multiple languages
- **Fuzzy matching**: Allow partial `doc_name` matches in queries

---

## 11. Approval Criteria

- [ ] Migration runs successfully on fresh database
- [ ] Migration runs successfully on existing database with data
- [ ] Upload with `doc_name` stores value in `documents` and chunk tables
- [ ] Upload without `doc_name` defaults to filename
- [ ] Query with `doc_name` filter returns only matching chunks
- [ ] Query without `doc_name` filter searches all chunks (backward compat)
- [ ] **API response includes `doc_name` in sources**
- [ ] **HTML results display `doc_name` for each source**
- [ ] UI forms include `doc_name` fields
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] No breaking changes to existing API contracts

---

## 12. References

- **Architecture**: `docs/ARCHITECTURE.md` §4 (Ingestion Pipeline), §5 (Query Pipeline), §6 (Database Layer)
- **Migrations**: `deploy/migrations/003_create_ingestion_status.sql`, `004_ingestion_fixes.sql`
- **Vector Store**: `src/app/ingestion/embedding/vector_store.py`
- **Ingestion Tasks**: `src/app/worker/ingestion_tasks.py`
- **Response Models**: `src/app/models/schemas.py`

---

**End of Plan**
