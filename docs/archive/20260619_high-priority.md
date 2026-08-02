# High Priority Refactoring Items

This document describes the high priority refactoring items that have been completed.

## 1. SQL Injection Risk Mitigation

### Problem
Multiple locations in the codebase used f-strings to construct SQL queries with table names, creating potential SQL injection vulnerabilities.

### Solution
Created a `TableRepository` in `repositories/table_repository.py` that provides:
- `validate_table_name()`: Validates table names against a safe pattern
- `quote_ident()`: Safely quotes table names using PostgreSQL double-quote identifier syntax
- Safe table operations (truncate, drop, delete, etc.)

### Files Modified
- `repositories/table_repository.py` (new)
- `repositories/__init__.py` (new)
- `api/routes/document_routes.py`
- `ingestion/embedding/vector_store.py`
- `graph_processing/extraction_service.py`

### Usage
```python
from repositories.table_repository import TableRepository, quote_ident

# Using the repository
repo = TableRepository(conn)
tables = await repo.list_chunk_tables()
await repo.truncate_table(table_name)

# Using safe quoting directly
safe_name = quote_ident(table_name)
await conn.execute(f"SELECT * FROM {safe_name}")
```

## 2. Duplicated Table Query Logic Extraction

### Problem
The same SQL query to find chunk tables was duplicated in 3 places:
- `get_table_count()`
- `list_tables()`
- `get_database_stats()`

### Solution
Extracted the common query into `TableRepository.list_chunk_tables()` method.

### Benefits
- Single source of truth for the query
- Easier to maintain and update
- Consistent behavior across endpoints

## 3. Connection Pooling

### Problem
`VectorStore._get_connection()` created a new database connection for each operation, leading to:
- Connection overhead
- Potential connection exhaustion under load
- No connection reuse

### Solution
Created `ConnectionPoolManager` in `repositories/connection_pool.py` that:
- Manages a pool of connections per connection string
- Provides thread-safe pool creation with locks
- Supports pool cleanup and shutdown

### Files Modified
- `repositories/connection_pool.py` (new)
- `ingestion/embedding/vector_store.py`

### Configuration
The pool uses default settings:
- `min_size`: 2 connections
- `max_size`: 10 connections

### Usage
```python
from repositories.connection_pool import ConnectionPoolManager

pool = await ConnectionPoolManager.get_pool(connection_string)
# Use pool.acquire() to get connections
# Connections are automatically returned to pool when released
```

## 4. Entity Extraction Caching

### Problem
Entity extraction was performed every time a chunk was processed, even if the same text had been extracted before. This led to:
- Redundant LLM API calls
- Increased latency
- Higher API costs

### Solution
Implemented `EntityCache` in `graph_processing/entity_cache.py` that:
- Caches extracted entities based on content hash (SHA-256)
- Supports TTL (time-to-live) for cache entries
- Can be enabled/disabled via configuration

### Files Modified
- `graph_processing/entity_cache.py` (new)
- `graph_processing/entity_extraction.py`

### Configuration
```python
from graph_processing.entity_cache import EntityCache

EntityCache.configure(enabled=True, ttl_seconds=3600)
```

### Benefits
- Reduces LLM API calls for repeated content
- Faster entity extraction for cached content
- Configurable cache TTL

## 5. Graph Processing - Ollama as Default LLM Provider

### Problem
Gemini API was the default for graph processing, but:
- Slow response times
- Rate limiting issues
- API costs

### Solution
- Added `llm_provider` configuration option in `GraphConfig`
- Made Ollama the default provider (local, fast, free)
- Gemini remains available as an option

### Configuration
Set in `.env`:
```bash
GRAPH_LLM_PROVIDER=ollama  # or "gemini"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b
```

### Files Modified
- `config/graph_config.py`
- `graph_processing/extraction_service.py`

### Benefits
- Faster entity/relationship extraction
- No API costs for local models
- No rate limiting issues
- Can still use Gemini for higher quality when needed
