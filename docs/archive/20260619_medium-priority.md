# Medium Priority Refactoring Items

This document describes the medium priority refactoring items that should be addressed after high priority items.

## 1. Break Up Large Functions

### Problem
Several functions exceed 100+ lines, making them hard to understand, test, and maintain:
- `vector_store.py:ingest_document()` (~250 lines)
- `document_routes.py:upload_and_process()` (~150 lines)
- `extraction_service.py:extract_from_chunks()` (~140 lines)

### Recommended Solution
Apply the Single Responsibility Principle by extracting smaller, focused functions:

#### For `ingest_document()`:
```python
async def ingest_document(self, ...):
    file_info = self._extract_file_info(file_path)
    chunks = await self._process_document_to_chunks(file_path, ...)
    valid_chunks = self._filter_valid_chunks(chunks)
    cleaned_texts = self._apply_text_cleaning(valid_chunks)
    embeddings = await self._generate_embeddings(cleaned_texts)
    await self._store_chunks_with_embeddings(valid_chunks, embeddings)
    return document_id
```

#### For `upload_and_process()`:
```python
async def upload_and_process(...):
    await self._validate_upload(file, chunk_size, ...)
    temp_path = await self._save_temp_file(file)
    try:
        result = await self._process_document(temp_path, ...)
        return self._build_success_response(result)
    finally:
        await self._cleanup_temp_file(temp_path)
```

### Benefits
- Easier to understand and test
- Better error handling
- Reusable components

## 2. Implement Proper Dependency Injection

### Problem
Current dependency management uses:
- Global state (`config` object)
- Manual parameter passing
- Circular import workarounds

### Recommended Solution
Use FastAPI's `Depends()` system properly:

```python
# Create dependency providers
async def get_db_pool() -> asyncpg.Pool:
    return await ConnectionPoolManager.get_pool(...)

async def get_table_repository(pool = Depends(get_db_pool)) -> TableRepository:
    return TableRepository(pool)

# Use in routes
@router.get("/tables")
async def list_tables(repo: TableRepository = Depends(get_table_repository)):
    tables = await repo.list_chunk_tables()
    return {"tables": tables}
```

### Benefits
- Clearer dependencies
- Easier testing with mocks
- No circular imports

## 3. Add Rate Limiting

### Problem
API endpoints have no rate limiting, which could lead to:
- API abuse
- Resource exhaustion
- Gemini API quota exhaustion

### Recommended Solution
Use `slowapi` for rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/upload")
@limiter.limit("10/minute")
async def upload_and_process(request: Request, ...):
    ...

@router.post("/query")
@limiter.limit("30/minute")
async def query_documents(request: Request, ...):
    ...
```

### Configuration
```python
# Different limits for different endpoints
UPLOAD_LIMIT = "10/minute"
QUERY_LIMIT = "30/minute"
STATS_LIMIT = "60/minute"
```

### Benefits
- Prevents API abuse
- Protects backend resources
- Ensures fair usage

## 4. Centralize Configuration

### Problem
Configuration is scattered across:
- `config/app_config.py`
- `config/graph_config.py`
- `.env` file
- Hardcoded defaults in function signatures

### Recommended Solution
Create a unified configuration hierarchy:

```python
# config/settings.py
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    # ...

class LLMSettings(BaseSettings):
    provider: str = "ollama"
    ollama_model: str = "deepseek-r1:8b"
    gemini_model: str = "gemini-2.5-flash"
    # ...

class AppSettings(BaseSettings):
    db: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    # ...

settings = AppSettings()
```

### Benefits
- Single source of truth
- Type-safe configuration
- Easier to test with different configs

## 5. Improve Error Handling Consistency

### Problem
Inconsistent error handling:
- Some endpoints raise `HTTPException`
- Others return error dicts
- Error messages expose internal details

### Recommended Solution
Create a centralized error handler:

```python
# api/errors.py
class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, public_message: str = None):
        self.message = message
        self.status_code = status_code
        self.public_message = public_message or "An internal error occurred"

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    logfire.error(f"AppError: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.public_message}
    )
```

### Usage
```python
@router.post("/upload")
async def upload_and_process(...):
    if not file:
        raise AppError("No file provided", 400, "No file was uploaded")
    
    try:
        result = await process_document(...)
    except ProcessingError as e:
        raise AppError(str(e), 500, "Document processing failed")
```

### Benefits
- Consistent error responses
- No internal details exposed
- Easier to log and monitor errors

## 6. Add Context Managers for Database Connections

### Problem
Manual connection management with try/finally blocks:
```python
conn = await pool.acquire()
try:
    # do work
finally:
    await pool.release(conn)
```

### Recommended Solution
Create async context managers:

```python
# repositories/connection_pool.py
class PoolConnection:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.conn: Optional[asyncpg.Connection] = None
    
    async def __aenter__(self) -> asyncpg.Connection:
        self.conn = await self.pool.acquire()
        return self.conn
    
    async def __aexit__(self, *args):
        if self.conn:
            await self.pool.release(self.conn)

# Usage
async with PoolConnection(pool) as conn:
    result = await conn.fetch("SELECT * FROM table")
```

### Benefits
- Cleaner code
- Automatic resource cleanup
- Less error-prone

## 7. Standardize Naming Conventions

### Problem
Inconsistent naming:
- Mix of snake_case and camelCase
- `chunk_id` vs `chunkId` in different parts
- Inconsistent API response formats

### Recommended Solution
Establish and enforce naming conventions:
- Python code: snake_case
- JSON API responses: camelCase
- Database columns: snake_case

Use Pydantic's `alias` for serialization:
```python
class ChunkResponse(BaseModel):
    chunk_id: str = Field(alias="chunkId")
    document_id: str = Field(alias="documentId")
    
    class Config:
        populate_by_name = True
```

### Benefits
- Consistent codebase
- Better API documentation
- Easier for frontend integration
