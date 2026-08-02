# Low Priority Refactoring Items

This document describes the low priority refactoring items that can be addressed when time permits.

## 1. Add Comprehensive Tests

### Problem
The `tests/` directory exists but has minimal test coverage. Critical functionality lacks unit and integration tests.

### Recommended Solution
Add tests for:
- **Unit tests**: Individual functions and classes
- **Integration tests**: API endpoints with test database
- **Mock tests**: External services (LLM providers, etc.)

### Test Structure
```
tests/
├── unit/
│   ├── test_table_repository.py
│   ├── test_entity_cache.py
│   ├── test_text_cleaning.py
│   └── test_chunking.py
├── integration/
│   ├── test_upload_endpoint.py
│   ├── test_query_endpoint.py
│   └── test_vector_store.py
└── conftest.py
```

### Example Test
```python
# tests/unit/test_entity_cache.py
import pytest
from graph_processing.entity_cache import EntityCache

def test_entity_cache_hit():
    EntityCache.configure(enabled=True, ttl_seconds=3600)
    text = "Test text for caching"
    entities = [{"name": "Test", "type": "CONCEPT"}]
    
    EntityCache.set(text, entities)
    result = EntityCache.get(text)
    
    assert result == entities

def test_entity_cache_miss():
    EntityCache.clear()
    result = EntityCache.get("Non-existent text")
    assert result is None
```

### Benefits
- Prevents regressions
- Documents expected behavior
- Enables safe refactoring

## 2. Improve Type Hints

### Problem
Many functions lack return type hints or use generic types like `Dict` instead of specific types.

### Recommended Solution
Add comprehensive type hints:

```python
# Before
async def extract_entities_from_chunk(self, chunk_id, chunk_text, confidence_threshold=0.6):
    ...

# After
async def extract_entities_from_chunk(
    self,
    chunk_id: UUID,
    chunk_text: str,
    confidence_threshold: float = 0.6
) -> List[Entity]:
    ...
```

### Use Specific Types
```python
# Instead of Dict[str, Any]
class Entity(BaseModel):
    entity_id: UUID
    name: str
    type: str
    confidence: float
    metadata: Dict[str, Any]

# Use the specific type
async def get_entity_by_id(entity_id: UUID) -> Optional[Entity]:
    ...
```

### Benefits
- Better IDE support
- Catch type errors early
- Self-documenting code

## 3. Remove Dead Code

### Problem
Codebase contains:
- Disabled graph routes (`graph_routes.py`)
- Deprecated functions (`chunk_text()`)
- Unused imports and variables

### Recommended Solution
1. **Audit disabled code**: Remove or clearly mark as experimental
2. **Remove deprecated functions**: After ensuring no callers
3. **Clean up imports**: Use tools like `autoflake`

### Commands
```bash
# Find unused imports
autoflake --check --remove-all-unused-imports -r .

# Find dead code
vulture . --min-confidence 80
```

### Benefits
- Cleaner codebase
- Faster imports
- Less confusion

## 4. Standardize Logging

### Problem
Inconsistent logging:
- Mix of `print()`, `logger.info()`, and `logfire.info()`
- No structured logging format
- Inconsistent log levels

### Recommended Solution
Standardize on `logfire` for all logging:

```python
# Create a logger module
# utils/logging.py
import logfire

def get_logger(name: str):
    return logfire.bind(name=name)

# Usage
logger = get_logger(__name__)
logger.info("Processing document", document_id=doc_id, file_size=size)
logger.error("Processing failed", error=str(e), exc_info=True)
```

### Remove print statements
```bash
# Find all print statements
grep -r "print(" --include="*.py" .

# Replace with proper logging
```

### Benefits
- Consistent log format
- Better observability
- Easier debugging

## 5. Add API Documentation

### Problem
API endpoints lack comprehensive documentation:
- No OpenAPI/Swagger descriptions
- Missing request/response examples
- No authentication documentation

### Recommended Solution
Add comprehensive FastAPI documentation:

```python
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and process a document",
    description="""
    Upload a document (PDF, DOCX, TXT) and process it into chunks.
    
    The document will be:
    1. Validated for type and size
    2. Converted to text/markdown
    3. Chunked into semantic segments
    4. Embedded and stored in vector database
    
    **Authentication**: Requires `x_app_password` header if configured.
    """,
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid file or parameters"},
        403: {"description": "Authentication failed"},
    }
)
async def upload_and_process(...):
    ...
```

### Benefits
- Better developer experience
- Auto-generated API docs
- Easier frontend integration

## 6. Optimize Database Queries

### Problem
Some queries are inefficient:
- N+1 query patterns in `graph_service.py`
- Missing indexes on frequently queried columns
- Unnecessary data fetching

### Recommended Solution
1. **Fix N+1 queries**: Use JOINs or batch queries
2. **Add indexes**: For frequently queried columns
3. **Use EXPLAIN ANALYZE**: To identify slow queries

### Example Fix
```python
# Before (N+1 queries)
for entity_id in entity_ids:
    entity = await conn.fetchrow("SELECT * FROM entities WHERE id = $1", entity_id)

# After (single query)
entities = await conn.fetch("SELECT * FROM entities WHERE id = ANY($1)", entity_ids)
```

### Benefits
- Faster query execution
- Reduced database load
- Better scalability

## 7. Add Performance Monitoring

### Problem
No visibility into:
- Query performance
- API response times
- Resource usage

### Recommended Solution
Add performance monitoring with `logfire`:

```python
# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logfire.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        duration_ms=process_time * 1000,
        status_code=response.status_code
    )
    return response
```

### Benefits
- Identify performance bottlenecks
- Monitor system health
- Data-driven optimization

## 8. Implement Graceful Shutdown

### Problem
Application doesn't handle shutdown gracefully:
- Database connections not closed
- Background tasks not stopped
- In-flight requests may be interrupted

### Recommended Solution
Add shutdown hooks:

```python
@app.on_event("shutdown")
async def shutdown_event():
    logfire.info("Shutting down application")
    
    # Close connection pools
    await ConnectionPoolManager.close_all()
    
    # Stop background tasks
    if celery_app:
        celery_app.control.shutdown()
    
    logfire.info("Shutdown complete")
```

### Benefits
- Clean resource cleanup
- No data loss
- Better reliability

## 9. Add Configuration Validation

### Problem
No validation that required environment variables are set at startup.

### Recommended Solution
Add startup validation:

```python
# config/validator.py
def validate_config():
    required_vars = ["POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise ConfigurationError(f"Missing required environment variables: {missing}")

# In app startup
@app.on_event("startup")
async def startup_event():
    validate_config()
    logfire.info("Configuration validated")
```

### Benefits
- Fail fast on misconfiguration
- Clear error messages
- Easier debugging

## 10. Document Architecture Decisions

### Problem
Architecture decisions are not documented, making it hard for new developers to understand:
- Why certain patterns were chosen
- Trade-offs made
- Constraints considered

### Recommended Solution
Create Architecture Decision Records (ADRs):

```markdown
# ADR-001: Use pgvector for Vector Storage

## Status
Accepted

## Context
We need to store and search document embeddings efficiently.

## Decision
Use pgvector extension for PostgreSQL.

## Consequences
- Pros:
  - Integrated with existing PostgreSQL database
  - Supports HNSW indexing for fast similarity search
  - ACID compliant
- Cons:
  - Limited to PostgreSQL ecosystem
  - May not scale as well as dedicated vector databases
```

### Benefits
- Knowledge preservation
- Easier onboarding
- Better decision making
