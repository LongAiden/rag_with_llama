# Common Docker Commands

Quick reference for day-to-day Docker operations in this project.

---

## Service Management

```bash
# Start all services
docker compose --profile observability up --build

# Start without optional services (no langfuse)
docker compose up --build

# Stop all services
docker compose down

# Restart specific services after code changes (~30s)
docker compose restart app celery_worker_upload celery_worker_ingestion celery_beat

# Full clean rebuild (deletes all data)
docker compose down -v
docker system prune -a
docker build -f deploy/deployment/Dockerfile.base -t rag-base:latest .
docker compose --profile observability up --build

# Start optional services
docker compose --profile dev up -d pgadmin        # DB admin UI on :5050
docker compose --profile observability up -d langfuse  # LLM observability on :3000
```

## Logs

```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f app
docker compose logs -f celery_worker_upload
docker compose logs -f celery_worker_ingestion

# Last N lines
docker compose logs --tail=100 celery_worker_upload

# Grep for specific patterns
docker compose logs celery_worker_upload | grep "parse_pdf summary"
docker compose logs celery_worker_upload | grep "Stage .* completed"
docker compose logs celery_worker_upload | grep "VLM call #"
```

## Ingestion Status

```bash
# Recent documents and their stages
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT id, file_name, stage, attempts, claimed_at, last_error FROM documents ORDER BY created_at DESC LIMIT 10;"

# Documents stuck in a processing stage (likely OOM-killed worker)
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT id, file_name, stage, attempts, claimed_at FROM documents WHERE stage IN ('parsing','chunking','embedding');"

# Failed / error documents
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT id, file_name, stage, attempts, last_error FROM documents WHERE stage IN ('error','failed');"

# Reset all stuck documents back to registered
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "UPDATE documents SET stage='registered', claimed_at=NULL, claimed_by=NULL, attempts=0 WHERE stage IN ('parsing','chunking','embedding');"

# Reset a specific document by ID
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "UPDATE documents SET stage='registered', claimed_at=NULL, claimed_by=NULL, attempts=0 WHERE id='<DOCUMENT_ID>';"

# Mark a failed document for retry (reset attempts)
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "UPDATE documents SET stage='registered', claimed_at=NULL, claimed_by=NULL, attempts=0 WHERE id='<DOCUMENT_ID>' AND stage='failed';"

# Delete a document and its chunks entirely
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "DELETE FROM documents WHERE id='<DOCUMENT_ID>';"
```

## Database Inspection

```bash
# List all tables
docker compose exec -T postgres psql -U admin -d rag_db -c "\dt"

# Count documents per stage
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT stage, COUNT(*) FROM documents GROUP BY stage ORDER BY stage;"

# Count chunks per domain
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT domain, COUNT(*) FROM documents WHERE stage='embedded' GROUP BY domain ORDER BY domain;"

# List domains with document counts
docker compose exec -T postgres psql -U admin -d rag_db \
  -c "SELECT d.name, d.display_name, COUNT(doc.id) AS docs FROM domains d LEFT JOIN documents doc ON doc.domain = d.name GROUP BY d.name, d.display_name ORDER BY d.display_name;"

# Open interactive psql shell
docker compose exec postgres psql -U admin -d rag_db

# Check Redis
docker compose exec redis redis-cli
docker compose exec redis redis-cli keys '*'
```

## Resource Monitoring

```bash
# Live resource usage (CPU, memory)
docker stats --format "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"

# Check for OOM kills
docker inspect rag_celery_worker_upload --format '{{.State.OOMKilled}}'
docker inspect rag_celery_worker_ingestion --format '{{.State.OOMKilled}}'

# Check container health
docker compose ps
```

## Testing

```bash
# All tests in Docker
docker compose --profile test run --rm test

# Unit tests only (no DB)
docker compose --profile test run --rm test pytest tests/unit -v

# Integration tests (requires Postgres)
docker compose --profile test run --rm test pytest tests/integration -v

# Specific test file
docker compose --profile test run --rm test pytest tests/unit/test_f23_structure_preservation.py -v

# With coverage
docker compose --profile test run --rm test pytest --cov=app --cov-report=html --cov-report=term

# Rebuild test image after code changes
docker compose --profile test build test
```

## Performance Profiling

```bash
# Capture a full parse run for analysis
docker compose logs -f celery_worker_upload | tee logs/parse_$(date +%Y%m%d_%H%M).log

# Extract key metrics from a parse run
grep "parse_pdf summary" logs/parse_*.log
grep "Converted pages" logs/parse_*.log
grep "Stage .* completed" logs/parse_*.log
grep -c "VLM call #" logs/parse_*.log

# Check Docker VM CPU count (should match docker-compose cpus limits)
docker info --format '{{.NCPU}}'
```

## Cleanup

```bash
# Remove stopped containers
docker compose rm -f

# Remove volumes (deletes all data)
docker compose down -v

# Remove all unused images, containers, volumes
docker system prune -a --volumes

# Remove specific image to force rebuild
docker rmi rag_with_llama:latest
```

## Shell Access

```bash
# Shell into the app container
docker compose exec app bash

# Shell into postgres
docker compose exec postgres bash

# Run a one-off command in the app container
docker compose exec app python -c "from app.config.app_config import AppSettings; print(AppSettings().pdf_parser_backend)"
```
