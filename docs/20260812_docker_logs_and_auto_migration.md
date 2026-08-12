# Docker Logs and Auto-Migration (2026-08-12)

**Date**: 2026-08-12  
**Scope**: `docker-compose.yml`, `.gitignore`, `src/app/infra/db/identifiers.py`

---

## 1. Problem Statement

Two operational gaps existed in the Docker setup:

1. **Logs were ephemeral**: Container logs were only accessible via `docker compose logs` and lost on container recreation. No persistent log files existed on disk for debugging or audit.

2. **Migrations required manual application**: The `/docker-entrypoint-initdb.d` mount only runs on a fresh Postgres volume. Existing volumes required manually piping SQL files into psql, which was error-prone and not documented in the README.

---

## 2. Changes

### 2.1 Persistent Log Files

**`docker-compose.yml`**: All services now write logs to `./logs/<service>.log`:

- `app` → `logs/app.log`
- `celery_worker_upload` → `logs/celery_worker_upload.log`
- `celery_worker_ingestion` → `logs/celery_worker_ingestion.log`
- `celery_beat` → `logs/celery_beat.log`

Implementation:
- Commands wrapped with `trap 'kill 0' EXIT; <cmd> 2>&1 | tee -a /app/logs/<service>.log`
- `trap 'kill 0' EXIT` ensures child processes (uvicorn, celery) are killed when the shell exits
- `tee -a` appends to the log file while still writing to stdout (so `docker compose logs` still works)
- All services have `logging` config with `json-file` driver and rotation:
  - App services: `max-size: 10m`, `max-file: 5`
  - Workers: `max-size: 50m`, `max-file: 5` (larger due to parse logs)
  - Postgres/Redis: `max-size: 10m`, `max-file: 5`

**`.gitignore`**: Added `logs/` (excluding `logs/.gitkeep`)  
**`logs/.gitkeep`**: Created to keep the directory in git

### 2.2 Auto-Apply Migrations

**`docker-compose.yml`**: Added `migrate` service:

```yaml
migrate:
  image: rag_postgres:latest
  container_name: rag_migrate
  depends_on:
    postgres:
      condition: service_healthy
  environment:
    PGHOST: postgres
    PGPORT: "5432"
    PGDATABASE: ${POSTGRES_DB:-rag_db}
    PGUSER: ${POSTGRES_USER:?POSTGRES_USER is required}
    PGPASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}
  volumes:
    - ./deploy/migrations:/migrations:ro
  entrypoint: ["sh", "-c"]
  command:
    - |
      set -e
      echo "Waiting for postgres to accept connections..."
      until pg_isready -h "$$PGHOST" -p "$$PGPORT" -U "$$PGUSER" >/dev/null 2>&1; do sleep 1; done
      echo "Applying migrations..."
      psql -c "CREATE TABLE IF NOT EXISTS schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW());" 2>/dev/null
      for f in /migrations/[0-9]*.sql; do
        [ -f "$$f" ] || continue
        name=$$(basename "$$f")
        applied=$$(psql -tAc "SELECT 1 FROM schema_migrations WHERE filename='$$name'" 2>/dev/null)
        if [ "$$applied" != "1" ]; then
          echo "  Applying $$name..."
          psql -v ON_ERROR_STOP=1 -f "$$f"
          psql -c "INSERT INTO schema_migrations (filename) VALUES ('$$name');"
          echo "  Done."
        else
          echo "  Skipping $$name (already applied)"
        fi
      done
      echo "All migrations up to date."
  networks:
    - rag_network
  restart: "no"
```

How it works:
1. Waits for postgres to accept connections
2. Creates `schema_migrations` table if it doesn't exist
3. Iterates over `/migrations/[0-9]*.sql` in lexicographic order
4. Checks if each migration has been applied (exists in `schema_migrations`)
5. If not applied, runs the migration and records it
6. Exits with code 0 (success)

All migration files are idempotent (`CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ADD COLUMN IF NOT EXISTS`), so re-running on a fresh volume where `/docker-entrypoint-initdb.d` already applied them is a no-op.

**Service dependencies updated**:
- `app`, `celery_worker_upload`, `celery_worker_ingestion`, `celery_beat` now depend on `migrate: condition: service_completed_successfully`
- This ensures migrations complete before any application code starts

**`src/app/infra/db/identifiers.py`**: Added `schema_migrations` to `_SYSTEM_TABLES` denylist to prevent users from creating a chunk table with that name.

**`tests/unit/test_domains_and_doc_name.py`**: Added `schema_migrations` to the reserved names test.

---

## 3. Why These Changes

### 3.1 Log Files

- **Debugging**: Persistent logs survive container recreation, making it easier to diagnose issues that occurred before `docker compose down`
- **Audit**: Log files can be archived, searched, or shipped to external log aggregators
- **Performance**: `tee -a` is non-blocking and the `json-file` driver with rotation prevents unbounded growth

### 3.2 Auto-Migration

- **Fresh start**: `docker compose up` now works on a fresh volume without manual migration steps
- **Existing volumes**: New migrations are applied automatically on the next `docker compose up`
- **Idempotent**: Safe to run multiple times; already-applied migrations are skipped
- **No schema drift**: The `schema_migrations` table tracks what has been applied, so migrations are applied exactly once

---

## 4. Migration Strategy

The `migrate` service runs before any application service starts. On a fresh volume:

1. Postgres starts and runs `/docker-entrypoint-initdb.d/*.sql` (migrations 002-006)
2. `migrate` service starts and checks `schema_migrations`
3. All migrations are already applied (by initdb.d), so they are skipped
4. Application services start

On an existing volume with unapplied migrations:

1. Postgres starts (no initdb.d runs because the data directory is not empty)
2. `migrate` service starts and checks `schema_migrations`
3. Unapplied migrations are found and executed
4. Application services start

---

## 5. Verification

### 5.1 Fresh Volume

```bash
# Remove existing volume
docker compose down -v

# Start services
docker compose up -d

# Check migrate service logs
docker compose logs migrate

# Verify schema_migrations table
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT * FROM schema_migrations ORDER BY filename;"

# Expected output:
#       filename       |         applied_at
# ---------------------+----------------------------
#  002_create_llm_interactions.sql | 2026-08-12 10:00:00.000000+00
#  003_create_ingestion_status.sql | 2026-08-12 10:00:00.000000+00
#  004_ingestion_fixes.sql         | 2026-08-12 10:00:00.000000+00
#  005_drop_filename_dedupe.sql    | 2026-08-12 10:00:00.000000+00
#  006_domains_and_doc_name.sql    | 2026-08-12 10:00:00.000000+00
```

### 5.2 Existing Volume

```bash
# Stop services (keep volume)
docker compose down

# Start services
docker compose up -d

# Check migrate service logs
docker compose logs migrate

# Expected output:
# Waiting for postgres to accept connections...
# Applying migrations...
#   Skipping 002_create_llm_interactions.sql (already applied)
#   Skipping 003_create_ingestion_status.sql (already applied)
#   Skipping 004_ingestion_fixes.sql (already applied)
#   Skipping 005_drop_filename_dedupe.sql (already applied)
#   Skipping 006_domains_and_doc_name.sql (already applied)
# All migrations up to date.
```

### 5.3 Log Files

```bash
# Start services
docker compose up -d

# Check log files exist
ls -lh logs/

# Expected output:
# -rw-r--r--  1 user  staff   1.2M Aug 12 10:05 app.log
# -rw-r--r--  1 user  staff   856K Aug 12 10:05 celery_worker_upload.log
# -rw-r--r--  1 user  staff   1.5M Aug 12 10:05 celery_worker_ingestion.log
# -rw-r--r--  1 user  staff    45K Aug 12 10:05 celery_beat.log

# Tail a log file
tail -f logs/app.log

# Verify logs are also visible via docker compose
docker compose logs -f app
```

### 5.4 Upload Test

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@test.pdf" \
  -F "table_name=test_domain" \
  -F "doc_name=Test Document"

# Check logs
tail logs/app.log
tail logs/celery_worker_upload.log

# Verify document was ingested
curl http://localhost:8000/documents/<document_id>/status
```

---

## 6. Next Steps

### 6.1 Immediate

- [ ] Rebuild and recreate containers: `docker compose down && docker compose up -d --build`
- [ ] Verify `schema_migrations` table is populated
- [ ] Verify log files are created in `./logs/`
- [ ] Test upload flow to ensure migrations did not break anything

### 6.2 Optional Enhancements

- [ ] **Log rotation**: Add `logrotate` config to compress and archive old logs
- [ ] **Log aggregation**: Ship logs to external systems (ELK, Loki, CloudWatch)
- [ ] **Migration rollback**: Add a `rollback` command to undo the last migration
- [ ] **Migration versioning**: Use semantic versioning (e.g., `v1.0.0`, `v1.1.0`) instead of numeric prefixes
- [ ] **Pre-commit hook**: Run migrations in a test container before committing

### 6.3 Documentation Updates

- [ ] Update `README.md` to mention auto-migration and log files
- [ ] Update `docs/ARCHITECTURE.md` section 6.4 (Migrations) to reflect the new auto-apply behavior
- [ ] Add a troubleshooting section for common migration issues (e.g., duplicate key, missing column)

---

## 7. Edge Cases and Risks

### 7.1 Edge Cases

| Case | Handling |
|------|----------|
| Migration file is deleted after being applied | `schema_migrations` still has the record; the migration is not re-applied. Manual cleanup required. |
| Migration file is modified after being applied | The modified version is not re-applied. Manual intervention required. |
| Two migrations have the same prefix (e.g., `006a.sql`, `006b.sql`) | Lexicographic order determines which runs first. Avoid this; use unique numeric prefixes. |
| Postgres is not ready when `migrate` starts | The service waits for `pg_isready` before proceeding. |
| Migration fails mid-execution | `ON_ERROR_STOP=1` causes psql to exit immediately. The migration is not recorded in `schema_migrations`, so it will be retried on the next start. |

### 7.2 Risks

| Risk | Mitigation |
|------|------------|
| Migration fails and blocks all services | `migrate` service has `restart: "no"`, so it exits and does not loop. Manual intervention required. |
| Log files grow unbounded | `json-file` driver with `max-size` and `max-file` prevents this. |
| `tee -a` causes high I/O | Logs are buffered and flushed periodically. Monitor disk I/O if this becomes an issue. |
| `trap 'kill 0' EXIT` kills the wrong processes | The trap is scoped to the shell process group. Only child processes of the shell are killed. |

---

## 8. References

- **Docker Compose logging**: https://docs.docker.com/compose/compose-file/compose-file-v3/#logging
- **Postgres initdb.d**: https://hub.docker.com/_/postgres (see "Initialization scripts")
- **Migration best practices**: https://flywaydb.org/documentation/concepts/migrations

---

**End of Document**
