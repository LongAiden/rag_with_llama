# Docker Setup Documentation

## Overview
This project uses a multi-service Docker setup for the RAG (Retrieval-Augmented Generation) application with PostgreSQL, Redis, Celery workers, and optional observability tools.

## Services

### Core Services
- **postgres**: PostgreSQL 16 with pgvector, PostGIS, and pgRouting extensions
- **redis**: Redis 7 (Celery broker/backend)
- **app**: FastAPI application (main API server)
- **celery_worker**: Background task processor

### Optional Services
- **langfuse**: LLM observability UI (profile: `observability`)
- **pgadmin**: Database management UI (profile: `dev`)
- **test**: Test runner service (profile: `test`)

## Recent Fixes & Improvements

### Critical Fixes
1. **Dockerfile.postgres COPY path** (`deployment/Dockerfile.postgres:31`)
   - **Issue**: Invalid path `../migrations/*.sql` (Docker cannot access parent directories)
   - **Fix**: Changed to `migrations/*.sql` (context is project root)

2. **Environment variable consistency**
   - **Issue**: Mismatched variable names between `docker-compose.yml` and `.env.example`
   - **Fix**: Aligned all services to use consistent naming from `.env.example`

### Security Improvements
1. **Removed hardcoded credentials**
   - **Before**: Default values `admin/admin` for PostgreSQL, pgAdmin
   - **After**: Required variables with no defaults (`${VAR:?message}` syntax)
   - **Impact**: Forces users to set credentials explicitly via `.env` file

2. **Services updated**:
   - `postgres`: POSTGRES_USER, POSTGRES_PASSWORD now required
   - `app`: Database credentials now required
   - `celery_worker`: Database credentials now required
   - `langfuse`: Database credentials now required
   - `pgadmin`: PGADMIN_EMAIL, PGADMIN_PASSWORD now required
   - `test`: Database credentials now required

### Structural Improvements
1. **Dockerfile.test completeness**
   - **Added missing directories**: `repositories/`, `observability/`, `migrations/`
   - **Why**: Prevents import errors when running tests

2. **Redis healthcheck**
   - **Added**: `redis-cli ping` healthcheck
   - **Updated dependencies**: Changed from `service_started` to `service_healthy`
   - **Benefit**: Ensures Redis is actually ready before dependent services start

3. **Dockerfile cleanup**
   - **Fixed**: opencv-python uninstall command (added `|| true` to prevent failures)
   - **Benefit**: More robust build process

## Quick Start

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env with your actual credentials and API keys
```

**Required variables** (no defaults):
- `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `PGADMIN_EMAIL`, `PGADMIN_PASSWORD` (if using pgadmin profile)
- `GOOGLE_API_KEY` (for Gemini LLM)

### 2. Build Images
```bash
# Build base image (one-time, ~8-10 min)
docker build -f deployment/Dockerfile.base -t rag-base:latest .

# Build app image (~1-2 min)
docker compose build app
```

### 3. Start Services
```bash
# Core services only
docker compose up -d

# With observability (Langfuse)
docker compose --profile observability up -d

# With pgAdmin (development)
docker compose --profile dev up -d
```

### 4. Run Tests
```bash
docker compose --profile test run --rm test
```

## Port Mappings
| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| Langfuse | 3000 | http://localhost:3000 |
| pgAdmin | 5050 | http://localhost:5050 |

## Volumes
- `postgres_data`: Persistent PostgreSQL data
- `upload_data`: Uploaded files
- `pgadmin_data`: pgAdmin configuration

## Health Checks
All core services have health checks:
- **postgres**: `pg_isready` (10s interval)
- **redis**: `redis-cli ping` (10s interval)
- **app**: `curl http://localhost:8000/health` (30s interval, 40s start period)

## Resource Limits
Each service has CPU/memory limits configured via `deploy.resources`:
- Prevents resource exhaustion
- Ensures fair allocation across services
- Adjust based on your workload

## Troubleshooting

### Build fails on Dockerfile.postgres
- Ensure you're building from project root: `docker build -f deployment/Dockerfile.postgres .`
- Check that `migrations/` directory exists with `.sql` files

### Services won't start
- Verify `.env` file has all required variables
- Check logs: `docker compose logs <service>`
- Ensure ports aren't already in use

### Database connection errors
- Wait for postgres healthcheck to pass: `docker compose ps`
- Verify credentials in `.env` match across all services

## Development Mode
For hot-reload during development:
```bash
# Code is mounted as volume in docker-compose.yml
docker compose up app
# Edit files locally, changes reflect immediately
```

## Production Considerations
1. **Change all default credentials** in `.env`
2. **Use strong passwords** for PostgreSQL and pgAdmin
3. **Set proper API keys** (GOOGLE_API_KEY, etc.)
4. **Configure Langfuse secrets** if using observability
5. **Consider using Docker secrets** for sensitive data
6. **Review resource limits** based on production workload
7. **Enable TLS/SSL** for external access
8. **Set up backups** for `postgres_data` volume
