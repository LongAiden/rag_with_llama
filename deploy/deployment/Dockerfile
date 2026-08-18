# ============================================
# RAG + Knowledge Graph Application Dockerfile
# Multi-stage build for optimal caching
# ============================================
# USAGE:
# 1. First build the base image (one-time, ~8-10 min):
#    docker build -f deploy/deployment/Dockerfile.base -t rag-base:latest .
# 2. Then build the app (fast, ~1-2 min):
#    docker build -f deploy/deployment/Dockerfile -t rag-app:latest .
# ============================================

# Global ARGs (must be before FROM to be used in FROM)
ARG BASE_IMAGE=rag-base:latest
ARG PYTHON_VERSION=3.11

# Use pre-built base image with ML dependencies
FROM ${BASE_IMAGE} AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install application dependencies (fast - ML packages already in base)
COPY deploy/deployment/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

# Stage 2: Runtime image (stays lean)
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HF_HOME=/home/appuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence_transformers \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# Runtime dependencies only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Application code lives under src/ so `app` is a single importable package
# (PYTHONPATH=/app/src above). Migrations are copied for reference only —
# Postgres applies them from its own initdb mount, not from this image.
COPY src/ ./src/
COPY deploy/migrations/ ./deploy/migrations/
COPY docs/images/ ./docs/images/

# Required directories for runtime state. data/parsed and data/chunks hold the
# per-stage ingestion artifacts written by the parse and chunk stages.
RUN mkdir -p /app/data/parsed /app/data/chunks /app/data/input/raw /app/logs

# Security: Run as non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/sentence_transformers && \
    chown -R appuser:appuser /home/appuser/.cache /app

# Pre-download the default embedding and reranker models so neither workers nor
# the app have to fetch them on first use. This avoids the runtime memory spike
# and cold-start latency. The reranker is baked here (not in an hf_cache volume)
# because the app container does not mount hf_cache — only the celery workers do
# — so without this it would land in the writable layer and be re-fetched on
# every --force-recreate.
USER appuser
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config.app_config import DEFAULT_EMBEDDING_MODEL, AppSettings
SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
CrossEncoder(AppSettings().rerank_model)
PY

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application using uvicorn
# Note: --reload is not used in production for better performance
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
