# ============================================
# RAG + Knowledge Graph Application Dockerfile
# Multi-stage build for optimal caching
# ============================================
# USAGE:
# 1. First build the base image (one-time, ~8-10 min):
#    docker build -f deployment/Dockerfile.base -t rag-base:latest .
# 2. Then build the app (fast, ~1-2 min):
#    docker build -f deployment/Dockerfile -t rag-app:latest .
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
COPY deployment/requirements.txt .
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
    PYTHONPATH=/app

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

# Copy application code by module for better caching on code changes
COPY api/ ./api/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY models/ ./models/
COPY worker/ ./worker/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY infra/ ./infra/

# Required directories for runtime state
RUN mkdir -p /app/data /app/logs /app/input/raw

# Security: Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application using uvicorn
# Note: --reload is not used in production for better performance
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
