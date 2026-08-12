#!/bin/bash
# Build script for RAG application Docker images
# Usage: ./deploy/deployment/build.sh [--no-cache] [--base-only] [--app-only]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BUILD_BASE=true
BUILD_APP=true
EXTRA_ARGS=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --no-cache)
            EXTRA_ARGS="--no-cache"
            ;;
        --base-only)
            BUILD_APP=false
            ;;
        --app-only)
            BUILD_BASE=false
            ;;
    esac
done

# Build base image (heavy ML dependencies)
if [ "$BUILD_BASE" = true ]; then
    echo "========================================"
    echo "Building base image (ML dependencies)..."
    echo "========================================"
    docker build -f deploy/deployment/Dockerfile.base -t rag-base:latest $EXTRA_ARGS .
    echo "Base image built successfully!"
fi

# Build application image
if [ "$BUILD_APP" = true ]; then
    echo "========================================"
    echo "Building application image..."
    echo "========================================"
    docker build -f deploy/deployment/Dockerfile -t rag-app:latest $EXTRA_ARGS .
    echo "Application image built successfully!"
fi

echo "========================================"
echo "Ensuring langfuse_db exists..."
echo "========================================"
docker compose exec postgres psql -U admin -tc \
    "SELECT 1 FROM pg_database WHERE datname = 'langfuse_db'" \
    | grep -q 1 \
    && echo "langfuse_db already exists, skipping." \
    || docker compose exec postgres psql -U admin -c "CREATE DATABASE langfuse_db;"

echo "========================================"
echo "Build complete!"
echo "========================================"
