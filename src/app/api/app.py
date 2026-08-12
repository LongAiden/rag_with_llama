"""
Main FastAPI application for RAG (Retrieval-Augmented Generation) system.
Integrates vector search and LLM generation.

This module only wires the app together. Everything else lives in:
- config/app_config.py:            configuration and environment management
- api/dependencies.py:             shared config + pipeline cache injected via Depends
- api/validators.py:               request validation and authentication
- retrieval/search.py:             document search (vector + BM25 + rerank)
- retrieval/llm_operations.py:     LLM-based response generation
- api/routes/*.py:                 endpoint definitions, mounted below
- ingestion/chunking/chunker_factory.py: chunking strategies

Routes are owned entirely by their route modules. They used to be declared twice —
once with a decorator there and once as a wrapper here — with only the wrappers
reachable, so adding a route the obvious way silently 404'd.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import config
from app.api.routes import (
    admin_routes,
    document_routes,
    domain_routes,
    observability_routes,
    query_routes,
    table_routes,
)

# Initialize FastAPI application
app = FastAPI(title="pgvector RAG API", version="1.0.0")
app.mount("/images", StaticFiles(directory="docs/images"), name="images")

observability_routes.set_connection_string(config.connection_string)

app.include_router(query_routes.router)
app.include_router(document_routes.router)
app.include_router(table_routes.router)
app.include_router(domain_routes.router)
app.include_router(admin_routes.router)
app.include_router(observability_routes.router)


# ===============================================
# NOTE: Do not run this file directly!
# Use Docker to start the application:
#   docker compose up
# ===============================================
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
