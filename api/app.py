"""
Main FastAPI application for RAG (Retrieval-Augmented Generation) system.
Integrates vector search and LLM generation.

This module serves as the entry point and orchestrator for the entire RAG pipeline.
Core functionality is distributed across specialized modules:
- config/app_config.py: Configuration and environment management
- api/validators.py:    Request validation and authentication
- retrieval/search.py:  Document search (vector + BM25 rerank)
- retrieval/llm_operations.py: LLM-based response generation
- api/routes/document_routes.py: FastAPI endpoint definitions
- ingestion/chunking/chunker_factory.py: Chunking strategies
"""

from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.app_config import AppConfig, DEFAULT_TABLE_NAME, DEFAULT_EMBEDDING_MODEL
from ingestion.embedding import ChunkEmbeddingPipeline
from models.schemas import QueryRequest, UploadResponse, RAGResponse
# Global configuration
config = AppConfig()


# Initialize FastAPI application
app = FastAPI(title="pgvector RAG API", version="1.0.0")
app.mount("/images", StaticFiles(directory="docs/images"), name="images")


async def get_pipeline(table_name: str = DEFAULT_TABLE_NAME):
    """
    Get or create ChunkEmbeddingPipeline for the specified table.
    Implements lazy initialization pattern for performance.

    Args:
        table_name: Database table name for chunk storage

    Returns:
        ChunkEmbeddingPipeline instance
    """
    if config.pipeline is None or config.pipeline.vector_store.table_name != table_name:
        config.pipeline = ChunkEmbeddingPipeline(
            db_params=config.db_params,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            table_name=table_name
        )
    return config.pipeline


# Observability routes
import api.routes.observability_routes as _obs_routes
_obs_routes._connection_string = config.connection_string
app.include_router(_obs_routes.router)

# Import route handlers from split route modules
from api.routes.document_routes import (
    upload_and_process,
    get_supported_types,
    get_document_status,
    delete_document,
)
from api.routes.query_routes import (
    home,
    query_documents,
    query_documents_form,
)
from api.routes.table_routes import (
    get_table_count,
    list_tables,
    delete_table,
)
from api.routes.admin_routes import (
    get_database_stats,
    health_check,
)


# Re-register routes with dependency injection
@app.get("/", response_class=HTMLResponse)
async def home_route():
    return await home()


@app.post("/upload", response_model=UploadResponse)
async def upload_route(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    table_name: str = Form("document_chunks"),
    parse_backend: str = Form("ollama"),
    access_password: Optional[str] = Form(None),
    x_app_password: Optional[str] = Header(default=None),
):
    table_name = table_name.strip() or DEFAULT_TABLE_NAME
    return await upload_and_process(
        file=file,
        chunk_size=chunk_size,
        table_name=table_name,
        parse_backend=parse_backend,
        access_password=access_password,
        x_app_password=x_app_password,
        config=config,
        get_pipeline=get_pipeline
    )


@app.post("/query", response_model=RAGResponse)
async def query_route(
    request: QueryRequest,
    x_app_password: Optional[str] = Header(default=None),
):
    return await query_documents(
        request=request,
        x_app_password=x_app_password,
        config=config,
        get_pipeline=get_pipeline
    )


@app.post("/query-form", response_class=HTMLResponse)
async def query_form_route(
    query: str = Form(...),
    limit: int = Form(5),
    threshold: float = Form(0.3),
    table_name: str = Form(DEFAULT_TABLE_NAME),
    model: str = Form("gemini-2.5-flash"),
    access_password: Optional[str] = Form(None),
):
    table_name = table_name.strip() or DEFAULT_TABLE_NAME
    return await query_documents_form(
        query=query,
        limit=limit,
        threshold=threshold,
        table_name=table_name,
        model=model,
        access_password=access_password,
        config=config,
        get_pipeline=get_pipeline
    )


@app.get("/tables/count")
async def tables_count_route():
    return await get_table_count(get_pipeline=get_pipeline)


@app.get("/tables")
async def tables_route():
    return await list_tables(get_pipeline=get_pipeline)


@app.get("/stats", response_class=HTMLResponse)
async def stats_route():
    return await get_database_stats(get_pipeline=get_pipeline)


@app.get("/health", response_class=HTMLResponse)
async def health_route():
    return await health_check(get_pipeline=get_pipeline, config=config)


@app.get("/supported-types")
async def supported_types_route():
    return await get_supported_types(config=config)


@app.get("/documents/{document_id}/status")
async def document_status_route(document_id: str):
    return await get_document_status(document_id=document_id, config=config)


@app.delete("/documents/{document_id}")
async def delete_document_route(
    document_id: str,
    delete_chunks: bool = True,
    delete_raw_file: bool = True,
    x_app_password: Optional[str] = Header(default=None),
):
    return await delete_document(
        document_id=document_id,
        delete_chunks=delete_chunks,
        delete_raw_file=delete_raw_file,
        x_app_password=x_app_password,
        config=config,
        get_pipeline=get_pipeline,
    )


@app.delete("/table/{table_name}")
async def delete_table_route(
    table_name: str,
    x_app_password: Optional[str] = Header(default=None),
):
    return await delete_table(
        table_name=table_name,
        x_app_password=x_app_password,
        config=config,
        get_pipeline=get_pipeline
    )


# ===============================================
# NOTE: Do not run this file directly!
# Use Docker to start the application:
#   docker compose up
# ===============================================
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
