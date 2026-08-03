"""
Shared FastAPI dependencies.

Lives outside api/app.py so route modules can depend on the config and the
pipeline cache without importing the app (which imports them).

Before this existed, every route was declared twice: once with a decorator in its
route module, and again in api/app.py as a wrapper that passed `config` and
`get_pipeline` positionally. Only the wrappers were reachable, so a route added the
obvious way — with a decorator — silently 404'd. Routers are now mounted directly
and these providers supply what the handlers need.
"""

import asyncio
from typing import Dict

from app.config.app_config import AppConfig, DEFAULT_EMBEDDING_MODEL, DEFAULT_TABLE_NAME
from app.ingestion.embedding import ChunkEmbeddingPipeline

# Global configuration
config = AppConfig()

# One pipeline per table, kept for the process lifetime. A single-slot cache keyed on
# table name would rebuild — and therefore reload the SentenceTransformer, seconds of
# blocking work — whenever consecutive requests target different tables, which is
# caller-controlled. worker/ingestion_tasks.py uses the same pattern.
_PIPELINES: Dict[str, ChunkEmbeddingPipeline] = {}
_PIPELINE_LOCK = asyncio.Lock()


async def get_pipeline(table_name: str = DEFAULT_TABLE_NAME) -> ChunkEmbeddingPipeline:
    """
    Get or create the ChunkEmbeddingPipeline for a table.

    Args:
        table_name: Database table name for chunk storage

    Returns:
        ChunkEmbeddingPipeline instance
    """
    pipeline = _PIPELINES.get(table_name)
    if pipeline is None:
        async with _PIPELINE_LOCK:
            pipeline = _PIPELINES.get(table_name)
            if pipeline is None:
                # Constructing one loads an embedding model — blocking CPU work,
                # so it goes to a worker thread.
                pipeline = await asyncio.to_thread(
                    ChunkEmbeddingPipeline,
                    db_params=config.db_params,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    table_name=table_name,
                )
                _PIPELINES[table_name] = pipeline

    # config.pipeline is still read by /health.
    config.pipeline = pipeline
    return pipeline


def forget_pipeline(table_name: str) -> None:
    """Drop the cached pipeline for a table, after that table is deleted."""
    _PIPELINES.pop(table_name, None)
    if config.pipeline is not None and config.pipeline.vector_store.table_name == table_name:
        config.pipeline = None


# --- FastAPI dependency providers -------------------------------------------------
# These return the objects themselves rather than being called per request, so the
# handlers keep their existing signatures.

def get_config() -> AppConfig:
    return config


def get_pipeline_factory():
    return get_pipeline


def get_forget_pipeline():
    return forget_pipeline
