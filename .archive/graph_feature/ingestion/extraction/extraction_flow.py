"""
Entity extraction flow orchestration.
Handles document-level entity and relationship extraction with Celery integration.
"""

import os
import uuid
import logfire
from typing import Dict, Optional

from api.validators import entity_extraction_enabled


async def get_db_pool(config):
    """
    Get or create the shared asyncpg connection pool.

    Args:
        config: Application configuration object

    Returns:
        asyncpg connection pool
    """
    import asyncpg

    if getattr(config, "graph_pool", None) and not config.graph_pool._closed:
        return config.graph_pool

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = config.db_params['host']
        port = config.db_params['port']
        dbname = config.db_params['dbname']
        user = config.db_params['user']
        password = config.db_params['password']
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    config.graph_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
    return config.graph_pool


async def run_entity_extraction_for_document(
    document_id: str,
    filename: Optional[str],
    table_name: str,
    config
) -> Dict:
    """
    Shared entity extraction flow used by both API requests and Celery workers.
    Returns a summary dict instead of raising so uploads are resilient.

    Args:
        document_id: UUID of the document
        filename: Original filename
        table_name: Database table name
        config: Application configuration object

    Returns:
        Dict with extraction summary (status, entities_extracted, relationships_extracted, task_id)
    """
    if not entity_extraction_enabled():
        logfire.info(
            "Entity extraction disabled via configuration",
            document_id=document_id,
            table_name=table_name,
        )
        return {
            "status": "disabled",
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "task_id": None,
        }

    pool = None
    try:
        from graph_processing.extraction_service import create_extraction_service

        pool = await get_db_pool(config)
        extraction_service = await create_extraction_service(
            pool, table_name=table_name)

        logfire.info(
            "Starting entity extraction",
            document_id=document_id,
            table_name=table_name,
        )

        extraction_result = await extraction_service.extract_from_document(
            document_id=uuid.UUID(document_id),
            verbose=True  # Enable verbose logging to see what's happening
        )

        entities_extracted = extraction_result['total_entities']
        relationships_extracted = extraction_result['total_relationships']

        logfire.info(
            "Entity extraction completed",
            document_id=document_id,
            filename=filename,
            entities_extracted=entities_extracted,
            relationships_extracted=relationships_extracted,
            chunks_successful=extraction_result.get('successful'),
            chunks_failed=extraction_result.get('failed'),
            total_chunks=extraction_result.get('total_chunks'),
        )

        return {
            "status": "completed",
            "entities_extracted": entities_extracted,
            "relationships_extracted": relationships_extracted,
            "chunks_successful": extraction_result.get('successful'),
            "chunks_failed": extraction_result.get('failed'),
            "task_id": None,
        }

    except Exception as e:
        logfire.error(
            "Entity extraction failed",
            document_id=document_id,
            filename=filename,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "status": "error",
            "error": str(e),
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "task_id": None,
        }
    finally:
        if pool:
            await pool.close()
