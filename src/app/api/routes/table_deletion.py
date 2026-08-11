"""Shared chunk-table teardown, used by DELETE /table/{name} and DELETE /domains/{name}.

Dropping a chunk table is three coupled steps, not one. Doing only the DROP leaves
the ingestion status rows behind at stage='embedded' pointing at a table that no
longer exists, and leaves a stale pipeline (with its embedding model and pool)
cached against the dead table. Both endpoints need all three, so the sequence lives
here rather than being duplicated.
"""

from typing import Any, Dict

import logfire
from fastapi import HTTPException

from app.infra.db import IngestionRepository, TableRepository


async def drop_chunk_table(
    table_name: str,
    config,
    get_pipeline,
    forget_pipeline,
    missing_table_ok: bool = False,
) -> Dict[str, Any]:
    """Drop a chunk table, delete its ingestion status rows, and evict its pipeline.

    Args:
        table_name: Chunk table to drop. Must already be validated by the caller.
        config: App config, for the connection string and legacy pipeline slot.
        get_pipeline: Pipeline factory dependency.
        forget_pipeline: Pipeline cache eviction dependency; may be None.
        missing_table_ok: When True, a table that does not exist is not an error —
            used by domain deletion, where the registry row can outlive the table
            (a domain created but never uploaded to has no table yet).

    Returns:
        Dict with `table_existed`, `estimated_rows_deleted`, and `documents_removed`.

    Raises:
        HTTPException: 404 if the table does not exist and `missing_table_ok` is False.
    """
    pipeline_instance = await get_pipeline(table_name)
    async with pipeline_instance.vector_store.connection() as conn:
        repo = TableRepository(conn)

        with logfire.span("table_existence_check"):
            table_exists = await repo.table_exists(table_name)
            row_count = await repo.get_table_row_estimate(table_name) if table_exists else 0
            logfire.info(
                "Table existence check completed",
                table_exists=table_exists,
                estimated_rows=row_count,
            )

        if not table_exists and not missing_table_ok:
            logfire.warn("Table deletion failed - table does not exist", table_name=table_name)
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' does not exist",
            )

        if table_exists:
            # DROP removes the data too — TRUNCATE first only buys an extra
            # exclusive lock.
            with logfire.span("table_schema_deletion"):
                await repo.drop_table(table_name)
                logfire.info(
                    "Table dropped successfully",
                    table_name=table_name,
                    rows_deleted=row_count,
                )

    # Drop the ingestion status rows that pointed at this table. Leaving them behind
    # strands every document at stage='embedded' against a table that no longer
    # exists, and the file is then hard to reason about on re-upload.
    with logfire.span("ingestion_status_cleanup"):
        ingestion_repo = IngestionRepository(connection_string=config.connection_string)
        removed = await ingestion_repo.delete_documents_for_table(table_name)
        logfire.info(
            "Ingestion status rows removed",
            table_name=table_name,
            documents_removed=len(removed),
        )

    if forget_pipeline is not None:
        forget_pipeline(table_name)
    elif table_name == pipeline_instance.vector_store.table_name:
        config.pipeline = None
    logfire.info("Pipeline cache evicted", table_name=table_name)

    return {
        "table_existed": table_exists,
        "estimated_rows_deleted": row_count,
        "documents_removed": len(removed),
    }
