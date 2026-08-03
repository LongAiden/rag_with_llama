"""API routes for table management."""
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import logfire
from fastapi import APIRouter, Depends, HTTPException, Header

from app.api.dependencies import get_config, get_forget_pipeline, get_pipeline_factory
from app.api.validators import require_access_password, validate_table_name
from app.infra.db import ConnectionPoolManager, IngestionRepository, TableRepository

router = APIRouter()


@asynccontextmanager
async def _table_connection(connection_string: str):
    """Borrow a pooled connection without instantiating a pipeline."""
    pool = await ConnectionPoolManager.get_pool(connection_string)
    async with pool.acquire() as conn:
        yield conn


@router.get("/tables/count")
async def get_table_count(config=Depends(get_config)):
    """Return the number of chunk tables in the database."""
    try:
        async with _table_connection(config.connection_string) as conn:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()
            return {"table_count": len(table_names), "table_names": table_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count tables: {str(e)}")


@router.get("/tables")
async def list_tables(config=Depends(get_config)):
    """List all chunk tables in the database with row counts."""
    try:
        async with _table_connection(config.connection_string) as conn:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()

            result = []
            for tname in table_names:
                counts = await repo.get_table_row_counts(tname)
                result.append({
                    "table_name": tname,
                    "documents": counts['documents'],
                    "chunks": counts['chunks'],
                })

            return {"tables": result, "total_tables": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@router.delete("/table/{table_name}")
async def delete_table(
    table_name: str,
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
    get_pipeline=Depends(get_pipeline_factory),
    forget_pipeline=Depends(get_forget_pipeline),
):
    """Delete a specific table from the database (optimized for speed)."""
    require_access_password(x_app_password)
    validate_table_name(table_name)
    with logfire.span("table_deletion", table_name=table_name):
        logfire.info("Starting table deletion", table_name=table_name)

        try:
            pipeline_instance = await get_pipeline(table_name)
            async with pipeline_instance.vector_store.connection() as conn:
                repo = TableRepository(conn)

                with logfire.span("table_existence_check"):
                    table_exists = await repo.table_exists(table_name)
                    row_count = await repo.get_table_row_estimate(table_name)
                    logfire.info(
                        "Table existence check completed",
                        table_exists=table_exists,
                        estimated_rows=row_count,
                    )

                if not table_exists:
                    logfire.warn("Table deletion failed - table does not exist", table_name=table_name)
                    raise HTTPException(
                        status_code=404,
                        detail=f"Table '{table_name}' does not exist",
                    )

                # DROP removes the data too — TRUNCATE first only buys an extra
                # exclusive lock.
                with logfire.span("table_schema_deletion"):
                    await repo.drop_table(table_name)
                    logfire.info(
                        "Table dropped successfully",
                        table_name=table_name,
                        rows_deleted=row_count,
                    )

            # Drop the ingestion status rows that pointed at this table. Leaving them
            # behind strands every document at stage='embedded' against a table that
            # no longer exists, and the (file_name, target_table_name) unique key then
            # rejects re-uploads as duplicates.
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

            logfire.info(
                "Table deletion completed successfully",
                table_name=table_name,
                estimated_rows_deleted=row_count,
                documents_removed=len(removed),
            )

            return {
                "status": "success",
                "message": f"Table '{table_name}' deleted successfully",
                "table_name": table_name,
                "estimated_rows_deleted": row_count,
                "documents_removed": len(removed),
                "timestamp": str(uuid.uuid1().time),
            }

        except HTTPException:
            raise
        except Exception as e:
            logfire.error(
                "Table deletion failed with unexpected error",
                table_name=table_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete table '{table_name}': {str(e)}",
            )
