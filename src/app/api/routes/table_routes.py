"""API routes for table management."""
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import logfire
from fastapi import APIRouter, Depends, HTTPException, Header

from app.api.dependencies import get_config, get_forget_pipeline, get_pipeline_factory
from app.api.routes.table_deletion import drop_chunk_table
from app.api.validators import require_access_password, validate_table_name
from app.infra.db import ConnectionPoolManager, TableRepository

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
            result = await drop_chunk_table(
                table_name,
                config=config,
                get_pipeline=get_pipeline,
                forget_pipeline=forget_pipeline,
            )

            logfire.info(
                "Table deletion completed successfully",
                table_name=table_name,
                estimated_rows_deleted=result["estimated_rows_deleted"],
                documents_removed=result["documents_removed"],
            )

            return {
                "status": "success",
                "message": f"Table '{table_name}' deleted successfully",
                "table_name": table_name,
                "estimated_rows_deleted": result["estimated_rows_deleted"],
                "documents_removed": result["documents_removed"],
                "domain_removed": result["domain_removed"],
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
