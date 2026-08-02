"""API routes for table management."""
import uuid
from typing import Optional

import logfire
from fastapi import APIRouter, HTTPException, Header

from api.validators import require_access_password, validate_table_name
from config.app_config import DEFAULT_TABLE_NAME
from infra.db import TableRepository

router = APIRouter()


@router.get("/tables/count")
async def get_table_count(get_pipeline=None, pipeline=None):
    """Return the number of chunk tables in the database."""
    try:
        if pipeline is None:
            pipeline = await get_pipeline()
        async with pipeline.vector_store.connection() as conn:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()
            return {"table_count": len(table_names), "table_names": table_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count tables: {str(e)}")


@router.get("/tables")
async def list_tables(get_pipeline=None):
    """List all chunk tables in the database with row counts."""
    try:
        pipeline = await get_pipeline(DEFAULT_TABLE_NAME)
        async with pipeline.vector_store.connection() as conn:
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
    config=None,
    get_pipeline=None,
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

                with logfire.span("table_data_deletion"):
                    await repo.truncate_table(table_name)
                    logfire.info(
                        "Table data truncated successfully",
                        table_name=table_name,
                        rows_deleted=row_count,
                    )

                with logfire.span("table_schema_deletion"):
                    await repo.drop_table(table_name)
                    logfire.info(
                        "Table schema dropped successfully",
                        table_name=table_name,
                    )

                if table_name == pipeline_instance.vector_store.table_name:
                    config.pipeline = None
                    logfire.info(
                        "Pipeline reset due to current table deletion",
                        table_name=table_name,
                    )

                logfire.info(
                    "Table deletion completed successfully",
                    table_name=table_name,
                    estimated_rows_deleted=row_count,
                )

                return {
                    "status": "success",
                    "message": f"Table '{table_name}' deleted successfully",
                    "table_name": table_name,
                    "estimated_rows_deleted": row_count,
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
