"""API routes for admin/observability endpoints: stats and health."""
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_config, get_pipeline_factory
from app.api.renderer import render
from app.config.app_config import DEFAULT_TABLE_NAME
from app.infra.db import ConnectionPoolManager, TableRepository

router = APIRouter()


@asynccontextmanager
async def _admin_connection(connection_string: str):
    """Borrow a pooled connection without instantiating a pipeline."""
    pool = await ConnectionPoolManager.get_pool(connection_string)
    async with pool.acquire() as conn:
        yield conn


@router.get("/stats", response_class=HTMLResponse)
async def get_database_stats(config=Depends(get_config)):
    """Get database statistics from ALL chunk tables."""
    try:
        async with _admin_connection(config.connection_string) as conn:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()

        print(f"\n📊 Found chunk tables: {', '.join(table_names) if table_names else 'none'}")

        if not table_names:
            stats = {
                'total_documents': 0,
                'total_chunks': 0,
                'avg_text_length': 0,
                'earliest_chunk': None,
                'latest_chunk': None,
            }
            table_display = DEFAULT_TABLE_NAME
        else:
            total_docs = 0
            total_chunks = 0
            total_text_length = 0
            earliest = None
            latest = None

            async with _admin_connection(config.connection_string) as conn:
                repo = TableRepository(conn)
                for table_name in table_names:
                    result = await repo.get_table_stats(table_name)

                    total_docs += result['documents'] or 0
                    total_chunks += result['chunks'] or 0
                    total_text_length += result['total_text_length'] or 0

                    print(f"  {table_name}: {result['documents']} docs, {result['chunks']} chunks")

                    if result['earliest'] and (earliest is None or result['earliest'] < earliest):
                        earliest = result['earliest']
                    if result['latest'] and (latest is None or result['latest'] > latest):
                        latest = result['latest']

            stats = {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'avg_text_length': total_text_length / total_chunks if total_chunks > 0 else 0,
                'earliest_chunk': earliest,
                'latest_chunk': latest,
            }

            table_display = f"ALL TABLES ({len(table_names)} tables)"
            print(f"📊 TOTAL: {total_docs} documents, {total_chunks} chunks\n")

        return render(
            "stats.html",
            total_documents=f"{stats['total_documents']:,}",
            total_chunks=f"{stats['total_chunks']:,}",
            avg_text_length=f"{stats['avg_text_length']:.0f}",
            avg_chunks_per_doc=f"{stats['total_chunks'] // max(stats['total_documents'], 1):.0f}",
            total_tables=len(table_names) if table_names else 1,
            table_name=table_display,
            earliest_chunk=str(stats['earliest_chunk']) if stats['earliest_chunk'] else 'No documents yet',
            latest_chunk=str(stats['latest_chunk']) if stats['latest_chunk'] else 'No documents yet',
        )

    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render("stats_error.html", error_message=str(e))


@router.get("/health", response_class=HTMLResponse)
async def health_check(
    get_pipeline=Depends(get_pipeline_factory),
    config=Depends(get_config),
):
    """Health check endpoint to verify system status."""
    try:
        pipeline = None
        if config is not None and config.pipeline is not None:
            pipeline = config.pipeline
        elif get_pipeline is not None:
            pipeline = await get_pipeline(DEFAULT_TABLE_NAME)

        if pipeline is None:
            return render("health_error.html", error_message="No pipeline initialized yet")

        stats = await pipeline.get_stats()

        db_status = "healthy" if stats['total_chunks'] >= 0 else "error"
        status_icon = "✅" if db_status == "healthy" else "❌"
        status_color = "#28a745" if db_status == "healthy" else "#dc3545"

        return render(
            "health_check.html",
            status_color=status_color,
            status_icon=status_icon,
            db_status_upper=db_status.upper(),
            embedding_model=pipeline.embedding_generator.model_name,
            table_name=pipeline.vector_store.table_name,
            total_documents=f"{stats['total_documents']:,}",
            total_chunks=f"{stats['total_chunks']:,}",
            embedding_dim=pipeline.embedding_generator.embedding_dim,
            avg_text_length=f"{stats['avg_text_length']:.0f}",
            timestamp=str(uuid.uuid1().time),
        )

    except Exception as e:
        return render("health_error.html", error_message=str(e))
