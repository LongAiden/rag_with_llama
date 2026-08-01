"""
API routes for the RAG application.
FastAPI endpoints for upload, query, stats, health checks, and table management.
"""
import os
import uuid
import traceback
import logfire
from pathlib import Path
from typing import Optional, List

from celery import chain
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Header
from fastapi.responses import HTMLResponse

from api.config import DEFAULT_TABLE_NAME, DEFAULT_CHUNKING_SIMILARITY
from api.validators import (
    validate_upload_params,
    require_access_password,
)
from repositories.ingestion_repository import IngestionRepository
from repositories.table_repository import TableRepository, validate_table_name
from retrieval.search import perform_document_search
from worker.ingestion_tasks import parse_document_task, chunk_document_task, embed_document_task

INPUT_RAW_DIR = Path(os.getenv("INPUT_RAW_DIR", "input/raw"))

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    langfuse_context = type("_Noop", (), {
        "update_current_trace": staticmethod(lambda **_: None),
        "update_current_observation": staticmethod(lambda **_: None),
    })()
    def observe(**__):
        def decorator(fn): return fn
        return decorator
# Lazy import to avoid circular dependency - imported in upload_and_process function
from api.templates import (
    HOME_PAGE_HTML,
    SEARCH_RESULTS_HTML,
    SEARCH_ERROR_HTML,
    STATS_PAGE_HTML,
    STATS_ERROR_HTML,
    HEALTH_CHECK_HTML,
    HEALTH_ERROR_HTML
)
from models.models import QueryRequest, UploadResponse, RAGResponse

# Create router
router = APIRouter()


async def get_pipeline_for_routes(table_name: str, get_pipeline_func):
    """Helper to get pipeline instance."""
    return await get_pipeline_func(table_name)


@router.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload and search forms."""
    return HOME_PAGE_HTML


@router.post("/upload", response_model=UploadResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    table_name: str = Form("document_chunks"),
    parse_backend: str = Form(""),
    access_password: Optional[str] = Form(None),
    x_app_password: Optional[str] = Header(default=None),
    # Dependencies injected by main app
    config=None,
    get_pipeline=None
):
    """Upload a document, persist the raw file, and queue it for processing."""
    require_access_password(access_password or x_app_password)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    validate_upload_params(chunk_size, file.content_type)

    document_id = str(uuid.uuid4())
    INPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = INPUT_RAW_DIR / f"{document_id}_{file.filename}"

    try:
        content = await file.read()
        with open(raw_path, "wb") as f:
            f.write(content)

        validation_result = config.file_validator.validate_file(str(raw_path))
        if not validation_result.is_valid:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.error_message}"
            )

        repo = IngestionRepository(connection_string=config.connection_string)
        await repo.register_document(
            doc_id=document_id,
            file_name=file.filename,
            raw_storage_path=str(raw_path.resolve()),
            file_size=validation_result.file_size,
            content_type=file.content_type or "application/octet-stream",
            target_table_name=table_name,
            chunk_size=chunk_size,
            parse_backend=parse_backend,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": validation_result.file_size,
                "upload_timestamp": str(uuid.uuid1().time),
                "validation_passed": True,
            },
        )

        task_chain = chain(
            parse_document_task.s(document_id),
            chunk_document_task.s(document_id),
            embed_document_task.s(document_id),
        )
        async_task = task_chain.apply_async(queue="upload")

        logfire.info(
            "Upload queued for Celery worker",
            document_id=document_id,
            filename=file.filename,
            task_id=async_task.id,
            table_name=table_name,
        )

        return UploadResponse(
            status="queued",
            document_id=document_id,
            filename=file.filename,
            message=f"Upload queued for processing. Task {async_task.id}",
            chunks_created=None,
            task_id=async_task.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logfire.error(
            "Upload processing failed",
            document_id=document_id,
            filename=file.filename if file.filename else "unknown",
            error_type=type(e).__name__,
            error=str(e),
            traceback=tb,
        )
        print(f"[UPLOAD ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(e).__name__}: {e}"
        )


async def _execute_traced_search(
    query: str,
    table_name: str,
    limit: int,
    threshold: float,
    model: str,
    session_id: Optional[str],
    document_ids: Optional[List[str]],
    config,
    get_pipeline,
    enable_reranking: bool = True,
    rerank_top_k: int = 5,
) -> RAGResponse:
    """Shared search logic used by both JSON and form endpoints.
    
    NO @observe decorator here — receives heavy objects (config, get_pipeline)
    that cannot be serialized. Tracing wrappers live in each route handler.
    """
    pipeline = await get_pipeline(table_name)
    return await perform_document_search(
        query=query,
        limit=limit,
        threshold=threshold,
        pipeline=pipeline,
        config=config,
        document_ids=document_ids,
        table_name=table_name,
        model=model,
        session_id=session_id,
        enable_reranking=enable_reranking,
        rerank_top_k=rerank_top_k,
    )


@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: QueryRequest,
    x_app_password: Optional[str] = Header(default=None),
    # Dependencies injected by main app
    config=None,
    get_pipeline=None
):
    """Query documents using pgvector similarity search + LLM generation with optional reranking."""
    require_access_password(x_app_password)
    try:
        table_name = (request.table_name or DEFAULT_TABLE_NAME).strip()

        @observe(name="rag_query")
        async def _run_search(query: str):
            langfuse_context.update_current_trace(
                session_id=request.session_id,
                metadata={"table_name": table_name},
            )
            return await _execute_traced_search(
                query=query,
                table_name=table_name,
                limit=request.limit,
                threshold=request.threshold,
                model=request.model,
                session_id=request.session_id,
                document_ids=request.document_ids,
                config=config,
                get_pipeline=get_pipeline,
                enable_reranking=request.enable_reranking,
                rerank_top_k=request.rerank_top_k or 5,
            )

        result = await _run_search(request.query)
        # Return the structured RAGResponse directly
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query-form", response_class=HTMLResponse)
async def query_documents_form(
    query: str = Form(...),
    limit: int = Form(5),
    threshold: float = Form(0.3),
    table_name: str = Form(DEFAULT_TABLE_NAME),
    model: str = Form("gemini-2.5-flash"),
    access_password: Optional[str] = Form(None),
    # Dependencies injected by main app
    config=None,
    get_pipeline=None
):
    """Query documents using form data (for HTML form submission) with optional reranking."""
    require_access_password(access_password)
    try:
        @observe(name="rag_query")
        async def _run_search(query: str):
            return await _execute_traced_search(
                query=query,
                table_name=table_name,
                limit=limit,
                threshold=threshold,
                model=model,
                session_id=None,
                document_ids=None,
                config=config,
                get_pipeline=get_pipeline,
                enable_reranking=True,
                rerank_top_k=5,
            )

        result = await _run_search(query)

        # Build sources HTML with hybrid scores
        sources_html = ''.join([f"""
        <div class="source-item">
            <strong>Source {i+1}</strong> (Similarity: {source.similarity:.1%}{f", BM25: {source.bm25_score:.3f}" if source.bm25_score else ""}{f", RRF: {source.rrf_score:.3f}" if source.rrf_score else ""}{f", Rerank: {source.rerank_score:.3f}" if source.rerank_score is not None else ""})<br>
            <em>Document: {source.document_id[:8]}... | Page: {source.page_number or 'N/A'}</em><br><br>
            <div style="white-space: pre-wrap; word-wrap: break-word;">{source.text}</div>
        </div>
        """ for i, source in enumerate(result.sources)])

        # Use template with substitutions
        stats = result.search_stats
        input_tok = stats.input_tokens
        output_tok = stats.output_tokens
        total_tok = stats.total_tokens
        token_display = (
            f"↑ {input_tok:,} in · ↓ {output_tok:,} out · Σ {total_tok:,} total"
            if total_tok else "N/A"
        )
        html_content = SEARCH_RESULTS_HTML.format(
            query=query,
            answer=result.answer.strip(),
            source_count=len(result.sources),
            sources_html=sources_html,
            chunks_found=stats.chunks_found,
            avg_similarity=f"{stats.avg_similarity:.1%}",
            search_method=stats.search_method,
            table_used=result.table_used,
            threshold_used=f"{stats.threshold_used:.1%}",
            confidence=f"{stats.confidence:.1%}" if stats.confidence else "N/A",
            word_count=stats.word_count or 0,
            graph_enriched="Yes" if stats.graph_enriched else "No",
            token_display=token_display
        )

        return html_content

    except Exception as e:
        # Check if it's a rate limit error and provide helpful message
        error_msg = str(e)
        if any(indicator in error_msg.lower() for indicator in ["rate limit", "quota exceeded", "429", "resource exhausted"]):
            error_msg = (
                f"⚠️ API Rate Limit Reached\n\n"
                f"The Gemini API rate limit has been exceeded. This typically happens during entity extraction.\n\n"
                f"Please try again in a minute or two.\n\n"
                f"Technical details: {error_msg}"
            )

        # Return error page using template
        return SEARCH_ERROR_HTML.format(error_message=error_msg)


@router.get("/tables/count")
async def get_table_count(get_pipeline=None, pipeline=None):
    """Return the number of chunk tables in the database."""
    try:
        if pipeline is None:
            pipeline = await get_pipeline()
        conn = await pipeline.vector_store._get_connection()
        try:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()
            return {"table_count": len(table_names), "table_names": table_names}
        finally:
            await pipeline.vector_store._release_connection(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count tables: {str(e)}")


@router.get("/tables")
async def list_tables(get_pipeline=None):
    """List all chunk tables in the database with row counts."""
    try:
        pipeline = await get_pipeline(DEFAULT_TABLE_NAME)
        conn = await pipeline.vector_store._get_connection()

        try:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()

            result = []
            for tname in table_names:
                counts = await repo.get_table_row_counts(tname)
                result.append({
                    "table_name": tname,
                    "documents": counts['documents'],
                    "chunks": counts['chunks']
                })

            return {"tables": result, "total_tables": len(result)}

        finally:
            await pipeline.vector_store._release_connection(conn)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@router.get("/stats", response_class=HTMLResponse)
async def get_database_stats(get_pipeline=None):
    """Get database statistics from ALL chunk tables."""
    try:
        pipeline = await get_pipeline(DEFAULT_TABLE_NAME)
        conn = await pipeline.vector_store._get_connection()

        try:
            repo = TableRepository(conn)
            table_names = await repo.list_chunk_tables()

            print(f"\n📊 Found chunk tables: {', '.join(table_names) if table_names else 'none'}")

            if not table_names:
                stats = await pipeline.get_stats()
                table_display = pipeline.vector_store.table_name
            else:
                total_docs = 0
                total_chunks = 0
                total_text_length = 0
                earliest = None
                latest = None

                for table_name in table_names:
                    result = await repo.get_table_stats(table_name)

                    total_docs += result['docs'] or 0
                    total_chunks += result['chunks'] or 0
                    total_text_length += result['total_text_length'] or 0

                    print(f"  {table_name}: {result['docs']} docs, {result['chunks']} chunks")

                    if result['earliest'] and (earliest is None or result['earliest'] < earliest):
                        earliest = result['earliest']
                    if result['latest'] and (latest is None or result['latest'] > latest):
                        latest = result['latest']

                stats = {
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                    'avg_text_length': total_text_length / total_chunks if total_chunks > 0 else 0,
                    'earliest_chunk': earliest,
                    'latest_chunk': latest
                }

                table_display = f"ALL TABLES ({len(table_names)} tables)"
                print(f"📊 TOTAL: {total_docs} documents, {total_chunks} chunks\n")

            return STATS_PAGE_HTML.format(
                total_documents=f"{stats['total_documents']:,}",
                total_chunks=f"{stats['total_chunks']:,}",
                avg_text_length=f"{stats['avg_text_length']:.0f}",
                avg_chunks_per_doc=f"{stats['total_chunks'] // max(stats['total_documents'], 1):.0f}",
                total_tables=len(table_names) if table_names else 1,
                embedding_model=pipeline.embedding_generator.model_name,
                embedding_dim=pipeline.embedding_generator.embedding_dim,
                table_name=table_display,
                earliest_chunk=str(stats['earliest_chunk']) if stats['earliest_chunk'] else 'No documents yet',
                latest_chunk=str(stats['latest_chunk']) if stats['latest_chunk'] else 'No documents yet'
            )

        finally:
            await pipeline.vector_store._release_connection(conn)

    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        import traceback
        traceback.print_exc()
        return STATS_ERROR_HTML.format(error_message=str(e))


@router.get("/health", response_class=HTMLResponse)
async def health_check(get_pipeline=None, config=None):
    """Health check endpoint to verify system status."""
    try:
        # Reuse existing pipeline if available; don't create document_chunks on every health check
        pipeline = None
        if config is not None and config.pipeline is not None:
            pipeline = config.pipeline
        elif get_pipeline is not None:
            pipeline = await get_pipeline(DEFAULT_TABLE_NAME)

        if pipeline is None:
            return HEALTH_ERROR_HTML.format(error_message="No pipeline initialized yet")

        stats = await pipeline.get_stats()

        # Check database connection
        db_status = "healthy" if stats['total_chunks'] >= 0 else "error"
        status_icon = "✅" if db_status == "healthy" else "❌"
        status_color = "#28a745" if db_status == "healthy" else "#dc3545"

        html_content = HEALTH_CHECK_HTML.format(
            status_color=status_color,
            status_icon=status_icon,
            db_status_upper=db_status.upper(),
            embedding_model=pipeline.embedding_generator.model_name,
            table_name=pipeline.vector_store.table_name,
            total_documents=f"{stats['total_documents']:,}",
            total_chunks=f"{stats['total_chunks']:,}",
            embedding_dim=pipeline.embedding_generator.embedding_dim,
            avg_text_length=f"{stats['avg_text_length']:.0f}",
            timestamp=str(uuid.uuid1().time)
        )
        return html_content

    except Exception as e:
        return HEALTH_ERROR_HTML.format(error_message=str(e))


@router.get("/supported-types")
async def get_supported_types(config=None):
    """Get information about supported file types and validation config (using processor registry)."""
    from ingestion.processors.page_utils import get_page_number_for_position, get_supported_file_types, list_available_processors

    # Get supported types from processor registry (dynamic based on registered processors)
    supported_extensions = get_supported_file_types()
    processors = list_available_processors()

    return {
        "supported_extensions": supported_extensions,
        "max_file_size_mb": config.file_validator.config.max_file_size_mb,
        "supported_types": [ext.replace('.', '') for ext in supported_extensions],
        "registered_processors": [str(processor) for processor in processors],
        "vector_store_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "database_backend": "PostgreSQL + pgvector",
            "chunking_method": "semantic_chunking_with_chonkie",
            "processor_pattern": "Abstract Method + Factory Method"
        }
    }


@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    config=None,
):
    """Return the ingestion status of a document from the status DB."""
    try:
        repo = IngestionRepository(connection_string=config.connection_string)
        status = await repo.get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "document_id": status["id"],
            "file_name": status["file_name"],
            "stage": status["stage"],
            "attempts": status["attempts"],
            "chunk_count": status["chunk_count"],
            "last_error": status["last_error"],
            "created_at": status["created_at"],
            "updated_at": status["updated_at"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document status: {type(e).__name__}: {e}"
        )


@router.delete("/table/{table_name}")
async def delete_table(
    table_name: str,
    x_app_password: Optional[str] = Header(default=None),
    config=None,
    get_pipeline=None
):
    """Delete a specific table from the database (optimized for speed)."""
    require_access_password(x_app_password)
    validate_table_name(table_name)
    with logfire.span("table_deletion", table_name=table_name):
        logfire.info("Starting table deletion", table_name=table_name)

        try:
            pipeline_instance = await get_pipeline(table_name)
            conn = await pipeline_instance.vector_store._get_connection()

            try:
                repo = TableRepository(conn)

                with logfire.span("table_existence_check"):
                    table_exists = await repo.table_exists(table_name)
                    row_count = await repo.get_table_row_estimate(table_name)

                    logfire.info("Table existence check completed",
                                 table_exists=table_exists,
                                 estimated_rows=row_count)

                if not table_exists:
                    logfire.warn("Table deletion failed - table does not exist",
                                 table_name=table_name)
                    raise HTTPException(
                        status_code=404,
                        detail=f"Table '{table_name}' does not exist"
                    )

                with logfire.span("table_data_deletion"):
                    await repo.truncate_table(table_name)
                    logfire.info("Table data truncated successfully",
                                 table_name=table_name,
                                 rows_deleted=row_count)

                with logfire.span("table_schema_deletion"):
                    await repo.drop_table(table_name)
                    logfire.info("Table schema dropped successfully",
                                 table_name=table_name)

                if table_name == pipeline_instance.vector_store.table_name:
                    config.pipeline = None
                    logfire.info("Pipeline reset due to current table deletion",
                                 table_name=table_name)

                logfire.info("Table deletion completed successfully",
                             table_name=table_name,
                             estimated_rows_deleted=row_count)

                return {
                    "status": "success",
                    "message": f"Table '{table_name}' deleted successfully",
                    "table_name": table_name,
                    "estimated_rows_deleted": row_count,
                    "timestamp": str(uuid.uuid1().time)
                }

            finally:
                await pipeline_instance.vector_store._release_connection(conn)

        except HTTPException:
            raise
        except Exception as e:
            logfire.error("Table deletion failed with unexpected error",
                          table_name=table_name,
                          error=str(e),
                          error_type=type(e).__name__)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete table '{table_name}': {str(e)}"
            )
