"""API routes for the chat UI and RAG queries."""
from typing import Optional, List

from fastapi import APIRouter, Depends, Form, Header, HTTPException
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_config, get_pipeline_factory
from app.api.renderer import render
from app.api.validators import require_access_password, validate_table_name
from app.config.app_config import DEFAULT_TABLE_NAME
from app.infra.db import DomainRepository
from app.models.schemas import QueryRequest, RAGResponse
from app.retrieval.search import perform_document_search

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    langfuse_context = type("_Noop", (), {
        "update_current_trace": staticmethod(lambda **_: None),
        "update_current_observation": staticmethod(lambda **_: None),
    })()
    def observe(**__):
        def decorator(fn):
            return fn
        return decorator


router = APIRouter()


async def _resolve_search_target(
    config,
    domain: Optional[str],
    table_name: Optional[str],
    doc_name: Optional[str],
    document_ids: Optional[List[str]],
) -> tuple[str, Optional[List[str]]]:
    """Turn (domain, table_name, doc_name) into a chunk table and a document filter.

    `domain` wins over `table_name` when both are sent — the registry knows which
    physical table a domain lives in, the caller may not. `doc_name` is resolved to
    document ids here rather than pushed into the SQL: a name can match several
    uploads, and ids are what the search path already filters on.

    Raises HTTPException, so callers must invoke this outside the try block that
    turns errors into 500s.
    """
    table = (table_name or DEFAULT_TABLE_NAME).strip() or DEFAULT_TABLE_NAME
    domain = (domain or "").strip()
    repo = DomainRepository(connection_string=config.connection_string)

    if domain:
        validate_table_name(domain)
        row = await repo.get_domain(domain)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Domain '{domain}' does not exist")
        table = row["table_name"]

    validate_table_name(table)

    doc_name = (doc_name or "").strip()
    if doc_name and not document_ids:
        document_ids = await repo.find_document_ids_by_name(doc_name, scope=table)
        if not document_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No document named '{doc_name}' in '{table}'",
            )

    return table, document_ids


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
    search_mode: str = "vector",
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
        search_mode=search_mode,
    )


@router.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload and search forms."""
    return render("home.html")


@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: QueryRequest,
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
    get_pipeline=Depends(get_pipeline_factory),
):
    """Query documents using pgvector similarity search + LLM generation."""
    require_access_password(x_app_password)
    table_name, document_ids = await _resolve_search_target(
        config,
        domain=request.domain,
        table_name=request.table_name,
        doc_name=request.doc_name,
        document_ids=request.document_ids,
    )
    try:
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
                document_ids=document_ids,
                config=config,
                get_pipeline=get_pipeline,
                enable_reranking=request.enable_reranking,
                rerank_top_k=request.rerank_top_k or 5,
                search_mode=request.search_mode,
            )

        result = await _run_search(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query-form", response_class=HTMLResponse)
async def query_documents_form(
    query: str = Form(...),
    limit: int = Form(5),
    threshold: float = Form(0.3),
    table_name: str = Form(DEFAULT_TABLE_NAME),
    domain: str = Form(""),
    doc_name: str = Form(""),
    model: str = Form("gemini-2.5-flash"),
    search_mode: str = Form("vector"),
    access_password: Optional[str] = Form(None),
    config=Depends(get_config),
    get_pipeline=Depends(get_pipeline_factory),
):
    """Query documents using form data (for HTML form submission)."""
    require_access_password(access_password)
    table_name, document_ids = await _resolve_search_target(
        config,
        domain=domain,
        table_name=table_name,
        doc_name=doc_name,
        document_ids=None,
    )
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
                document_ids=document_ids,
                config=config,
                get_pipeline=get_pipeline,
                enable_reranking=True,
                rerank_top_k=5,
                search_mode=search_mode,
            )

        result = await _run_search(query)

        sources = [
            {
                "index": i + 1,
                "similarity": source.similarity,
                "bm25_score": source.bm25_score,
                "rrf_score": source.rrf_score,
                "rerank_score": source.rerank_score,
                "doc_name": source.doc_name or "Unknown",
                "document_id": source.document_id[:8] if source.document_id else "Unknown",
                "page_number": source.page_number or 'N/A',
                "text": source.text,
            }
            for i, source in enumerate(result.sources)
        ]

        stats = result.search_stats
        input_tok = stats.input_tokens
        output_tok = stats.output_tokens
        total_tok = stats.total_tokens
        token_display = (
            f"↑ {input_tok:,} in · ↓ {output_tok:,} out · Σ {total_tok:,} total"
            if total_tok else "N/A"
        )

        return render(
            "search_results.html",
            query=query,
            answer=result.answer.strip(),
            source_count=len(result.sources),
            sources=sources,
            chunks_found=stats.chunks_found,
            avg_similarity=f"{stats.avg_similarity:.1%}",
            search_method=stats.search_method,
            table_used=result.table_used,
            threshold_used=f"{stats.threshold_used:.1%}",
            confidence=f"{stats.confidence:.1%}" if stats.confidence else "N/A",
            word_count=stats.word_count or 0,
            graph_enriched="Yes" if stats.graph_enriched else "No",
            token_display=token_display,
        )

    except Exception as e:
        error_msg = str(e)
        if any(indicator in error_msg.lower() for indicator in ["rate limit", "quota exceeded", "429", "resource exhausted"]):
            error_msg = (
                f"⚠️ API Rate Limit Reached\n\n"
                f"The API rate limit has been exceeded.\n\n"
                f"Please try again in a minute or two.\n\n"
                f"Technical details: {error_msg}"
            )
        return render("search_error.html", error_message=error_msg)
