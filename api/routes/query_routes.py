"""API routes for the chat UI and RAG queries."""
from typing import Optional, List

from fastapi import APIRouter, Form, Header, HTTPException
from fastapi.responses import HTMLResponse

from api.renderer import render
from api.validators import require_access_password
from config.app_config import DEFAULT_TABLE_NAME
from models.schemas import QueryRequest, RAGResponse
from retrieval.search import perform_document_search

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


@router.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload and search forms."""
    return render("home.html")


@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: QueryRequest,
    x_app_password: Optional[str] = Header(default=None),
    config=None,
    get_pipeline=None,
):
    """Query documents using pgvector similarity search + LLM generation."""
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
    config=None,
    get_pipeline=None,
):
    """Query documents using form data (for HTML form submission)."""
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

        sources_html = ''.join([f"""
        <div class="source-item">
            <strong>Source {i+1}</strong> (Similarity: {source.similarity:.1%}{f", BM25: {source.bm25_score:.3f}" if source.bm25_score else ""}{f", RRF: {source.rrf_score:.3f}" if source.rrf_score else ""}{f", Rerank: {source.rerank_score:.3f}" if source.rerank_score is not None else ""})<br>
            <em>Document: {source.document_id[:8]}... | Page: {source.page_number or 'N/A'}</em><br><br>
            <div style="white-space: pre-wrap; word-wrap: break-word;">{source.text}</div>
        </div>
        """ for i, source in enumerate(result.sources)])

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
            sources_html=sources_html,
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
