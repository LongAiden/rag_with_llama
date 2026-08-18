"""
Document retrieval and search logic.
Handles vector search, BM25 reranking, and response generation.
"""

import asyncio
import re
import time
import logfire
from typing import List, Optional
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

from app.retrieval.utils import merge_with_rrf
from app.retrieval.llm_operations import generate_llm_response
from app.models.schemas import RAGResponse, RAGSource, RAGResponseMetadata
from app.infra.telemetry import InteractionPayload, log_interaction

# Strong references to in-flight fire-and-forget tasks. asyncio only keeps a weak
# reference to a running task, so without this the interaction log can be collected
# before it is written.
_BACKGROUND_TASKS: set = set()


async def perform_document_search(
    query: str,
    limit: int,
    threshold: float,
    pipeline,
    config,
    document_ids: Optional[List[str]] = None,
    table_name: str = "document_chunks",
    model: str = "gemini-2.5-flash",
    session_id: Optional[str] = None,
    enable_reranking: bool = True,
    rerank_top_k: int = 5,
    search_mode: str = "vector",
) -> RAGResponse:
    """
    Common document search logic with optional reranking.

    Args:
        query: Search query string
        limit: Maximum results on the non-reranked path. Ignored when
            enable_reranking=True — rerank_top_k controls the final count.
        threshold: Similarity threshold for filtering results
        pipeline: ChunkEmbeddingPipeline instance
        config: Application configuration object
        document_ids: Optional list of document IDs to filter by
        table_name: Database table name
        search_mode: "vector" for vector-only, "hybrid" for vector+BM25+RRF

    Returns:
        RAGResponse with answer, sources, and metadata
    """
    if session_id:
        langfuse_context.update_current_trace(
            session_id=session_id,
            metadata={"table_name": table_name},
        )

    with logfire.span("document_search",
                     query=query[:100],
                     limit=limit,
                     threshold=threshold,
                     table_name=table_name,
                     search_mode=search_mode):

        # Candidate depth for the reranker. Guarded because rerank_top_k slices
        # AFTER predict(): a depth below top_k would silently return fewer than
        # the caller asked for.
        candidate_depth = max(config.settings.vector_search_limit, rerank_top_k)
        logfire.info("Retrieval knobs",
                     candidate_depth=candidate_depth,
                     rerank_top_k=rerank_top_k,
                     rerank_model=config.settings.rerank_model,
                     rerank_max_length=config.settings.rerank_max_length)
        with logfire.span("embedding_generation_for_search"):
            logfire.info("Generating embeddings for search query",
                        query_length=len(query),
                        embedding_model=pipeline.embedding_generator.model_name)

            vector_results = await pipeline.search_documents(
                query=query,
                limit=candidate_depth,
                threshold=threshold,
                document_ids=document_ids
            )

            logfire.info("Vector search completed",
                        results_found=len(vector_results),
                        avg_similarity=sum(r['similarity'] for r in vector_results) / len(vector_results) if vector_results else 0)

        if search_mode == "hybrid":
            with logfire.span("bm25_retrieval"):
                bm25_results = await pipeline.vector_store.search_bm25(
                    query=query,
                    limit=candidate_depth,
                    document_ids=document_ids,
                )
                logfire.info("BM25 search completed", results_found=len(bm25_results))

            with logfire.span("rrf_merge"):
                merged_results = merge_with_rrf(vector_results, bm25_results)
                logfire.info("RRF merge completed", merged_count=len(merged_results))
        else:
            merged_results = vector_results

        avg_rerank_score = None
        reranking_enabled = enable_reranking

        if enable_reranking and merged_results:
            with logfire.span("cross_encoder_reranking"):
                try:
                    from app.retrieval.utils import get_reranker
                    reranker = await asyncio.to_thread(get_reranker, config)
                    original_by_id = {r['chunk_id']: r for r in merged_results}
                    reranked = await asyncio.to_thread(
                        reranker.rerank,
                        query=query,
                        results=merged_results,
                        top_k=rerank_top_k,
                    )
                    # RerankResult carries only the fields the reranker itself needs,
                    # so anything not restored from original_by_id here is silently
                    # dropped from the response. doc_name joins bm25_score/rrf_score
                    # for exactly that reason.
                    merged_results = [{
                        'chunk_id': r.chunk_id,
                        'text': r.text,
                        'document_id': r.document_id,
                        'doc_name': original_by_id[r.chunk_id].get('doc_name'),
                        'metadata': r.metadata,
                        'similarity': r.similarity,
                        'bm25_score': original_by_id[r.chunk_id].get('bm25_score', 0.0),
                        'rrf_score': original_by_id[r.chunk_id].get('rrf_score', 0.0),
                        'rerank_score': r.rerank_score,
                    } for r in reranked]
                    avg_rerank_score = sum(r['rerank_score'] for r in merged_results) / len(merged_results) if merged_results else None
                    logfire.info("Cross-encoder reranking completed",
                               final_results=len(merged_results),
                               avg_rerank_score=avg_rerank_score)
                except Exception as e:
                    logfire.error("Cross-encoder reranking failed, using vector scores only", error=str(e))
                    reranking_enabled = False
                    merged_results = merged_results[:rerank_top_k]
        else:
            merged_results = merged_results[:limit]

        results = merged_results

        search_method = "vector" + ("_crossencoder" if reranking_enabled else "")
        if search_mode == "hybrid":
            search_method = "hybrid_bm25_vector" + ("_crossencoder" if reranking_enabled else "")

        if not results:
            no_results_msg = "No relevant documents found with the specified similarity threshold."
            return RAGResponse(
                query=query,
                answer=no_results_msg,
                sources=[],
                search_stats=RAGResponseMetadata(
                    chunks_found=0,
                    avg_similarity=0.0,
                    search_method=search_method,
                    threshold_used=threshold,
                    word_count=len(no_results_msg.split()),
                    confidence=0.0,
                    reranking_enabled=reranking_enabled,
                    avg_rerank_score=avg_rerank_score
                ),
                table_used=table_name
            )

        _STRUCTURAL_RE = re.compile(
            r'\b(how many|list all|all the|count|enumerate|what are the|steps in|'
            r'number of|how much|summarize all|every)\b',
            re.IGNORECASE
        )
        section_context_blocks = []
        if _STRUCTURAL_RE.search(query):
            seen_section_doc = set()
            for r in results:
                sp = (r.get('metadata') or {}).get('section_path', '')
                doc_id = r.get('document_id', '')
                if sp and (sp, doc_id) not in seen_section_doc:
                    seen_section_doc.add((sp, doc_id))
                    siblings = await pipeline.vector_store.get_chunks_by_section(
                        section_path=sp,
                        document_ids=[doc_id],
                        limit=15,
                    )
                    if siblings:
                        combined = "\n\n".join(s['text'] for s in siblings)
                        sib_name = siblings[0].get('doc_name') or r.get('doc_name')
                        doc_label = f" — {sib_name}" if sib_name else ""
                        section_context_blocks.append(
                            f"[Section context{doc_label}: {sp}]\n{combined}"
                        )
            logfire.info("Sibling expansion",
                         sections_expanded=len(section_context_blocks))

        with logfire.span("context_building"):
            context_parts = []

            context_parts.extend(section_context_blocks)

            seen_page_contexts: set = set()

            for i, result in enumerate(results):
                page_info = ""
                page_num = (result.get('metadata') or {}).get('page_number')
                if page_num is not None:
                    page_info = f" (Page {page_num})"

                chunk_text = result['text']
                page_content = (result.get('metadata') or {}).get('page_content', '')

                # Name the source in the prompt so the model can attribute in prose
                # ("According to Linear Algebra, ...") instead of the UI being the
                # only place provenance shows up.
                doc_name_label = result.get('doc_name')
                source_label = f"Source {i+1}" + (f" — {doc_name_label}" if doc_name_label else "")

                doc_id = result.get('document_id', '')
                page_key = (doc_id, page_num if page_num is not None else 'no_page')
                if (
                    page_content
                    and page_content.strip() != chunk_text.strip()
                    and page_key not in seen_page_contexts
                ):
                    source_block = (
                        f"[{source_label}{page_info}]\n"
                        f"[Matched chunk]: {chunk_text}\n"
                        f"[Full page context]:\n{page_content}"
                    )
                    seen_page_contexts.add(page_key)
                else:
                    source_block = f"[{source_label}{page_info}]: {chunk_text}"

                context_parts.append(source_block)

            context = "\n\n---\n\n".join(context_parts)

            logfire.info("Context built",
                        total_context_parts=len(context_parts),
                        context_length=len(context))

        t0 = time.monotonic()
        llm_response = await generate_llm_response(query, context, results, model=model)
        latency_ms = int((time.monotonic() - t0) * 1000)

        avg_similarity = sum(r['similarity'] for r in results) / len(results)

        _task = asyncio.create_task(log_interaction(
            InteractionPayload(
                question=query,
                answer=llm_response.answer,
                model=model,
                backend=llm_response.metadata.get("method", "unknown"),
                latency_ms=latency_ms,
                sources_used=len(results),
                table_name=table_name,
                rerank_method="cross_encoder" if reranking_enabled else "vector_only",
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
                session_id=session_id,
            ),
            config.connection_string,
        ))
        _BACKGROUND_TASKS.add(_task)
        _task.add_done_callback(_BACKGROUND_TASKS.discard)

        return RAGResponse(
            query=query,
            answer=llm_response.answer,
            sources=[
                RAGSource(
                    chunk_id=r['chunk_id'],
                    text=r['text'],
                    similarity=round(r['similarity'], 3),
                    document_id=r['document_id'],
                    doc_name=r.get('doc_name'),
                    page_number=(r.get('metadata') or {}).get('page_number'),
                    metadata=r.get('metadata') or {},
                    rerank_score=round(r['rerank_score'], 3) if 'rerank_score' in r else None,
                    bm25_score=round(r.get('bm25_score', 0.0), 3),
                    rrf_score=round(r.get('rrf_score', 0.0), 3),
                ) for r in results
            ],
            search_stats=RAGResponseMetadata(
                chunks_found=len(results),
                avg_similarity=round(avg_similarity, 3),
                search_method=search_method,
                threshold_used=threshold,
                word_count=llm_response.word_count,
                confidence=llm_response.confidence,
                reranking_enabled=reranking_enabled,
                avg_rerank_score=round(avg_rerank_score, 3) if avg_rerank_score else None,
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
            ),
            table_used=table_name
        )
