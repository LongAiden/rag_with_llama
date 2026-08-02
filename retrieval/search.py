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

from retrieval.utils import merge_with_rrf
from retrieval.llm_operations import generate_llm_response
from models.models import RAGResponse, RAGSource, RAGResponseMetadata
from infra.telemetry import InteractionPayload, log_interaction


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
) -> RAGResponse:
    """
    Common document search logic with optional reranking.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        threshold: Similarity threshold for filtering results
        pipeline: ChunkEmbeddingPipeline instance
        config: Application configuration object
        document_ids: Optional list of document IDs to filter by
        table_name: Database table name

    Returns:
        RAGResponse with answer, sources, and metadata
    """
    # Attach session_id to the active Langfuse trace (created by the route wrapper)
    if session_id:
        langfuse_context.update_current_trace(
            session_id=session_id,
            metadata={"table_name": table_name},
        )

    with logfire.span("document_search",
                     query=query[:100],  # Truncate long queries for logging
                     limit=limit,
                     threshold=threshold,
                     table_name=table_name):

        # Step 1: pgvector similarity search (semantic retrieval)
        HYBRID_LIMIT = 20
        with logfire.span("embedding_generation_for_search"):
            logfire.info("Generating embeddings for search query",
                        query_length=len(query),
                        embedding_model=pipeline.embedding_generator.model_name)

            vector_results = await pipeline.search_documents(
                query=query,
                limit=HYBRID_LIMIT,
                threshold=threshold,
                document_ids=document_ids
            )

            logfire.info("Vector search completed",
                        results_found=len(vector_results),
                        avg_similarity=sum(r['similarity'] for r in vector_results) / len(vector_results) if vector_results else 0)

        # Step 2: BM25 search (lexical retrieval)
        with logfire.span("bm25_retrieval"):
            bm25_results = await pipeline.vector_store.search_bm25(
                query=query,
                limit=HYBRID_LIMIT,
                document_ids=document_ids,
            )
            logfire.info("BM25 search completed", results_found=len(bm25_results))

        # Step 3: RRF merge of vector + BM25 results
        with logfire.span("rrf_merge"):
            merged_results = merge_with_rrf(vector_results, bm25_results)
            logfire.info("RRF merge completed", merged_count=len(merged_results))

        # Step 4: Cross-encoder reranking (if enabled)
        avg_rerank_score = None
        reranking_enabled = enable_reranking

        if enable_reranking and merged_results:
            with logfire.span("cross_encoder_reranking"):
                try:
                    from retrieval.utils import get_reranker
                    reranker = get_reranker(config)
                    original_by_id = {r['chunk_id']: r for r in merged_results}
                    reranked = reranker.rerank(
                        query=query,
                        results=merged_results,
                        top_k=rerank_top_k,
                    )
                    merged_results = [{
                        'chunk_id': r.chunk_id,
                        'text': r.text,
                        'document_id': r.document_id,
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
                    logfire.error("Cross-encoder reranking failed, using RRF scores only", error=str(e))
                    reranking_enabled = False
                    merged_results = merged_results[:rerank_top_k]
        else:
            merged_results = merged_results[:limit]

        results = merged_results

        if not results:
            no_results_msg = "No relevant documents found with the specified similarity threshold."
            return RAGResponse(
                query=query,
                answer=no_results_msg,
                sources=[],
                search_stats=RAGResponseMetadata(
                    chunks_found=0,
                    avg_similarity=0.0,
                    search_method="hybrid_bm25_vector" + ("_crossencoder" if reranking_enabled else ""),
                    threshold_used=threshold,
                    word_count=len(no_results_msg.split()),
                    confidence=0.0,
                    reranking_enabled=reranking_enabled,
                    avg_rerank_score=avg_rerank_score
                ),
                table_used=table_name
            )

        # Step 1.6: Sibling expansion for structural queries (how many, list all, count…)
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
                        section_context_blocks.append(
                            f"[Section context: {sp}]\n{combined}"
                        )
            logfire.info("Sibling expansion",
                         sections_expanded=len(section_context_blocks))

        # Step 2: Build context from retrieved chunks with page numbers and full page content
        with logfire.span("context_building"):
            context_parts = []

            # Prepend section context blocks so the LLM sees complete sections first
            context_parts.extend(section_context_blocks)

            # Track which page contexts have already been included to avoid duplicates
            seen_page_contexts: set = set()

            for i, result in enumerate(results):
                page_info = ""
                page_num = (result.get('metadata') or {}).get('page_number')
                if page_num is not None:
                    page_info = f" (Page {page_num})"

                chunk_text = result['text']
                page_content = (result.get('metadata') or {}).get('page_content', '')

                # Include the full page context block only when it adds new information
                # and we haven't already emitted the same page context for this document
                doc_id = result.get('document_id', '')
                page_key = (doc_id, page_num if page_num is not None else 'no_page')
                if (
                    page_content
                    and page_content.strip() != chunk_text.strip()
                    and page_key not in seen_page_contexts
                ):
                    source_block = (
                        f"[Source {i+1}{page_info}]\n"
                        f"[Matched chunk]: {chunk_text}\n"
                        f"[Full page context]:\n{page_content}"
                    )
                    seen_page_contexts.add(page_key)
                else:
                    source_block = f"[Source {i+1}{page_info}]: {chunk_text}"

                context_parts.append(source_block)

            context = "\n\n---\n\n".join(context_parts)

            logfire.info("Context built",
                        total_context_parts=len(context_parts),
                        context_length=len(context))

        # Step 3: Generate response with LLM
        t0 = time.monotonic()
        llm_response = await generate_llm_response(query, context, results, config.agent, model=model)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Calculate search statistics
        avg_similarity = sum(r['similarity'] for r in results) / len(results)

        # Fire-and-forget: persist interaction to llm_interactions table (and Langfuse if configured)
        asyncio.create_task(log_interaction(
            InteractionPayload(
                question=query,
                answer=llm_response.answer,
                model=model,
                backend=llm_response.metadata.get("method", "unknown"),
                latency_ms=latency_ms,
                sources_used=len(results),
                table_name=table_name,
                rerank_method="cross_encoder" if reranking_enabled else "rrf_only",
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
                session_id=session_id,
            ),
            config.connection_string,
        ))

        # Create structured response
        return RAGResponse(
            query=query,
            answer=llm_response.answer,
            sources=[
                RAGSource(
                    chunk_id=r['chunk_id'],
                    text=r['text'],
                    similarity=round(r['similarity'], 3),
                    document_id=r['document_id'],
                    page_number=r.get('metadata', {}).get('page_number'),
                    metadata=r.get('metadata', {}),
                    rerank_score=round(r['rerank_score'], 3) if 'rerank_score' in r else None,
                    bm25_score=round(r.get('bm25_score', 0.0), 3),
                    rrf_score=round(r.get('rrf_score', 0.0), 3),
                ) for r in results
            ],
            search_stats=RAGResponseMetadata(
                chunks_found=len(results),
                avg_similarity=round(avg_similarity, 3),
                search_method="hybrid_bm25_vector" + ("_crossencoder" if reranking_enabled else ""),
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
