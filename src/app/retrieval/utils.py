"""
Utility functions for the RAG application.
Rank fusion (RRF) and lazy reranker construction.
"""

import threading
from typing import List, Dict

from app.retrieval.reranking import Reranker


def merge_with_rrf(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    Args:
        vector_results: Results from pgvector (sorted by similarity desc)
        bm25_results: Results from BM25 (sorted by bm25_score desc)
        k: RRF constant (default 60)

    Returns:
        Merged list sorted by rrf_score descending, deduplicated by chunk_id
    """
    vector_ranks = {r['chunk_id']: i + 1 for i, r in enumerate(vector_results)}
    bm25_ranks = {r['chunk_id']: i + 1 for i, r in enumerate(bm25_results)}

    all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    lookup = {r['chunk_id']: r for r in vector_results}
    for r in bm25_results:
        if r['chunk_id'] not in lookup:
            lookup[r['chunk_id']] = r
    bm25_scores_by_id = {r['chunk_id']: r.get('bm25_score', 0.0) for r in bm25_results}

    merged = []
    for cid in all_chunk_ids:
        base = lookup[cid].copy()
        vec_rank = vector_ranks.get(cid, float('inf'))
        bm25_rank = bm25_ranks.get(cid, float('inf'))
        rrf_score = (1.0 / (k + vec_rank)) + (1.0 / (k + bm25_rank))

        base['rrf_score'] = rrf_score
        base['bm25_score'] = bm25_scores_by_id.get(cid, 0.0)
        base.setdefault('similarity', 0.0)
        merged.append(base)

    merged.sort(key=lambda x: x['rrf_score'], reverse=True)
    return merged


_RERANKER_LOCK = threading.Lock()


def get_reranker(config) -> Reranker:
    """
    Get or initialize the reranker (lazy initialization).

    Loading a CrossEncoder takes seconds, so callers run this in a worker thread.
    The lock keeps two concurrent first-calls from each loading their own copy.

    Args:
        config: Application configuration object

    Returns:
        Reranker instance
    """
    if config.reranker is not None:
        return config.reranker

    with _RERANKER_LOCK:
        if config.reranker is None:
            # From AppSettings, not os.getenv: pydantic-settings also reads .env,
            # which a bare os.getenv would silently ignore.
            rerank_model = config.settings.rerank_model
            config.reranker = Reranker(
                model_name=rerank_model,
                max_length=config.settings.rerank_max_length,
            )
            print(f"✓ Reranker initialized with model: {rerank_model}")
    return config.reranker


def preload_reranker(config) -> None:
    """Eagerly construct the cross-encoder. Call from app startup in a thread.

    Thin wrapper around get_reranker: the lock inside get_reranker already
    serialises first construction, so no extra guard is needed here. The
    preload_reranker setting is checked by the caller (app.py lifespan), not
    here — this function is a plain "do it now" primitive.
    """
    get_reranker(config)
