"""
Reranking module for improving RAG retrieval quality.

This module provides reranking functionality using cross-encoder models
to refine the initial retrieval results from pgvector similarity search.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import CrossEncoder
import logfire


@dataclass
class RerankedResult:
    """Reranked search result with scores."""
    chunk_id: str
    text: str
    document_id: str
    metadata: Dict[str, Any]
    similarity: float  # Original similarity score from pgvector
    rerank_score: float  # New score from cross-encoder
    original_rank: int  # Position before reranking
    new_rank: int  # Position after reranking


class Reranker:
    """
    Reranker using cross-encoder models for semantic relevance scoring.

    Cross-encoders process query and document together, providing more accurate
    relevance scores than bi-encoder similarity alone.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize reranker with a cross-encoder model.

        Args:
            model_name: Cross-encoder model name. Popular options:
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, good quality)
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (slower, better quality)
                - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (fastest, lower quality)
        """
        self.model_name = model_name
        logfire.info("Initializing reranker", model_name=model_name)

        try:
            self.model = CrossEncoder(model_name)
            logfire.info("Reranker initialized successfully", model_name=model_name)
        except Exception as e:
            logfire.error("Failed to initialize reranker",
                         model_name=model_name,
                         error=str(e))
            raise

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_all: bool = False
    ) -> List[RerankedResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: The search query
            results: List of search results from pgvector with fields:
                - chunk_id, text, document_id, metadata, similarity
            top_k: Number of top results to return (None = return all)
            return_all: If True, return all results even if top_k is specified

        Returns:
            List of RerankedResult objects sorted by rerank_score (descending)
        """
        if not results:
            return []

        with logfire.span("reranking",
                         query=query[:100],
                         num_results=len(results),
                         top_k=top_k):

            # Prepare query-document pairs for cross-encoder
            pairs = [[query, result['text']] for result in results]

            # Get rerank scores
            logfire.info("Computing rerank scores",
                        num_pairs=len(pairs),
                        model=self.model_name)

            rerank_scores = self.model.predict(pairs)

            # Create reranked results
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, rerank_scores)):
                reranked_results.append({
                    'chunk_id': result['chunk_id'],
                    'text': result['text'],
                    'document_id': result['document_id'],
                    'metadata': result.get('metadata', {}),
                    'similarity': result['similarity'],
                    'rerank_score': float(score),
                    'original_rank': i + 1
                })

            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Assign new ranks
            for i, result in enumerate(reranked_results):
                result['new_rank'] = i + 1

            # Convert to RerankedResult objects
            final_results = [
                RerankedResult(
                    chunk_id=r['chunk_id'],
                    text=r['text'],
                    document_id=r['document_id'],
                    metadata=r['metadata'],
                    similarity=r['similarity'],
                    rerank_score=r['rerank_score'],
                    original_rank=r['original_rank'],
                    new_rank=r['new_rank']
                )
                for r in reranked_results
            ]

            # Apply top_k filtering if specified
            if top_k is not None and not return_all:
                final_results = final_results[:top_k]

            # Log reranking statistics
            if final_results:
                avg_rerank_score = sum(r.rerank_score for r in final_results) / len(final_results)
                rank_changes = sum(1 for r in final_results if r.original_rank != r.new_rank)

                logfire.info("Reranking completed",
                           results_returned=len(final_results),
                           avg_rerank_score=avg_rerank_score,
                           rank_changes=rank_changes,
                           top_score=final_results[0].rerank_score if final_results else None)

            return final_results

    def get_score_statistics(self, results: List[RerankedResult]) -> Dict[str, float]:
        """
        Calculate statistics about reranking scores.

        Args:
            results: List of reranked results

        Returns:
            Dictionary with score statistics
        """
        if not results:
            return {
                'min_score': 0.0,
                'max_score': 0.0,
                'avg_score': 0.0,
                'median_score': 0.0
            }

        scores = [r.rerank_score for r in results]
        scores_sorted = sorted(scores)

        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_score': sum(scores) / len(scores),
            'median_score': scores_sorted[len(scores_sorted) // 2]
        }


class HybridScorer:
    """
    Combine similarity scores and rerank scores using different strategies.
    """

    @staticmethod
    def weighted_average(
        results: List[RerankedResult],
        similarity_weight: float = 0.3,
        rerank_weight: float = 0.7
    ) -> List[RerankedResult]:
        """
        Combine scores using weighted average.

        Args:
            results: List of reranked results
            similarity_weight: Weight for original similarity score (0-1)
            rerank_weight: Weight for rerank score (0-1)

        Returns:
            Results sorted by combined score
        """
        # Normalize weights
        total_weight = similarity_weight + rerank_weight
        similarity_weight /= total_weight
        rerank_weight /= total_weight

        # Calculate combined scores
        for result in results:
            combined_score = (
                result.similarity * similarity_weight +
                result.rerank_score * rerank_weight
            )
            # Store combined score in metadata
            result.metadata['combined_score'] = combined_score

        # Sort by combined score
        results.sort(key=lambda x: x.metadata['combined_score'], reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.new_rank = i + 1

        return results

    @staticmethod
    def reciprocal_rank_fusion(
        results: List[RerankedResult],
        k: int = 60
    ) -> List[RerankedResult]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).

        Args:
            results: List of reranked results
            k: Constant for RRF formula (typically 60)

        Returns:
            Results sorted by RRF score
        """
        # Calculate RRF scores
        for result in results:
            # RRF score = 1/(k + rank_similarity) + 1/(k + rank_rerank)
            # We need to create rankings based on scores
            similarity_rank = result.original_rank
            rerank_rank = result.new_rank

            rrf_score = (
                1.0 / (k + similarity_rank) +
                1.0 / (k + rerank_rank)
            )
            result.metadata['rrf_score'] = rrf_score

        # Sort by RRF score
        results.sort(key=lambda x: x.metadata['rrf_score'], reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.new_rank = i + 1

        return results


# Example usage and testing
if __name__ == "__main__":
    # Example mock results for testing
    mock_results = [
        {
            'chunk_id': '1',
            'text': 'Machine learning is a subset of artificial intelligence.',
            'document_id': 'doc1',
            'metadata': {'page': 1},
            'similarity': 0.85
        },
        {
            'chunk_id': '2',
            'text': 'Deep learning uses neural networks with multiple layers.',
            'document_id': 'doc1',
            'metadata': {'page': 2},
            'similarity': 0.82
        },
        {
            'chunk_id': '3',
            'text': 'The weather today is sunny and warm.',
            'document_id': 'doc2',
            'metadata': {'page': 1},
            'similarity': 0.80
        }
    ]

    # Initialize reranker
    reranker = Reranker()

    # Test query
    query = "What is machine learning?"

    # Rerank results
    reranked = reranker.rerank(query, mock_results, top_k=2)

    print(f"\nQuery: {query}\n")
    print("Reranked Results:")
    for result in reranked:
        print(f"Rank {result.new_rank} (was {result.original_rank}): "
              f"Similarity={result.similarity:.3f}, "
              f"Rerank Score={result.rerank_score:.3f}")
        print(f"  Text: {result.text[:80]}...")

    # Get statistics
    stats = reranker.get_score_statistics(reranked)
    print(f"\nScore Statistics: {stats}")
