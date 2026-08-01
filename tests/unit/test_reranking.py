"""
Unit tests for retrieval/reranking.py.

Covers:
- Reranker initialization (mocked CrossEncoder)
- Reranker.rerank (scoring, sorting, top_k, empty results)
- Reranker.get_score_statistics
- HybridScorer.weighted_average
- HybridScorer.reciprocal_rank_fusion
- RerankedResult dataclass
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import fields

from retrieval.reranking import Reranker, RerankedResult, HybridScorer


@pytest.fixture
def mock_cross_encoder():
    with patch('retrieval.reranking.CrossEncoder') as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        yield mock_model


@pytest.fixture
def reranker(mock_cross_encoder):
    return Reranker(model_name='test-model')


@pytest.fixture
def sample_results():
    return [
        {
            'chunk_id': '1',
            'text': 'Machine learning is great',
            'document_id': 'doc1',
            'metadata': {'page': 1},
            'similarity': 0.9,
        },
        {
            'chunk_id': '2',
            'text': 'Deep learning uses neural networks',
            'document_id': 'doc1',
            'metadata': {'page': 2},
            'similarity': 0.85,
        },
        {
            'chunk_id': '3',
            'text': 'The weather is sunny',
            'document_id': 'doc2',
            'metadata': {'page': 1},
            'similarity': 0.7,
        },
    ]


class TestRerankedResult:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(RerankedResult)}
        expected = {'chunk_id', 'text', 'document_id', 'metadata',
                    'similarity', 'rerank_score', 'original_rank', 'new_rank'}
        assert field_names == expected

    def test_creation(self):
        result = RerankedResult(
            chunk_id='1', text='text', document_id='doc1',
            metadata={}, similarity=0.9, rerank_score=0.95,
            original_rank=1, new_rank=1,
        )
        assert result.chunk_id == '1'
        assert result.rerank_score == 0.95


class TestRerankerInit:
    def test_model_name_stored(self, reranker):
        assert reranker.model_name == 'test-model'

    def test_model_loaded(self, reranker, mock_cross_encoder):
        assert reranker.model is not None


class TestRerankerRerank:
    def test_rerank_returns_results(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.7, 0.3])
        results = reranker.rerank("What is ML?", sample_results)
        assert len(results) == 3

    def test_rerank_sorted_by_score(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.3, 0.9, 0.7])
        results = reranker.rerank("What is ML?", sample_results)
        assert results[0].rerank_score >= results[1].rerank_score >= results[2].rerank_score

    def test_rerank_top_k(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.7, 0.3])
        results = reranker.rerank("What is ML?", sample_results, top_k=2)
        assert len(results) == 2

    def test_rerank_empty_results(self, reranker):
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_assigns_ranks(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.7, 0.3])
        results = reranker.rerank("query", sample_results)
        ranks = [r.new_rank for r in results]
        assert ranks == [1, 2, 3]

    def test_rerank_preserves_original_rank(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.3, 0.9, 0.7])
        results = reranker.rerank("query", sample_results)
        original_ranks = {r.chunk_id: r.original_rank for r in results}
        assert original_ranks['1'] == 1
        assert original_ranks['2'] == 2
        assert original_ranks['3'] == 3

    def test_return_all_ignores_top_k(self, reranker, mock_cross_encoder, sample_results):
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.7, 0.3])
        results = reranker.rerank("query", sample_results, top_k=1, return_all=True)
        assert len(results) == 3


class TestGetScoreStatistics:
    def test_statistics_computed(self, reranker):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.9, 0.95, 1, 1),
            RerankedResult('2', 't', 'd', {}, 0.8, 0.70, 2, 2),
            RerankedResult('3', 't', 'd', {}, 0.7, 0.30, 3, 3),
        ]
        stats = reranker.get_score_statistics(results)
        assert stats['min_score'] == 0.30
        assert stats['max_score'] == 0.95
        assert stats['avg_score'] == pytest.approx((0.95 + 0.70 + 0.30) / 3)

    def test_empty_results(self, reranker):
        stats = reranker.get_score_statistics([])
        assert stats['min_score'] == 0.0
        assert stats['max_score'] == 0.0
        assert stats['avg_score'] == 0.0


class TestHybridScorerWeightedAverage:
    def test_weighted_average_sorts(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.5, 0.9, 1, 2),
            RerankedResult('2', 't', 'd', {}, 0.9, 0.3, 2, 1),
        ]
        sorted_results = HybridScorer.weighted_average(
            results, similarity_weight=0.3, rerank_weight=0.7
        )
        assert sorted_results[0].metadata['combined_score'] >= sorted_results[1].metadata['combined_score']

    def test_weighted_average_updates_ranks(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.5, 0.9, 1, 2),
            RerankedResult('2', 't', 'd', {}, 0.9, 0.3, 2, 1),
        ]
        sorted_results = HybridScorer.weighted_average(results)
        assert sorted_results[0].new_rank == 1
        assert sorted_results[1].new_rank == 2

    def test_weights_normalized(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.8, 0.8, 1, 1),
        ]
        HybridScorer.weighted_average(results, similarity_weight=0.3, rerank_weight=0.7)
        expected = 0.8 * 0.3 + 0.8 * 0.7
        assert results[0].metadata['combined_score'] == pytest.approx(expected)


class TestHybridScorerRRF:
    def test_rrf_computes_scores(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.9, 0.8, 1, 1),
            RerankedResult('2', 't', 'd', {}, 0.8, 0.7, 2, 2),
        ]
        sorted_results = HybridScorer.reciprocal_rank_fusion(results, k=60)
        for r in sorted_results:
            assert 'rrf_score' in r.metadata

    def test_rrf_sorted_desc(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.9, 0.8, 1, 1),
            RerankedResult('2', 't', 'd', {}, 0.8, 0.7, 2, 2),
        ]
        sorted_results = HybridScorer.reciprocal_rank_fusion(results)
        assert sorted_results[0].metadata['rrf_score'] >= sorted_results[1].metadata['rrf_score']

    def test_rrf_updates_ranks(self):
        results = [
            RerankedResult('1', 't', 'd', {}, 0.9, 0.8, 1, 1),
        ]
        sorted_results = HybridScorer.reciprocal_rank_fusion(results)
        assert sorted_results[0].new_rank == 1
