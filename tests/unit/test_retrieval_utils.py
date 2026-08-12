"""
Unit tests for retrieval/utils.py.

Covers:
- merge_with_rrf (fusion, deduplication, sorting)
"""
import pytest
import numpy as np

from app.retrieval.utils import merge_with_rrf


class TestMergeWithRRF:
    def test_merges_two_lists(self):
        vector_results = [
            {"chunk_id": "1", "text": "a", "similarity": 0.9},
            {"chunk_id": "2", "text": "b", "similarity": 0.8},
        ]
        bm25_results = [
            {"chunk_id": "2", "text": "b", "bm25_score": 5.0},
            {"chunk_id": "3", "text": "c", "bm25_score": 3.0},
        ]
        merged = merge_with_rrf(vector_results, bm25_results)
        assert len(merged) == 3

    def test_deduplicates_by_chunk_id(self):
        vector_results = [
            {"chunk_id": "1", "text": "a", "similarity": 0.9},
        ]
        bm25_results = [
            {"chunk_id": "1", "text": "a", "bm25_score": 5.0},
        ]
        merged = merge_with_rrf(vector_results, bm25_results)
        assert len(merged) == 1
        assert merged[0]["chunk_id"] == "1"

    def test_sorted_by_rrf_score_desc(self):
        vector_results = [
            {"chunk_id": "1", "text": "a", "similarity": 0.9},
            {"chunk_id": "2", "text": "b", "similarity": 0.8},
        ]
        bm25_results = [
            {"chunk_id": "1", "text": "a", "bm25_score": 5.0},
            {"chunk_id": "2", "text": "b", "bm25_score": 3.0},
        ]
        merged = merge_with_rrf(vector_results, bm25_results)
        for i in range(len(merged) - 1):
            assert merged[i]["rrf_score"] >= merged[i + 1]["rrf_score"]

    def test_rrf_score_present(self):
        vector_results = [{"chunk_id": "1", "text": "a", "similarity": 0.9}]
        bm25_results = [{"chunk_id": "1", "text": "a", "bm25_score": 5.0}]
        merged = merge_with_rrf(vector_results, bm25_results)
        assert "rrf_score" in merged[0]

    def test_empty_both_lists(self):
        merged = merge_with_rrf([], [])
        assert merged == []

    def test_empty_vector_results(self):
        bm25_results = [{"chunk_id": "1", "text": "a", "bm25_score": 5.0}]
        merged = merge_with_rrf([], bm25_results)
        assert len(merged) == 1

    def test_empty_bm25_results(self):
        vector_results = [{"chunk_id": "1", "text": "a", "similarity": 0.9}]
        merged = merge_with_rrf(vector_results, [])
        assert len(merged) == 1

    def test_custom_k_parameter(self):
        vector_results = [{"chunk_id": "1", "text": "a", "similarity": 0.9}]
        bm25_results = [{"chunk_id": "1", "text": "a", "bm25_score": 5.0}]
        merged_default = merge_with_rrf(vector_results, bm25_results, k=60)
        merged_custom = merge_with_rrf(vector_results, bm25_results, k=10)
        assert merged_default[0]["rrf_score"] != merged_custom[0]["rrf_score"]
