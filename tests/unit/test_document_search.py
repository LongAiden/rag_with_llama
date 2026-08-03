"""
Unit tests for retrieval.search.perform_document_search.

This is the function that orchestrates the whole query path (vector search →
BM25 → RRF → rerank → context build → LLM). It previously had no coverage at
all, which is how a reference to a non-existent `config.agent` attribute
survived: it only fires when the search returns at least one chunk, so the
empty-result path kept working and hid it.

Everything external (pgvector, BM25, cross-encoder, LLM, telemetry) is mocked,
so these tests are fast and need no database or model weights.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import RAGResponse, SimpleRAGResponse
from app.retrieval.search import perform_document_search


def _chunk(chunk_id: str, text: str, similarity: float = 0.9, page: int = 1):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document_id": "doc-1",
        "metadata": {"page_number": page, "page_content": f"full page {page}", "section_path": ""},
        "similarity": similarity,
    }


@pytest.fixture
def pipeline():
    """A ChunkEmbeddingPipeline stand-in returning two vector hits and one BM25 hit."""
    vector_store = MagicMock()
    vector_store.search_bm25 = AsyncMock(return_value=[
        {**_chunk("c2", "second chunk"), "bm25_score": 3.2},
    ])
    vector_store.get_chunks_by_section = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.embedding_generator = SimpleNamespace(model_name="all-MiniLM-L6-v2", embedding_dim=384)
    pipe.vector_store = vector_store
    pipe.search_documents = AsyncMock(return_value=[
        _chunk("c1", "first chunk", similarity=0.91),
        _chunk("c2", "second chunk", similarity=0.72),
    ])
    return pipe


@pytest.fixture
def config():
    """A minimal AppConfig stand-in.

    Deliberately built from the real attribute set of AppConfig. Using a bare
    MagicMock here would auto-create any attribute the code asks for and would
    not have caught the `config.agent` bug.
    """
    return SimpleNamespace(
        connection_string="postgresql://u:p@localhost:5432/db",
        reranker=None,
        pipeline=None,
    )


@pytest.fixture
def llm_response():
    return SimpleRAGResponse(
        answer="The answer is 42.",
        confidence=0.9,
        word_count=4,
        sources_used=2,
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        metadata={"method": "gemini", "model": "gemini-2.5-flash"},
    )


@pytest.fixture(autouse=True)
def no_telemetry():
    """Never touch the interactions DB from unit tests."""
    with patch("app.retrieval.search.log_interaction", new=AsyncMock(return_value=None)):
        yield


class TestSuccessfulSearch:
    """The path that was broken: a query that actually retrieves chunks."""

    @pytest.mark.asyncio
    async def test_returns_rag_response_when_chunks_are_found(self, pipeline, config, llm_response):
        with patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is the answer",
                limit=5,
                threshold=0.3,
                pipeline=pipeline,
                config=config,
                enable_reranking=False,
            )

        assert isinstance(result, RAGResponse)
        assert result.answer == "The answer is 42."
        assert len(result.sources) == 2
        assert result.search_stats.chunks_found == 2

    @pytest.mark.asyncio
    async def test_does_not_touch_undefined_config_attributes(self, pipeline, config, llm_response):
        """Regression guard: the LLM call must only use attributes AppConfig defines.

        `config` is a SimpleNamespace, so any stray `config.<missing>` raises
        AttributeError rather than silently succeeding.
        """
        with patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is the answer",
                limit=5,
                threshold=0.3,
                pipeline=pipeline,
                config=config,
                enable_reranking=False,
            )

        assert result.answer

    @pytest.mark.asyncio
    async def test_llm_is_called_without_an_agent_argument(self, pipeline, config, llm_response):
        """generate_llm_response takes (query, context, results, model=...) only."""
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            await perform_document_search(
                query="what is the answer",
                limit=5,
                threshold=0.3,
                pipeline=pipeline,
                config=config,
                model="gemini-2.5-flash",
                enable_reranking=False,
            )

        args, kwargs = mock_llm.call_args
        assert len(args) == 3, f"expected (query, context, results), got {len(args)} positional args"
        assert kwargs == {"model": "gemini-2.5-flash"}

    @pytest.mark.asyncio
    async def test_context_includes_retrieved_chunk_text(self, pipeline, config, llm_response):
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            await perform_document_search(
                query="what is the answer",
                limit=5,
                threshold=0.3,
                pipeline=pipeline,
                config=config,
                enable_reranking=False,
            )

        context = mock_llm.call_args[0][1]
        assert "first chunk" in context
        assert "second chunk" in context


class TestNoResults:
    """The path that always worked — kept so the early return stays intact."""

    @pytest.mark.asyncio
    async def test_returns_placeholder_without_calling_the_llm(self, pipeline, config):
        pipeline.search_documents = AsyncMock(return_value=[])
        pipeline.vector_store.search_bm25 = AsyncMock(return_value=[])

        mock_llm = AsyncMock()
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            result = await perform_document_search(
                query="nothing matches this",
                limit=5,
                threshold=0.99,
                pipeline=pipeline,
                config=config,
            )

        assert result.search_stats.chunks_found == 0
        assert result.sources == []
        mock_llm.assert_not_called()


class TestReranking:
    """Reranking is optional and must degrade to RRF rather than fail the query."""

    @pytest.mark.asyncio
    async def test_reranker_failure_falls_back_to_rrf(self, pipeline, config, llm_response):
        with patch("app.retrieval.utils.get_reranker", side_effect=RuntimeError("model missing")), \
             patch("app.retrieval.search.generate_llm_response", new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is the answer",
                limit=5,
                threshold=0.3,
                pipeline=pipeline,
                config=config,
                enable_reranking=True,
                rerank_top_k=2,
            )

        assert result.search_stats.reranking_enabled is False
        assert result.answer == "The answer is 42."
