"""
Unit tests for the domain registry and the `doc_name` label.

No database and no model weights: everything external is mocked. The test that
matters most here is TestRerankPreservesDocName — the rerank block in
retrieval/search.py rebuilds result dicts from RerankedResult, which carries no
doc_name, so any field not explicitly restored from `original_by_id` is dropped
silently. That is the same bug shape that already forced bm25_score and rrf_score
to be restored there.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infra.db.identifiers import validate_table_name
from app.ingestion.embedding.chunk import Chunk
from app.models.schemas import (
    DomainDocument,
    DomainInfo,
    QueryRequest,
    RAGSource,
    SimpleRAGResponse,
    UploadResponse,
)
from app.retrieval.reranking import RerankedResult
from app.retrieval.search import perform_document_search


class TestChunkDocName:
    def test_doc_name_is_stored(self):
        chunk = Chunk(id="c1", document_id="d1", text="t", embedding=[0.1],
                      doc_name="Linear Algebra")
        assert chunk.doc_name == "Linear Algebra"

    def test_doc_name_defaults_to_none(self):
        """Pre-006 call sites construct Chunks without it and must keep working."""
        chunk = Chunk(id="c1", document_id="d1", text="t", embedding=[0.1])
        assert chunk.doc_name is None


class TestSchemas:
    def test_rag_source_accepts_doc_name(self):
        source = RAGSource(chunk_id="c1", text="t", similarity=0.9,
                           document_id="d1", doc_name="Linear Algebra")
        assert source.doc_name == "Linear Algebra"

    def test_rag_source_without_doc_name_still_validates(self):
        """doc_name is additive — existing consumers must not break."""
        source = RAGSource(chunk_id="c1", text="t", similarity=0.9, document_id="d1")
        assert source.doc_name is None

    def test_query_request_accepts_domain_and_doc_name(self):
        req = QueryRequest(query="q", domain="math", doc_name="Linear Algebra")
        assert req.domain == "math"
        assert req.doc_name == "Linear Algebra"

    def test_query_request_without_either_uses_table_name(self):
        req = QueryRequest(query="q")
        assert req.domain is None
        assert req.doc_name is None
        assert req.table_name == "document_chunks"

    def test_upload_response_carries_doc_name_and_domain(self):
        resp = UploadResponse(status="queued", document_id="d1", filename="a.pdf",
                              message="ok", doc_name="Linear Algebra", domain="math")
        assert (resp.doc_name, resp.domain) == ("Linear Algebra", "math")

    def test_domain_info_defaults_document_count_to_zero(self):
        info = DomainInfo(name="math", display_name="Mathematics", table_name="math")
        assert info.document_count == 0

    def test_domain_document_allows_a_document_still_ingesting(self):
        doc = DomainDocument(document_id="d1", file_name="a.pdf", stage="parsing")
        assert doc.chunk_count is None
        assert doc.doc_name is None


class TestReservedTableNames:
    """Regression guard: _SYSTEM_TABLES was declared but never consulted."""

    @pytest.mark.parametrize("name", [
        "domains", "documents", "document_parsed", "document_chunked",
        "llm_interactions", "entities", "relationships",
    ])
    def test_application_tables_are_rejected(self, name):
        with pytest.raises(ValueError, match="Reserved table name"):
            validate_table_name(name)

    def test_case_insensitive(self):
        with pytest.raises(ValueError, match="Reserved table name"):
            validate_table_name("Domains")

    @pytest.mark.parametrize("name", ["math", "document_chunks", "history_v2"])
    def test_ordinary_domain_names_still_pass(self, name):
        assert validate_table_name(name) == name

    @pytest.mark.parametrize("name", ["My Domain!", "drop table", "1math", "a" * 64])
    def test_invalid_identifiers_are_rejected(self, name):
        with pytest.raises(ValueError, match="Invalid table name"):
            validate_table_name(name)


def _chunk(chunk_id: str, doc_name: str, similarity: float = 0.9):
    return {
        "chunk_id": chunk_id,
        "text": f"text of {chunk_id}",
        "document_id": "doc-1",
        "doc_name": doc_name,
        "metadata": {"page_number": 12, "page_content": "", "section_path": ""},
        "similarity": similarity,
    }


@pytest.fixture
def pipeline():
    vector_store = MagicMock()
    vector_store.search_bm25 = AsyncMock(return_value=[
        {**_chunk("c2", "Calculus"), "bm25_score": 3.2},
    ])
    vector_store.get_chunks_by_section = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.embedding_generator = SimpleNamespace(model_name="all-MiniLM-L6-v2", embedding_dim=384)
    pipe.vector_store = vector_store
    pipe.search_documents = AsyncMock(return_value=[
        _chunk("c1", "Linear Algebra", similarity=0.91),
        _chunk("c2", "Calculus", similarity=0.72),
    ])
    return pipe


@pytest.fixture
def config():
    return SimpleNamespace(
        connection_string="postgresql://u:p@localhost:5432/db",
        reranker=None,
        pipeline=None,
    )


@pytest.fixture
def llm_response():
    return SimpleRAGResponse(
        answer="A vector is an element of a vector space.",
        confidence=0.9,
        word_count=8,
        sources_used=2,
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        metadata={"method": "gemini", "model": "gemini-2.5-flash"},
    )


@pytest.fixture(autouse=True)
def no_telemetry():
    with patch("app.retrieval.search.log_interaction", new=AsyncMock(return_value=None)):
        yield


class TestSearchCarriesDocName:
    @pytest.mark.asyncio
    async def test_vector_mode_sources_have_doc_name(self, pipeline, config, llm_response):
        with patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert [s.doc_name for s in result.sources] == ["Linear Algebra", "Calculus"]

    @pytest.mark.asyncio
    async def test_hybrid_mode_sources_have_doc_name(self, pipeline, config, llm_response):
        """RRF copies the base result dict, so doc_name must survive the merge."""
        with patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
                search_mode="hybrid",
            )

        assert result.sources
        assert all(s.doc_name for s in result.sources)

    @pytest.mark.asyncio
    async def test_context_sent_to_the_llm_names_the_source(self, pipeline, config, llm_response):
        """The model can only attribute in prose if the name is in the prompt."""
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        context = mock_llm.call_args[0][1]
        assert "Source 1 — Linear Algebra" in context

    @pytest.mark.asyncio
    async def test_missing_doc_name_falls_back_to_an_unlabelled_source(
        self, pipeline, config, llm_response
    ):
        """Chunks written before the migration have doc_name NULL."""
        pipeline.search_documents = AsyncMock(return_value=[
            {**_chunk("c1", "Linear Algebra"), "doc_name": None},
        ])
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].doc_name is None
        assert "[Source 1 (Page 12)]" in mock_llm.call_args[0][1]


class TestRerankPreservesDocName:
    """The reranker returns RerankedResult, which has no doc_name field."""

    def test_reranked_result_does_not_carry_doc_name(self):
        assert not hasattr(
            RerankedResult(chunk_id="c1", text="t", document_id="d1", metadata={},
                           similarity=0.9, rerank_score=1.0, original_rank=0, new_rank=0),
            "doc_name",
        )

    @pytest.mark.asyncio
    async def test_doc_name_survives_reranking(self, pipeline, config, llm_response):
        def _rerank(query, results, top_k=None):
            # Mirrors the real reranker: only the fields RerankedResult declares
            # come back, in a new order.
            ordered = sorted(results, key=lambda r: r["similarity"])
            return [
                RerankedResult(
                    chunk_id=r["chunk_id"], text=r["text"], document_id=r["document_id"],
                    metadata=r["metadata"], similarity=r["similarity"],
                    rerank_score=9.0 - i, original_rank=i, new_rank=i,
                )
                for i, r in enumerate(ordered[:top_k or len(ordered)])
            ]

        fake_reranker = SimpleNamespace(rerank=_rerank)
        with patch("app.retrieval.utils.get_reranker", return_value=fake_reranker), \
             patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config,
                enable_reranking=True, rerank_top_k=2,
            )

        assert result.search_stats.reranking_enabled is True
        assert [s.doc_name for s in result.sources] == ["Calculus", "Linear Algebra"]
