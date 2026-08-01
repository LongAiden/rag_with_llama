"""
Unit tests for models/models.py.

Covers:
- QueryRequest validation (field constraints)
- UploadResponse creation
- SupportedFileType enum
- FileValidationResult / FileValidationConfig
- RAGSource, RAGResponse, RAGResponseMetadata
- SimpleRAGResponse
"""
import pytest
from pydantic import ValidationError

from models.models import (
    QueryRequest,
    UploadResponse,
    SupportedFileType,
    FileValidationResult,
    FileValidationConfig,
    RAGSource,
    RAGResponseMetadata,
    RAGResponse,
    SimpleRAGResponse,
)


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(query="What is ML?")
        assert req.query == "What is ML?"
        assert req.limit == 5
        assert req.threshold == 0.3

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 1001)

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=21)

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", threshold=-0.1)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", threshold=1.1)

    def test_optional_fields(self):
        req = QueryRequest(
            query="test",
            document_ids=["doc1"],
            enable_reranking=False,
            rerank_top_k=5,
            model="gemini-2.5-flash",
            table_name="custom_table",
            session_id="sess-123",
        )
        assert req.document_ids == ["doc1"]
        assert req.enable_reranking is False
        assert req.rerank_top_k == 5
        assert req.session_id == "sess-123"


class TestUploadResponse:
    def test_minimal_creation(self):
        resp = UploadResponse(
            status="success",
            document_id="abc-123",
            filename="test.pdf",
            message="Processed",
        )
        assert resp.status == "success"
        assert resp.chunks_created is None
        assert resp.table_count is None
        assert resp.task_id is None

    def test_full_creation(self):
        resp = UploadResponse(
            status="success",
            document_id="abc-123",
            filename="test.pdf",
            message="Processed",
            chunks_created=10,
            table_count=3,
            task_id="task-456",
        )
        assert resp.chunks_created == 10
        assert resp.table_count == 3
        assert resp.task_id == "task-456"


class TestSupportedFileType:
    def test_enum_values(self):
        assert SupportedFileType.PDF == "pdf"
        assert SupportedFileType.DOCX == "docx"
        assert SupportedFileType.TXT == "txt"

    def test_is_string_enum(self):
        assert isinstance(SupportedFileType.PDF, str)


class TestFileValidationConfig:
    def test_default_values(self):
        config = FileValidationConfig()
        assert config.max_file_size_mb == 50
        assert ".pdf" in config.allowed_extensions

    def test_zero_size_rejected(self):
        with pytest.raises(ValidationError):
            FileValidationConfig(max_file_size_mb=0)

    def test_negative_size_rejected(self):
        with pytest.raises(ValidationError):
            FileValidationConfig(max_file_size_mb=-1)

    def test_custom_values(self):
        config = FileValidationConfig(max_file_size_mb=100, allowed_extensions=[".pdf"])
        assert config.max_file_size_mb == 100
        assert config.allowed_extensions == [".pdf"]


class TestFileValidationResult:
    def test_valid_result(self):
        result = FileValidationResult(
            filename="test.pdf",
            file_type=SupportedFileType.PDF,
            is_valid=True,
            file_size=1024,
        )
        assert result.error_message is None

    def test_invalid_result(self):
        result = FileValidationResult(
            filename="test.xyz",
            file_type=None,
            is_valid=False,
            file_size=0,
            error_message="Unsupported",
        )
        assert result.error_message == "Unsupported"


class TestRAGSource:
    def test_minimal_creation(self):
        source = RAGSource(
            chunk_id="c1",
            text="Some text",
            similarity=0.85,
            document_id="d1",
        )
        assert source.page_number is None
        assert source.metadata == {}
        assert source.rerank_score is None
        assert source.bm25_score is None
        assert source.rrf_score is None
        assert source.graph_entities == []

    def test_similarity_bounds(self):
        with pytest.raises(ValidationError):
            RAGSource(chunk_id="c1", text="t", similarity=-0.1, document_id="d1")
        with pytest.raises(ValidationError):
            RAGSource(chunk_id="c1", text="t", similarity=1.1, document_id="d1")

    def test_full_creation(self):
        source = RAGSource(
            chunk_id="c1",
            text="text",
            similarity=0.9,
            document_id="d1",
            page_number=5,
            metadata={"key": "val"},
            rerank_score=0.95,
            bm25_score=3.5,
            rrf_score=0.03,
            graph_entities=[{"name": "BERT"}],
        )
        assert source.page_number == 5
        assert source.rerank_score == 0.95


class TestRAGResponseMetadata:
    def test_creation(self):
        meta = RAGResponseMetadata(
            chunks_found=10,
            avg_similarity=0.85,
            search_method="pgvector_cosine",
            threshold_used=0.3,
        )
        assert meta.chunks_found == 10
        assert meta.word_count is None
        assert meta.confidence is None
        assert meta.graph_enriched is None


class TestRAGResponse:
    def test_creation(self):
        meta = RAGResponseMetadata(
            chunks_found=5,
            avg_similarity=0.8,
            search_method="pgvector_cosine",
            threshold_used=0.3,
        )
        resp = RAGResponse(
            query="What is ML?",
            answer="ML is ...",
            sources=[],
            search_stats=meta,
        )
        assert resp.query == "What is ML?"
        assert resp.table_used is None


class TestSimpleRAGResponse:
    def test_creation(self):
        resp = SimpleRAGResponse(
            answer="Answer text",
            word_count=10,
            sources_used=3,
        )
        assert resp.confidence is None
        assert resp.metadata == {}

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SimpleRAGResponse(answer="a", confidence=-0.1, word_count=1, sources_used=1)
        with pytest.raises(ValidationError):
            SimpleRAGResponse(answer="a", confidence=1.1, word_count=1, sources_used=1)
