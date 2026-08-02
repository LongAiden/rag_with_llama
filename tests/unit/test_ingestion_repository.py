"""
Unit tests for the ingestion repository.

Tests the status DB operations for the stage-based ingestion pipeline.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


class AsyncContextManagerMock:
    """Mock for async context manager (async with pool.acquire() as conn)."""

    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TransactionContextManagerMock:
    """Mock for async transaction context manager (async with conn.transaction())."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    conn = MagicMock()
    conn.fetchrow = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.execute = AsyncMock()
    conn.transaction = MagicMock(return_value=TransactionContextManagerMock())
    pool.acquire.return_value = AsyncContextManagerMock(conn)
    return pool, conn


@pytest.fixture
def repo(mock_pool):
    """Create an IngestionRepository with a mock pool."""
    from infra.db import IngestionRepository
    pool, _ = mock_pool
    return IngestionRepository(connection_string="postgresql://test:test@localhost/test", pool=pool)


class TestRegisterDocument:
    """Tests for register_document method."""

    @pytest.mark.asyncio
    async def test_register_new_document(self, repo, mock_pool):
        """Test registering a new document returns the inserted row."""
        _, conn = mock_pool
        expected_row = {
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "registered",
            "attempts": 0,
        }
        conn.fetchrow = AsyncMock(return_value=expected_row)

        result = await repo.register_document(
            doc_id="test-doc-id",
            file_name="test.pdf",
            raw_storage_path="/tmp/test.pdf",
            file_size=1024,
            content_type="application/pdf",
        )

        assert result["id"] == "test-doc-id"
        assert result["file_name"] == "test.pdf"
        assert result["stage"] == "registered"
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_same_filename_creates_a_new_document(self, repo, mock_pool):
        """Re-uploading a filename registers a second, independent document.

        Filename de-duplication was removed (migration 005): the INSERT has no
        ON CONFLICT clause, so there is exactly one round trip and the caller always
        gets back the row it just created, never a pre-existing one.
        """
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "new-doc-id",
            "file_name": "test.pdf",
            "stage": "registered",
        })

        result = await repo.register_document(
            doc_id="new-doc-id",
            file_name="test.pdf",
            raw_storage_path="/tmp/test.pdf",
            file_size=1024,
        )

        assert result["id"] == "new-doc-id"
        assert result["stage"] == "registered"
        assert conn.fetchrow.call_count == 1
        assert "ON CONFLICT" not in conn.fetchrow.call_args[0][0]


class TestClaimDocument:
    """Tests for claim_document method."""

    @pytest.mark.asyncio
    async def test_claim_registered_document(self, repo, mock_pool):
        """Test claiming a registered document transitions to processing stage."""
        _, conn = mock_pool
        claimed_row = {
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "parsing",
            "claimed_by": "worker-1",
        }
        conn.fetchrow = AsyncMock(return_value=claimed_row)

        result = await repo.claim_document(
            doc_id="test-doc-id",
            current_stage="registered",
            processing_stage="parsing",
            worker_id="worker-1",
            timeout_minutes=30,
        )

        assert result is not None
        assert result["id"] == "test-doc-id"
        assert result["stage"] == "parsing"

    @pytest.mark.asyncio
    async def test_claim_returns_none_when_not_claimable(self, repo, mock_pool):
        """Test claiming returns None when the document is not in the expected stage."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        result = await repo.claim_document(
            doc_id="test-doc-id",
            current_stage="registered",
            processing_stage="parsing",
            worker_id="worker-1",
            timeout_minutes=30,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_claim_is_scoped_to_the_requested_document(self, repo, mock_pool):
        """Claiming must target one id and one stage, never 'whichever row is next'.

        Regression guard: an unscoped claim moved an unrelated document into a
        processing stage and left the claim behind, stranding it until the sweep.
        """
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        await repo.claim_document(
            doc_id="test-doc-id",
            current_stage="registered",
            processing_stage="parsing",
            worker_id="worker-1",
            timeout_minutes=30,
        )

        sql, *params = conn.fetchrow.call_args[0]
        assert "WHERE id = $1" in sql
        assert "AND stage = $2" in sql
        assert "LIMIT" not in sql.upper()
        assert params[0] == "test-doc-id"
        assert params[1] == "registered"


class TestTransitionMethods:
    """Tests for stage transition methods."""

    @pytest.mark.asyncio
    async def test_transition_to_parsed(self, repo, mock_pool):
        """Test transitioning to parsed stage stores parsed text."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "stage": "parsed",
        })

        await repo.transition_to_parsed(
            doc_id="test-doc-id",
            parsed_text="This is parsed text.",
            parser_used="gemini-docling",
            file_type="pdf",
            metadata={"file_type": "pdf"},
        )

        assert conn.fetchrow.call_count == 2
        first_call_sql = conn.fetchrow.call_args_list[0][0][0]
        assert "document_parsed" in first_call_sql
        # Retries must overwrite the artifact, not append a second row.
        assert "ON CONFLICT (document_id) DO UPDATE" in first_call_sql
        second_call_sql = conn.fetchrow.call_args_list[1][0][0]
        assert "stage = 'parsed'" in second_call_sql

    @pytest.mark.asyncio
    async def test_transition_to_parsed_persists_file_type(self, repo, mock_pool):
        """file_type must land on the documents row.

        Regression guard: without a real column the chunking stage always saw an
        empty file_type and silently downgraded every PDF to the generic chunker,
        losing page numbers and section paths.
        """
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"id": "test-doc-id", "stage": "parsed"})

        await repo.transition_to_parsed(
            doc_id="test-doc-id",
            parsed_text="text",
            parser_used="gemini-docling",
            file_type="pdf",
        )

        sql, *params = conn.fetchrow.call_args_list[1][0]
        assert "file_type" in sql
        assert "pdf" in params

    @pytest.mark.asyncio
    async def test_transition_to_chunked(self, repo, mock_pool):
        """Test transitioning to chunked stage stores chunk data."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "stage": "chunked",
        })

        chunks = [
            {"text": "chunk 1", "page_number": 1},
            {"text": "chunk 2", "page_number": 1},
        ]

        await repo.transition_to_chunked(
            doc_id="test-doc-id",
            chunks=chunks,
            chunk_size=512,
            metadata={"parser_used": "gemini-docling"},
        )

        assert conn.fetchrow.call_count == 2
        first_call_sql = conn.fetchrow.call_args_list[0][0][0]
        assert "document_chunked" in first_call_sql
        assert "ON CONFLICT (document_id) DO UPDATE" in first_call_sql
        second_call_sql = conn.fetchrow.call_args_list[1][0][0]
        assert "stage = 'chunked'" in second_call_sql

        # chunks are passed as a list, relying on the pool's jsonb codec.
        chunks_param = conn.fetchrow.call_args_list[0][0][2]
        assert chunks_param == chunks

    @pytest.mark.asyncio
    async def test_transition_to_embedded(self, repo, mock_pool):
        """Test transitioning to embedded stage updates chunk count."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "stage": "embedded",
        })

        await repo.transition_to_embedded(doc_id="test-doc-id")

        conn.fetchrow.assert_called_once()
        call_args = conn.fetchrow.call_args
        assert "stage = 'embedded'" in call_args[0][0]


class TestErrorHandling:
    """Tests for error recording and retry logic."""

    @pytest.mark.asyncio
    async def test_record_error_increments_attempts(self, repo, mock_pool):
        """Test recording an error increments the attempts counter."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "stage": "error",
            "attempts": 1,
        })

        await repo.record_error(
            doc_id="test-doc-id",
            error="Parse failed: invalid PDF",
            max_attempts=2,
        )

        conn.fetchrow.assert_called_once()
        call_args = conn.fetchrow.call_args
        assert "attempts = attempts + 1" in call_args[0][0]
        assert "last_error" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_record_error_marks_failed_when_max_attempts(self, repo, mock_pool):
        """Test recording error marks document as failed when max attempts reached."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "stage": "failed",
            "attempts": 2,
        })

        await repo.record_error(
            doc_id="test-doc-id",
            error="Parse failed",
            max_attempts=2,
        )

        call_args = conn.fetchrow.call_args
        assert "WHEN attempts + 1 >= $3 THEN 'failed'" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_record_error_stores_failing_stage(self, repo, mock_pool):
        """The failing stage is recorded so a retry can resume from it."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"id": "test-doc-id", "stage": "error"})

        await repo.record_error(
            doc_id="test-doc-id",
            error="Embedding failed",
            max_attempts=2,
            stage="embed",
        )

        sql, *params = conn.fetchrow.call_args[0]
        assert "error_stage" in sql
        assert "embed" in params


class TestStaleClaims:
    """Tests for stale claim reset."""

    @pytest.mark.asyncio
    async def test_reset_stale_claims(self, repo, mock_pool):
        """Test resetting stale claims returns count of reset documents."""
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="UPDATE 3")

        result = await repo.reset_stale_claims(timeout_minutes=30)

        assert result == 3
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_error_documents(self, repo, mock_pool):
        """Test resetting error documents for retry."""
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="UPDATE 2")

        result = await repo.reset_error_documents(max_attempts=2)

        assert result == 2
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_error_documents_resumes_from_last_artifact(self, repo, mock_pool):
        """A retry must not redo completed stages.

        Re-parsing through a VLM backend is the most expensive stage; an embedding
        failure should resume at 'chunked', not restart at 'registered'.
        """
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="UPDATE 1")

        await repo.reset_error_documents(max_attempts=2)

        sql = conn.execute.call_args[0][0]
        assert "error_stage = 'embed'" in sql and "THEN 'chunked'" in sql
        assert "error_stage = 'chunk'" in sql and "THEN 'parsed'" in sql


class TestQueryMethods:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_get_document_status(self, repo, mock_pool):
        """Test getting document status returns the row."""
        _, conn = mock_pool
        expected = {
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "parsed",
            "attempts": 0,
        }
        conn.fetchrow = AsyncMock(return_value=expected)

        result = await repo.get_document_status("test-doc-id")

        assert result["id"] == "test-doc-id"
        assert result["stage"] == "parsed"

    @pytest.mark.asyncio
    async def test_get_document_status_returns_none_for_missing(self, repo, mock_pool):
        """Test getting status for non-existent document returns None."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        result = await repo.get_document_status("non-existent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_is_path_registered(self, repo, mock_pool):
        """The scan de-duplicates on the stored raw path.

        Regression guard: uploads are written as '<uuid>_<name>' but registered
        under '<name>', so a filename-only check re-registered every uploaded file
        as a second document and duplicated its chunks in the vector store.
        """
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"?column?": 1})

        result = await repo.is_path_registered("/app/input/raw/uuid_test.pdf")

        assert result is True
        sql, *params = conn.fetchrow.call_args[0]
        assert "raw_storage_path = $1" in sql
        assert params[0] == "/app/input/raw/uuid_test.pdf"

    @pytest.mark.asyncio
    async def test_delete_document_returns_deleted_row(self, repo, mock_pool):
        """Deleting returns the row so callers can clean up file and chunks."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "raw_storage_path": "/app/input/raw/uuid_test.pdf",
        })

        result = await repo.delete_document("test-doc-id")

        assert result["id"] == "test-doc-id"
        assert "DELETE FROM documents" in conn.fetchrow.call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_document_returns_none_when_missing(self, repo, mock_pool):
        """Deleting a non-existent document returns None."""
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        assert await repo.delete_document("nope") is None

    @pytest.mark.asyncio
    async def test_get_pending_doc_ids(self, repo, mock_pool):
        """Test getting pending document IDs by stage."""
        _, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[
            {"id": "doc-1"},
            {"id": "doc-2"},
        ])

        result = await repo.get_pending_doc_ids(["registered", "parsed"])

        assert result == ["doc-1", "doc-2"]

    @pytest.mark.asyncio
    async def test_get_parsed_artifact(self, repo, mock_pool):
        """Test getting parsed artifact for a document."""
        _, conn = mock_pool
        expected = {
            "parsed_text": "This is parsed text.",
            "parser_used": "gemini-docling",
        }
        conn.fetchrow = AsyncMock(return_value=expected)

        result = await repo.get_parsed("test-doc-id")

        assert result["parsed_text"] == "This is parsed text."

    @pytest.mark.asyncio
    async def test_get_chunked_artifact(self, repo, mock_pool):
        """Test getting chunked artifact for a document."""
        _, conn = mock_pool
        expected = {
            "chunks": [{"text": "chunk 1"}, {"text": "chunk 2"}],
            "chunk_count": 2,
        }
        conn.fetchrow = AsyncMock(return_value=expected)

        result = await repo.get_chunked("test-doc-id")

        assert result["chunk_count"] == 2
        assert len(result["chunks"]) == 2
