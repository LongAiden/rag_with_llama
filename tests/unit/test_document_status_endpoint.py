"""
Unit tests for the document status and delete endpoints.

Tests /documents/{document_id}/status and DELETE /documents/{document_id}.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = MagicMock()
    config.connection_string = "postgresql://admin:admin@localhost/rag_db"
    return config


@pytest.fixture
def mock_repo():
    """Create a mock IngestionRepository."""
    repo = AsyncMock()
    return repo


class TestGetDocumentStatus:
    """Tests for GET /documents/{document_id}/status endpoint."""

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_get_status_success(self, MockRepo, mock_config, mock_repo):
        """Test getting status for a valid document."""
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "parsed",
            "attempts": 0,
            "chunk_count": None,
            "last_error": None,
            "created_at": "2026-08-02T10:00:00Z",
            "updated_at": "2026-08-02T10:05:00Z",
        })

        from app.api.routes.document_routes import get_document_status
        result = await get_document_status("test-doc-id", config=mock_config)

        assert result["document_id"] == "test-doc-id"
        assert result["file_name"] == "test.pdf"
        assert result["stage"] == "parsed"
        assert result["attempts"] == 0

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_get_status_not_found(self, MockRepo, mock_config, mock_repo):
        """Test getting status for non-existent document returns 404."""
        from fastapi import HTTPException

        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value=None)

        from app.api.routes.document_routes import get_document_status
        with pytest.raises(HTTPException) as exc_info:
            await get_document_status("non-existent-id", config=mock_config)

        assert exc_info.value.status_code == 404
        assert "Document not found" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_get_status_with_error(self, MockRepo, mock_config, mock_repo):
        """Test getting status for a document that had errors."""
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "error",
            "attempts": 1,
            "chunk_count": None,
            "last_error": "Parse failed: invalid PDF structure",
            "created_at": "2026-08-02T10:00:00Z",
            "updated_at": "2026-08-02T10:05:00Z",
        })

        from app.api.routes.document_routes import get_document_status
        result = await get_document_status("test-doc-id", config=mock_config)

        assert result["stage"] == "error"
        assert result["attempts"] == 1
        assert "Parse failed" in result["last_error"]

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_get_status_embedded(self, MockRepo, mock_config, mock_repo):
        """Test getting status for a fully processed document."""
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "embedded",
            "attempts": 0,
            "chunk_count": 15,
            "last_error": None,
            "created_at": "2026-08-02T10:00:00Z",
            "updated_at": "2026-08-02T10:10:00Z",
        })

        from app.api.routes.document_routes import get_document_status
        result = await get_document_status("test-doc-id", config=mock_config)

        assert result["stage"] == "embedded"
        assert result["chunk_count"] == 15

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_get_status_repository_error(self, MockRepo, mock_config, mock_repo):
        """Test repository errors are handled gracefully."""
        from fastapi import HTTPException

        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(side_effect=Exception("DB connection failed"))

        from app.api.routes.document_routes import get_document_status
        with pytest.raises(HTTPException) as exc_info:
            await get_document_status("test-doc-id", config=mock_config)

        assert exc_info.value.status_code == 500
        assert "Failed to get document status" in exc_info.value.detail


class TestDeleteDocument:
    """Tests for DELETE /documents/{document_id}."""

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Auth is asserted separately; these tests exercise the delete behaviour."""
        with patch("app.api.routes.document_routes.require_access_password"):
            yield

    @pytest.mark.asyncio
    async def test_delete_requires_access_password(self, mock_config):
        """The endpoint is password-guarded when APP_ACCESS_PASSWORD is configured."""
        from fastapi import HTTPException
        import app.api.routes.document_routes as routes

        with patch(
            "app.api.routes.document_routes.require_access_password",
            side_effect=HTTPException(status_code=403, detail="Invalid access password"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await routes.delete_document("test-doc-id", config=mock_config, get_pipeline=None)

        assert exc_info.value.status_code == 403

    @pytest.fixture
    def status_row(self, tmp_path):
        raw_file = tmp_path / "uuid_test.pdf"
        raw_file.write_text("raw bytes")
        return {
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "stage": "embedded",
            "target_table_name": "document_chunks",
            "raw_storage_path": str(raw_file),
        }, raw_file

    @pytest.fixture
    def mock_get_pipeline(self):
        pipeline = AsyncMock()
        pipeline.delete_document = AsyncMock(return_value=12)

        async def _get_pipeline(table_name):
            return pipeline

        return _get_pipeline, pipeline

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_delete_removes_row_chunks_and_raw_file(
        self, MockRepo, mock_config, mock_repo, status_row, mock_get_pipeline, tmp_path
    ):
        """Deleting clears the status row, the vector chunks and the raw file."""
        import app.api.routes.document_routes as routes

        row, raw_file = status_row
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value=row)
        mock_repo.delete_document = AsyncMock(return_value=row)
        get_pipeline, pipeline = mock_get_pipeline

        with patch.object(routes, "INPUT_RAW_DIR", tmp_path):
            result = await routes.delete_document(
                "test-doc-id", config=mock_config, get_pipeline=get_pipeline
            )

        assert result["status"] == "deleted"
        assert result["chunks_deleted"] == 12
        assert result["raw_file_deleted"] is True
        assert not raw_file.exists()
        pipeline.delete_document.assert_called_once_with("test-doc-id")
        mock_repo.delete_document.assert_called_once_with("test-doc-id")

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_delete_not_found(self, MockRepo, mock_config, mock_repo):
        """Deleting a non-existent document returns 404."""
        from fastapi import HTTPException
        import app.api.routes.document_routes as routes

        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await routes.delete_document("nope", config=mock_config, get_pipeline=None)

        assert exc_info.value.status_code == 404
        mock_repo.delete_document.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_delete_refuses_raw_file_outside_input_dir(
        self, MockRepo, mock_config, mock_repo, mock_get_pipeline, tmp_path
    ):
        """A raw path outside INPUT_RAW_DIR is left alone, not unlinked."""
        import app.api.routes.document_routes as routes

        outsider = tmp_path / "elsewhere.pdf"
        outsider.write_text("do not delete me")
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "elsewhere.pdf",
            "target_table_name": "document_chunks",
            "raw_storage_path": str(outsider),
        })
        mock_repo.delete_document = AsyncMock(return_value={})
        get_pipeline, _ = mock_get_pipeline

        input_dir = tmp_path / "input_raw"
        input_dir.mkdir()
        with patch.object(routes, "INPUT_RAW_DIR", input_dir):
            result = await routes.delete_document(
                "test-doc-id", config=mock_config, get_pipeline=get_pipeline
            )

        assert result["raw_file_deleted"] is False
        assert outsider.exists()

    @pytest.mark.asyncio
    @patch("app.api.routes.document_routes.IngestionRepository")
    async def test_delete_can_keep_chunks_and_file(
        self, MockRepo, mock_config, mock_repo, status_row, mock_get_pipeline, tmp_path
    ):
        """Flags allow clearing only the status row."""
        import app.api.routes.document_routes as routes

        row, raw_file = status_row
        MockRepo.return_value = mock_repo
        mock_repo.get_document_status = AsyncMock(return_value=row)
        mock_repo.delete_document = AsyncMock(return_value=row)
        get_pipeline, pipeline = mock_get_pipeline

        with patch.object(routes, "INPUT_RAW_DIR", tmp_path):
            result = await routes.delete_document(
                "test-doc-id",
                delete_chunks=False,
                delete_raw_file=False,
                config=mock_config,
                get_pipeline=get_pipeline,
            )

        assert result["chunks_deleted"] == 0
        assert result["raw_file_deleted"] is False
        assert raw_file.exists()
        pipeline.delete_document.assert_not_called()
