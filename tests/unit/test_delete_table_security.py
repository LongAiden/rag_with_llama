"""
Unit tests for DELETE /table/{table_name} endpoint security.

Covers:
- Authentication enforcement (require_access_password)
- SQL injection prevention (validate_table_name)

Uses FastAPI TestClient with mocked pipeline and config — no real DB required.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient


def _make_app():
    """Build a minimal FastAPI app with only the delete route wired up."""
    from fastapi import FastAPI, Header
    from typing import Optional
    from app.api.routes.table_routes import delete_table

    app = FastAPI()

    mock_config = MagicMock()
    mock_config.pipeline = None
    mock_config.connection_string = "postgresql://test:test@localhost/test"

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"table_exists": True, "estimated_rows": 5})
    mock_conn.fetchval = AsyncMock(return_value=True)
    mock_conn.execute = AsyncMock(return_value="DELETE 0")

    mock_vector_store = MagicMock()
    mock_vector_store.table_name = "some_other_table"

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store = mock_vector_store

    @asynccontextmanager
    async def mock_connection():
        yield mock_conn

    mock_pipeline.vector_store.connection = mock_connection

    async def mock_get_pipeline(table_name):
        return mock_pipeline

    @app.delete("/table/{table_name}")
    async def delete_table_route(
        table_name: str,
        x_app_password: Optional[str] = Header(default=None),
    ):
        with patch("app.api.routes.table_deletion.IngestionRepository") as mock_ing_repo_cls, \
             patch("app.api.routes.table_deletion.DomainRepository") as mock_dom_repo_cls:
            mock_ing_repo = AsyncMock()
            mock_ing_repo.delete_documents_for_table = AsyncMock(return_value=[])
            mock_ing_repo_cls.return_value = mock_ing_repo

            mock_dom_repo = AsyncMock()
            mock_dom_repo.delete_domain = AsyncMock(return_value=False)
            mock_dom_repo_cls.return_value = mock_dom_repo

            return await delete_table(
                table_name=table_name,
                x_app_password=x_app_password,
                config=mock_config,
                get_pipeline=mock_get_pipeline,
                forget_pipeline=None,
            )

    return app


@pytest.fixture(scope="module")
def client():
    app = _make_app()
    with TestClient(app) as c:
        yield c


class TestDeleteAuthentication:
    """Auth checks — require_access_password enforcement."""

    def test_no_password_when_auth_enabled(self, client):
        with patch.dict("os.environ", {"APP_ACCESS_PASSWORD": "secret"}):
            resp = client.delete("/table/document_chunks")
        assert resp.status_code == 403

    def test_wrong_password_rejected(self, client):
        with patch.dict("os.environ", {"APP_ACCESS_PASSWORD": "secret"}):
            resp = client.delete(
                "/table/document_chunks",
                headers={"X-App-Password": "wrong"},
            )
        assert resp.status_code == 403

    def test_correct_password_accepted(self, client):
        with patch.dict("os.environ", {"APP_ACCESS_PASSWORD": "secret"}):
            resp = client.delete(
                "/table/document_chunks",
                headers={"X-App-Password": "secret"},
            )
        assert resp.status_code == 200

    def test_no_auth_configured_allows_request(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/document_chunks")
        assert resp.status_code == 200


class TestDeleteSQLInjection:
    """SQL injection payloads must be blocked before reaching the DB."""

    def test_semicolon_payload_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/foo%3B%20DROP%20TABLE%20x%3B--")
        assert resp.status_code == 400

    def test_inline_comment_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            # URL-encoded "foo--bar"
            resp = client.delete("/table/foo--bar")
        assert resp.status_code == 400

    def test_space_in_name_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/foo%20bar")
        assert resp.status_code == 400

    def test_dot_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/schema.table")
        assert resp.status_code == 400

    def test_path_traversal_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/..%2Fetc%2Fpasswd")
        assert resp.status_code == 400

    def test_starts_with_digit_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/1badname")
        assert resp.status_code == 400

    def test_too_long_blocked(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete(f"/table/{'a' * 64}")
        assert resp.status_code == 400


class TestDeleteValidNames:
    """Valid table names must pass validation and reach the handler."""

    def test_standard_name(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/document_chunks")
        assert resp.status_code == 200

    def test_underscore_prefix(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/_my_table")
        assert resp.status_code == 200

    def test_alphanumeric_with_numbers(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.delete("/table/chunks_v2")
        assert resp.status_code == 200
