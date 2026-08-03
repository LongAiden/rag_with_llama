"""
Unit tests for infra/db/identifiers.py (pure functions only).

Covers:
- validate_table_name (valid/invalid names)
- quote_ident
"""
from unittest.mock import AsyncMock

import pytest

from app.infra.db.identifiers import validate_table_name, quote_ident
from app.infra.db.table_repository import TableRepository


class TestValidateTableName:
    @pytest.mark.parametrize("name", [
        "document_chunks",
        "test_table",
        "_private",
        "Table123",
        "a",
        "a" + "_" * 62,
    ])
    def test_valid_names(self, name):
        assert validate_table_name(name) == name

    @pytest.mark.parametrize("name", [
        "",
        "1starts_with_digit",
        "has space",
        "has-hyphen",
        "has.dot",
        "has;semicolon",
        "a" * 64,
        "drop table",
    ])
    def test_invalid_names(self, name):
        with pytest.raises(ValueError, match="Invalid table name"):
            validate_table_name(name)


class TestQuoteIdent:
    def test_quotes_valid_name(self):
        assert quote_ident("document_chunks") == '"document_chunks"'

    def test_quotes_simple_name(self):
        assert quote_ident("test") == '"test"'

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError):
            quote_ident("invalid name")


class TestListChunkTables:
    """Tests for TableRepository.list_chunk_tables."""

    @pytest.mark.asyncio
    async def test_hides_default_table_when_empty(self):
        """The auto-created default table is omitted from listings if it has no rows."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"table_name": "document_chunks"}])
        conn.fetchrow = AsyncMock(return_value={"has_rows": False})

        repo = TableRepository(conn)
        result = await repo.list_chunk_tables()

        assert result == []
        sql, *_ = conn.fetchrow.call_args[0]
        assert 'FROM "document_chunks"' in sql

    @pytest.mark.asyncio
    async def test_keeps_default_table_when_it_has_data(self):
        """The default table is listed once it contains data."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"table_name": "document_chunks"}])
        conn.fetchrow = AsyncMock(return_value={"has_rows": True})

        repo = TableRepository(conn)
        result = await repo.list_chunk_tables()

        assert result == ["document_chunks"]

    @pytest.mark.asyncio
    async def test_keeps_other_tables_even_when_empty(self):
        """User-created tables are listed regardless of row count."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"table_name": "test"}])

        repo = TableRepository(conn)
        result = await repo.list_chunk_tables()

        assert result == ["test"]
        conn.fetchrow.assert_not_called()
