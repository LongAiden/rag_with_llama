"""
Unit tests for the /stats aggregation in api/routes/admin_routes.py.

The route read `result['docs']` while `TableRepository.get_table_stats` aliases
that column `documents`. `asyncpg.Record.__getitem__` raises `KeyError` for an
unknown column, and `str(KeyError('docs'))` is `"'docs'"` — which is exactly the
text the page rendered as "Failed to Load Statistics — Error: 'docs'". The bug
only fired once a chunk table had data, so the empty-database path hid it.

These tests pin the route to the column names the SQL actually produces.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.routes import admin_routes


# The exact aliases produced by TableRepository.get_table_stats.
STATS_ROW_KEYS = ("documents", "chunks", "total_text_length", "earliest", "latest")


def _stats_row(documents=3, chunks=42, total_text_length=8400, earliest=None, latest=None):
    """A row that behaves like asyncpg.Record: unknown keys raise KeyError."""
    data = {
        "documents": documents,
        "chunks": chunks,
        "total_text_length": total_text_length,
        "earliest": earliest,
        "latest": latest,
    }
    row = MagicMock()
    row.__getitem__.side_effect = lambda key: data[key]
    return row


@asynccontextmanager
async def _fake_connection(_connection_string):
    yield MagicMock()


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.connection_string = "postgresql://test"
    return cfg


def _patched_repo(table_names, rows):
    repo = MagicMock()
    repo.list_chunk_tables = AsyncMock(return_value=table_names)
    repo.get_table_stats = AsyncMock(side_effect=rows)
    return repo


class TestStatsAggregation:
    async def test_populated_tables_render_the_dashboard(self, config):
        repo = _patched_repo(["document_chunks"], [_stats_row()])

        with patch.object(admin_routes, "_admin_connection", _fake_connection), \
             patch.object(admin_routes, "TableRepository", return_value=repo), \
             patch.object(admin_routes, "render", return_value="OK") as render:
            result = await admin_routes.get_database_stats(config=config)

        assert result == "OK"
        assert render.call_args.args[0] == "stats.html"
        assert render.call_args.kwargs["total_documents"] == "3"
        assert render.call_args.kwargs["total_chunks"] == "42"

    async def test_totals_sum_across_tables(self, config):
        repo = _patched_repo(
            ["table_a", "table_b"],
            [_stats_row(documents=3, chunks=42), _stats_row(documents=5, chunks=58)],
        )

        with patch.object(admin_routes, "_admin_connection", _fake_connection), \
             patch.object(admin_routes, "TableRepository", return_value=repo), \
             patch.object(admin_routes, "render", return_value="OK") as render:
            await admin_routes.get_database_stats(config=config)

        assert render.call_args.kwargs["total_documents"] == "8"
        assert render.call_args.kwargs["total_chunks"] == "100"

    async def test_never_reads_a_column_the_query_does_not_produce(self, config):
        """The regression guard: any key outside STATS_ROW_KEYS raises, and the
        route's blanket `except` would turn that into the stats error page."""
        repo = _patched_repo(["document_chunks"], [_stats_row()])

        with patch.object(admin_routes, "_admin_connection", _fake_connection), \
             patch.object(admin_routes, "TableRepository", return_value=repo), \
             patch.object(admin_routes, "render", return_value="OK") as render:
            await admin_routes.get_database_stats(config=config)

        assert render.call_args.args[0] != "stats_error.html"

    async def test_empty_database_still_renders(self, config):
        repo = _patched_repo([], [])

        with patch.object(admin_routes, "_admin_connection", _fake_connection), \
             patch.object(admin_routes, "TableRepository", return_value=repo), \
             patch.object(admin_routes, "render", return_value="OK") as render:
            await admin_routes.get_database_stats(config=config)

        assert render.call_args.args[0] == "stats.html"
        assert render.call_args.kwargs["total_documents"] == "0"


class TestQueryAliasesMatch:
    def test_route_reads_only_columns_the_sql_aliases(self):
        """Static check tying the two files together, so renaming one side is caught."""
        import inspect
        import re

        from app.infra.db.table_repository import TableRepository

        sql = inspect.getsource(TableRepository.get_table_stats)
        aliases = set(re.findall(r"\bas (\w+)", sql))

        route = inspect.getsource(admin_routes.get_database_stats)
        read_keys = set(re.findall(r"result\['(\w+)'\]", route))

        assert read_keys, "expected the route to read the stats row by column name"
        assert read_keys <= aliases, f"route reads columns the query never produces: {read_keys - aliases}"
