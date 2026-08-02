from typing import List, Dict, Any

import asyncpg

from config.app_config import DEFAULT_TABLE_NAME
from infra.db.identifiers import quote_ident, validate_table_name

_SYSTEM_TABLES = frozenset([
    'entities', 'relationships', 'entity_nodes', 'entity_edges',
])

CHUNK_TABLES_QUERY = """
    SELECT DISTINCT t1.table_name
    FROM information_schema.columns t1
    WHERE t1.table_schema = 'public'
      AND t1.column_name = 'document_id'
      AND EXISTS (
          SELECT 1 FROM information_schema.columns t2
          WHERE t2.table_name = t1.table_name
            AND t2.table_schema = 'public'
            AND t2.column_name = 'embedding'
      )
      AND t1.table_name NOT IN ('entities', 'relationships', 'entity_nodes', 'entity_edges')
    ORDER BY t1.table_name
"""


class TableRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def list_chunk_tables(self) -> List[str]:
        rows = await self.conn.fetch(CHUNK_TABLES_QUERY)
        tables = [row['table_name'] for row in rows]

        # The default chunk table is auto-created on first pipeline use. If it has
        # never held any data, don't include it in listings so the table list stays
        # empty until a user actually creates/uploads to a table.
        if DEFAULT_TABLE_NAME in tables and await self._table_is_empty(DEFAULT_TABLE_NAME):
            tables.remove(DEFAULT_TABLE_NAME)

        return tables

    async def _table_is_empty(self, table_name: str) -> bool:
        safe_name = quote_ident(table_name)
        row = await self.conn.fetchrow(
            f"SELECT EXISTS (SELECT 1 FROM {safe_name} LIMIT 1) AS has_rows"
        )
        return not row['has_rows']

    async def table_exists(self, table_name: str) -> bool:
        validate_table_name(table_name)
        return await self.conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            table_name,
        )

    async def get_table_row_estimate(self, table_name: str) -> int:
        validate_table_name(table_name)
        return await self.conn.fetchval(
            """
            SELECT COALESCE(
                (SELECT reltuples::bigint FROM pg_catalog.pg_class WHERE relname = $1),
                0
            )
            """,
            table_name,
        )

    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        safe_name = quote_ident(table_name)
        return await self.conn.fetchrow(f"""
            SELECT
                COUNT(DISTINCT document_id) as documents,
                COUNT(*) as chunks,
                COALESCE(SUM(LENGTH(text)), 0) as total_text_length,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM {safe_name}
        """)

    async def get_table_row_counts(self, table_name: str) -> Dict[str, int]:
        safe_name = quote_ident(table_name)
        row = await self.conn.fetchrow(f"""
            SELECT
                COUNT(DISTINCT document_id) as documents,
                COUNT(*) as chunks
            FROM {safe_name}
        """)
        return dict(row)

    async def truncate_table(self, table_name: str) -> None:
        safe_name = quote_ident(table_name)
        await self.conn.execute(f"TRUNCATE TABLE {safe_name} CASCADE")

    async def drop_table(self, table_name: str) -> None:
        safe_name = quote_ident(table_name)
        await self.conn.execute(f"DROP TABLE {safe_name} CASCADE")

    async def delete_chunks_by_document_id(self, table_name: str, document_id: str) -> int:
        safe_name = quote_ident(table_name)
        result = await self.conn.execute(
            f"DELETE FROM {safe_name} WHERE document_id = $1",
            document_id,
        )
        return int(result.split()[-1]) if result else 0
