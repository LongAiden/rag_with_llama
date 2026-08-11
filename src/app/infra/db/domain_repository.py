"""
Domain registry repository.

A domain is a named bucket of documents backed 1:1 by one pgvector chunk table.
This repository owns the `domains` registry and the `documents.domain` membership
column only — it knows nothing about how chunks are stored or searched.

See docs/plans/20260812_domains_and_doc_name.md.
"""

import os
from typing import Any, Dict, List, Optional

import asyncpg

from app.infra.db.identifiers import validate_table_name
from app.infra.db.pool import ConnectionPoolManager
from app.infra.db.table_repository import TableRepository


class DomainRepository:
    """CRUD over the `domains` registry."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool: Optional[asyncpg.Pool] = None,
    ):
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError("DATABASE_URL or connection_string is required")
        self._pool = pool

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await ConnectionPoolManager.get_pool(self.connection_string)
        return self._pool

    async def list_domains(self, reconcile: bool = True) -> List[Dict[str, Any]]:
        """List domains with their document counts.

        When `reconcile` is set, chunk tables that exist without a registry row are
        registered first. Tables can appear out of band — created directly by a
        pipeline before migration 006, or by hand — and a domain list that omitted
        them would hide data the /tables endpoint still shows.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if reconcile:
                await self._reconcile_chunk_tables(conn)
            rows = await conn.fetch(
                """
                SELECT d.name, d.display_name, d.description, d.table_name,
                       d.created_at, d.updated_at,
                       COUNT(doc.id) AS document_count
                FROM domains d
                LEFT JOIN documents doc ON doc.domain = d.name
                GROUP BY d.name, d.display_name, d.description, d.table_name,
                         d.created_at, d.updated_at
                ORDER BY d.display_name
                """
            )
            return [dict(r) for r in rows]

    async def _reconcile_chunk_tables(self, conn: asyncpg.Connection) -> None:
        """Register a domain row for any chunk table that lacks one."""
        table_names = await TableRepository(conn).list_chunk_tables()
        for table_name in table_names:
            try:
                validate_table_name(table_name)
            except ValueError:
                # A table whose name the app would refuse to accept cannot be a
                # domain; skip it rather than failing the whole listing.
                continue
            await conn.execute(
                """
                INSERT INTO domains (name, display_name, table_name)
                VALUES ($1, $2, $1)
                ON CONFLICT (name) DO NOTHING
                """,
                table_name,
                _default_display_name(table_name),
            )

    async def get_domain(self, name: str) -> Optional[Dict[str, Any]]:
        """Return one domain row with its document count, or None."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.name, d.display_name, d.description, d.table_name,
                       d.created_at, d.updated_at,
                       COUNT(doc.id) AS document_count
                FROM domains d
                LEFT JOIN documents doc ON doc.domain = d.name
                WHERE d.name = $1
                GROUP BY d.name, d.display_name, d.description, d.table_name,
                         d.created_at, d.updated_at
                """,
                name,
            )
            return dict(row) if row else None

    async def create_domain(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a domain, or return the existing one under the same name.

        `name` is validated as a SQL identifier because it is also the chunk table
        name. The chunk table itself is not created here — VectorStore does that
        lazily on first write, as it always has.
        """
        validate_table_name(name)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO domains (name, display_name, description, table_name)
                VALUES ($1, $2, $3, $1)
                ON CONFLICT (name) DO NOTHING
                """,
                name,
                display_name or _default_display_name(name),
                description,
            )
        domain = await self.get_domain(name)
        if domain is None:
            raise RuntimeError(f"Domain {name!r} vanished immediately after insert")
        return domain

    async def ensure_domain(self, name: str) -> Dict[str, Any]:
        """Return the domain, creating it if it does not exist yet.

        Used by /upload so uploading into a new domain works without a separate
        create call — matching the implicit table creation that already happens.
        """
        return await self.get_domain(name) or await self.create_domain(name)

    async def list_documents(self, name: str) -> List[Dict[str, Any]]:
        """List the documents in a domain, whatever their ingestion stage.

        Reads `documents`, not the chunk table, so a document still mid-ingest is
        visible with its stage instead of silently missing.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, doc_name, file_name, stage, chunk_count, created_at
                FROM documents
                WHERE domain = $1
                ORDER BY COALESCE(doc_name, file_name)
                """,
                name,
            )
            return [dict(r) for r in rows]

    async def find_document_ids_by_name(
        self,
        doc_name: str,
        scope: Optional[str] = None,
    ) -> List[str]:
        """Resolve a document name to ids, optionally scoped to one domain/table.

        Returns a list, not one id: uploading the same book twice produces two
        documents with the same name, so a name is not a key. Callers filter on
        the returned ids, which are.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id
                FROM documents
                WHERE doc_name = $1
                  AND ($2::text IS NULL OR domain = $2 OR target_table_name = $2)
                """,
                doc_name,
                scope,
            )
            return [str(r["id"]) for r in rows]

    async def delete_domain(self, name: str) -> bool:
        """Delete the registry row. Returns False if it did not exist.

        Dropping the chunk table and the documents rows is the caller's job — see
        the shared drop helper used by DELETE /table and DELETE /domains.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM domains WHERE name = $1", name)
        return result.split()[-1] != "0"


def _default_display_name(name: str) -> str:
    """`linear_algebra` -> `Linear Algebra`."""
    return name.replace("_", " ").title()
