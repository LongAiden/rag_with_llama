"""
Ingestion status repository.

Provides atomic operations on the `documents` status table and the intermediate
`document_parsed` / `document_chunked` artifact tables.
"""

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

import asyncpg

from repositories.connection_pool import ConnectionPoolManager


class IngestionRepository:
    """Repository for tracking ingestion pipeline state and intermediate artifacts."""

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

    async def register_document(
        self,
        doc_id: str,
        file_name: str,
        raw_storage_path: str,
        file_size: int,
        content_type: Optional[str] = None,
        target_table_name: str = "document_chunks",
        chunk_size: int = 512,
        parse_backend: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new document or return the existing row by filename (POC dedupe)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    id, file_name, raw_storage_path, file_size, content_type,
                    target_table_name, chunk_size, parse_backend, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (file_name) DO NOTHING
                RETURNING *
                """,
                doc_id,
                file_name,
                raw_storage_path,
                file_size,
                content_type,
                target_table_name,
                chunk_size,
                parse_backend,
                metadata or {},
            )
            if row is None:
                row = await conn.fetchrow(
                    "SELECT * FROM documents WHERE file_name = $1",
                    file_name,
                )
            return dict(row)

    async def claim_next_document(
        self,
        current_stage: str,
        processing_stage: str,
        worker_id: str,
        timeout_minutes: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the next document in current_stage and move it to
        processing_stage. Returns None if nothing is available.
        """
        pool = await self._get_pool()
        timeout = timedelta(minutes=timeout_minutes)
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM documents
                    WHERE stage = $1
                      AND (claimed_at IS NULL OR claimed_at < NOW() - $2::interval)
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                    current_stage,
                    timeout,
                )
                if row is None:
                    return None
                await conn.execute(
                    """
                    UPDATE documents
                    SET stage = $1,
                        claimed_at = NOW(),
                        claimed_by = $2,
                        updated_at = NOW()
                    WHERE id = $3
                    """,
                    processing_stage,
                    worker_id,
                    row["id"],
                )
                result = dict(row)
                result["stage"] = processing_stage
                result["claimed_by"] = worker_id
                return result

    async def transition_to_parsed(
        self,
        doc_id: str,
        parsed_text: str,
        parser_used: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store parsed text and mark document as parsed."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                parsed_row = await conn.fetchrow(
                    """
                    INSERT INTO document_parsed (document_id, parsed_text, parser_used, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    doc_id,
                    parsed_text,
                    parser_used,
                    metadata or {},
                )
                doc_row = await conn.fetchrow(
                    """
                    UPDATE documents
                    SET stage = 'parsed',
                        parsed_id = $1,
                        claimed_at = NULL,
                        claimed_by = NULL,
                        updated_at = NOW()
                    WHERE id = $2
                    RETURNING *
                    """,
                    parsed_row["id"],
                    doc_id,
                )
                return dict(doc_row)

    async def transition_to_chunked(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        chunk_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store chunk artifacts and mark document as chunked."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                chunked_row = await conn.fetchrow(
                    """
                    INSERT INTO document_chunked (document_id, chunks, chunk_size, chunk_count, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    doc_id,
                    chunks,
                    chunk_size,
                    len(chunks),
                    metadata or {},
                )
                doc_row = await conn.fetchrow(
                    """
                    UPDATE documents
                    SET stage = 'chunked',
                        chunked_id = $1,
                        chunk_count = $2,
                        claimed_at = NULL,
                        claimed_by = NULL,
                        updated_at = NOW()
                    WHERE id = $3
                    RETURNING *
                    """,
                    chunked_row["id"],
                    len(chunks),
                    doc_id,
                )
                return dict(doc_row)

    async def transition_to_embedded(self, doc_id: str) -> Dict[str, Any]:
        """Mark document as embedded and clear its claim."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE documents
                SET stage = 'embedded',
                    claimed_at = NULL,
                    claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                doc_id,
            )
            return dict(row) if row else {}

    async def record_error(
        self,
        doc_id: str,
        error: str,
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        """Record an error and move the document to 'error' or terminal 'failed'."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE documents
                SET attempts = attempts + 1,
                    last_error = $2,
                    stage = CASE
                        WHEN attempts + 1 >= $3 THEN 'failed'
                        ELSE 'error'
                    END,
                    claimed_at = NULL,
                    claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                doc_id,
                str(error)[:2000],
                max_attempts,
            )
            return dict(row) if row else {}

    async def reset_stale_claims(self, timeout_minutes: int = 30) -> int:
        """Reset documents stuck in a processing stage for longer than the timeout."""
        pool = await self._get_pool()
        timeout = timedelta(minutes=timeout_minutes)
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET stage = CASE stage
                        WHEN 'parsing' THEN 'registered'
                        WHEN 'chunking' THEN 'parsed'
                        WHEN 'embedding' THEN 'chunked'
                        ELSE stage
                    END,
                    claimed_at = NULL,
                    claimed_by = NULL,
                    updated_at = NOW()
                WHERE stage IN ('parsing', 'chunking', 'embedding')
                  AND claimed_at < NOW() - $1::interval
                """,
                timeout,
            )
        return _parse_count(result)

    async def reset_error_documents(self, max_attempts: int = 2) -> int:
        """Reset errored documents that still have retry attempts left."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET stage = 'registered',
                    claimed_at = NULL,
                    claimed_by = NULL,
                    updated_at = NOW()
                WHERE stage = 'error'
                  AND attempts < $1
                """,
                max_attempts,
            )
        return _parse_count(result)

    async def get_pending_doc_ids(self, stages: List[str]) -> List[str]:
        """Return IDs of unclaimed documents waiting in the given idle stages."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id
                FROM documents
                WHERE stage = ANY($1)
                  AND claimed_at IS NULL
                ORDER BY created_at
                """,
                stages,
            )
        return [r["id"] for r in rows]

    async def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a document's status row."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1",
                doc_id,
            )
            return dict(row) if row else None

    async def is_file_registered(self, file_name: str) -> bool:
        """Check whether a filename has already been registered."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM documents WHERE file_name = $1",
                file_name,
            )
            return row is not None

    async def get_parsed(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM document_parsed WHERE document_id = $1",
                doc_id,
            )
            return dict(row) if row else None

    async def get_chunked(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM document_chunked WHERE document_id = $1",
                doc_id,
            )
            return dict(row) if row else None


def _parse_count(command_result: str) -> int:
    """Parse 'UPDATE N' style asyncpg command result."""
    try:
        return int(command_result.split()[1])
    except Exception:
        return 0
