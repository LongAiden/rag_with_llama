"""
Ingestion status repository.

Provides atomic operations on the `documents` status table and the intermediate
`document_parsed` / `document_chunked` artifact tables.
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg

from app.infra.db.pool import ConnectionPoolManager


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
        doc_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new document. Always creates a new row.

        There is no filename de-duplication: re-uploading a file produces a second
        independent document with its own id. See
        migrations/005_drop_filename_dedupe.sql.

        `doc_name` is the human-readable label (defaults to the filename stem) and
        `domain` is the registry membership; `target_table_name` remains the
        physical chunk table the worker writes to.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    id, file_name, raw_storage_path, file_size, content_type,
                    target_table_name, chunk_size, parse_backend, metadata,
                    doc_name, domain
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                doc_name or Path(file_name).stem,
                domain,
            )
            return dict(row)

    async def claim_document(
        self,
        doc_id: str,
        current_stage: str,
        processing_stage: str,
        worker_id: str,
        timeout_minutes: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically claim one specific document and move it to processing_stage.

        Returns None if the document does not exist, is not in current_stage, or is
        already claimed by a live worker — in which case no row is modified. This is
        the claim used by the stage tasks, which are always dispatched for a known
        document id; claiming "whichever row is next" would strand other documents
        in a processing stage.
        """
        pool = await self._get_pool()
        timeout = timedelta(minutes=timeout_minutes)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE documents
                SET stage = $3,
                    claimed_at = NOW(),
                    claimed_by = $4,
                    updated_at = NOW()
                WHERE id = $1
                  AND stage = $2
                  AND (claimed_at IS NULL OR claimed_at < NOW() - $5::interval)
                RETURNING *
                """,
                doc_id,
                current_stage,
                processing_stage,
                worker_id,
                timeout,
            )
            return dict(row) if row else None

    async def transition_to_parsed(
        self,
        doc_id: str,
        parsed_text: str,
        parser_used: str,
        file_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store parsed text and mark document as parsed.

        `file_type` is persisted on the documents row because the chunking stage
        branches on it (markdown/page-aware chunking for PDFs).
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                parsed_row = await conn.fetchrow(
                    """
                    INSERT INTO document_parsed (document_id, parsed_text, parser_used, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (document_id) DO UPDATE SET
                        parsed_text = EXCLUDED.parsed_text,
                        parser_used = EXCLUDED.parser_used,
                        metadata    = EXCLUDED.metadata,
                        created_at  = NOW()
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
                        file_type = COALESCE(NULLIF($3, ''), file_type),
                        claimed_at = NULL,
                        claimed_by = NULL,
                        error_stage = NULL,
                        updated_at = NOW()
                    WHERE id = $2
                    RETURNING *
                    """,
                    parsed_row["id"],
                    doc_id,
                    file_type,
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
                    ON CONFLICT (document_id) DO UPDATE SET
                        chunks      = EXCLUDED.chunks,
                        chunk_size  = EXCLUDED.chunk_size,
                        chunk_count = EXCLUDED.chunk_count,
                        metadata    = EXCLUDED.metadata,
                        created_at  = NOW()
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
                        error_stage = NULL,
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
                    error_stage = NULL,
                    last_error = NULL,
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
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an error and move the document to 'error' or terminal 'failed'.

        `stage` is the pipeline stage that raised, stored so that a retry can resume
        from the last completed artifact instead of re-parsing from scratch.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE documents
                SET attempts = attempts + 1,
                    last_error = $2,
                    error_stage = COALESCE($4, error_stage),
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
                stage,
            )
            return dict(row) if row else {}

    async def reset_stale_claims(self, timeout_minutes: int = 30, max_attempts: int = 2) -> int:
        """Reset documents stuck in a processing stage for longer than the timeout.
        
        Increments attempts and moves to 'failed' if max_attempts is reached,
        preventing infinite re-dispatch loops when workers are killed (e.g., OOM).
        """
        pool = await self._get_pool()
        timeout = timedelta(minutes=timeout_minutes)
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET attempts = attempts + 1,
                    stage = CASE
                        WHEN attempts + 1 >= $2 THEN 'failed'
                        ELSE CASE stage
                            WHEN 'parsing' THEN 'registered'
                            WHEN 'chunking' THEN 'parsed'
                            WHEN 'embedding' THEN 'chunked'
                            ELSE stage
                        END
                    END,
                    last_error = CASE
                        WHEN attempts + 1 >= $2 THEN 'Worker killed (stale claim exceeded max attempts)'
                        ELSE last_error
                    END,
                    claimed_at = NULL,
                    claimed_by = NULL,
                    updated_at = NOW()
                WHERE stage IN ('parsing', 'chunking', 'embedding')
                  AND claimed_at < NOW() - $1::interval
                """,
                timeout,
                max_attempts,
            )
        return _parse_count(result)

    async def reset_error_documents(self, max_attempts: int = 2) -> int:
        """Reset errored documents that still have retry attempts left.

        Resumes from the last completed artifact rather than always restarting at
        'registered' — re-parsing a PDF through a VLM backend is the most expensive
        stage and should not be repeated for an embedding failure.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET stage = CASE
                        WHEN error_stage = 'embed' AND chunked_id IS NOT NULL THEN 'chunked'
                        WHEN error_stage = 'chunk' AND parsed_id IS NOT NULL THEN 'parsed'
                        ELSE 'registered'
                    END,
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

    async def delete_documents_for_table(self, target_table_name: str) -> List[Dict[str, Any]]:
        """Delete every documents row targeting a chunk table, returning the rows.

        Called when that chunk table is dropped. Without this the status rows survive
        at stage='embedded' pointing at a table that no longer exists, and the unique
        key on (file_name, target_table_name) then rejects any re-upload as a
        duplicate — the file becomes permanently un-ingestable.

        The artifact tables clean themselves up via ON DELETE CASCADE.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # If the ingestion status table has never been created (e.g. migrations
            # not run), there is nothing to clean up and the chunk table drop should
            # still succeed.
            table_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'documents')",
            )
            if not table_exists:
                return []
            rows = await conn.fetch(
                "DELETE FROM documents WHERE target_table_name = $1 RETURNING *",
                target_table_name,
            )
            return [dict(r) for r in rows]

    async def is_path_registered(self, raw_storage_path: str) -> bool:
        """Check whether a raw file path has already been registered.

        The directory scan must key on the stored path, not the display filename:
        uploads land on disk as '<uuid>_<name>' but register under '<name>', so a
        filename check would re-register every uploaded file as a new document.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM documents WHERE raw_storage_path = $1",
                raw_storage_path,
            )
            return row is not None

    async def delete_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Delete a document's status row, cascading to its parsed/chunked artifacts.

        Returns the deleted row (so callers can clean up the raw file and vector
        chunks), or None if it did not exist.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM documents WHERE id = $1 RETURNING *",
                doc_id,
            )
            return dict(row) if row else None

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
