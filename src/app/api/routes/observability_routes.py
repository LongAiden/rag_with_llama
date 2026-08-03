"""
Observability API routes.
Provides query history, aggregate stats, and time-series metrics
from the llm_interactions table.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.infra.db.pool import ConnectionPoolManager

router = APIRouter(prefix="/observability", tags=["observability"])

# Connection string injected at registration time by set_connection_string().
_connection_string: str = ""


def set_connection_string(connection_string: str) -> None:
    """Wire this router to a database. Called once during app startup."""
    global _connection_string
    _connection_string = connection_string


def get_connection_string() -> str:
    return _connection_string


@asynccontextmanager
async def _connection():
    """Borrow a pooled connection.

    Uses the shared ConnectionPoolManager rather than asyncpg.connect() so these
    endpoints do not open a new TCP/TLS/auth handshake per request.
    """
    try:
        pool = await ConnectionPoolManager.get_pool(get_connection_string())
        conn = await pool.acquire()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    try:
        yield conn
    finally:
        await pool.release(conn)


@router.get("/stats")
async def get_stats(days: int = Query(default=7, ge=1, le=365)) -> Dict[str, Any]:
    """Aggregate stats grouped by model and backend for the last N days."""
    async with _connection() as conn:
        rows = await conn.fetch(
            """
            SELECT
                model,
                backend,
                COUNT(*)                                                    AS total_queries,
                ROUND(AVG(input_tokens))                                    AS avg_input_tokens,
                ROUND(AVG(output_tokens))                                   AS avg_output_tokens,
                ROUND(AVG(total_tokens))                                    AS avg_total_tokens,
                SUM(total_tokens)                                           AS sum_total_tokens,
                ROUND(AVG(latency_ms))                                      AS avg_latency_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)) AS p95_latency_ms,
                ROUND(AVG(sources_used))                                    AS avg_sources_used
            FROM llm_interactions
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL
            GROUP BY model, backend
            ORDER BY total_queries DESC
            """,
            str(days),
        )
        return {
            "period_days": days,
            "groups": [dict(r) for r in rows],
        }


@router.get("/history")
async def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    model: Optional[str] = None,
    backend: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Paginated interaction log with optional filters."""
    async with _connection() as conn:
        offset = (page - 1) * page_size
        total = await conn.fetchval(
            """
            SELECT COUNT(*) FROM llm_interactions
            WHERE ($1::text IS NULL OR model = $1)
              AND ($2::text IS NULL OR backend = $2)
              AND ($3::timestamptz IS NULL OR created_at >= $3)
              AND ($4::timestamptz IS NULL OR created_at <= $4)
            """,
            model, backend, since, until,
        )
        rows = await conn.fetch(
            """
            SELECT id, session_id, question, answer, model, backend,
                   input_tokens, output_tokens, total_tokens,
                   latency_ms, sources_used, table_name, rerank_method, created_at
            FROM llm_interactions
            WHERE ($1::text IS NULL OR model = $1)
              AND ($2::text IS NULL OR backend = $2)
              AND ($3::timestamptz IS NULL OR created_at >= $3)
              AND ($4::timestamptz IS NULL OR created_at <= $4)
            ORDER BY created_at DESC
            LIMIT $5 OFFSET $6
            """,
            model, backend, since, until, page_size, offset,
        )
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [dict(r) for r in rows],
        }


@router.get("/metrics")
async def get_metrics(
    bucket: str = Query(default="hour", pattern="^(hour|day|week)$"),
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Time-bucketed query counts, token usage, and latency for charting."""
    since = since or datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    until = until or datetime.now(timezone.utc)
    async with _connection() as conn:
        rows = await conn.fetch(
            """
            SELECT
                DATE_TRUNC($1, created_at)  AS bucket,
                COUNT(*)                     AS query_count,
                ROUND(AVG(total_tokens))     AS avg_total_tokens,
                SUM(total_tokens)            AS sum_total_tokens,
                ROUND(AVG(latency_ms))       AS avg_latency_ms
            FROM llm_interactions
            WHERE created_at >= $2 AND created_at <= $3
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            bucket, since, until,
        )
        return {
            "bucket": bucket,
            "since": since.isoformat(),
            "until": until.isoformat(),
            "data": [dict(r) for r in rows],
        }
