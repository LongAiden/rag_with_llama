"""
LLM interaction logger.

Persists every query/answer cycle to llm_interactions via asyncpg.
Langfuse tracing is handled upstream via @observe() decorators in
retrieval/search.py and retrieval/llm_operations.py.

Token counts are owned exclusively here — Logfire handles status/errors only.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class InteractionPayload:
    question: str
    answer: str
    model: str
    backend: str
    latency_ms: int
    sources_used: int
    table_name: str
    rerank_method: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    session_id: Optional[str] = None


async def log_interaction(payload: InteractionPayload, connection_string: str) -> None:
    """
    Fire-and-forget coroutine — INSERT into llm_interactions via asyncpg.
    Errors are swallowed so the query path is never affected.
    """
    try:
        conn = await asyncpg.connect(connection_string)
        try:
            await conn.execute(
                """
                INSERT INTO llm_interactions (
                    session_id, question, answer, model, backend,
                    input_tokens, output_tokens, total_tokens,
                    latency_ms, sources_used, table_name, rerank_method
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                payload.session_id,
                payload.question,
                payload.answer,
                payload.model,
                payload.backend,
                payload.input_tokens,
                payload.output_tokens,
                payload.total_tokens,
                payload.latency_ms,
                payload.sources_used,
                payload.table_name,
                payload.rerank_method,
            )
        finally:
            await conn.close()
    except Exception as db_err:
        logger.error("llm_logger: DB insert failed (non-fatal): %s", db_err)
