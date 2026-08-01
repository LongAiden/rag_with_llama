"""
Stage-based ingestion Celery tasks.

The pipeline is split into parse → chunk → embed. Each task claims a row in
the `documents` status table, does one stage, and persists the intermediate
artifact to Postgres. The embedding step reuses the existing vector store logic.
"""

import asyncio
import mimetypes
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from celery import chain

from worker.celery_app import celery_app

INPUT_RAW_DIR = Path(os.getenv("INPUT_RAW_DIR", "input/raw"))
CLAIM_TIMEOUT_MINUTES = int(os.getenv("INGESTION_CLAIM_TIMEOUT_MINUTES", "30"))
MAX_ATTEMPTS = int(os.getenv("INGESTION_MAX_ATTEMPTS", "2"))

_config = None
_repo = None
_pipeline_cache = {}


def get_config():
    global _config
    if _config is None:
        from config.app_config import AppConfig
        _config = AppConfig()
    return _config


def get_repo():
    global _repo
    if _repo is None:
        from repositories.ingestion_repository import IngestionRepository
        _repo = IngestionRepository(connection_string=get_config().connection_string)
    return _repo


async def get_pipeline(table_name: str):
    if table_name not in _pipeline_cache:
        from config.app_config import DEFAULT_EMBEDDING_MODEL
        from ingestion.embedding.vector_store import ChunkEmbeddingPipeline
        config = get_config()
        _pipeline_cache[table_name] = ChunkEmbeddingPipeline(
            db_params=config.db_params,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            table_name=table_name,
        )
    return _pipeline_cache[table_name]


async def _parse_document(doc_id: str) -> Dict[str, Any]:
    """Claim a registered document, parse it, and move it to 'parsed'."""
    repo = get_repo()
    worker_id = f"{os.getpid()}-parse"

    doc = await repo.claim_next_document(
        current_stage="registered",
        processing_stage="parsing",
        worker_id=worker_id,
        timeout_minutes=CLAIM_TIMEOUT_MINUTES,
    )
    if doc is None or doc["id"] != doc_id:
        return {"status": "skipped", "document_id": doc_id}

    try:
        pipeline = await get_pipeline(doc["target_table_name"])
        parsed = await pipeline.parse_file(
            doc["raw_storage_path"],
            doc_id,
            parse_backend=doc.get("parse_backend") or "",
        )
        await repo.transition_to_parsed(
            doc_id,
            parsed["parsed_text"],
            parsed["parser_used"],
            metadata={"file_type": parsed["file_type"]},
        )
        return {"status": "parsed", "document_id": doc_id}
    except Exception as exc:
        await repo.record_error(doc_id, str(exc), MAX_ATTEMPTS)
        raise


async def _chunk_document(doc_id: str) -> Dict[str, Any]:
    """Claim a parsed document, chunk it, and move it to 'chunked'."""
    repo = get_repo()
    worker_id = f"{os.getpid()}-chunk"

    doc = await repo.claim_next_document(
        current_stage="parsed",
        processing_stage="chunking",
        worker_id=worker_id,
        timeout_minutes=CLAIM_TIMEOUT_MINUTES,
    )
    if doc is None or doc["id"] != doc_id:
        return {"status": "skipped", "document_id": doc_id}

    try:
        parsed = await repo.get_parsed(doc_id)
        if not parsed:
            raise ValueError(f"Parsed artifact missing for {doc_id}")

        from config.app_config import DEFAULT_CHUNKING_SIMILARITY

        pipeline = await get_pipeline(doc["target_table_name"])
        chunks = pipeline.chunk_parsed_document(
            {
                "parsed_text": parsed["parsed_text"],
                "file_type": doc.get("file_type", ""),
                "filename": doc["file_name"],
            },
            chunk_size=doc.get("chunk_size", 512),
            similarity_threshold=DEFAULT_CHUNKING_SIMILARITY,
            chunker_type=None,
        )

        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                "text": getattr(chunk, "text", ""),
                "token_count": getattr(chunk, "token_count", None),
                "start_index": getattr(chunk, "start_index", None),
                "end_index": getattr(chunk, "end_index", None),
                "page_number": getattr(chunk, "page_number", 1),
                "section_path": getattr(chunk, "section_path", ""),
                "full_content": getattr(chunk, "full_content", ""),
            })

        await repo.transition_to_chunked(
            doc_id,
            chunk_dicts,
            chunk_size=doc.get("chunk_size", 512),
            metadata={"parser_used": parsed.get("parser_used")},
        )
        return {"status": "chunked", "document_id": doc_id, "chunk_count": len(chunk_dicts)}
    except Exception as exc:
        await repo.record_error(doc_id, str(exc), MAX_ATTEMPTS)
        raise


async def _embed_document(doc_id: str) -> Dict[str, Any]:
    """Claim a chunked document, embed it, and move it to 'embedded'."""
    repo = get_repo()
    worker_id = f"{os.getpid()}-embed"

    doc = await repo.claim_next_document(
        current_stage="chunked",
        processing_stage="embedding",
        worker_id=worker_id,
        timeout_minutes=CLAIM_TIMEOUT_MINUTES,
    )
    if doc is None or doc["id"] != doc_id:
        return {"status": "skipped", "document_id": doc_id}

    try:
        chunked = await repo.get_chunked(doc_id)
        if not chunked:
            raise ValueError(f"Chunked artifact missing for {doc_id}")

        chunks = [SimpleNamespace(**c) for c in chunked["chunks"]]
        from config.app_config import DEFAULT_CHUNKING_SIMILARITY

        pipeline = await get_pipeline(doc["target_table_name"])

        metadata = dict(doc.get("metadata") or {})
        metadata.update({
            "filename": doc["file_name"],
            "content_type": doc.get("content_type"),
            "file_size": doc.get("file_size"),
            "validation_passed": True,
        })

        await pipeline.embed_chunks(
            chunks=chunks,
            document_id=doc_id,
            chunk_size=doc.get("chunk_size", 512),
            similarity_threshold=DEFAULT_CHUNKING_SIMILARITY,
            filename=doc["file_name"],
            file_type=doc.get("file_type", ""),
            file_size=doc.get("file_size", 0),
            parser_used=chunked.get("metadata", {}).get("parser_used", ""),
            metadata=metadata,
        )
        await repo.transition_to_embedded(doc_id)
        return {"status": "embedded", "document_id": doc_id}
    except Exception as exc:
        await repo.record_error(doc_id, str(exc), MAX_ATTEMPTS)
        raise


async def _register_and_dispatch() -> Dict[str, Any]:
    """
    Weekly orchestrator:
    - sweep stale claims
    - reset errored documents for retry
    - scan input/raw/ and register new files
    - dispatch processing chains for pending documents
    """
    repo = get_repo()

    stale = await repo.reset_stale_claims(CLAIM_TIMEOUT_MINUTES)
    retried = await repo.reset_error_documents(MAX_ATTEMPTS)

    INPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    registered = 0
    for entry in INPUT_RAW_DIR.iterdir():
        if not entry.is_file():
            continue
        file_name = entry.name
        if await repo.is_file_registered(file_name):
            continue
        doc_id = str(uuid.uuid4())
        await repo.register_document(
            doc_id=doc_id,
            file_name=file_name,
            raw_storage_path=str(entry.resolve()),
            file_size=entry.stat().st_size,
            content_type=mimetypes.guess_type(str(entry))[0],
            target_table_name=os.getenv("DEFAULT_TABLE_NAME", "document_chunks"),
            chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "512")),
            parse_backend=os.getenv("DEFAULT_PARSE_BACKEND", ""),
            metadata={"source": "weekly_scan"},
        )
        registered += 1

    pending = await repo.get_pending_doc_ids(["registered", "parsed", "chunked"])
    dispatched = 0
    for pending_id in pending:
        status = await repo.get_document_status(pending_id)
        if not status:
            continue
        stage = status.get("stage")
        if stage in ("registered", "error"):
            task_chain = chain(
                parse_document_task.s(pending_id),
                chunk_document_task.s(pending_id),
                embed_document_task.s(pending_id),
            )
        elif stage == "parsed":
            task_chain = chain(
                chunk_document_task.s(pending_id),
                embed_document_task.s(pending_id),
            )
        elif stage == "chunked":
            task_chain = embed_document_task.s(pending_id)
        else:
            continue
        task_chain.apply_async()
        dispatched += 1

    return {
        "status": "ok",
        "stale_reset": stale,
        "retried": retried,
        "registered": registered,
        "dispatched": dispatched,
    }


async def _sweep_stale_documents() -> Dict[str, Any]:
    """Reset documents stuck in a processing stage for too long."""
    repo = get_repo()
    count = await repo.reset_stale_claims(CLAIM_TIMEOUT_MINUTES)
    return {"status": "ok", "stale_reset": count}


@celery_app.task(name="worker.ingestion_tasks.parse_document")
def parse_document_task(doc_id: str) -> Dict[str, Any]:
    return asyncio.run(_parse_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.chunk_document")
def chunk_document_task(doc_id: str) -> Dict[str, Any]:
    return asyncio.run(_chunk_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.embed_document")
def embed_document_task(doc_id: str) -> Dict[str, Any]:
    return asyncio.run(_embed_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.register_and_dispatch")
def register_and_dispatch_task() -> Dict[str, Any]:
    return asyncio.run(_register_and_dispatch())


@celery_app.task(name="worker.ingestion_tasks.sweep_stale_documents")
def sweep_stale_documents_task() -> Dict[str, Any]:
    return asyncio.run(_sweep_stale_documents())
