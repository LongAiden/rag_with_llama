"""
Stage-based ingestion Celery tasks.

The pipeline is split into parse → chunk → embed. Each task claims *its own*
document row in the `documents` status table, does one stage, and persists the
intermediate artifact to Postgres. The embedding step reuses the vector store logic.

Two things here are load-bearing and easy to break:

* Chains are built from **immutable** signatures (``.si``). A plain ``.s`` chain
  passes each task's return value as the next task's first positional argument,
  which does not match ``task(doc_id)``.
* Every task runs on **one persistent event loop per worker process**. Using
  ``asyncio.run`` per task closes the loop while the process-wide asyncpg pool
  still holds connections bound to it, so the second task in a process fails with
  "Event loop is closed".
"""

import asyncio
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Optional

from celery import chain

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

INPUT_RAW_DIR = Path(os.getenv("INPUT_RAW_DIR", "input/raw"))
CLAIM_TIMEOUT_MINUTES = int(os.getenv("INGESTION_CLAIM_TIMEOUT_MINUTES", "30"))
MAX_ATTEMPTS = int(os.getenv("INGESTION_MAX_ATTEMPTS", "2"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))

UPLOAD_QUEUE = os.getenv("CELERY_UPLOAD_QUEUE", "upload")
INGESTION_QUEUE = os.getenv("CELERY_INGESTION_QUEUE", "ingestion")

# Stage a document must be in before a given task may claim it, and the
# in-progress stage it is moved to while that task runs.
STAGE_TRANSITIONS = {
    "parse": ("registered", "parsing"),
    "chunk": ("parsed", "chunking"),
    "embed": ("chunked", "embedding"),
}

_LOOP: Optional[asyncio.AbstractEventLoop] = None
_PIPELINES: Dict[str, Any] = {}


def _run(coro: Awaitable[Any]) -> Any:
    """Run a coroutine on this worker process's persistent event loop.

    The loop is deliberately never closed: the asyncpg pools cached in
    ConnectionPoolManager bind their connections to whichever loop created them.
    """
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP.run_until_complete(coro)


def _get_config():
    from config.app_config import AppConfig
    return AppConfig()


def _get_repo(config):
    from infra.db import IngestionRepository
    return IngestionRepository(connection_string=config.connection_string)


def _get_pipeline(config, table_name: str):
    """Return the embedding pipeline for a table, cached per worker process.

    Constructing one loads a SentenceTransformer model, so this must not happen
    per task — and only the embed stage needs it at all. Parse and chunk use the
    pipeline's stateless helpers directly.
    """
    if table_name not in _PIPELINES:
        from config.app_config import DEFAULT_EMBEDDING_MODEL
        from ingestion.embedding.vector_store import ChunkEmbeddingPipeline

        logger.info("Loading embedding pipeline for table %s", table_name)
        _PIPELINES[table_name] = ChunkEmbeddingPipeline(
            db_params=config.db_params,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            table_name=table_name,
        )
    return _PIPELINES[table_name]


def build_ingestion_chain(doc_id: str, from_stage: str = "registered", queue: str = INGESTION_QUEUE):
    """Build the remaining pipeline for a document as a Celery chain.

    Signatures are immutable (``.si``) so each stage receives only ``doc_id``, and
    the queue is set per task rather than relying on chain-level option
    propagation, so every stage provably lands on a worker that consumes it.

    Returns None when the stage has no work left (e.g. already embedded).
    """
    steps = {
        "registered": (parse_document_task, chunk_document_task, embed_document_task),
        "error": (parse_document_task, chunk_document_task, embed_document_task),
        "parsed": (chunk_document_task, embed_document_task),
        "chunked": (embed_document_task,),
    }.get(from_stage)

    if not steps:
        return None
    return chain(*[task.si(doc_id).set(queue=queue) for task in steps])


async def _run_stage(
    doc_id: str,
    stage_name: str,
    work: Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Claim this document for `stage_name`, run `work`, record failures.

    Claiming is by document id, not "next row in stage": a task dispatched for one
    document must never move another document into a processing stage, because the
    claim it would leave behind blocks that document until the stale sweep.
    """
    from_stage, processing_stage = STAGE_TRANSITIONS[stage_name]

    config = _get_config()
    repo = _get_repo(config)
    worker_id = f"{os.getpid()}-{stage_name}"

    doc = await repo.claim_document(
        doc_id=doc_id,
        current_stage=from_stage,
        processing_stage=processing_stage,
        worker_id=worker_id,
        timeout_minutes=CLAIM_TIMEOUT_MINUTES,
    )
    if doc is None:
        logger.info(
            "Stage %s skipped for %s: not in stage '%s', or claimed by another worker",
            stage_name, doc_id, from_stage,
        )
        return {"status": "skipped", "stage": stage_name, "document_id": doc_id}

    try:
        return await work(repo, config, doc)
    except Exception as exc:
        logger.exception("Stage %s failed for %s", stage_name, doc_id)
        await repo.record_error(doc_id, str(exc), MAX_ATTEMPTS, stage=stage_name)
        raise


def _chunk_to_dict(chunk: Any) -> Dict[str, Any]:
    """Flatten a chunker output object into the JSONB artifact shape."""
    return {
        "text": getattr(chunk, "text", ""),
        "token_count": getattr(chunk, "token_count", None),
        "start_index": getattr(chunk, "start_index", None),
        "end_index": getattr(chunk, "end_index", None),
        "page_number": getattr(chunk, "page_number", 1),
        "section_path": getattr(chunk, "section_path", ""),
        "full_content": getattr(chunk, "full_content", ""),
    }


async def _parse_document(doc_id: str) -> Dict[str, Any]:
    """Claim a registered document, parse it, and move it to 'parsed'."""

    async def work(repo, config, doc):
        from ingestion.embedding.vector_store import ChunkEmbeddingPipeline

        parsed = await ChunkEmbeddingPipeline.parse_file(
            doc["raw_storage_path"],
            doc_id,
            parse_backend=doc.get("parse_backend") or "",
        )
        await repo.transition_to_parsed(
            doc_id,
            parsed["parsed_text"],
            parsed["parser_used"],
            file_type=parsed["file_type"],
            metadata={
                "file_type": parsed["file_type"],
                # Non-PDF page numbers are resolved from this at chunk time.
                "page_mapping": parsed.get("page_mapping") or [],
            },
        )
        return {
            "status": "parsed",
            "document_id": doc_id,
            "parser_used": parsed["parser_used"],
        }

    return await _run_stage(doc_id, "parse", work)


async def _chunk_document(doc_id: str) -> Dict[str, Any]:
    """Claim a parsed document, chunk it, and move it to 'chunked'."""

    async def work(repo, config, doc):
        from config.app_config import DEFAULT_CHUNKING_SIMILARITY
        from ingestion.embedding.vector_store import ChunkEmbeddingPipeline

        parsed = await repo.get_parsed(doc_id)
        if not parsed:
            raise ValueError(f"Parsed artifact missing for {doc_id}")

        parsed_metadata = parsed.get("metadata") or {}
        # file_type drives markdown/page-aware chunking for PDFs — an empty value
        # silently downgrades every PDF to the generic chunker.
        file_type = doc.get("file_type") or parsed_metadata.get("file_type", "")
        chunk_size = doc.get("chunk_size") or DEFAULT_CHUNK_SIZE

        chunks = ChunkEmbeddingPipeline.chunk_parsed_document(
            {
                "parsed_text": parsed["parsed_text"],
                "file_type": file_type,
                "filename": doc["file_name"],
                "page_mapping": parsed_metadata.get("page_mapping") or [],
            },
            chunk_size=chunk_size,
            similarity_threshold=DEFAULT_CHUNKING_SIMILARITY,
            chunker_type=None,
        )

        chunk_dicts = [_chunk_to_dict(chunk) for chunk in chunks]
        await repo.transition_to_chunked(
            doc_id,
            chunk_dicts,
            chunk_size=chunk_size,
            metadata={
                "parser_used": parsed.get("parser_used"),
                "file_type": file_type,
            },
        )
        return {"status": "chunked", "document_id": doc_id, "chunk_count": len(chunk_dicts)}

    return await _run_stage(doc_id, "chunk", work)


async def _embed_document(doc_id: str) -> Dict[str, Any]:
    """Claim a chunked document, embed it, and move it to 'embedded'."""

    async def work(repo, config, doc):
        from config.app_config import DEFAULT_CHUNKING_SIMILARITY

        chunked = await repo.get_chunked(doc_id)
        if not chunked:
            raise ValueError(f"Chunked artifact missing for {doc_id}")

        chunks = [SimpleNamespace(**c) for c in chunked["chunks"]]
        chunked_metadata = chunked.get("metadata") or {}
        chunk_size = doc.get("chunk_size") or DEFAULT_CHUNK_SIZE
        file_type = doc.get("file_type") or chunked_metadata.get("file_type", "")

        metadata = dict(doc.get("metadata") or {})
        metadata.update({
            "filename": doc["file_name"],
            "content_type": doc.get("content_type"),
            "file_size": doc.get("file_size"),
            "validation_passed": True,
        })

        pipeline = _get_pipeline(config, doc["target_table_name"])
        await pipeline.embed_chunks(
            chunks=chunks,
            document_id=doc_id,
            chunk_size=chunk_size,
            similarity_threshold=DEFAULT_CHUNKING_SIMILARITY,
            filename=doc["file_name"],
            file_type=file_type,
            file_size=doc.get("file_size") or 0,
            parser_used=chunked_metadata.get("parser_used") or "",
            metadata=metadata,
        )
        await repo.transition_to_embedded(doc_id)
        return {"status": "embedded", "document_id": doc_id, "chunk_count": len(chunks)}

    return await _run_stage(doc_id, "embed", work)


async def _dispatch_pending(repo) -> int:
    """Queue the remaining pipeline for every idle document that has work left."""
    pending = await repo.get_pending_doc_ids(["registered", "parsed", "chunked"])
    dispatched = 0
    for pending_id in pending:
        status = await repo.get_document_status(pending_id)
        if not status:
            continue
        task_chain = build_ingestion_chain(
            pending_id,
            from_stage=status.get("stage"),
            queue=INGESTION_QUEUE,
        )
        if task_chain is None:
            continue
        task_chain.apply_async()
        dispatched += 1
    return dispatched


async def _scan_input_dir(repo) -> int:
    """Register files sitting in INPUT_RAW_DIR that are not tracked yet."""
    INPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    registered = 0

    for entry in sorted(INPUT_RAW_DIR.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue

        raw_path = str(entry.resolve())
        # Key on the stored path: uploads are written as '<uuid>_<name>' but
        # registered under '<name>', so a filename check re-registers them.
        if await repo.is_path_registered(raw_path):
            continue
        if await repo.is_file_registered(entry.name):
            continue

        await repo.register_document(
            doc_id=str(uuid.uuid4()),
            file_name=entry.name,
            raw_storage_path=raw_path,
            file_size=entry.stat().st_size,
            content_type=mimetypes.guess_type(raw_path)[0],
            target_table_name=os.getenv("DEFAULT_TABLE_NAME", "document_chunks"),
            chunk_size=DEFAULT_CHUNK_SIZE,
            parse_backend=os.getenv("DEFAULT_PARSE_BACKEND", ""),
            metadata={"source": "directory_scan"},
        )
        registered += 1

    return registered


async def _recover_and_dispatch() -> Dict[str, Any]:
    """
    Recovery pass (frequent):
    - release claims held by workers that died mid-stage
    - reset errored documents that still have attempts left
    - re-queue anything idle with work remaining

    The reset alone is not enough: a released document has no task in flight, so
    without the dispatch it would sit idle until the next directory scan.
    """
    config = _get_config()
    repo = _get_repo(config)

    stale = await repo.reset_stale_claims(CLAIM_TIMEOUT_MINUTES)
    retried = await repo.reset_error_documents(MAX_ATTEMPTS)
    dispatched = await _dispatch_pending(repo)

    return {
        "status": "ok",
        "stale_reset": stale,
        "retried": retried,
        "dispatched": dispatched,
    }


async def _register_and_dispatch() -> Dict[str, Any]:
    """
    Full scan (weekly): everything the recovery pass does, plus registering new
    files that were dropped into INPUT_RAW_DIR outside the upload API.
    """
    config = _get_config()
    repo = _get_repo(config)

    stale = await repo.reset_stale_claims(CLAIM_TIMEOUT_MINUTES)
    retried = await repo.reset_error_documents(MAX_ATTEMPTS)
    registered = await _scan_input_dir(repo)
    dispatched = await _dispatch_pending(repo)

    return {
        "status": "ok",
        "stale_reset": stale,
        "retried": retried,
        "registered": registered,
        "dispatched": dispatched,
    }


@celery_app.task(name="worker.ingestion_tasks.parse_document")
def parse_document_task(doc_id: str) -> Dict[str, Any]:
    return _run(_parse_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.chunk_document")
def chunk_document_task(doc_id: str) -> Dict[str, Any]:
    return _run(_chunk_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.embed_document")
def embed_document_task(doc_id: str) -> Dict[str, Any]:
    return _run(_embed_document(doc_id))


@celery_app.task(name="worker.ingestion_tasks.register_and_dispatch")
def register_and_dispatch_task() -> Dict[str, Any]:
    return _run(_register_and_dispatch())


@celery_app.task(name="worker.ingestion_tasks.recover_and_dispatch")
def recover_and_dispatch_task() -> Dict[str, Any]:
    return _run(_recover_and_dispatch())
