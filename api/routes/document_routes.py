"""API routes for document upload, status, and deletion."""
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logfire
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Header

from api.dependencies import get_config, get_pipeline_factory
from api.validators import validate_upload_params, require_access_password, validate_table_name
from config.app_config import AppSettings, DEFAULT_TABLE_NAME
from infra.db import IngestionRepository
from models.schemas import UploadResponse
from worker.ingestion_tasks import UPLOAD_QUEUE, build_ingestion_chain

# Via AppSettings so this agrees with the worker and honours .env.
INPUT_RAW_DIR = Path(AppSettings().input_raw_dir)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    table_name: str = Form("document_chunks"),
    parse_backend: str = Form(""),
    access_password: Optional[str] = Form(None),
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
):
    """Upload a document, persist the raw file, and queue it for processing."""
    require_access_password(access_password or x_app_password)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    table_name = (table_name or "").strip() or DEFAULT_TABLE_NAME
    validate_upload_params(chunk_size, file.content_type)
    validate_table_name(table_name)

    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid file name")

    document_id = str(uuid.uuid4())
    INPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = INPUT_RAW_DIR / f"{document_id}_{safe_filename}"

    try:
        content = await file.read()
        with open(raw_path, "wb") as f:
            f.write(content)

        validation_result = config.file_validator.validate_file(str(raw_path))
        if not validation_result.is_valid:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.error_message}",
            )

        repo = IngestionRepository(connection_string=config.connection_string)
        # Always registers a new document — re-uploading the same filename creates a
        # second independent document rather than being rejected as a duplicate.
        await repo.register_document(
            doc_id=document_id,
            file_name=safe_filename,
            raw_storage_path=str(raw_path.resolve()),
            file_size=validation_result.file_size,
            content_type=file.content_type or "application/octet-stream",
            target_table_name=table_name,
            chunk_size=chunk_size,
            parse_backend=parse_backend,
            metadata={
                "filename": safe_filename,
                "content_type": file.content_type,
                "file_size": validation_result.file_size,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_passed": True,
            },
        )

        task_chain = build_ingestion_chain(document_id, from_stage="registered", queue=UPLOAD_QUEUE)
        async_task = task_chain.apply_async()

        logfire.info(
            "Upload queued for Celery worker",
            document_id=document_id,
            filename=safe_filename,
            task_id=async_task.id,
            table_name=table_name,
        )

        return UploadResponse(
            status="queued",
            document_id=document_id,
            filename=safe_filename,
            message="Upload queued for processing. Poll /documents/{id}/status for progress.",
            chunks_created=None,
            task_id=async_task.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        tb = traceback.format_exc()
        logfire.error(
            "Upload processing failed",
            document_id=document_id,
            filename=safe_filename,
            error_type=type(e).__name__,
            error=str(e),
            traceback=tb,
        )
        print(f"[UPLOAD ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(e).__name__}: {e}",
        )


@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    config=Depends(get_config),
):
    """Return the ingestion status of a document from the status DB."""
    try:
        repo = IngestionRepository(connection_string=config.connection_string)
        status = await repo.get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "document_id": status["id"],
            "file_name": status["file_name"],
            "stage": status["stage"],
            "attempts": status["attempts"],
            "chunk_count": status["chunk_count"],
            "last_error": status["last_error"],
            "created_at": status["created_at"],
            "updated_at": status["updated_at"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document status: {type(e).__name__}: {e}",
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    delete_chunks: bool = True,
    delete_raw_file: bool = True,
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
    get_pipeline=Depends(get_pipeline_factory),
):
    """Delete a document from the status DB so it can be re-ingested."""
    require_access_password(x_app_password)
    try:
        repo = IngestionRepository(connection_string=config.connection_string)
        status = await repo.get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")

        chunks_deleted = 0
        if delete_chunks:
            table_name = status.get("target_table_name") or DEFAULT_TABLE_NAME
            validate_table_name(table_name)
            pipeline = await get_pipeline(table_name)
            chunks_deleted = await pipeline.delete_document(document_id)

        raw_file_deleted = False
        if delete_raw_file and status.get("raw_storage_path"):
            raw_path = Path(status["raw_storage_path"])
            try:
                raw_path.resolve().relative_to(INPUT_RAW_DIR.resolve())
            except ValueError:
                logfire.warn(
                    "Skipping raw file delete outside INPUT_RAW_DIR",
                    document_id=document_id,
                    raw_storage_path=str(raw_path),
                )
            else:
                raw_file_deleted = raw_path.exists()
                raw_path.unlink(missing_ok=True)

        await repo.delete_document(document_id)

        logfire.info(
            "Document deleted",
            document_id=document_id,
            chunks_deleted=chunks_deleted,
            raw_file_deleted=raw_file_deleted,
        )
        return {
            "status": "deleted",
            "document_id": document_id,
            "file_name": status.get("file_name"),
            "chunks_deleted": chunks_deleted,
            "raw_file_deleted": raw_file_deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {type(e).__name__}: {e}",
        )


@router.get("/supported-types")
async def get_supported_types(config=Depends(get_config)):
    """Get information about supported file types and validation config."""
    from ingestion.processors.page_utils import get_supported_file_types, list_available_processors

    supported_extensions = get_supported_file_types()
    processors = list_available_processors()

    return {
        "supported_extensions": supported_extensions,
        "max_file_size_mb": config.file_validator.config.max_file_size_mb,
        "supported_types": [ext.replace('.', '') for ext in supported_extensions],
        "registered_processors": [str(processor) for processor in processors],
        "vector_store_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "database_backend": "PostgreSQL + pgvector",
            "chunking_method": "semantic_chunking_with_chonkie",
            "processor_pattern": "Abstract Method + Factory Method",
        }
    }


