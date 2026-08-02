"""
Validation utilities for the RAG application.
Handles parameter validation, authentication, and feature flags.
"""

import os
from typing import Optional
from fastapi import HTTPException

from api.config import ALLOWED_CONTENT_TYPES
from infra.db.identifiers import validate_table_name as _validate_table_name


def validate_table_name(table_name: str):
    """
    Validate that a table name is a safe PostgreSQL identifier.
    Blocks SQL injection via semicolons, comments, spaces, and special chars.

    Raises:
        HTTPException: 400 if the name does not match safe identifier rules
    """
    try:
        _validate_table_name(table_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc


def validate_upload_params(chunk_size: int, content_type: str):
    """
    Validate upload parameters.

    Args:
        chunk_size: Size of text chunks for processing
        content_type: MIME type of uploaded file

    Raises:
        HTTPException: If parameters are invalid
    """
    if not (128 <= chunk_size <= 2048):
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size must be between 128-2048"
        )
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files supported"
        )


def require_access_password(provided_password: Optional[str]):
    """
    Enforce a simple shared password for browser-facing forms when configured.
    No-op if APP_ACCESS_PASSWORD/FRONTEND_PASSWORD is unset.

    Args:
        provided_password: Password provided by the user

    Raises:
        HTTPException: If password is required but incorrect
    """
    expected = os.getenv("APP_ACCESS_PASSWORD") or os.getenv("FRONTEND_PASSWORD")
    expected = expected.strip() if expected else ""

    if expected and provided_password != expected:
        raise HTTPException(status_code=403, detail="Invalid access password")


def celery_enabled() -> bool:
    """
    Check whether Celery offloading is enabled for entity extraction.

    Returns:
        bool: True if Celery is enabled
    """
    return os.getenv("USE_CELERY_FOR_EXTRACTION", "false").lower() == "true"


def celery_upload_enabled() -> bool:
    """
    Check whether uploads should be offloaded to Celery.

    Returns:
        bool: True if Celery upload processing is enabled
    """
    return os.getenv("USE_CELERY_FOR_UPLOAD", "false").lower() == "true"


def entity_extraction_enabled() -> bool:
    """
    Check whether entity extraction is enabled.

    Returns:
        bool: True if entity extraction is enabled
    """
    return os.getenv("ENABLE_ENTITY_EXTRACTION", "true").lower() == "true"
