"""
Configuration module for the RAG application.
"""

from .app_config import (
    AppConfig,
    AppSettings,
    DatabaseConfig,
    DEFAULT_TABLE_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNKING_SIMILARITY,
    ALLOWED_CONTENT_TYPES,
    get_ollama_model,
)

__all__ = [
    "AppConfig",
    "AppSettings",
    "DatabaseConfig",
    "DEFAULT_TABLE_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_CHUNKING_SIMILARITY",
    "ALLOWED_CONTENT_TYPES",
    "get_ollama_model",
]
