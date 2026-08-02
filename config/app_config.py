"""
Configuration management for the RAG application.
Handles environment setup, database configuration, and service initialization.
"""

import logging
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import logfire

from ingestion.validation.file_validator import FileValidator, FileValidationConfig

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TABLE_NAME = "document_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNKING_SIMILARITY = 0.5
ALLOWED_CONTENT_TYPES = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain'
]


class DatabaseConfig(BaseSettings):
    """Database configuration using pydantic-settings."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # Database settings with fallback aliases
    host: str = Field(default='localhost', validation_alias='POSTGRES_HOST')
    port: str = Field(default='5432', validation_alias='POSTGRES_PORT')
    dbname: str = Field(default='rag_db', validation_alias='POSTGRES_DB')
    user: str = Field(default='admin', validation_alias='POSTGRES_USER')
    password: str = Field(default='admin', validation_alias='POSTGRES_PASSWORD')

    def to_dict(self):
        """Convert to dictionary format for psycopg2/asyncpg."""
        return {
            'host': self.host,
            'port': self.port,
            'dbname': self.dbname,
            'user': self.user,
            'password': self.password
        }


class AppSettings(BaseSettings):
    """Application settings using pydantic-settings."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # Logfire
    logfire_write_token: Optional[str] = Field(default=None, validation_alias='LOGFIRE_WRITE_TOKEN')

    # Ollama
    ollama_base_url: str = Field(default='http://localhost:11434', validation_alias='OLLAMA_BASE_URL')
    ollama_model: str = Field(default='deepseek-r1:1.5b', validation_alias='OLLAMA_MODEL')
    ollama_vlm_model: str = Field(default='qwen3.5:9b', validation_alias='OLLAMA_VLM_MODEL')

    # Gemini/Google AI (kept for backward compatibility)
    google_api_key: Optional[str] = Field(default=None, validation_alias='GOOGLE_API_KEY')
    gemini_model: str = Field(default='gemini-2.5-flash', validation_alias='GEMINI_MODEL')

    # PDF parsing backend: "ollama" (Docling + Ollama VLM) or "gemini-docling" (Docling + Gemini)
    pdf_parser_backend: str = Field(default='ollama', validation_alias='PDF_PARSER_BACKEND')

    # Embedding
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)

    # Table
    table_name: str = Field(default=DEFAULT_TABLE_NAME)

    # Ingestion pipeline settings
    input_raw_dir: str = Field(default='input/raw', validation_alias='INPUT_RAW_DIR')
    ingestion_max_attempts: int = Field(default=2, validation_alias='INGESTION_MAX_ATTEMPTS')
    ingestion_claim_timeout_minutes: int = Field(default=30, validation_alias='INGESTION_CLAIM_TIMEOUT_MINUTES')
    default_chunk_size: int = Field(default=512, validation_alias='DEFAULT_CHUNK_SIZE')
    default_parse_backend: str = Field(default='ollama', validation_alias='DEFAULT_PARSE_BACKEND')

    # Langfuse observability (optional)
    langfuse_host: Optional[str] = Field(default=None, validation_alias='LANGFUSE_HOST')
    langfuse_public_key: Optional[str] = Field(default=None, validation_alias='LANGFUSE_PUBLIC_KEY')
    langfuse_secret_key: Optional[str] = Field(default=None, validation_alias='LANGFUSE_SECRET_KEY')


class AppConfig:
    """Global application configuration and service initialization."""

    def __init__(self, settings: Optional[AppSettings] = None, db_config: Optional[DatabaseConfig] = None):
        # Disable tokenizers parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load settings
        self.settings = settings or AppSettings()
        self.db_config = db_config or DatabaseConfig()

        # Initialize logfire
        self._configure_logfire()

        # Database configuration (backward compatible dict format)
        self.db_params = self.db_config.to_dict()

        # Service initialization (lazy loading for performance)
        self.file_validator = FileValidator(FileValidationConfig())
        self.pipeline = None  # Lazy initialization
        self.reranker = None  # Lazy initialization
        self.graph_pool = None  # Lazy initialization

    @property
    def connection_string(self) -> str:
        p = self.db_config
        return f"postgresql://{p.user}:{p.password}@{p.host}:{p.port}/{p.dbname}"

    def _configure_logfire(self):
        """Configure logfire with token from settings."""
        if self.settings.logfire_write_token:
            logfire.configure(token=self.settings.logfire_write_token)
            logger.info("Logfire configured successfully")
        else:
            logger.warning("LOGFIRE_WRITE_TOKEN not found, using default configuration")
            logfire.configure()


def get_ollama_model() -> str:
    """Get the configured Ollama model name."""
    settings = AppSettings()
    return settings.ollama_model
