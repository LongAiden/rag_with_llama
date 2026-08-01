"""
Configuration management for the RAG application.
Handles environment setup, database configuration, and service initialization.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

from ingestion.validation.file_validator import FileValidator, FileValidationConfig
from models.models import SimpleRAGResponse

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
        self.agent = self._configure_pydantic_ai()
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
            print("✓ Logfire configured successfully")
        else:
            print("⚠️ LOGFIRE_WRITE_TOKEN not found, using default configuration")
            logfire.configure()

    def _configure_pydantic_ai(self) -> Optional[Agent]:
        """Configure Pydantic AI Agent with Ollama via OpenAI-compatible endpoint."""
        try:
            model = OpenAIModel(
                self.settings.ollama_model,
                provider=OllamaProvider(base_url=self.settings.ollama_base_url),
            )

            agent = Agent(
                model,
                output_type=SimpleRAGResponse,
                system_prompt="""You are a precise RAG (Retrieval-Augmented Generation) assistant.
Your job is to answer the user's question using ONLY the provided document sources.

Each source is structured as:
  [Source N (Page P)]
  [Matched chunk]: The specific passage retrieved by semantic search. It may begin with a
    section prefix like [Chapter].[Section] that shows where in the document it lives.
  [Full page context]: The complete text of that page, giving you broader surrounding context.

How to use the sources:
1. Read the [Matched chunk] first — it is the most semantically relevant passage.
2. Consult [Full page context] to understand the surrounding information and fill gaps.
3. Use the section prefix (e.g. [Chapter 3].[Security Threats]) to identify which part of the
   document the chunk belongs to and include that in your citations when helpful.
4. ALWAYS extract and state the relevant information — never say "I cannot find it" when the
   sources clearly contain the answer.
5. Quote or paraphrase key facts, definitions, steps, and explanations directly from the text.
6. Cite sources and page numbers (e.g. "Source 2, Page 7") whenever available.
7. If multiple sources address the same topic, synthesize them into one coherent answer.
8. Only state that information is unavailable if it is genuinely absent from ALL provided sources.

Respond with:
- answer: A thorough, well-cited response grounded in the sources
- confidence: Float 0-1 reflecting how completely the sources answer the question
- word_count: Number of words in your answer
- sources_used: Number of sources used (provided in the message)
- metadata: Any additional relevant notes (e.g. ambiguities, conflicting sources)"""
            )

            print("✓ Pydantic AI Agent configured successfully (Ollama)")
            return agent
        except Exception as e:
            print(f"❌ Pydantic AI configuration failed: {e}")
        return None


def get_ollama_model() -> str:
    """Get the configured Ollama model name."""
    settings = AppSettings()
    return settings.ollama_model
