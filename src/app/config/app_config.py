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

from app.ingestion.validation.file_validator import FileValidator, FileValidationConfig

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

    # Ollama. Default matches docker-compose.yml and .env.example — the app runs in a
    # container, so 'localhost' would point at the container itself.
    ollama_base_url: str = Field(default='http://host.docker.internal:11434', validation_alias='OLLAMA_BASE_URL')
    ollama_model: str = Field(default='deepseek-r1:1.5b', validation_alias='OLLAMA_MODEL')
    ollama_vlm_model: str = Field(default='qwen3.5:0.8b', validation_alias='OLLAMA_VLM_MODEL')

    # Gemini/Google AI (kept for backward compatibility)
    google_api_key: Optional[str] = Field(default=None, validation_alias='GOOGLE_API_KEY')
    gemini_model: str = Field(default='gemini-2.5-flash', validation_alias='GEMINI_MODEL')

    # PDF parsing backend: "ollama" (Docling + Ollama VLM) or "gemini-docling" (Docling + Gemini)
    pdf_parser_backend: str = Field(default='ollama', validation_alias='PDF_PARSER_BACKEND')

    # Embedding
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    rerank_model: str = Field(default='cross-encoder/ms-marco-MiniLM-L-6-v2', validation_alias='RERANK_MODEL')
    # Cross-encoder input truncation. Chunks target DEFAULT_CHUNK_SIZE=512 tokens
    # and the model's own max_length is 512, so pairs run at the ceiling where
    # attention cost is quadratic. 256 roughly halves per-pair cost while the
    # reranker still SEES every candidate — unlike cutting vector_search_limit,
    # which removes candidates from consideration entirely.
    rerank_max_length: int = Field(default=256, validation_alias='RERANK_MAX_LENGTH')
    # Candidates pgvector returns before reranking. Was hardcoded HYBRID_LIMIT=20
    # in retrieval/search.py. Kept at 20: lowering it is a recall tradeoff, not a
    # free win — see docs/plans/20260818_retrieval_speed_vector_rerank.md Step 4.
    vector_search_limit: int = Field(default=20, validation_alias='VECTOR_SEARCH_LIMIT')
    # Final top-k after cross-encoder reranking. Was three literals in query_routes.py.
    rerank_top_k: int = Field(default=5, validation_alias='RERANK_TOP_K')
    # Eagerly load the cross-encoder on FastAPI startup so the first /query does
    # not pay the model-load hit. See app/api/app.py lifespan.
    preload_reranker: bool = Field(default=True, validation_alias='PRELOAD_RERANKER')

    # Table
    table_name: str = Field(default=DEFAULT_TABLE_NAME, validation_alias='DEFAULT_TABLE_NAME')

    # Celery queues
    celery_upload_queue: str = Field(default='upload', validation_alias='CELERY_UPLOAD_QUEUE')
    celery_ingestion_queue: str = Field(default='ingestion', validation_alias='CELERY_INGESTION_QUEUE')

    # Ingestion pipeline settings
    input_raw_dir: str = Field(default='data/input/raw', validation_alias='INPUT_RAW_DIR')
    # Derived, inspectable artifacts written by the parse and chunk stages. The DB
    # remains the source of truth; these exist so chunk boundaries can be read on
    # disk instead of queried out of JSONB.
    parsed_dir: str = Field(default='data/parsed', validation_alias='PARSED_DIR')
    chunks_dir: str = Field(default='data/chunks', validation_alias='CHUNKS_DIR')
    persist_ingestion_artifacts: bool = Field(
        default=True, validation_alias='PERSIST_INGESTION_ARTIFACTS'
    )
    preserve_math_notation: bool = Field(
        default=True, validation_alias='PRESERVE_MATH_NOTATION'
    )
    ingestion_max_attempts: int = Field(default=2, validation_alias='INGESTION_MAX_ATTEMPTS')
    ingestion_claim_timeout_minutes: int = Field(default=30, validation_alias='INGESTION_CLAIM_TIMEOUT_MINUTES')
    default_chunk_size: int = Field(default=512, validation_alias='DEFAULT_CHUNK_SIZE')
    default_parse_backend: str = Field(default='ollama', validation_alias='DEFAULT_PARSE_BACKEND')

    # PDF parse tuning. Defaults reproduce the previously hardcoded values
    # exactly, so exposing them changes no behaviour — they exist so the
    # CPU/thread and VLM-concurrency experiments are .env changes rather than
    # code edits. See docs/20260804_ingestion_performance_investigation.md.
    docling_num_threads: int = Field(default=2, validation_alias='DOCLING_NUM_THREADS')
    docling_page_batch_size: int = Field(default=40, validation_alias='DOCLING_PAGE_BATCH_SIZE')
    # TableFormer structure decoder: "accurate" or "fast". Docling time is not
    # uniform across a document — in a 504-page run one batch of 50 pages (the
    # dense multi-column index, classified as tables by the layout model) cost
    # 554.8s at 11.10 s/page against 1.47 s/page elsewhere, 42% of all docling
    # time. "fast" cut that batch to 121.1s and total docling 1333s -> 782s while
    # producing a structurally identical artifact (67 tables either way), so it
    # is the default. Set to "accurate" to revert if a document type shows table
    # loss. An unrecognised value logs and falls back to docling's own default.
    docling_tableformer_mode: str = Field(default='fast', validation_alias='DOCLING_TABLEFORMER_MODE')
    # 1: Ollama runs on the same host and serializes on one GPU. Measured on an
    # M1 — 3.87s/call at 1, 4.93s at 2, 20.62s at 4.
    vlm_concurrency: int = Field(default=1, validation_alias='VLM_CONCURRENCY')
    # qwen3.5 is a reasoning model and Ollama defaults thinking on: 87s/call
    # versus 2.3s, with the reasoning discarded unread. Tables additionally came
    # back empty every time thinking was on.
    ollama_vlm_think: bool = Field(default=False, validation_alias='OLLAMA_VLM_THINK')
    # VLM latency is pure decode at ~35 tok/s on this host, so elapsed IS the
    # output length, and Ollama's defaults leave it unbounded. Measured: the
    # same 218×54px crop returned 22 tokens in 1.55s and 342 tokens in 10.94s on
    # consecutive identical requests at temperature 0.8; the worst call of a
    # 191-call run took 93.3s. Greedy decoding gives 22-35 tokens and a verbatim
    # transcription instead of an invented one.
    ollama_vlm_temperature: float = Field(default=0.0, validation_alias='OLLAMA_VLM_TEMPERATURE')
    # Hard ceiling for the tail, not the primary lever — 384 leaves headroom
    # over the ~126-token honest description of a real figure.
    ollama_vlm_num_predict: int = Field(default=384, validation_alias='OLLAMA_VLM_NUM_PREDICT')
    # Page render resolution: docling renders at 72 * scale DPI and every VLM crop
    # is cut out of that page image, so this decides whether the model can read
    # anything at all. It was pinned at 0.6 (43 DPI, a 5px-tall 8pt glyph) with no
    # way to change it, and the measured result was confabulation: output length
    # ran INVERSELY to crop area, and a full-width screenshot delivered as 218x96px
    # came back as an invented salary table repeated six times. 2.0 = 144 DPI.
    # Costs ~290MB per 50-page batch and ~+0.3s/call of prefill.
    vlm_images_scale: float = Field(default=2.0, validation_alias='VLM_IMAGES_SCALE')
    # Short-side floor for sending a picture to the VLM, in POINTS so it survives a
    # change to vlm_images_scale. 113 of 191 calls in a 504-page run were strips
    # below this floor — rules and equation lines, not figures — and they consumed
    # 60% of the VLM budget hallucinating content that then gets embedded.
    # 107pt (1.48in) reproduces the previous 64px-at-scale-0.6 gate exactly.
    vlm_min_image_short_pt: float = Field(default=107.0, validation_alias='VLM_MIN_IMAGE_SHORT_PT')
    # Tables go to docling's TableFormer. A 0.8B VLM cannot read them — on
    # bert.pdf the 13-column GLUE table came back with headers "I, II, III, IV…"
    # and a small table came back with invented rows. Set true to restore the
    # old behaviour of routing complex tables to the VLM.
    vlm_tables: bool = Field(default=False, validation_alias='VLM_TABLES')

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
