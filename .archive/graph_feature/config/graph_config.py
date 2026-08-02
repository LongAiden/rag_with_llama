"""
Configuration for Graph and Entity Extraction.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class GraphConfig(BaseSettings):
    """Graph and entity extraction configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ============================================
    # LLM Provider Selection
    # ============================================
    llm_provider: str = Field(
        default="ollama",
        validation_alias="GRAPH_LLM_PROVIDER",
        description="LLM provider for graph extraction: 'ollama' (default, local) or 'gemini' (cloud, slower)"
    )

    # ============================================
    # LLM Configuration
    # ============================================
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Gemini API key for entity/relationship extraction"
    )
    gemini_model: str = Field(
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        description="Gemini model to use for extraction"
    )

    # ============================================
    # Ollama Configuration
    # ============================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="deepseek-r1:8b",
        validation_alias="OLLAMA_MODEL",
        description="Ollama text model for entity/relationship extraction"
    )
    ollama_vlm_model: str = Field(
        default="llama3.2-vision:11b",
        validation_alias="OLLAMA_VLM_MODEL",
        description="Ollama vision model for image content extraction"
    )

    # ============================================
    # API Retry & Timeout Configuration
    # ============================================
    gemini_request_timeout: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Timeout for Gemini API requests in seconds"
    )
    gemini_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for failed API calls"
    )
    gemini_retry_initial_delay: float = Field(
        default=2.0,
        ge=0.5,
        le=60.0,
        description="Initial delay before first retry in seconds"
    )
    gemini_retry_max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay between retries in seconds"
    )
    gemini_retry_exponential_base: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff base multiplier"
    )
    gemini_rate_limit_pause: float = Field(
        default=65.0,
        ge=10.0,
        le=600.0,
        description="Pause duration when rate limit is hit (seconds)"
    )

    # ============================================
    # Entity Extraction Configuration
    # ============================================
    entity_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for entity extraction (0-1)"
    )
    entity_extraction_enabled: bool = Field(
        default=True,
        description="Enable automatic entity extraction"
    )
    max_entities_per_chunk: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of entities to extract per chunk"
    )

    # ============================================
    # Relationship Extraction Configuration
    # ============================================
    relationship_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for relationship extraction (0-1)"
    )
    relationship_extraction_enabled: bool = Field(
        default=True,
        description="Enable automatic relationship extraction"
    )
    max_relationships_per_chunk: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of relationships to extract per chunk"
    )

    # ============================================
    # Graph Algorithm Configuration
    # ============================================
    default_max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Default maximum hops for connected entity queries"
    )
    pagerank_damping_factor: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Damping factor for PageRank algorithm"
    )
    pagerank_max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum iterations for PageRank algorithm"
    )

    # ============================================
    # Vector Embedding Configuration
    # ============================================
    entity_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for entity embeddings"
    )
    entity_embedding_dimension: int = Field(
        default=384,
        description="Dimension of entity embeddings (must match model)"
    )

    # ============================================
    # Batch Processing Configuration
    # ============================================
    batch_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of chunks to process per batch (smaller = safer for rate limits)"
    )
    enable_parallel_extraction: bool = Field(
        default=True,
        description="Enable parallel processing for batch extraction"
    )
    max_concurrent_api_calls: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum concurrent Gemini API calls (higher = faster but more rate limit risk)"
    )
    inter_batch_delay: float = Field(
        default=1.5,
        ge=0.0,
        le=30.0,
        description="Delay between batches in seconds (helps avoid rate limits)"
    )

    # ============================================
    # Entity Type Filtering
    # ============================================
    enabled_entity_types: List[str] = Field(
        default=[
            "ALGORITHM", "MODEL", "ARCHITECTURE", "TECHNIQUE",
            "DATASET", "METRIC", "TASK", "FRAMEWORK",
            "LAYER", "ACTIVATION", "LOSS_FUNCTION", "OPTIMIZER",
            "CONCEPT", "PAPER", "RESEARCHER", "ORGANIZATION"
        ],
        description="Enabled entity types for extraction (empty = all enabled)"
    )

    # ============================================
    # Relationship Type Filtering
    # ============================================
    enabled_relationship_types: List[str] = Field(
        default=[
            "USES", "TRAINED_ON", "IMPROVES", "OUTPERFORMS",
            "PART_OF", "BASED_ON", "IS_A", "EXTENDS",
            "SOLVES", "ADDRESSES", "EVALUATED_ON", "IMPLEMENTS"
        ],
        description="Enabled relationship types (empty = all enabled)"
    )

    # ============================================
    # Database Configuration
    # ============================================
    db_pool_min_size: int = Field(
        default=2,
        ge=1,
        description="Minimum database connection pool size"
    )
    db_pool_max_size: int = Field(
        default=10,
        ge=2,
        description="Maximum database connection pool size"
    )
    db_pool_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Database connection timeout in seconds"
    )

    # ============================================
    # Performance Configuration
    # ============================================
    enable_entity_caching: bool = Field(
        default=True,
        description="Enable caching for entity lookups"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache time-to-live in seconds (1 hour default)"
    )

    # ============================================
    # Logging Configuration
    # ============================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_entity_extraction: bool = Field(
        default=True,
        description="Log entity extraction details"
    )
    log_relationship_extraction: bool = Field(
        default=True,
        description="Log relationship extraction details"
    )

    # (rest unchanged)


_graph_config: Optional[GraphConfig] = None


def get_graph_config() -> GraphConfig:
    """Get the graph configuration singleton."""
    global _graph_config
    if _graph_config is None:
        _graph_config = GraphConfig()
    return _graph_config


# Export the config instance
graph_config = get_graph_config()


# ============================================
# Helper Functions
# ============================================

def is_entity_type_enabled(entity_type: str) -> bool:
    """Check if an entity type is enabled."""
    config = get_graph_config()
    if not config.enabled_entity_types:
        return True  # All enabled if list is empty
    return entity_type.upper() in [t.upper() for t in config.enabled_entity_types]


def is_relationship_type_enabled(relationship_type: str) -> bool:
    """Check if a relationship type is enabled."""
    config = get_graph_config()
    if not config.enabled_relationship_types:
        return True  # All enabled if list is empty
    return relationship_type.upper() in [t.upper() for t in config.enabled_relationship_types]


def get_extraction_config() -> dict:
    """Get extraction configuration as dictionary."""
    config = get_graph_config()
    return {
        "entity_confidence_threshold": config.entity_confidence_threshold,
        "relationship_confidence_threshold": config.relationship_confidence_threshold,
        "max_entities_per_chunk": config.max_entities_per_chunk,
        "max_relationships_per_chunk": config.max_relationships_per_chunk,
        "enabled_entity_types": config.enabled_entity_types,
        "enabled_relationship_types": config.enabled_relationship_types,
    }


# ============================================
# Environment Variables Reference
# ============================================
"""
Add these to your .env file:

# Graph Configuration
ENTITY_CONFIDENCE_THRESHOLD=0.6
RELATIONSHIP_CONFIDENCE_THRESHOLD=0.6
MAX_ENTITIES_PER_CHUNK=50
MAX_RELATIONSHIPS_PER_CHUNK=100
DEFAULT_MAX_HOPS=2

# LLM Configuration
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# API Retry & Timeout (NEW - for handling rate limits)
GEMINI_REQUEST_TIMEOUT=60.0           # Timeout per API request (seconds)
GEMINI_MAX_RETRIES=3                  # Max retry attempts on failure
GEMINI_RETRY_INITIAL_DELAY=2.0        # Initial delay before retry (seconds)
GEMINI_RETRY_MAX_DELAY=60.0           # Max delay between retries (seconds)
GEMINI_RETRY_EXPONENTIAL_BASE=2.0     # Exponential backoff multiplier
GEMINI_RATE_LIMIT_PAUSE=65.0          # Pause when rate limit hit (seconds)

# Performance & Rate Limiting
BATCH_SIZE=5                          # Chunks per batch (smaller = safer)
ENABLE_PARALLEL_EXTRACTION=true
MAX_CONCURRENT_API_CALLS=2            # Concurrent Gemini calls (1-5)
INTER_BATCH_DELAY=1.5                 # Seconds between batches
ENABLE_ENTITY_CACHING=true
CACHE_TTL_SECONDS=3600

# Logging
LOG_LEVEL=INFO
LOG_ENTITY_EXTRACTION=true
LOG_RELATIONSHIP_EXTRACTION=true
"""
