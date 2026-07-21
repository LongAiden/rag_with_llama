from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = None
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    rerank_top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of results to return after reranking")
    model: str = Field(default="deepseek-r1:1.5b", description="LLM model to use for response generation")
    table_name: str = Field(default="document_chunks", description="Database table to search")
    session_id: Optional[str] = Field(None, description="Optional session identifier for grouping interactions")


class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    message: str
    chunks_created: Optional[int] = None
    table_count: Optional[int] = Field(
        default=None,
        description="Number of chunk tables in the database after upload",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Celery task id if processing/extraction was queued",
    )


class SupportedFileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class FileValidationResult(BaseModel):
    """Result of file validation."""
    filename: str
    file_type: Optional[SupportedFileType]
    is_valid: bool
    file_size: int = Field(description="File size in bytes")
    error_message: Optional[str] = None


class FileValidationConfig(BaseModel):
    """Configuration for file validation."""
    max_file_size_mb: float = Field(
        default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(default=[".pdf", ".txt"])

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v


class RAGSource(BaseModel):
    """Information about a source used in RAG response."""
    chunk_id: str = Field(description="Unique identifier for the source chunk")
    text: str = Field(description="Text content of the chunk")
    similarity: float = Field(
        ge=0, le=1, description="Similarity score to query")
    document_id: str = Field(description="Document this chunk comes from")
    page_number: Optional[int] = Field(
        None, description="Page number where this chunk appears")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional source metadata")
    rerank_score: Optional[float] = Field(
        None, description="Reranking score from cross-encoder model")
    bm25_score: Optional[float] = Field(
        None, description="BM25 lexical score")
    rrf_score: Optional[float] = Field(
        None, description="Reciprocal Rank Fusion score combining vector and BM25")
    graph_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entities associated with this chunk from the knowledge graph"
    )


class RAGResponseMetadata(BaseModel):
    """Metadata for RAG response generation."""
    chunks_found: int = Field(description="Number of relevant chunks found")
    avg_similarity: float = Field(
        description="Average similarity score of used chunks")
    search_method: str = Field(
        description="Search method used (e.g., pgvector_cosine)")
    threshold_used: float = Field(description="Similarity threshold applied")
    word_count: Optional[int] = Field(
        None, description="Number of words in response")
    confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Response confidence score")
    reranking_enabled: Optional[bool] = Field(
        None, description="Whether reranking was applied")
    avg_rerank_score: Optional[float] = Field(
        None, description="Average reranking score")
    graph_enriched: Optional[bool] = Field(
        None, description="Whether knowledge graph entities were used to enrich results")
    input_tokens: Optional[int] = Field(None, description="Input tokens used by LLM")
    output_tokens: Optional[int] = Field(None, description="Output tokens used by LLM")
    total_tokens: Optional[int] = Field(None, description="Total tokens used by LLM")


class RAGResponse(BaseModel):
    """Structured response from RAG system using Pydantic AI."""
    query: str = Field(description="Original user query")
    answer: str = Field(description="Generated answer from LLM")
    sources: List[RAGSource] = Field(
        description="Sources used to generate the answer")
    search_stats: RAGResponseMetadata = Field(
        description="Metadata about the search and response generation")
    table_used: Optional[str] = Field(
        None, description="Database table used for search")


class SimpleRAGResponse(BaseModel):
    """Simplified response format for backward compatibility."""
    answer: str = Field(description="Generated answer")
    confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Response confidence")
    word_count: int = Field(description="Number of words in response")
    sources_used: int = Field(description="Number of sources used")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens used")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens used")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")
