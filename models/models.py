from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = None


class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    message: str
    chunks_created: Optional[int] = None


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
    max_file_size_mb: int = Field(
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
    similarity: float = Field(
        ge=0, le=1, description="Similarity score to query")
    document_id: str = Field(description="Document this chunk comes from")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional source metadata")


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
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")
