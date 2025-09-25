from typing import List, Optional
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
    allowed_extensions: List[str] = Field(default=[".pdf", ".docx", ".txt"])

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v
