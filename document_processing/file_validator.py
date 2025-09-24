"""
File validation script using Pydantic and pydantic_ai for checking PDF, DOCX, and TXT files.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, field_validator
from pydantic_ai import Agent, RunContext


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
    mime_type: Optional[str] = None
    error_message: Optional[str] = None


class FileValidationConfig(BaseModel):
    """Configuration for file validation."""
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(default=[".pdf", ".docx", ".txt"])

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v


class FileValidator:
    """File validator class for checking PDF, DOCX, and TXT files."""

    def __init__(self, config: FileValidationConfig = None):
        self.config = config or FileValidationConfig()
        self.mime_type_mapping = {
            "application/pdf": SupportedFileType.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": SupportedFileType.DOCX,
            "text/plain": SupportedFileType.TXT,
        }

        # Initialize pydantic_ai agent for validation
        # self.validation_agent = Agent(
        #     "openai:gpt-3.5-turbo",  # You can change this to your preferred model
        #     result_type=FileValidationResult,
        #     system_prompt="You are a file validation assistant. Analyze the file information and determine if it's a valid PDF, DOCX, or TXT file."
        # )

    def _check_file_extension(self, filename: str) -> Optional[SupportedFileType]:
        """Check file extension and return corresponding file type."""
        ext = Path(filename).suffix.lower()

        extension_mapping = {
            ".pdf": SupportedFileType.PDF,
            ".docx": SupportedFileType.DOCX,
            ".txt": SupportedFileType.TXT,
        }

        return extension_mapping.get(ext)

    def _check_mime_type(self, file_path: str) -> Optional[str]:
        """Check MIME type of the file."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type

    def _validate_file_size(self, file_path: str) -> bool:
        """Check if file size is within allowed limits."""
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            return file_size <= max_size_bytes
        except OSError:
            return False

    def _read_file_signature(self, file_path: str) -> bytes:
        """Read the first few bytes of the file to check signature."""
        try:
            with open(file_path, 'rb') as f:
                return f.read(16)  # Read first 16 bytes
        except (OSError, IOError):
            return b''

    def _validate_pdf_signature(self, signature: bytes) -> bool:
        """Validate PDF file signature."""
        return signature.startswith(b'%PDF-')

    def _validate_docx_signature(self, signature: bytes) -> bool:
        """Validate DOCX file signature (ZIP-based)."""
        return signature.startswith(b'PK\x03\x04') or signature.startswith(b'PK\x05\x06')

    def _validate_txt_content(self, file_path: str) -> bool:
        """Validate TXT file by checking if it contains readable text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read first 1024 characters
                content = f.read(1024)
                # Check if content is printable text
                return all(c.isprintable() or c.isspace() for c in content)
        except (UnicodeDecodeError, OSError):
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read(1024)
                    return all(ord(c) < 128 for c in content)  # ASCII check
            except (UnicodeDecodeError, OSError):
                return False

    def validate_file(self, file_path: str) -> FileValidationResult:
        """
        Validate a single file and return validation result.

        Args:
            file_path: Path to the file to validate

        Returns:
            FileValidationResult with validation details
        """
        filename = file_path

        # Check if file exists
        if not os.path.exists(file_path):
            return FileValidationResult(
                filename=filename,
                file_type=None,
                is_valid=False,
                file_size=0,
                error_message="File does not exist"
            )

        # Get file size
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            return FileValidationResult(
                filename=filename,
                file_type=None,
                is_valid=False,
                file_size=0,
                error_message="Cannot access file"
            )

        # Check file size
        if not self._validate_file_size(file_path):
            return FileValidationResult(
                filename=filename,
                file_type=None,
                is_valid=False,
                file_size=file_size,
                error_message=f"File size exceeds {self.config.max_file_size_mb}MB limit"
            )

        # Check extension
        file_type = self._check_file_extension(filename)
        if not file_type:
            return FileValidationResult(
                filename=filename,
                file_type=None,
                is_valid=False,
                file_size=file_size,
                error_message="Unsupported file extension"
            )

        # Check MIME type
        mime_type = self._check_mime_type(file_path)

        # Read file signature
        signature = self._read_file_signature(file_path)

        # Validate based on file type
        is_valid = False
        error_message = None

        if file_type == SupportedFileType.PDF:
            is_valid = self._validate_pdf_signature(signature)
            if not is_valid:
                error_message = "Invalid PDF file signature"

        elif file_type == SupportedFileType.DOCX:
            is_valid = self._validate_docx_signature(signature)
            if not is_valid:
                error_message = "Invalid DOCX file signature"

        elif file_type == SupportedFileType.TXT:
            is_valid = self._validate_txt_content(file_path)
            if not is_valid:
                error_message = "File does not contain valid text content"

        return FileValidationResult(
            filename=filename,
            file_type=file_type,
            is_valid=is_valid,
            file_size=file_size,
            mime_type=mime_type,
            error_message=error_message
        )

    def validate_files(self, file_paths: List[str]) -> List[FileValidationResult]:
        """
        Validate multiple files and return validation results.

        Args:
            file_paths: List of file paths to validate

        Returns:
            List of FileValidationResult objects
        """
        return [self.validate_file(file_path) for file_path in file_paths]
