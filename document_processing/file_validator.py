import os
from pathlib import Path
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class FileValidator:
    """File validator class for checking PDF, DOCX, and TXT files."""

    def __init__(self, config: FileValidationConfig = None):
        self.config = config or FileValidationConfig()

    def _check_file_extension(self, filename: str) -> Optional[SupportedFileType]:
        """Check file extension and return corresponding file type."""
        ext = Path(filename).suffix.lower()

        extension_mapping = {
            ".pdf": SupportedFileType.PDF,
            ".docx": SupportedFileType.DOCX,
            ".txt": SupportedFileType.TXT,
        }

        return extension_mapping.get(ext)

    def _validate_file_size(self, file_path: str) -> bool:
        """Check if file size is within allowed limits."""
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            return file_size <= max_size_bytes
        except OSError:
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

        # Simple validation - for PoC, just check if file is readable
        is_valid = True
        error_message = None

        try:
            # Basic file readability check
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read first byte
        except (OSError, IOError):
            is_valid = False
            error_message = "Cannot read file"

        return FileValidationResult(
            filename=filename,
            file_type=file_type,
            is_valid=is_valid,
            file_size=file_size,
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
