"""
TXT document processor implementation.
Handles extraction of text from plain text files.
"""

from typing import List, Tuple
from pathlib import Path

from .base_processor import DocumentProcessor


class TXTProcessor(DocumentProcessor):
    """
    Concrete implementation for processing plain text files.

    Simplest processor - just reads the file as text.
    Treats entire file as page 1.
    """

    def get_supported_extensions(self) -> List[str]:
        """TXT and Markdown files support."""
        return ['.txt', '.TXT', '.md', '.MD', '.markdown']

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid text file.

        Args:
            file_path: Path to the text file

        Returns:
            True if file exists and is readable
        """
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            print(f"File does not exist: {file_path}")
            return False

        # Check if it's a file (not directory)
        if not path.is_file():
            print(f"Path is not a file: {file_path}")
            return False

        # Try to read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try reading first few bytes to verify it's readable
                f.read(100)
            return True
        except Exception as e:
            print(f"Error validating text file {file_path}: {e}")
            return False

    def extract_text(self, file_path: str) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Extract text from plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            Tuple of (full_text, page_mapping)
            where page_mapping treats entire file as page 1

        Raises:
            ValueError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # For TXT files, create a simple page mapping (assume one page)
            if text:
                page_mapping = [(0, len(text) - 1, 1)]
            else:
                page_mapping = []

            return text, page_mapping

        except Exception as e:
            raise ValueError(f"Failed to extract text from file {file_path}: {e}")
