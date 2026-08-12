from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class PDFParserBase(ABC):

    @abstractmethod
    def parse_pdf(self, path: str | Path, output_path: Optional[str | Path] = None) -> str:
        """Parse a PDF and return markdown. Optionally write to output_path."""

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return a short identifier, e.g. 'ollama' or 'gemini-docling'."""
