from __future__ import annotations
from typing import TYPE_CHECKING

from app.ingestion.processors.pdf_parser_base import PDFParserBase

if TYPE_CHECKING:
    from app.config.app_config import AppSettings


def create_pdf_parser(backend: str, settings: "AppSettings") -> PDFParserBase:
    """
    Factory function. Returns the appropriate PDFParserBase implementation.

    Args:
        backend: "ollama" or "gemini-docling"
        settings: AppSettings instance (provides credentials/URLs)

    Raises:
        ValueError: unknown backend string or missing credentials
    """
    tuning = dict(
        docling_num_threads=settings.docling_num_threads,
        page_batch_size=settings.docling_page_batch_size,
        vlm_concurrency=settings.vlm_concurrency,
        vlm_tables=settings.vlm_tables,
    )

    if backend == "ollama":
        from app.ingestion.processors.ollama_pdf_parser import OllamaPDFParser
        return OllamaPDFParser(
            ollama_base_url=settings.ollama_base_url,
            vlm_model=settings.ollama_vlm_model,
            think=settings.ollama_vlm_think,
            **tuning,
        )

    if backend == "gemini-docling":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for gemini-docling backend")
        from app.ingestion.processors.gemini_docling_parser import GeminiDoclingParser
        return GeminiDoclingParser(
            api_key=settings.google_api_key,
            gemini_model=settings.gemini_model,
            **tuning,
        )

    raise ValueError(f"Unknown pdf_parser_backend: {backend!r}. Choose 'ollama' or 'gemini-docling'.")
