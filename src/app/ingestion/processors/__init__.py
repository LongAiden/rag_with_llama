"""
Document processor implementations using Abstract Method pattern.
"""

from .base_processor import DocumentProcessor
from .docx_processor import DOCXProcessor
from .txt_processor import TXTProcessor
from app.ingestion.processors.processor_factory import (
    ProcessorRegistry,
    get_processor_for_file,
    get_registry
)
from .pdf_parser_base import PDFParserBase
from .gemini_docling_parser import GeminiDoclingParser
from .ollama_pdf_parser import OllamaPDFParser
from .pdf_parser_factory import create_pdf_parser

__all__ = [
    'DocumentProcessor',
    'DOCXProcessor',
    'TXTProcessor',
    'ProcessorRegistry',
    'get_processor_for_file',
    'get_registry',
    'PDFParserBase',
    'GeminiDoclingParser',
    'OllamaPDFParser',
    'create_pdf_parser',
]
