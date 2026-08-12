"""
Document ingestion module.
Handles document processing, chunking, embedding, and validation.
"""

from app.ingestion.processors import (
    DocumentProcessor,
    DOCXProcessor,
    TXTProcessor,
    get_processor_for_file,
    get_registry
)
from app.ingestion.processors.page_utils import (
    get_supported_file_types,
    list_available_processors
)

__all__ = [
    # Processors
    'DocumentProcessor',
    'DOCXProcessor',
    'TXTProcessor',
    'get_processor_for_file',
    'get_registry',
    'get_supported_file_types',
    'list_available_processors',
]
