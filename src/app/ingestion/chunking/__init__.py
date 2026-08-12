"""
Markdown chunking and semantic processing.
"""

from .chunker_factory import (
    get_chunker,
    chunk_markdown,
    chunk_text,  # Deprecated alias for backward compatibility
    ChunkerType,
    LARGE_DOCUMENT_THRESHOLD_CHARS,
)

__all__ = [
    'get_chunker',
    'chunk_markdown',
    'chunk_text',  # Deprecated
    'ChunkerType',
    'LARGE_DOCUMENT_THRESHOLD_CHARS',
]
