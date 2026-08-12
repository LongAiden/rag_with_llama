"""
Chunking Strategy Factory
Provides multiple chunking strategies with a unified interface.

Input: Markdown text (converted from PDF via PDFToMarkdownConverter).

Strategies:
- TokenChunker: Fastest, simple token-based chunking
- RecursiveChunker: Balanced, respects text boundaries (DEFAULT for large docs)
- MarkdownChunker: Structure-aware, preserves markdown hierarchy (DEFAULT for markdown input)
- SemanticChunker: Highest quality, AI-powered (slowest, use for small docs)

Adaptive Selection:
- For documents > 100KB: Uses RecursiveChunker (fast, reliable)
- For documents < 100KB: Uses configured chunker (semantic if specified)
"""

from enum import Enum
from typing import List, Optional
import os
import re
import logfire


class ChunkerType(Enum):
    """Available chunking strategies."""
    TOKEN = "token"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


# Size threshold for adaptive chunking (100KB)
LARGE_DOCUMENT_THRESHOLD_CHARS = 100_000

# Global cache for chunkers (especially important for SemanticChunker)
_CHUNKER_CACHE = {}


def get_chunker(
    chunker_type: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.5,
    embedding_model: Optional[str] = None,
    text_length: Optional[int] = None
):
    """
    Factory function to create the appropriate chunker.

    The expected input text is **Markdown** (produced by PDFToMarkdownConverter).
    Choose a chunker strategy that best fits your quality / speed trade-off.

    Args:
        chunker_type: Type of chunker ("token", "recursive", "markdown", "semantic")
                     If None, uses CHUNKER_TYPE env var or defaults to "markdown"
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of overlapping tokens/sentences
        similarity_threshold: For semantic chunker only (0-1)
        embedding_model: For semantic chunker only
        text_length: Optional text length for adaptive chunker selection.
                    If provided and > LARGE_DOCUMENT_THRESHOLD_CHARS,
                    forces RecursiveChunker for performance.

    Returns:
        Configured chunker instance

    Examples:
        >>> # Use default (markdown-aware)
        >>> chunker = get_chunker(chunk_size=512)

        >>> # Use markdown chunker explicitly
        >>> chunker = get_chunker("markdown", chunk_size=512)

        >>> # Use token chunker explicitly
        >>> chunker = get_chunker("token", chunk_size=512)

        >>> # Use semantic chunker for small documents
        >>> chunker = get_chunker("semantic", chunk_size=512, similarity_threshold=0.3)

        >>> # Adaptive selection based on document size
        >>> chunker = get_chunker("semantic", text_length=500000)  # Will use recursive
    """
    from chonkie import TokenChunker, RecursiveChunker, SemanticChunker

    # Determine chunker type from env or parameter
    if chunker_type is None:
        chunker_type = os.getenv("CHUNKER_TYPE", "markdown").lower()
    else:
        chunker_type = chunker_type.lower()

    # Validate parameters
    if chunk_size < chunk_overlap:
        error_msg = f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})"
        logfire.error(error_msg, chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap)
        raise ValueError(error_msg)

    # Adaptive selection: Force RecursiveChunker for large documents
    # This prevents SemanticChunker from being too slow on large markdown documents
    if text_length is not None and text_length > LARGE_DOCUMENT_THRESHOLD_CHARS:
        if chunker_type == ChunkerType.SEMANTIC.value:
            print(f"[Adaptive] Document is large ({text_length:,} chars > {LARGE_DOCUMENT_THRESHOLD_CHARS:,}). "
                  f"Switching from SemanticChunker to RecursiveChunker for performance.")
            chunker_type = ChunkerType.RECURSIVE.value

    # Create appropriate chunker
    if chunker_type == ChunkerType.TOKEN.value:
        cache_key = f"token_{chunk_size}_{chunk_overlap}"
        if cache_key not in _CHUNKER_CACHE:
            _CHUNKER_CACHE[cache_key] = TokenChunker(
                tokenizer="character",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        return _CHUNKER_CACHE[cache_key]

    elif chunker_type == ChunkerType.RECURSIVE.value:
        # Uses hierarchical text splitting with generic delimiters
        cache_key = f"recursive_{chunk_size}"
        if cache_key not in _CHUNKER_CACHE:
            _CHUNKER_CACHE[cache_key] = RecursiveChunker(
                tokenizer="character",
                chunk_size=chunk_size
            )
        return _CHUNKER_CACHE[cache_key]

    elif chunker_type == ChunkerType.MARKDOWN.value:
        # Markdown-aware chunker: uses chonkie's built-in markdown recipe
        # which splits on headings, lists, code blocks, etc.
        # This is the recommended chunker when input is markdown from PDFToMarkdownConverter.
        cache_key = f"markdown_{chunk_size}"
        if cache_key not in _CHUNKER_CACHE:
            _CHUNKER_CACHE[cache_key] = RecursiveChunker.from_recipe(
                "markdown",
                lang="en",
                chunk_size=chunk_size
            )
        return _CHUNKER_CACHE[cache_key]

    elif chunker_type == ChunkerType.SEMANTIC.value:
        # Semantic chunker is HEAVY (loads models). Must cache.
        embedding_model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        cache_key = f"semantic_{chunk_size}_{similarity_threshold}_{embedding_model_name}"

        if cache_key not in _CHUNKER_CACHE:
            print(
                f"Initializing SemanticChunker (loading model: {embedding_model_name})...")
            _CHUNKER_CACHE[cache_key] = SemanticChunker(
                chunk_size=chunk_size,
                threshold=similarity_threshold,
                embedding_model=embedding_model_name
            )
        return _CHUNKER_CACHE[cache_key]

    else:
        raise ValueError(
            f"Invalid chunker_type: {chunker_type}. "
            f"Must be one of: {[e.value for e in ChunkerType]}"
        )


def chunk_markdown(
    markdown_text: str,
    chunker_type: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.5,
    embedding_model: Optional[str] = None
) -> List:
    """
    Convenience function to chunk markdown text with the specified strategy.

    Args:
        markdown_text: Markdown content to chunk (from PDFToMarkdownConverter)
        chunker_type: Type of chunker (None = use default/env var, defaults to "markdown")
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap size
        similarity_threshold: For semantic chunker
        embedding_model: For semantic chunker

    Returns:
        List of chunks

    Examples:
        >>> chunks = chunk_markdown(md_text)  # Uses default (markdown-aware)
        >>> chunks = chunk_markdown(md_text, chunker_type="token")  # Fast
        >>> chunks = chunk_markdown(md_text, chunker_type="semantic")  # Quality
    """
    chunker = get_chunker(
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model
    )

    chunks = chunker.chunk(markdown_text)

    # Build per-page content map from [Page N] markers (PDF-converted markdown)
    page_markers = list(re.finditer(r'\[Page (\d+)\]', markdown_text))

    if page_markers:
        # Map each page number → its content slice
        page_content_map: dict = {}
        for i, marker in enumerate(page_markers):
            page_num = int(marker.group(1))
            start = marker.end()
            end = page_markers[i + 1].start() if i + 1 < len(page_markers) else len(markdown_text)
            page_content_map[page_num] = markdown_text[start:end].strip()

        # Assign each chunk only the content of its own page (not the full document)
        first_page = int(page_markers[0].group(1))
        for chunk in chunks:
            start_idx = getattr(chunk, 'start_index', None)
            if start_idx is None:
                start_idx = 0
            chunk_page = first_page
            for marker in page_markers:
                if marker.start() > start_idx:
                    break
                chunk_page = int(marker.group(1))
            setattr(chunk, 'full_content', page_content_map.get(chunk_page, ''))
            setattr(chunk, 'page', chunk_page)
    else:
        # No page markers (plain markdown / non-PDF): store the full text as fallback
        for chunk in chunks:
            setattr(chunk, 'full_content', markdown_text)
            setattr(chunk, 'page', 'all')

    return chunks


# Backward-compatible alias
def chunk_text(
    text: str,
    chunker_type: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.5,
    embedding_model: Optional[str] = None
) -> List:
    """Deprecated: Use chunk_markdown() instead. Kept for backward compatibility."""
    import warnings
    warnings.warn(
        "chunk_text() is deprecated. Use chunk_markdown() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return chunk_markdown(
        markdown_text=text,
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model
    )
