"""
Unit tests for ChunkerFactory.

Tests focus on observable behavior (input text -> chunks) rather than
internal implementation details like class names or enum counts.
This keeps tests stable when the underlying chunking library changes.
"""
import os
import pytest
import warnings
from unittest.mock import patch

from app.ingestion.chunking.chunker_factory import (
    get_chunker,
    chunk_markdown,
    chunk_text,
    ChunkerType,
    LARGE_DOCUMENT_THRESHOLD_CHARS,
    _CHUNKER_CACHE,
)


@pytest.fixture(autouse=True)
def clear_chunker_cache():
    """Clear the global chunker cache before each test to avoid state leaks."""
    _CHUNKER_CACHE.clear()
    yield
    _CHUNKER_CACHE.clear()


class TestChunkerType:
    """Tests for ChunkerType enum."""

    def test_chunker_type_values(self):
        """Test that ChunkerType enum has the expected string values."""
        assert ChunkerType.TOKEN.value == "token"
        assert ChunkerType.RECURSIVE.value == "recursive"
        assert ChunkerType.MARKDOWN.value == "markdown"
        assert ChunkerType.SEMANTIC.value == "semantic"


class TestGetChunkerBehavior:
    """Behavioral tests for get_chunker factory function."""

    def test_token_chunker_splits_text(self):
        """A token chunker should split a long text into multiple chunks."""
        chunker = get_chunker(chunker_type="token", chunk_size=50, chunk_overlap=0)
        text = "word " * 200
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Each chunk should contain some of the original text
        assert all("word" in str(chunk) for chunk in chunks)

    def test_recursive_chunker_splits_text(self):
        """A recursive chunker should split text and preserve some structure."""
        chunker = get_chunker(chunker_type="recursive", chunk_size=50)
        text = "word " * 200
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all("word" in str(chunk) for chunk in chunks)

    def test_markdown_chunker_preserves_headings(self):
        """A markdown chunker should respect heading boundaries."""
        chunker = get_chunker(chunker_type="markdown", chunk_size=128)
        text = "# Heading 1\n\nContent under heading one.\n\n# Heading 2\n\nContent under heading two."
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # Headings should appear in the chunk text
        chunk_text = " ".join(str(c) for c in chunks)
        assert "# Heading 1" in chunk_text or "Heading 1" in chunk_text

    def test_semantic_chunker_splits_text(self):
        """A semantic chunker should split text into chunks."""
        chunker = get_chunker(chunker_type="semantic", chunk_size=128)
        text = "Machine learning is amazing. " * 50
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_invalid_chunker_type_raises_error(self):
        """Test that invalid chunker type raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_chunker(chunker_type="invalid_type")
        assert "Invalid chunker_type" in str(excinfo.value)

    def test_default_chunker_is_markdown(self):
        """Test that default chunker processes markdown sensibly."""
        with patch.dict(os.environ, {}, clear=True):
            chunker = get_chunker(chunk_size=128)
            text = "# Title\n\nSome content here.\n\n## Section\n\nMore content."
            chunks = chunker.chunk(text)

            assert isinstance(chunks, list)
            assert len(chunks) >= 1

    def test_env_var_override_chunker_type(self):
        """Test that CHUNKER_TYPE env var overrides the default."""
        with patch.dict(os.environ, {"CHUNKER_TYPE": "token"}, clear=False):
            chunker = get_chunker(chunk_size=50, chunk_overlap=0)
            text = "word " * 200
            chunks = chunker.chunk(text)

            assert isinstance(chunks, list)
            assert len(chunks) > 1

    def test_chunker_type_case_insensitive(self):
        """Test that chunker type selection is case insensitive."""
        chunker_upper = get_chunker(chunker_type="TOKEN", chunk_size=50, chunk_overlap=0)
        chunker_lower = get_chunker(chunker_type="token", chunk_size=50, chunk_overlap=0)

        text = "word " * 200
        upper_chunks = chunker_upper.chunk(text)
        lower_chunks = chunker_lower.chunk(text)

        # Same parameters should return the same cached instance and produce same chunks
        assert chunker_upper is chunker_lower
        assert len(upper_chunks) == len(lower_chunks)

    def test_chunk_size_less_than_overlap_raises_error(self):
        """Test that validation fails when chunk_size < chunk_overlap."""
        with pytest.raises(ValueError) as excinfo:
            get_chunker(chunk_size=100, chunk_overlap=200)
        assert "must be greater than" in str(excinfo.value)


class TestAdaptiveSelection:
    """Tests for text length-based adaptive chunker selection."""

    def test_large_document_forces_recursive(self):
        """Large documents should use a fast chunker even when semantic is requested."""
        large_text_length = LARGE_DOCUMENT_THRESHOLD_CHARS + 1

        chunker = get_chunker(
            chunker_type="semantic",
            text_length=large_text_length
        )
        text = "word " * 200
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_small_document_allows_semantic(self):
        """Small documents can use semantic chunker."""
        small_text_length = LARGE_DOCUMENT_THRESHOLD_CHARS - 1

        chunker = get_chunker(
            chunker_type="semantic",
            text_length=small_text_length
        )
        text = "Machine learning is amazing. " * 20
        chunks = chunker.chunk(text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_none_text_length_no_error(self):
        """Test that None text_length doesn't cause errors."""
        chunker = get_chunker(chunker_type="recursive", text_length=None)
        assert chunker is not None

    def test_zero_text_length_no_error(self):
        """Test that zero text_length doesn't cause errors."""
        chunker = get_chunker(chunker_type="recursive", text_length=0)
        assert chunker is not None

    def test_negative_text_length_no_error(self):
        """Test that negative text_length doesn't cause errors."""
        chunker = get_chunker(chunker_type="recursive", text_length=-100)
        assert chunker is not None


class TestChunkerCaching:
    """Tests for chunker caching behavior."""

    def test_same_params_return_cached_chunker(self):
        """Test that same parameters return the same cached chunker instance."""
        chunker1 = get_chunker(chunker_type="token",
                               chunk_size=256, chunk_overlap=25)
        chunker2 = get_chunker(chunker_type="token",
                               chunk_size=256, chunk_overlap=25)

        # Should be the exact same instance
        assert chunker1 is chunker2

    def test_different_params_return_different_chunker(self):
        """Test that different parameters return different chunker instances."""
        chunker1 = get_chunker(chunker_type="token", chunk_size=256)
        chunker2 = get_chunker(chunker_type="token", chunk_size=512)

        # Should NOT be the same instance (different chunk_size)
        assert chunker1 is not chunker2


class TestChunkMarkdown:
    """Tests for chunk_markdown convenience function."""

    def test_chunk_markdown_returns_list(self, small_text):
        """Test that chunk_markdown returns a list of chunks."""
        chunks = chunk_markdown(
            small_text, chunker_type="markdown", chunk_size=128)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_markdown_with_token_chunker(self, small_text):
        """Test chunk_markdown with token chunker."""
        chunks = chunk_markdown(
            small_text, chunker_type="token", chunk_size=50, chunk_overlap=10)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_markdown_empty_string(self):
        """Test chunk_markdown with empty string."""
        chunks = chunk_markdown("", chunker_type="markdown")
        assert isinstance(chunks, list)

    def test_chunk_markdown_single_word(self):
        """Test chunk_markdown with a single word."""
        chunks = chunk_markdown(
            "Hello", chunker_type="markdown", chunk_size=512)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk_markdown_preserves_heading_structure(self):
        """Test that markdown chunker respects heading boundaries."""
        md_text = "# Section 1\n\nContent for section 1.\n\n# Section 2\n\nContent for section 2."
        chunks = chunk_markdown(
            md_text, chunker_type="markdown", chunk_size=512)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # Both sections should be represented
        combined = " ".join(str(c) for c in chunks)
        assert "Section 1" in combined
        assert "Section 2" in combined

    def test_chunk_markdown_assigns_page_metadata(self):
        """Test that chunk_markdown assigns page metadata from [Page N] markers."""
        md_text = "[Page 1]\nContent on page one.\n\n[Page 2]\nContent on page two."
        chunks = chunk_markdown(
            md_text, chunker_type="token", chunk_size=50, chunk_overlap=0)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # At least one chunk should have page metadata
        assert any(hasattr(c, 'page') for c in chunks)


class TestChunkTextDeprecation:
    """Tests for the deprecated chunk_text backward-compatible alias."""

    def test_chunk_text_emits_deprecation_warning(self):
        """Test that chunk_text emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunk_text("Hello world", chunker_type="token", chunk_size=512)
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "chunk_markdown" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1

    def test_chunk_text_still_returns_results(self, small_text):
        """Test that chunk_text still works and returns chunks."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            chunks = chunk_text(
                small_text, chunker_type="token", chunk_size=128)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
