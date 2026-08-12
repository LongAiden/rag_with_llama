"""
Unit tests for EmbeddingGenerator.

All tests mock the underlying SentenceTransformer model so they are fast,
deterministic, and independent of the actual embedding model or its weights.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.ingestion.embedding.generator import EmbeddingGenerator


@pytest.fixture
def mock_sentence_transformer():
    """Fixture that patches SentenceTransformer with a deterministic mock."""
    def _make_mock(embedding_dim=384):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = embedding_dim

        # Deterministic encode: each unique text gets a unique but stable vector.
        # We use a simple hash-based vector for unit tests.
        def mock_encode(texts, **kwargs):
            single = isinstance(texts, str)
            if single:
                texts = [texts]

            result = []
            for text in texts:
                # Deterministic pseudo-random vector based on text content
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                vec = rng.random(embedding_dim).astype(np.float32)
                # Normalize to unit length
                vec = vec / np.linalg.norm(vec)
                result.append(vec)

            if single:
                return result[0]
            return np.array(result)

        mock_model.encode.side_effect = mock_encode
        return mock_model

    return _make_mock


@pytest.fixture
def generator(mock_sentence_transformer):
    """Create an EmbeddingGenerator with a mocked model."""
    with patch('app.ingestion.embedding.generator.SentenceTransformer') as mock_cls:
        mock_cls.return_value = mock_sentence_transformer(embedding_dim=384)
        gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        yield gen


class TestEmbeddingGeneratorInitialization:
    """Tests for EmbeddingGenerator initialization."""

    def test_model_name_is_stored(self, generator):
        """Test that the model name is stored."""
        assert generator.model_name == 'all-MiniLM-L6-v2'

    def test_embedding_dimension_is_set(self, generator):
        """Test that embedding dimension is correctly set from the model."""
        assert generator.embedding_dim == 384

    def test_custom_model_initialization(self, mock_sentence_transformer):
        """Test initialization with a different model name."""
        with patch('app.ingestion.embedding.generator.SentenceTransformer') as mock_cls:
            mock_cls.return_value = mock_sentence_transformer(embedding_dim=512)
            gen = EmbeddingGenerator(model_name='custom-model')
            assert gen.embedding_dim == 512


class TestEmbedText:
    """Tests for embed_text method."""

    def test_embed_single_text_returns_list(self, generator):
        """Test embedding a single text returns a list."""
        embedding = generator.embed_text("This is a test sentence.")

        assert isinstance(embedding, list)
        assert len(embedding) == generator.embedding_dim

    def test_embed_text_returns_floats(self, generator):
        """Test that embedding values are floats."""
        embedding = generator.embed_text("Test text")

        assert all(isinstance(x, float) for x in embedding)

    def test_embed_same_text_produces_same_embedding(self, generator):
        """Test that same text produces same embedding."""
        text = "Consistent embedding test"

        embedding1 = generator.embed_text(text)
        embedding2 = generator.embed_text(text)

        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_embed_different_texts_produce_different_embeddings(self, generator):
        """Test that different texts produce different embeddings."""
        text1 = "Machine learning is amazing"
        text2 = "The weather is sunny today"

        embedding1 = generator.embed_text(text1)
        embedding2 = generator.embed_text(text2)

        assert not np.allclose(embedding1, embedding2)

    def test_embed_text_calls_model_encode(self, generator):
        """Test that embed_text delegates to the underlying model."""
        generator.embed_text("Hello world")
        generator.model.encode.assert_called_with("Hello world")


class TestEmbedBatch:
    """Tests for embed_batch method."""

    def test_embed_batch_returns_list(self, generator, sample_chunk_texts):
        """Test that embed_batch returns a list of embeddings."""
        embeddings = generator.embed_batch(sample_chunk_texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_chunk_texts)

    def test_embed_batch_correct_dimensions(self, generator, sample_chunk_texts):
        """Test that each embedding in batch has correct dimensions."""
        embeddings = generator.embed_batch(sample_chunk_texts)

        for embedding in embeddings:
            assert len(embedding) == generator.embedding_dim

    def test_embed_batch_single_text(self, generator):
        """Test embed_batch with single text in list."""
        texts = ["Single text"]

        embeddings = generator.embed_batch(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == generator.embedding_dim

    def test_embed_batch_empty_list(self, generator):
        """Test embed_batch with empty list."""
        embeddings = generator.embed_batch([])

        assert embeddings == []

    def test_embed_batch_consistent_with_single(self, generator):
        """Test that batch and individual embeddings are consistent."""
        texts = ["First text", "Second text", "Third text"]

        batch_embeddings = generator.embed_batch(texts)
        individual_embeddings = [generator.embed_text(text) for text in texts]

        for batch_emb, ind_emb in zip(batch_embeddings, individual_embeddings):
            np.testing.assert_array_almost_equal(batch_emb, ind_emb)


class TestEmptyAndEdgeCases:
    """Tests for empty and edge case inputs."""

    def test_embed_empty_string(self, generator):
        """Test embedding empty string."""
        embedding = generator.embed_text("")

        assert embedding is not None
        assert len(embedding) == generator.embedding_dim

    def test_embed_whitespace_only(self, generator):
        """Test embedding whitespace-only string."""
        embedding = generator.embed_text("   ")

        assert embedding is not None
        assert len(embedding) == generator.embedding_dim

    def test_embed_very_long_text(self, generator):
        """Test embedding very long text."""
        long_text = "word " * 1000  # ~5000 characters

        embedding = generator.embed_text(long_text)

        assert embedding is not None
        assert len(embedding) == generator.embedding_dim

    def test_embed_special_characters(self, generator):
        """Test embedding text with special characters."""
        special_text = "α β γ δ ε → ← ∑ ∫ © ® ™ € £ ¥"

        embedding = generator.embed_text(special_text)

        assert embedding is not None
        assert len(embedding) == generator.embedding_dim

    def test_embed_unicode_text(self, generator):
        """Test embedding Unicode text."""
        unicode_text = "日本語テスト 中文测试 한국어 테스트"

        embedding = generator.embed_text(unicode_text)

        assert embedding is not None
        assert len(embedding) == generator.embedding_dim

    def test_embed_batch_with_none_replaced_by_empty_string(self, generator):
        """Test that None values in batch are replaced with empty strings."""
        embeddings = generator.embed_batch([None, "valid text"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == generator.embedding_dim
        assert len(embeddings[1]) == generator.embedding_dim


class TestEmbeddingNormalization:
    """Tests for embedding normalization properties using the mock."""

    def test_embeddings_are_normalized(self, generator):
        """Test that mocked embeddings are normalized (L2 norm ≈ 1)."""
        text = "Test text for normalization"

        embedding = generator.embed_text(text)

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.01), f"L2 norm is {norm}, expected ~1.0"
