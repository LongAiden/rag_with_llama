"""
Unit tests for the LLM Provider abstraction.

Tests the LLMProvider interface and concrete provider implementations
(Gemini, Ollama) without making real external API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graph_processing.llm_provider import LLMProvider
from graph_processing.gemini_provider import GeminiLLMProvider
from graph_processing.ollama_provider import OllamaLLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self):
        self.generate_calls = []
        self.mock_response = "Mock response text"

    async def generate_content(self, prompt: str):
        self.generate_calls.append(prompt)
        response = MagicMock()
        response.text = self.mock_response
        return response

    def extract_text_from_response(self, response):
        return response.text

    def get_provider_name(self):
        return "mock"


class TestLLMProviderInterface:
    """Test the LLMProvider abstract interface."""

    def test_mock_provider_implements_interface(self):
        """Test that MockLLMProvider correctly implements LLMProvider."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)
        assert provider.get_provider_name() == "mock"

    @pytest.mark.asyncio
    async def test_mock_provider_generate_content(self):
        """Test that MockLLMProvider can generate content."""
        provider = MockLLMProvider()
        provider.mock_response = "Test response"

        response = await provider.generate_content("Test prompt")
        text = provider.extract_text_from_response(response)

        assert text == "Test response"
        assert len(provider.generate_calls) == 1
        assert provider.generate_calls[0] == "Test prompt"

    @pytest.mark.asyncio
    async def test_mock_provider_tracks_multiple_calls(self):
        """Test that MockLLMProvider tracks multiple calls."""
        provider = MockLLMProvider()

        await provider.generate_content("Prompt 1")
        await provider.generate_content("Prompt 2")
        await provider.generate_content("Prompt 3")

        assert len(provider.generate_calls) == 3
        assert provider.generate_calls == ["Prompt 1", "Prompt 2", "Prompt 3"]


class TestGeminiLLMProvider:
    """Test the GeminiLLMProvider implementation."""

    def test_provider_name(self):
        """Test that GeminiLLMProvider returns correct provider name."""
        with patch('graph_processing.gemini_provider.genai'):
            provider = GeminiLLMProvider(api_key="test-key", model_name="gemini-2.5-flash")
            assert provider.get_provider_name() == "gemini"

    def test_extract_text_from_response(self):
        """Test text extraction from Gemini response."""
        with patch('graph_processing.gemini_provider.genai'):
            provider = GeminiLLMProvider(api_key="test-key", model_name="gemini-2.5-flash")

            mock_response = MagicMock()
            mock_response.text = "Extracted text content"

            result = provider.extract_text_from_response(mock_response)
            assert result == "Extracted text content"

    @pytest.mark.asyncio
    async def test_generate_content_calls_model(self):
        """Test that generate_content calls the underlying model."""
        with patch('graph_processing.gemini_provider.genai') as mock_genai:
            # Setup mock
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Generated response"
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            provider = GeminiLLMProvider(api_key="test-key", model_name="gemini-2.5-flash")

            response = await provider.generate_content("Test prompt")

            assert response == mock_response
            mock_model.generate_content_async.assert_called_once_with("Test prompt")


class TestOllamaLLMProvider:
    """Test the OllamaLLMProvider implementation."""

    def test_provider_name(self):
        """Test that OllamaLLMProvider returns correct provider name."""
        provider = OllamaLLMProvider(base_url="http://localhost:11434")
        assert provider.get_provider_name() == "ollama"

    def test_base_url_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        provider = OllamaLLMProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    def test_extract_text_from_response(self):
        """Test text extraction from Ollama JSON response."""
        provider = OllamaLLMProvider(base_url="http://localhost:11434")

        response = {"response": "Generated text from Ollama"}
        result = provider.extract_text_from_response(response)

        assert result == "Generated text from Ollama"

    @pytest.mark.asyncio
    async def test_generate_content_posts_to_api(self):
        """Test that generate_content POSTs to the Ollama generate endpoint."""
        provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model_name="deepseek-r1:8b"
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Ollama output"})

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch('graph_processing.ollama_provider.httpx.AsyncClient', return_value=mock_client):
            response = await provider.generate_content("Test prompt")

        assert response == {"response": "Ollama output"}
        mock_client.post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:8b", "prompt": "Test prompt", "stream": False}
        )

    @pytest.mark.asyncio
    async def test_generate_content_retries_on_error(self):
        """Test that generate_content retries on HTTP error."""
        provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model_name="deepseek-r1:8b"
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Success after retry"})

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        # First call fails, second succeeds
        mock_client.post = AsyncMock(side_effect=[
            Exception("Connection failed"),
            mock_response
        ])

        with patch('graph_processing.ollama_provider.httpx.AsyncClient', return_value=mock_client):
            response = await provider.generate_content("Test prompt")

        assert response == {"response": "Success after retry"}
        assert mock_client.post.call_count == 2


class TestProviderAbstraction:
    """Test that all providers adhere to the same interface contract."""

    @pytest.mark.parametrize("provider", [
        MockLLMProvider(),
    ])
    def test_all_providers_have_required_methods(self, provider):
        """Test that providers implement the required interface."""
        assert hasattr(provider, 'generate_content')
        assert hasattr(provider, 'extract_text_from_response')
        assert hasattr(provider, 'get_provider_name')
        assert callable(provider.generate_content)
        assert callable(provider.extract_text_from_response)
        assert callable(provider.get_provider_name)
        assert isinstance(provider.get_provider_name(), str)

    def test_provider_name_uniqueness(self):
        """Test that each provider has a distinct name."""
        with patch('graph_processing.gemini_provider.genai'):
            gemini = GeminiLLMProvider(api_key="test-key")
        ollama = OllamaLLMProvider(base_url="http://localhost:11434")
        mock = MockLLMProvider()

        names = {gemini.get_provider_name(), ollama.get_provider_name(), mock.get_provider_name()}
        assert len(names) == 3
