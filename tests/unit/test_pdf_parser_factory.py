"""
Unit tests for ingestion/processors/pdf_parser_factory.py.

Covers:
- create_pdf_parser with "ollama" backend
- create_pdf_parser with "gemini-docling" backend
- Missing GOOGLE_API_KEY for gemini-docling
- Unknown backend string
"""
import pytest
from unittest.mock import MagicMock, patch

from ingestion.processors.pdf_parser_factory import create_pdf_parser


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_vlm_model = "qwen3.5:0.8b"
    settings.google_api_key = "test-api-key"
    settings.gemini_model = "gemini-2.5-flash"
    return settings


class TestCreatePdfParser:
    def test_ollama_backend(self, mock_settings):
        with patch('ingestion.processors.pdf_parser_factory.OllamaPDFParser', create=True):
            with patch('ingestion.processors.ollama_pdf_parser.OllamaPDFParser') as mock_cls:
                mock_cls.return_value = MagicMock()
                parser = create_pdf_parser("ollama", mock_settings)
                assert parser is not None

    def test_gemini_docling_backend(self, mock_settings):
        with patch('ingestion.processors.gemini_docling_parser.GeminiDoclingParser') as mock_cls:
            mock_cls.return_value = MagicMock()
            parser = create_pdf_parser("gemini-docling", mock_settings)
            assert parser is not None

    def test_gemini_docling_missing_api_key(self):
        settings = MagicMock()
        settings.google_api_key = None
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
            create_pdf_parser("gemini-docling", settings)

    def test_unknown_backend_raises(self, mock_settings):
        with pytest.raises(ValueError, match="Unknown pdf_parser_backend"):
            create_pdf_parser("unknown-backend", mock_settings)

    def test_ollama_passes_correct_params(self, mock_settings):
        with patch('ingestion.processors.ollama_pdf_parser.OllamaPDFParser') as mock_cls:
            mock_cls.return_value = MagicMock()
            create_pdf_parser("ollama", mock_settings)
            mock_cls.assert_called_once_with(
                ollama_base_url="http://localhost:11434",
                vlm_model="qwen3.5:0.8b",
            )

    def test_gemini_passes_correct_params(self, mock_settings):
        with patch('ingestion.processors.gemini_docling_parser.GeminiDoclingParser') as mock_cls:
            mock_cls.return_value = MagicMock()
            create_pdf_parser("gemini-docling", mock_settings)
            mock_cls.assert_called_once_with(
                api_key="test-api-key",
                gemini_model="gemini-2.5-flash",
            )
