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

from app.ingestion.processors.pdf_parser_factory import create_pdf_parser


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_vlm_model = "qwen3.5:0.8b"
    settings.google_api_key = "test-api-key"
    settings.gemini_model = "gemini-2.5-flash"
    settings.docling_num_threads = 2
    settings.docling_page_batch_size = 50
    settings.vlm_concurrency = 1
    settings.vlm_tables = False
    settings.ollama_vlm_think = False
    settings.ollama_vlm_temperature = 0.0
    settings.ollama_vlm_num_predict = 384
    settings.vlm_min_image_short_px = 64
    return settings


class TestCreatePdfParser:
    def test_ollama_backend(self, mock_settings):
        with patch('app.ingestion.processors.pdf_parser_factory.OllamaPDFParser', create=True):
            with patch('app.ingestion.processors.ollama_pdf_parser.OllamaPDFParser') as mock_cls:
                mock_cls.return_value = MagicMock()
                parser = create_pdf_parser("ollama", mock_settings)
                assert parser is not None

    def test_gemini_docling_backend(self, mock_settings):
        with patch('app.ingestion.processors.gemini_docling_parser.GeminiDoclingParser') as mock_cls:
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
        with patch('app.ingestion.processors.ollama_pdf_parser.OllamaPDFParser') as mock_cls:
            mock_cls.return_value = MagicMock()
            create_pdf_parser("ollama", mock_settings)
            mock_cls.assert_called_once_with(
                ollama_base_url="http://localhost:11434",
                vlm_model="qwen3.5:0.8b",
                think=False,
                temperature=0.0,
                num_predict=384,
                docling_num_threads=2,
                page_batch_size=50,
                vlm_concurrency=1,
                vlm_tables=False,
                min_image_short_px=64,
            )

    def test_gemini_passes_correct_params(self, mock_settings):
        with patch('app.ingestion.processors.gemini_docling_parser.GeminiDoclingParser') as mock_cls:
            mock_cls.return_value = MagicMock()
            create_pdf_parser("gemini-docling", mock_settings)
            mock_cls.assert_called_once_with(
                api_key="test-api-key",
                gemini_model="gemini-2.5-flash",
                docling_num_threads=2,
                page_batch_size=50,
                vlm_concurrency=1,
                vlm_tables=False,
                min_image_short_px=64,
            )


class TestTuningIsWiredThrough:
    """The parse tuning levers must reach the parser, and must default to the
    values that were hardcoded before they were made configurable — otherwise
    exposing them silently changes behaviour and contaminates the baseline the
    performance investigation depends on."""

    def test_settings_reach_the_ollama_parser(self, mock_settings):
        mock_settings.docling_num_threads = 4
        mock_settings.docling_page_batch_size = 25
        mock_settings.vlm_concurrency = 8

        parser = create_pdf_parser("ollama", mock_settings)

        assert parser._docling_num_threads == 4
        assert parser._page_batch_size == 25
        assert parser._vlm_concurrency == 8

    def test_settings_reach_the_gemini_parser(self, mock_settings):
        mock_settings.docling_num_threads = 6
        mock_settings.docling_page_batch_size = 100
        mock_settings.vlm_concurrency = 3

        parser = create_pdf_parser("gemini-docling", mock_settings)

        assert parser._docling_num_threads == 6
        assert parser._page_batch_size == 100
        assert parser._vlm_concurrency == 3

    def test_app_settings_defaults(self):
        from app.config.app_config import AppSettings

        settings = AppSettings()

        # Unchanged from the values that were hardcoded before they were exposed.
        assert settings.docling_num_threads == 2
        # 40, lowered from 50 after a 702-page PDF was OOM-killed: docling holds
        # a rendered image for every page in the batch until it is released, so
        # this is the parse's working set (~5.2MB/page at images_scale 2.0).
        assert settings.docling_page_batch_size == 40
        # Measured on this stack: concurrency above 1 is slower, not faster,
        # because Ollama is local and serializes on one GPU.
        assert settings.vlm_concurrency == 1
        # Reasoning is 40x slower and returns nothing for tables.
        assert settings.ollama_vlm_think is False
        # Tables belong to docling's TableFormer, not a 0.8B VLM.
        assert settings.vlm_tables is False

    def test_think_and_vlm_tables_can_be_re_enabled(self, mock_settings):
        mock_settings.ollama_vlm_think = True
        mock_settings.vlm_tables = True

        parser = create_pdf_parser("ollama", mock_settings)

        assert parser._think is True
        assert parser._vlm_tables is True
