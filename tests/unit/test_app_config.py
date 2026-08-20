"""
Unit tests for the refactored AppConfig with pydantic-settings.

Tests the configuration classes and their behavior.
"""

import os
from unittest.mock import patch

import pytest


class TestDatabaseConfig:
    """Test the DatabaseConfig class."""

    def test_default_values(self):
        """Test that DatabaseConfig has correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            from app.config.app_config import DatabaseConfig
            config = DatabaseConfig()

            assert config.host == 'localhost'
            assert config.port == '5432'
            assert config.dbname == 'rag_db'
            assert config.user == 'admin'
            assert config.password == 'admin'

    def test_to_dict(self):
        """Test that to_dict returns correct format."""
        with patch.dict(os.environ, {}, clear=True):
            from app.config.app_config import DatabaseConfig
            config = DatabaseConfig()
            result = config.to_dict()

            assert isinstance(result, dict)
            assert 'host' in result
            assert 'port' in result
            assert 'dbname' in result
            assert 'user' in result
            assert 'password' in result

    def test_loads_from_environment(self):
        """Test that DatabaseConfig loads from environment variables."""
        env_vars = {
            'POSTGRES_HOST': 'custom-host',
            'POSTGRES_PORT': '5433',
            'POSTGRES_DB': 'custom_db',
            'POSTGRES_USER': 'custom_user',
            'POSTGRES_PASSWORD': 'custom_pass'
        }
        with patch.dict(os.environ, env_vars, clear=True):
            from app.config.app_config import DatabaseConfig

            config = DatabaseConfig()

            assert config.host == 'custom-host'
            assert config.port == '5433'
            assert config.dbname == 'custom_db'
            assert config.user == 'custom_user'
            assert config.password == 'custom_pass'


class TestAppSettings:
    """Test the AppSettings class."""

    def test_default_gemini_model(self):
        """Test that AppSettings has correct default Gemini model."""
        with patch.dict(os.environ, {}, clear=True):
            from app.config.app_config import AppSettings
            settings = AppSettings()

            assert settings.gemini_model == 'gemini-2.5-flash'

    def test_loads_google_api_key(self):
        """Test that AppSettings loads Google API key from environment."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-api-key'}, clear=True):
            from app.config.app_config import AppSettings
            settings = AppSettings()

            assert settings.google_api_key == 'test-api-key'

    def test_loads_logfire_token(self):
        """Test that AppSettings loads Logfire token from environment."""
        with patch.dict(os.environ, {'LOGFIRE_WRITE_TOKEN': 'test-token'}, clear=True):
            from app.config.app_config import AppSettings
            settings = AppSettings()

            assert settings.logfire_write_token == 'test-token'


class TestConfigureLogfire:
    """Logfire is optional — a failed configure must never crash AppConfig.

    Regression for the chunk-stage crash where a worker without
    LOGFIRE_WRITE_TOKEN and without local ``~/.logfire`` auth raised
    LogfireConfigError out of ``AppConfig.__init__``, killing the task.
    """

    def test_no_token_configure_raises_does_not_crash(self):
        """No token + logfire.configure() raises → AppConfig still constructs."""
        from logfire.exceptions import LogfireConfigError

        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.app_config.logfire.configure",
                       side_effect=LogfireConfigError("no config")):
                from app.config.app_config import AppConfig
                config = AppConfig()  # must not raise
                assert config.settings.logfire_write_token is None

    def test_bad_token_configure_raises_does_not_crash(self):
        """Token set but configure fails → AppConfig still constructs."""
        from logfire.exceptions import LogfireConfigError

        with patch.dict(os.environ, {'LOGFIRE_WRITE_TOKEN': 'bad-token'}, clear=True):
            with patch("app.config.app_config.logfire.configure",
                       side_effect=LogfireConfigError("bad token")):
                from app.config.app_config import AppConfig
                config = AppConfig()  # must not raise
                assert config.settings.logfire_write_token == 'bad-token'

    def test_configure_succeeds_no_token(self):
        """No token + configure succeeds (local auth) → normal path."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.app_config.logfire.configure") as mock_cfg:
                from app.config.app_config import AppConfig
                AppConfig()
                mock_cfg.assert_called_once_with()


class TestGetOllamaModel:
    """Test the get_ollama_model function."""

    def test_returns_default_when_not_set(self):
        """Test that get_ollama_model returns default when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            from app.config.app_config import get_ollama_model
            result = get_ollama_model()

            assert result == 'deepseek-r1:1.5b'

    def test_returns_configured_value(self):
        """Test that get_ollama_model returns configured value."""
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'llama3.2'}, clear=True):
            from app.config.app_config import get_ollama_model
            result = get_ollama_model()

            assert result == 'llama3.2'
