"""
Unit tests for config/graph_config.py.

Covers:
- GraphConfig defaults
- GraphConfig env var loading
- is_entity_type_enabled
- is_relationship_type_enabled
- get_extraction_config
- get_graph_config singleton
"""
import os
import pytest
from unittest.mock import patch

import config.graph_config as gc_module
from config.graph_config import (
    GraphConfig,
    get_graph_config,
    is_entity_type_enabled,
    is_relationship_type_enabled,
    get_extraction_config,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    gc_module._graph_config = None
    yield
    gc_module._graph_config = None


_ENV_KEYS = [
    "GRAPH_LLM_PROVIDER", "ENTITY_CONFIDENCE_THRESHOLD",
    "RELATIONSHIP_CONFIDENCE_THRESHOLD", "MAX_ENTITIES_PER_CHUNK",
    "MAX_RELATIONSHIPS_PER_CHUNK", "DEFAULT_MAX_HOPS",
    "BATCH_SIZE", "GEMINI_MAX_RETRIES", "GEMINI_RETRY_INITIAL_DELAY",
    "GEMINI_RETRY_EXPONENTIAL_BASE", "GOOGLE_API_KEY", "GEMINI_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_VLM_MODEL",
]


def _clean_env():
    return {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}


def _isolated_config(**overrides):
    return GraphConfig(_env_file="/dev/null", **overrides)


class TestGraphConfigDefaults:
    def test_default_llm_provider(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.llm_provider == "ollama"

    def test_default_entity_confidence(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.entity_confidence_threshold == 0.6

    def test_default_relationship_confidence(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.relationship_confidence_threshold == 0.6

    def test_default_max_entities(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.max_entities_per_chunk == 50

    def test_default_max_relationships(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.max_relationships_per_chunk == 100

    def test_default_max_hops(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.default_max_hops == 2

    def test_default_batch_size(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.batch_size == 20

    def test_default_embedding_dim(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.entity_embedding_dimension == 384

    def test_entity_types_not_empty(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert len(config.enabled_entity_types) > 0

    def test_relationship_types_not_empty(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert len(config.enabled_relationship_types) > 0

    def test_retry_config_defaults(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            config = _isolated_config()
            assert config.gemini_max_retries == 3
            assert config.gemini_retry_initial_delay == 2.0
            assert config.gemini_retry_exponential_base == 2.0


class TestGraphConfigValidation:
    def test_confidence_threshold_min(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with pytest.raises(Exception):
                _isolated_config(entity_confidence_threshold=-0.1)

    def test_confidence_threshold_max(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with pytest.raises(Exception):
                _isolated_config(entity_confidence_threshold=1.1)

    def test_max_entities_min(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with pytest.raises(Exception):
                _isolated_config(max_entities_per_chunk=0)

    def test_max_hops_range(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            with pytest.raises(Exception):
                _isolated_config(default_max_hops=0)
            with pytest.raises(Exception):
                _isolated_config(default_max_hops=6)


class TestGraphConfigEnvLoading:
    def test_loads_provider_from_env(self):
        with patch.dict(os.environ, {"GRAPH_LLM_PROVIDER": "gemini"}):
            config = GraphConfig()
            assert config.llm_provider == "gemini"

    def test_loads_batch_size_from_env(self):
        with patch.dict(os.environ, {"BATCH_SIZE": "10"}):
            config = GraphConfig()
            assert config.batch_size == 10


class TestGetGraphConfigSingleton:
    def test_returns_same_instance(self):
        gc_module._graph_config = None
        c1 = get_graph_config()
        c2 = get_graph_config()
        assert c1 is c2


class TestIsEntityTypeEnabled:
    def test_enabled_type(self):
        gc_module._graph_config = None
        assert is_entity_type_enabled("MODEL") is True

    def test_enabled_type_case_insensitive(self):
        gc_module._graph_config = None
        assert is_entity_type_enabled("model") is True

    def test_disabled_type(self):
        gc_module._graph_config = None
        assert is_entity_type_enabled("NONEXISTENT_TYPE") is False


class TestIsRelationshipTypeEnabled:
    def test_enabled_type(self):
        gc_module._graph_config = None
        assert is_relationship_type_enabled("USES") is True

    def test_enabled_type_case_insensitive(self):
        gc_module._graph_config = None
        assert is_relationship_type_enabled("uses") is True

    def test_disabled_type(self):
        gc_module._graph_config = None
        assert is_relationship_type_enabled("NONEXISTENT_TYPE") is False


class TestGetExtractionConfig:
    def test_returns_dict(self):
        gc_module._graph_config = None
        result = get_extraction_config()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        gc_module._graph_config = None
        result = get_extraction_config()
        expected_keys = [
            "entity_confidence_threshold",
            "relationship_confidence_threshold",
            "max_entities_per_chunk",
            "max_relationships_per_chunk",
            "enabled_entity_types",
            "enabled_relationship_types",
        ]
        for key in expected_keys:
            assert key in result
