"""
Unit tests for graph_processing/entity_cache.py.

Covers:
- configure (enable/disable, TTL)
- set/get round-trip
- TTL expiration
- disabled cache returns None
- clear
- stats
- hash normalization (case/whitespace)
"""
import pytest
from datetime import datetime
from unittest.mock import patch

from graph_processing.entity_cache import EntityCache


@pytest.fixture(autouse=True)
def reset_cache():
    EntityCache.clear()
    EntityCache.configure(enabled=True, ttl_seconds=3600)
    yield
    EntityCache.clear()
    EntityCache.configure(enabled=True, ttl_seconds=3600)


class TestEntityCacheBasicOperations:
    def test_set_and_get(self):
        entities = [{"name": "BERT", "type": "MODEL"}]
        EntityCache.set("BERT is a model", entities)
        result = EntityCache.get("BERT is a model")
        assert result == entities

    def test_get_missing_returns_none(self):
        result = EntityCache.get("nonexistent text")
        assert result is None

    def test_hash_normalization_case_insensitive(self):
        entities = [{"name": "GPT"}]
        EntityCache.set("Hello World", entities)
        result = EntityCache.get("hello world")
        assert result == entities

    def test_hash_normalization_strips_whitespace(self):
        entities = [{"name": "GPT"}]
        EntityCache.set("  Hello World  ", entities)
        result = EntityCache.get("Hello World")
        assert result == entities

    def test_overwrite_existing_entry(self):
        EntityCache.set("text", [{"name": "old"}])
        EntityCache.set("text", [{"name": "new"}])
        result = EntityCache.get("text")
        assert result == [{"name": "new"}]


class TestEntityCacheTTL:
    def test_expired_entry_returns_none(self):
        EntityCache.configure(enabled=True, ttl_seconds=10)
        EntityCache.set("text", [{"name": "old"}])

        fake_time = datetime.now()
        with patch('graph_processing.entity_cache.datetime') as mock_dt:
            mock_dt.now.return_value = fake_time
            EntityCache.set("text2", [{"name": "timed"}])

        cache_key = EntityCache._compute_hash("text2")
        EntityCache._cache[cache_key]['cached_at'] = datetime(2000, 1, 1)

        result = EntityCache.get("text2")
        assert result is None

    def test_non_expired_entry_returns_data(self):
        EntityCache.configure(enabled=True, ttl_seconds=3600)
        EntityCache.set("fresh text", [{"name": "fresh"}])
        result = EntityCache.get("fresh text")
        assert result == [{"name": "fresh"}]


class TestEntityCacheDisabled:
    def test_disabled_get_returns_none(self):
        EntityCache.configure(enabled=False)
        EntityCache._cache["test"] = {"entities": [{"name": "x"}], "cached_at": datetime.now()}
        result = EntityCache.get("anything")
        assert result is None

    def test_disabled_set_is_noop(self):
        EntityCache.configure(enabled=False)
        EntityCache.set("text", [{"name": "x"}])
        assert len(EntityCache._cache) == 0


class TestEntityCacheClear:
    def test_clear_empties_cache(self):
        EntityCache.set("a", [{"name": "a"}])
        EntityCache.set("b", [{"name": "b"}])
        assert len(EntityCache._cache) == 2

        EntityCache.clear()
        assert len(EntityCache._cache) == 0


class TestEntityCacheStats:
    def test_stats_structure(self):
        stats = EntityCache.stats()
        assert "enabled" in stats
        assert "size" in stats
        assert "ttl_seconds" in stats

    def test_stats_reflects_size(self):
        EntityCache.set("a", [{"name": "a"}])
        EntityCache.set("b", [{"name": "b"}])
        stats = EntityCache.stats()
        assert stats["size"] == 2

    def test_stats_reflects_config(self):
        EntityCache.configure(enabled=True, ttl_seconds=7200)
        stats = EntityCache.stats()
        assert stats["enabled"] is True
        assert stats["ttl_seconds"] == 7200
