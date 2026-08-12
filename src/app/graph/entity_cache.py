import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EntityCache:
    _cache: Dict[str, Dict[str, Any]] = {}
    _enabled: bool = True
    _ttl_seconds: int = 3600

    @classmethod
    def configure(cls, enabled: bool = True, ttl_seconds: int = 3600) -> None:
        cls._enabled = enabled
        cls._ttl_seconds = ttl_seconds
        logger.info(f"EntityCache configured: enabled={enabled}, ttl={ttl_seconds}s")

    @classmethod
    def _compute_hash(cls, text: str) -> str:
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    @classmethod
    def get(cls, text: str) -> Optional[List[Dict[str, Any]]]:
        if not cls._enabled:
            return None

        cache_key = cls._compute_hash(text)
        cached = cls._cache.get(cache_key)

        if cached is None:
            return None

        cached_at = cached.get('cached_at')
        if cached_at:
            age = (datetime.now() - cached_at).total_seconds()
            if age > cls._ttl_seconds:
                del cls._cache[cache_key]
                return None

        logger.debug(f"Entity cache hit for text hash: {cache_key[:16]}...")
        return cached['entities']

    @classmethod
    def set(cls, text: str, entities: List[Dict[str, Any]]) -> None:
        if not cls._enabled:
            return

        cache_key = cls._compute_hash(text)
        cls._cache[cache_key] = {
            'entities': entities,
            'cached_at': datetime.now(),
        }
        logger.debug(f"Entity cached for text hash: {cache_key[:16]}...")

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()
        logger.info("Entity cache cleared")

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        return {
            'enabled': cls._enabled,
            'size': len(cls._cache),
            'ttl_seconds': cls._ttl_seconds,
        }
