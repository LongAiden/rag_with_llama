import asyncio
import json
from typing import Optional, Dict
import asyncpg
import logfire


async def _init_connection(conn: asyncpg.Connection) -> None:
    """
    Register json/jsonb codecs so Python dicts and lists round-trip natively.

    Without this asyncpg maps json/jsonb to `str`, which means passing a dict as
    a query argument raises DataError and reading a JSONB column returns raw text.
    """
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(
            type_name,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


class ConnectionPoolManager:
    _instances: Dict[str, asyncpg.Pool] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    _global_lock = asyncio.Lock()

    @classmethod
    async def get_pool(
        cls,
        connection_string: str,
        min_size: int = 2,
        max_size: int = 10,
    ) -> asyncpg.Pool:
        key = connection_string

        if key in cls._instances:
            return cls._instances[key]

        async with cls._global_lock:
            if key in cls._instances:
                return cls._instances[key]

            if key not in cls._locks:
                cls._locks[key] = asyncio.Lock()

        async with cls._locks[key]:
            if key in cls._instances:
                return cls._instances[key]

            pool = await asyncpg.create_pool(
                connection_string,
                min_size=min_size,
                max_size=max_size,
                init=_init_connection,
            )
            cls._instances[key] = pool
            logfire.info("Connection pool created", min_size=min_size, max_size=max_size)
            return pool

    @classmethod
    async def close_pool(cls, connection_string: str) -> None:
        key = connection_string
        if key in cls._instances:
            pool = cls._instances.pop(key)
            await pool.close()
            logfire.info("Connection pool closed")

    @classmethod
    async def close_all(cls) -> None:
        for key, pool in list(cls._instances.items()):
            await pool.close()
            logfire.info("Connection pool closed", connection_string=key[:50])
        cls._instances.clear()
