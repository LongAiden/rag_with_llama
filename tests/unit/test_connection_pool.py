"""
Unit tests for the shared asyncpg connection pool.

The pool registers json/jsonb codecs. Without them asyncpg maps those types to
`str`: passing a dict as a query argument raises DataError, and reading a JSONB
column hands back raw text. Both broke the ingestion status DB, which stores
document metadata and chunk artifacts as JSONB.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_pool_is_created_with_a_json_codec_initialiser():
    """create_pool must receive an init callback, or JSONB columns break."""
    from app.infra.db.pool import ConnectionPoolManager

    pool = MagicMock()
    with patch("asyncpg.create_pool", AsyncMock(return_value=pool)) as mock_create:
        ConnectionPoolManager._instances.clear()
        try:
            result = await ConnectionPoolManager.get_pool("postgresql://t:t@localhost/t")
        finally:
            ConnectionPoolManager._instances.clear()

    assert result is pool
    assert mock_create.call_args.kwargs["init"] is not None


@pytest.mark.asyncio
async def test_init_registers_json_and_jsonb_codecs():
    """Both json and jsonb are registered, with json.dumps/loads round-tripping."""
    from app.infra.db.pool import _init_connection

    conn = MagicMock()
    conn.set_type_codec = AsyncMock()

    await _init_connection(conn)

    registered = {call.args[0]: call.kwargs for call in conn.set_type_codec.call_args_list}
    assert set(registered) == {"json", "jsonb"}
    for kwargs in registered.values():
        assert kwargs["encoder"] is json.dumps
        assert kwargs["decoder"] is json.loads
        assert kwargs["schema"] == "pg_catalog"


@pytest.mark.asyncio
async def test_codec_round_trips_the_ingestion_artifact_shapes():
    """The encoder/decoder pair must survive the shapes the pipeline stores."""
    from app.infra.db.pool import _init_connection

    conn = MagicMock()
    conn.set_type_codec = AsyncMock()
    await _init_connection(conn)

    encoder = conn.set_type_codec.call_args_list[0].kwargs["encoder"]
    decoder = conn.set_type_codec.call_args_list[0].kwargs["decoder"]

    document_metadata = {"filename": "test.pdf", "file_size": 1024, "validation_passed": True}
    chunk_artifact = [
        {"text": "chunk 1", "page_number": 1, "section_path": "[A]", "token_count": None},
        {"text": "chunk 2", "page_number": 2, "section_path": "", "token_count": 42},
    ]

    assert decoder(encoder(document_metadata)) == document_metadata
    assert decoder(encoder(chunk_artifact)) == chunk_artifact
