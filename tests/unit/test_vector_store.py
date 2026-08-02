"""
Unit tests for VectorStore.

These tests mock the asyncpg connection pool so they run without a real
database. They verify SQL generation, parameter passing, and result shaping
for the core retrieval/ingestion interface.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from ingestion.embedding.vector_store import VectorStore
from ingestion.embedding.chunk import Chunk


@pytest.fixture
def connection_params():
    return {
        'host': 'localhost',
        'port': '5432',
        'dbname': 'rag_db',
        'user': 'admin',
        'password': 'admin'
    }


@pytest.fixture
def vector_store(connection_params):
    return VectorStore(connection_params, table_name="test_chunks")


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool with acquire/release context manager."""
    pool = MagicMock()
    conn = AsyncMock()

    async def _acquire():
        return conn

    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.release = AsyncMock()

    return pool, conn


@pytest.fixture
def mock_chunks():
    return [
        Chunk(
            id="chunk_1",
            document_id="doc_1",
            text="First chunk text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"page": 1}
        ),
        Chunk(
            id="chunk_2",
            document_id="doc_1",
            text="Second chunk text",
            embedding=[0.4, 0.5, 0.6],
            metadata={"page": 2}
        )
    ]


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization and helpers."""

    def test_connection_string(self, vector_store):
        """Test that connection string is built correctly."""
        expected = "postgresql://admin:admin@localhost:5432/rag_db"
        assert vector_store._build_connection_string() == expected

    def test_safe_table_name_quotes_identifier(self, vector_store):
        """Test that safe_table_name wraps the table name in quotes."""
        assert vector_store.safe_table_name == '"test_chunks"'


class TestVectorStoreAddChunks:
    """Tests for VectorStore.add_chunks with mocked pool."""

    @pytest.mark.asyncio
    async def test_add_chunks_executes_insert(self, vector_store, mock_pool, mock_chunks):
        """Test that add_chunks inserts chunk data into the DB."""
        pool, conn = mock_pool

        with patch('ingestion.embedding.vector_store.ConnectionPoolManager') as mock_mgr:
            mock_mgr.get_pool = AsyncMock(return_value=pool)

            await vector_store.add_chunks(mock_chunks)

        # Pool should be requested once
        mock_mgr.get_pool.assert_awaited_once()
        # Connection should be acquired and released
        pool.acquire.return_value.__aenter__.assert_awaited_once()
        pool.acquire.return_value.__aexit__.assert_awaited_once()
        # executemany should be called with the chunk data
        assert conn.executemany.called
        call_args = conn.executemany.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        assert 'INSERT INTO' in sql
        assert vector_store.safe_table_name in sql
        assert len(params) == len(mock_chunks)
        assert params[0][0] == "chunk_1"
        assert params[0][2] == "First chunk text"


class TestVectorStoreSearch:
    """Tests for VectorStore.search_similar_chunks with mocked pool."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, vector_store, mock_pool):
        """Test that search returns formatted dictionaries."""
        pool, conn = mock_pool

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            'id': 'chunk_1',
            'text': 'Result text',
            'metadata': {'page': 1},
            'document_id': 'doc_1',
            'similarity': 0.95
        }[key]
        mock_row.__contains__ = lambda self, key: key in {
            'id', 'text', 'metadata', 'document_id', 'similarity'
        }
        conn.fetch = AsyncMock(return_value=[mock_row])

        with patch('ingestion.embedding.vector_store.ConnectionPoolManager') as mock_mgr:
            mock_mgr.get_pool = AsyncMock(return_value=pool)

            results = await vector_store.search_similar_chunks(
                query_embedding=[0.1, 0.2, 0.3],
                limit=5,
                threshold=0.7
            )

        assert len(results) == 1
        assert results[0]['chunk_id'] == 'chunk_1'
        assert results[0]['text'] == 'Result text'
        assert results[0]['document_id'] == 'doc_1'
        assert results[0]['similarity'] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_search_filters_by_document_ids(self, vector_store, mock_pool):
        """Test that search filters by document_ids when provided."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        with patch('ingestion.embedding.vector_store.ConnectionPoolManager') as mock_mgr:
            mock_mgr.get_pool = AsyncMock(return_value=pool)

            await vector_store.search_similar_chunks(
                query_embedding=[0.1, 0.2, 0.3],
                limit=5,
                threshold=0.7,
                document_ids=["doc_1", "doc_2"]
            )

        sql = conn.fetch.call_args[0][0]
        assert 'document_id = ANY($3)' in sql


class TestVectorStoreDelete:
    """Tests for VectorStore.delete_document_chunks with mocked pool."""

    @pytest.mark.asyncio
    async def test_delete_document_chunks(self, vector_store, mock_pool):
        """Test deleting chunks for a document."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 5")

        with patch('ingestion.embedding.vector_store.ConnectionPoolManager') as mock_mgr:
            mock_mgr.get_pool = AsyncMock(return_value=pool)

            deleted = await vector_store.delete_document_chunks("doc_1")

        assert deleted == 5
        sql = conn.execute.call_args[0][0]
        assert 'DELETE FROM' in sql
        assert vector_store.safe_table_name in sql
        assert conn.execute.call_args[0][1] == "doc_1"


class TestVectorStoreStats:
    """Tests for VectorStore.get_collection_stats with mocked pool."""

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, vector_store, mock_pool):
        """Test getting collection statistics."""
        pool, conn = mock_pool

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            'total_chunks': 10,
            'total_documents': 2,
            'avg_text_length': 150.5,
            'earliest_chunk': MagicMock(isoformat=MagicMock(return_value='2024-01-01T00:00:00')),
            'latest_chunk': MagicMock(isoformat=MagicMock(return_value='2024-01-02T00:00:00'))
        }[key]
        conn.fetchrow = AsyncMock(return_value=mock_row)

        with patch('ingestion.embedding.vector_store.ConnectionPoolManager') as mock_mgr:
            mock_mgr.get_pool = AsyncMock(return_value=pool)

            stats = await vector_store.get_collection_stats()

        assert stats['total_chunks'] == 10
        assert stats['total_documents'] == 2
        assert stats['avg_text_length'] == pytest.approx(150.5)
        assert stats['earliest_chunk'] == '2024-01-01T00:00:00'
        assert stats['latest_chunk'] == '2024-01-02T00:00:00'
