"""
Integration tests for PostgreSQL + pgvector database connection.

These tests verify:
1. Basic database connectivity
2. pgvector extension installation
3. Table creation with vector columns
4. Vector similarity operations
"""
import pytest
import asyncpg
from typing import List


class TestDatabaseConnection:
    """Test suite for PostgreSQL database connection."""

    @pytest.mark.asyncio
    async def test_database_connection(self, db_params):
        """Test basic database connectivity."""
        conn = None
        try:
            conn = await asyncpg.connect(**db_params)
            assert conn is not None, "Failed to establish database connection"

            # Verify connection is active
            result = await conn.fetchval("SELECT 1")
            assert result == 1, "Database connection not responding correctly"

        finally:
            if conn:
                await conn.close()

    @pytest.mark.asyncio
    async def test_pgvector_extension(self, db_connection):
        """Test that pgvector extension is installed and available."""
        # Try to create the extension (idempotent operation)
        await db_connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Verify extension exists
        result = await db_connection.fetchval(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
        )
        assert result == 1, "pgvector extension is not installed"

    @pytest.mark.asyncio
    async def test_create_vector_table(self, db_connection, test_table_name, cleanup_test_table, embedding_dim):
        """Test creating a table with vector column."""
        # Create table with vector column (uses embedding_dim from fixture)
        create_table_query = f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({embedding_dim}),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        await db_connection.execute(create_table_query)

        # Verify table exists
        result = await db_connection.fetchval(
            f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{test_table_name}'
            """
        )
        assert result == 1, f"Table {test_table_name} was not created"

        # Verify vector column exists and has correct dimension
        column_info = await db_connection.fetchrow(
            f"""
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_name = '{test_table_name}' AND column_name = 'embedding'
            """
        )
        assert column_info is not None, "Vector column 'embedding' not found"

    @pytest.mark.asyncio
    async def test_insert_and_query_vectors(self, db_connection, test_table_name, cleanup_test_table, embedding_dim):
        """Test inserting vectors and performing similarity search."""
        # Create test table
        create_table_query = f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({embedding_dim})
        );
        """
        await db_connection.execute(create_table_query)

        # Create sample vectors (embedding_dim dimensions)
        import random
        random.seed(42)

        def generate_random_vector(dim: int = embedding_dim) -> List[float]:
            """Generate a random normalized vector."""
            vector = [random.random() for _ in range(dim)]
            # Normalize
            magnitude = sum(x**2 for x in vector) ** 0.5
            return [x / magnitude for x in vector]

        # Insert test vectors
        test_data = [
            ("doc1", "Machine learning is amazing", generate_random_vector()),
            ("doc2", "Python is great for data science", generate_random_vector()),
            ("doc3", "Neural networks are powerful", generate_random_vector()),
        ]

        for doc_id, text, embedding in test_data:
            await db_connection.execute(
                f"INSERT INTO {test_table_name} (id, text, embedding) VALUES ($1, $2, $3)",
                doc_id, text, embedding
            )

        # Verify insertion
        count = await db_connection.fetchval(f"SELECT COUNT(*) FROM {test_table_name}")
        assert count == 3, "Not all vectors were inserted"

        # Test similarity search using cosine distance
        query_vector = generate_random_vector()
        results = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {test_table_name}
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_vector
        )

        assert len(results) == 3, "Similarity search did not return expected results"
        assert all('similarity' in dict(r) for r in results), "Similarity scores missing"

        # Verify results are ordered by similarity (descending)
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True), "Results not properly ordered by similarity"

    @pytest.mark.asyncio
    async def test_vector_index_creation(self, db_connection, test_table_name, cleanup_test_table, embedding_dim):
        """Test creating IVFFLAT index for vector similarity search."""
        # Create test table
        create_table_query = f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            embedding vector({embedding_dim})
        );
        """
        await db_connection.execute(create_table_query)

        # Insert some dummy vectors to create index
        import random
        random.seed(42)
        for i in range(10):
            vector = [random.random() for _ in range(embedding_dim)]
            await db_connection.execute(
                f"INSERT INTO {test_table_name} (id, embedding) VALUES ($1, $2)",
                f"doc_{i}", vector
            )

        # Create IVFFLAT index
        index_name = f"{test_table_name}_embedding_idx"
        await db_connection.execute(
            f"""
            CREATE INDEX {index_name} ON {test_table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10);
            """
        )

        # Verify index exists
        result = await db_connection.fetchval(
            f"""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = '{test_table_name}' AND indexname = '{index_name}'
            """
        )
        assert result == 1, f"IVFFLAT index {index_name} was not created"

    @pytest.mark.asyncio
    async def test_connection_pool(self, db_params):
        """Test database connection pooling."""
        pool = None
        try:
            # Create connection pool
            pool = await asyncpg.create_pool(
                **db_params,
                min_size=2,
                max_size=5,
                timeout=10.0
            )

            assert pool is not None, "Failed to create connection pool"

            # Test acquiring connection from pool
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1, "Pool connection not working"

            # Test multiple concurrent connections
            async def test_query(pool):
                async with pool.acquire() as conn:
                    return await conn.fetchval("SELECT 1")

            import asyncio
            results = await asyncio.gather(*[test_query(pool) for _ in range(3)])
            assert all(r == 1 for r in results), "Concurrent connections failed"

        finally:
            if pool:
                await pool.close()

    @pytest.mark.asyncio
    async def test_jsonb_metadata_storage(self, db_connection, test_table_name, cleanup_test_table):
        """Test storing and querying JSONB metadata."""
        # Create test table with JSONB column
        create_table_query = f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT,
            metadata JSONB
        );
        """
        await db_connection.execute(create_table_query)

        # Insert data with JSONB metadata
        import json
        metadata = {
            "source": "test_document.pdf",
            "page": 5,
            "author": "Test Author",
            "tags": ["machine learning", "AI"]
        }

        await db_connection.execute(
            f"INSERT INTO {test_table_name} (id, text, metadata) VALUES ($1, $2, $3)",
            "doc1", "Test text", json.dumps(metadata)
        )

        # Query and verify JSONB data
        result = await db_connection.fetchrow(
            f"SELECT metadata FROM {test_table_name} WHERE id = 'doc1'"
        )

        retrieved_metadata = result['metadata']
        assert retrieved_metadata['source'] == "test_document.pdf"
        assert retrieved_metadata['page'] == 5
        assert "machine learning" in retrieved_metadata['tags']
