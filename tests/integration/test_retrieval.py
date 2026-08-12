"""
Tests for retrieval process.

These tests verify vector similarity search mechanics using a deterministic
fake embedding model. They focus on retrieval behavior (ordering, thresholds,
limits, metadata preservation) rather than semantic relevance, so they remain
stable when the embedding model or document corpus changes.

Requirements: a running PostgreSQL with pgvector.
"""
import pytest
import numpy as np
import json


class TestRetrieval:
    """Test suite for document retrieval process."""

    @pytest.fixture
    async def populated_test_table(self, db_connection, test_table_name,
                                   embedding_model, embedding_dim, sample_texts):
        """Fixture that creates and populates a test table with deterministic embeddings."""
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

        embeddings = embedding_model.encode(sample_texts)

        for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
            metadata = {
                "source": f"test_doc_{i}.txt",
                "index": i
            }

            await db_connection.execute(
                f"""
                INSERT INTO {test_table_name} (id, text, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                f"chunk_{i}", text, embedding.tolist(), json.dumps(metadata)
            )

        yield test_table_name

        # Cleanup
        try:
            await db_connection.execute(f"DROP TABLE IF EXISTS {test_table_name};")
        except Exception as e:
            print(f"Cleanup warning: {e}")

    @pytest.mark.asyncio
    async def test_basic_similarity_search(self, db_connection, populated_test_table,
                                           embedding_model, embedding_dim):
        """Test basic vector similarity search returns ordered results."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        results = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 5
            """,
            query_embedding
        )

        assert len(results) > 0, "No results returned from similarity search"
        assert len(results) <= 5, "Too many results returned"

        for result in results:
            assert 'id' in dict(result), "Result missing 'id'"
            assert 'text' in dict(result), "Result missing 'text'"
            assert 'similarity' in dict(result), "Result missing 'similarity'"

        # Results should be ordered by descending similarity
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True), \
            "Results not properly ordered by similarity"

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, db_connection, populated_test_table,
                                                   embedding_model):
        """Test retrieval with similarity threshold filters correctly."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()
        threshold = 0.5

        results = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            WHERE (1 - (embedding <=> $1)) >= $2
            ORDER BY embedding <=> $1
            LIMIT 10
            """,
            query_embedding, threshold
        )

        for result in results:
            assert result['similarity'] >= threshold, \
                f"Result has similarity {result['similarity']:.3f} below threshold {threshold}"

    @pytest.mark.asyncio
    async def test_retrieval_with_limit(self, db_connection, populated_test_table,
                                        embedding_model):
        """Test that limit parameter works correctly."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        for limit in [1, 2, 3]:
            results = await db_connection.fetch(
                f"""
                SELECT id, text, (1 - (embedding <=> $1)) as similarity
                FROM {populated_test_table}
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding, limit
            )

            assert len(results) == limit, f"Expected {limit} results, got {len(results)}"

    @pytest.mark.asyncio
    async def test_retrieval_with_metadata(self, db_connection, populated_test_table,
                                           embedding_model):
        """Test retrieval includes metadata."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        results = await db_connection.fetch(
            f"""
            SELECT id, text, metadata, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_embedding
        )

        assert len(results) > 0, "No results returned"

        for result in results:
            assert 'metadata' in dict(result), "Result missing metadata"
            metadata = result['metadata']
            assert 'source' in metadata, "Metadata missing 'source'"
            assert 'index' in metadata, "Metadata missing 'index'"

    @pytest.mark.asyncio
    async def test_retrieval_consistency(self, db_connection, populated_test_table,
                                         embedding_model):
        """Test that same query returns consistent results."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        results1 = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 5
            """,
            query_embedding
        )

        results2 = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 5
            """,
            query_embedding
        )

        assert len(results1) == len(results2), "Result counts differ"

        for r1, r2 in zip(results1, results2):
            assert r1['id'] == r2['id'], "Result IDs differ"
            assert abs(r1['similarity'] - r2['similarity']) < 1e-6, \
                "Similarity scores differ"

    @pytest.mark.asyncio
    async def test_retrieval_similarity_scores_in_valid_range(self, db_connection,
                                                               populated_test_table,
                                                               embedding_model):
        """Test that similarity scores are in valid range."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        results = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 10
            """,
            query_embedding
        )

        for result in results:
            similarity = result['similarity']
            assert -1.0 <= similarity <= 1.0, \
                f"Similarity score {similarity} out of valid range"

    @pytest.mark.asyncio
    async def test_end_to_end_retrieval_pipeline(self, db_connection, test_table_name,
                                                 cleanup_test_table, embedding_dim):
        """Test complete end-to-end retrieval pipeline with controlled vectors."""
        await db_connection.execute(f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({embedding_dim}),
            metadata JSONB
        );
        """)

        # Use simple orthogonal-ish vectors so we know exactly what should be retrieved
        base = np.eye(embedding_dim, dtype=np.float32)
        documents = [
            {"id": "doc1", "text": "Python programming", "vec": base[0]},
            {"id": "doc2", "text": "Machine learning", "vec": base[1]},
            {"id": "doc3", "text": "Unrelated topic", "vec": base[2]},
        ]

        for doc in documents:
            await db_connection.execute(
                f"""
                INSERT INTO {test_table_name} (id, text, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                doc["id"], doc["text"], doc["vec"].tolist(),
                json.dumps({"category": doc["id"]})
            )

        # Query with the exact same vector as doc1 -> doc1 must be top
        query_embedding = base[0].tolist()
        results = await db_connection.fetch(
            f"""
            SELECT id, text, metadata, (1 - (embedding <=> $1)) as similarity
            FROM {test_table_name}
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_embedding
        )

        assert len(results) > 0, "End-to-end pipeline returned no results"
        assert results[0]['id'] == 'doc1', f"Expected doc1 as top result, got {results[0]['id']}"
        assert results[0]['metadata']['category'] == 'doc1', "Metadata not correctly preserved"
        # Similarity with itself should be ~1.0
        assert abs(results[0]['similarity'] - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_retrieval_with_different_similarity_functions(self, db_connection,
                                                                 populated_test_table,
                                                                 embedding_model):
        """Test retrieval with different similarity/distance functions."""
        query = sample_texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        cosine_results = await db_connection.fetch(
            f"""
            SELECT id, (1 - (embedding <=> $1)) as similarity
            FROM {populated_test_table}
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_embedding
        )

        assert len(cosine_results) > 0, "Cosine distance search failed"

        l2_results = await db_connection.fetch(
            f"""
            SELECT id, embedding <-> $1 as distance
            FROM {populated_test_table}
            ORDER BY embedding <-> $1
            LIMIT 3
            """,
            query_embedding
        )

        assert len(l2_results) > 0, "L2 distance search failed"

        # Both should return results
        assert len(cosine_results) == len(l2_results), \
            "Different distance functions returned different result counts"
