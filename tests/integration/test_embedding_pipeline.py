"""
Integration tests for the embedding pipeline.

These tests verify the mechanics of the embedding pipeline (create table,
generate deterministic embeddings, store, retrieve) without relying on
semantic relevance of a real embedding model.

Requirements: a running PostgreSQL with pgvector.
"""
import json
import uuid
import numpy as np
import pytest


class TestEmbeddingPipeline:
    """Integration tests for the embedding pipeline."""

    @pytest.mark.asyncio
    async def test_embedding_pipeline_integration(self, db_connection, test_table_name,
                                                   cleanup_test_table, embedding_model,
                                                   embedding_dim):
        """Test end-to-end embedding pipeline with database storage."""
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

        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is widely used for data science.",
            "Neural networks are inspired by the brain."
        ]

        embeddings = embedding_model.encode(texts)

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk_id = str(uuid.uuid4())
            metadata = {"source": "test", "index": i}

            await db_connection.execute(
                f"""
                INSERT INTO {test_table_name} (id, text, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                chunk_id, text, embedding.tolist(), json.dumps(metadata)
            )

        count = await db_connection.fetchval(f"SELECT COUNT(*) FROM {test_table_name}")
        assert count == 3, f"Expected 3 rows, got {count}"

        # Retrieve using one of the stored texts' exact embedding
        query = texts[0]
        query_embedding = embedding_model.encode(query).tolist()

        results = await db_connection.fetch(
            f"""
            SELECT id, text, (1 - (embedding <=> $1)) as similarity
            FROM {test_table_name}
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_embedding
        )

        assert len(results) > 0, "No results from pipeline"
        # The query matches one stored text exactly, so the top result should be that text
        assert results[0]['text'] == query

    @pytest.mark.asyncio
    async def test_batch_embedding_storage(self, db_connection, test_table_name,
                                           cleanup_test_table, embedding_model,
                                           embedding_dim):
        """Test batch embedding and storage."""
        await db_connection.execute(f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({embedding_dim})
        );
        """)

        texts = [f"Document number {i} with some content." for i in range(50)]
        embeddings = embedding_model.encode(texts)

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            await db_connection.execute(
                f"INSERT INTO {test_table_name} (id, text, embedding) VALUES ($1, $2, $3)",
                str(uuid.uuid4()), text, embedding.tolist()
            )

        count = await db_connection.fetchval(f"SELECT COUNT(*) FROM {test_table_name}")
        assert count == 50, f"Expected 50 rows, got {count}"

    @pytest.mark.asyncio
    async def test_similarity_search_with_controlled_vectors(self, db_connection,
                                                              test_table_name,
                                                              cleanup_test_table,
                                                              embedding_dim):
        """Test that similarity search ordering matches exact vector distances."""
        await db_connection.execute(f"""
        CREATE TABLE {test_table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({embedding_dim}),
            category TEXT
        );
        """)

        # Use orthogonal basis vectors for deterministic distances
        base = np.eye(embedding_dim, dtype=np.float32)
        documents = [
            {"id": "doc0", "text": "Document zero", "category": "A"},
            {"id": "doc1", "text": "Document one", "category": "B"},
            {"id": "doc2", "text": "Document two", "category": "C"},
        ]

        for i, doc in enumerate(documents):
            await db_connection.execute(
                f"""
                INSERT INTO {test_table_name} (id, text, embedding, category)
                VALUES ($1, $2, $3, $4)
                """,
                doc["id"], doc["text"], base[i].tolist(), doc["category"]
            )

        # Query with doc0's exact vector -> doc0 must be top
        query_embedding = base[0].tolist()
        results = await db_connection.fetch(
            f"""
            SELECT id, category, (1 - (embedding <=> $1)) as similarity
            FROM {test_table_name}
            ORDER BY embedding <=> $1
            LIMIT 2
            """,
            query_embedding
        )

        assert len(results) == 2
        assert results[0]['id'] == 'doc0'
        assert results[0]['category'] == 'A'
        # Cosine similarity of identical vectors is ~1.0
        assert abs(results[0]['similarity'] - 1.0) < 1e-5
