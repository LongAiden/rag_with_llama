"""pgvector-backed vector store for chunk storage and similarity search."""

import asyncio
import traceback
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

import asyncpg
import logfire

from app.infra.db import ConnectionPoolManager, quote_ident, validate_table_name
from app.ingestion.embedding.chunk import Chunk


class VectorStore:
    """Vector store using pgvector for efficient similarity search."""

    def __init__(self, connection_params: Dict[str, str], table_name: str = "chunks"):
        """
        Initialize vector store with pgvector support.
        Args:
            connection_params: Database connection parameters
            table_name: Name of the chunks table
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.connection_string = self._build_connection_string()
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def safe_table_name(self) -> str:
        return quote_ident(self.table_name)

    def _build_connection_string(self) -> str:
        """Build asyncpg connection string from parameters."""
        return f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['dbname']}"

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            self._pool = await ConnectionPoolManager.get_pool(self.connection_string)
        return self._pool

    @asynccontextmanager
    async def connection(self):
        """Borrow a pooled connection for the duration of the block.

        Every query in this class goes through here. Releasing via
        ``async with pool.acquire()`` means the connection returns to the pool on
        exceptions and on cancellation too — releasing manually after the last
        statement leaks one connection per failed query, which exhausts the pool.

        The pgvector extension is created once in _initialize_database rather than
        per acquire: it costs a round trip on every query and needs elevated
        privileges every time.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            yield conn

    async def _initialize_database(self):
        """Initialize database with pgvector extension and table."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                # Index names are identifiers too. Validate the table name first,
                # then derive and quote the index names from it — interpolating
                # the raw name here would bypass the check that safe_table_name
                # happens to perform a few lines further down.
                validate_table_name(self.table_name)
                embedding_idx = f'"{self.table_name}_embedding_idx"'
                document_id_idx = f'"{self.table_name}_document_id_idx"'
                doc_name_idx = f'"{self.table_name}_doc_name_idx"'

                async with self.connection() as conn:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # Create table with proper vector column
                    # Assuming 384-dimensional embeddings for all-MiniLM-L6-v2
                    # Adjust dimension based on your model
                    await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.safe_table_name} (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector(384),  -- Adjust dimension as needed
                        metadata JSONB,
                        doc_name TEXT,
                        entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """)

                    # Tables created before migration 006 self-heal here rather than
                    # waiting for a manual migration run against every chunk table.
                    await conn.execute(f"""
                    ALTER TABLE {self.safe_table_name}
                    ADD COLUMN IF NOT EXISTS doc_name TEXT;
                    """)

                    # Create index for similarity search
                    await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {embedding_idx}
                    ON {self.safe_table_name} USING hnsw (embedding vector_cosine_ops);
                    """)

                    # Create index on document_id for filtering
                    await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {document_id_idx}
                    ON {self.safe_table_name} (document_id);
                    """)

                    # Create index on doc_name for name-based listing/filtering
                    await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {doc_name_idx}
                    ON {self.safe_table_name} (doc_name);
                    """)

                    self._initialized = True
                    print(f"Database initialized with table: {self.table_name}")

            except Exception as e:
                tb = traceback.format_exc()
                logfire.error(
                    "Database initialization failed",
                    table_name=self.table_name,
                    error_type=type(e).__name__,
                    error=str(e),
                    traceback=tb,
                )
                print(f"[DB INIT ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
                raise

    async def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """Add chunks to vector store using batch insert for efficiency."""
        try:
            if not self._initialized:
                await self._initialize_database()

            # Prepare data for batch insert
            chunk_data = []
            for chunk in chunks:
                # Convert embedding list to proper pgvector format
                embedding_str = '[' + ','.join(map(str, chunk.embedding)) + ']'

                # metadata is passed as a dict — the pool registers a jsonb codec
                # (see infra/db/pool.py), so asyncpg encodes it.
                chunk_data.append((
                    chunk.id,
                    chunk.document_id,
                    chunk.text,
                    embedding_str,
                    chunk.metadata if chunk.metadata else {},
                    chunk.doc_name,
                ))

            # Use asyncpg's executemany for efficient batch insert
            insert_sql = f"""
            INSERT INTO {self.safe_table_name} (id, document_id, text, embedding, metadata, doc_name)
            VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                text        = EXCLUDED.text,
                embedding   = EXCLUDED.embedding,
                metadata    = EXCLUDED.metadata,
                doc_name    = EXCLUDED.doc_name;
            """

            # Process in batches
            async with self.connection() as conn:
                for i in range(0, len(chunk_data), batch_size):
                    batch = chunk_data[i:i + batch_size]
                    await conn.executemany(insert_sql, batch)

            print(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            tb = traceback.format_exc()
            logfire.error(
                "Chunk insertion failed",
                table_name=self.table_name,
                num_chunks=len(chunks),
                error_type=type(e).__name__,
                error=str(e),
                traceback=tb,
            )
            print(f"[ADD CHUNKS ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
            raise

    async def search_similar_chunks(self, query_embedding: List[float],
                                    limit: int = 5, threshold: float = 0.7,
                                    document_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for similar chunks using pgvector cosine similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Similarity threshold (0-1, higher = more similar)
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of similar chunks with metadata
        """
        try:
            if not self._initialized:
                await self._initialize_database()

            # Build query with optional document filtering
            base_query = f"""
                SELECT
                    id,
                    text,
                    metadata,
                    document_id,
                    doc_name,
                    (1 - (embedding <=> $1::vector)) as similarity
                FROM {self.safe_table_name}
                WHERE (1 - (embedding <=> $1::vector)) >= $2
            """

            # Convert query embedding to proper pgvector format
            query_embedding_str = '[' + \
                ','.join(map(str, query_embedding)) + ']'
            params = [query_embedding_str, threshold]

            if document_ids:
                base_query += " AND document_id = ANY($3)"
                params.append(document_ids)
                base_query += """
                    ORDER BY embedding <=> $1
                    LIMIT $4
                """
                params.append(limit)
            else:
                base_query += """
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """
                params.append(limit)

            async with self.connection() as conn:
                rows = await conn.fetch(base_query, *params)

            return [
                {
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
                    'doc_name': row['doc_name'],
                    'similarity': float(row['similarity'])
                }
                for row in rows
            ]

        except Exception as e:
            print(f"Error searching chunks: {e}")
            raise

    async def search_bm25(
        self,
        query: str,
        limit: int = 20,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Lexical retrieval using BM25Okapi over all chunks in the table.

        Args:
            query: Search query string
            limit: Maximum number of results
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of chunks with bm25_score, sorted descending
        """
        from rank_bm25 import BM25Okapi
        import numpy as np

        try:
            if not self._initialized:
                await self._initialize_database()

            base_query = f"""
                SELECT id, text, metadata, document_id, doc_name
                FROM {self.safe_table_name}
            """
            params: list = []

            if document_ids:
                base_query += " WHERE document_id = ANY($1)"
                params.append(document_ids)

            async with self.connection() as conn:
                rows = await conn.fetch(base_query, *params)

            if not rows:
                return []

            def _score() -> "np.ndarray":
                """Tokenise, index, and score. Pure CPU — must not run on the loop."""
                corpus = [row['text'].lower().split() for row in rows]
                bm25 = BM25Okapi(corpus)
                return bm25.get_scores(query.lower().split())

            # Building the index is O(corpus) on every call; keeping it on the event
            # loop stalls every other request for the duration.
            bm25_scores = await asyncio.to_thread(_score)

            top_indices = np.argsort(bm25_scores)[::-1][:limit]

            results = []
            for idx in top_indices:
                row = rows[idx]
                results.append({
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
                    'doc_name': row['doc_name'],
                    'bm25_score': float(bm25_scores[idx]),
                })

            return results

        except Exception as e:
            print(f"Error in BM25 search: {e}")
            raise

    async def get_chunks_by_section(
        self,
        section_path: str,
        document_ids: List[str],
        limit: int = 20,
    ) -> List[Dict]:
        """Return all chunks that share the same section_path, ordered by chunk_index."""
        if not section_path:
            return []
        try:
            if not self._initialized:
                await self._initialize_database()
            query = f"""
                SELECT id, text, metadata, document_id, doc_name
                FROM {self.safe_table_name}
                WHERE metadata->>'section_path' = $1
                  AND document_id = ANY($2)
                ORDER BY (metadata->>'chunk_index')::int
                LIMIT $3
            """
            async with self.connection() as conn:
                rows = await conn.fetch(query, section_path, document_ids, limit)
            return [
                {
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
                    'doc_name': row['doc_name'],
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Error fetching chunks by section: {e}")
            return []

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        try:
            if not self._initialized:
                await self._initialize_database()

            async with self.connection() as conn:
                result = await conn.execute(
                    f"DELETE FROM {self.safe_table_name} WHERE document_id = $1", document_id)
            deleted_count = int(result.split()[-1]) if result else 0
            print(
                f"Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            validate_table_name(self.table_name)
            async with self.connection() as conn:
                exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = $1)",
                    self.table_name,
                )
                if not exists:
                    return {
                        'total_chunks': 0,
                        'total_documents': 0,
                        'avg_text_length': 0,
                        'earliest_chunk': None,
                        'latest_chunk': None,
                    }

                row = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT document_id) as total_documents,
                    AVG(LENGTH(text)) as avg_text_length,
                    MIN(created_at) as earliest_chunk,
                    MAX(created_at) as latest_chunk
                FROM {self.safe_table_name}
                """)

            stats = {
                'total_chunks': row['total_chunks'],
                'total_documents': row['total_documents'],
                'avg_text_length': float(row['avg_text_length']) if row['avg_text_length'] else 0,
                'earliest_chunk': row['earliest_chunk'].isoformat() if row['earliest_chunk'] else None,
                'latest_chunk': row['latest_chunk'].isoformat() if row['latest_chunk'] else None
            }

            return stats
        except Exception as e:
            print(f"Error getting stats: {type(e).__name__}: {e}")
            raise
