from ingestion.processors.processor_factory import get_processor_for_file
from ingestion.processors.page_utils import get_page_number_for_position
from ingestion.text_cleaning.cleaners import TextCleaningPipeline
from infra.db import quote_ident, ConnectionPoolManager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import asyncio
import os
import re
import traceback
import uuid
import asyncpg
import logfire
from pathlib import Path


def _extract_section_hierarchy(markdown: str, position: int) -> str:
    """
    Extract the heading hierarchy (H1 > H2 > H3) at a given character position.

    Returns a prefix like "[Chapter 1].[Section 2].[Subsection A]"
    or empty string if no headings precede the position.
    """
    segment = markdown[:position]
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    hierarchy: Dict[int, str] = {}
    for match in heading_pattern.finditer(segment):
        level = len(match.group(1))
        title = match.group(2).strip()
        # Remove inline markdown from heading titles (bold, italic, backticks)
        title = re.sub(r'[*_`]', '', title).strip()
        hierarchy[level] = title
        # Clear any deeper levels — they're no longer in scope
        for deeper in list(hierarchy.keys()):
            if deeper > level:
                del hierarchy[deeper]

    if not hierarchy:
        return ""

    parts = [hierarchy[lvl] for lvl in sorted(hierarchy.keys())]
    return "[" + "].[".join(parts) + "]"


def _extract_page_content(markdown: str, page_number: int) -> str:
    """
    Extract the full text content of a specific page from markdown
    that contains [Page N] markers produced by PDFToMarkdownConverter.
    """
    page_markers = list(re.finditer(r'\[Page (\d+)\]', markdown))
    if not page_markers:
        return ""

    start_pos = None
    end_pos = None
    for i, marker in enumerate(page_markers):
        if int(marker.group(1)) == page_number:
            start_pos = marker.end()  # content starts right after the marker
            end_pos = page_markers[i + 1].start() if i + 1 < len(page_markers) else len(markdown)
            break

    if start_pos is None:
        return ""

    return markdown[start_pos:end_pos].strip()

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Chunk:
    """Chunk data structure to match your existing interface."""
    id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict] = None


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.

        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Args:
            text: Input text
        Returns:
            List of embedding values
        """
        # Validate and clean input text
        if text is None:
            logfire.warn("Embedding called with None text, using empty string")
            text = ""
        elif not isinstance(text, str):
            logfire.warn("Embedding called with non-string text, converting to string",
                         text_type=type(text).__name__)
            text = str(text)

        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Args:
            texts: List of input texts
        Returns:
            List of embedding lists
        """
        # Validate and clean input texts - be very defensive
        valid_texts = []
        none_count = 0
        non_string_count = 0
        problematic_items = []

        for i, text in enumerate(texts):
            try:
                if text is None:
                    valid_texts.append("")  # Replace None with empty string
                    none_count += 1
                elif isinstance(text, str):
                    valid_texts.append(text)
                elif isinstance(text, (list, dict, tuple)):
                    # Complex types that can't be directly converted
                    logfire.error("Complex type encountered in embedding batch",
                                  index=i,
                                  text_type=type(text).__name__,
                                  text_value=str(text)[:200])
                    # Fallback to string conversion
                    valid_texts.append(str(text))
                    non_string_count += 1
                    problematic_items.append((i, type(text).__name__))
                else:
                    # Convert non-string to string
                    converted = str(text)
                    valid_texts.append(converted)
                    non_string_count += 1
            except Exception as e:
                logfire.error("Failed to process text at index",
                              index=i,
                              error=str(e),
                              text_type=type(text).__name__)
                valid_texts.append("")  # Use empty string as fallback
                problematic_items.append((i, f"error: {str(e)}"))

        # Log warnings if we encountered invalid texts
        if none_count > 0 or non_string_count > 0:
            logfire.warn("Invalid texts encountered in batch embedding",
                         total_texts=len(texts),
                         none_count=none_count,
                         non_string_count=non_string_count,
                         problematic_items_count=len(problematic_items))

        # Final safety check - ensure all items are strings
        final_texts = []
        for i, text in enumerate(valid_texts):
            if not isinstance(text, str):
                logfire.error("Non-string made it through validation",
                              index=i,
                              type=type(text).__name__)
                final_texts.append(str(text))
            else:
                final_texts.append(text)

        # Ensure we have a proper list of proper strings with aggressive filtering
        safe_texts = []
        for idx, x in enumerate(final_texts):
            try:
                # Convert to string if needed
                if not isinstance(x, str):
                    text = str(x)
                else:
                    text = x

                # Replace empty strings with space
                if len(text.strip()) == 0:
                    text = " "

                # Ensure it's a plain Python str (not bytes, not subclass)
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')

                # Final conversion to ensure plain str type
                text = str(text)

                safe_texts.append(text)
            except Exception as e:
                logfire.error(
                    "Failed to process text in final cleanup", index=idx, error=str(e))
                safe_texts.append(" ")  # Fallback to space

        # Verify the conversion worked
        logfire.info("Preparing to encode batch",
                     total_texts=len(safe_texts),
                     all_strings=all(isinstance(t, str) for t in safe_texts),
                     sample_lengths=[len(t) for t in safe_texts[:5]])

        try:
            # Force each text through encode/decode to ensure pure Python strings
            texts_to_encode = []
            for t in safe_texts:
                # Convert to bytes and back to ensure pure string
                clean_text = str(t).encode(
                    'utf-8', errors='replace').decode('utf-8')
                texts_to_encode.append(clean_text)

            # Encode in batches for efficiency (batch_size=32)
            batch_size = 32
            all_embeddings = []

            logfire.info("Starting batch embedding",
                         total_texts=len(texts_to_encode),
                         batch_size=batch_size,
                         num_batches=(len(texts_to_encode) + batch_size - 1) // batch_size)

            for i in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[i:i + batch_size]
                batch_num = i // batch_size + 1

                try:
                    # Encode this batch
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=len(batch)
                    )
                    all_embeddings.extend([emb.tolist()
                                          for emb in batch_embeddings])

                    if batch_num % 10 == 0:  # Log every 10 batches
                        logfire.info(f"Processed batch {batch_num}",
                                     embeddings_so_far=len(all_embeddings))
                except Exception as batch_error:
                    logfire.error(f"Batch {batch_num} failed, falling back to one-by-one",
                                  error=str(batch_error),
                                  batch_size=len(batch))
                    # Fallback: encode one by one for this batch
                    for text in batch:
                        try:
                            embedding = self.model.encode(
                                text, show_progress_bar=False, convert_to_numpy=True)
                            all_embeddings.append(embedding.tolist())
                        except Exception:
                            all_embeddings.append([0.0] * self.embedding_dim)

            logfire.info("Batch embedding completed",
                         total_texts=len(texts_to_encode),
                         total_embeddings=len(all_embeddings))

            return all_embeddings
        except Exception as e:
            logfire.error("Embedding encoding failed completely",
                          error=str(e),
                          error_type=type(e).__name__,
                          total_texts=len(safe_texts))
            # Log samples for debugging
            for i, text in enumerate(safe_texts[:3]):
                logfire.error(f"Sample text {i}",
                              text_type=type(text).__name__,
                              text_repr=repr(text)[:200],
                              text_len=len(text) if isinstance(text, str) else 0)
            raise


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

    async def _get_connection(self):
        """Get a database connection from the pool with pgvector support."""
        pool = await self._get_pool()
        conn = await pool.acquire()
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        return conn

    async def _release_connection(self, conn):
        """Release a connection back to the pool."""
        pool = await self._get_pool()
        await pool.release(conn)

    async def _initialize_database(self):
        """Initialize database with pgvector extension and table."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                conn = await self._get_connection()
                try:
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
                        entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """)

                    # Create index for similarity search
                    await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.safe_table_name} USING hnsw (embedding vector_cosine_ops);
                    """)

                    # Create index on document_id for filtering
                    await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_document_id_idx
                    ON {self.safe_table_name} (document_id);
                    """)

                    self._initialized = True
                    print(f"Database initialized with table: {self.table_name}")
                finally:
                    await self._release_connection(conn)

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

            conn = await self._get_connection()

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
                    chunk.metadata if chunk.metadata else {}
                ))

            # Use asyncpg's executemany for efficient batch insert
            insert_sql = f"""
            INSERT INTO {self.safe_table_name} (id, document_id, text, embedding, metadata)
            VALUES ($1, $2, $3, $4::vector, $5::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                text        = EXCLUDED.text,
                embedding   = EXCLUDED.embedding,
                metadata    = EXCLUDED.metadata;
            """

            # Process in batches
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i + batch_size]
                await conn.executemany(insert_sql, batch)

            await self._release_connection(conn)
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

            conn = await self._get_connection()

            # Build query with optional document filtering
            base_query = f"""
                SELECT
                    id,
                    text,
                    metadata,
                    document_id,
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

            rows = await conn.fetch(base_query, *params)

            results = []
            for row in rows:
                results.append({
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
                    'similarity': float(row['similarity'])
                })

            await self._release_connection(conn)
            return results

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

            conn = await self._get_connection()

            base_query = f"""
                SELECT id, text, metadata, document_id
                FROM {self.safe_table_name}
            """
            params: list = []

            if document_ids:
                base_query += " WHERE document_id = ANY($1)"
                params.append(document_ids)

            rows = await conn.fetch(base_query, *params)
            await self._release_connection(conn)

            if not rows:
                return []

            corpus = [row['text'].lower().split() for row in rows]
            bm25 = BM25Okapi(corpus)
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)

            top_indices = np.argsort(bm25_scores)[::-1][:limit]

            results = []
            for idx in top_indices:
                row = rows[idx]
                results.append({
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
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
            conn = await self._get_connection()
            query = f"""
                SELECT id, text, metadata, document_id
                FROM {self.safe_table_name}
                WHERE metadata->>'section_path' = $1
                  AND document_id = ANY($2)
                ORDER BY (metadata->>'chunk_index')::int
                LIMIT $3
            """
            rows = await conn.fetch(query, section_path, document_ids, limit)
            await self._release_connection(conn)
            return [
                {
                    'chunk_id': row['id'],
                    'text': row['text'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
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

            conn = await self._get_connection()
            result = await conn.execute(
                f"DELETE FROM {self.safe_table_name} WHERE document_id = $1", document_id)
            deleted_count = int(result.split()[-1]) if result else 0
            await self._release_connection(conn)
            print(
                f"Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            if not self._initialized:
                await self._initialize_database()

            conn = await self._get_connection()
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

            await self._release_connection(conn)
            return stats
        except Exception as e:
            print(f"Error getting stats: {type(e).__name__}: {e}")
            raise


class ChunkEmbeddingPipeline:
    """Complete pipeline for chunking documents and storing embeddings."""

    def __init__(self, db_params: Dict[str, str], embedding_model: str,
                 table_name: str):
        """
        Initialize the pipeline.
        Args:
            db_params: Database connection parameters
            embedding_model: SentenceTransformer model name
            table_name: Name of the chunks table
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore(db_params, table_name)

    @staticmethod
    async def parse_file(
        file_path: str,
        document_id: str,
        parse_backend: str = "",
    ) -> Dict[str, Any]:
        """
        Parse a document into raw text/markdown without chunking or embedding.

        Static because parsing needs no embedding model — the ingestion worker
        calls it on the class so the parse stage does not pay for loading one.

        Returns:
            Dict with parsed_text, parser_used, filename, file_type, file_size,
            and page_mapping (for non-PDF files).
        """
        file_path = Path(file_path)
        filename = file_path.name
        file_type = file_path.suffix.lower().replace('.', '')
        file_size = file_path.stat().st_size

        print(f"Parsing document: {filename} (ID: {document_id})")

        if file_path.suffix.lower() == '.pdf':
            from ingestion.processors.pdf_parser_factory import create_pdf_parser
            from config.app_config import AppSettings

            settings = AppSettings()
            backend = parse_backend or settings.pdf_parser_backend
            parser = create_pdf_parser(backend, settings)
            parsed_text = parser.parse_pdf(str(file_path), output_path=None)
            parser_used = parser.get_backend_name()
            page_mapping: List[Any] = []
        else:
            processor = get_processor_for_file(str(file_path))
            if not processor.validate_file(str(file_path)):
                raise ValueError(f"Invalid file: {file_path}")
            parsed_text, page_mapping = processor.extract_text(str(file_path))
            parser_used = processor.__class__.__name__

        logfire.info("Document parsed",
                     document_id=document_id,
                     parser_used=parser_used,
                     file_type=file_type,
                     file_size=file_size,
                     text_length=len(parsed_text))

        return {
            "parsed_text": parsed_text,
            "parser_used": parser_used,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "page_mapping": page_mapping,
        }

    @staticmethod
    def chunk_parsed_document(
        parsed_result: Dict[str, Any],
        chunk_size: int = 512,
        similarity_threshold: float = 0.5,
        chunker_type: Optional[str] = None,
    ) -> List[Any]:
        """
        Chunk parsed text. For PDFs, uses markdown-aware chunking and page
        markers; for DOCX/TXT, uses the standard chunker factory.

        Static because chunking needs no embedding model — see parse_file.
        """
        parsed_text = parsed_result["parsed_text"]
        file_type = parsed_result["file_type"]
        filename = parsed_result["filename"]

        if file_type == 'pdf':
            from ingestion.chunking.chunker_factory import chunk_markdown

            chunks = chunk_markdown(
                parsed_text,
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=50,
                similarity_threshold=similarity_threshold,
            )

            page_content_cache: Dict[int, str] = {}
            last_section_prefix = ""
            for chunk in chunks:
                if hasattr(chunk, 'start_index') and chunk.start_index is not None:
                    segment = parsed_text[:chunk.start_index]
                    page_matches = list(re.finditer(r'\[Page (\d+)\]', segment))
                    chunk.page_number = int(page_matches[-1].group(1)) if page_matches else 1

                    section_prefix = _extract_section_hierarchy(parsed_text, chunk.start_index)
                    if not section_prefix:
                        section_prefix = last_section_prefix
                    else:
                        last_section_prefix = section_prefix

                    chunk.section_path = section_prefix
                    if section_prefix:
                        chunk.text = f"{section_prefix} - {chunk.text}"
                else:
                    chunk.page_number = 1
                    chunk.section_path = last_section_prefix
                    if last_section_prefix:
                        chunk.text = f"{last_section_prefix} - {chunk.text}"

                pg = chunk.page_number
                if pg not in page_content_cache:
                    page_content_cache[pg] = _extract_page_content(parsed_text, pg)
                chunk.full_content = page_content_cache.get(pg, "")
        else:
            from ingestion.chunking.chunker_factory import get_chunker

            chunker = get_chunker(
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=50,
                similarity_threshold=similarity_threshold,
                text_length=len(parsed_text),
            )
            chunks = chunker.chunk(parsed_text)
            page_mapping = parsed_result.get("page_mapping", [])
            for chunk in chunks:
                if hasattr(chunk, 'start_index') and page_mapping:
                    chunk.page_number = get_page_number_for_position(chunk.start_index, page_mapping)
                else:
                    chunk.page_number = 1
                chunk.section_path = ""
                chunk.full_content = parsed_text

        print(f"Created {len(chunks)} chunks from {filename}")
        return chunks

    async def embed_chunks(
        self,
        chunks: List[Any],
        document_id: str,
        chunk_size: int,
        similarity_threshold: float,
        filename: str,
        file_type: str,
        file_size: int,
        parser_used: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Clean, embed, and store chunks in the vector DB.
        """
        valid_chunks = []
        invalid_chunks = 0

        for chunk in chunks:
            text = getattr(chunk, 'text', None)
            if text is None or (isinstance(text, str) and len(text.strip()) == 0):
                invalid_chunks += 1
                logfire.warn("Skipping chunk with None or empty text",
                             chunk_info=str(chunk)[:100])
            else:
                valid_chunks.append(chunk)

        if invalid_chunks > 0:
            logfire.warn("Filtered out invalid chunks during processing",
                         total_chunks=len(chunks),
                         invalid_chunks=invalid_chunks,
                         valid_chunks=len(valid_chunks))

        if not valid_chunks:
            raise ValueError(
                f"No valid chunks created from document {document_id}. "
                f"All {len(chunks)} chunks had None or empty text.")

        text_cleaner = TextCleaningPipeline()
        chunk_texts = []
        for chunk in valid_chunks:
            text = getattr(chunk, 'text', '')
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            elif not isinstance(text, str):
                text = str(text)
            text = text_cleaner.clean(text, log_steps=False)
            chunk.text = text
            chunk_texts.append(text)

        logfire.info("Extracted chunk texts for embedding",
                     num_chunks=len(chunk_texts),
                     sample_types=[type(t).__name__ for t in chunk_texts[:3]])

        logfire.info("Stage: generating embeddings", num_chunks=len(chunk_texts))
        print("Generating embeddings...")
        try:
            embeddings = self.embedding_generator.embed_batch(chunk_texts)
        except Exception as e:
            tb = traceback.format_exc()
            logfire.error("Stage FAILED: embedding generation",
                          error_type=type(e).__name__, error=str(e), traceback=tb)
            print(f"[EMBED ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
            raise

        logfire.info("Stage: embeddings generated", num_embeddings=len(embeddings))
        chunks = valid_chunks

        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            page_number = getattr(chunk, 'page_number', 1)
            raw_page_content = getattr(chunk, 'full_content', '') or ''

            if len(raw_page_content) > 10000:
                logfire.warn("Large page_content in metadata",
                             page_number=page_number,
                             char_count=len(raw_page_content))

            chunk_metadata = {
                'chunk_index': i,
                'token_count': getattr(chunk, 'token_count', None),
                'start_index': getattr(chunk, 'start_index', None),
                'end_index': getattr(chunk, 'end_index', None),
                'page_number': page_number,
                'section_path': getattr(chunk, 'section_path', ''),
                'page_content': raw_page_content,
                'full_content': raw_page_content,
                'chunk_size': chunk_size,
                'similarity_threshold': similarity_threshold,
                'embedding_model': self.embedding_generator.model_name,
                'embedding_dimension': len(embedding),
                'filename': filename,
                'file_type': file_type,
                'file_size': file_size,
                'parser_used': parser_used,
            }

            if metadata:
                chunk_metadata.update(metadata)

            chunk_objects.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_metadata,
            ))

        logfire.info("Stage: inserting chunks into DB",
                     num_chunks=len(chunk_objects),
                     table_name=self.vector_store.table_name)
        print("Inserting chunks into database using pgvector...")
        try:
            await self.vector_store.add_chunks(chunk_objects)
        except Exception as e:
            tb = traceback.format_exc()
            logfire.error("Stage FAILED: DB insertion",
                          num_chunks=len(chunk_objects),
                          table_name=self.vector_store.table_name,
                          error_type=type(e).__name__, error=str(e), traceback=tb)
            print(f"[INSERT ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
            raise

        logfire.info("Stage: DB insertion complete",
                     num_chunks=len(chunk_objects),
                     document_id=document_id)

    async def ingest_document(self, file_path: str, chunk_size: int = 512,
                              similarity_threshold: float = 0.5,
                              document_id: str = None, metadata: Dict = None,
                              chunker_type: str = None,
                              parse_backend: str = "") -> str:
        """
        Backward-compatible wrapper: parse, chunk, embed, and store.

        Args:
            file_path: Path to the document file
            chunk_size: Maximum tokens per chunk
            similarity_threshold: Similarity threshold for chunking
            document_id: Optional document ID (if None, will generate one)
            metadata: Additional metadata for the document

        Returns:
            Document ID
        """
        file_path = Path(file_path)
        filename = file_path.name

        if document_id is None:
            document_id = str(uuid.uuid4())

        print(f"Processing document: {filename} (ID: {document_id})")

        parsed = await self.parse_file(
            file_path=str(file_path),
            document_id=document_id,
            parse_backend=parse_backend,
        )
        chunks = self.chunk_parsed_document(
            parsed,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            chunker_type=chunker_type,
        )
        await self.embed_chunks(
            chunks=chunks,
            document_id=document_id,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            filename=parsed["filename"],
            file_type=parsed["file_type"],
            file_size=parsed["file_size"],
            parser_used=parsed["parser_used"],
            metadata=metadata,
        )

        print(f"Successfully processed {filename} -> Document ID: {document_id}")
        return document_id

    async def search_documents(self, query: str, limit: int = 5, threshold: float = 0.5,
                               document_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for relevant document chunks using pgvector.
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of relevant chunks
        """
        query_embedding = self.embedding_generator.embed_text(query)
        return await self.vector_store.search_similar_chunks(
            query_embedding, limit, threshold, document_ids
        )

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.vector_store.delete_document_chunks(document_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return await self.vector_store.get_collection_stats()
