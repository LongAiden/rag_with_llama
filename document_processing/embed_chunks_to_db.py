import sys
import os
import json
import uuid
import psycopg2
import numpy as np
import PyPDF2
from pathlib import Path

from pgvector.psycopg2 import register_vector
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import your existing chunking functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


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
        self._initialize_database()

    def _get_connection(self):
        """Get database connection with pgvector support."""
        conn = psycopg2.connect(**self.connection_params)
        # Register pgvector types
        register_vector(conn)
        return conn

    def _initialize_database(self):
        """Initialize database with pgvector extension and table."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create table with proper vector column
                # Assuming 384-dimensional embeddings for all-MiniLM-L6-v2
                # Adjust dimension based on your model
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(384),  -- Adjust dimension as needed
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # Create index for similarity search
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
                """)

                # Create index on document_id for filtering
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_document_id_idx 
                ON {self.table_name} (document_id);
                """)

            conn.commit()
            conn.close()
            print(f"Database initialized with table: {self.table_name}")

        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """Add chunks to vector store using batch insert for efficiency."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # Prepare data for batch insert
                chunk_data = []
                for chunk in chunks:
                    chunk_data.append((
                        chunk.id,
                        chunk.document_id,
                        chunk.text,
                        # Convert to numpy array for pgvector
                        np.array(chunk.embedding),
                        json.dumps(
                            chunk.metadata) if chunk.metadata else json.dumps({})
                    ))

                # Use execute_values for efficient batch insert
                insert_sql = f"""
                INSERT INTO {self.table_name} (id, document_id, text, embedding, metadata)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
                """

                # Process in batches
                for i in range(0, len(chunk_data), batch_size):
                    batch = chunk_data[i:i + batch_size]
                    execute_values(
                        cur, insert_sql, batch,
                        template=None, page_size=batch_size
                    )

            conn.commit()
            conn.close()
            print(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            print(f"Error adding chunks: {e}")
            raise

    def search_similar_chunks(self, query_embedding: List[float],
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
            conn = self._get_connection()
            with conn.cursor() as cursor:
                # Convert query embedding to numpy array
                query_vector = np.array(query_embedding)

                # Build query with optional document filtering
                base_query = f"""
                    SELECT
                        id,
                        text,
                        metadata,
                        document_id,
                        (1 - (embedding <=> %s)) as similarity
                    FROM {self.table_name}
                    WHERE (1 - (embedding <=> %s)) >= %s
                """

                params = [query_vector, query_vector, threshold]

                if document_ids:
                    base_query += " AND document_id = ANY(%s)"
                    params.append(document_ids)

                base_query += """
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """
                params.extend([query_vector, limit])

                cursor.execute(base_query, params)

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'chunk_id': row[0],
                        'text': row[1],
                        'metadata': row[2] if isinstance(row[2], (dict, type(None))) else json.loads(row[2]),
                        'document_id': row[3],
                        'similarity': float(row[4])
                    })

                conn.close()
                return results

        except Exception as e:
            print(f"Error searching chunks: {e}")
            raise

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE document_id = %s", (document_id,))
                deleted_count = cur.rowcount
            conn.commit()
            conn.close()
            print(
                f"Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT document_id) as total_documents,
                    AVG(LENGTH(text)) as avg_text_length,
                    MIN(created_at) as earliest_chunk,
                    MAX(created_at) as latest_chunk
                FROM {self.table_name}
                """)

                row = cur.fetchone()
                stats = {
                    'total_chunks': row[0],
                    'total_documents': row[1],
                    'avg_text_length': float(row[2]) if row[2] else 0,
                    'earliest_chunk': row[3].isoformat() if row[3] else None,
                    'latest_chunk': row[4].isoformat() if row[4] else None
                }

            conn.close()
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
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

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 512,
                   similarity_threshold: float = 0.5) -> List:
        """
        Chunk text using Chonkie SemanticChunker.

        Args:
            text: Input text to chunk
            chunk_size: Maximum tokens per chunk
            similarity_threshold: Similarity threshold for semantic chunking

        Returns:
            List of chunks
        """
        # Use model name string instead of SentenceTransformer object
        chunker = SemanticChunker(
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            embedding_model=self.embedding_generator.model_name
        )
        return chunker.chunk(text)

    def process_document(self, file_path: str, chunk_size: int = 512,
                         similarity_threshold: float = 0.5,
                         document_id: str = None, metadata: Dict = None) -> str:
        """
        Process a document: extract text, chunk, embed, and store.

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
        file_type = file_path.suffix.lower().replace('.', '')
        file_size = file_path.stat().st_size

        # Generate document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        print(f"Processing document: {filename} (ID: {document_id})")

        # Extract text based on file type
        if file_type == 'pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_type in ['txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        print(f"Extracted {len(text)} characters from {filename}")

        # Chunk the text
        chunks = self.chunk_text(text, chunk_size, similarity_threshold)
        print(f"Created {len(chunks)} chunks")

        # Prepare chunks for embedding
        chunk_texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batch
        print("Generating embeddings...")
        embeddings = self.embedding_generator.embed_batch(chunk_texts)

        # Create Chunk objects using your interface
        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                'chunk_index': i,
                'token_count': chunk.token_count,
                'start_index': getattr(chunk, 'start_index', None),
                'end_index': getattr(chunk, 'end_index', None),
                'chunk_size': chunk_size,
                'similarity_threshold': similarity_threshold,
                'embedding_model': self.embedding_generator.model_name,
                'embedding_dimension': len(embedding),
                'filename': filename,
                'file_type': file_type,
                'file_size': file_size
            }

            # Add any additional metadata
            if metadata:
                chunk_metadata.update(metadata)

            chunk_obj = Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_metadata
            )
            chunk_objects.append(chunk_obj)

        # Use your add_chunks method with pgvector
        print("Inserting chunks into database using pgvector...")
        self.vector_store.add_chunks(chunk_objects)

        print(
            f"Successfully processed {filename} -> Document ID: {document_id}")
        return document_id

    def search_documents(self, query: str, limit: int = 5, threshold: float = 0.7,
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
        return self.vector_store.search_similar_chunks(
            query_embedding, limit, threshold, document_ids
        )

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return self.vector_store.delete_document_chunks(document_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_collection_stats()
