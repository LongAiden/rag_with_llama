#!/usr/bin/env python3
"""
Embed Chunks and Insert to RAG Database
This script embeds text chunks and inserts them into a PostgreSQL database with pgvector.

Features:
- Load chunks from various sources (files, direct text)
- Generate embeddings using SentenceTransformers
- Insert chunks with embeddings into PostgreSQL with pgvector
- Support for metadata storage
- Batch processing for efficiency

Requirements:
- pip install psycopg2-binary pgvector sentence-transformers chonkie PyPDF2
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from chonkie import SemanticChunker
import PyPDF2

# Import your existing chunking functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logger = logging.getLogger(__name__)


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
    """Vector store using your existing add_chunks method interface."""

    def __init__(self, connection_params: Dict[str, str], table_name: str = "chunks"):
        """
        Initialize vector store.
        Args:
            connection_params: Database connection parameters
            table_name: Name of the chunks table
        """
        self.connection_params = connection_params
        self.table_name = table_name

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.connection_params)

    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to vector store using your existing method"""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                for chunk in chunks:
                    insert_sql = f"""
                    INSERT INTO {self.table_name} (id, document_id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """
                    cur.execute(insert_sql, (
                        chunk.id,
                        chunk.document_id,
                        chunk.text,
                        chunk.embedding,
                        chunk.metadata or {}
                    ))
            conn.commit()
            conn.close()
            logger.info(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise

    def search_similar_chunks(self, query_embedding: List[float],
                              limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of similar chunks with metadata
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        id,
                        text,
                        metadata,
                        document_id,
                        (1 - (embedding <=> %s::vector)) as similarity
                    FROM {self.table_name}
                    WHERE (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, threshold, query_embedding, limit))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'chunk_id': row[0],
                        'text': row[1],
                        'metadata': row[2],
                        'document_id': row[3],
                        'similarity': float(row[4])
                    })

                conn.close()
                return results

        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise


class ChunkEmbeddingPipeline:
    """Complete pipeline for chunking documents and storing embeddings."""

    def __init__(self, db_params: Dict[str, str], embedding_model: str = 'all-MiniLM-L6-v2',
                 table_name: str = "chunks"):
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

        # Use your add_chunks method
        print("Inserting chunks into database using add_chunks method...")
        self.vector_store.add_chunks(chunk_objects)

        print(
            f"Successfully processed {filename} -> Document ID: {document_id}")
        return document_id

    def search_documents(self, query: str, limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of relevant chunks
        """
        query_embedding = self.embedding_generator.embed_text(query)
        return self.vector_store.search_similar_chunks(query_embedding, limit, threshold)


def main():
    """Example usage of the chunk embedding pipeline."""

    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': '5432',
        'dbname': 'rag_db',
        'user': 'admin',  # Replace with your DB user
        'password': 'your_password'  # Replace with your DB password
    }

    # Initialize pipeline
    pipeline = ChunkEmbeddingPipeline(
        db_params=db_params,
        embedding_model='all-MiniLM-L6-v2',
        table_name='chunks'  # Your existing table name
    )

    try:
        # Process the PDF document
        pdf_path = "../docs/llama2.pdf"

        if os.path.exists(pdf_path):
            document_id = pipeline.process_document(
                file_path=pdf_path,
                chunk_size=512,
                similarity_threshold=0.5,
                document_id=None,  # Will generate new UUID
                metadata={
                    'source': 'llama2_paper',
                    'processed_by': 'chonkie_pipeline',
                    'processing_date': datetime.now().isoformat()
                }
            )

            print(
                f"\nDocument processed successfully! Document ID: {document_id}")

            # Test search functionality
            print("\nTesting search functionality...")
            query = "What is Llama 2?"
            results = pipeline.search_documents(query, limit=3, threshold=0.3)

            print(f"\nSearch results for: '{query}'")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(
                    f"\nResult {i} (Similarity: {result['similarity']:.3f}):")
                print(f"Document ID: {result['document_id']}")
                print(f"Text: {result['text'][:200]}...")
                metadata = result.get('metadata', {})
                if isinstance(metadata, dict) and 'filename' in metadata:
                    print(f"File: {metadata['filename']}")
                print("-" * 30)

        else:
            print(f"PDF file not found: {pdf_path}")
            print("Please make sure the file exists or update the path.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
