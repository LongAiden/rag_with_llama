#!/usr/bin/env python3
"""
Example usage of the chunk embedding pipeline.
Simple script to demonstrate how to use embed_chunks_to_db.py
"""

import os
from datetime import datetime
from embed_chunks_to_db import ChunkEmbeddingPipeline

def main():
    """Simple example of processing a PDF and storing embeddings."""

    # Database connection parameters - UPDATE THESE!
    db_params = {
        'host': 'localhost',
        'port': '5432',
        'dbname': 'rag_db',
        'user': 'admin',
        'password': 'your_password'  # Update with your actual password
    }

    print("🚀 Starting RAG Database Pipeline...")

    # Initialize the pipeline
    try:
        pipeline = ChunkEmbeddingPipeline(
            db_params=db_params,
            embedding_model='all-MiniLM-L6-v2'  # Fast, lightweight model
        )
        print("✅ Pipeline initialized successfully!")

    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("Make sure PostgreSQL is running and connection parameters are correct.")
        return

    try:
        # Process the Llama2 PDF
        pdf_path = "../docs/llama2.pdf"

        if os.path.exists(pdf_path):
            print(f"\n📄 Processing PDF: {pdf_path}")

            document_id = pipeline.process_document(
                file_path=pdf_path,
                chunk_size=512,              # Moderate chunk size
                similarity_threshold=0.5,    # Balanced threshold
                metadata={
                    'source': 'llama2_research_paper',
                    'category': 'AI/ML',
                    'processed_by': 'chonkie_semantic_chunker',
                    'processing_date': datetime.now().isoformat()
                }
            )

            print(f"✅ Document processed! ID: {document_id}")

            # Test search functionality
            print("\n🔍 Testing search functionality...")

            # Example queries
            test_queries = [
                "What is Llama 2?",
                "training methodology",
                "model architecture",
                "performance evaluation"
            ]

            for query in test_queries:
                print(f"\nQuery: '{query}'")
                print("-" * 40)

                results = pipeline.search_documents(
                    query=query,
                    limit=2,           # Top 2 results
                    threshold=0.3      # Lower threshold for more results
                )

                if results:
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. Similarity: {result['similarity']:.3f}")
                        print(f"     Text: {result['text'][:150]}...")
                        print()
                else:
                    print("  No results found.")

        else:
            print(f"❌ PDF file not found: {pdf_path}")
            print("Please check the file path.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

    finally:
        pipeline.close()
        print("\n🏁 Pipeline closed successfully!")

if __name__ == "__main__":
    main()