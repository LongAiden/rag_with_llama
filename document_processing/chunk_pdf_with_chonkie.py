import os
from pathlib import Path
import PyPDF2
from chonkie import SemanticChunker, TokenChunker, WordChunker, SentenceChunker
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file using PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from all pages
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

    return text


def chunk_with_semantic_chunker(text, chunk_size=512, similarity_threshold=0.5, embedding_model=None):
    """
    Chunk text using Chonkie's SemanticChunker with custom embedding model.

    Args:
        text (str): Text to chunk
        chunk_size (int): Maximum tokens per chunk
        similarity_threshold (float): Similarity threshold for semantic chunking
        embedding_model: Custom embedding model (SentenceTransformer or similar)

    Returns:
        list: List of chunks
    """
    if embedding_model:
        chunker = SemanticChunker(
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model
        )
    else:
        # Use default embedding model
        chunker = SemanticChunker(
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold
        )

    chunks = chunker.chunk(text)
    return chunks


def save_chunks_to_file(chunks, output_path, chunker_type):
    """
    Save chunks to a text file.

    Args:
        chunks (list): List of chunks
        output_path (str): Output file path
        chunker_type (str): Type of chunker used
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Chunks created using {chunker_type}\n")
        f.write("=" * 50 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            f.write(f"Chunk {i}:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{chunk.text}\n\n")
            f.write(f"Tokens: {chunk.token_count}\n")
            if hasattr(chunk, 'start_index'):
                f.write(f"Start Index: {chunk.start_index}\n")
            if hasattr(chunk, 'end_index'):
                f.write(f"End Index: {chunk.end_index}\n")
            f.write("\n" + "="*50 + "\n\n")


def main():
    # Path to your PDF file
    pdf_path = "docs/llama2.pdf"

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print("Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters from PDF")

    # Create output directory
    output_dir = "chunked_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 2: Setting up embedding model...")
    # Pass model name as string - Chonkie will load it internally
    embedding_model_name = 'all-MiniLM-L6-v2'
    print(f"  - Using model: {embedding_model_name}")

    print("\nStep 3: Chunking with SemanticChunker and custom embedding model")
    semantic_chunks = chunk_with_semantic_chunker(
        text,
        chunk_size=512,
        similarity_threshold=0.7,
        embedding_model=embedding_model_name
    )
    save_chunks_to_file(
        semantic_chunks, f"{output_dir}/semantic_chunks_custom.txt", "SemanticChunker (Custom Model)")
    print(f"Created {len(semantic_chunks)} semantic chunks with custom model")

    print(f"\nStep 4: All chunks saved to '{output_dir}' directory")

    # Print sample chunks
    print("\nSample chunks:")
    print("="*50)

    if semantic_chunks:
        print("First Semantic Chunk:")
        print(f"Text: {semantic_chunks[0].text[:200]}...")
        print(f"Tokens: {semantic_chunks[0].token_count}")
        print()


if __name__ == "__main__":
    main()
