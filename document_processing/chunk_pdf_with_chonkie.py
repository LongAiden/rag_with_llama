import PyPDF2
from docx import Document
from chonkie import SemanticChunker


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


def extract_text_from_docx(docx_path):
    """
    Extract text from DOCX file using python-docx.
    Args:
        docx_path (str): Path to the DOCX file
    Returns:
        str: Extracted text from all paragraphs and tables
    """
    try:
        doc = Document(docx_path)
        text = ""

        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"

        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        raise ValueError(f"Failed to extract text from DOCX file: {e}")


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


def process_document(file_path, chunk_size=512, similarity_threshold=0.5, embedding_model=None):
    """
    Process a document (PDF, DOCX, or TXT) and return chunks.
    Args:
        file_path (str): Path to the document file
        chunk_size (int): Maximum tokens per chunk
        similarity_threshold (float): Similarity threshold for semantic chunking
        embedding_model: Custom embedding model
    Returns:
        list: List of chunks
    """
    from pathlib import Path

    file_path = Path(file_path)
    file_type = file_path.suffix.lower().replace('.', '')

    print(f"Processing {file_type.upper()} file: {file_path.name}")

    # Extract text based on file type
    if file_type == 'pdf':
        text = extract_text_from_pdf(str(file_path))
    elif file_type == 'docx':
        text = extract_text_from_docx(str(file_path))
    elif file_type == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(
            f"Unsupported file type: {file_type}. Supported types: PDF, DOCX, TXT")

    print(f"Extracted {len(text)} characters from {file_path.name}")

    # Chunk the text
    chunks = chunk_with_semantic_chunker(
        text, chunk_size, similarity_threshold, embedding_model)
    print(f"Created {len(chunks)} chunks")

    return chunks
