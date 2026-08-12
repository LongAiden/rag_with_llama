"""
Unit test specific fixtures.

These fixtures are used for unit tests that don't require external dependencies
like databases or API connections.
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_pdf_path(temp_dir):
    """Create a minimal valid PDF file for testing."""
    pdf_path = temp_dir / "valid.pdf"
    # Minimal valid PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF
"""
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


@pytest.fixture
def fake_pdf_path(temp_dir):
    """Create a fake PDF file (text file with .pdf extension)."""
    fake_pdf = temp_dir / "fake.pdf"
    fake_pdf.write_text("This is not a real PDF file, just plain text.")
    return str(fake_pdf)


@pytest.fixture
def valid_txt_path(temp_dir):
    """Create a valid text file for testing."""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text("This is a sample text file for testing purposes.")
    return str(txt_path)


@pytest.fixture
def large_text():
    """Generate a large text for chunking tests (>100KB)."""
    sentence = "This is a sample sentence for testing text chunking functionality. "
    # Generate ~150KB of text
    return sentence * 3000  # ~150KB


@pytest.fixture
def small_text():
    """Generate a small text for chunking tests (<100KB)."""
    return "This is a small text sample for testing. " * 50  # ~2KB


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for unit tests."""
    mock = MagicMock()
    # Mock encode to return a list of 384-dimension vectors
    mock.encode.return_value = [[0.1] * 384]
    mock.get_sentence_embedding_dimension.return_value = 384
    return mock


@pytest.fixture
def sample_chunk_texts():
    """Sample texts for embedding batch tests."""
    return [
        "First chunk of text for embedding.",
        "Second chunk with different content.",
        "Third chunk about machine learning.",
        "Fourth chunk discussing databases.",
        "Fifth chunk about retrieval systems."
    ]
