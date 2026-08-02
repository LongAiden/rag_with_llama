"""
Integration test fixtures.

These fixtures require external dependencies like databases and API connections.
"""
import os
import sys
import pytest
import asyncio
import asyncpg
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


@pytest.fixture(scope="session")
def db_params():
    """Database connection parameters from environment variables."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'rag_db'),
        'user': os.getenv('POSTGRES_USER', 'admin'),
        'password': os.getenv('POSTGRES_PASSWORD', 'admin')
    }


@pytest.fixture(scope="session")
def gemini_api_key():
    """Gemini API key from environment variables."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-google-api-key-here':
        pytest.skip("GOOGLE_API_KEY not configured in .env file")
    return api_key


@pytest.fixture(scope="session")
def gemini_model():
    """Gemini model name from environment variables."""
    return os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')


@pytest.fixture(scope="session")
def logfire_token():
    """Logfire write token from environment variables."""
    return os.getenv('LOGFIRE_WRITE_TOKEN', '')


class FakeEmbeddingModel:
    """Deterministic fake embedding model for integration tests.

    Replaces a real SentenceTransformer so retrieval tests are fast and
    reproducible. Vectors are deterministically derived from the input text.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        result = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.random(self.embedding_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            result.append(vec)

        if single:
            return result[0]
        return np.array(result)

    def get_sentence_embedding_dimension(self):
        return self.embedding_dim


@pytest.fixture(scope="session")
def embedding_model():
    """Shared deterministic embedding model for tests."""
    return FakeEmbeddingModel(embedding_dim=384)


@pytest.fixture(scope="session")
def embedding_dim():
    """Embedding dimension used by the fake model."""
    return 384


@pytest.fixture
async def db_connection(db_params):
    """Create a database connection for a single test."""
    conn = await asyncpg.connect(**db_params)
    try:
        # Ensure pgvector extension is enabled
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        yield conn
    finally:
        await conn.close()


@pytest.fixture
async def test_table_name():
    """Generate a unique test table name."""
    import uuid
    return f"test_chunks_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def cleanup_test_table(db_connection, test_table_name):
    """Cleanup fixture that drops the test table after the test."""
    yield
    # Cleanup: drop test table
    try:
        await db_connection.execute(f"DROP TABLE IF EXISTS {test_table_name};")
    except Exception as e:
        print(f"Warning: Could not drop test table {test_table_name}: {e}")

