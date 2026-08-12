# Project Architecture Summary

**Last Updated**: 2026-03-14

---

## Quick Overview

A **RAG (Retrieval-Augmented Generation)** system that:
1. Converts PDFs to Markdown using a selectable parser: PyMuPDF (default), Docling + Ollama VLM (local), or Docling + Gemini Vision (cloud)
2. Chunks the Markdown with structure-aware chunkers
3. Stores chunk embeddings in pgvector
4. Retrieves relevant chunks via vector search + BM25 reranking
5. Generates answers with a selectable LLM: **Gemini 2.5 Flash** (cloud) or **DeepSeek-R1 8B** via Ollama (local)

---

## End-to-End Workflow

```
┌──────────────┐    ┌──────────────────────────────────┐    ┌─────────────┐
│  POST /upload│───▶│ Parser (select via parse_backend) │───▶│ input/      │
│  (PDF file)  │    │ • PyMuPDF (default)               │    │  markdown/  │
│              │    │ • Docling + Ollama VLM (local)    │    │             │
│              │    │ • Docling + Gemini Vision (cloud)  │    │             │
└──────────────┘    └──────────────────────────────────┘    └──────┬──────┘
                                                        │
                    ┌───────────────────────────────────▼──────┐
                    │  MarkdownChunker (chonkie)                │
                    │  Splits on headings, paragraphs, lists    │
                    └───────────────────┬───────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │  EmbeddingGenerator (all-MiniLM-L6-v2)    │
                    │  → stored in PostgreSQL + pgvector         │
                    └───────────────────────────────────────────┘

┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ POST /query  │───▶│ Vector Search        │───▶│ BM25 Reranking   │
│  (question)  │    │ (pgvector cosine)    │    │ (top-5 results)  │
└──────────────┘    └─────────────────────┘    └────────┬─────────┘
                                                         │
                                              ┌──────────▼──────────────────────┐
                                              │  LLM (select via model param)   │
                                              │  • Gemini 2.5 Flash (cloud)     │
                                              │  • DeepSeek-R1 8B / Ollama      │
                                              └──────────┬──────────────────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │  RAGResponse (JSON)  │
                                              │  answer + sources    │
                                              └──────────────────────┘
```

---

## Project Structure

```
rag_with_llama/
│
├── input/                        # 📁 Input/Output files
│   ├── pdf/                      # Source PDFs (e.g. llama2.pdf)
│   └── markdown/                 # Converted Markdown output
│
├── ingestion/                    # 📥 Document Ingestion Pipeline
│   ├── processors/               # File-type processors (Abstract Method Pattern)
│   │   ├── base_processor.py     # Abstract base class
│   │   ├── pdf_processor.py      # PDF → text extraction
│   │   ├── pdf_to_markdown.py    # PDF → Markdown (PDFToMarkdownConverter)
│   │   ├── process_pdf_to_md.py  # Functional helper for PDF→Markdown
│   │   ├── docx_processor.py     # DOCX processing
│   │   ├── txt_processor.py      # TXT processing
│   │   ├── processor_factory.py  # Factory: picks processor by file type
│   │   └── page_utils.py         # Shared page-number utilities
│   ├── chunking/                 # Text Chunking
│   │   ├── chunker_factory.py    # Factory: token/recursive/markdown/semantic
│   │   └── legacy/               # Deprecated chunkers
│   ├── embedding/                # Vector Embeddings
│   │   └── vector_store.py       # ChunkEmbeddingPipeline + pgvector storage
│   ├── text_cleaning/            # Text normalization
│   │   └── cleaners.py
│   ├── extraction/               # (Disabled) Entity extraction helpers
│   │   └── extraction_flow.py
│   └── validation/               # File security validation
│       └── file_validator.py
│
├── retrieval/                    # 🔍 Search & Answer Generation
│   ├── search.py                 # Main search: vector → rerank → LLM
│   ├── reranking.py              # BM25Reranker class
│   ├── llm_operations.py         # LLM response generation (Gemini or Ollama)
│   └── utils.py                  # rerank_bm25() helper
│
├── api/                          # 🌐 FastAPI Application
│   ├── app.py                    # App factory, lifespan, middleware
│   ├── config.py                 # Re-export shim → config/app_config.py
│   ├── validators.py             # Request validation helpers
│   ├── templates.py              # Inline HTML templates
│   └── routes/
│       └── document_routes.py    # Upload, query, stats, health endpoints
│
├── config/                       # ⚙️ Configuration (single source of truth)
│   ├── app_config.py             # AppConfig, AppSettings, DatabaseConfig
│   └── graph_config.py           # Graph settings (currently unused)
│
├── models/                       # 📊 Pydantic Data Models
│   ├── models.py                 # RAGResponse, RAGSource, QueryRequest, etc.
│   └── graph_models.py           # Graph entity/relationship models
│
├── worker/                       # ⚙️ Celery Background Workers
│   ├── celery_app.py             # Celery configuration
│   └── tasks.py                  # Async upload processing task
│
├── graph_processing/             # 🕸️ Knowledge Graph (currently disabled)
│   ├── entity_extraction.py
│   ├── relationship_extraction.py
│   ├── graph_service.py
│   └── clear_graph_data.py       # Utility to wipe graph tables
│
├── tests/                        # 🧪 Tests
│   ├── unit/                     # Isolated unit tests (no DB needed)
│   └── integration/              # Tests requiring PostgreSQL + pgvector
│
├── docs/                         # 📚 Documentation
│   ├── CHUNKING_STRATEGIES.md
│   ├── PROJECT_ARCHITECTURE_SUMMARY.md  ← you are here
│   └── TESTING.md
│
├── docker-compose.yml            # Services: app, postgres, redis, worker, test
└── README.md
```

---

## Module Responsibilities

| Module | Does what | Key entry point |
|--------|-----------|-----------------|
| `ingestion/processors/` | PDF/DOCX/TXT → Markdown/text | `PDFToMarkdownConverter`, `get_processor_for_file()` |
| `ingestion/chunking/` | Markdown → Chunks | `get_chunker()`, `chunk_markdown()` |
| `ingestion/embedding/` | Chunks → Vectors → DB | `ChunkEmbeddingPipeline` |
| `retrieval/search.py` | Query → Vector search → Rerank → LLM | `perform_document_search()` |
| `retrieval/llm_operations.py` | Chunks → LLM answer | `generate_llm_response()` |
| `api/routes/document_routes.py` | HTTP upload & query | `POST /upload`, `POST /query` |
| `config/app_config.py` | All app settings | `AppConfig`, `AppSettings`, `DatabaseConfig` |
| `worker/tasks.py` | Async upload via Celery | `process_upload_task` |

---

## Design Patterns

### 1. Abstract Method + Factory (Processors)

```python
# base_processor.py - Abstract contract
class DocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, file_path) -> Tuple[str, PageMapping]: ...

# processor_factory.py - Factory picks the right one
processor = get_processor_for_file("report.pdf")   # → PDFProcessor
processor = get_processor_for_file("notes.docx")   # → DOCXProcessor
```

### 2. Factory (Chunkers)

```python
# chunker_factory.py
chunker = get_chunker("markdown")   # → MarkdownChunker (DEFAULT)
chunker = get_chunker("token")      # → TokenChunker (fastest)
chunker = get_chunker("semantic")   # → SemanticChunker (best quality)
```

### 3. Lazy Initialization (AppConfig)

```python
# config/app_config.py
config.pipeline = None    # Created on first request
config.reranker = None    # Created on first use
```

### 4. Re-export Shim (backward compatibility)

```python
# api/config.py - keeps old imports working after config moved to config/
from config.app_config import AppConfig, AppSettings, ...
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Home page (HTML upload/search UI) |
| `POST` | `/upload` | Upload + process a document |
| `POST` | `/query` | Ask a question, get RAG answer |
| `GET` | `/stats` | Database statistics |
| `GET` | `/health` | Health check |
| `GET` | `/supported-types` | List accepted file types |
| `DELETE` | `/table/{name}` | Delete a document table |

---

## Configuration

All configuration lives in `config/app_config.py`. Key settings:

| Setting | Env var | Default |
|---------|---------|---------|
| Database host | `POSTGRES_HOST` | `localhost` |
| Database name | `POSTGRES_DB` | `rag_db` |
| Gemini model | `GEMINI_MODEL` | `gemini-2.5-flash` |
| Ollama endpoint | `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` |
| Ollama VLM model | `OLLAMA_VLM_MODEL` | `qwen2.5vl:7b` |
| Embedding model | *(hardcoded)* | `all-MiniLM-L6-v2` |
| Table name | *(param)* | `document_chunks` |
| Chunker type | `CHUNKER_TYPE` | `markdown` |
| Google API key | `GOOGLE_API_KEY` | - *(optional, Gemini only)* |
| Logfire token | `LOGFIRE_WRITE_TOKEN` | - |

---

## Key Data Flow (Code)

```python
# 1. Convert PDF to Markdown
from ingestion.processors.pdf_to_markdown import PDFToMarkdownConverter
converter = PDFToMarkdownConverter()
markdown = converter.convert("input/pdf/llama2.pdf", output="input/markdown/llama2.md")

# 2. Chunk the Markdown
from ingestion.chunking.chunker_factory import chunk_markdown
chunks = chunk_markdown(markdown, chunk_size=512)

# 3. Embed + store
from ingestion.embedding.vector_store import ChunkEmbeddingPipeline
pipeline = ChunkEmbeddingPipeline(db_params, "all-MiniLM-L6-v2", "document_chunks")
doc_id = await pipeline.process_document("input/markdown/llama2.md")

# 4. Query
from retrieval.search import perform_document_search
result = await perform_document_search(
    query="What safety measures does Llama 2 have?",
    limit=10, threshold=0.5, pipeline=pipeline, config=config
)
# result.answer, result.sources, result.search_stats
```

---

## What Is Disabled / Not Yet Active

| Feature | Location | Status |
|---------|----------|--------|
| Knowledge graph (entity extraction) | `graph_processing/` | Disabled - code exists, not wired to upload |
| Graph enrichment in search | `retrieval/search.py` | Removed |
| Graph API routes | `api/routes/graph_routes.py` | Not registered in app.py |
| Celery upload (async) | `worker/tasks.py` | Available but `celery_upload_enabled=False` by default |

---

## For New Developers

**Start here to understand the flow:**
1. [api/routes/document_routes.py](../api/routes/document_routes.py) - `upload_and_process()` and `query_documents()`
2. [ingestion/processors/pdf_to_markdown.py](../ingestion/processors/pdf_to_markdown.py) - PDF conversion
3. [ingestion/chunking/chunker_factory.py](../ingestion/chunking/chunker_factory.py) - Chunking strategies
4. [retrieval/search.py](../retrieval/search.py) - Search + rerank + LLM

**Run locally:**
```bash
docker-compose up -d postgres          # Start database
uvicorn api.app:app --reload           # Start API (http://localhost:8000)
```

**Run tests:**
```bash
pytest tests/unit -v                   # Fast, no DB needed
pytest tests/integration -v            # Requires running postgres
```
