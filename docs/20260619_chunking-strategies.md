# Chunking Strategies Guide

## Overview

This project processes documents through a two-step pipeline before chunking:

```
input/pdf/ → Parser (choose one) → input/markdown/ → Chunker → Vector DB
               • PyMuPDF (default)
               • Docling + Ollama VLM (local)
               • Docling + Gemini Vision (cloud)
```

Since the chunker always receives **Markdown text** (not raw PDF text), the default chunker is the **MarkdownChunker**, which is aware of markdown structure (headings, lists, code blocks, etc.).

---

## Available Strategies

### 1. MarkdownChunker (DEFAULT) 📄
**Best for markdown input - structure-aware splitting**

- **Speed**: Fast
- **Quality**: Excellent - respects markdown hierarchy (headings, sections, lists)
- **Use when**: Default for all documents processed through PDFToMarkdownConverter

Splits on markdown-specific boundaries in order:
1. `#` / `##` / `###` headings
2. Code blocks (` ``` `)
3. Paragraphs (`\n\n`)
4. List items
5. Line breaks

### 2. RecursiveChunker ⚡
**Balanced - generic text boundary splitting**

- **Speed**: Fast (10–20 seconds for 15 MB)
- **Quality**: Good - respects natural text boundaries (paragraphs, sentences)
- **Use when**: Non-markdown input or when markdown structure should be ignored

Splits on natural boundaries in order:
1. Paragraphs (`\n\n`)
2. Line breaks (`\n`)
3. Sentences (`. `, `! `, `? `)
4. Clauses (`, `, `; `)
5. Words (` `)
6. Characters (fallback)

### 3. TokenChunker 🚀
**Fastest - simple token-based splitting**

- **Speed**: Very fast (3–5 seconds for 15 MB)
- **Quality**: Predictable chunk sizes; ignores text structure
- **Use when**: Speed is critical, chunk boundary quality is not important

Simple character/token counting with overlap.

### 4. SemanticChunker 🧠
**Highest quality - AI-powered semantic boundaries**

- **Speed**: Very slow (5–30 minutes for 15 MB)
- **Quality**: Best possible - finds topic changes in embedding space
- **Use when**: Small documents (< 500 KB) where quality matters most

Uses sentence embeddings to detect semantic topic transitions.
**Note**: Automatically falls back to RecursiveChunker for documents > 100 KB.

---

## How to Use

### Method 1: Environment Variable (Global Setting)

```bash
# .env
CHUNKER_TYPE=markdown   # Options: markdown (default), recursive, token, semantic
```

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - CHUNKER_TYPE=markdown
```

### Method 2: API Parameter (Per Request)

```bash
# Use default (MarkdownChunker)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=512"

# Use TokenChunker (fastest)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "chunker_type=token"

# Use RecursiveChunker
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "chunker_type=recursive"

# Use SemanticChunker (slow, best quality)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "chunker_type=semantic"
```

### Method 3: Python Code

```python
from ingestion.chunking.chunker_factory import chunk_markdown, get_chunker

# Use default (markdown-aware)
chunks = chunk_markdown(markdown_text, chunk_size=512)

# Use specific strategy
chunks = chunk_markdown(markdown_text, chunker_type="token")
chunks = chunk_markdown(markdown_text, chunker_type="recursive")
chunks = chunk_markdown(markdown_text, chunker_type="semantic")
```

---

## Performance Comparison

| Document Size | Token | Recursive | Markdown | Semantic |
|--------------|-------|-----------|----------|----------|
| 1 MB PDF     | 1 s   | 3 s       | 3 s      | 2 min    |
| 5 MB PDF     | 3 s   | 10 s      | 10 s     | 10 min   |
| 15 MB PDF    | 5 s   | 20 s      | 20 s     | 30 min   |

*Times include PDF → Markdown conversion. SemanticChunker auto-falls-back to Recursive above 100 KB.*

---

## Recommendations

### 📚 Large Documents (> 5 MB)
**Use: TokenChunker** (`chunker_type=token`)
- Fastest, good quality for most RAG tasks

### 📄 Most Documents - Markdown Input (DEFAULT)
**Use: MarkdownChunker** (default, no parameter needed)
- Best for documents converted via PDFToMarkdownConverter
- Preserves section structure for better retrieval
- **RECOMMENDED for this project's standard workflow**

### 📝 Small Documents (< 500 KB) - Quality Critical
**Use: SemanticChunker** (`chunker_type=semantic`)
- Perfect semantic boundaries
- Very slow on larger documents

---

## Advanced Configuration

```python
from ingestion.chunking.chunker_factory import get_chunker

# Markdown (default): Structure-aware
chunker = get_chunker("markdown", chunk_size=512)

# Token: Adjust overlap
chunker = get_chunker("token", chunk_size=512, chunk_overlap=100)

# Recursive: Same parameters
chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=50)

# Semantic: Adjust threshold (lower = larger chunks, faster)
chunker = get_chunker("semantic", chunk_size=512, similarity_threshold=0.3)

# Adaptive: auto-downgrade to Recursive if text is large
chunker = get_chunker("semantic", text_length=len(markdown_text))
```

---

## Troubleshooting

### Document taking too long to process?
→ Switch to `token` or `recursive` chunker

### Want better section boundaries?
→ Use `markdown` (default) or `semantic` (for small docs)

### Need consistent chunk sizes?
→ Use `token` chunker

### SemanticChunker falling back to Recursive?
→ Expected for documents > 100 KB - this is automatic performance protection

### Default chunker not working?
→ Check `CHUNKER_TYPE` environment variable
→ Restart services after changing env vars
