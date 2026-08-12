# Chunk Context Enrichment

## Overview

This project enriches each chunk with two pieces of structural context before it is embedded and stored:

1. **Section / sub-section prefix** — the heading hierarchy (H1 → H2 → H3) that contains the chunk.
2. **Full page content** — the complete text of the page the chunk came from.

These enrichments are added during ingestion and consumed during retrieval to improve answer quality, citation accuracy, and handling of structural questions.

---

## Pipeline

```
PDF → Markdown converter → chunks → enrich with section prefix + page content → embed → pgvector
```

Key files:

- `ingestion/embedding/vector_store.py` — adds section prefix and page content during ingestion.
- `ingestion/chunking/chunker_factory.py` — `chunk_markdown()` preserves `[Page N]` markers produced by the PDF-to-Markdown parser for page extraction.
- `retrieval/search.py` — builds the LLM prompt from chunks, section context, and full page content.
- `config/app_config.py` — system prompt instructs the LLM how to use each block.

---

## 1. Section / Sub-section Prefix

### What it does

For every chunk, the code walks backwards through the Markdown to find the current heading hierarchy and builds a prefix such as:

```
[Chapter 1].[Section 2].[Subsection A]
```

This prefix is:

- prepended to the chunk text before embedding and storage, and
- stored in metadata as `section_path`.

### Code location

`ingestion/embedding/vector_store.py`, lines 675–690:

```python
section_prefix = _extract_section_hierarchy(markdown, chunk.start_index)
if not section_prefix:
    section_prefix = last_section_prefix
else:
    last_section_prefix = section_prefix

chunk.section_path = section_prefix
if section_prefix:
    chunk.text = f"{section_prefix} - {chunk.text}"
```

### Purpose

- **Disambiguation**: tells the LLM which part of the document the chunk belongs to.
- **Citations**: enables answers like "According to Section 2.3, …".
- **Sibling expansion**: structural queries (`how many`, `list all`, `count`, `enumerate`, etc.) use `section_path` to retrieve all chunks in the same section, reducing fragmentation.

### Sibling expansion in retrieval

`retrieval/search.py`, lines 128–153:

```python
if _STRUCTURAL_RE.search(query):
    siblings = await pipeline.vector_store.get_chunks_by_section(
        section_path=sp,
        document_ids=[doc_id],
        limit=15,
    )
```

This is triggered by queries such as:

- "How many steps are in Section 3?"
- "List all requirements in Chapter 2."
- "Summarize all subsections under Security."

---

## 2. Full Page Content in Metadata

### What it does

After chunking, the full text of the chunk’s page is extracted from `[Page N]` markers and stored in metadata:

```python
chunk_metadata = {
    'page_content': raw_page_content,
    'full_content': raw_full_content,
    ...
}
```

### Code location

`ingestion/embedding/vector_store.py`, lines 692–695 and 797–811:

```python
pg = chunk.page_number
if pg not in page_content_cache:
    page_content_cache[pg] = _extract_page_content(markdown, pg)
```

### Purpose

- **Surrounding context**: the matched chunk may be short; the full page helps the model resolve pronouns, acronyms, and references defined earlier on the same page.
- **Completeness**: the system prompt explicitly instructs the LLM to consult `[Full page context]` when the chunk alone is insufficient.
- **Avoids false negatives**: the model is less likely to say "I cannot find it" when the answer spans multiple chunks on the same page.

### Deduplication at query time

The retrieval layer includes each page context only once per `(document_id, page_number)` pair to avoid bloating the prompt:

```python
page_key = (doc_id, page_num if page_num is not None else 'no_page')
if (
    page_content
    and page_content.strip() != chunk_text.strip()
    and page_key not in seen_page_contexts
):
    source_block = (
        f"[Source {i+1}{page_info}]\n"
        f"[Matched chunk]: {chunk_text}\n"
        f"[Full page context]:\n{page_content}"
    )
    seen_page_contexts.add(page_key)
```

---

## Final Prompt Structure

During query time, each source block sent to the LLM looks like this:

```
[Source N (Page P)]
[Matched chunk]: [H1].[H2] - relevant passage
[Full page context]:
  ...entire page text...
```

The system prompt (`config/app_config.py`, lines 140–166) tells the LLM to:

1. Read the `[Matched chunk]` first.
2. Consult `[Full page context]` for surrounding information.
3. Use the section prefix to identify which part of the document the chunk belongs to.
4. Cite sources and page numbers whenever available.

---

## Pros and Cons

### Pros

- **Better answer quality**: the LLM can resolve references, acronyms, and pronouns that appear earlier on the same page but outside the retrieved chunk.
- **More accurate citations**: the section prefix and page number make it easy for the model to cite "Section 2.3, Page 7" instead of guessing.
- **Reduced fragmentation**: sibling expansion lets structural questions (`list all`, `how many`, `count`) gather all chunks under the same heading instead of relying on a single chunk.
- **Lower hallucination**: the system prompt explicitly tells the model to use `[Full page context]`, so it is less likely to claim information is missing when it is present elsewhere on the page.
- **No extra model calls**: section and page extraction are rule-based operations done during ingestion, so query latency is not increased by enrichment itself.
- **Retrieval stays fast**: embeddings are computed on the original chunk text plus a short prefix; page content is only added to the prompt at query time.

### Cons

- **Larger prompt size**: including full page content for every unique page can consume significant context window, especially for dense documents.
- **Higher token cost**: larger prompts mean more input tokens per LLM call, which increases cost and can slow generation.
- **Metadata bloat**: every chunk stores `page_content` and `full_content` in JSONB; this increases storage and can slow large metadata reads.
- **Diminishing returns for small chunks**: if `chunk_size` is small, many chunks share the same page, so the same page context is emitted repeatedly unless deduplication is applied.
- **Quality depends on source format**: PDFs with `[Page N]` markers work best; DOCX/TXT use estimated page mapping, and plain Markdown without markers gets little or no page context.
- **Risk of distracting the model**: very large page contexts can bury the actual matched chunk, especially if the LLM has limited attention to long contexts.
- **Harder to tune**: balancing chunk size, page size, and context window requires experimentation for each document type.

---

## Recommendations

- Keep the default enrichment enabled; it is the main reason the RAG pipeline can answer questions that span multiple nearby chunks.
- If page content is consistently very large, consider:
  - increasing `chunk_size` so fewer chunks share the same page, or
  - truncating `page_content` to a fixed character limit instead of storing the whole page.
- For documents without `[Page N]` markers (plain Markdown, DOCX, TXT), `page_content` falls back to empty or estimated page text; quality depends on the source format.
