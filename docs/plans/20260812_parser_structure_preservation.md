# F23 — preserve the structure docling already gives us

> Follows F22 (43 DPI → 144 DPI, crop padding re-expressed in points), which is
> complete. Scope chosen by the repo owner: **structure preservation only**.
> Junk removal, OCR fallback/resume, and the memory-headroom decision are
> recorded at the bottom as deliberately deferred.

## Context

F21 and F22 fixed *speed* (1683s → 1080s) and *legibility* (the VLM was reading
43 DPI thumbnails and confabulating). Both left the assembler alone. Reading the
F22 artifact shows the assembler is now the lossy stage: docling classifies the
page correctly and `_process_page` throws most of that classification away.

Measured on `data/parsed/e6b7fabf-…_NLTK.md` (504 pages, 12,378 lines):

| defect | evidence |
|---|---|
| code emitted as prose | 631 `>>>` lines, **0** fenced blocks |
| captions unbound | `Figure 1-1. Downloading the NLTK Book Collection…` is a loose paragraph beside the description |
| VLM output has no boundary | 79 `<figure_type>` tags, **0** surviving `<figure>` wrappers |
| list markers gone | 3 bullet lines, 0 numbered lines in 12,378 |

A direct docling probe of pages 29–32 in the running worker confirms the labels
are there and correct:

```
LABELS p29-32: {'code': 9, 'text': 26, 'picture': 3, 'section_header': 4, 'table': 1, 'caption': 1}
'code'    | '>>> def lexical_diversity(text): ...     return len(text) / len(set(text)) ... >>>'
'caption' | 'Table 1-1. Lexical diversity of various genres in the Brown Corpus'
```

### Root cause

`CodeItem` and `ListItem` both subclass `TextItem`
(`CodeItem → FloatingItem → TextItem`, `ListItem → TextItem`), so the
`elif isinstance(item, TextItem)` branch at
[gemini_docling_parser.py:691](../../src/app/ingestion/processors/gemini_docling_parser.py#L691)
catches them first and emits `item.text` bare. `item.label`, `item.marker`,
`item.enumerated`, `item.code_language` and `item.captions` are never read.

The missing `<figure>` wrapper is the same class of bug that
`_strip_html_wrappers` was written to solve: the prompt asks for the tag, and a
0.8B model does not reliably emit it. Tables get their wrapper added in code
(`md = f"<table>\n\n{md}\n\n</table>"`); figures do not. The consequence is that
a generated description is byte-indistinguishable from book text once chunked,
so a hallucination is retrieved and cited as source.

### Why this matters for retrieval, not just tidiness

The 1.09 MB artifact is over `LARGE_DOCUMENT_THRESHOLD_CHARS` (100 KB), so
`chunker_factory` picks `RecursiveChunker` — generic delimiters, no markdown
awareness. Fences and figure tags do not change chunking *today*. They are worth
adding anyway: they mark provenance in the retrieved text, they stop code and
prose from being embedded as one undifferentiated blob, and they are the
precondition for ever switching a large document to `MarkdownChunker`.

## Approach

All changes are in `_process_page` and its helpers in
`src/app/ingestion/processors/gemini_docling_parser.py`. `OllamaPDFParser`
inherits the whole assembler, so nothing there changes.

### 1. Dispatch on `item.label`, not on `isinstance`

Keep the `elif isinstance(item, TextItem)` branch as the entry point, but branch
inside it on `getattr(item.label, "value", "")`. Dispatching on the label rather
than adding more `isinstance` checks is deliberate: docling's MRO is what caused
this bug, and label dispatch is immune to a future `XItem(TextItem)`.

```python
elif isinstance(item, TextItem):
    label = getattr(item.label, "value", "")
    if label == "code":
        md = _format_code(item)
    elif label == "list_item":
        md = _format_list_item(item)
    else:
        md = (item.text or "").strip()
    if not md:
        continue
    ordered.append((col, y, x, md))
```

### 2. `_format_code` — fence, and restore doctest line breaks

Docling has **already** joined the code block's text cells with spaces by the
time we see it, so fencing alone leaves one long line. For interactive sessions
the `>>>` / `...` prompts *are* the line boundaries and can be restored exactly;
that covers essentially all code in this book. Non-REPL code stays as docling
gave it — fenced but unbroken. State that limit in the docstring rather than
attempting bbox-cell reconstruction.

```python
_DOCTEST_SPLIT_RE = re.compile(r"\s+(?=(?:>>>|\.\.\.)\s)")
```

Applied to the probe's item this yields the original four lines, indentation
included, because the split consumes only the whitespace *before* the prompt.
Strip the trailing bare `>>>` that docling leaves on most blocks.
`code_language` is a `CodeLanguageLabel` enum — take `.value`, lowercase it, and
treat `unknown` as no language.

### 3. `_format_list_item` — use the marker docling parsed

`ListItem` carries `marker` and `enumerated`. Prefix with `item.marker` when
present, else `1.` for enumerated and `-` otherwise.

### 4. Bind captions to their figure and table

`FloatingItem.caption_text(doc)` returns the joined caption for a `PictureItem`
or `TableItem`. Two changes:

- Emit the caption **inside** the block it belongs to.
- Suppress the same text from being emitted again as a loose paragraph. Collect
  `{ref.cref for it in items for ref in getattr(it, "captions", [])}` in the
  pre-pass that already builds `adjacent_texts` / `skip_ids`, and skip any item
  whose `self_ref` is in that set. Use `self_ref`, not `id()` — the existing
  `skip_ids` uses `id()`, which is fine for same-batch objects but is not what
  docling gives us for a resolved reference.

### 5. Wrap VLM output deterministically

Add a `_wrap_figure(md, caption)` helper that strips whatever `<figure>` tags the
model did or did not emit, then re-adds them, with the caption as a
`<figure_caption>` line inside the block. Apply it on **both** VLM paths — the
synchronous one at line 627 and the future-result loop at line 703.

While in that loop, carry the item kind in the task tuple
(`vlm_tasks.append((col, y, x, future, kind, caption))`) and delete the
`_item_sort_key(item) == (y, x)` re-lookup at
[gemini_docling_parser.py:704](../../src/app/ingestion/processors/gemini_docling_parser.py#L704).
That line decides table-vs-figure wrapping by re-scanning `items` for a matching
sort key; it is fragile and becomes wrong the moment two items share a key.

`_strip_stray_headers` already tracks `<figure>` to decide "inside a block", but
it runs inside `_call_vlm`, *before* the wrapper is applied — so its behaviour is
unchanged. Verify this rather than assuming it.

### 6. Make the whole-document post-passes fence-aware — **the trap**

`_fix_markdown_headings` and `_normalize_tables_in_markdown` run over the joined
markdown and today only track `<table>` regions. Once fenced code exists they
will corrupt it:

- `_fix_markdown_headings` rule 1 promotes any `N. Title` line to `##`, and
  rule 2 promotes any short ALL-CAPS line. Both match lines that occur inside
  Python sessions.
- `_normalize_tables_in_markdown` rewrites any line starting with `|`.

Add a shared ```` ``` ```` toggle to both, in the same style as their existing
`in_table` flag. This is silent-corruption territory, so it is not optional.

`_clean_html` strips `<br>` and `<tr>/<td>/<th>` document-wide and will hit code
containing those strings. Pre-existing, out of scope, worth a comment.

### Not doing

- **No prompt change.** F22 showed the prompt is not the lever here, and the
  `<figure>` wrapper is being enforced in code precisely because three prompt
  versions failed to get it emitted.
- **No chunker change.** Switching a 1 MB document to `MarkdownChunker` is a
  separate, measurable decision.
- **No bbox-cell reconstruction** for non-REPL code.

## Verification

Static, on the host — no re-parse needed for most of it:

```bash
PYTHONPATH=src python3 -c "
from app.ingestion.processors.gemini_docling_parser import _fix_markdown_headings, _normalize_tables_in_markdown
s = '\`\`\`python\n1. not a heading\nINTRODUCTION\n| not a table |\n\`\`\`\n'
assert _fix_markdown_headings(s) == s
assert _normalize_tables_in_markdown(s) == s
print('fence guards ok')
"
```

Then one full 504-page run through `celery_worker_upload` (the same run that
still owes us the F22 crop-padding validation — fold the two together):

```bash
docker compose restart celery_worker_upload
# upload NLTK.pdf
docker compose logs celery_worker_upload > f23.log
```

| metric | F22 | F23 expectation | reads as |
|---|---|---|---|
| fenced blocks | 0 | **> 500** | code is marked |
| `>>>` lines outside a fence | 631 | **≈ 0** | all of it, not some |
| `<figure>` … `</figure>` pairs | 0 | **= `vlm_calls`** | every description is bounded |
| `<figure_caption>` | 0 | **> 40** | captions bound, not floating |
| `vlm_calls` | 78 | **≈ 85** | F22 padding fix: 7 recovered + 8 newly detected at 144 DPI |
| `tables` | 64 | **64** | nothing here touches TableFormer |
| total parse | 1083s | **≈ 1083s** | this is all string work |
| `peak_rss` | 2965 MB | **≈ 2965 MB** | unchanged; still 83% of the 3.5G limit |

**The real check is by eye**, on page 30 of the artifact: the
`>>> sorted(set(text3))` session must appear as a fenced block with `>>>` and
`...` on separate lines, and `Table 1-1. Lexical diversity of various genres in
the Brown Corpus` must appear inside its table block rather than beside it.

Two factory tests are already failing (`tests/unit/test_pdf_parser_factory.py`)
from F21/F22 constructor changes and are the repo owner's to fix; this change
adds nothing to them.

## Deferred, with the evidence, so it is not re-derived later

- **Junk in the index**: 213 orphan `|-|` rows outside any `<table>` block
  (~42% of pages), 4 table blocks with no rows, 4 dotted-leader TOC
  pseudo-tables, 11 pages that parsed to zero characters.
- **Robustness**: `do_ocr=False` is hardcoded in `_build_converter`, so a scanned
  PDF produces a near-empty artifact and the pipeline reports success. No
  resume — an 18-minute parse that dies at page 480 restarts at page 1, even
  though page batches plus the streamed artifact are already a natural
  checkpoint.
- **Memory**: peak RSS 2965 MB against 3584 MiB. Either drop `VLM_IMAGES_SCALE`
  to 1.5 (≈2540 MB) or raise the worker limit. Still undecided.
