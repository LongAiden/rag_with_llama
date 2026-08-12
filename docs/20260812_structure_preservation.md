# F23 — Structure Preservation (2026-08-12)

> Follows F22 (144 DPI, crop padding in points). Plan:
> `docs/plans/20260812_parser_structure_preservation.md`.

## Problem

`_process_page` threw away most of docling's structural classification. `CodeItem`
and `ListItem` both subclass `TextItem`, so the `isinstance(item, TextItem)` branch
caught them first and emitted bare `item.text`. The VLM's `<figure>` wrapper was
never enforced in code. Measured on the 504-page NLTK artifact:

| defect | count |
|---|---|
| `>>>` lines, 0 fenced blocks | 631 |
| `<figure>` wrappers surviving | 0 / 79 |
| `<figure_caption>` bound to parent | 0 |
| list markers preserved | 3 / ~thousands |

## Changes

All in `src/app/ingestion/processors/gemini_docling_parser.py`.

### New helpers

- **`_format_code(item)`** — fences code items with `` ```lang ``. Restores
  doctest line breaks by splitting on whitespace before `>>>` / `...` prompts.
  Strips trailing bare `>>>` and `...` that docling appends. `code_language`
  enum `.value` used as the fence language; `unknown` omitted.

- **`_format_list_item(item)`** — emits `marker text` when `item.marker` is
  present, `1. text` for enumerated items, `- text` otherwise.

- **`_wrap_figure(md, caption)`** — strips any `<figure>` tags the VLM did or
  did not emit, re-adds them deterministically with `<figure_caption>` inside.
  Applied on both the synchronous and async VLM paths.

### Label dispatch (§1)

The `TextItem` branch now dispatches on `getattr(item.label, "value", "")`:
`"code"` → `_format_code`, `"list_item"` → `_format_list_item`, else bare text.
Immune to future `XItem(TextItem)` subclasses — the MRO trap that caused this
bug.

### Caption binding (§4)

A pre-pass collects caption text via `item.caption_text(doc)` for all
`FloatingItem`s. Loose `TextItem`s whose text matches a collected caption are
skipped, so captions appear only inside their parent block.

### VLM task tuple (§5)

`vlm_tasks` now carries `(col, y, x, future, kind, caption)` where `kind` is
`"figure"` or `"table"`. The fragile `_item_sort_key` re-lookup at the result
loop — which decided table-vs-figure wrapping by re-scanning all items for a
matching sort key — is deleted. Two items sharing a sort key can no longer
mis-route the wrapping.

### Fence-aware post-passes (§6)

`_fix_markdown_headings`, `_normalize_tables_in_markdown`, and
`_strip_stray_headers` all track an `in_fence` flag (```` ``` ```` toggle).
Content inside fenced code blocks is passed through unchanged. Without this,
`1. not a heading` inside a Python session would be promoted to `##`, and
`| not a table |` would be rewritten as a markdown table.

## Tests

`tests/unit/test_f23_structure_preservation.py` — 27 tests:

- `TestFormatCode` (7): empty, plain, language, unknown language, doctest split,
  trailing prompt strip, indentation preservation.
- `TestFormatListItem` (4): empty, marker, enumerated, plain.
- `TestWrapFigure` (4): plain wrap, caption, strip existing, strip partial.
- `TestFenceAwareFixMarkdownHeadings` (5): outside promoted, inside unchanged,
  mixed, ALLCAPS inside/outside.
- `TestFenceAwareNormalizeTables` (3): outside normalized, inside unchanged,
  mixed.
- `TestFenceAwareStripStrayHeaders` (4): stray removed, fence preserved, figure
  preserved, mixed.

Existing `test_pdf_parser_streaming.py` (10 tests) confirmed passing.

## Expected metrics (504-page NLTK.pdf)

| metric | F22 | F23 expectation |
|---|---|---|
| fenced blocks | 0 | > 500 |
| `>>>` lines outside a fence | 631 | ~ 0 |
| `<figure>` pairs | 0 | = vlm_calls |
| `<figure_caption>` | 0 | > 40 |
| total parse time | 1083s | ~ 1083s (string work only) |
| peak RSS | 2965 MB | ~ 2965 MB |

## Not in scope

- No prompt changes — the `<figure>` wrapper is enforced in code.
- No chunker change — `MarkdownChunker` for large documents is a separate decision.
- No bbox-cell reconstruction for non-REPL code.
- `_clean_html` still strips `<br>` and `<tr>/<td>/<th>` document-wide — pre-existing,
  will hit code containing those strings. Tracked but not fixed here.
