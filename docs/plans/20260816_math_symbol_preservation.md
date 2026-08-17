# Preserve math / special symbols in RAG answers

**Date**: 2026-08-16
**Status**: planned, not implemented.
**Branch**: `refactor/document-ingestion`

## Context

Answers returned by the RAG pipeline strip mathematical and special notation. A source
line reading `∇f(x) ≤ ε for all x ∈ X` reaches the LLM as
`[math] f(x) <= epsilon for all x [math] X`, so the model cannot reproduce it even if
asked to. The goal is for the answer text to carry the original symbols.

Two things cause this, and prompts are only half of it:

1. **`MathNotationNormalizer` destroys symbols at ingest time.**
   [`cleaners.py:82`](../../src/app/ingestion/text_cleaning/cleaners.py) replaces the entire
   Unicode Mathematical Operators block (`∀ ∃ ∈ ∇ ⊂ ∪ ≡ ⊗ …`) with the literal string
   `" [math] "`, and `:85` deletes Mathematical Alphanumeric Symbols outright. Greek
   letters become `alpha`/`beta`. The cleaned text is written back onto the chunk at
   [`pipeline.py:317`](../../src/app/ingestion/embedding/pipeline.py), so the **stored**
   `text` column is already lossy — no prompt or UI change can recover it without a
   re-ingest.

2. **The RAG prompt forbids the behaviour we want.**
   [`prompts.py:131`](../../src/app/ingestion/processors/prompts.py) says *"Do not copy
   sentences verbatim from the context"*. A formula must be copied verbatim. The Gemini
   backend carries a hand-duplicated copy of the same prompt inline at
   [`llm_operations.py:76-94`](../../src/app/retrieval/llm_operations.py), so any fix
   applied to one silently misses the other.

**Chosen encoding: Unicode passthrough.** The prompt will ask the model to reproduce
symbols exactly as they appear in the context, not to emit LaTeX. This works with the UI
as it stands today — [`home.html:717`](../../src/app/api/templates/home.html) escapes the
answer into a `white-space: pre-wrap` div, which displays Unicode correctly but would show
`$x^2$` literally. A KaTeX renderer and the `VLM_MIN_IMAGE_SHORT_PX` image gate
(ARCHITECTURE §4.6.7 — it deliberately skips "equation lines") are **out of scope**.

---

## Part 1 — Prompt changes

### `VLM_IMAGE_PROMPT` (Gemini) — update

Its existing rules already demand verbatim transcription and forbid paraphrase; the math
rule is a natural extension and Gemini follows instructions reliably. Add one rule to the
list:

> `- Transcribe mathematical expressions exactly, preserving every symbol (∑ ∫ ∇ α ≤ ² ₁).`
> `  Do NOT transliterate a symbol into a word and do NOT describe an equation in prose.`

### `VLM_TABLE_PROMPT` (Gemini) — update

Cells routinely carry `σ`, `≤`, `x²`. Add two rules:

> `- Preserve mathematical symbols, Greek letters, superscripts and subscripts exactly as`
> `  printed. Do NOT transliterate them into words.`
> `- If a cell's content contains a literal | character, escape it as \|.`

The escape rule matters because `_normalize_tables_in_markdown`
([`gemini_docling_parser.py:341`](../../src/app/ingestion/processors/gemini_docling_parser.py))
treats any line starting with `|` as a table row and will split an unescaped pipe into a
spurious column.

### `OLLAMA_IMAGE_PROMPT` — **do not change**

This is a deliberate non-change. The comment block at
[`prompts.py:49-79`](../../src/app/ingestion/processors/prompts.py) records three measured
iterations showing that every added rule costs output budget on `qwen3.5:0.8b`, and that
negative formatting constraints get ignored or recited back. The `_strip_html_wrappers`
docstring ([`gemini_docling_parser.py:273-287`](../../src/app/ingestion/processors/gemini_docling_parser.py))
reaches the same conclusion independently: *"at 0.8B the model does not reliably honour
negative formatting constraints, so these are normalised here, where the outcome is
guaranteed rather than requested."* [`20260812_structure_preservation.md`](../20260812_structure_preservation.md)
§"Not in scope" says the same. The prompt also already routes equations to verbatim output
at `:93-95`. The real Ollama-side blocker is the image gate, not the prompt.

### `OLLAMA_TABLE_PROMPT` — **do not change**

Dead by default: `VLM_TABLES=false` sends all tables to docling's TableFormer
(ARCHITECTURE §4.6.5).

### `OLLAMA_RAG_PROMPT_TEMPLATE` — update, and make it the single shared template

Replace the blanket rule at `:131` with a narrowed rule plus an explicit carve-out:

> `- Summarize the relevant information in your own words. Do not copy whole sentences`
> `  verbatim from the context.`
> `- EXCEPTION: reproduce formulas, equations, symbols, units and identifiers EXACTLY as`
> `  they appear in the context, including any markup already present. Never transliterate`
> `  a symbol into a word — write ∑, not "sum".`

Then rename it `RAG_PROMPT_TEMPLATE` and have **both** backends use it: delete the inline
f-string at [`llm_operations.py:76-94`](../../src/app/retrieval/llm_operations.py) and
format the shared template in `GeminiBackend`, exactly as `OllamaBackend` already does at
`:34`. The two texts are identical today apart from line wrapping, so this is a
de-duplication, not a behaviour change for Gemini beyond the new rule. Update the single
import at `:21`.

---

## Part 2 — Cleaner changes

All in [`cleaners.py`](../../src/app/ingestion/text_cleaning/cleaners.py).

1. **Drop `MathNotationNormalizer` from the default chain** (`:224-231`). Keep the class —
   it stays available for opt-in use via the existing `strategies=` argument and
   `add_strategy()`. Gate the default with `preserve_math: bool = True` on
   `TextCleaningPipeline.__init__`, sourced from a new `PRESERVE_MATH_NOTATION` setting in
   `AppSettings` following the `persist_ingestion_artifacts` pattern at
   [`app_config.py:97-99`](../../src/app/config/app_config.py), read where the pipeline is
   constructed at `pipeline.py:308`. Re-ingesting a large corpus is expensive; a one-env-var
   revert is worth the line.

2. **`UnicodeNormalizer('NFKC')` → `NFC` in the default chain** (`:227`). NFKC is a
   compatibility mapping: it turns `²` into `2` and `½` into `1⁄2` *before* anything else
   runs. Two consequences worth noting — it is why the `'²': '^2'` and `'½': '1/2'` entries
   at `:61` and `:139` are currently unreachable dead code, and it is a second, independent
   path by which superscripts are lost. Keep the constructor argument so NFKC stays
   available.

3. **Scope `TableStructurePreserver`'s pipe rewrite** (`:107`). `re.sub(r'\s*\|\s*', ' | ')`
   currently rewrites every pipe in the document, mangling `|x|` and `‖v‖` in prose. Apply
   it only to lines that look like table rows (`^\s*\|.*\|\s*$`); leave other lines alone.

4. **Trim `SpecialSymbolNormalizer`'s replacement map** (`:121-140`). Remove the currency
   entries (`€ → EUR`, `£ → GBP`, `¥ → YEN`) and the fraction entries — with NFC in place
   the fraction rules become live for the first time and would newly destroy `½`. Keep the
   quotes, dashes, ellipsis and bullet normalization: those help BM25 and lose nothing the
   user cares about.

---

## Files to modify

| File | Change |
|---|---|
| `src/app/ingestion/processors/prompts.py` | 2 VLM prompt rules; RAG template rewrite + rename to `RAG_PROMPT_TEMPLATE` |
| `src/app/retrieval/llm_operations.py` | Delete the duplicated inline Gemini prompt; use the shared template; fix the import at `:21` |
| `src/app/ingestion/text_cleaning/cleaners.py` | Default-chain, NFC, pipe-scoping, symbol-map changes above |
| `src/app/config/app_config.py` | `preserve_math_notation` setting |
| `src/app/ingestion/embedding/pipeline.py` | Pass the setting into `TextCleaningPipeline()` at `:308` |
| `tests/unit/test_text_cleaning.py` | See below |
| `.env.example`, `docs/ARCHITECTURE.md` | See below |

`ollama_pdf_parser.py:100-103` swaps the Gemini prompts for the Ollama ones by comparing
against the same constants it imports, so editing the Gemini prompt text needs no change
there — but re-read it during implementation to confirm the identity/equality check still
holds after the edit.

### Tests

`tests/unit/test_text_cleaning.py` (22 tests). Two will fail by design and must be
re-pointed at an explicitly-constructed legacy pipeline so both behaviours stay covered:

- `test_normalizes_greek_letters` (`:67`) — asserts `α → "alpha"`
- `test_normalizes_math_symbols` (`:78`) — asserts `∑ → "sum"`

`test_normalizes_comparison_operators` (`:89`) and the fraction test (`:159`) are already
written with `or "≤" not in result`-style escape hatches and pass either way.

New tests to add:
- Default pipeline preserves `∀ ∃ ∈ ∇ α ∑` and never emits the literal `[math]`
- Default pipeline preserves `x²` (the NFC regression guard)
- `|x| > 0` in prose survives; a real `| a | b |` table row is still normalized
- `€100` survives
- `MathNotationNormalizer` still transliterates when explicitly included in `strategies=`

### Docs

- `docs/ARCHITECTURE.md` §4.6.5 item 3 documents the old transliterating behaviour and must
  be rewritten; item 5 needs the currency/fraction removal noted.
- `docs/ARCHITECTURE.md` §4.6.10 line 366 is already stale — it lists `prompts.py` as
  holding `_VLM_IMAGE_PROMPT`, `_VLM_TABLE_PROMPT` (leading underscores are the *import
  aliases*, not the names) and omits the three Ollama prompts and the RAG template. Fix
  while in there.
- `.env.example` + ARCHITECTURE §7 env table: add `PRESERVE_MATH_NOTATION`.

---

## Verification

**Unit** — baseline is 9 failures / 605 passed on this branch (ARCHITECTURE §10.3); the
count must not grow:

```bash
python -m pytest tests/unit/test_text_cleaning.py -v
python -m pytest tests/unit -q --ignore=tests/unit/test_pdf_to_markdown.py
```

**End-to-end** — the stored text is already lossy, so verification *requires* a fresh
parse. Commands for the repo owner to run:

```bash
docker compose up -d --build app celery_worker_upload
docker compose logs -f celery_worker_upload
```

1. Upload a symbol-heavy PDF through the UI into a scratch domain.
2. Inspect the parse artifact — symbols should be intact, no `[math]`:
   `rg -n '\[math\]' data/parsed/ data/chunks/ ; rg -n '[∑∫∇α≤∈]' data/chunks/<id>_*/0000.md`
3. Confirm the DB agrees:
   `SELECT count(*) FROM <domain_table> WHERE text LIKE '%[math]%';` → expect 0.
4. Query the domain in the chat UI with a question whose answer contains a formula.
   Expect the symbols rendered in the answer bubble, not spelled-out words and not
   literal `$…$`.
5. Run the same query against both `gemini-*` and an Ollama model to confirm the shared
   template took effect on both backends.

**Rollback** — set `PRESERVE_MATH_NOTATION=false` and re-ingest to restore the previous
cleaning behaviour.

## Not in scope

- KaTeX / markdown rendering at `home.html:717` — deferred with the Unicode-passthrough decision.
- `VLM_MIN_IMAGE_SHORT_PX` image-gate carve-out for equation strips (ARCHITECTURE §4.6.7).
  Display equations rendered as thin image strips remain dropped at parse time; this costs
  VLM calls and parse time to change and needs its own measurement against the §4.6.8
  summary line.
- `OLLAMA_IMAGE_PROMPT` / `OLLAMA_TABLE_PROMPT`, for the reasons recorded above.
- Existing chunks. They stay lossy permanently until their documents are re-ingested.
