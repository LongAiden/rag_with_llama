# One batch of fifty pages was a third of the parse

**Date**: 2026-08-11
**Status**: implemented and measured end-to-end on the 504-page reference document.
Both parts passed. Unit tests remain outstanding (the repo owner writes them).

## Context

`docs/20260805_vlm_output_length_and_image_gate.md` (F18) capped VLM output with
`num_predict: 384` and added a 64px short-side image gate. F19 (commit `7631584`) then
bounded the image prompt. Between them the 504-page NLTK book went from 2194s to 1683s,
and VLM truncations from 28 of 79 calls to zero.

That left docling at **79% of the parse**, which reads like a wall — until you look at
where the time actually goes. Per-batch rates from the F19 run:

| batch | elapsed | rate |
|---|---|---|
| 1–50 | 111.9s | 2.24 s/page |
| 51–100 | 80.0s | 1.60 s/page |
| 101–150 | 69.9s | 1.40 s/page |
| 151–200 | 65.9s | 1.32 s/page |
| 201–250 | 81.7s | 1.63 s/page |
| 251–300 | 74.5s | 1.49 s/page |
| **301–350** | **147.9s** | **2.96 s/page** |
| 351–400 | 72.7s | 1.45 s/page |
| 401–450 | 68.6s | 1.37 s/page |
| **451–500** | **554.8s** | **11.10 s/page** |
| 501–504 | 5.1s | 1.29 s/page |

**One batch of 50 pages was 42% of all docling time and 33% of the entire parse.** Docling
was not slow. Docling was fast everywhere except one place, and the average hid it.

The suspect: pages 451–500 are the book's index, laid out as dense multi-column tables
that docling's layout model classifies as tables and hands to TableFormer. Zero VLM calls
fall in that range, and per-page assembly there is normal, so the 555s is pure
`DocumentConverter.convert` time. The same outlier appears in the F18 run at 11.07 s/page,
so it was reproducible rather than a host fluke.

Two prompt findings ran alongside it. F19's *"Do NOT mention that something is absent"*
had left absence-reporting essentially untouched, and its *"at most 3 short sentences (60
words)"* produced a 95-word median — a 0.8B model does not count words.

## Part A — the TableFormer outlier

### Confirming the cause before changing anything

`scripts/f21_tableformer_probe.py` drives `DocumentConverter` directly over
`page_range=(451, 500)` under three configurations. `OllamaPDFParser` accepts `max_pages`
but not a start page, so timing 1–450 against 1–500 would have been the wrong shape.

```
baseline ACCURATE+match      540.8s  10.82s/page  tables=15
FAST+match                   129.7s   2.59s/page  tables=15
FAST, no cell matching       125.1s   2.50s/page  tables=15
```

The baseline reproduces F19's 554.8s to within 2.5%, which was the precondition for
believing any of this: had it not, the cause would not have been TableFormer.

**`fast` is a 76% win at no measured cost.** `tables=15` under both modes — on the hardest
table layout in the document, the faster decoder finds exactly the same structures.

**Cell matching is not the cost.** Disabling it saves 4.6s of 129.7s, **3.5%**. So the
540.8s was almost entirely the ACCURATE structure decoder. There is nothing to buy by
turning cell matching off, and cell accuracy is now effectively free, so
`_build_converter` keeps `do_cell_matching=True`.

The fourth configuration (`table_structure OFF`) was not run. 67 real tables elsewhere in
the document depend on table structure, so it could not have been adopted whatever its
number.

### What changed in the code

The mode is configurable rather than hardcoded, following the constant + constructor-arg +
env-var pattern already used by `_DEFAULT_MIN_IMAGE_SHORT_PX`:

- [`gemini_docling_parser.py`](../src/app/ingestion/processors/gemini_docling_parser.py) —
  `_DEFAULT_TABLEFORMER_MODE = "fast"`, a `tableformer_mode` constructor argument, and
  `_resolve_tableformer_mode()`, which maps the string onto docling's enum and falls back
  to docling's own default **with a warning** if the enum is missing or the value is a
  typo. A bad env var cannot take a parse down.
- [`ollama_pdf_parser.py`](../src/app/ingestion/processors/ollama_pdf_parser.py) — threads
  the argument through to `super().__init__()`.
- [`app_config.py`](../src/app/config/app_config.py) — `docling_tableformer_mode`, read
  from `DOCLING_TABLEFORMER_MODE`.
- [`pdf_parser_factory.py`](../src/app/ingestion/processors/pdf_parser_factory.py) — passes
  it in the shared `tuning` dict, so both backends get it.
- `docker-compose.yml` — `DOCLING_TABLEFORMER_MODE: ${DOCLING_TABLEFORMER_MODE:-fast}` on
  all four services carrying the `DOCLING_*` block.

Reverting is one environment variable: `DOCLING_TABLEFORMER_MODE=accurate`.

**No heuristic disables TableFormer on "index-looking" pages.** That is a content
classifier disguised as a performance fix, and it was ruled out of scope.

## Part B — the image prompt, versions 2 and 3

### v2 failed in an instructive way

v2 deleted the enumerated checklist rather than negating it, and replaced the word count
with a sentence count. Output dropped — mean 148 → 105 tokens over an 80-page probe — but
it introduced a defect worth recording.

The rule read: *"The first names what the image is. The second states its single most
important content."* **The model recited that sentence back as its output.** Any rule that
describes the *shape* of the answer is recitable at this model size. v3 states the bound
without narrating what each sentence should carry.

v2 also broke grounding. The page-1 cover became one fluent sentence of recalled knowledge
about the book — naming spaCy, which does not appear on the cover — with the whales and
the banner unmentioned. That is a **grounding** failure, not a length failure, so the
planned "loosen to three sentences" contingency would not have fixed it and was not
applied. v3 forbids outside knowledge and licenses `Unclear image.` as an answer, so
fabricating is no longer the model's only way to fill the space.

### Three `Do NOT` rules had zero effect, so they moved into code

Across v1, v2 and v3, rules banning HTML tags and demanding the description start on the
line *after* `</figure_type>` were simply ignored: the v3 probe still put 8 of 17
descriptions on the `<figure_type>` line and emitted 8 stray `<p>`/`<div>` tags.

Format defects that a 0.8B model will not obey are a code problem, not a prompt problem.
`_strip_html_wrappers()` in `gemini_docling_parser.py` now removes `<p>`/`<div>`/`<span>`
wrappers and forces the newline after `</figure_type>`, deterministically, in both
backends' sanitizer chains. It runs after `_strip_code_fences` and before
`_strip_stray_headers`, and it leaves `<table>` blocks and already-correct output untouched.

### Hallucination is a capacity limit, not a prompt bug

The page-25 crop (an NLTK download dialog) was invented three different ways across three
prompt versions — a COPPER/IRON/CHROMOS/PHENOL spreadsheet with the value -348679 under
v2, an ID/Name/Department/Job Title/Salary Range table under v3. No prompt edit moved it.
This is the ceiling of `qwen3.5:0.8b` on dense screenshots, and it is the single remaining
`done=length` truncation in the final run. It needs a larger model, not another rule.

> **Correction — this conclusion was wrong (2026-08-11, F22).** The model was not at its
> capacity limit; it could not see the image. `images_scale` was pinned at **0.6**, so
> docling rendered pages at **43 DPI** and `_expand_and_crop` handed the VLM a **218×96px**
> thumbnail of a full-page-width screenshot — an 8pt glyph is 5 pixels tall at that
> resolution. Across all 79 calls, output length runs *inversely* to crop area: the four
> longest descriptions are the four **smallest** crops. That is the signature of
> confabulation from an illegible input, and it explains what this section could not — why
> three prompt versions failed on this crop alone, and why v3's explicit `Unclear image.`
> escape hatch went unused 79 times out of 79. The model does not experience the crop as
> unclear; it experiences it as a table it can almost read. See F22 for the fix
> (`VLM_IMAGES_SCALE`, and the image gate re-expressed in points so it stops drifting when
> the scale moves).

## Result

Same 504-page document, same host (Apple M1, 16 GB, local Ollama, `qwen3.5:0.8b`), through
`celery_worker_upload`:

```
parse_pdf summary: pages=504 total=1045s docling=782s (75%) assembly=263s (25%)
                   vlm_wait=261s vlm_calls=79 vlm_failures=0 peak_rss=2000MB
```

| metric | F18 | F19 | **F21** | target | |
|---|---|---|---|---|---|
| total | 2194s | 1683s | **1045s** | ≤1250s | ✅ −38% |
| docling | 1600s | 1333s | **782s** | ≤900s | ✅ |
| pages 451–500 | 553s | 554.8s | **121.1s** | ≤150s | ✅ −78% |
| pages 301–350 | — | 147.9s | **73.5s** | — | ✅ |
| `vlm_wait` | 592s | 348s | **261s** | ≤260s | 1s over |
| mean output tok | ~318 | 160 | **109** | ≤110 | ✅ |
| `vlm_calls` | 79 | 79 | **79** | exactly 79 | ✅ guard rail held |
| `vlm_failures` | 0 | 0 | **0** | 0 | ✅ |
| `done=length` | 28 | 0 | **1** | 0 | ⚠️ |
| `peak_rss` | 1979 MB | 2524 MB | **2000 MB** | <3000 MB | ✅ |
| tables | 67 | 67 | **67** | 67 | ✅ |
| filler clauses | 19 | 15 | **3** | ≤5 | ✅ −80% |
| inline descriptions | — | 2 | **0** | 0 | ✅ |

**The mode change is not lossy, and that is verified structurally rather than by a single
count.** The F21 and F19 artifacts agree exactly: 67 `<table>`, 66 `</table>` closers, 1
empty block. The isolated probe could only show this for 50 index pages; the full run shows
it for the whole document.

**The second outlier resolved too.** Pages 301–350 went 147.9s → 73.5s, landing on the
healthy 1.47 s/page rate. It was table-bearing, just not pathologically so.

**Peak RSS fell 524 MB** (2524 → 2000), which was not a goal. F19 had peaked at 98.6% of
the old 2560 MiB ceiling — one batch from the SIGKILL that prompted raising the worker to
3.5G. Headroom is now ~1.5 G. The `fast` checkpoint is 139 MB against `accurate`'s 203 MB,
and prompt v3 accumulates less text, but the full 524 MB is not accounted for; the RSS
climb still deserves its own investigation before a much larger document is attempted.

**Two blemishes, both marginal.** `vlm_wait` missed its 260s target by one second. One call
truncated at `num_predict` — the page-25 dialog described above.

### A measurement trap worth recording

`scripts/f21_census.py` originally counted stray HTML over the whole artifact and reported
2 `<p>` and 6 `<span>` in the F21 output, which looked like `_strip_html_wrappers()` had
failed. It had not. Those tags are the **book's own content**: Chapter 3 teaches HTML
parsing and prints a `<p class=MsoNormal>sleep <span ...>[sli:p]` dictionary entry, then
discusses the `<p>` element in prose. F19 scores the same lines. Scoped to the description
windows, both counts are **0**.

The same class of error bit the filler metric earlier: the original census stopped reading
at the first blank line, so it only ever saw paragraph 1 of multi-paragraph descriptions.
That made the F18→F19 filler count look like a regression (11 → 12) when the true movement
over the whole description was 19 → 15, a 21% improvement. Both measures are now reported
side by side so the historic numbers stay comparable.

## Still to verify

1. **Unit tests.** Outstanding since F18: the `options` payload assertion, the
   `done=length` warning, the short-side gate, and now `_resolve_tableformer_mode()`'s
   fallback on an unknown value and `_strip_html_wrappers()`'s six cases.
   `tests/unit/test_ollama_vlm_call.py` and `test_vlm_table_routing.py` are the pattern.
2. **`fast` on other document types.** Verified lossless on this book's index and body. A
   drop in `tables` on a different document means reverting `DOCLING_TABLEFORMER_MODE` for
   that corpus.
2b. ~~**A larger VLM for dense screenshots.**~~ **Superseded by F22** — the problem was
   43 DPI input, not model capacity. See the correction above.
3. **The remaining index cost.** `fast` is 2.42 s/page on pages 451–500 against the 1.47
   s/page healthy rate — better than 11.10, not equal to the rest. Closing that would need
   per-page-region policy, which is a design question, not a tuning one.

## Out of scope, noted

- **`VLM_MIN_IMAGE_SHORT_PX` stays at 64.** Raising it to 80 would drop 23 calls and 90s,
  to 100 would drop 40 calls and 164s — but the gate silently discards page content, and
  the measured defects were about *what the model writes*, not how many crops it sees.
- **The empty `<table>` block** (1 of 67). TableFormer, not the VLM, and present identically
  before and after this change. It belongs to F16's open question about TableFormer output
  quality.
- **The dead `<figure>` wrapper rules.** The model emitted 0 wrappers in 79 calls while
  obeying `<figure_type>` 79/79. Four prompt rules are dead weight it discards, costing
  prefill only. Removing them is a separate question with no measured payoff.
- [llm_operations.py:37](../src/app/retrieval/llm_operations.py#L37) — the Q&A path still
  sends a bare `/api/generate` payload with no `keep_alive`, no `think` and no `options`,
  against `deepseek-r1:1.5b`, itself a reasoning model. The same class of bug F14 fixed on
  the parse path, still live on the query path. Worth its own ticket.

## References

- **F14–F17**: `docs/20260805_vlm_thinking_and_table_routing.md`
- **F18**: `docs/20260805_vlm_output_length_and_image_gate.md`
- **F19**: commit `7631584`
- **Working plan and full measurement log**:
  `docs/plans/20260811_tableformer_outlier_and_prompt_v2.md`
- **Architecture**: `docs/ARCHITECTURE.md` §15.7–15.9
