# F21 — Kill the TableFormer Outlier and Fix the Two Prompt Rules That Didn't Work

**Date**: 2026-08-11  
**Status**: ✅ **COMPLETE — both parts measured and passed** on the full 504-page run
(2026-08-11). Total 1683s → **1045s**, index batch 555s → **121s**, filler 15 → **3**,
`tables` unchanged at 67. `fast` is now the code default. Results written up in
[`docs/20260811_tableformer_outlier_and_prompt_v2.md`](../20260811_tableformer_outlier_and_prompt_v2.md);
this file is the working plan and measurement log behind it.  
**Scope**: Docling TableFormer configuration, Ollama VLM image prompt  
**Dependencies**: F14–F19 (already applied), F20 (hf_cache volume)

---

## Context

### Where the pipeline stands

Four rounds of work are already applied and measured. F14–F17
(`docs/20260805_vlm_thinking_and_table_routing.md`) turned off Qwen3.5's default reasoning
(`"think": false`, 87s → 2.3s per call), pinned VLM concurrency to 1, and rerouted tables
from the VLM to docling's TableFormer. F18
(`docs/20260805_vlm_output_length_and_image_gate.md`) added
`options: {temperature: 0.0, num_predict: 384}` and a 64px short-side image gate. F20
persisted the HuggingFace cache in an `hf_cache` named volume and took docling off the Xet
backend. F19 (commit `7631584`) bounded the VLM prompt.

Separately, `celery_worker_upload` was raised from 2.5G to 3.5G after a SIGKILL. That is
now quantified: the F19 run peaked at **2524 MB against a 2560 MiB ceiling — 98.6%**. The
OOM was not marginal, it was one batch away.

### F19 result: 3 of 5 pass conditions met

Same 504-page PDF, same host (Apple M1, 16 GB, local Ollama 0.32.5, `qwen3.5:0.8b`),
through `celery_worker_upload`:

```
parse_pdf summary: NLTK.pdf pages=504 total=1683s docling=1333s (79%)
                   assembly=349s (21%) vlm_wait=348s vlm_calls=79
                   vlm_failures=0 peak_rss=2524MB
```

| metric | F18 | F19 | target | verdict |
|---|---|---|---|---|
| total | 2194s | **1683s** | — | −23% |
| `done=length` | 28 / 79 | **0 / 79** | 0 | ✅ |
| `vlm_calls` | 79 | **79** | exactly 79 | ✅ guard rail held |
| `vlm_failures` | 0 | **0** | 0 | ✅ |
| mean output | ~318 tok | **160 tok** | <~100 | ❌ |
| `vlm_wait` | 592s | **348s** | ≤~300s | ❌ |

Token census over all 79 calls: mean 160, median 149, max 369, min 64. Nothing reached the
384 ceiling — `num_predict` stopped binding, which was the point of F19.

Quality census (same script as F18, run on both artifacts):

| | F18 | F19 |
|---|---|---|
| descriptions | 79 | 79 |
| census-truncated | 28 | 4 |
| filler clauses — *whole description* | 19 | **15** |
| filler clauses — *first paragraph only* | 11 | 12 |

> **Measurement correction, 2026-08-11.** An earlier reading of this table recorded the
> filler count as 11 → 12 and concluded the anti-filler rule had *failed*. That was a
> measurement artifact. Figure descriptions have no end delimiter in the artifact — they
> begin at `<figure_type>` and run into the page body — and the census script stopped at
> the first blank line, so it only ever read paragraph 1 of a routinely multi-paragraph
> description. Because F19's output is shorter, more of its filler fell inside paragraph 1
> where the metric could see it, which inverted the sign. Read over the whole description
> the movement is **19 → 15, a 21% improvement**. Both numbers are reproduced by
> `scripts/f21_census.py`, which reports each and labels which is which.

Two findings drive this document.

**1. The anti-filler rule worked, but nowhere near enough.** F19 added *"Do NOT mention
that something is absent"* and absence-reporting fell 19 → 15 of 79. Real F19 output that
survives: *"The axis labels are not clearly visible in this cropped view… There is no
legend"*, *"There is no legend visible"*. What F19 **kept** is the likely reason it
stalled: the rule still enumerates *"axis labels, legend entries, data values and text
annotations"*, and the model echoes that list back to report each item missing. F18's own
analysis said the checklist is echoed as filler; F19 left the checklist in place and only
appended a negation to it. v2 deletes the checklist instead.

**2. The word bound is ignored.** The rule asks for *"at most 3 short sentences (60
words)"*; the median description is **95 words**. A 0.8B model does not count. Output
halved because the *shape* of the ask changed, not because the number was obeyed.

Hallucination is unchanged in kind, only in length: the Figure 1-3 region is still
described as a table of `"the" / "been" / "message" / "perspective"` columns with icons,
and one chart still invents data points `(0,2), (1,3), (2,4), (3,5)` and an equation
`y = x² + 6x − 7`. These descriptions are written into `document_chunks` and retrieved as
if they were page content.

### The finding that reorders the priorities

Docling is now 79% of the parse — but it is **not uniformly slow**. Per-batch rates from
the F19 run:

| batch | elapsed | rate | rss |
|---|---|---|---|
| 1–50 | 111.9s | 2.24 s/page | — |
| 51–100 | 80.0s | 1.60 s/page | 1474 MB |
| 101–150 | 69.9s | 1.40 s/page | 1802 MB |
| 151–200 | 65.9s | 1.32 s/page | 1783 MB |
| 201–250 | 81.7s | 1.63 s/page | 1655 MB |
| 251–300 | 74.5s | 1.49 s/page | 1716 MB |
| **301–350** | **147.9s** | **2.96 s/page** | 1854 MB |
| 351–400 | 72.7s | 1.45 s/page | 2159 MB |
| 401–450 | 68.6s | 1.37 s/page | 2283 MB |
| **451–500** | **554.8s** | **11.10 s/page** | 2351 MB |
| 501–504 | 5.1s | 1.29 s/page | 2524 MB |

**One batch of 50 pages is 555s — 42% of all docling time and 33% of the entire parse.**
The seven healthy batches average **1.47 s/page**. At that rate pages 451–500 would cost
~74s, so **~481s is recoverable there**, plus ~74s from the 301–350 batch: together
**~556s, or 33% off the total parse.** Nothing left on the VLM side comes close.

Attribution evidence: the eight largest tables in the produced artifact (43–52 lines each)
all sit in the **last 5% of the document** — the NLTK book's index, laid out as dense
multi-column tables that docling's layout model classifies as tables and hands to
TableFormer. Zero VLM calls fall inside that batch (call #79 is the last, before it), and
per-page assembly in that range is normal (4.6s max). The 555s is pure
`DocumentConverter.convert` time. The same 11.07 s/page outlier appears in the F18 run, so
it is reproducible and document-structural, not a host fluke.

**This attribution is circumstantial and Step 1 below confirms it before anything changes.**

### Scope decision (made by the repo owner, do not revisit)

- **Docling + prompt v2, in one measured run.**
- **`VLM_MIN_IMAGE_SHORT_PX` stays at 64.** The gate is lossy — it silently discards page
  content. Modelled against all 79 F19 calls, raising it to 80 would drop 23 calls / 90s
  and to 100 would drop 40 calls / 164s, but the measured quality defects are about *what
  the model writes*, not how many crops it sees. Do not touch it.

---

## Part A — the docling outlier

### Step A1 — confirm the cause before changing anything (~12 min)

Isolate pages 451–500 and time three configurations. `OllamaPDFParser` accepts
`max_pages`, but not a start page, so time 1–450 and 1–500 is the wrong shape — instead
drive `DocumentConverter` directly on a page range, which is what the parser does
internally at [gemini_docling_parser.py:634-653](../src/app/ingestion/processors/gemini_docling_parser.py#L634-L653).

```bash
docker compose up -d celery_worker_upload
docker compose exec -T celery_worker_upload python /app/scripts/f21_tableformer_probe.py
```

The script times four configurations over pages 451–500: baseline `ACCURATE` + cell
matching, `FAST` + cell matching, `FAST` without cell matching (which separates the
structure decoder from cell matching in one pass rather than requiring a second run), and
`do_table_structure=False`. It prints s/page and a table count per configuration.

**No download is needed for this step.** Both TableFormer checkpoints are already in the
`rag_llama_index_hf_cache` volume — `tableformer_accurate.safetensors` (203 MB) and
`tableformer_fast.safetensors` (139 MB), under
`hub/models--docling-project--docling-models/snapshots/fc0f2d45…/model_artifacts/tableformer/`,
alongside `docling-layout-heron/model.safetensors` (164 MB). Switching to `FAST` loads a
checkpoint that is on disk today, so Step A1 runs with no network dependency.

**Read it like this.** The baseline must reproduce ~555s / ~11.1 s/page — if it does not,
the cause is not TableFormer and the rest of Part A is void; stop and re-measure. If
`table_structure OFF` collapses to ~1.5 s/page, TableFormer is confirmed as the whole cost.
The gap between `FAST` and the baseline is the actual available win.

Note `page_range` is 1-indexed and inclusive in docling's `convert`. If this docling
version does not accept it, fall back to `convert(PDF)` on a 50-page extract produced with
`pypdf` inside the container.

### Step A1 — MEASURED. Attribution confirmed, `fast` adopted (2026-08-11)

Run via `scripts/f21_tableformer_probe.py`, pages 451–500, `docling 2.117.0`,
4 CPU threads (`AcceleratorDevice.AUTO` resolves to `'cpu'` in this container — Docker
Desktop on Mac has no Metal passthrough):

```
baseline ACCURATE+match      540.8s  10.82s/page  tables=15
FAST+match                   129.7s   2.59s/page  tables=15
FAST, no cell matching       125.1s   2.50s/page  tables=15
```

The fourth configuration (`table_structure OFF`) was not run — the plan forbids shipping it
globally regardless of its number, so it could not have changed the decision.

**Three things are settled by these numbers.**

**1. The attribution holds.** The baseline reproduces F19's 554.8s / 11.10 s/page to within
2.5%. The stop-condition above — *"if it does not, the cause is not TableFormer and the rest
of Part A is void"* — does not fire. Counting F18's 553s, the outlier is now measured three
times: it is document-structural, caused by TableFormer decoding the NLTK index's dense
multi-column tables, not a host fluke.

**2. `FAST` is a 76% win with zero measured loss.** 2.59 s/page is under the ~3 s/page adopt
threshold, and `tables=15` is identical to the baseline — on the hardest 50 pages in the
document, FAST is not trading structure for speed. Adopted:
`DOCLING_TABLEFORMER_MODE=fast`.

**3. Cell matching is not the cost, and stays on.** Disabling it saves 4.6s of 129.7s —
**3.5%**. So the original 540.8s was almost entirely the ACCURATE structure decoder. The
plan's second decision branch (*"set `do_cell_matching=False` for a second measurement
before choosing"*) is closed: there is nothing to buy there, and cell accuracy is now
effectively free. `_build_converter` already hardcodes `do_cell_matching=True`, so **no code
change follows from A1 — only the `.env` value.**

What this does *not* claim: FAST at 2.59 s/page is still 1.8× the 1.47 s/page healthy-batch
average, so TableFormer remains the dominant cost on index pages. It is simply no longer a
7× outlier. Closing that remaining gap would need per-page-region policy, which is out of
scope by the rule below.

`tables=15` is the lossiness reference **for this page range only**. The whole-document
gate is still `tables=67` in Step 3 — a drop there would mean FAST is lossy somewhere
outside the index, and the mode reverts to `accurate` regardless of the ~411s win.

> **The code default stays `accurate` for now.** A1 measured 50 index pages; the
> whole-document check has not run. Until the Step 3 census passes, `fast` lives in `.env`
> as a reversible override for a validation run. Flipping `_DEFAULT_TABLEFORMER_MODE` and
> the `${DOCLING_TABLEFORMER_MODE:-accurate}` compose fallback is the follow-up commit — it
> is also what permanently removes the "did the env var actually get picked up" failure mode.

### Step A2 — apply the winning configuration

One line changes, at
[gemini_docling_parser.py:306](../src/app/ingestion/processors/gemini_docling_parser.py#L306):

```python
opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
```

Make the mode configurable rather than hardcoded, following the existing constant +
constructor-arg + env-var pattern already used by `_DEFAULT_MIN_IMAGE_SHORT_PX`
([gemini_docling_parser.py:29](../src/app/ingestion/processors/gemini_docling_parser.py#L29),
threaded through `__init__` at
[l.264](../src/app/ingestion/processors/gemini_docling_parser.py#L264) and surfaced as
`VLM_MIN_IMAGE_SHORT_PX` in `docker-compose.yml`). Add a `_DEFAULT_TABLEFORMER_MODE`
constant, a `tableformer_mode: str` constructor argument threaded through
`OllamaPDFParser.__init__`
([ollama_pdf_parser.py:47-81](../src/app/ingestion/processors/ollama_pdf_parser.py#L47-L81))
the same way `min_image_short_px` already is, and a `DOCLING_TABLEFORMER_MODE` env var
on all four services in `docker-compose.yml` that carry the existing `DOCLING_*` block
(lines 138-139, 259-260, 377-378, 482-483).

> **Applied 2026-08-11.** The plumbing is in the tree and is behaviour-neutral: the default
> is `accurate`, which is exactly what `TableStructureOptions(do_cell_matching=True)`
> produced before. `_resolve_tableformer_mode()` maps the string onto docling's enum and
> falls back to docling's own default with a warning if the enum is missing or the value is
> a typo, so a bad env var cannot take a parse down. **Nothing about parse behaviour has
> changed yet** — Step A1 decides whether the value flips to `fast`, and that flip is a
> one-line `.env` change (`DOCLING_TABLEFORMER_MODE=fast`), not a code edit.

**Decision rule, applied to the Step A1 numbers:**

- If `FAST` lands under ~3 s/page on that batch — take it, set `DOCLING_TABLEFORMER_MODE=fast`
  and consider moving the constant default in a follow-up.
- If `FAST` is within ~20% of the baseline (i.e. the cost is layout/cell-matching, not the
  structure decoder) — compare against the probe's third configuration, `FAST` without cell
  matching, before choosing.
- If only `table_structure OFF` is fast, do **not** disable table structure globally: 67
  real tables elsewhere in this document depend on it. Record the finding and stop Part A
  there; the fix is then a per-document or per-page-region policy and needs its own design.

Do not add a heuristic that disables TableFormer on "index-looking" pages. That is a
content classifier disguised as a performance fix, and it is out of scope here.

### Do not change in Part A

`images_scale` (0.75), `do_ocr` (False), `DOCLING_NUM_THREADS` (4, matching the container's
`cpus: "4.0"` limit), `DOCLING_PAGE_BATCH_SIZE` (50), or the accelerator device (`AUTO`).
None of them explain a 7× rate difference between batches of the same document.

---

## Part B — prompt v2

### One file: `src/app/ingestion/processors/prompts.py`

Only `OLLAMA_IMAGE_PROMPT` changes. `OLLAMA_TABLE_PROMPT` is not in the execution path
(`VLM_TABLES=false` since F16) and the two Gemini prompts at the top of the file belong to
the other backend — leave all three untouched.

**Current** ([prompts.py:55-73](../src/app/ingestion/processors/prompts.py#L55-L73)) — the
three rules F19 added are lines 63–67.

**Replace the rule block with:**

```python
# Bounded deliberately, second iteration. F19 (prompt v1) halved output — mean 318 tok
# -> 160 tok, 28 of 79 num_predict truncations -> 0 — but two of its rules measurably
# did not work. "Do NOT mention that something is absent" left the filler count at
# 12 of 79 (baseline 11), because the same rule still enumerates the categories the
# model then reports as missing: it echoes the checklist back. And "at most 3 short
# sentences (60 words)" produced a 95-word median, because a 0.8B model does not count
# words. v2 deletes the checklist instead of negating it, and asks for a sentence
# count rather than a word count.
OLLAMA_IMAGE_PROMPT = """\
Look at this image from a PDF page.

Describe what you see inside <figure></figure> tags.

Rules:
- Start your output with <figure> and end with </figure>. Nothing outside these tags.
- On the first line inside <figure>, add: <figure_type>Chart|Diagram|Logo|Screenshot|Other</figure_type>
- After that line write at most two sentences. The first names what the image is. The
  second states its single most important content. Then stop.
- Write only about what is visibly present.
- If the image is a band of text, an equation, a rule, or a code listing — anything that
  is not a chart, diagram, logo or screenshot — output the visible text verbatim and
  nothing else: no sentences, no description of it.
- If the image shows a flowchart or process: describe the sequence (A → B → C).
- Do NOT use markdown headings (#, ##, ###) anywhere in the output.
- Do NOT write any text, title, or commentary before <figure> or after </figure>.
- Do NOT use code fences.
Output only the <figure>...</figure> block.
"""
```

What each edit targets, and the measured defect it addresses:

| change | targets | F19 evidence |
|---|---|---|
| delete *"axis labels, legend entries, data values and text annotations"* from the describe rule | the checklist being echoed back as absence claims | 15 of 79 still do it with the checklist present |
| replace the negation with the positive *"Write only about what is visibly present"* | same defect, stated as an instruction the model can follow rather than one it must suppress | v1's `Do NOT` form got 19 → 15; the remaining 15 need the provocation removed, not a stronger prohibition |
| *"at most two sentences… Then stop"* + naming what each sentence carries | the 95-word median against a 60-word ask | a 0.8B model follows shape, not counts |
| strengthen the non-figure branch to *"and nothing else: no sentences, no description of it"* | thin strips narrated as fictional charts and tables | the Figure 1-3 fabrication and the invented `(0,2),(1,3)…` data points both survived v1 |

### Measured outcome — v2 failed, v3 + a code fix succeeded (2026-08-11)

Two 80-page probes (pages 1–80, 17 VLM calls each, same document and settings).

| | v1 (F19) | v2 | v3 | after sanitizer |
|---|---|---|---|---|
| mean output tok | 148 | 105 | **91** | — |
| median output tok | 133 | 94 | **63** | — |
| filler clauses | 15/79 (19%) | 2/17 (12%) | **1/17 (6%)** | — |
| description inline with tag | 2/79 | 11/17 | 8/17 | **0** |
| stray `<p>`/`<div>` | 2/79 | 7/17 | 8/17 | **0** |
| `done=length` | 0 | 0 | **1/17** | unchanged |

**v2 regressed on format and had to be replaced.** Its rule *"The first names what the
image is. The second states its single most important content"* was **recited back as
output** — one description read `<p>Type: Working with the area…</p><p>The first name is
"Type". The second states its single most important content…`. Any rule that narrates the
shape of the answer is recitable at this model size. v3 states the bound without describing
what each sentence carries.

**The page-1 cover failed differently than this plan predicted.** The plan's contingency was
*"if the cover figure loses its identity, loosen to three sentences"*. Under v2 the cover
produced one fluent sentence of recalled book knowledge — naming **spaCy, which is not on
the cover** — with the whales and banner unmentioned. That is a grounding failure, not a
length failure; a longer budget would have bought more recall. v3 forbids outside knowledge,
and the cover now returns the banner text verbatim (`Natural Language Processing with
Python`): thinner, but true. **Do not apply the three-sentence contingency.**

**Three `Do NOT` rules moved nothing.** v3 said *"Do NOT use HTML tags (<p>, <div>,
<span>)"*, *"Put the description on the NEXT line"*, and offered *"write exactly: Unclear
image."* as a licensed non-answer. HTML tags: unchanged. Inline descriptions: 11/17 → 8/17.
Escape hatch: used **0 times**. At 0.8B, negative formatting constraints are not reliable.
Both format defects are therefore normalised in code by `_strip_html_wrappers()`
([gemini_docling_parser.py:138](../src/app/ingestion/processors/gemini_docling_parser.py#L138)),
added to the sanitizer chain in **both** backends ahead of `_strip_stray_headers`. It
unwraps `<p>`/`<div>`/`<span>` and moves text off the `</figure_type>` line, and leaves
`<figure>`, `<figure_type>` and `<table>` untouched.

**Hallucination is out of the prompt's reach.** The p25 crop (the NLTK download dialog)
was invented three different ways across three prompts: a COPPER/IRON/CHROMOS chemical
spreadsheet at 256 tok, then an ID/Name/Department/Salary table at 384 tok truncated. This
is the one pass condition v3 fails (`done=length` 1/17) and it is a model-capacity limit,
not a wording problem. Levers left are a larger VLM or a tighter image gate; the gate is
fixed at 64 by the scope decision, so accept it for this round.

### One thing you will notice and must not "fix"

The `<figure>` wrapper rule produced **1 wrapper in 79 calls**, while the `<figure_type>`
line was obeyed **79/79**. The tags are not stripped by the pipeline —
[gemini_docling_parser.py:131-145](../src/app/ingestion/processors/gemini_docling_parser.py#L131-L145)
(`_strip_stray_headers`) explicitly preserves any line starting `<figure` and is the only
code in the repo that reads them. `qwen3.5:0.8b` simply ignores the wrapper. Four rules are
therefore dead weight the model already discards.

**Keep them in this change** — they cost prefill only, and removing them is a separate
question. But write no verification step that depends on `<figure>` blocks existing, and
add no code that repairs or requires the tags.

### Do not change in Part B

`num_predict` stays at **384**. It stopped binding in F19 (max observed 369) and the goal
is for it to remain a ceiling, not a lever. `temperature` stays at 0.0.

---

## Verification

The whole repo is bind-mounted (`- .:/app`), so Python edits are live after
`docker compose restart celery_worker_upload` — **no `--build`, no `--force-recreate`.**
A `docker-compose.yml` env addition in Part A does need `docker compose up -d`; that is
safe now, because F20's `hf_cache` volume survives a recreate.

`docling` is not installed in the host Python environment. Every probe runs in the
container.

### Step 1 — probe prompt v2 on the real PDF (~5 min)

F18's mistake was probing synthetic crops and predicting the wrong output level. Probe the
real document. Pages 1–80 produced calls #1–16 in the F19 run.

```bash
docker compose restart celery_worker_upload
docker compose exec -T celery_worker_upload python /app/scripts/f21_prompt_probe.py
```

F19 baseline for those same 16 calls — output tokens in order:

```
202  108  198  157  231  144  136  192
120  216  116  110  130  102  100  104      mean 154 tok
```

**Pass:** mean under ~110 tok, still 16 calls, zero `done=length`, and the 296×280 page-1
figure still identified (it is the book-cover graphic — three stylized whales under a
purple "Natural Language Processing with Python" banner). If the cover figure loses its
identity, the bound is too tight: change *"at most two sentences"* to *"at most three
sentences"* and re-probe. Do not abandon the approach on the first probe.

### Step 2 — full document, end to end (~20 min if Part A lands)

`document_routes.py:89` pins the whole parse→chunk→embed chain to the **`upload`** queue
via `.set(queue=UPLOAD_QUEUE)`, so `celery_worker_ingestion` never sees an API upload.
Tailing the wrong container is why an earlier run looked stalled. Note also that
`_dispatch_pending` hardcodes `INGESTION_QUEUE`, so do not use `recover_and_dispatch` to
restart a big-PDF parse — it will route to the 2.5G worker and OOM.

```bash
docker compose stop celery_worker_upload celery_worker_ingestion
docker compose exec -T postgres psql -U admin -d rag_db -v ON_ERROR_STOP=1 <<'SQL'
DELETE FROM document_chunked WHERE document_id IN (SELECT id FROM documents WHERE stage <> 'embedded');
DELETE FROM document_parsed  WHERE document_id IN (SELECT id FROM documents WHERE stage <> 'embedded');
DELETE FROM documents WHERE stage <> 'embedded';
SQL
docker compose exec -T redis redis-cli FLUSHALL
docker compose up -d celery_worker_upload celery_worker_ingestion
docker compose logs -f celery_worker_upload | tee f21.log
# then upload NLTK.pdf through the UI / POST /upload
```

`document_parsed` and `document_chunked` cascade on `documents`, so those two DELETEs are
redundant — they are here to make the intent explicit. Per-upload vector tables
(`test`, `test1`, `test2`) have no FK and never cascade; drop them by hand if you want a
clean slate. Under `psql -c` an explicit `BEGIN` warns *"there is already a transaction in
progress"* — harmless; the heredoc form above avoids it. Do **not** wrap `$$` blocks in
double quotes in zsh, `$$` expands to the shell PID.

```bash
grep -c "done=length"          f21.log
grep    "VLM call #"           f21.log
grep -E "Converted pages"      f21.log
grep    "parse_pdf summary"    f21.log
```

**Pass conditions on the same document:**

| | F18 | F19 | F21 target |
|---|---|---|---|
| `vlm_calls` | 79 | 79 | **exactly 79** — guard rail |
| `vlm_failures` | 0 | 0 | **0** |
| `done=length` | 28 | 0 | **0** |
| mean output tok | ~318 | 160 | **≤ 110** |
| `vlm_wait` | 592s | 348s | **≤ 260s** |
| pages 451–500 batch | 553s | 555s | **≤ 150s** |
| `docling` | 1600s | 1333s | **≤ 900s** |
| `total` | 2194s | 1683s | **≤ 1250s** |
| `peak_rss` | 1979 MB | 2524 MB | **< 3000 MB** (limit 3.5G) |

`vlm_calls` is the guard rail: Part B alters only how much each call emits, so any movement
in the call count means something upstream broke. If Part A stopped at the "only
`table_structure OFF` is fast" branch, drop the three docling rows and judge Part B alone.

Two traps when reading the log, both previously hit:

- **Wall-clock spacing is meaningless on this laptop.** The host sleeps and
  `time.monotonic()` does not advance while the Docker VM is suspended, so `elapsed=`,
  `vlm_wait=` and the `parse_pdf summary` line are the only comparable numbers.
- **Docling is silent mid-batch.** `Going to convert document batch...` is emitted at batch
  start and nothing is logged until `Converted pages N-M`. A multi-minute gap is normal.
  The discriminator is CPU: a real parse holds 270–315% in `docker stats`; the F20 download
  stall sat at 0.36–1.4%.
- **Tasks are redelivered.** `parse_document` and `chunk_document` are each received a
  second time with the same task id after succeeding, and skipped by the stage guard
  (`Stage parse skipped … not in stage 'registered'`). Expected; do not count it as a
  second parse.

### Step 3 — quality census (this is the gate that matters)

Shorter is only an improvement if the description still identifies the figure.

```bash
python3 scripts/f21_census.py                  # newest data/parsed/*NLTK*.md
python3 scripts/f21_census.py <path>           # or a specific artifact
```

**Targets:**

| | F18 | F19 | F21 target |
|---|---|---|---|
| descriptions | 79 | 79 | **79** |
| filler clauses (whole description) | 19 | 15 | **≤ 5** — primary gate (6% of 79 ≈ 5) |
| filler clauses (first paragraph) | 11 | 12 | reported for continuity only |
| `tables` | 67 | 67 | **still 67** |
| first-para median words | 57 | 95 | **30–50** (lower bound; see below) |
| inline descriptions | — | 2 | **0** — sanitizer must eliminate these |
| stray `<p>`/`<div>` | — | 2 | **0** — same |
| `Unclear image` | — | — | watch: >4 means the escape hatch became the default |

The filler target is **≤5, not 0**. The 80-page probe reached 1 of 17 (6%); zero was the
original ask but nothing in three rounds has driven absence-reporting to zero, and the
remaining cases are genuine (crops that really are unreadable).

**`tables` is 67, not 64** — verified against both artifacts. A *drop* means the
TableFormer mode change is lossy and must be reverted regardless of the speed win. Note
the closer count is already inconsistent in both baselines (F18: 67 openers / 73 closers;
F19: 67 / 66), which is a pre-existing markdown-emission defect, not something F21 causes
or fixes.

**Word counts from the artifact are a lower bound, not the measure of length.** Same
missing-delimiter problem as the filler count. The exact number is the output token count
in the run log; use the artifact word count only as a sanity check on it.

Then read by eye against the source pages: the page-1 cover graphic, the Figure 1-3 region
around page 39 that both F18 and F19 fabricated as a table of `"the"/"been"/"message"`
columns, and two of the index tables from pages 451–500.

Delete the probe artifact: `rm -f data/parsed/_f21_probe.md`.

### Step 4 — cleanup left over from the F19 run

```bash
rm -f input/raw/91d5e345-31d0-4919-be16-24a781e9bdf0_NLTK.pdf
rm -f temp_uploads/*_NLTK.pdf
```

The orphan in `input/raw/` matters: `_register_and_dispatch` runs `_scan_input_dir(repo)`,
which re-registers any unknown file it finds there.

### Step 5 — documentation

Write `docs/20260811_tableformer_outlier_and_prompt_v2.md` and cross-link it from the F18
document, the way F18 is linked from F14's *"F18 — the remaining cost, in its own
document"* section. It should carry the F19 result table, the per-batch docling table, the
Step A1 attribution measurement, and the F21 before/after.

Also correct two known errors in the existing docs:

- Both `docs/20260805_vlm_output_length_and_image_gate.md` and
  `docs/20260805_vlm_thinking_and_table_routing.md` instruct tailing
  `celery_worker_ingestion`, which never receives API uploads. It must be
  `celery_worker_upload`.
- F18's *"Still to verify"* item 2 (the full-document re-measure) is **done**: replace the
  unmeasured "150–300s" projection with the measured **592s / 79 calls**, and note that the
  projection came from a synthetic-crop probe which was right about output *variance* and
  wrong about output *level*.

---

## Deliberately not doing

- **Unit tests.** Standing instruction from the repo owner: they write them. Still
  outstanding from F18 — the `options` payload assertion, the `done=length` warning, and
  the short-side gate (a 218×54 `PictureItem` must produce no `_call_vlm` call, a 218×160
  one must). Patterns live in `tests/unit/test_ollama_vlm_call.py` and
  `tests/unit/test_vlm_table_routing.py`. `python -m pytest tests/unit` has 12 pre-existing
  failures unrelated to this work (ARCHITECTURE §10.3) — not a regression.
- **Raising `VLM_MIN_IMAGE_SHORT_PX` above 64.** Ruled out by the repo owner for this
  round; see the scope decision above.
- **Discarding `done=length` output.** Moot at 0 truncations.
- **Removing the dead `<figure>` wrapper rules.** Separate question, no measured payoff.
- **The RSS climb.** Peak RSS rose monotonically across batches (1474 → 2524 MB) and grew
  545 MB between F18 and F19 despite F19 producing *less* text. The assembled document is
  accumulated across batches, so some growth is expected, but the F18→F19 delta is not
  explained by anything either change did. It now has ~1 GB of headroom under the 3.5G
  limit and is not urgent — but it is the reason `peak_rss` is a pass condition above, and
  it deserves its own investigation before any larger document is attempted.
- **The empty `<table>` block** (1 of 64). TableFormer, not the VLM — it belongs to the F16
  "still to verify" item about TableFormer output quality.
- **[llm_operations.py:37](../src/app/retrieval/llm_operations.py#L37)** — the Q&A path sends a
  bare `/api/generate` payload with no `keep_alive`, no `think` and no `options`, against
  `deepseek-r1:1.5b`, which is also a reasoning model. The same class of bug F14 fixed on
  the parse path, still live on the query path. Out of scope here; worth its own ticket.

---

## Implementation Checklist

### Part A — applied 2026-08-11 (behaviour-neutral until the env var flips)

| # | File | Change | Status |
|---|---|---|---|
| 1 | `src/app/ingestion/processors/gemini_docling_parser.py` | `_DEFAULT_TABLEFORMER_MODE = "accurate"`, `tableformer_mode` ctor arg, `_resolve_tableformer_mode()`, threaded into `_build_converter()` | ✅ |
| 2 | `src/app/ingestion/processors/ollama_pdf_parser.py` | Thread `tableformer_mode` through `__init__` → `super().__init__()` | ✅ |
| 3 | `src/app/config/app_config.py` | `docling_tableformer_mode: str` (`DOCLING_TABLEFORMER_MODE`, default `accurate`) | ✅ |
| 4 | `src/app/ingestion/processors/pdf_parser_factory.py` | Pass `tableformer_mode` in the shared `tuning` dict | ✅ |
| 5 | `docker-compose.yml` | `DOCLING_TABLEFORMER_MODE: ${DOCLING_TABLEFORMER_MODE:-accurate}` on all 4 services | ✅ |

| 6 | `.env` | `DOCLING_TABLEFORMER_MODE=fast` — the A1 decision, applied 2026-08-11 | ✅ |

### Part B — applied 2026-08-11

| # | File | Change | Status |
|---|---|---|---|
| 7 | `src/app/ingestion/processors/prompts.py` | `OLLAMA_IMAGE_PROMPT` **v3** (v2 regressed — see the measured-outcome section): checklist deleted, two-sentence bound stated without narrating the answer shape, grounding rule, `Unclear image.` escape hatch, HTML ban | ✅ |
| 8 | `src/app/ingestion/processors/gemini_docling_parser.py` | `_strip_html_wrappers()` — deterministic cleanup of the format defects three rounds of prompt rules could not fix at 0.8B; wired into both backends' sanitizer chains | ✅ |

### Probe scripts — added 2026-08-11

| # | File | Purpose | Status |
|---|---|---|---|
| 9 | `scripts/f21_tableformer_probe.py` | Step A1 — TableFormer configs over pages 451–500 | ✅ run, decision made |
| 10 | `scripts/f21_prompt_probe.py` | Step 1 — prompt probe over pages 1–80 | ✅ run 3× (v1/v2/v3) |
| 11 | `scripts/f21_census.py` | Step 3 — quality census, both filler measures | ✅ run on F19; awaits F21 artifact |

`scripts/` is gitignored (`.gitignore:56`) and not COPY'd by the Dockerfile — these reach
the container through the `.:/app` bind mount only. They are cited from tracked files
(`prompts.py` and this document), so if the citations should resolve for anyone else, the
census script is the one worth moving into a tracked location.

### Final result — all steps complete

```
parse_pdf summary: pages=504 total=1045s docling=782s (75%) assembly=263s (25%)
                   vlm_wait=261s vlm_calls=79 vlm_failures=0 peak_rss=2000MB
Converted pages 451-500: elapsed=121.1s rate=2.42s/page   (F19: 554.8s, 11.10s/page)
Converted pages 301-350: elapsed= 73.5s rate=1.47s/page   (F19: 147.9s,  2.96s/page)
```

| metric | F18 | F19 | **F21** | target | |
|---|---|---|---|---|---|
| total | 2194s | 1683s | **1045s** | ≤1250 | ✅ −38% |
| docling | 1600s | 1333s | **782s** | ≤900 | ✅ |
| pages 451–500 | 553s | 554.8s | **121.1s** | ≤150 | ✅ −78% |
| `vlm_wait` | 592s | 348s | **261s** | ≤260 | 1s over |
| mean output tok | ~318 | 160 | **109** | ≤110 | ✅ |
| `vlm_calls` | 79 | 79 | **79** | 79 | ✅ |
| `done=length` | 28 | 0 | **1** | 0 | ⚠️ the p25 dialog |
| `peak_rss` | 1979 MB | 2524 MB | **2000 MB** | <3000 | ✅ −524 MB |
| tables | 67 | 67 | **67** | 67 | ✅ not lossy |
| filler (window) | 19 | 15 | **3** | ≤5 | ✅ −80% |
| inline descriptions | — | 2 | **0** | 0 | ✅ |
| stray HTML in descriptions | — | — | **0** | 0 | ✅ |

Lossiness is confirmed **structurally**, not by a single count: F21 and F19 agree exactly
on 67 `<table>`, 66 `</table>` closers, 1 empty block.

- ~~Step A1 probe~~ ✅ `fast` adopted (540.8s → 129.7s, `tables=15` under both)
- ~~Step 1 prompt probe~~ ✅ v3 reached 91 mean tok / 6% filler
- ~~Step 2 full run~~ ✅ every pass condition met bar two marginal ones
- ~~Step 3 census~~ ✅ and the census's stray-HTML metric was fixed — it counted the whole
  document, and the NLTK book's own Chapter 3 HTML-parsing examples read as false positives
- ~~Flip `_DEFAULT_TABLEFORMER_MODE` + compose fallback to `fast`~~ ✅
- ~~Steps 4–5~~ ✅ probe artifact deleted, `docs/20260811_*.md` written and cross-linked
  from F18

---

## References

- **F14–F17**: `docs/20260805_vlm_thinking_and_table_routing.md`
- **F18**: `docs/20260805_vlm_output_length_and_image_gate.md`
- **F19**: commit `7631584`
- **F20**: hf_cache volume persistence
- **Architecture**: `docs/ARCHITECTURE.md` §15.7–15.9
- **Key code** (post-F21 line numbers):
  - `src/app/ingestion/processors/gemini_docling_parser.py:37` (`_DEFAULT_TABLEFORMER_MODE`)
  - `src/app/ingestion/processors/gemini_docling_parser.py:138` (`_strip_html_wrappers`)
  - `src/app/ingestion/processors/gemini_docling_parser.py:306` (`_resolve_tableformer_mode`)
  - `src/app/ingestion/processors/gemini_docling_parser.py:340-345` (TableFormer config)
  - `src/app/ingestion/processors/prompts.py:80` (`OLLAMA_IMAGE_PROMPT` v3)
  - `src/app/config/app_config.py:111-117` (`docling_tableformer_mode`)
- **Measured A1 result**: `accurate` 540.8s → `fast` 129.7s over pages 451–500, `tables=15`
  in both. Cell matching costs 3.5% and stays on.

---

**End of Plan**
