# The VLM was reasoning, and it was reading tables it cannot read

**Date**: 2026-08-05
**Status**: measured and fixed. One item outstanding — docling's TableFormer output has
not been compared side by side against the VLM's (see [Still to verify](#still-to-verify)).

## Context

`docs/20260804_ingestion_performance_investigation.md` (F1-F13) diagnosed a ~500-page PDF
taking ~1 hour, on a **Windows/WSL2** host with **Ollama on a separate machine**. It
concluded the parse was CPU-bound on docling and deferred all tuning until a baseline run
existed.

This document covers the same question asked on the **Mac dev host**, where two of those
assumptions do not hold:

| | WSL2 investigation | This machine |
|---|---|---|
| Host | WSL2 VM, 8 logical processors | **Apple M1** — 4 performance + 4 efficiency cores, 16 GB |
| Ollama | separate machine (pure network I/O) | **local** (`/Applications/Ollama.app`), sharing the laptop |
| Docker VM | 8101 MB | 6.769 GiB, 1 GB swap |

Rather than wait for the hour-long baseline, the VLM was measured directly against the
local Ollama (0.32.5, `qwen3.5:0.8b`), using synthetic images first and then real page
crops from `input/raw/…_bert.pdf`. The result reframes the problem: the dominant cost was
never docling, and one of the fixes is to stop calling the VLM for a job it cannot do.

---

## F14 — the VLM is a reasoning model, and thinking was on

`qwen3.5:0.8b` reports `capabilities: [completion, vision, tools, **thinking**]`. The
payload in `_call_vlm` sent `model`, `prompt`, `images`, `stream`, `keep_alive` — no
`think` field — so Ollama defaulted it on.

| Prompt (700×500 synthetic table) | as shipped | `"think": false` | |
|---|---|---|---|
| `OLLAMA_IMAGE_PROMPT` | 85.02s — 3555 out tokens, 15,632 ch thinking | **6.46s** — 213 tokens | **13×** |
| `OLLAMA_TABLE_PROMPT` | 86.55s — 3621 out tokens, 14,586 ch thinking | **2.15s** — 80 tokens | **40×** |

The reasoning is then **discarded unread**. `_call_vlm` reads only
`response.json()["response"]`, and Ollama returns reasoning in a separate `thinking` key.
Every figure and table was paying 30-90s to generate ~15 KB of text nobody looks at.

### How to turn it off

Three ways were tried against the real BERT table crop:

| Method | Latency | Thinking | Answer |
|---|---|---|---|
| as shipped | 87.35s | 13,841 ch | **0 ch** |
| `"think": false` in the request body | **2.27s** | 0 | 191 ch ✅ |
| `/no_think` appended to the prompt | 87.35s | 14,658 ch | **0 ch** ❌ |

`/no_think` is a **Qwen3** convention and has no effect on qwen3.5. The request field is
the API equivalent of typing `/set nothink` in `ollama run` — that command sets a session
flag which Ollama sends as `think: false` on the request — and it is the only thing that
works here.

It is also safe across models: Ollama returns 200 for `think: false` on the non-thinking
`deepseek-r1:1.5b`, so swapping the VLM later will not break the call.

**This supersedes F13's explanation.** `keep_alive` was a real fix and stays, but the
27-33s per call F13 attributed to cold model loads was mostly reasoning tokens — F13's own
`eval_count` in the thousands does not fit a cold load.

### F14b — thinking returned *nothing* for tables, silently

Every table call with thinking on returned an **empty** `response` after ~3600 reasoning
tokens — 3 out of 3, synthetic and real. This was not an error path: `_call_vlm` returned
`""`, and `_process_page` wrapped it into `<table>\n\n\n\n</table>`. The table disappeared
from the parsed output with no warning, no failure count, and nothing in the log.

**Fixed** — a blank response now counts as a failure, logs a warning naming the model, and
falls back to `[IMAGE]`, so it surfaces in `vlm_failures` on the `parse_pdf summary:` line.

---

## F15 — VLM concurrency makes it worse when Ollama is local

F5 assumed Ollama was remote, so parallel calls were free network waits. Locally it
serializes on one GPU. Four identical calls with `think: false`:

| `VLM_CONCURRENCY` | 4 calls | per call |
|---|---|---|
| 1 | 15.47s | **3.87s** |
| 2 *(old default)* | 19.73s | 4.93s (**+27%**) |
| 4 | 82.48s | 20.62s (**+433%**) |

Default changed to **1**. Raise it only if Ollama moves to a separate machine — which is
the configuration F5 was written against.

---

## F16 — a 0.8B VLM cannot read tables, and the routing rule sent it the hardest ones

Real crops from `input/raw/…_bert.pdf` (the BERT paper), `think: false`:

| Content | Latency | Output |
|---|---|---|
| **Figure 1** — architecture diagram, p3 | **2.78s** | Correct `<figure_type>Diagram</figure_type>` and an accurate description of the pre-training/fine-tuning comparison. **Genuinely good.** |
| **Table 1** — GLUE results, 13 columns, p6 | 4.48s | Garbage. Headers came out `I, II, III, IV…`, values mangled, markdown malformed |
| **Table 4** — small, p7 | 8.82s | **Hallucinated.** Invented `Births:`, `9.86%`, and a literal `A B C D E F G H I J K…` row that appears nowhere in the paper |

The model captions figures well and cannot do table OCR at all. That is a capacity limit,
not a thinking limit — `think: false` made it fast and *wrong* instead of slow and *empty*.

**The pipeline already had a better table extractor.** `_process_page` sent a table to the
VLM only when `_is_complex_table()` was true (rows > 8 **and** cols > 6); every other table
went through `item.export_to_markdown(doc)` — docling's TableFormer, a purpose-built
table-structure model. So the rule routed precisely the *hardest* tables to the *weakest*
extractor, which is exactly where the garbage above came from.

**Fixed** — the VLM table branch is gated on `VLM_TABLES`, default **false**, so every
table goes to TableFormer. `_is_complex_table` is untouched and still applies when the flag
is on, so `VLM_TABLES=true` restores the previous behaviour exactly. That is worth doing
only with a substantially larger vision model.

---

## F17 — the Docker Desktop CPU slider had not been applied

The Resources panel showed **CPU 6 / Memory 7 GB**. The daemon disagreed:

```
docker info  →  CPUs: 4        Total Memory: 6.769GiB
```

Memory applied; **CPU did not** — that slider needs *Apply & Restart*. Any thread-count
experiment run before `docker info --format '{{.NCPU}}'` reads 6 is measured on the wrong
machine, so this is the first step of any parse-side tuning.

Two related problems on the same host:

- **CPU ceilings summed to 9.0** against 4 real VM CPUs (postgres 2 + app 2 + upload 2 +
  ingestion 2 + langfuse 1).
- **`langfuse` was not profile-gated** despite its own comment saying
  `docker compose --profile observability up langfuse`. It had no `profiles:` key — the
  only two in the file were `dev` (pgadmin) and `test` — so a plain `docker compose up -d`
  started it and it held 1G of a 6.77 GiB VM.

**Fixed** — workers raised to `cpus: "4.0"`, postgres and app dropped to `"1.0"` (measured
peaks: 44 MiB and idle), `langfuse` given `profiles: [observability]`. `docker compose
config --services` now lists six services and langfuse is not among them.

---

## What changed

| File | Change |
|---|---|
| `src/app/ingestion/processors/ollama_pdf_parser.py` | `"think"` in the payload; blank response → counted failure + `[IMAGE]` |
| `src/app/ingestion/processors/gemini_docling_parser.py` | VLM table branch gated on `_vlm_tables`; `_VLM_CONCURRENCY` 2 → 1 |
| `src/app/config/app_config.py` | `ollama_vlm_think`, `vlm_tables`, `vlm_concurrency` 2 → 1 |
| `src/app/ingestion/processors/pdf_parser_factory.py` | Threads both new settings through |
| `docker-compose.yml` | `OLLAMA_VLM_THINK`, `VLM_TABLES`, `VLM_CONCURRENCY` 1, `DOCLING_NUM_THREADS` 4, worker `cpus` 4.0, postgres/app 1.0, `langfuse` profile-gated |
| `.env.example` | Each of the above with its measured justification |

New settings, all measured rather than guessed:

| Env var | Default | Why |
|---|---|---|
| `OLLAMA_VLM_THINK` | `false` | F14 — 87s → 2.3s, reasoning discarded unread |
| `VLM_TABLES` | `false` | F16 — VLM table output is garbage or empty; TableFormer is better |
| `VLM_CONCURRENCY` | `1` | F15 — 2 is 27% slower, 4 is 5× slower against a local Ollama |
| `DOCLING_NUM_THREADS` | `4` | F2 + F17 — targets the M1's 4 performance cores |

Tests: `tests/unit/test_ollama_vlm_call.py` (think flag, blank-response guard, `thinking`
field never merged into the markdown) and `tests/unit/test_vlm_table_routing.py` (no VLM
call for any table by default; `vlm_tables=True` restores the complexity-gated path). Both
verified to fail against the pre-fix code.

---

## Still to verify

1. **Docling's TableFormer output on `bert.pdf` Table 1.** `docling_core` is not installed
   in the local env, so it could not be run side by side against the VLM output above. The
   prior is strong — it is a dedicated table-structure model and the pipeline already
   trusted it for every simple table — but it is a prior, not a measurement. After the
   first ingest, check Table 1 in `data/parsed/<id>_bert.md` against page 6 of the PDF. If
   TableFormer is also poor, `VLM_TABLES=true` reverts and the real fix is a larger VLM.
2. **The parse-side numbers.** F17's CPU changes and `DOCLING_NUM_THREADS=4` are still
   unmeasured on this machine. Apply the Docker slider, confirm `docker info` reads 6, then
   compare `rate=Xs/page` from the `Converted pages` log lines on the same document and
   page range — F12's lesson is that a table-heavy book versus a prose-heavy one produces a
   4× difference from document complexity alone.

## Verification

```bash
docker compose up -d --build --force-recreate celery_worker_upload
docker compose logs -f celery_worker_upload | tee assemble.log

grep "VLM call #"        assemble.log   # elapsed= low single digits, not 27-90s
grep "parse_pdf summary" assemble.log   # vlm_calls drops too — tables no longer routed here
grep "Converted pages"   assemble.log   # rate=Xs/page for the parse side
docker info --format '{{.NCPU}}'        # must read 6 before trusting any thread number
```

- `vlm_wait` should fall by roughly an order of magnitude against the 3593s in F13.
- `vlm_failures` may now be **non-zero**. That is F14b becoming visible, not a regression —
  any count above zero means content was being dropped silently before.

## Out of scope, noted

[llm_operations.py:37](src/app/retrieval/llm_operations.py#L37) sends the same bare payload
to `/api/generate` for the Q&A answer with `deepseek-r1:1.5b` — also a reasoning model,
also with no `keep_alive` and no `think`. Same class of problem on the query path; not
touched here. The embedding stage is untouched, as it is already fast.
