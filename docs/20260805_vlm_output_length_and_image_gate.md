# The VLM was fast per token and nobody was counting the tokens

**Date**: 2026-08-05
**Status**: implemented and smoke-tested against the live Ollama. Two things outstanding —
unit tests, and the full-document re-measure (see [Still to verify](#still-to-verify)).

## Context

`docs/20260805_vlm_thinking_and_table_routing.md` (F14–F17) turned reasoning off for the
PDF-parsing VLM and measured 85–87s → 2–6s per call. That fix is real and is live. But
the first full run with it applied still showed this:

```
VLM call #187: model=qwen3.5:0.8b, img=218x57px, 6.7KB, elapsed=17.14s
[436/504] elapsed=17.1s vlm_wait=2121s
  p437: image (218×54px) → VLM
VLM call #188: model=qwen3.5:0.8b, img=218x54px, 10.9KB, elapsed=12.02s
```

12–17s for a 218×54px image, against F14's measured 2.78s for a full-page architecture
diagram. This document is why, and it is a different cause from F14 — not a regression of
it.

Everything below is measured on the **Mac dev host** (Apple M1, 16 GB, Ollama 0.32.5
local, `qwen3.5:0.8b` Q8_0, 4096 context), the same machine as F14–F17.

---

## First, what it is not

| Check | Result |
|---|---|
| `docker exec … grep '"think"' ollama_pdf_parser.py` | present — the running image has the F14 fix |
| `docker exec … env \| grep OLLAMA_VLM_THINK` | `false` |
| `curl /api/ps` | model resident, `size_vram` 1.1 GB — not reloading |
| worker `command:` | both workers `-c 1` — `ForkPoolWorker-6` is `--max-tasks-per-child=1` recycling, not concurrency 6 |

So: thinking is off, the model is warm, and one parse is running. F14's explanations are
all exhausted.

---

## F18 — latency is decode, and output length was unbounded

The whole `celery_worker_ingestion` log of that run (504-page PDF, 191 calls) reduced to:

| | |
|---|---|
| total VLM wait | **2245s** |
| mean / median | 11.75s / 8.3s |
| range | **1.04s – 93.30s** |
| correlation with image bytes | **none** — 26.8KB → 14.5s, 6.5KB → 92.5s |

A 90× spread with no relationship to the input is not a slow machine. Probing the same
Ollama with the pipeline's own `OLLAMA_IMAGE_PROMPT` on synthetic crops:

| Input | Sampling | Latency | Tokens out |
|---|---|---|---|
| 218×54 equation strip | Ollama defaults | **10.94s** | 342 — an invented flowchart: "Line 2 has labels for 'w' and 'b'" |
| *the same request again* | Ollama defaults | **1.55s** | 22 |
| 218×54 equation strip | `temperature: 0` | **1.03s** | 22 — verbatim transcription |
| 218×133 figure | Ollama defaults | 4.37s | 126 |
| 218×133 figure | `temperature: 0` | **1.49s** | 35 |

Two facts fall out.

**Latency is pure decode.** `prompt_eval_count` is ~206 tokens in **0.26s** whatever the
crop — Qwen-VL pads small images up to a fixed tile budget, so a 6KB sliver prefills the
same as a real figure, which is exactly why image size predicts nothing — and then
`eval_duration ≈ elapsed` at a steady **~35 tok/s**. Elapsed *is* the output length.

**Output length was unbounded.** `_call_vlm` sent `model`, `prompt`, `images`, `stream`,
`keep_alive`, `think` — no `options` — so Ollama applied its defaults: `temperature 0.8`,
`top_p 0.9`, `num_predict -1`. On a 0.8B model handed a near-empty crop, sampling wanders
into invented content. The first two rows above are **byte-identical requests**: 22 tokens
one run, 342 the next. The 93.3s worst case is ~3200 tokens, i.e. generating until it hits
the 4096-token context. Nine calls over 30s accounted for 625s — 28% of the entire budget.

**Fixed** — `"options": {"temperature": 0.0, "num_predict": 384}`, both configurable.
Temperature is the lever: greedy decoding stops the wandering and, in the runs above,
produced a *correct* transcription where default sampling produced a fabricated diagram.
`num_predict` is only the ceiling that bounds the tail; at 384 it does not bind on the
~126-token honest description of a real figure.

### F18b — the size gate never caught strips

`_process_page` skipped a picture only when `width < 150` **and** `height < 150`, so it
only ever caught square icons. Every full-column, 40–60px-tall strip — horizontal rules,
equation lines, header bands — went to the VLM. From the same 191 calls:

| | calls | VLM time | share |
|---|---|---|---|
| height < 48px | 71 | 801s | 36% |
| **height < 64px** | **113** | **1348s** | **60%** |
| height < 100px | 153 | 1866s | 83% |

60% of the VLM budget was spent hallucinating descriptions of things that are not figures
— and unlike a slow call, that output does not merely cost time, it is embedded into the
chunk table and retrieved as if it were page content.

**Fixed** — `VLM_MIN_IMAGE_SHORT_PX`, default 64, gates on the **short** side. At
`images_scale=0.6` that is ~1.5in on the page, which every real figure clears. The old
both-dimensions rule is untouched and still applies first. Nothing worth keeping is lost:
the alternative for these crops is invented text entering the index.

> **Superseded by F22.** The setting is now `VLM_MIN_IMAGE_SHORT_PT`, default **107pt**,
> expressed in points rather than rendered pixels. The threshold above was pixels *at a
> particular `images_scale`*, so it silently changed physical meaning whenever the render
> resolution moved — and F22 moves it to 2.0. 107 × 0.6 = 64.2px, so the gate's behaviour
> at the old scale is unchanged; it simply no longer drifts.

### F18c — the log could not answer the question it existed to answer

F14 and F18 are the same sentence — "the model emitted far too many tokens" — with
different causes, and the `VLM call #` line carried only latency. Distinguishing them
required an out-of-band experiment against Ollama, twice.

The line now carries the token counts that were already sitting unread in the response
JSON:

```
VLM call #188: model=qwen3.5:0.8b, img=218x54px, 10.9KB, elapsed=12.02s,
               in=206 out=418 tok, 34.8 tok/s, done=stop
```

`done_reason == "length"` additionally logs a warning: that is `num_predict` truncating an
answer mid-sentence, which means either the cap is too low or the crop should never have
been sent. Neither is visible otherwise.

---

## Two traps in reading that log

- **The 05:08→06:09 hole in the timeline is the host sleeping**, not a stall. Wall-clock
  spacing between `Converted pages` lines is not usable on a laptop; `elapsed=` and
  `vlm_wait=` are.
- **`documents` held two `parsing` rows for the same PDF**, claimed 04:44 and 06:43 — two
  parses against the same single-GPU Ollama. `VLM_CONCURRENCY` bounds concurrency *within*
  one parse; `celery_worker_upload` and `celery_worker_ingestion` are separate processes
  and neither knows about the other, so it cannot bound this. F15 says that costs 27% at
  two-way. The 191 calls above all predate the second claim, so the numbers are clean, but
  any re-measure must run one document at a time.

---

## What changed

| File | Change |
|---|---|
| `src/app/ingestion/processors/ollama_pdf_parser.py` | `"options"` with `temperature` / `num_predict` in the payload; `in=`/`out=` tokens, `tok/s` and `done_reason` on the `VLM call #` line; warning on `done=length` |
| `src/app/ingestion/processors/gemini_docling_parser.py` | `_DEFAULT_MIN_IMAGE_SHORT_PX`; short-side skip in `_process_page`, logged as `too thin` |
| `src/app/config/app_config.py` | `ollama_vlm_temperature`, `ollama_vlm_num_predict`, `vlm_min_image_short_px` |
| `src/app/ingestion/processors/pdf_parser_factory.py` | Threads all three through; `min_image_short_px` joins the shared `tuning` dict so both backends get it |
| `docker-compose.yml` | The three new vars on all four app-image services |
| `.env.example` | Each with its measured justification |
| `tests/unit/test_pdf_parser_factory.py` | Kwarg assertions updated for the new signature |

| Env var | Default | Why |
|---|---|---|
| `OLLAMA_VLM_TEMPERATURE` | `0.0` | F18 — 342 tokens vs 22 on byte-identical requests at 0.8 |
| `OLLAMA_VLM_NUM_PREDICT` | `384` | F18 — ceiling for the tail; the worst call ran to 93.3s / ~3200 tokens |
| `VLM_MIN_IMAGE_SHORT_PX` | `64` | F18b — sub-64px strips were 113 of 191 calls and 60% of the VLM budget |

Smoke-tested end to end through `create_pdf_parser("ollama", AppSettings())` against the
live Ollama:

```
VLM call #1: img=218x54px,  elapsed=2.99s, in=205 out=85 tok, 43.1 tok/s, done=stop
VLM call #2: img=218x133px, elapsed=2.45s, in=219 out=79 tok, 43.5 tok/s, done=stop
```

`python -m pytest tests/unit` is back to its 12 pre-existing failures (ARCHITECTURE §10.3)
— nothing new.

---

## Verification

Clear the stuck rows and the queued tasks first, or the workers resume the old run against
pre-fix document ids:

```bash
docker compose stop celery_worker_upload celery_worker_ingestion
docker compose exec -T postgres psql -U admin -d rag_db -v ON_ERROR_STOP=1 \
  -c "DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE stage='parsing'); DELETE FROM documents WHERE stage='parsing';"
docker compose exec -T redis redis-cli FLUSHALL
```

Then re-run **one** document — an env change needs `--force-recreate`, a plain `restart`
will not pick it up:

```bash
docker compose up -d --build --force-recreate celery_worker_upload
docker compose logs -f celery_worker_upload | tee f18.log

grep "VLM call #"        f18.log   # out= well under 384, no elapsed over ~15s
grep "done=length"       f18.log   # ideally empty; if not, raise OLLAMA_VLM_NUM_PREDICT
grep -c "too thin"       f18.log   # ~113 strips skipped that used to be sent
grep "parse_pdf summary" f18.log   # measured: vlm_calls 79, vlm_wait 592s
```

> **Correction.** This block originally named `celery_worker_ingestion`. API uploads never
> reach that worker: `document_routes.py:89` pins the whole parse→chunk→embed chain to the
> **`upload`** queue via `.set(queue=UPLOAD_QUEUE)`. Tailing the ingestion worker is why an
> early run appeared to stall.

Pass condition on the same document: **`vlm_wait` under ~400s** against the 2245s
baseline, and no single call over ~15s. `vlm_failures` should be unchanged.

Then check quality, not just speed: open `data/parsed/<id>_*.md`, confirm the surviving
`<figure>` blocks still describe real figures, and confirm no page lost content that only
existed inside a skipped strip.

---

## F21 — the docling half, in its own document

This document ends with the VLM accounting for a shrinking share of the parse and docling
for 79% of it. That framing turned out to be misleading: docling was fast everywhere except
**one batch of 50 pages**, which alone was 42% of all docling time — the book's index,
handed to TableFormer as dense multi-column tables.

`docs/20260811_tableformer_outlier_and_prompt_v2.md` measures it, switches the TableFormer
structure decoder to `fast` (that batch: 555s → 121s, with an artifact structurally
identical to the `accurate` one), and takes the image prompt through two more versions plus
a `_strip_html_wrappers()` sanitizer for the format rules a 0.8B model will not obey.
Total 1683s → **1045s**, filler clauses 15 → **3**, peak RSS 2524 → **2000 MB**.

---

## Still to verify

1. **Unit tests.** Deliberately left to the author of this change's follow-up: the
   `options` payload, the `done=length` warning, and the short-side gate (a 218×54
   `PictureItem` must produce no `_call_vlm` call, a 218×160 one must). The existing
   `tests/unit/test_ollama_vlm_call.py` and `test_vlm_table_routing.py` are the pattern.
2. ~~**The full-document re-measure.**~~ **Done — measured 2026-08-10.** The full 504-page
   run gave `vlm_calls=79`, `vlm_wait=592s`, `total=2194s`, `vlm_failures=0`, mean output
   ~318 tok with **28 of 79 calls truncated** at the `num_predict` ceiling. The projection
   of 150–300s above was wrong by roughly 2–4×. It came from a synthetic-crop probe, which
   turned out to predict output *variance* correctly and output *level* badly: real figures
   from the document are far denser than the probe crops, so the model wrote much more.
   Prompt v1 (F19) then brought this to 348s / 160 tok / 0 truncations, and prompt v3
   (F21) to **261s / 109 tok / 1 truncation**. See
   `docs/20260811_tableformer_outlier_and_prompt_v2.md`.
3. **Output quality on real figures at `temperature: 0`.** The smoke test's second call
   emitted an HTML `<picture style="font-family: …">` fragment, which is a prompt problem
   rather than a sampling one, but greedy decoding will make whatever the prompt elicits
   deterministic instead of occasionally lucky. Compare a handful of `<figure>` blocks in
   `data/parsed/` against the source pages.

## Out of scope, noted

- [llm_operations.py:37](src/app/retrieval/llm_operations.py#L37) sends the same bare
  payload to `/api/generate` for the Q&A answer with `deepseek-r1:1.5b` — a reasoning
  model, with no `keep_alive`, no `think`, and now demonstrably no `options` either. The
  same three findings apply to the query path. Untouched here.
- **Reusing an `httpx.Client`** across VLM calls. Connect plus prefill is 0.26s of what
  used to be a 12s call — worth measuring only after this lands.
- **A larger VLM.** Still the real answer for tables (F16), but it makes figures slower,
  and 35 tok/s is what this host gives.
