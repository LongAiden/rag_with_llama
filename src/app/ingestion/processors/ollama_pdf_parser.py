import base64
import io
import logging
import time
from typing import Optional

import httpx
from PIL import Image as PILImage

from app.ingestion.processors.gemini_docling_parser import (
    GeminiDoclingParser,
    _strip_code_fences,
    _strip_html_wrappers,
    _strip_stray_headers,
    _normalize_tables_in_markdown,
    _DEFAULT_DOCLING_NUM_THREADS,
    _DEFAULT_IMAGES_SCALE,
    _DEFAULT_MIN_IMAGE_SHORT_PT,
    _DEFAULT_TABLEFORMER_MODE,
    _DEFAULT_VLM_TABLES,
    _DOCLING_PAGE_BATCH_SIZE,
    _VLM_CONCURRENCY,
)
from app.ingestion.processors.prompts import (
    VLM_IMAGE_PROMPT as _VLM_IMAGE_PROMPT,
    VLM_TABLE_PROMPT as _VLM_TABLE_PROMPT,
    OLLAMA_IMAGE_PROMPT as _OLLAMA_IMAGE_PROMPT,
    OLLAMA_TABLE_PROMPT as _OLLAMA_TABLE_PROMPT,
)

logger = logging.getLogger(__name__)


class OllamaPDFParser(GeminiDoclingParser):
    """
    Hybrid PDF → Markdown using Docling layout extraction and a locally-hosted
    Ollama vision model for complex tables and figures.

    The only difference from GeminiDoclingParser is `_call_vlm`, which routes to
    Ollama with simpler prompts and falls back to [IMAGE] on failure, plus the
    tuned defaults below. `parse_pdf` is inherited — the batched, streaming
    conversion lives in the base class so the two backends cannot drift apart.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        vlm_model: str = "qwen3.5:0.8b",
        vlm_timeout: float = 300.0,
        keep_alive: str = "30m",
        think: bool = False,
        temperature: float = 0.0,
        num_predict: int = 384,
        images_scale: float = _DEFAULT_IMAGES_SCALE,
        complex_table_rows: int = 8,
        complex_table_cols: int = 6,
        max_pages: Optional[int] = None,
        h1_min_height: float = 20.0,
        h2_min_height: float = 11.0,
        h3_min_height: float = 9.0,
        min_image_px: int = 150,
        docling_num_threads: int = _DEFAULT_DOCLING_NUM_THREADS,
        page_batch_size: int = _DOCLING_PAGE_BATCH_SIZE,
        vlm_concurrency: int = _VLM_CONCURRENCY,
        vlm_tables: bool = _DEFAULT_VLM_TABLES,
        min_image_short_pt: float = _DEFAULT_MIN_IMAGE_SHORT_PT,
        tableformer_mode: str = _DEFAULT_TABLEFORMER_MODE,
    ):
        super().__init__(
            api_key=None,
            gemini_model=None,
            rpm_limit=999,
            images_scale=images_scale,
            complex_table_rows=complex_table_rows,
            complex_table_cols=complex_table_cols,
            h1_min_height=h1_min_height,
            h2_min_height=h2_min_height,
            h3_min_height=h3_min_height,
            min_image_px=min_image_px,
            max_pages=max_pages,
            docling_num_threads=docling_num_threads,
            page_batch_size=page_batch_size,
            vlm_concurrency=vlm_concurrency,
            vlm_tables=vlm_tables,
            min_image_short_pt=min_image_short_pt,
            tableformer_mode=tableformer_mode,
        )
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._vlm_model = vlm_model
        self._vlm_timeout = vlm_timeout
        self._keep_alive = keep_alive
        self._think = think
        self._temperature = temperature
        self._num_predict = num_predict

    def get_backend_name(self) -> str:
        return "ollama-docling"

    def _call_vlm(self, pil_img: PILImage.Image, prompt: str) -> str:
        if prompt is _VLM_IMAGE_PROMPT or prompt == _VLM_IMAGE_PROMPT:
            prompt = _OLLAMA_IMAGE_PROMPT
        elif prompt is _VLM_TABLE_PROMPT or prompt == _VLM_TABLE_PROMPT:
            prompt = _OLLAMA_TABLE_PROMPT

        started = time.monotonic()
        try:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_size_kb = len(buf.getvalue()) / 1024
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            timeout = httpx.Timeout(
                connect=10.0,
                read=self._vlm_timeout,
                write=10.0,
                pool=10.0,
            )
            response = httpx.post(
                f"{self._ollama_base_url}/api/generate",
                json={
                    "model": self._vlm_model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    # Without this Ollama unloads the model 5 minutes after its
                    # last use, so a gap between figure-bearing pages makes the
                    # next call pay a full cold model load.
                    "keep_alive": self._keep_alive,
                    # qwen3.5 is a reasoning model and Ollama defaults thinking
                    # ON. Measured on a real BERT table crop: 87.35s thinking
                    # versus 2.27s without, and the reasoning is discarded
                    # unread — it arrives in a separate `thinking` field. This
                    # request field is the API equivalent of `/set nothink`;
                    # appending `/no_think` to the prompt is a Qwen3 convention
                    # and has no effect here.
                    "think": self._think,
                    # Latency here is pure decode — measured against this same
                    # local Ollama, prefill is 206 tokens in 0.26s regardless of
                    # crop size, then eval_duration ≈ elapsed at ~35 tok/s. So
                    # elapsed IS the output length, and Ollama's defaults
                    # (temperature 0.8, num_predict -1) leave it unbounded: the
                    # same 218×54px equation strip came back as 22 tokens in
                    # 1.55s and as 342 tokens of invented flowchart in 10.94s on
                    # consecutive identical requests, and the worst call of a
                    # 191-call run took 93.3s (~3200 tokens, i.e. running to the
                    # 4096 context). Greedy decoding collapses that to 22-35
                    # tokens and transcribes the crop instead of inventing one;
                    # num_predict is the ceiling for the tail, not the lever.
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._num_predict,
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload["response"]
            raw = _strip_code_fences(raw)
            raw = _strip_html_wrappers(raw)
            raw = _strip_stray_headers(raw)
            raw = _normalize_tables_in_markdown(raw)

            elapsed = time.monotonic() - started

            # A 200 with an empty body is a silent content loss, not a success:
            # with thinking on, every table call returned exactly this after
            # ~3600 reasoning tokens, and _process_page then emitted an empty
            # <table> with nothing in the log to show the table had been dropped.
            if not raw.strip():
                self._record_vlm_call(elapsed, failed=True)
                logger.warning(
                    f"VLM returned an empty response after {elapsed:.2f}s "
                    f"(model={self._vlm_model}, think={self._think}, "
                    f"img={pil_img.width}x{pil_img.height}px) — falling back to [IMAGE]"
                )
                return "[IMAGE]"

            call_no = self._record_vlm_call(elapsed)
            # Token counts, not just latency: elapsed alone cannot distinguish a
            # slow host from a model that decided to emit 3000 tokens, which is
            # what made F14/F18 take two rounds of investigation to tell apart.
            in_tok = payload.get("prompt_eval_count") or 0
            out_tok = payload.get("eval_count") or 0
            eval_ns = payload.get("eval_duration") or 0
            tok_s = out_tok / (eval_ns / 1e9) if eval_ns else 0.0
            done_reason = payload.get("done_reason", "?")
            logger.info(
                f"VLM call #{call_no}: model={self._vlm_model}, "
                f"img={pil_img.width}x{pil_img.height}px, {img_size_kb:.1f}KB, "
                f"elapsed={elapsed:.2f}s, in={in_tok} out={out_tok} tok, "
                f"{tok_s:.1f} tok/s, done={done_reason}"
            )
            # "length" means num_predict truncated the answer mid-sentence.
            # Either the cap is too low or this crop should never have been sent.
            if done_reason == "length":
                logger.warning(
                    f"VLM call #{call_no} hit num_predict={self._num_predict} "
                    f"(img={pil_img.width}x{pil_img.height}px) — output truncated"
                )
            return raw
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._record_vlm_call(elapsed, failed=True)
            # The status and body matter: a non-vision model answers 400 with an
            # explanatory message, and the [IMAGE] fallback below is otherwise
            # completely silent about why every figure was dropped.
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" status={exc.response.status_code} body={exc.response.text[:300]!r}"
            logger.warning(
                f"VLM call failed after {elapsed:.2f}s "
                f"({type(exc).__name__}: {exc}){detail} — falling back to [IMAGE]"
            )
            return "[IMAGE]"
