import base64
import io
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image as PILImage

from app.ingestion.processors.gemini_docling_parser import (
    GeminiDoclingParser,
    _strip_code_fences,
    _strip_stray_headers,
    _normalize_tables_in_markdown,
    _clean_html,
    _fix_table_closing_tags,
    _fix_markdown_headings,
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

    Key differences from GeminiDoclingParser:
    - _call_vlm: routes to Ollama with simpler prompts + fallback on failure
    - parse_pdf: page-batched conversion; heading levels are assigned via bbox height (inherited from base)
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        vlm_model: str = "qwen3.5:0.8b",
        vlm_timeout: float = 300.0,
        images_scale: float = 0.6,
        complex_table_rows: int = 8,
        complex_table_cols: int = 6,
        max_pages: Optional[int] = None,
        h1_min_height: float = 20.0,
        h2_min_height: float = 11.0,
        h3_min_height: float = 9.0,
        min_image_px: int = 150,
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
        )
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._vlm_model = vlm_model
        self._vlm_timeout = vlm_timeout
        self._max_pages = max_pages

    def get_backend_name(self) -> str:
        return "ollama-docling"

    def _call_vlm(self, pil_img: PILImage.Image, prompt: str) -> str:
        if prompt is _VLM_IMAGE_PROMPT or prompt == _VLM_IMAGE_PROMPT:
            prompt = _OLLAMA_IMAGE_PROMPT
        elif prompt is _VLM_TABLE_PROMPT or prompt == _VLM_TABLE_PROMPT:
            prompt = _OLLAMA_TABLE_PROMPT

        try:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_size_kb = len(buf.getvalue()) / 1024
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            self._vlm_calls += 1
            logger.info(
                f"VLM call #{self._vlm_calls}: model={self._vlm_model}, "
                f"img={pil_img.width}x{pil_img.height}px, {img_size_kb:.1f}KB, "
                f"url={self._ollama_base_url}/api/generate"
            )
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
                },
                timeout=timeout,
            )
            response.raise_for_status()
            raw = response.json()["response"]
            raw = _strip_code_fences(raw)
            raw = _strip_stray_headers(raw)
            raw = _normalize_tables_in_markdown(raw)
            return raw
        except Exception as exc:
            logger.warning(f"VLM call failed ({type(exc).__name__}: {exc}) — falling back to [IMAGE]")
            return "[IMAGE]"

    def parse_pdf(self, path, output_path=None) -> str:
        """Parse PDF using page-batched Docling conversion + Ollama VLM for images/tables."""
        self._vlm_calls = 0
        pdf_path = str(path)

        total_pages = self._count_pages(pdf_path)
        if self._max_pages is not None:
            total_pages = min(total_pages, self._max_pages)
        
        logger.info(f"Docling converting (page-batched): {Path(pdf_path).name}, {total_pages} pages")
        
        converter = self._build_converter()
        batch_docs = {}
        
        for batch_start in range(1, total_pages + 1, _DOCLING_PAGE_BATCH_SIZE):
            batch_end = min(batch_start + _DOCLING_PAGE_BATCH_SIZE - 1, total_pages)
            logger.info(f"  Converting pages {batch_start}-{batch_end}...")
            
            conv = converter.convert(pdf_path, page_range=(batch_start, batch_end))
            doc = conv.document
            batch_docs[(batch_start, batch_end)] = doc
        
        page_items: dict[int, list] = defaultdict(list)
        page_doc_map: dict[int, any] = {}
        
        for (batch_start, batch_end), doc in batch_docs.items():
            for item, _ in doc.iterate_items():
                if item.prov:
                    page_no = item.prov[0].page_no
                    page_items[page_no].append(item)
                    page_doc_map[page_no] = doc

        logger.info(f"Assembling {total_pages} pages with Ollama VLM ({self._vlm_model})")

        pages_md = []
        out_file = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_file = open(output_path, "w", encoding="utf-8")

        try:
            with ThreadPoolExecutor(max_workers=_VLM_CONCURRENCY) as executor:
                for page_no in range(1, total_pages + 1):
                    print(f"[{page_no}/{total_pages}] assembling … ", end="", flush=True)
                    try:
                        doc = page_doc_map.get(page_no)
                        if not doc:
                            logger.warning(f"  p{page_no}: no document found, skipping")
                            continue
                        
                        page_md = self._process_page(
                            page_no=page_no,
                            items=page_items.get(page_no, []),
                            doc=doc,
                            executor=executor
                        )
                        page_md = _normalize_tables_in_markdown(page_md)
                        page_md = _clean_html(page_md)
                        page_md = _fix_table_closing_tags(page_md)

                        chunk = page_md + "\n\n---\n\n"
                        pages_md.append(chunk)
                        if out_file:
                            out_file.write(chunk)
                            out_file.flush()
                        print("done")
                    except Exception as exc:
                        print(f"ERROR: {exc}")
                        logger.error(f"[page {page_no}] {exc}", exc_info=True)
        finally:
            if out_file:
                out_file.close()

        markdown = _fix_markdown_headings("".join(pages_md))
        print(f"\nDone. Ollama VLM calls: {self._vlm_calls} | pages: {total_pages}")
        return markdown
