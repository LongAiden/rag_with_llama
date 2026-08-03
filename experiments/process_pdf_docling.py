import re
import os
import time
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# ── Docling ───────────────────────────────────────────────────────────────────
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    TableItem,
    PictureItem,
    TextItem,
    SectionHeaderItem,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("hybrid_parser")

# ── Prompts ───────────────────────────────────────────────────────────────────
VLM_TABLE_PROMPT = """\
This is a cropped region from a PDF page containing a table.
Extract and render the table content.

Rules:
- Wrap the output in <table></table> tags (closing tag must be </table>).
- Optionally include <table_caption>Title</table_caption> before the table \
if a caption or title is visible above or below it.
- Use GitHub-flavoured markdown table syntax.
- Separator row must use ONLY dashes: |-|-| (no colons, no padding spaces).
- Do NOT use any HTML inside cells (no <br>, no <td>, no <tr>).
- If a cell contains multiple lines of text, join them with a single space \
in the SAME cell.
- Ensure every row has the same number of columns.
- If a cell is merged vertically (one value applies to multiple rows), keep \
the value in the top row only.
- If the table has multiple header rows, include all of them in order.
- Preserve all cell values exactly as they appear.
- Output only the <table>...</table> block — no commentary or preamble.
- Do NOT use code fences.
"""

VLM_IMAGE_PROMPT = """\
This is an image, chart, diagram, figure, or visual element cropped from a PDF page.
Extract and preserve ALL content inside <figure></figure> tags.

Rules:
- Transcribe ALL visible text EXACTLY as it appears, in its original language.
  Do NOT translate, paraphrase, or describe text — copy it verbatim.
- If the image contains Japanese text, output it in Japanese. NEVER translate it to English.
- Do NOT write any English description or summary of the diagram structure.
- Preserve logical reading order (top-to-bottom, left-to-right within each column).
- For diagrams or flowcharts: transcribe ONLY the visible text labels, captions,
  and annotations verbatim in the original language. No English descriptions.
- For charts/graphs: transcribe all axis labels, legend entries, and data values.
- For colored boxes, banners, or callout regions: include ALL text inside them.
- If the image also contains a table, render it as a GFM markdown table.
- ALL transcribed text must be placed inside the <figure>...</figure> block.
- Output only the <figure>...</figure> block — no commentary or preamble.
- Do NOT use code fences.
"""

# ── Config ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_RPM_LIMIT = 5    # safe limit (model allows 30, keep buffer)
IMAGES_SCALE = 1.0   # docling render resolution (2× = ~144 dpi equivalent)
COMPLEX_TABLE_ROWS = 8     # tables with more rows  → Gemini
COMPLEX_TABLE_COLS = 6     # tables with more cols  → Gemini
PAGE_COUNT_THRESHOLD = 50  # docs with more pages use parse_pdf_page_by_page


# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_calls: int = GEMINI_RPM_LIMIT, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque = deque()

    def wait(self):
        while True:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
            if len(self._calls) < self.max_calls:
                break
            sleep_for = self.period - (now - self._calls[0])
            logger.info(
                f"[rate limiter] quota reached, sleeping {sleep_for:.1f}s …")
            time.sleep(sleep_for)
        self._calls.append(time.monotonic())


# ── Post-processing helpers ───────────────────────────────────────────────────
def _normalize_table(table_md: str) -> str:
    lines = table_md.splitlines()
    result: list[str] = []
    for line in lines:
        if not line.strip().startswith("|"):
            result.append(line)
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[-:\s]+", c) for c in cells if c):
            result.append("|" + "|".join(["-"] * len(cells)) + "|")
            continue
        if cells and cells[0] == "" and any(c for c in cells[1:]):
            if result:
                prev = [c.strip()
                        for c in result[-1].strip().strip("|").split("|")]
                for i, cont in enumerate(cells):
                    if cont and i < len(prev):
                        prev[i] = (prev[i] + " " + cont).strip()
                result[-1] = "| " + " | ".join(prev) + " |"
                continue
        result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


def _clean_html(md: str) -> str:
    md = re.sub(r"\s*<br\s*/?>\s*", " ", md, flags=re.IGNORECASE)
    md = re.sub(r"</?t[rdh]\b[^>]*>",  "", md, flags=re.IGNORECASE)
    return md


def _strip_code_fences(md: str) -> str:
    md = re.sub(r"^```(?:markdown)?\s*\n?", "", md, flags=re.IGNORECASE)
    md = re.sub(r"\n?```\s*$",             "", md, flags=re.IGNORECASE)
    return md.strip()


def _fix_table_closing_tags(md: str) -> str:
    lines, result, in_table = md.splitlines(), [], False
    for line in lines:
        s = line.strip()
        if s == "<table>":
            if in_table:
                result.append("</table>")
                in_table = False
            else:
                result.append(line)
                in_table = True
        elif s == "</table>":
            result.append(line)
            in_table = False
        else:
            result.append(line)
    return "\n".join(result)


def normalize_tables_in_markdown(md: str) -> str:
    out, buf = [], []

    def flush():
        if buf:
            out.extend(_normalize_table("\n".join(buf)).splitlines())
            buf.clear()
    for line in md.splitlines():
        if line.strip().startswith("|"):
            buf.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


# ── Main parser ───────────────────────────────────────────────────────────────
class HybridDoclingGeminiParser:
    """
    Hybrid PDF → Markdown parser using only Docling + Gemini (no fitz).

    Image acquisition (all via docling):
      PictureItem  → item.get_image(doc)            (pre-cropped by docling)
      TableItem    → item.get_image(doc)            (cropped from page image)
      Page image   → doc.pages[n].image.pil_image   (full page, for reference)

    Docling pipeline options that make this work:
      generate_page_images=True       required for item.get_image() fallback
      generate_picture_images=True    pre-crops PictureItems
      images_scale=IMAGES_SCALE       controls resolution of all rendered images
    """

    def __init__(
        self,
        api_key: str = GOOGLE_API_KEY,
        gemini_model: str = GEMINI_MODEL,
        rpm_limit: int = GEMINI_RPM_LIMIT,
        images_scale: float = IMAGES_SCALE,
        complex_table_rows: int = COMPLEX_TABLE_ROWS,
        complex_table_cols: int = COMPLEX_TABLE_COLS,
    ):
        self._api_key = api_key
        self._gemini_model = gemini_model
        self._rate_limiter = RateLimiter(rpm_limit)
        self._images_scale = images_scale
        self._complex_table_rows = complex_table_rows
        self._complex_table_cols = complex_table_cols
        self._genai_model = None
        self._vlm_calls: int = 0

    # ── Docling converter ─────────────────────────────────────────────────────

    def _build_converter(self) -> DocumentConverter:
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True)
        opts.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )
        # ── Image rendering (replaces fitz) ──────────────────────────────
        opts.generate_page_images = True   # enables item.get_image(doc) crop
        opts.generate_picture_images = True   # pre-crops PictureItems
        opts.images_scale = self._images_scale
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts)}
        )

    # ── Gemini client ─────────────────────────────────────────────────────────

    def _get_model(self):
        if self._genai_model is not None:
            return self._genai_model
        import google.generativeai as genai
        genai.configure(api_key=self._api_key)
        self._genai_model = genai.GenerativeModel(self._gemini_model)
        return self._genai_model

    def _call_gemini(self, pil_img: PILImage.Image, prompt: str, retries: int = 3) -> str:
        """Rate-limited Gemini call. Accepts a PIL Image directly — no bytes conversion."""
        self._rate_limiter.wait()
        model = self._get_model()
        for attempt in range(retries):
            try:
                self._vlm_calls += 1
                return model.generate_content([pil_img, prompt]).text
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err or "resource_exhausted" in err:
                    if "per_day" in err or "day" in err:
                        raise RuntimeError(
                            "[Gemini] Daily quota exhausted.") from exc
                    wait = 60
                    logger.warning(
                        f"RPM limit hit (attempt {attempt+1}/{retries}), waiting {wait}s …"
                    )
                    time.sleep(wait)
                    self._rate_limiter._calls.clear()
                    self._rate_limiter.wait()
                else:
                    raise
        raise RuntimeError(f"Gemini call failed after {retries} retries")

    def _call_vlm(self, pil_img: PILImage.Image, prompt: str) -> str:
        raw = self._call_gemini(pil_img, prompt)
        raw = _strip_code_fences(raw)
        raw = normalize_tables_in_markdown(raw)
        return raw

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _is_complex_table(self, table: TableItem) -> bool:
        try:
            return (
                table.data.num_rows > self._complex_table_rows
                or table.data.num_cols > self._complex_table_cols
            )
        except Exception:
            return False


    def _count_pages(self, pdf_path: str) -> int:
        """Count PDF pages without loading into docling."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        count = len(pdf)
        pdf.close()
        return count

    @staticmethod
    def _item_sort_key(item) -> tuple[float, float]:
        """Return (y, x) sort key in top-left origin from docling provenance."""
        if not item.prov:
            return (0.0, 0.0)
        bbox = item.prov[0].bbox
        # Docling bbox: l/t/r/b in BOTTOMLEFT origin → t is top (largest y).
        # For sort order we want top-of-element first (largest bbox.t = closest to top).
        # Negate t so highest-on-page items sort first.
        return (-bbox.t, bbox.l)

    # ── Per-page assembly ─────────────────────────────────────────────────────

    def _find_same_band_items(self, anchor, all_items, h_gap_thresh: float = 20.0):
        """Return TextItems/SectionHeaderItems in the same horizontal Y-band as
        *anchor* (PictureItem) and horizontally adjacent within h_gap_thresh pts.
        These are merged into the anchor picture Gemini crop."""
        if not anchor.prov:
            return []
        ab = anchor.prov[0].bbox
        a_page = anchor.prov[0].page_no
        a_height = max(ab.t - ab.b, 1.0)
        result = []
        for item in all_items:
            if item is anchor:
                continue
            if not isinstance(item, (TextItem, SectionHeaderItem)):
                continue
            if not item.prov or item.prov[0].page_no != a_page:
                continue
            ib = item.prov[0].bbox
            v_overlap = min(ab.t, ib.t) - max(ab.b, ib.b)
            if v_overlap / a_height < 0.2:
                continue
            if ib.l > ab.r:
                h_gap = ib.l - ab.r
            elif ib.r < ab.l:
                h_gap = ab.l - ib.r
            else:
                h_gap = 0
            if h_gap <= h_gap_thresh:
                result.append(item)
        return result

    def _expand_and_crop(self, doc, page_no: int, bboxes, padding: int = 8):
        """Crop the union of *bboxes* from the page image (BOTTOMLEFT → PIL).
        At images_scale pts/px: pix = coord * images_scale."""
        page = doc.pages.get(page_no)
        if page is None or page.image is None or page.image.pil_image is None:
            return None
        pil_full = page.image.pil_image
        img_w, img_h = pil_full.size
        ph = page.size.height
        l = min(b.l for b in bboxes)
        bot = min(b.b for b in bboxes)
        r = max(b.r for b in bboxes)
        t = max(b.t for b in bboxes)
        sc = self._images_scale
        pix_l = max(0,     int(l * sc) - padding)
        pix_t = max(0,     int((ph - t) * sc) - padding)
        pix_r = min(img_w, int(r * sc) + padding)
        pix_b = min(img_h, int((ph - bot) * sc) + padding)
        if pix_r <= pix_l or pix_b <= pix_t:
            return None
        return pil_full.crop((pix_l, pix_t, pix_r, pix_b))

    def _process_page(self, page_no: int, items: list, doc) -> str:
        """Assemble one page markdown in reading order.

        Pre-pass: text items in the same horizontal band as a PictureItem are
        merged into its Gemini crop (expanded bbox) and skipped individually.
        This prevents adjacent styled text from being garbled by docling OCR.
        """
        # ── Pre-pass ─────────────────────────────────────────────────────────
        adjacent_texts: dict = {}
        skip_ids: set = set()
        for item in items:
            if not isinstance(item, PictureItem) or not item.prov:
                continue
            band = self._find_same_band_items(item, items)
            if band:
                adjacent_texts[id(item)] = band
                skip_ids.update(id(t) for t in band)

        # ── Main pass ─────────────────────────────────────────────────────────
        ordered: list = []

        for item in items:
            if not item.prov:
                continue
            y, x = self._item_sort_key(item)

            if id(item) in skip_ids:
                continue  # absorbed into adjacent picture Gemini crop

            # ── Image → docling crop → Gemini ────────────────────────────────
            if isinstance(item, PictureItem):
                adj = adjacent_texts.get(id(item))
                if adj:
                    bboxes = [item.prov[0].bbox] + \
                        [t.prov[0].bbox for t in adj]
                    pil = self._expand_and_crop(
                        doc, item.prov[0].page_no, bboxes)
                    if pil is None:
                        pil = item.get_image(doc)
                    label = f"image+{len(adj)} text-items"
                else:
                    pil = item.get_image(doc)
                    label = "image"
                if pil is None:
                    logger.warning(
                        f"  p{page_no}: PictureItem has no image, skipping")
                    continue
                logger.info(
                    f"  p{page_no}: {label} ({pil.width}×{pil.height}px) → Gemini")
                md = self._call_vlm(pil, VLM_IMAGE_PROMPT).strip()

            # ── Complex table → docling crop → Gemini ────────────────────────
            elif isinstance(item, TableItem) and self._is_complex_table(item):
                pil = item.get_image(doc)
                if pil is None:
                    logger.warning(
                        f"  p{page_no}: complex table has no image, falling back to Docling"
                    )
                    try:
                        md = item.export_to_markdown()
                    except Exception:
                        md = str(item.data)
                    md = f"<table>\n\n{md}\n\n</table>"
                else:
                    logger.info(
                        f"  p{page_no}: complex table "
                        f"({item.data.num_rows}×{item.data.num_cols}, "
                        f"{pil.width}×{pil.height}px) → Gemini"
                    )
                    md = self._call_vlm(pil, VLM_TABLE_PROMPT).strip()
                    if not md.startswith("<table>"):
                        md = f"<table>\n\n{md}\n\n</table>"

            # ── Simple table → Docling ────────────────────────────────────────
            elif isinstance(item, TableItem):
                logger.debug(
                    f"  p{page_no}: simple table "
                    f"({item.data.num_rows}×{item.data.num_cols}) → Docling"
                )
                try:
                    md = item.export_to_markdown()
                except Exception:
                    try:
                        md = item.export_to_dataframe().to_markdown(index=False)
                    except Exception:
                        md = str(item.data)
                md = f"<table>\n\n{md}\n\n</table>"

            # ── Section header → Docling ──────────────────────────────────────
            elif isinstance(item, SectionHeaderItem):
                level = "#" * min(max(item.level, 1), 6)
                md = f"{level} {item.text}"

            # ── Body text → Docling ───────────────────────────────────────────
            elif isinstance(item, TextItem):
                md = (item.text or "").strip()
                if not md:
                    continue

            else:
                continue

            ordered.append((y, x, md))

        ordered.sort(key=lambda t: (t[0], t[1]))
        body = "\n\n".join(md for _, _, md in ordered)
        return f"[Page {page_no}]\n\n{body}"

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        page_threshold: int = PAGE_COUNT_THRESHOLD,
    ) -> str:
        """
        Auto-routing entry point. Counts pages first, then selects the best method:
          < page_threshold pages → parse_pdf()         (fast, all pages in RAM at once)
          ≥ page_threshold pages → parse_pdf_page_by_page()  (slower, low peak memory)
        """
        total_pages = self._count_pages(pdf_path)
        if total_pages < page_threshold:
            logger.info(
                f"[parse] {total_pages} pages < threshold ({page_threshold}) "
                f"→ using parse_pdf (full-document, fast)"
            )
            return self.parse_pdf(pdf_path, output_path)
        else:
            logger.warning(
                f"[parse] {total_pages} pages ≥ threshold ({page_threshold}) "
                f"→ using parse_pdf_page_by_page (low-memory, slower)"
            )
            return self.parse_pdf_page_by_page(pdf_path, output_path)

    def parse_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Parse a PDF to markdown. No fitz — image rendering is 100% via docling.

        Args:
            pdf_path:    Path to the PDF file.
            output_path: Optional output path; each page is flushed immediately.
        Returns:
            Full markdown string.
        """
        self._vlm_calls = 0
        pdf_path = str(pdf_path)

        logger.info(
            f"═══ HybridDoclingGeminiParser (fitz-free): {Path(pdf_path).name} ═══")

        # ── Step 1: Docling full-document conversion ───────────────────────────
        # generate_page_images + generate_picture_images enable image access
        # via item.get_image(doc) without any fitz calls.
        logger.info("Step 1/3  Docling converting "
                    "(OCR + layout + tables + page/picture image rendering) …")
        conv = self._build_converter().convert(pdf_path)
        doc = conv.document

        # ── Step 2: Group elements by page ────────────────────────────────────
        logger.info("Step 2/3  Grouping elements by page …")
        page_items: dict[int, list] = defaultdict(list)
        for item, _level in doc.iterate_items():
            if item.prov:
                page_items[item.prov[0].page_no].append(item)

        total_pages = len(doc.pages)
        total_items = sum(len(v) for v in page_items.values())
        logger.info(f"{total_pages} pages, {total_items} elements")

        # ── Step 3: Page-by-page assembly ─────────────────────────────────────
        logger.info("Step 3/3  Assembling pages …")
        pages_md = []
        out_file = None

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_file = open(output_path, "w", encoding="utf-8")

        try:
            for page_no in range(1, total_pages + 1):
                print(f"[{page_no}/{total_pages}] … ", end="", flush=True)
                try:
                    page_md = self._process_page(
                        page_no=page_no,
                        items=page_items.get(page_no, []),
                        doc=doc,
                    )
                    page_md = normalize_tables_in_markdown(page_md)
                    page_md = _clean_html(page_md)
                    page_md = _fix_table_closing_tags(page_md)

                    chunk = page_md + "\n\n---\n\n"
                    pages_md.append(chunk)

                    if out_file:
                        out_file.write(chunk)
                        out_file.flush()

                    print("done")
                except Exception as exc:
                    if out_file:
                        out_file.write(chunk)
                        out_file.flush()
                    print(f"ERROR: {exc}")
                    logger.error(f"[page {page_no}] {exc}", exc_info=True)
        finally:
            if out_file:
                out_file.close()

        markdown = "".join(pages_md)
        if output_path:
            logger.info(f"Saved → {output_path}")

        print(f"\nDone. Gemini calls: {self._vlm_calls} (cropped regions only)  "
              f"|  pages: {total_pages}")
        return markdown

    # ── Alternative: true page-by-page conversion ─────────────────────────────
    def parse_pdf_page_by_page(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Alternative entry point: converts one page at a time using
        docling's page_range=(n, n) parameter.

        Trade-off: lower peak memory, but each convert() call re-initialises
        the docling pipeline, making this significantly slower than parse_pdf()
        for large documents. Use only when memory is the bottleneck.
        """
        self._vlm_calls = 0
        pdf_path = str(pdf_path)

        total_pages = self._count_pages(pdf_path)
        logger.info(f"Total pages: {total_pages}")

        pages_md = []
        out_file = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_file = open(output_path, "w", encoding="utf-8")

        converter = self._build_converter()

        try:
            for page_no in range(1, total_pages + 1):
                print(f"[{page_no}/{total_pages}] converting … ",
                      end="", flush=True)
                try:
                    # Convert exactly one page — docling page_range is 1-based inclusive
                    conv = converter.convert(
                        pdf_path, page_range=(page_no, page_no))
                    doc = conv.document

                    items = []
                    for item, _level in doc.iterate_items():
                        if item.prov:
                            items.append(item)

                    page_md = self._process_page(
                        page_no=page_no,
                        items=items,
                        doc=doc,
                    )
                    page_md = normalize_tables_in_markdown(page_md)
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

        markdown = "".join(pages_md)
        if output_path:
            logger.info(f"Saved → {output_path}")

        print(f"\nDone. Gemini calls: {self._vlm_calls} (cropped regions only)  "
              f"|  pages: {total_pages}")
        return markdown
