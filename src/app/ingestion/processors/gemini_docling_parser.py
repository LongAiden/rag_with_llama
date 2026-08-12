import re
import time
import logging
import threading
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from app.ingestion.processors.pdf_parser_base import PDFParserBase
from app.ingestion.processors.prompts import VLM_IMAGE_PROMPT as _VLM_IMAGE_PROMPT, VLM_TABLE_PROMPT as _VLM_TABLE_PROMPT

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_RPM_LIMIT = 10
_DEFAULT_COMPLEX_TABLE_ROWS = 8
_DEFAULT_COMPLEX_TABLE_COLS = 6
# Docling renders page images at 72 * images_scale DPI, and _expand_and_crop cuts
# every VLM crop out of that page image — so this single number decides whether
# the VLM can read anything. It was effectively 0.6 (OllamaPDFParser overrode the
# old 0.75 default and nothing exposed it), i.e. 43 DPI, at which an 8pt glyph is
# 5px tall. Measured consequence over a 504-page run: output length was inversely
# correlated with crop area — the four longest descriptions were the four
# SMALLEST crops (218x96 -> 384 tok and truncated, 200x67 -> 362, 211x85 -> 325,
# 187x72 -> 311) while the largest crops came back at 227-263 tok. A full-width
# screenshot of the NLTK downloader arrived as 218x96px and was described as an
# invented ID/Name/Department/Salary table repeated six times. That is
# confabulation from an illegible input, not verbosity, and no prompt fixes it:
# three prompt versions failed on that one crop, and v3's explicit
# "write exactly: Unclear image." escape hatch went unused in 79 of 79 calls.
#
# 2.0 = 144 DPI, 16px for an 8pt glyph. Costs checked before adopting: a Letter
# page is ~5.8MB, so ~290MB per 50-page batch against ~1.5G of headroom, and
# ~+270 image tokens (~+0.3s) per call in prefill. Decode dominates latency and
# should shorten as the model stops guessing.
_DEFAULT_IMAGES_SCALE = 2.0
_DEFAULT_MIN_IMAGE_PX = 150
# Short-side floor, in POINTS (1/72in) rather than rendered pixels, so it keeps
# its meaning when _DEFAULT_IMAGES_SCALE moves. The both-dimensions rule above
# only ever caught square icons, so every full-column 40-60px-tall strip — rules,
# equation lines, header bands — still went to the VLM: 113 of 191 calls in a
# 504-page run were under the floor and burned 1348s (60% of the VLM budget)
# producing invented descriptions that then get embedded.
#
# 107pt ~= 1.48in. This reproduces the previous behaviour exactly: the old
# constant was 64 pixels at images_scale 0.6, and 107 * 0.6 = 64.2. Expressed in
# pixels it would have silently become a 0.44in gate at scale 2.0, admitting
# every thin strip it exists to reject.
_DEFAULT_MIN_IMAGE_SHORT_PT = 107
# Crop padding, also in points, for the same reason. 13pt * 0.6 = 7.8 ~= the 8px
# it replaces, so behaviour at the old scale is unchanged. Because the gate
# measures the padded crop, the two constants together define the real floor:
# an item passes when (content_pt + 2*13) >= 107, i.e. content >= 81pt — at any
# images_scale, which is the whole point. Leaving this at a fixed 8px made the
# floor 80pt at scale 0.6 but 99pt at scale 2.0 and silently dropped 9 figures.
_DEFAULT_CROP_PADDING_PT = 13
_DOCLING_PAGE_BATCH_SIZE = 50
# TableFormer structure decoder: "accurate" or "fast". docling time on a 504-page
# book is not uniform — pages 451-500 (the dense multi-column index, which the
# layout model classifies as tables) took 554.8s at 11.10 s/page against 1.47
# s/page across the seven healthy batches: 42% of all docling time in one batch
# of 50, reproduced three times.
#
# "fast" because it is a large win at no measured cost. Isolated on those 50
# pages: 540.8s -> 129.7s (-76%) with tables=15 under both modes. Over the whole
# document: that batch 554.8s -> 121.1s, docling 1333s -> 782s, and the artifact
# is structurally identical to the "accurate" one — 67 <table>, 66 closers, 1
# empty block, the same counts either way. Disabling do_cell_matching on top of
# this saves a further 3.5% (129.7s -> 125.1s), which is not worth the cell
# accuracy, so _build_converter keeps it on.
#
# Both checkpoints ship in the same HF snapshot (fast 139 MB, accurate 203 MB),
# so switching costs no download. Set DOCLING_TABLEFORMER_MODE=accurate to revert
# if a document type shows table loss; an unknown value falls back to docling's
# own default with a warning rather than failing the parse.
_DEFAULT_TABLEFORMER_MODE = "fast"
# 1, not 2: Ollama runs on the same host and serializes on one GPU. Measured on
# an M1 with think disabled — 3.87s/call at 1, 4.93s at 2, 20.62s at 4.
_VLM_CONCURRENCY = 1
_DEFAULT_DOCLING_NUM_THREADS = 2
# Tables go to docling's TableFormer, not the VLM. See _process_page.
_DEFAULT_VLM_TABLES = False


def _rss_mb() -> float:
    """Resident set size in MB, or 0.0 where /proc is unavailable.

    Deliberately dependency-free (no psutil): the number is only needed inside
    the Linux worker container, where it is what shows whether peak memory is
    flat across batches or growing with total page count.
    """
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ── Rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, max_calls: int = _DEFAULT_RPM_LIMIT, period: float = 60.0):
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
            logger.info(f"[rate limiter] quota reached, sleeping {sleep_for:.1f}s …")
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
                prev = [c.strip() for c in result[-1].strip().strip("|").split("|")]
                for i, cont in enumerate(cells):
                    if cont and i < len(prev):
                        prev[i] = (prev[i] + " " + cont).strip()
                result[-1] = "| " + " | ".join(prev) + " |"
                continue
        result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


def _clean_html(md: str) -> str:
    md = re.sub(r"\s*<br\s*/?>\s*", " ", md, flags=re.IGNORECASE)
    md = re.sub(r"</?t[rdh]\b[^>]*>", "", md, flags=re.IGNORECASE)
    return md


def _strip_code_fences(md: str) -> str:
    md = re.sub(r"^```(?:markdown)?\s*\n?", "", md, flags=re.IGNORECASE)
    md = re.sub(r"\n?```\s*$", "", md, flags=re.IGNORECASE)
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


def _strip_html_wrappers(md: str) -> str:
    """Unwrap HTML block tags from VLM prose and put the description on its own line.

    `qwen3.5:0.8b` wraps descriptions in <p>/<div>/<span> and frequently emits the
    description on the </figure_type> line itself. Measured over an 80-page probe:
    7 stray <p> and 1 <div> across 17 descriptions, and 8 of 17 inline — against
    2 of 79 inline in the previous full run. Three rounds of prompt rules, the
    last of which said "Do NOT use HTML tags (<p>, <div>, <span>)" and "Put the
    description on the NEXT line" in as many words, moved neither number. At 0.8B
    the model does not reliably honour negative formatting constraints, so these
    are normalised here, where the outcome is guaranteed rather than requested.

    <figure>, <figure_type> and <table> are deliberately left alone: they are the
    structural tags the assembler and `_strip_stray_headers` depend on.
    """
    # Closing block tags are paragraph breaks; opening tags carry no content.
    md = re.sub(r"</(?:p|div)\s*>", "\n", md, flags=re.IGNORECASE)
    md = re.sub(r"<(?:p|div|span)\b[^>]*>", "", md, flags=re.IGNORECASE)
    md = re.sub(r"</span\s*>", "", md, flags=re.IGNORECASE)
    # Move description text that landed on the </figure_type> line onto the next.
    md = re.sub(r"(</figure_type>)[ \t]*(?=\S)", r"\1\n", md, flags=re.IGNORECASE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _strip_stray_headers(md: str) -> str:
    """Remove markdown heading lines that appear outside <figure>, <table>, or fenced blocks."""
    lines, result, inside, in_fence = md.splitlines(), [], False, False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue
        if stripped.startswith('<figure') or stripped.startswith('<table'):
            inside = True
        if stripped.startswith('</figure>') or stripped.startswith('</table>'):
            result.append(line)
            inside = False
            continue
        if not inside and re.match(r'^#{1,6}\s', stripped):
            continue
        result.append(line)
    return '\n'.join(result)


def _detect_column_split(items: list, page_width: float) -> float | None:
    """Return page midpoint if items cluster into two columns, else None."""
    centroids = []
    for item in items:
        if item.prov:
            b = item.prov[0].bbox
            centroids.append((b.l + b.r) / 2)
    if len(centroids) < 4:
        return None
    split = page_width / 2
    left   = sum(1 for c in centroids if c < page_width * 0.45)
    right  = sum(1 for c in centroids if c > page_width * 0.55)
    middle = sum(1 for c in centroids if page_width * 0.45 <= c <= page_width * 0.55)
    total  = len(centroids)
    if left >= 2 and right >= 2 and middle / total < 0.15 and (left + right) / total > 0.70:
        return split
    return None


def _normalize_tables_in_markdown(md: str) -> str:
    out, buf = [], []
    in_fence = False

    def flush():
        if buf:
            out.extend(_normalize_table("\n".join(buf)).splitlines())
            buf.clear()

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flush()
            out.append(line)
            continue
        if in_fence:
            flush()
            out.append(line)
            continue
        if stripped.startswith("|"):
            buf.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


_NUMBERED_HEADING_RE = re.compile(
    r'^(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]{2,60})$'
)
_ALLCAPS_LINE_RE = re.compile(r'^[A-Z][A-Z\s\-/]{4,50}$')


def _fix_markdown_headings(md: str) -> str:
    """
    Post-processing pass to promote plain-text lines that look like headings
    but were emitted as TextItem by Docling (not SectionHeaderItem).

    Rules (applied per-line, skipping content inside <table>…</table> and
    fenced code blocks):
      1. Numbered sections: "1. Title", "1.1 Title", "2.3.4 Title" → ## or ###
      2. ALL-CAPS short line → ##
    Lines already starting with '#' are left unchanged.
    """
    lines = md.splitlines()
    result = []
    in_table = False
    in_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue

        if in_fence:
            result.append(line)
            continue

        # Track table regions to avoid corrupting table content
        if '<table>' in stripped.lower():
            in_table = True
        if '</table>' in stripped.lower():
            in_table = False
            result.append(line)
            continue

        if in_table or stripped.startswith('#') or not stripped:
            result.append(line)
            continue

        # Rule 1: numbered section pattern
        m = _NUMBERED_HEADING_RE.match(stripped)
        if m:
            numbering = m.group(1).rstrip('.')
            depth = numbering.count('.')  # "1" → 0, "1.1" → 1, "1.1.1" → 2
            prefix = '##' if depth == 0 else '###'
            result.append(f"{prefix} {stripped}")
            continue

        # Rule 2: ALL-CAPS short line (e.g. "INTRODUCTION", "KEY CONCEPTS")
        if _ALLCAPS_LINE_RE.match(stripped):
            result.append(f"## {stripped}")
            continue

        result.append(line)

    return "\n".join(result)


_DOCTEST_SPLIT_RE = re.compile(r"\s+(?=(?:>>>|\.\.\.)\s)")


def _format_code(item) -> str:
    """Fence a code item and restore doctest line breaks.

    Docling joins code cells with spaces by the time we see them, so fencing
    alone leaves one long line. For interactive sessions the `>>>` / `...`
    prompts are the line boundaries and can be restored exactly — the split
    consumes only the whitespace before the prompt, preserving indentation.
    Non-REPL code stays fenced but unbroken; bbox-cell reconstruction is out
    of scope.
    """
    text = (item.text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+>>>\s*$", "", text)
    text = re.sub(r"\s+\.\.\.\s*$", "", text)
    if ">>>" in text or "..." in text:
        lines = _DOCTEST_SPLIT_RE.split(text)
        text = "\n".join(lines)
    lang = getattr(item, "code_language", None)
    lang_str = ""
    if lang is not None:
        lang_val = getattr(lang, "value", str(lang)).lower()
        if lang_val and lang_val != "unknown":
            lang_str = lang_val
    return f"```{lang_str}\n{text}\n```"


def _format_list_item(item) -> str:
    """Emit a list item with its parsed marker."""
    text = (item.text or "").strip()
    if not text:
        return ""
    marker = getattr(item, "marker", None)
    if marker:
        return f"{marker} {text}"
    enumerated = getattr(item, "enumerated", False)
    prefix = "1." if enumerated else "-"
    return f"{prefix} {text}"


def _wrap_figure(md: str, caption: str = "") -> str:
    """Wrap VLM output in a deterministic <figure> block.

    Strips whatever <figure> tags the model did or did not emit, then re-adds
    them with the caption as a <figure_caption> line inside the block. The
    0.8B model does not reliably emit the wrapper, so it is enforced here.
    """
    md = re.sub(r"</?figure[^>]*>", "", md, flags=re.IGNORECASE).strip()
    inner = md
    if caption:
        inner = f"{md}\n<figure_caption>{caption}</figure_caption>"
    return f"<figure>\n{inner}\n</figure>"


# ── Main parser ───────────────────────────────────────────────────────────────
class GeminiDoclingParser(PDFParserBase):
    """
    Hybrid PDF → Markdown using Docling layout extraction and Gemini
    for complex tables (> complex_table_rows rows or > complex_table_cols cols)
    and figures. Image rendering is 100% via docling.
    """

    def __init__(
        self,
        api_key: str,
        gemini_model: str = "gemini-2.5-flash",
        rpm_limit: int = _DEFAULT_RPM_LIMIT,
        images_scale: float = _DEFAULT_IMAGES_SCALE,
        complex_table_rows: int = _DEFAULT_COMPLEX_TABLE_ROWS,
        complex_table_cols: int = _DEFAULT_COMPLEX_TABLE_COLS,
        h1_min_height: float = 20.0,
        h2_min_height: float = 11.0,
        h3_min_height: float = 9.0,
        min_image_px: int = _DEFAULT_MIN_IMAGE_PX,
        max_pages: Optional[int] = None,
        docling_num_threads: int = _DEFAULT_DOCLING_NUM_THREADS,
        page_batch_size: int = _DOCLING_PAGE_BATCH_SIZE,
        vlm_concurrency: int = _VLM_CONCURRENCY,
        vlm_tables: bool = _DEFAULT_VLM_TABLES,
        min_image_short_pt: float = _DEFAULT_MIN_IMAGE_SHORT_PT,
        tableformer_mode: str = _DEFAULT_TABLEFORMER_MODE,
    ):
        self._api_key = api_key
        self._gemini_model = gemini_model
        self._rate_limiter = _RateLimiter(rpm_limit)
        self._images_scale = images_scale
        self._complex_table_rows = complex_table_rows
        self._complex_table_cols = complex_table_cols
        self._h1_min_height = h1_min_height
        self._h2_min_height = h2_min_height
        self._h3_min_height = h3_min_height
        self._min_image_px = min_image_px
        self._min_image_short_pt = min_image_short_pt
        # Resolved once: the gate is authored in points but compared against
        # rendered pixels, and the two only agree at one value of images_scale.
        self._min_image_short_px = int(round(min_image_short_pt * images_scale))
        logger.info(
            "VLM image gate: %.0fpt (%.2fin) = %dpx at images_scale %.2f (%.0f DPI)",
            min_image_short_pt,
            min_image_short_pt / 72.0,
            self._min_image_short_px,
            images_scale,
            72.0 * images_scale,
        )
        self._max_pages = max_pages
        self._docling_num_threads = docling_num_threads
        self._page_batch_size = page_batch_size
        self._vlm_concurrency = vlm_concurrency
        self._vlm_tables = vlm_tables
        self._tableformer_mode = tableformer_mode
        self._genai_model = None

        # VLM counters are written from `vlm_concurrency` pool threads. Without
        # the lock they silently under-report, which would defeat the whole
        # point of measuring the VLM share of the parse.
        self._vlm_stats_lock = threading.Lock()
        self._vlm_calls: int = 0
        self._vlm_seconds: float = 0.0
        self._vlm_failures: int = 0
        # `_vlm_seconds` is total time spent inside VLM calls on the pool threads;
        # `_vlm_blocked_seconds` is how much of that the assembly loop actually
        # waited for. Pipelining is meant to drive the second toward zero while
        # leaving the first unchanged, so both are needed to see it worked.
        self._vlm_blocked_seconds: float = 0.0

    def get_backend_name(self) -> str:
        return "gemini-docling"

    # ── Docling converter ─────────────────────────────────────────────────────

    def _resolve_tableformer_mode(self):
        """Map the configured mode string onto docling's `TableFormerMode`.

        Returns None when the enum is missing or the string is unrecognised, in
        which case `TableStructureOptions` keeps its own default. A typo in an
        env var must not take the parse down.
        """
        try:
            from docling.datamodel.pipeline_options import TableFormerMode
        except ImportError:
            logger.warning(
                "TableFormerMode not importable in this docling version; "
                "using the built-in table structure default"
            )
            return None
        try:
            return TableFormerMode(str(self._tableformer_mode).strip().lower())
        except ValueError:
            logger.warning(
                "Unknown DOCLING_TABLEFORMER_MODE %r; expected one of %s. "
                "Falling back to the built-in default.",
                self._tableformer_mode,
                [m.value for m in TableFormerMode],
            )
            return None

    def _build_converter(self):
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        ts_kwargs = {"do_cell_matching": True}
        mode = self._resolve_tableformer_mode()
        if mode is not None:
            ts_kwargs["mode"] = mode
        opts.table_structure_options = TableStructureOptions(**ts_kwargs)
        opts.accelerator_options = AcceleratorOptions(
            num_threads=self._docling_num_threads, device=AcceleratorDevice.AUTO
        )
        opts.generate_page_images = True
        opts.generate_picture_images = True
        opts.images_scale = self._images_scale
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    # ── Gemini client ─────────────────────────────────────────────────────────

    def _get_model(self):
        if self._genai_model is not None:
            return self._genai_model
        import google.generativeai as genai
        genai.configure(api_key=self._api_key)
        self._genai_model = genai.GenerativeModel(self._gemini_model)
        return self._genai_model

    def _record_vlm_call(self, elapsed: float, failed: bool = False) -> int:
        """Accumulate VLM timing under the lock; returns the new call count."""
        with self._vlm_stats_lock:
            self._vlm_calls += 1
            self._vlm_seconds += elapsed
            if failed:
                self._vlm_failures += 1
            return self._vlm_calls

    def _call_gemini(self, pil_img: PILImage.Image, prompt: str, retries: int = 3) -> str:
        self._rate_limiter.wait()
        model = self._get_model()
        for attempt in range(retries):
            started = time.monotonic()
            try:
                text = model.generate_content([pil_img, prompt]).text
                self._record_vlm_call(time.monotonic() - started)
                return text
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err or "resource_exhausted" in err:
                    if "per_day" in err or "day" in err:
                        self._record_vlm_call(time.monotonic() - started, failed=True)
                        raise RuntimeError("[Gemini] Daily quota exhausted.") from exc
                    wait = 60
                    logger.warning(f"RPM limit hit (attempt {attempt+1}/{retries}), waiting {wait}s …")
                    time.sleep(wait)
                    self._rate_limiter._calls.clear()
                    self._rate_limiter.wait()
                else:
                    self._record_vlm_call(time.monotonic() - started, failed=True)
                    raise
        self._record_vlm_call(0.0, failed=True)
        raise RuntimeError(f"Gemini call failed after {retries} retries")

    def _call_vlm(self, pil_img: PILImage.Image, prompt: str) -> str:
        raw = self._call_gemini(pil_img, prompt)
        raw = _strip_code_fences(raw)
        raw = _strip_html_wrappers(raw)
        raw = _strip_stray_headers(raw)
        raw = _normalize_tables_in_markdown(raw)
        return raw

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_complex_table(self, table) -> bool:
        try:
            return (
                table.data.num_rows > self._complex_table_rows
                and table.data.num_cols > self._complex_table_cols
            )
        except Exception:
            return False

    def _count_pages(self, pdf_path: str) -> int:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        count = len(pdf)
        pdf.close()
        return count

    @staticmethod
    def _item_sort_key(item) -> tuple[float, float]:
        if not item.prov:
            return (0.0, 0.0)
        bbox = item.prov[0].bbox
        return (-bbox.t, bbox.l)

    # ── Per-page assembly ─────────────────────────────────────────────────────

    def _find_same_band_items(self, anchor, all_items, h_gap_thresh: float = 20.0):
        from docling_core.types.doc import TextItem, SectionHeaderItem
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
            h_gap = max(0, ib.l - ab.r) if ib.l > ab.r else max(0, ab.l - ib.r)
            if h_gap <= h_gap_thresh:
                result.append(item)
        return result

    def _expand_and_crop(self, doc, page_no: int, bboxes, padding_pt: float = _DEFAULT_CROP_PADDING_PT):
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
        # Padding scales with the render, or the short-side gate silently tightens
        # as resolution rises: the gate sees crop = content*scale + 2*padding, so a
        # fixed 8px pad made the effective page-space floor 80pt at scale 0.6 but
        # 99pt at scale 2.0 — 24% stricter, which dropped 9 real figures.
        padding = max(1, int(round(padding_pt * sc)))
        pix_l = max(0,     int(l * sc) - padding)
        pix_t = max(0,     int((ph - t) * sc) - padding)
        pix_r = min(img_w, int(r * sc) + padding)
        pix_b = min(img_h, int((ph - bot) * sc) + padding)
        if pix_r <= pix_l or pix_b <= pix_t:
            return None
        return pil_full.crop((pix_l, pix_t, pix_r, pix_b))

    def _build_page(
        self, page_no: int, items: list, doc, executor: Optional[ThreadPoolExecutor] = None
    ) -> tuple[list, list]:
        """Walk one page's items, submitting VLM work without waiting for it.

        Returns `(ordered, vlm_tasks)`. Everything that needs `doc` happens here —
        crops are cut eagerly into independent PIL images and `ordered` holds only
        strings — so the caller may `del doc` before handing these to
        `_finalize_page`. That is what lets a batch's VLM calls run underneath the
        next batch's `convert()`.
        """
        from docling_core.types.doc import TableItem, PictureItem, TextItem, SectionHeaderItem

        page = doc.pages.get(page_no)
        page_width = page.size.width if page and page.size else None
        split_x = _detect_column_split(items, page_width) if page_width else None

        def _col_idx(item) -> int:
            if split_x is None or not item.prov:
                return 0
            cx = (item.prov[0].bbox.l + item.prov[0].bbox.r) / 2
            return 0 if cx <= split_x else 1

        adjacent_texts: dict = {}
        skip_ids: set = set()
        caption_texts: set = set()
        for item in items:
            if not isinstance(item, PictureItem) or not item.prov:
                continue
            band = self._find_same_band_items(item, items)
            if band:
                adjacent_texts[id(item)] = band
                skip_ids.update(id(t) for t in band)
        for item in items:
            try:
                cap = getattr(item, "caption_text", None)
                if callable(cap):
                    cap_text = cap(doc)
                    if cap_text:
                        caption_texts.add(cap_text.strip())
            except Exception:
                pass

        ordered: list = []
        vlm_tasks: list[tuple[int, float, float, Future, str, str]] = []

        for item in items:
            if not item.prov:
                continue
            y, x = self._item_sort_key(item)
            col = _col_idx(item)

            if id(item) in skip_ids:
                continue
            if isinstance(item, TextItem):
                item_text = (item.text or "").strip()
                if item_text and item_text in caption_texts:
                    continue

            if isinstance(item, PictureItem):
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                adj = adjacent_texts.get(id(item))
                if adj:
                    bboxes = [item.prov[0].bbox] + [t.prov[0].bbox for t in adj]
                    pil = self._expand_and_crop(doc, item.prov[0].page_no, bboxes)
                    if pil is None:
                        pil = item.get_image(doc)
                    label = f"image+{len(adj)} text-items"
                else:
                    pil = item.get_image(doc)
                    label = "image"
                if pil is None:
                    logger.warning(f"  p{page_no}: PictureItem has no image, skipping")
                    continue
                if self._min_image_px > 0 and (pil.width < self._min_image_px and pil.height < self._min_image_px):
                    print(f"  p{page_no}: {label} too small ({pil.width}×{pil.height}px < {self._min_image_px}px), skipping VLM", flush=True)
                    continue
                if self._min_image_short_px > 0 and min(pil.width, pil.height) < self._min_image_short_px:
                    print(
                        f"  p{page_no}: {label} too thin ({pil.width}×{pil.height}px, "
                        f"short side < {self._min_image_short_px}px), skipping VLM", flush=True
                    )
                    continue
                print(f"  p{page_no}: {label} ({pil.width}×{pil.height}px) → VLM", flush=True)
                
                if executor:
                    future = executor.submit(self._call_vlm, pil, _VLM_IMAGE_PROMPT)
                    vlm_tasks.append((col, y, x, future, "figure", caption))
                else:
                    md = self._call_vlm(pil, _VLM_IMAGE_PROMPT).strip()
                    md = _wrap_figure(md, caption)
                    ordered.append((col, y, x, md))

            # Tables go to docling's TableFormer unless the VLM is explicitly
            # opted in. Measured on bert.pdf with qwen3.5:0.8b: the 13-column
            # GLUE table came back with headers "I, II, III, IV…", and a small
            # table came back with invented rows. The old rule sent only tables
            # with >8 rows AND >6 cols here — i.e. the hardest ones to the
            # weakest extractor.
            elif isinstance(item, TableItem) and self._vlm_tables and self._is_complex_table(item):
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                pil = item.get_image(doc)
                if pil is None:
                    logger.warning(f"  p{page_no}: complex table has no image, falling back to Docling")
                    try:
                        md = item.export_to_markdown(doc)
                    except Exception:
                        md = str(item.data)
                    if caption:
                        md = f"{md}\n<figure_caption>{caption}</figure_caption>"
                    md = f"<table>\n\n{md}\n\n</table>"
                    ordered.append((col, y, x, md))
                else:
                    print(
                        f"  p{page_no}: complex table "
                        f"({item.data.num_rows}×{item.data.num_cols}, "
                        f"{pil.width}×{pil.height}px) → VLM", flush=True
                    )
                    
                    if executor:
                        future = executor.submit(self._call_vlm, pil, _VLM_TABLE_PROMPT)
                        vlm_tasks.append((col, y, x, future, "table", caption))
                    else:
                        md = self._call_vlm(pil, _VLM_TABLE_PROMPT).strip()
                        if not md.startswith("<table>"):
                            md = f"<table>\n\n{md}\n\n</table>"
                        if caption:
                            md = md.replace("</table>", f"<figure_caption>{caption}</figure_caption>\n</table>")
                        ordered.append((col, y, x, md))

            elif isinstance(item, TableItem):
                logger.debug(f"  p{page_no}: simple table ({item.data.num_rows}×{item.data.num_cols}) → Docling")
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                try:
                    md = item.export_to_markdown(doc)
                except Exception:
                    try:
                        md = item.export_to_dataframe().to_markdown(index=False)
                    except Exception:
                        md = str(item.data)
                if caption:
                    md = f"{md}\n<figure_caption>{caption}</figure_caption>"
                md = f"<table>\n\n{md}\n\n</table>"
                ordered.append((col, y, x, md))

            elif isinstance(item, SectionHeaderItem):
                bbox_height = item.prov[0].bbox.t - item.prov[0].bbox.b
                if bbox_height > self._h1_min_height:
                    prefix = "#"
                elif bbox_height > self._h2_min_height:
                    prefix = "##"
                elif bbox_height > self._h3_min_height:
                    prefix = "###"
                else:
                    md = (item.text or "").strip()
                    if not md:
                        continue
                    ordered.append((col, y, x, md))
                    continue
                md = f"{prefix} {item.text}"
                ordered.append((col, y, x, md))

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

            else:
                continue

        return ordered, vlm_tasks

    def _finalize_page(self, page_no: int, ordered: list, vlm_tasks: list) -> str:
        """Join `_build_page`'s VLM futures and render the page markdown."""
        if vlm_tasks:
            for col, y, x, future, kind, caption in vlm_tasks:
                # Time the stall OUTSIDE the stats lock. `_record_vlm_call` takes
                # that same lock from the pool thread, so holding it across
                # `.result()` would have the main thread waiting on a future whose
                # thread is waiting on the main thread — a hard hang.
                blocked_started = time.monotonic()
                try:
                    md = future.result().strip()
                    if kind == "table":
                        if not md.startswith("<table>"):
                            md = f"<table>\n\n{md}\n\n</table>"
                        if caption:
                            md = md.replace("</table>", f"<figure_caption>{caption}</figure_caption>\n</table>")
                    else:
                        md = _wrap_figure(md, caption)
                    ordered.append((col, y, x, md))
                except Exception as exc:
                    logger.error(f"VLM call failed: {exc}")
                    ordered.append((col, y, x, "[IMAGE]"))
                finally:
                    blocked = time.monotonic() - blocked_started
                    with self._vlm_stats_lock:
                        self._vlm_blocked_seconds += blocked

        ordered.sort(key=lambda t: (t[0], t[1], t[2]))
        body = "\n\n".join(md for _, _, _, md in ordered)
        return f"[PAGE:{page_no}]\n\n{body}"

    def _process_page(self, page_no: int, items: list, doc, executor: Optional[ThreadPoolExecutor] = None) -> str:
        """Build and finalize one page in a single call.

        The synchronous path. `parse_pdf` pipelines the two halves instead, but
        this stays the simple entry point for callers and tests.
        """
        ordered, vlm_tasks = self._build_page(page_no, items, doc, executor)
        return self._finalize_page(page_no, ordered, vlm_tasks)

    def _emit_batch(self, built: list, pages_md: list, out_file, total_pages: int) -> None:
        """Finalize, normalize and write out one already-built batch.

        `built` is `[(page_no, ordered, vlm_tasks), …]` from `_build_page`. It
        holds no reference to a `DoclingDocument`, so the caller releases the doc
        before calling this.
        """
        for page_no, ordered, vlm_tasks in built:
            page_started = time.monotonic()
            try:
                page_md = self._finalize_page(page_no, ordered, vlm_tasks)
                page_md = _normalize_tables_in_markdown(page_md)
                page_md = _clean_html(page_md)
                page_md = _fix_table_closing_tags(page_md)

                chunk = page_md + "\n\n---\n\n"
                pages_md.append(chunk)
                if out_file:
                    out_file.write(chunk)
                    out_file.flush()
            except Exception as exc:
                logger.error(f"[page {page_no}] finalize failed: {exc}", exc_info=True)
                continue

            page_elapsed = time.monotonic() - page_started
            if page_elapsed > 1.0:
                with self._vlm_stats_lock:
                    vlm_wait = self._vlm_seconds
                    vlm_blocked = self._vlm_blocked_seconds
                logger.info(
                    f"[{page_no}/{total_pages}] elapsed={page_elapsed:.1f}s "
                    f"vlm_wait={vlm_wait:.0f}s blocked={vlm_blocked:.0f}s"
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_pdf(self, path: str | Path, output_path: Optional[str | Path] = None) -> str:
        """Parse a PDF to markdown, one page batch at a time, VLM work pipelined.

        Each batch is converted, built and released before the next `convert()`
        runs, so peak memory is O(batch size) rather than O(total pages): the
        `DoclingDocument`s — and the rendered page images they carry — are never
        all alive at once.

        The *join* of a batch's VLM futures is deferred by one batch, so those
        calls decode while docling converts the next batch instead of after it:

            convert(N) → build(N) → convert(N+1) → emit(N) → build(N+1) → …

        Markdown still comes out in page order; streaming to `output_path` just
        lags one batch behind. `vlm_blocked` in the summary is the metric that
        shows the overlap working — it should be roughly one batch's VLM time,
        not the whole run's.
        """
        with self._vlm_stats_lock:
            self._vlm_calls = 0
            self._vlm_seconds = 0.0
            self._vlm_failures = 0
            self._vlm_blocked_seconds = 0.0

        pdf_path = str(path)
        name = Path(pdf_path).name
        started = time.monotonic()

        total_pages = self._count_pages(pdf_path)
        if self._max_pages is not None:
            total_pages = min(total_pages, self._max_pages)

        logger.info(
            f"═══ {self.get_backend_name()}: {name} — {total_pages} pages, "
            f"batch={self._page_batch_size}, vlm_concurrency={self._vlm_concurrency} ═══"
        )

        converter = self._build_converter()

        pages_md: list[str] = []
        out_file = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_file = open(output_path, "w", encoding="utf-8")

        docling_seconds = 0.0
        assembly_seconds = 0.0
        peak_rss = _rss_mb()

        # The batch whose VLM futures are in flight. Emitting it is deferred until
        # after the NEXT convert() so the VLM decode overlaps docling's CPU work
        # instead of following it. Depth stays 1: per batch, convert is ~68s
        # against ~26s of VLM, so one batch of lag already absorbs all of it.
        pending: Optional[list] = None

        try:
            with ThreadPoolExecutor(max_workers=self._vlm_concurrency) as executor:
                for batch_start in range(1, total_pages + 1, self._page_batch_size):
                    batch_end = min(batch_start + self._page_batch_size - 1, total_pages)
                    batch_pages = batch_end - batch_start + 1

                    convert_started = time.monotonic()
                    doc = converter.convert(pdf_path, page_range=(batch_start, batch_end)).document
                    convert_elapsed = time.monotonic() - convert_started
                    docling_seconds += convert_elapsed

                    # Page numbers stay absolute inside a page_range slice, so
                    # `prov[0].page_no` needs no per-batch offset.
                    page_items: dict[int, list] = defaultdict(list)
                    for item, _level in doc.iterate_items():
                        if item.prov:
                            page_items[item.prov[0].page_no].append(item)

                    rss = _rss_mb()
                    peak_rss = max(peak_rss, rss)
                    logger.info(
                        f"Converted pages {batch_start}-{batch_end}: "
                        f"elapsed={convert_elapsed:.1f}s "
                        f"rate={convert_elapsed / max(batch_pages, 1):.2f}s/page "
                        f"rss={rss:.0f}MB"
                    )

                    # Build this batch: walk the items and submit VLM work, but do
                    # not join it. Everything that touches `doc` happens here.
                    assemble_started = time.monotonic()
                    built: list = []
                    for page_no in range(batch_start, batch_end + 1):
                        try:
                            ordered, vlm_tasks = self._build_page(
                                page_no=page_no,
                                items=page_items.get(page_no, []),
                                doc=doc,
                                executor=executor,
                            )
                            built.append((page_no, ordered, vlm_tasks))
                        except Exception as exc:
                            logger.error(f"[page {page_no}] build failed: {exc}", exc_info=True)
                            continue
                    assembly_seconds += time.monotonic() - assemble_started

                    # Release the batch before converting the next one. This is
                    # what keeps peak memory flat in total page count; dropping
                    # it is what made a 504-page PDF OOM the worker. It must also
                    # happen BEFORE the emit below, or two batches' docs overlap.
                    del doc, page_items
                    peak_rss = max(peak_rss, _rss_mb())

                    # Emit the previous batch. Its futures resolved during the
                    # convert() above, so these joins are nearly free.
                    if pending is not None:
                        emit_started = time.monotonic()
                        self._emit_batch(pending, pages_md, out_file, total_pages)
                        assembly_seconds += time.monotonic() - emit_started

                    pending = built

                # The final batch has no convert() left to hide under, so its VLM
                # tail is the one unavoidable serial cost. Stay inside the `with`:
                # outside it, the executor's shutdown(wait=True) would run first
                # and do the same waiting with the log order scrambled.
                if pending is not None:
                    emit_started = time.monotonic()
                    self._emit_batch(pending, pages_md, out_file, total_pages)
                    assembly_seconds += time.monotonic() - emit_started
                    pending = None
                peak_rss = max(peak_rss, _rss_mb())
        finally:
            if out_file:
                out_file.close()

        markdown = _fix_markdown_headings("".join(pages_md))

        total_elapsed = time.monotonic() - started
        with self._vlm_stats_lock:
            vlm_calls = self._vlm_calls
            vlm_seconds = self._vlm_seconds
            vlm_failures = self._vlm_failures
            vlm_blocked = self._vlm_blocked_seconds

        def _pct(value: float) -> float:
            return (value / total_elapsed * 100) if total_elapsed else 0.0

        logger.info(
            f"parse_pdf summary: {name} pages={total_pages} total={total_elapsed:.0f}s "
            f"docling={docling_seconds:.0f}s ({_pct(docling_seconds):.0f}%) "
            f"assembly={assembly_seconds:.0f}s ({_pct(assembly_seconds):.0f}%) "
            f"vlm_wait={vlm_seconds:.0f}s vlm_blocked={vlm_blocked:.0f}s "
            f"vlm_calls={vlm_calls} "
            f"vlm_failures={vlm_failures} peak_rss={peak_rss:.0f}MB"
        )
        if vlm_failures:
            model = getattr(self, "_vlm_model", None) or self._gemini_model
            logger.warning(
                f"{vlm_failures}/{vlm_calls} VLM calls failed for {name} (model={model}) — "
                f"those figures and tables fell back to [IMAGE]"
            )
        if output_path:
            logger.info(f"Saved → {output_path}")

        return markdown
