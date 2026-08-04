"""
Unit tests for the batched, streaming `parse_pdf` in GeminiDoclingParser.

The behaviour under test is memory, not output: an earlier version converted
every page batch up front and kept all the resulting `DoclingDocument`s (and the
rendered page images they carry) alive until the end of the run, so peak memory
scaled with *total* pages and a 504-page PDF OOM-killed the Celery worker.
Batching bounded docling's working memory but not its retained memory.

`test_batch_document_is_released_before_next_convert` is the one that catches a
regression: it holds only weak references, so if `parse_pdf` starts retaining
batch documents again the weakref stays alive and the test fails.
"""
import gc
import weakref
from unittest.mock import patch

import pytest

from app.ingestion.processors.gemini_docling_parser import GeminiDoclingParser


class FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no


class FakeItem:
    """Stands in for a docling TextItem: only `.prov` is read before assembly."""
    def __init__(self, page_no):
        self.prov = [FakeProv(page_no)]


class FakeDocument:
    """Weakref-able stand-in for DoclingDocument."""
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self._items = [FakeItem(p) for p in range(start, end + 1)]

    def iterate_items(self):
        for item in self._items:
            yield item, 0


class FakeConversionResult:
    def __init__(self, document):
        self.document = document


class RecordingConverter:
    """Records each convert() call and whether the previous document is still alive.

    Deliberately keeps only a weakref to each document it hands out, so the only
    thing that can keep one alive is `parse_pdf` itself.
    """
    def __init__(self):
        self.page_ranges = []
        self.doc_refs = []
        self.alive_at_convert = []

    def convert(self, pdf_path, page_range=None):
        gc.collect()
        self.alive_at_convert.append([ref() is not None for ref in self.doc_refs])

        self.page_ranges.append(page_range)
        doc = FakeDocument(*page_range)
        self.doc_refs.append(weakref.ref(doc))
        return FakeConversionResult(doc)


@pytest.fixture
def parser():
    return GeminiDoclingParser(
        api_key="test-key",
        page_batch_size=10,
        vlm_concurrency=1,
    )


@pytest.fixture
def converter():
    return RecordingConverter()


def _fake_process_page(page_no, items, doc, executor=None):
    """Plain function, not a Mock: a Mock's call_args_list would itself retain
    every `doc` it was called with, which would defeat the memory-release test."""
    return f"[PAGE:{page_no}]"


def _run(parser, converter, total_pages, output_path=None):
    """Drive parse_pdf against the fake converter, stubbing out page assembly."""
    with patch.object(parser, "_count_pages", return_value=total_pages), \
         patch.object(parser, "_build_converter", return_value=converter), \
         patch.object(parser, "_process_page", new=_fake_process_page):
        return parser.parse_pdf("/nonexistent/fake.pdf", output_path=output_path)


class TestBatching:
    def test_convert_is_called_once_per_batch_with_absolute_page_ranges(self, parser, converter):
        _run(parser, converter, total_pages=25)

        assert converter.page_ranges == [(1, 10), (11, 20), (21, 25)]

    def test_single_batch_document_shorter_than_batch_size(self, parser, converter):
        _run(parser, converter, total_pages=3)

        assert converter.page_ranges == [(1, 3)]

    def test_max_pages_caps_the_conversion(self, converter):
        parser = GeminiDoclingParser(api_key="k", page_batch_size=10, max_pages=12)
        _run(parser, converter, total_pages=500)

        assert converter.page_ranges == [(1, 10), (11, 12)]


class TestMemoryRelease:
    def test_batch_document_is_released_before_next_convert(self, parser, converter):
        """No batch document may survive into the next convert() call.

        This is the regression guard for the OOM: with the old `batch_docs`
        dict every document was still alive here.
        """
        _run(parser, converter, total_pages=50)

        assert len(converter.page_ranges) == 5
        # First convert has nothing before it; every later one must see only dead refs.
        for call_index, alive in enumerate(converter.alive_at_convert):
            assert not any(alive), (
                f"convert() call {call_index} still holds "
                f"{sum(alive)} earlier batch document(s) in memory"
            )

    def test_no_document_survives_the_parse(self, parser, converter):
        _run(parser, converter, total_pages=30)
        gc.collect()

        assert [ref() for ref in converter.doc_refs] == [None, None, None]


class TestPageOrdering:
    def test_pages_are_assembled_in_document_order_across_batches(self, parser, converter):
        markdown = _run(parser, converter, total_pages=25)

        positions = [markdown.index(f"[PAGE:{n}]") for n in range(1, 26)]
        assert positions == sorted(positions)

    def test_page_numbers_stay_absolute_past_the_first_batch(self, parser, converter):
        """The bug this guards: pages renumbered per batch would emit 1-10 four times."""
        markdown = _run(parser, converter, total_pages=25)

        for page_no in range(1, 26):
            assert f"[PAGE:{page_no}]" in markdown
        assert markdown.count("[PAGE:1]") == 1

    def test_streams_to_output_file_as_it_goes(self, parser, converter, tmp_path):
        out = tmp_path / "parsed.md"
        _run(parser, converter, total_pages=25, output_path=str(out))

        written = out.read_text(encoding="utf-8")
        assert "[PAGE:1]" in written
        assert "[PAGE:25]" in written


class TestVlmStats:
    def test_counters_reset_between_documents(self, parser, converter):
        parser._record_vlm_call(1.5)
        parser._record_vlm_call(2.5, failed=True)
        assert parser._vlm_calls == 2

        _run(parser, converter, total_pages=5)

        assert parser._vlm_calls == 0
        assert parser._vlm_seconds == 0.0
        assert parser._vlm_failures == 0

    def test_record_vlm_call_accumulates(self, parser):
        assert parser._record_vlm_call(1.0) == 1
        assert parser._record_vlm_call(2.0, failed=True) == 2

        assert parser._vlm_seconds == pytest.approx(3.0)
        assert parser._vlm_failures == 1
