"""
Unit tests for which extractor handles tables in `_process_page`.

Measured on real crops from `input/raw/…_bert.pdf` with `qwen3.5:0.8b`:

- Table 1 (GLUE results, 13 columns) came back with headers `I, II, III, IV…`
  and mangled values.
- Table 4 (small) came back with invented content — `Births:`, `9.86%`, and a
  literal `A B C D E F G H I J K…` row that appears nowhere in the paper.
- With thinking on, every table call returned an empty response after ~3600
  reasoning tokens.

The old rule sent a table to the VLM only when `_is_complex_table()` was true
(rows > 8 *and* cols > 6), so it routed precisely the hardest tables to the
weakest extractor. Docling's TableFormer — already used for every simple table
via `item.export_to_markdown(doc)` — handles all of them now, and `vlm_tables`
restores the old path for anyone who wants it.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.processors.gemini_docling_parser import GeminiDoclingParser


class FakeBBox:
    l, r, t, b = 0.0, 100.0, 100.0, 50.0


class FakeProv:
    page_no = 1
    bbox = FakeBBox()


class FakeTableData:
    def __init__(self, rows, cols):
        self.num_rows = rows
        self.num_cols = cols


def _fake_table(rows, cols, markdown="| a | b |\n|---|---|\n| 1 | 2 |"):
    """A stand-in for docling's TableItem, patched in as the real class below."""
    table = MagicMock()
    table.prov = [FakeProv()]
    table.data = FakeTableData(rows, cols)
    table.export_to_markdown.return_value = markdown
    table.get_image.return_value = MagicMock(width=800, height=600)
    return table


@pytest.fixture
def doc():
    document = MagicMock()
    document.pages.get.return_value = None  # no page size → no column-split logic
    return document


def _run_process_page(parser, items, doc):
    """Call _process_page with docling's item classes swapped for our fakes.

    `_process_page` imports the real classes at call time and dispatches on
    isinstance, so the table fakes must *be* TableItem for the branch under test
    to be reached.
    """
    table_cls = type(items[0])
    stub = MagicMock()
    stub.TableItem = table_cls
    stub.PictureItem = type("PictureItem", (), {})
    stub.TextItem = type("TextItem", (), {})
    stub.SectionHeaderItem = type("SectionHeaderItem", (), {})

    with patch.dict("sys.modules", {"docling_core.types.doc": stub}):
        return parser._process_page(page_no=1, items=items, doc=doc, executor=None)


class TestTablesGoToDocling:
    @pytest.mark.parametrize("rows,cols", [(20, 13), (9, 7), (3, 2)])
    def test_no_vlm_call_for_any_table_by_default(self, doc, rows, cols):
        parser = GeminiDoclingParser(api_key="k")
        table = _fake_table(rows, cols)

        with patch.object(parser, "_call_vlm") as call_vlm:
            markdown = _run_process_page(parser, [table], doc)

        call_vlm.assert_not_called()
        table.export_to_markdown.assert_called_once()
        assert "| a | b |" in markdown

    def test_complex_table_reaches_the_vlm_when_opted_in(self, doc):
        parser = GeminiDoclingParser(api_key="k", vlm_tables=True)
        table = _fake_table(20, 13)

        with patch.object(parser, "_call_vlm", return_value="<table>vlm</table>") as call_vlm:
            markdown = _run_process_page(parser, [table], doc)

        call_vlm.assert_called_once()
        assert "vlm" in markdown

    def test_simple_table_still_uses_docling_when_opted_in(self, doc):
        """`vlm_tables=True` restores the old rule, which is complexity-gated —
        it does not send every table to the VLM."""
        parser = GeminiDoclingParser(api_key="k", vlm_tables=True)
        table = _fake_table(3, 2)

        with patch.object(parser, "_call_vlm") as call_vlm:
            _run_process_page(parser, [table], doc)

        call_vlm.assert_not_called()
        table.export_to_markdown.assert_called_once()


class TestDefaults:
    def test_vlm_tables_defaults_to_false_on_both_backends(self):
        from app.ingestion.processors.ollama_pdf_parser import OllamaPDFParser

        assert GeminiDoclingParser(api_key="k")._vlm_tables is False
        assert OllamaPDFParser()._vlm_tables is False
