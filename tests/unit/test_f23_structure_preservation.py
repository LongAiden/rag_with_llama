"""Unit tests for F23 structure-preservation helpers in gemini_docling_parser.

Tests the pure functions added to preserve docling's structural classification:
- _format_code: fencing and doctest line-break restoration
- _format_list_item: marker-aware list item emission
- _wrap_figure: deterministic <figure> wrapping
- Fence-aware post-passes: _fix_markdown_headings, _normalize_tables_in_markdown,
  _strip_stray_headers
"""
import pytest

from app.ingestion.processors.gemini_docling_parser import (
    _fix_markdown_headings,
    _format_code,
    _format_list_item,
    _normalize_tables_in_markdown,
    _strip_stray_headers,
    _wrap_figure,
)


class FakeLabel:
    def __init__(self, value):
        self.value = value


class FakeCodeItem:
    def __init__(self, text, language="unknown"):
        self.text = text
        self.code_language = FakeLabel(language)


class FakeListItem:
    def __init__(self, text, marker=None, enumerated=False):
        self.text = text
        self.marker = marker
        self.enumerated = enumerated


class TestFormatCode:
    def test_empty_text_returns_empty(self):
        assert _format_code(FakeCodeItem("")) == ""
        assert _format_code(FakeCodeItem(None)) == ""

    def test_plain_code_gets_fenced(self):
        result = _format_code(FakeCodeItem("x = 1"))
        assert result == "```\nx = 1\n```"

    def test_language_is_included(self):
        result = _format_code(FakeCodeItem("x = 1", "python"))
        assert result == "```python\nx = 1\n```"

    def test_unknown_language_is_omitted(self):
        result = _format_code(FakeCodeItem("x = 1", "unknown"))
        assert result == "```\nx = 1\n```"

    def test_doctest_lines_are_split(self):
        text = ">>> def foo(): ...     return 1 ... >>>"
        result = _format_code(FakeCodeItem(text))
        assert ">>> def foo():" in result
        assert "...     return 1" in result
        lines = result.split("\n")
        assert lines[1] == ">>> def foo():"
        assert lines[2] == "...     return 1"

    def test_trailing_bare_prompt_is_stripped(self):
        text = ">>> x = 1 >>>"
        result = _format_code(FakeCodeItem(text))
        assert result == "```\n>>> x = 1\n```"

    def test_indentation_is_preserved(self):
        text = ">>> def f(): ...     pass"
        result = _format_code(FakeCodeItem(text))
        assert "...     pass" in result


class TestFormatListItem:
    def test_empty_text_returns_empty(self):
        assert _format_list_item(FakeListItem("")) == ""
        assert _format_list_item(FakeListItem(None)) == ""

    def test_marker_is_used_when_present(self):
        result = _format_list_item(FakeListItem("item text", marker="*"))
        assert result == "* item text"

    def test_enumerated_uses_number_when_no_marker(self):
        result = _format_list_item(FakeListItem("item text", enumerated=True))
        assert result == "1. item text"

    def test_plain_list_uses_dash_when_no_marker(self):
        result = _format_list_item(FakeListItem("item text"))
        assert result == "- item text"


class TestWrapFigure:
    def test_wraps_plain_text(self):
        result = _wrap_figure("a description", "")
        assert result == "<figure>\na description\n</figure>"

    def test_includes_caption(self):
        result = _wrap_figure("desc", "Figure 1. Caption")
        assert "<figure_caption>Figure 1. Caption</figure_caption>" in result
        assert result.startswith("<figure>")
        assert result.endswith("</figure>")

    def test_strips_existing_figure_tags(self):
        result = _wrap_figure("<figure>already wrapped</figure>", "")
        assert result.count("<figure>") == 1
        assert result.count("</figure>") == 1

    def test_strips_partial_figure_tags(self):
        result = _wrap_figure("<figure>desc", "cap")
        assert result.count("<figure>") == 1
        assert result.count("</figure>") == 1


class TestFenceAwareFixMarkdownHeadings:
    def test_headings_outside_fence_are_promoted(self):
        result = _fix_markdown_headings("1. Real Heading\ntext")
        assert result == "## 1. Real Heading\ntext"

    def test_headings_inside_fence_are_unchanged(self):
        s = "```python\n1. not a heading\nINTRODUCTION\n```"
        assert _fix_markdown_headings(s) == s

    def test_mixed_fence_and_non_fence(self):
        s = "1. Real Heading\n```python\n1. not a heading\n```\n2. Another Heading"
        result = _fix_markdown_headings(s)
        assert result == "## 1. Real Heading\n```python\n1. not a heading\n```\n## 2. Another Heading"

    def test_allcaps_inside_fence_unchanged(self):
        s = "```\nINTRODUCTION\n```"
        assert _fix_markdown_headings(s) == s

    def test_allcaps_outside_fence_promoted(self):
        result = _fix_markdown_headings("INTRODUCTION")
        assert result == "## INTRODUCTION"


class TestFenceAwareNormalizeTables:
    def test_tables_outside_fence_are_normalized(self):
        s = "| a | b |\n|---|---|\n| 1 | 2 |"
        result = _normalize_tables_in_markdown(s)
        assert "| a | b |" in result

    def test_pipe_lines_inside_fence_are_unchanged(self):
        s = "```python\n| not a table |\n```"
        assert _normalize_tables_in_markdown(s) == s

    def test_mixed_content(self):
        s = "| real table |\n```\n| not table |\n```\n| another real |"
        result = _normalize_tables_in_markdown(s)
        assert "| real table |" in result
        assert "| not table |" in result
        assert "| another real |" in result


class TestFenceAwareStripStrayHeaders:
    def test_stray_header_outside_block_is_removed(self):
        s = "# stray header\ntext"
        result = _strip_stray_headers(s)
        assert "# stray header" not in result
        assert "text" in result

    def test_header_inside_fence_is_preserved(self):
        s = "```python\n# this is a comment\n```"
        result = _strip_stray_headers(s)
        assert "# this is a comment" in result

    def test_header_inside_figure_is_preserved(self):
        s = "<figure>\n# caption header\n</figure>"
        result = _strip_stray_headers(s)
        assert "# caption header" in result

    def test_mixed_fence_and_stray(self):
        s = "```python\n# comment\n```\n# stray"
        result = _strip_stray_headers(s)
        assert "# comment" in result
        assert "# stray" not in result
