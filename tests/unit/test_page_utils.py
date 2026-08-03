"""
Unit tests for ingestion/processors/page_utils.py.

Covers:
- get_page_number_for_position with various mappings
- Boundary positions
- Empty mapping
- Position before/after all mapped content
"""
import pytest

from app.ingestion.processors.page_utils import get_page_number_for_position


class TestGetPageNumberForPosition:
    def test_position_in_first_page(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        assert get_page_number_for_position(50, mapping) == 1

    def test_position_in_second_page(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        assert get_page_number_for_position(150, mapping) == 2

    def test_position_at_start_boundary(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        assert get_page_number_for_position(0, mapping) == 1

    def test_position_at_end_boundary(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        assert get_page_number_for_position(100, mapping) == 1

    def test_position_at_page_start_boundary(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        assert get_page_number_for_position(101, mapping) == 2

    def test_position_before_all_pages(self):
        mapping = [(50, 100, 1), (101, 200, 2)]
        result = get_page_number_for_position(10, mapping)
        assert result == 1

    def test_position_after_all_pages(self):
        mapping = [(0, 100, 1), (101, 200, 2)]
        result = get_page_number_for_position(500, mapping)
        assert result == 2

    def test_empty_mapping_returns_default(self):
        result = get_page_number_for_position(50, [])
        assert result == 1

    def test_single_page_mapping(self):
        mapping = [(0, 1000, 1)]
        assert get_page_number_for_position(500, mapping) == 1

    def test_multiple_pages(self):
        mapping = [(0, 50, 1), (51, 100, 2), (101, 150, 3), (151, 200, 4)]
        assert get_page_number_for_position(75, mapping) == 2
        assert get_page_number_for_position(125, mapping) == 3
        assert get_page_number_for_position(175, mapping) == 4

    def test_non_contiguous_pages(self):
        mapping = [(0, 50, 1), (100, 150, 3)]
        assert get_page_number_for_position(25, mapping) == 1
        assert get_page_number_for_position(125, mapping) == 3
