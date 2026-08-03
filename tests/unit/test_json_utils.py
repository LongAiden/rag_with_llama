"""
Unit tests for graph_processing/json_utils.py.

Covers:
- JSONParser.extract_and_parse (array + object)
- Markdown code block extraction
- Balanced bracket extraction
- Regex extraction
- JSON repair (trailing commas, duplicate commas)
- parse_with_fallback defaults
- safe_json_loads
"""
import json
import pytest

from app.graph.json_utils import JSONParser, safe_json_loads


class TestExtractAndParseArray:
    def test_valid_json_array(self):
        text = '[{"name": "BERT"}, {"name": "GPT"}]'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result == [{"name": "BERT"}, {"name": "GPT"}]

    def test_json_array_in_markdown_block(self):
        text = '```json\n[{"id": 1}]\n```'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result == [{"id": 1}]

    def test_json_array_with_surrounding_text(self):
        text = 'Here are the entities:\n[{"name": "Adam"}]\nDone.'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result == [{"name": "Adam"}]

    def test_empty_array(self):
        text = '[]'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result == []

    def test_nested_array(self):
        text = '[{"a": [1, 2, 3]}]'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result == [{"a": [1, 2, 3]}]


class TestExtractAndParseObject:
    def test_valid_json_object(self):
        text = '{"key": "value"}'
        result = JSONParser.extract_and_parse(text, expected_type="object")
        assert result == {"key": "value"}

    def test_json_object_in_markdown_block(self):
        text = '```json\n{"status": "ok"}\n```'
        result = JSONParser.extract_and_parse(text, expected_type="object")
        assert result == {"status": "ok"}

    def test_empty_object(self):
        text = '{}'
        result = JSONParser.extract_and_parse(text, expected_type="object")
        assert result == {}


class TestJSONRepair:
    def test_trailing_comma_in_array(self):
        text = '[{"a": 1}, {"b": 2},]'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result is not None
        assert len(result) == 2

    def test_trailing_comma_in_object(self):
        text = '{"a": 1, "b": 2,}'
        result = JSONParser.extract_and_parse(text, expected_type="object")
        assert result is not None
        assert result["a"] == 1

    def test_duplicate_commas(self):
        text = '[{"a": 1},,{"b": 2}]'
        result = JSONParser.extract_and_parse(text, expected_type="array")
        assert result is not None


class TestParseWithFallback:
    def test_returns_parsed_on_success(self):
        text = '[1, 2, 3]'
        result = JSONParser.parse_with_fallback(text, expected_type="array")
        assert result == [1, 2, 3]

    def test_returns_empty_list_on_failure_array(self):
        text = 'this is not json at all'
        result = JSONParser.parse_with_fallback(text, expected_type="array")
        assert result == []

    def test_returns_empty_dict_on_failure_object(self):
        text = 'this is not json at all'
        result = JSONParser.parse_with_fallback(text, expected_type="object")
        assert result == {}


class TestSafeJsonLoads:
    def test_valid_json(self):
        result = safe_json_loads('{"a": 1}')
        assert result == {"a": 1}

    def test_invalid_json_returns_default(self):
        result = safe_json_loads('not json', default={"fallback": True})
        assert result == {"fallback": True}

    def test_invalid_json_default_none(self):
        result = safe_json_loads('not json')
        assert result is None

    def test_valid_list(self):
        result = safe_json_loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_empty_string(self):
        result = safe_json_loads('', default="empty")
        assert result == "empty"


class TestBalancedBracketExtraction:
    def test_array_with_nested_objects(self):
        text = 'prefix [{"a": {"b": 1}}] suffix'
        result = JSONParser._try_balanced_extraction(text, "array")
        assert result == [{"a": {"b": 1}}]

    def test_object_with_nested_arrays(self):
        text = 'prefix {"items": [1, 2]} suffix'
        result = JSONParser._try_balanced_extraction(text, "object")
        assert result == {"items": [1, 2]}

    def test_no_brackets_returns_none(self):
        result = JSONParser._try_balanced_extraction("no brackets here", "array")
        assert result is None

    def test_string_with_brackets_inside(self):
        text = '[{"text": "value with ] bracket"}]'
        result = JSONParser._try_balanced_extraction(text, "array")
        assert result is not None
        assert result[0]["text"] == "value with ] bracket"
