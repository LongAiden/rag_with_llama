"""
Unit tests for validate_table_name() in api/validators.py.

Covers the SQL injection prevention regex:
  ^[a-zA-Z_][a-zA-Z0-9_]{0,62}$
"""
import pytest
from fastapi import HTTPException

from app.api.validators import validate_table_name


class TestValidTableNames:
    """Names that must pass validation."""

    def test_simple_lowercase(self):
        validate_table_name("chunks")

    def test_default_table_name(self):
        validate_table_name("document_chunks")

    def test_starts_with_underscore(self):
        validate_table_name("_my_table")

    def test_alphanumeric_mix(self):
        validate_table_name("table_v2_final")

    def test_uppercase(self):
        validate_table_name("DocumentChunks")

    def test_single_char(self):
        validate_table_name("t")

    def test_exactly_63_chars(self):
        # 1 letter + 62 underscores = 63 chars total (max allowed)
        validate_table_name("a" + "_" * 62)


class TestSQLInjectionPayloads:
    """Payloads that must be blocked with HTTP 400."""

    def test_semicolon_drop(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("foo; DROP TABLE document_chunks; --")
        assert exc.value.status_code == 400

    def test_inline_comment(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("foo--bar")
        assert exc.value.status_code == 400

    def test_space_in_name(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("foo bar")
        assert exc.value.status_code == 400

    def test_single_quote(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("foo'bar")
        assert exc.value.status_code == 400

    def test_double_quote(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name('foo"bar')
        assert exc.value.status_code == 400

    def test_parenthesis(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("foo()")
        assert exc.value.status_code == 400


class TestInvalidTableNames:
    """Other invalid names that must be blocked."""

    def test_empty_string(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("")
        assert exc.value.status_code == 400

    def test_starts_with_digit(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("1table")
        assert exc.value.status_code == 400

    def test_hyphen(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("my-table")
        assert exc.value.status_code == 400

    def test_dot(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("schema.table")
        assert exc.value.status_code == 400

    def test_too_long(self):
        # 64 chars — one over the PostgreSQL limit
        with pytest.raises(HTTPException) as exc:
            validate_table_name("a" * 64)
        assert exc.value.status_code == 400

    def test_slash(self):
        with pytest.raises(HTTPException) as exc:
            validate_table_name("../../etc/passwd")
        assert exc.value.status_code == 400
