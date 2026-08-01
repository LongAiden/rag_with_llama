"""
Unit tests for repositories/table_repository.py (pure functions only).

Covers:
- validate_table_name (valid/invalid names)
- quote_ident
"""
import pytest

from repositories.table_repository import validate_table_name, quote_ident


class TestValidateTableName:
    @pytest.mark.parametrize("name", [
        "document_chunks",
        "test_table",
        "_private",
        "Table123",
        "a",
        "a" + "_" * 62,
    ])
    def test_valid_names(self, name):
        assert validate_table_name(name) == name

    @pytest.mark.parametrize("name", [
        "",
        "1starts_with_digit",
        "has space",
        "has-hyphen",
        "has.dot",
        "has;semicolon",
        "a" * 64,
        "drop table",
    ])
    def test_invalid_names(self, name):
        with pytest.raises(ValueError, match="Invalid table name"):
            validate_table_name(name)


class TestQuoteIdent:
    def test_quotes_valid_name(self):
        assert quote_ident("document_chunks") == '"document_chunks"'

    def test_quotes_simple_name(self):
        assert quote_ident("test") == '"test"'

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError):
            quote_ident("invalid name")
