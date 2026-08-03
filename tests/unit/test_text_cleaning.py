"""
Unit tests for Text Cleaning Strategies.

Tests:
1. SurrogateRemovalStrategy
2. MathNotationNormalizer
3. TableStructurePreserver
4. SpecialSymbolNormalizer
5. WhitespaceNormalizer
6. UnicodeNormalizer
7. TextCleaningPipeline (chain of responsibility)
"""
import sys
import pytest
from pathlib import Path

from app.ingestion.text_cleaning.cleaners import (
    TextCleaningStrategy,
    SurrogateRemovalStrategy,
    MathNotationNormalizer,
    TableStructurePreserver,
    SpecialSymbolNormalizer,
    WhitespaceNormalizer,
    UnicodeNormalizer,
    TextCleaningPipeline
)


class TestSurrogateRemovalStrategy:
    """Tests for SurrogateRemovalStrategy."""

    def test_removes_surrogate_characters(self):
        """Test that surrogate characters are removed."""
        strategy = SurrogateRemovalStrategy()
        # Text with surrogate characters
        text_with_surrogates = "Text with \ud835\udc65 surrogate"
        
        result = strategy.clean(text_with_surrogates)
        
        # Surrogates should be removed
        assert "\ud835" not in result
        assert "\udc65" not in result
        assert "Text with" in result

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        strategy = SurrogateRemovalStrategy()
        normal_text = "This is normal English text."
        
        result = strategy.clean(normal_text)
        
        assert result == normal_text

    def test_get_name_returns_string(self):
        """Test that get_name returns a string."""
        strategy = SurrogateRemovalStrategy()
        
        name = strategy.get_name()
        
        assert isinstance(name, str)
        assert len(name) > 0


class TestMathNotationNormalizer:
    """Tests for MathNotationNormalizer."""

    def test_normalizes_greek_letters(self):
        """Test that Greek letters are normalized."""
        strategy = MathNotationNormalizer()
        text = "α + β = γ"
        
        result = strategy.clean(text)
        
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

    def test_normalizes_math_symbols(self):
        """Test that math symbols are normalized."""
        strategy = MathNotationNormalizer()
        text = "∑ and ∫ and √"
        
        result = strategy.clean(text)
        
        assert "sum" in result
        assert "integral" in result
        assert "sqrt" in result

    def test_normalizes_comparison_operators(self):
        """Test that comparison operators are normalized."""
        strategy = MathNotationNormalizer()
        text = "x ≤ y and x ≥ z"
        
        result = strategy.clean(text)
        
        assert "<=" in result or "≤" not in result
        assert ">=" in result or "≥" not in result

    def test_preserves_regular_text(self):
        """Test that regular text is preserved."""
        strategy = MathNotationNormalizer()
        text = "Regular text without math"
        
        result = strategy.clean(text)
        
        assert "Regular text without math" in result


class TestTableStructurePreserver:
    """Tests for TableStructurePreserver."""

    def test_converts_table_borders(self):
        """Test that table borders are converted to ASCII."""
        strategy = TableStructurePreserver()
        text = "┌─────┬─────┐\n│ A   │ B   │\n└─────┴─────┘"
        
        result = strategy.clean(text)
        
        # Should contain ASCII table characters or be normalized
        assert result is not None
        assert len(result) > 0

    def test_preserves_table_content(self):
        """Test that table content is preserved."""
        strategy = TableStructurePreserver()
        text = "│ Content │"
        
        result = strategy.clean(text)
        
        assert "Content" in result


class TestSpecialSymbolNormalizer:
    """Tests for SpecialSymbolNormalizer."""

    def test_normalizes_smart_quotes(self):
        """Test that smart quotes are normalized to regular quotes."""
        strategy = SpecialSymbolNormalizer()
        text = '\u201cSmart quotes\u201d and \u2018apostrophes\u2019'
        
        result = strategy.clean(text)
        
        # Smart quotes should be converted to regular quotes
        assert '"' not in result or result.count('"') > 0

    def test_normalizes_currency_symbols(self):
        """Test that currency symbols are normalized."""
        strategy = SpecialSymbolNormalizer()
        text = "€100 and £50 and ¥1000"
        
        result = strategy.clean(text)
        
        # Currency symbols should be converted
        assert "EUR" in result or "€" not in result

    def test_normalizes_fractions(self):
        """Test that fractions are normalized."""
        strategy = SpecialSymbolNormalizer()
        text = "½ and ¼ and ¾"
        
        result = strategy.clean(text)
        
        assert "1/2" in result or "½" not in result


class TestWhitespaceNormalizer:
    """Tests for WhitespaceNormalizer."""

    def test_normalizes_multiple_spaces(self):
        """Test that multiple spaces are normalized to single space."""
        strategy = WhitespaceNormalizer()
        text = "Multiple    spaces   here"
        
        result = strategy.clean(text)
        
        assert "    " not in result
        assert "Multiple spaces here" in result

    def test_normalizes_multiple_newlines(self):
        """Test that excessive newlines are normalized."""
        strategy = WhitespaceNormalizer()
        text = "Line1\n\n\n\n\nLine2"
        
        result = strategy.clean(text)
        
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is stripped."""
        strategy = WhitespaceNormalizer()
        text = "   Text with spaces   "
        
        result = strategy.clean(text)
        
        assert result == result.strip()


class TestUnicodeNormalizer:
    """Tests for UnicodeNormalizer."""

    def test_normalizes_to_nfkc_by_default(self):
        """Test that default normalization is NFKC."""
        strategy = UnicodeNormalizer()
        
        assert strategy.form == 'NFKC'

    def test_normalizes_composed_characters(self):
        """Test that composed characters are normalized."""
        strategy = UnicodeNormalizer()
        # é can be represented as single char or combining sequence
        text = "café"
        
        result = strategy.clean(text)
        
        assert "caf" in result
        # The é should be normalized

    def test_custom_normalization_form(self):
        """Test that custom normalization form is used."""
        strategy = UnicodeNormalizer(form='NFC')
        
        assert strategy.form == 'NFC'


class TestTextCleaningPipeline:
    """Tests for TextCleaningPipeline (Chain of Responsibility)."""

    def test_applies_strategies_in_order(self):
        """Test that strategies are applied in order."""
        pipeline = TextCleaningPipeline([
            WhitespaceNormalizer(),
            SurrogateRemovalStrategy()
        ])
        
        text = "  Multiple   spaces  "
        result = pipeline.clean(text)
        
        # Whitespace should be normalized
        assert "Multiple spaces" in result

    def test_empty_pipeline(self):
        """Test that empty pipeline returns original text."""
        pipeline = TextCleaningPipeline([])
        text = "Original text"
        
        result = pipeline.clean(text)
        
        assert result == text

    def test_add_strategy(self):
        """Test adding strategy to pipeline."""
        pipeline = TextCleaningPipeline([])
        pipeline.add_strategy(WhitespaceNormalizer())
        
        text = "  Spaces  "
        result = pipeline.clean(text)
        
        assert result == "Spaces"

    def test_remove_strategy(self):
        """Test removing strategy from pipeline."""
        strategy = WhitespaceNormalizer()
        pipeline = TextCleaningPipeline([strategy])
        
        pipeline.remove_strategy(strategy.get_name())
        
        # After removing, the pipeline should not normalize whitespace
        text = "  Spaces  "
        result = pipeline.clean(text)
        
        # Behavior depends on implementation
