"""
Text Cleaning Module for RAG System
Handles tables, math notation, and special symbols using Strategy and Chain of Responsibility patterns.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logfire


class TextCleaningStrategy(ABC):
    """Abstract base class for text cleaning strategies."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the input text according to the strategy."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this cleaning strategy."""
        pass


class SurrogateRemovalStrategy(TextCleaningStrategy):
    """Remove Unicode surrogate characters that cause UTF-8 encoding errors."""

    def clean(self, text: str) -> str:
        """Remove surrogate pairs (U+D800 to U+DFFF)."""
        if not isinstance(text, str):
            text = str(text)

        # Remove surrogate characters
        text = re.sub(r'[\ud800-\udfff]', '', text)

        # Ensure valid UTF-8
        return text.encode('utf-8', errors='ignore').decode('utf-8')

    def get_name(self) -> str:
        return "surrogate_removal"


class MathNotationNormalizer(TextCleaningStrategy):
    """Normalize mathematical notation to ASCII equivalents or descriptions."""

    MATH_REPLACEMENTS = {
        # Greek letters
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma',
        'τ': 'tau', 'φ': 'phi', 'ω': 'omega',

        # Math operators
        '×': ' * ', '÷': ' / ', '±': ' +/- ',
        '≤': ' <= ', '≥': ' >= ', '≠': ' != ',
        '≈': ' ~= ', '∞': ' infinity ',

        # Superscripts (common)
        '²': '^2', '³': '^3', '⁴': '^4',

        # Arrows
        '→': ' -> ', '←': ' <- ', '↔': ' <-> ',

        # Math symbols
        '∑': ' sum ', '∏': ' product ', '∫': ' integral ',
        '√': ' sqrt ', '∂': ' partial ',
    }

    def clean(self, text: str) -> str:
        """Replace math notation with ASCII equivalents."""
        if not isinstance(text, str):
            text = str(text)

        # Apply replacements
        for math_char, replacement in self.MATH_REPLACEMENTS.items():
            text = text.replace(math_char, replacement)

        # Handle remaining math symbols by removing them or converting to description
        # Remove mathematical operators block (U+2200 to U+22FF)
        text = re.sub(r'[\u2200-\u22ff]', ' [math] ', text)

        # Remove mathematical alphanumeric symbols (U+1D400 to U+1D7FF)
        text = re.sub(r'[\U0001d400-\U0001d7ff]', '', text)

        return text

    def get_name(self) -> str:
        return "math_notation_normalizer"


class TableStructurePreserver(TextCleaningStrategy):
    """Preserve table structure using ASCII table formatting."""

    def clean(self, text: str) -> str:
        """Clean table separators and preserve structure."""
        if not isinstance(text, str):
            text = str(text)

        # Normalize table separators
        text = re.sub(r'[│┃║]', ' | ', text)  # Vertical lines
        text = re.sub(r'[─━═]', '-', text)     # Horizontal lines
        text = re.sub(r'[┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬]', '+', text)  # Corners and intersections

        # Preserve cells by ensuring spaces around pipe symbols
        text = re.sub(r'\s*\|\s*', ' | ', text)

        # Clean up excessive separators
        text = re.sub(r'-{4,}', '----', text)

        return text

    def get_name(self) -> str:
        return "table_structure_preserver"


class SpecialSymbolNormalizer(TextCleaningStrategy):
    """Normalize special symbols and punctuation."""

    SYMBOL_REPLACEMENTS = {
        # Quotes
        '"': '"', '"': '"', ''': "'", ''': "'",
        '«': '"', '»': '"', '‹': "'", '›': "'",

        # Dashes
        '–': '-', '—': '-', '−': '-',

        # Ellipsis
        '…': '...',

        # Bullets
        '•': '*', '◦': '-', '▪': '*', '▫': '-',

        # Currency (keep common ones, remove obscure)
        '€': 'EUR', '£': 'GBP', '¥': 'YEN',

        # Fractions
        '½': '1/2', '⅓': '1/3', '¼': '1/4', '¾': '3/4',
    }

    def clean(self, text: str) -> str:
        """Normalize special symbols to ASCII equivalents."""
        if not isinstance(text, str):
            text = str(text)

        # Apply symbol replacements
        for symbol, replacement in self.SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

        return text

    def get_name(self) -> str:
        return "special_symbol_normalizer"


class WhitespaceNormalizer(TextCleaningStrategy):
    """Normalize whitespace and control characters."""

    def clean(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        if not isinstance(text, str):
            text = str(text)

        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char == '\n' or char == '\t' or not unicodedata.category(char).startswith('C'))

        # Normalize multiple spaces to single space
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize multiple newlines to maximum 2 (preserve paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove spaces at start/end of lines
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n[ \t]+', '\n', text)

        return text.strip()

    def get_name(self) -> str:
        return "whitespace_normalizer"


class UnicodeNormalizer(TextCleaningStrategy):
    """Normalize Unicode to standard forms."""

    def __init__(self, form: str = 'NFKC'):
        """
        Initialize with Unicode normalization form.

        Args:
            form: Unicode normalization form (NFC, NFKC, NFD, NFKD)
                  NFKC is recommended for text processing (compatibility composition)
        """
        self.form = form

    def clean(self, text: str) -> str:
        """Normalize Unicode to specified form."""
        if not isinstance(text, str):
            text = str(text)

        return unicodedata.normalize(self.form, text)

    def get_name(self) -> str:
        return f"unicode_normalizer_{self.form}"


class TextCleaningPipeline:
    """
    Chain of Responsibility pattern for text cleaning.
    Applies multiple cleaning strategies in sequence.
    """

    def __init__(self, strategies: Optional[List[TextCleaningStrategy]] = None):
        """
        Initialize the cleaning pipeline.

        Args:
            strategies: List of cleaning strategies to apply in order.
                       If None, uses default pipeline.
        """
        if strategies is None:
            # Default pipeline for RAG system
            self.strategies = [
                SurrogateRemovalStrategy(),
                UnicodeNormalizer('NFKC'),
                MathNotationNormalizer(),
                TableStructurePreserver(),
                SpecialSymbolNormalizer(),
                WhitespaceNormalizer(),
            ]
        else:
            self.strategies = strategies

    def clean(self, text: str, log_steps: bool = False) -> str:
        """
        Apply all cleaning strategies in sequence.

        Args:
            text: Input text to clean
            log_steps: Whether to log each cleaning step

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)

        if log_steps:
            logfire.info("Starting text cleaning pipeline",
                        initial_length=len(text),
                        num_strategies=len(self.strategies))

        cleaned_text = text
        for strategy in self.strategies:
            try:
                before_length = len(cleaned_text)
                cleaned_text = strategy.clean(cleaned_text)
                after_length = len(cleaned_text)

                if log_steps and before_length != after_length:
                    logfire.info(f"Applied {strategy.get_name()}",
                               length_change=after_length - before_length,
                               before=before_length,
                               after=after_length)
            except Exception as e:
                logfire.error(f"Strategy {strategy.get_name()} failed",
                            error=str(e),
                            strategy=strategy.get_name())
                # Continue with next strategy even if one fails
                continue

        if log_steps:
            logfire.info("Text cleaning pipeline completed",
                        original_length=len(text),
                        final_length=len(cleaned_text),
                        reduction=len(text) - len(cleaned_text))

        return cleaned_text

    def add_strategy(self, strategy: TextCleaningStrategy, position: Optional[int] = None):
        """Add a strategy to the pipeline."""
        if position is None:
            self.strategies.append(strategy)
        else:
            self.strategies.insert(position, strategy)

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name."""
        for i, strategy in enumerate(self.strategies):
            if strategy.get_name() == strategy_name:
                self.strategies.pop(i)
                return True
        return False
