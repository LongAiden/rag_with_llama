# Text Cleaning Module

Robust text cleaning system for RAG pipelines using design patterns to handle tables, mathematical notation, and special symbols.

## Design Patterns Used

### 1. **Strategy Pattern**
Each cleaning operation is encapsulated in a strategy class:
- `SurrogateRemovalStrategy` - Removes Unicode surrogates
- `MathNotationNormalizer` - Converts math symbols to ASCII
- `TableStructurePreserver` - Preserves table formatting
- `SpecialSymbolNormalizer` - Normalizes special characters
- `WhitespaceNormalizer` - Cleans whitespace
- `UnicodeNormalizer` - Normalizes Unicode forms

### 2. **Chain of Responsibility**
`TextCleaningPipeline` applies strategies in sequence, each processing the output of the previous one.

### 3. **Factory Pattern**
`TextCleanerFactory` creates pre-configured pipelines for different use cases.

## Features

### Handles Edge Cases

✅ **Unicode Surrogates** - Removes `\ud800-\udfff` that cause UTF-8 encoding errors
✅ **Mathematical Notation** - Converts Greek letters, operators, and symbols to ASCII
✅ **Table Structures** - Preserves tables with ASCII formatting
✅ **Special Symbols** - Normalizes quotes, bullets, currencies, fractions
✅ **Whitespace** - Normalizes while preserving paragraph structure
✅ **Control Characters** - Removes invalid control characters

## Usage

### Basic Usage

```python
from ingestion.text_cleaning import TextCleanerFactory

# Create default cleaner
cleaner = TextCleanerFactory.create_default_cleaner()

# Clean text
text = "Distance formula: d = \ud835\udc65 between two points"
cleaned = cleaner.clean(text)
```

### Pre-configured Pipelines

```python
# Default (recommended for RAG)
cleaner = TextCleanerFactory.create_default_cleaner()

# Aggressive (more normalization)
cleaner = TextCleanerFactory.create_aggressive_cleaner()

# Minimal (only essential cleaning)
cleaner = TextCleanerFactory.create_minimal_cleaner()
```

### Custom Configuration

```python
cleaner = TextCleanerFactory.create_custom_cleaner(
    remove_surrogates=True,
    normalize_unicode=True,
    normalize_math=True,
    preserve_tables=True,
    normalize_symbols=True,
    normalize_whitespace=True
)
```

### Custom Pipeline

```python
from ingestion.text_cleaning import (
    TextCleaningPipeline,
    SurrogateRemovalStrategy,
    MathNotationNormalizer,
    WhitespaceNormalizer
)

# Build custom pipeline
pipeline = TextCleaningPipeline([
    SurrogateRemovalStrategy(),
    MathNotationNormalizer(),
    WhitespaceNormalizer()
])

cleaned = pipeline.clean(text, log_steps=True)
```

## Cleaning Strategies

### SurrogateRemovalStrategy

Removes Unicode surrogate characters (`U+D800` to `U+DFFF`) that cause database encoding errors.

**Input:**
```
Distance formula: d = 𝕩 between two points
```

**Output:**
```
Distance formula: d =  between two points
```

### MathNotationNormalizer

Converts mathematical notation to ASCII equivalents.

**Conversions:**
- Greek letters: `α → alpha`, `β → beta`, `π → pi`
- Operators: `× → *`, `÷ → /`, `≈ → ~=`
- Symbols: `∑ → sum`, `∫ → integral`, `√ → sqrt`

**Input:**
```
α + β ≈ γ × π²
```

**Output:**
```
alpha + beta ~= gamma * pi^2
```

### TableStructurePreserver

Preserves table structure using ASCII formatting.

**Input:**
```
┌─────┬──────┐
│ A   │  B   │
├─────┼──────┤
│ 1   │  2   │
└─────┴──────┘
```

**Output:**
```
+-----+------+
| A   |  B   |
+-----+------+
| 1   |  2   |
+-----+------+
```

### SpecialSymbolNormalizer

Normalizes special symbols to ASCII.

**Conversions:**
- Quotes: `"text" → "text"`, `'text' → 'text'`
- Currencies: `€100 → EUR100`, `£50 → GBP50`
- Fractions: `½ → 1/2`, `¼ → 1/4`
- Bullets: `• item → * item`

### WhitespaceNormalizer

Normalizes whitespace while preserving paragraph structure.

**Rules:**
- Multiple spaces → single space
- Multiple newlines (≥3) → double newline (preserves paragraphs)
- Removes trailing/leading whitespace on lines

### UnicodeNormalizer

Normalizes Unicode to standard forms (NFKC by default).

**Benefit:** Ensures consistent character representation across different encodings.

## Integration

### Automatic Integration

The text cleaning pipeline is automatically integrated into the document processing workflow in `vector_store.py`:

```python
# Automatically applied during document processing
text_cleaner = TextCleanerFactory.create_default_cleaner()
cleaned_text = text_cleaner.clean(chunk.text)
```

### Manual Usage

```python
from ingestion.text_cleaning import TextCleanerFactory

cleaner = TextCleanerFactory.create_default_cleaner()

# Clean individual text
cleaned = cleaner.clean(raw_text)

# Clean with logging
cleaned = cleaner.clean(raw_text, log_steps=True)
```

## Testing

Run the test suite:

```bash
python test_text_cleaning.py
```

Tests cover:
1. Surrogate character removal
2. Math notation normalization
3. Table structure preservation
4. Special symbol normalization
5. Combined real-world examples
6. Custom configurations
7. Minimal vs aggressive cleaning

## Performance

- **Overhead:** ~1-5ms per chunk (negligible)
- **Memory:** Minimal (no buffering)
- **Parallelizable:** Yes (stateless strategies)

## Extensibility

### Add Custom Strategy

```python
from ingestion.text_cleaning import TextCleaningStrategy

class CustomStrategy(TextCleaningStrategy):
    def clean(self, text: str) -> str:
        # Your cleaning logic
        return processed_text

    def get_name(self) -> str:
        return "custom_strategy"

# Add to pipeline
pipeline.add_strategy(CustomStrategy())
```

### Remove Strategy

```python
pipeline.remove_strategy("math_notation_normalizer")
```

## Common Use Cases

### Case 1: PDF with Mathematical Equations

```python
# Use default cleaner (includes math normalization)
cleaner = TextCleanerFactory.create_default_cleaner()
```

### Case 2: Tables from DOCX/Excel

```python
# Ensures table structure is preserved
cleaner = TextCleanerFactory.create_custom_cleaner(
    preserve_tables=True,
    normalize_math=False  # Might have numbers that look like math
)
```

### Case 3: Web Scraping (Lots of HTML entities)

```python
# Aggressive normalization
cleaner = TextCleanerFactory.create_aggressive_cleaner()
```

### Case 4: Plain Text (Minimal Processing)

```python
# Only essential cleaning
cleaner = TextCleanerFactory.create_minimal_cleaner()
```

## Troubleshooting

### Issue: Text is over-normalized

**Solution:** Use minimal cleaner or custom configuration:
```python
cleaner = TextCleanerFactory.create_minimal_cleaner()
```

### Issue: Math symbols important for semantic meaning

**Solution:** Disable math normalization:
```python
cleaner = TextCleanerFactory.create_custom_cleaner(
    normalize_math=False
)
```

### Issue: Performance concerns

**Solution:** Strategies are lightweight and stateless. For large batches, consider:
```python
cleaner = TextCleanerFactory.create_minimal_cleaner()
```

## Architecture

```
TextCleaningStrategy (ABC)
    ├── SurrogateRemovalStrategy
    ├── MathNotationNormalizer
    ├── TableStructurePreserver
    ├── SpecialSymbolNormalizer
    ├── WhitespaceNormalizer
    └── UnicodeNormalizer

TextCleaningPipeline
    └── Applies strategies in sequence

TextCleanerFactory
    ├── create_default_cleaner()
    ├── create_aggressive_cleaner()
    ├── create_minimal_cleaner()
    └── create_custom_cleaner(...)
```

## Best Practices

1. **Use default cleaner** for most RAG applications
2. **Log steps** during development to understand transformations
3. **Test with real data** from your domain
4. **Customize carefully** - over-normalization loses information
5. **Monitor logfire** for cleaning failures

## Future Enhancements

- [ ] Language-specific cleaning strategies
- [ ] Domain-specific vocabularies (medical, legal, etc.)
- [ ] Configurable math symbol preservation rules
- [ ] HTML/Markdown structure preservation
- [ ] Async pipeline support
- [ ] Parallel strategy execution
