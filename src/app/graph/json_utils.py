"""
Robust JSON parsing utilities for handling LLM responses.
Handles common JSON errors from Gemini API responses.
"""

import json
import re
import logfire
from typing import Any, Optional


class JSONParser:
    """Robust JSON parser with error recovery for LLM responses."""

    @staticmethod
    def extract_and_parse(text: str, expected_type: str = "array") -> Optional[Any]:
        """
        Extract and parse JSON from LLM response with multiple fallback strategies.

        Args:
            text: Raw text from LLM response
            expected_type: "array" or "object"

        Returns:
            Parsed JSON data or None if parsing fails
        """
        # Strategy 1: Try markdown code blocks first
        result = JSONParser._try_markdown_extraction(text, expected_type)
        if result is not None:
            return result

        # Strategy 2: Try finding JSON with balanced brackets
        result = JSONParser._try_balanced_extraction(text, expected_type)
        if result is not None:
            return result

        # Strategy 3: Try basic regex extraction
        result = JSONParser._try_regex_extraction(text, expected_type)
        if result is not None:
            return result

        # Strategy 4: Try repair common JSON errors
        result = JSONParser._try_json_repair(text, expected_type)
        if result is not None:
            return result

        logfire.error("All JSON extraction strategies failed",
                     text_preview=text[:500])
        return None

    @staticmethod
    def _try_markdown_extraction(text: str, expected_type: str) -> Optional[Any]:
        """Try to extract JSON from markdown code blocks."""
        try:
            if expected_type == "array":
                pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
            else:
                pattern = r'```(?:json)?\s*(\{.*?\})\s*```'

            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                data = json.loads(json_str)
                logfire.info("JSON extracted from markdown code block")
                return data
        except Exception as e:
            logfire.debug(f"Markdown extraction failed: {e}")
        return None

    @staticmethod
    def _try_balanced_extraction(text: str, expected_type: str) -> Optional[Any]:
        """
        Extract JSON by finding balanced brackets.
        More reliable than greedy regex for nested structures.
        """
        try:
            start_char = '[' if expected_type == "array" else '{'
            end_char = ']' if expected_type == "array" else '}'

            # Find first occurrence of start character
            start_idx = text.find(start_char)
            if start_idx == -1:
                return None

            # Track bracket depth to find matching end
            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                char = text[i]

                # Handle string escapes
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                # Track string boundaries
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                # Only count brackets outside strings
                if not in_string:
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            # Found matching bracket
                            json_str = text[start_idx:i+1]
                            data = json.loads(json_str)
                            logfire.info("JSON extracted using balanced bracket matching")
                            return data

        except Exception as e:
            logfire.debug(f"Balanced extraction failed: {e}")
        return None

    @staticmethod
    def _try_regex_extraction(text: str, expected_type: str) -> Optional[Any]:
        """Try simple regex extraction as fallback."""
        try:
            if expected_type == "array":
                pattern = r'\[.*\]'
            else:
                pattern = r'\{.*\}'

            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0).strip()
                # Limit size to prevent hanging on huge malformed JSON
                if len(json_str) > 500000:  # 500KB limit
                    logfire.warn("JSON too large, truncating", size=len(json_str))
                    return None

                data = json.loads(json_str)
                logfire.info("JSON extracted using regex")
                return data
        except Exception as e:
            logfire.debug(f"Regex extraction failed: {e}")
        return None

    @staticmethod
    def _try_json_repair(text: str, expected_type: str) -> Optional[Any]:
        """Try to repair common JSON errors."""
        try:
            # Extract potential JSON
            if expected_type == "array":
                match = re.search(r'\[.*\]', text, re.DOTALL)
            else:
                match = re.search(r'\{.*\}', text, re.DOTALL)

            if not match:
                return None

            json_str = match.group(0)

            # Repair strategies
            repairs = [
                # Remove trailing commas
                lambda s: re.sub(r',(\s*[}\]])', r'\1', s),
                # Fix unescaped quotes in strings (risky, only basic cases)
                lambda s: re.sub(r'([^\\])"([^",:}\]]*)"([^,:}\]]*)"', r'\1"\2\\\"\3"', s),
                # Remove duplicate commas
                lambda s: re.sub(r',\s*,', ',', s),
                # Fix missing commas between objects
                lambda s: re.sub(r'}\s*{', '},{', s),
                lambda s: re.sub(r']\s*\[', '],[', s),
            ]

            for repair in repairs:
                try:
                    repaired = repair(json_str)
                    data = json.loads(repaired)
                    logfire.info("JSON parsed after repair")
                    return data
                except:
                    continue

        except Exception as e:
            logfire.debug(f"JSON repair failed: {e}")
        return None

    @staticmethod
    def parse_with_fallback(
        text: str,
        expected_type: str = "array",
        context: str = ""
    ) -> Any:
        """
        Parse JSON with comprehensive error handling and logging.

        Args:
            text: Raw text containing JSON
            expected_type: "array" or "object"
            context: Description of what's being parsed (for logging)

        Returns:
            Parsed JSON or empty list/dict depending on expected_type
        """
        result = JSONParser.extract_and_parse(text, expected_type)

        if result is not None:
            return result

        # Return empty structure as fallback
        default = [] if expected_type == "array" else {}

        logfire.error("JSON parsing completely failed, using empty fallback",
                     context=context,
                     text_length=len(text),
                     text_preview=text[:500])

        return default


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logfire.error("JSON decode error",
                     error=str(e),
                     line=e.lineno if hasattr(e, 'lineno') else None,
                     col=e.colno if hasattr(e, 'colno') else None)
        return default
    except Exception as e:
        logfire.error("Unexpected error parsing JSON",
                     error=str(e),
                     error_type=type(e).__name__)
        return default
