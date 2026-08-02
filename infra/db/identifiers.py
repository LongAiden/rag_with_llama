"""Safe PostgreSQL identifier helpers used by the DB layer."""

import re

_SAFE_TABLE_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')

_SYSTEM_TABLES = frozenset([
    'entities', 'relationships', 'entity_nodes', 'entity_edges',
])


def validate_table_name(table_name: str) -> str:
    """Return the table name if it is a safe PostgreSQL identifier.

    Raises:
        ValueError: If the name contains characters outside letters, digits, or
            underscores, or exceeds 63 characters.
    """
    if not _SAFE_TABLE_PATTERN.match(table_name):
        raise ValueError(
            f"Invalid table name: {table_name!r}. "
            "Use only letters, digits, and underscores (max 63 chars, must start with letter/underscore)."
        )
    return table_name


def quote_ident(name: str) -> str:
    """Validate ``name`` and return it as a double-quoted identifier."""
    validate_table_name(name)
    return f'"{name}"'
