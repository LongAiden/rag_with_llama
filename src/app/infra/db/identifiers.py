"""Safe PostgreSQL identifier helpers used by the DB layer."""

import re

_SAFE_TABLE_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')

# Names a chunk table may never take. Two groups:
#
# - Graph tables, which are unwired (ARCHITECTURE.md 9.3) but whose names are
#   kept reserved so a user cannot create or drop a chunk table over them.
# - Application tables. Without these, an upload with table_name='documents'
#   reaches VectorStore._initialize_database(), whose CREATE TABLE IF NOT EXISTS
#   silently matches the existing status table and then fails on INSERT with a
#   column error. 'domains' joined this list in migration 006.
_SYSTEM_TABLES = frozenset([
    'entities', 'relationships', 'entity_nodes', 'entity_edges',
    'domains', 'documents', 'document_parsed', 'document_chunked',
    'llm_interactions', 'schema_migrations',
])


def validate_table_name(table_name: str) -> str:
    """Return the table name if it is a safe, non-reserved PostgreSQL identifier.

    Raises:
        ValueError: If the name contains characters outside letters, digits, or
            underscores, exceeds 63 characters, or is a reserved system table.
    """
    if not _SAFE_TABLE_PATTERN.match(table_name):
        raise ValueError(
            f"Invalid table name: {table_name!r}. "
            "Use only letters, digits, and underscores (max 63 chars, must start with letter/underscore)."
        )
    if table_name.lower() in _SYSTEM_TABLES:
        raise ValueError(
            f"Reserved table name: {table_name!r}. "
            f"These names belong to the application schema: {', '.join(sorted(_SYSTEM_TABLES))}."
        )
    return table_name


def quote_ident(name: str) -> str:
    """Validate ``name`` and return it as a double-quoted identifier."""
    validate_table_name(name)
    return f'"{name}"'
