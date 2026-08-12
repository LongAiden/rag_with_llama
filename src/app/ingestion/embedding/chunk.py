"""Chunk data model for the embedding pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Chunk:
    """Chunk data structure to match your existing interface."""
    id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict] = None
    # Human-readable name of the source document, denormalized onto every chunk row
    # so search results can be attributed without joining `documents`. A snapshot of
    # documents.doc_name at ingest time, not a mirror of it.
    doc_name: Optional[str] = None
