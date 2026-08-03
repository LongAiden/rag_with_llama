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
