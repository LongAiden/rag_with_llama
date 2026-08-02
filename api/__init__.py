"""
API layer - FastAPI application and routes.
"""

from .app import app
from .dependencies import get_pipeline

__all__ = [
    'app',
    'get_pipeline',
]
