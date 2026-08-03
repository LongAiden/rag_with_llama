"""
Factory for creating appropriate document processors.
Implements the Factory Method pattern to automatically select the right processor.
"""

from typing import List, Optional
from pathlib import Path

from .base_processor import DocumentProcessor
from .docx_processor import DOCXProcessor
from .txt_processor import TXTProcessor


class ProcessorRegistry:
    """
    Registry for document processors.

    Automatically selects the appropriate processor based on file extension.
    Uses the Factory Method pattern to create processor instances.
    """

    def __init__(self):
        """Initialize registry with all available processors."""
        self._processors: List[DocumentProcessor] = []
        self._register_default_processors()

    def _register_default_processors(self):
        """Register all built-in processors.

        PDFs are handled by the docling-based parsers in the pipeline, not by the
        generic processor registry.
        """
        self.register(DOCXProcessor())
        self.register(TXTProcessor())

    def register(self, processor: DocumentProcessor):
        """
        Register a new processor.

        Args:
            processor: DocumentProcessor instance to register
        """
        self._processors.append(processor)
        print(f"Registered processor: {processor}")

    def get_processor(self, file_path: str) -> DocumentProcessor:
        """
        Get the appropriate processor for a file.

        Args:
            file_path: Path to the file to process

        Returns:
            DocumentProcessor instance that can handle the file

        Raises:
            ValueError: If no processor can handle the file type
        """
        file_extension = Path(file_path).suffix.lower()

        # Try each processor to see if it can handle the file
        for processor in self._processors:
            if processor.can_process(file_path):
                return processor

        # No processor found
        supported_extensions = []
        for processor in self._processors:
            supported_extensions.extend(processor.get_supported_extensions())

        raise ValueError(
            f"No processor found for file type '{file_extension}'. "
            f"Supported types: {', '.join(set(supported_extensions))}"
        )

    def list_processors(self) -> List[DocumentProcessor]:
        """
        Get list of all registered processors.

        Returns:
            List of all registered processors
        """
        return self._processors.copy()

    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions across all processors.

        Returns:
            List of all supported extensions
        """
        extensions = []
        for processor in self._processors:
            extensions.extend(processor.get_supported_extensions())
        return list(set(ext.lower() for ext in extensions))


# Global singleton registry instance
_global_registry: Optional[ProcessorRegistry] = None


def get_registry() -> ProcessorRegistry:
    """
    Get the global processor registry (singleton pattern).

    Returns:
        Global ProcessorRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ProcessorRegistry()
    return _global_registry


def get_processor_for_file(file_path: str) -> DocumentProcessor:
    """
    Convenience function to get processor for a file.

    Args:
        file_path: Path to the file

    Returns:
        Appropriate DocumentProcessor instance

    Raises:
        ValueError: If no processor can handle the file
    """
    registry = get_registry()
    return registry.get_processor(file_path)
