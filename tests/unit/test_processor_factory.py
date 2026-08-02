"""
Unit tests for ingestion/processors/processor_factory.py.

Covers:
- ProcessorRegistry registration and lookup
- get_processor for supported/unsupported extensions
- get_supported_extensions
- Singleton pattern
- get_processor_for_file convenience function
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ingestion.processors.processor_factory import (
    ProcessorRegistry,
    get_registry,
    get_processor_for_file,
    _global_registry,
)
import ingestion.processors.processor_factory as pf_module


class TestProcessorRegistry:
    def test_default_processors_registered(self):
        registry = ProcessorRegistry()
        processors = registry.list_processors()
        assert len(processors) == 2

    def test_get_supported_extensions(self):
        registry = ProcessorRegistry()
        extensions = registry.get_supported_extensions()
        assert ".txt" in extensions
        assert ".docx" in extensions

    def test_get_processor_for_txt(self):
        registry = ProcessorRegistry()
        processor = registry.get_processor("test.txt")
        assert processor is not None
        assert processor.can_process("test.txt")

    def test_get_processor_for_docx(self):
        registry = ProcessorRegistry()
        processor = registry.get_processor("test.docx")
        assert processor is not None
        assert processor.can_process("test.docx")

    def test_get_processor_unsupported_raises(self):
        registry = ProcessorRegistry()
        with pytest.raises(ValueError, match="No processor found"):
            registry.get_processor("test.xyz")

    def test_register_custom_processor(self):
        registry = ProcessorRegistry()
        mock_processor = MagicMock()
        mock_processor.get_supported_extensions.return_value = [".custom"]
        mock_processor.can_process.return_value = True

        registry.register(mock_processor)
        assert mock_processor in registry.list_processors()

    def test_list_processors_returns_copy(self):
        registry = ProcessorRegistry()
        procs1 = registry.list_processors()
        procs2 = registry.list_processors()
        assert procs1 is not procs2


class TestGetRegistrySingleton:
    def test_returns_same_instance(self):
        pf_module._global_registry = None
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
        pf_module._global_registry = None


class TestGetProcessorForFile:
    def test_returns_processor_for_docx(self):
        pf_module._global_registry = None
        processor = get_processor_for_file("document.docx")
        assert processor is not None
        pf_module._global_registry = None

    def test_raises_for_unsupported(self):
        pf_module._global_registry = None
        with pytest.raises(ValueError):
            get_processor_for_file("document.xyz")
        pf_module._global_registry = None
