"""
Unit tests for on-disk ingestion artifacts.

The parse stage writes the markdown it produced, and the chunk stage writes one
file per chunk plus an index, so a bad chunk boundary can be inspected without
querying JSONB. These artifacts are a debugging aid, never the source of truth —
which is why every writer here is expected to degrade to a no-op rather than
fail the ingestion that produced them.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.ingestion import artifacts


@pytest.fixture
def artifact_dirs(tmp_path, monkeypatch):
    """Point both artifact writers at a temp directory."""
    parsed = tmp_path / "parsed"
    chunks = tmp_path / "chunks"
    monkeypatch.setattr(artifacts, "PARSED_DIR", parsed)
    monkeypatch.setattr(artifacts, "CHUNKS_DIR", chunks)
    monkeypatch.setattr(artifacts, "PERSIST_ARTIFACTS", True)
    return SimpleNamespace(parsed=parsed, chunks=chunks)


def make_chunk(text, **kwargs):
    """Build a chunker-like object with the attributes the writer reads."""
    defaults = {
        "token_count": 12,
        "start_index": 0,
        "end_index": len(text),
        "page_number": 1,
        "section_path": "",
        "full_content": "the whole page",
    }
    defaults.update(kwargs)
    return SimpleNamespace(text=text, **defaults)


DOC_ID = "3f2b1c4d-0000-4000-8000-abcdefabcdef"


class TestWriteParsedDocument:
    """data/parsed/<document_id>_<stem>.md"""

    def test_writes_markdown_next_to_document_id(self, artifact_dirs):
        path = artifacts.write_parsed_document(
            DOC_ID, "# Title\n\nBody text.", filename="llama2.pdf"
        )

        assert path is not None
        assert path.parent == artifact_dirs.parsed
        assert path.name == f"{DOC_ID}_llama2.md"
        assert path.read_text(encoding="utf-8") == "# Title\n\nBody text."

    def test_creates_parent_directory(self, artifact_dirs):
        assert not artifact_dirs.parsed.exists()

        artifacts.write_parsed_document(DOC_ID, "text", filename="a.pdf")

        assert artifact_dirs.parsed.is_dir()

    def test_falls_back_to_document_id_without_filename(self, artifact_dirs):
        path = artifacts.write_parsed_document(DOC_ID, "text")

        assert path.name == f"{DOC_ID}.md"

    def test_reparse_overwrites_rather_than_appends(self, artifact_dirs):
        artifacts.write_parsed_document(DOC_ID, "first pass", filename="a.pdf")
        path = artifacts.write_parsed_document(DOC_ID, "second pass", filename="a.pdf")

        assert path.read_text(encoding="utf-8") == "second pass"

    def test_disabled_writes_nothing(self, artifact_dirs, monkeypatch):
        monkeypatch.setattr(artifacts, "PERSIST_ARTIFACTS", False)

        assert artifacts.write_parsed_document(DOC_ID, "text", filename="a.pdf") is None
        assert not artifact_dirs.parsed.exists()

    def test_empty_text_still_written(self, artifact_dirs):
        """An empty parse result is itself the thing worth inspecting."""
        path = artifacts.write_parsed_document(DOC_ID, "", filename="a.pdf")

        assert path is not None
        assert path.read_text(encoding="utf-8") == ""


class TestWriteChunkArtifacts:
    """data/chunks/<document_id>_<stem>/{0000.md, index.json}"""

    def test_creates_one_subdirectory_per_document(self, artifact_dirs):
        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk("alpha"), make_chunk("beta")], filename="llama2.pdf"
        )

        assert directory is not None
        assert directory.parent == artifact_dirs.chunks
        assert directory.name == f"{DOC_ID}_llama2"
        assert directory.is_dir()

    def test_writes_one_file_per_chunk_zero_padded(self, artifact_dirs):
        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk(f"chunk {i}") for i in range(3)], filename="a.pdf"
        )

        assert sorted(p.name for p in directory.glob("*.md")) == [
            "0000.md", "0001.md", "0002.md",
        ]
        assert (directory / "0001.md").read_text(encoding="utf-8") == "chunk 1"

    def test_index_records_metadata_for_every_chunk(self, artifact_dirs):
        chunks = [
            make_chunk("alpha", page_number=1, section_path="[Intro]",
                       token_count=5, start_index=0, end_index=5),
            make_chunk("beta", page_number=3, section_path="[Intro].[Body]",
                       token_count=7, start_index=5, end_index=9),
        ]

        directory = artifacts.write_chunk_artifacts(DOC_ID, chunks, filename="llama2.pdf")
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))

        assert index["document_id"] == DOC_ID
        assert index["file_name"] == "llama2.pdf"
        assert index["chunk_count"] == 2
        assert index["chunks"][1] == {
            "index": 1,
            "file": "0001.md",
            "page_number": 3,
            "section_path": "[Intro].[Body]",
            "token_count": 7,
            "start_index": 5,
            "end_index": 9,
            "char_count": len("beta"),
        }

    def test_index_omits_page_content(self, artifact_dirs):
        """full_content repeats the whole page per chunk — it would dwarf the index."""
        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk("alpha", full_content="x" * 5000)], filename="a.pdf"
        )

        raw = (directory / "index.json").read_text(encoding="utf-8")

        assert "xxxx" not in raw
        assert "full_content" not in raw

    def test_rechunk_replaces_previous_chunk_files(self, artifact_dirs):
        """A retry that produces fewer chunks must not leave the old tail behind."""
        artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk(f"old {i}") for i in range(5)], filename="a.pdf"
        )

        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk("new 0")], filename="a.pdf"
        )

        assert sorted(p.name for p in directory.glob("*.md")) == ["0000.md"]
        assert (directory / "0000.md").read_text(encoding="utf-8") == "new 0"

    def test_empty_chunk_list_writes_index_only(self, artifact_dirs):
        directory = artifacts.write_chunk_artifacts(DOC_ID, [], filename="a.pdf")

        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))

        assert index["chunk_count"] == 0
        assert index["chunks"] == []
        assert list(directory.glob("*.md")) == []

    def test_handles_chunks_missing_optional_attributes(self, artifact_dirs):
        """Chunker outputs vary; a bare object with only .text must not crash."""
        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [SimpleNamespace(text="bare")], filename="a.pdf"
        )

        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))

        assert (directory / "0000.md").read_text(encoding="utf-8") == "bare"
        assert index["chunks"][0]["page_number"] == 1
        assert index["chunks"][0]["token_count"] is None

    def test_disabled_writes_nothing(self, artifact_dirs, monkeypatch):
        monkeypatch.setattr(artifacts, "PERSIST_ARTIFACTS", False)

        assert artifacts.write_chunk_artifacts(DOC_ID, [make_chunk("a")]) is None
        assert not artifact_dirs.chunks.exists()


class TestPathSafety:
    """document_id and filename both reach the filesystem as path components."""

    @pytest.mark.parametrize("bad_id", [
        "../../etc/passwd",
        "..",
        "a/b",
        "a\\b",
        "",
        "with space",
        "semi;colon",
    ])
    def test_rejects_unsafe_document_id(self, artifact_dirs, bad_id):
        assert artifacts.write_parsed_document(bad_id, "text") is None
        assert artifacts.write_chunk_artifacts(bad_id, [make_chunk("a")]) is None

    def test_strips_directory_components_from_filename(self, artifact_dirs):
        path = artifacts.write_parsed_document(
            DOC_ID, "text", filename="../../../etc/passwd.pdf"
        )

        assert path.parent == artifact_dirs.parsed
        assert path.name == f"{DOC_ID}_passwd.md"

    def test_sanitises_awkward_filename_characters(self, artifact_dirs):
        path = artifacts.write_parsed_document(
            DOC_ID, "text", filename="my report (final) v2.pdf"
        )

        assert path.name == f"{DOC_ID}_my_report_final_v2.md"

    def test_filename_that_sanitises_to_nothing_falls_back(self, artifact_dirs):
        path = artifacts.write_parsed_document(DOC_ID, "text", filename="....pdf")

        assert path.name == f"{DOC_ID}.md"


class TestPipelineWiring:
    """The parse and chunk stages must actually call the writers."""

    async def test_parse_file_writes_parsed_markdown(self, tmp_path, monkeypatch):
        from app.ingestion.embedding import pipeline as pipeline_module

        source = tmp_path / "notes.txt"
        source.write_text("hello world", encoding="utf-8")

        writer = MagicMock()
        monkeypatch.setattr(pipeline_module, "write_parsed_document", writer)

        await pipeline_module.ChunkEmbeddingPipeline.parse_file(str(source), DOC_ID)

        writer.assert_called_once()
        assert writer.call_args.args[0] == DOC_ID
        assert writer.call_args.args[1] == "hello world"
        assert writer.call_args.kwargs["filename"] == "notes.txt"

    @staticmethod
    def _stub_chunker(monkeypatch):
        """Replace the real chunker — this asserts on wiring, not on chunking."""
        from app.ingestion.chunking import chunker_factory

        chunker = MagicMock()
        chunker.chunk.return_value = [make_chunk("alpha"), make_chunk("beta")]
        monkeypatch.setattr(chunker_factory, "get_chunker", lambda **kwargs: chunker)
        return chunker

    def test_chunk_parsed_document_writes_chunk_artifacts(self, monkeypatch):
        from app.ingestion.embedding import pipeline as pipeline_module

        self._stub_chunker(monkeypatch)
        writer = MagicMock()
        monkeypatch.setattr(pipeline_module, "write_chunk_artifacts", writer)

        chunks = pipeline_module.ChunkEmbeddingPipeline.chunk_parsed_document(
            {
                "parsed_text": "Some plain text to split into chunks.",
                "file_type": "txt",
                "filename": "notes.txt",
                "page_mapping": [],
            },
            document_id=DOC_ID,
        )

        writer.assert_called_once()
        assert writer.call_args.args[0] == DOC_ID
        assert list(writer.call_args.args[1]) == list(chunks)
        assert writer.call_args.kwargs["filename"] == "notes.txt"

    def test_chunk_parsed_document_skips_write_without_document_id(self, monkeypatch):
        """Without an id there is no per-document folder to write into."""
        from app.ingestion.embedding import pipeline as pipeline_module

        self._stub_chunker(monkeypatch)
        writer = MagicMock()
        monkeypatch.setattr(pipeline_module, "write_chunk_artifacts", writer)

        pipeline_module.ChunkEmbeddingPipeline.chunk_parsed_document({
            "parsed_text": "Some plain text.",
            "file_type": "txt",
            "filename": "notes.txt",
            "page_mapping": [],
        })

        writer.assert_not_called()


class TestFailuresNeverPropagate:
    """Artifacts are for humans; losing one must not fail an ingestion."""

    def test_parsed_write_error_returns_none(self, artifact_dirs, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)

        assert artifacts.write_parsed_document(DOC_ID, "text", filename="a.pdf") is None

    def test_chunk_write_error_returns_none(self, artifact_dirs, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)

        assert artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk("a")], filename="a.pdf"
        ) is None

    def test_unserialisable_chunk_metadata_does_not_raise(self, artifact_dirs):
        """Chunker attributes reach json.dumps unvalidated (e.g. a numpy count)."""
        class Weird:
            def __repr__(self):
                return "Weird()"

        directory = artifacts.write_chunk_artifacts(
            DOC_ID, [make_chunk("alpha", token_count=Weird())], filename="a.pdf"
        )

        assert directory is not None
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        assert index["chunks"][0]["token_count"] == "Weird()"
