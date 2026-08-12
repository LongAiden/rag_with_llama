"""On-disk artifacts for the parse and chunk stages.

Postgres stays the source of truth: `document_parsed` holds the parsed text and
`document_chunked` holds the serialized chunks. These files exist so a human can
*look* at them. Before this, chunks went from `chunk_markdown()` straight into
the vector store in memory, so when embedding quality looked off there was
nothing on disk to inspect — only JSONB to query.

Layout::

    data/parsed/<document_id>_<stem>.md      one file per parsed document
    data/chunks/<document_id>_<stem>/
        0000.md                              chunk text, one file per chunk
        0001.md
        index.json                           per-chunk metadata

The `<document_id>_<stem>` naming mirrors `data/input/raw/<document_id>_<filename>`,
so a document's raw, parsed, and chunked forms all sort together under the same
prefix.

Nothing here raises. A failed artifact write is logged and skipped, because a
debugging aid must never fail an ingestion that otherwise succeeded.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import logfire

from app.config.app_config import AppSettings

_SETTINGS = AppSettings()

PARSED_DIR = Path(_SETTINGS.parsed_dir)
CHUNKS_DIR = Path(_SETTINGS.chunks_dir)
PERSIST_ARTIFACTS = _SETTINGS.persist_ingestion_artifacts

# document_id becomes a path component, so it is allowlisted rather than escaped.
# Real ids are UUID4 strings; anything else is refused outright.
_SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Characters kept in the human-readable part of a filename. Everything else
# collapses to a single underscore.
_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(filename: str) -> str:
    """Reduce a filename to a safe, readable path fragment ('' if nothing is left)."""
    base = Path(filename).name  # drops any directory components, including '..'
    stem = Path(base).stem
    stem = _UNSAFE_STEM_CHARS.sub("_", stem)
    # Strip after truncating too, so a cut mid-name cannot leave a trailing separator.
    return stem[:80].strip("._-")


def _artifact_name(document_id: str, filename: str) -> Optional[str]:
    """Build `<document_id>_<stem>`, or None when the id is not path-safe."""
    if not _SAFE_DOCUMENT_ID.match(document_id or ""):
        logfire.warn("Refusing to write ingestion artifact for unsafe document id",
                     document_id=str(document_id)[:100])
        return None

    stem = _safe_stem(filename or "")
    return f"{document_id}_{stem}" if stem else document_id


def write_parsed_document(
    document_id: str,
    parsed_text: str,
    *,
    filename: str = "",
) -> Optional[Path]:
    """Write the parse stage's markdown to `data/parsed/`.

    Args:
        document_id: The document's id; also the artifact's filename prefix.
        parsed_text: Markdown or plain text produced by the parser.
        filename: Original filename, used for the readable half of the name.

    Returns:
        The path written, or None if artifacts are disabled, the id is unsafe,
        or the write failed.
    """
    if not PERSIST_ARTIFACTS:
        return None

    name = _artifact_name(document_id, filename)
    if name is None:
        return None

    try:
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PARSED_DIR / f"{name}.md"
        path.write_text(parsed_text or "", encoding="utf-8")
    # See write_chunk_artifacts: an artifact is a debugging aid, so no failure
    # writing one may propagate into the stage that produced it.
    except Exception as exc:
        logfire.warn("Could not write parsed artifact",
                     document_id=document_id,
                     error_type=type(exc).__name__, error=str(exc))
        return None

    logfire.info("Parsed artifact written",
                 document_id=document_id, path=str(path), chars=len(parsed_text or ""))
    return path


def _chunk_index_entry(index: int, chunk: Any) -> Dict[str, Any]:
    """Metadata row for one chunk.

    `full_content` is deliberately excluded: it holds the chunk's entire source
    page, so every chunk on a page would repeat that page's text into the index.
    """
    text = getattr(chunk, "text", "") or ""
    return {
        "index": index,
        "file": f"{index:04d}.md",
        "page_number": getattr(chunk, "page_number", 1),
        "section_path": getattr(chunk, "section_path", "") or "",
        "token_count": getattr(chunk, "token_count", None),
        "start_index": getattr(chunk, "start_index", None),
        "end_index": getattr(chunk, "end_index", None),
        "char_count": len(text),
    }


def write_chunk_artifacts(
    document_id: str,
    chunks: Sequence[Any],
    *,
    filename: str = "",
) -> Optional[Path]:
    """Write one file per chunk, plus an index, to `data/chunks/<document>/`.

    The document's directory is cleared first: a retry that produces fewer chunks
    than the previous run would otherwise leave the old tail behind, and the
    stale files would read as real output.

    Args:
        document_id: The document's id; also the directory's name prefix.
        chunks: Chunker output objects (only `.text` is required).
        filename: Original filename, used for the readable half of the name.

    Returns:
        The directory written, or None if artifacts are disabled, the id is
        unsafe, or the write failed.
    """
    if not PERSIST_ARTIFACTS:
        return None

    name = _artifact_name(document_id, filename)
    if name is None:
        return None

    directory = CHUNKS_DIR / name
    try:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            entry = _chunk_index_entry(i, chunk)
            (directory / entry["file"]).write_text(
                getattr(chunk, "text", "") or "", encoding="utf-8"
            )
            entries.append(entry)

        index = {
            "document_id": document_id,
            "file_name": filename,
            "chunk_count": len(entries),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "chunks": entries,
        }
        (directory / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    # Deliberately broad: besides OSError, chunker attributes reach json.dumps
    # unvalidated (a numpy token_count raises TypeError, for one). Whatever the
    # cause, losing a debugging dump must not fail the ingestion behind it.
    except Exception as exc:
        logfire.warn("Could not write chunk artifacts",
                     document_id=document_id,
                     error_type=type(exc).__name__, error=str(exc))
        return None

    logfire.info("Chunk artifacts written",
                 document_id=document_id, path=str(directory), chunk_count=len(entries))
    return directory
