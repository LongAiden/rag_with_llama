"""Complete pipeline for chunking documents and storing embeddings."""

import asyncio
import os
import re
import traceback
import uuid
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import logfire

from app.ingestion.artifacts import write_chunk_artifacts, write_parsed_document
from app.ingestion.embedding.chunk import Chunk
from app.ingestion.embedding.generator import EmbeddingGenerator
from app.ingestion.embedding.vector_store import VectorStore
from app.ingestion.processors.page_utils import get_page_number_for_position
from app.ingestion.processors.processor_factory import get_processor_for_file
from app.ingestion.text_cleaning.cleaners import TextCleaningPipeline

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class _MarkdownStructure:
    """Page-marker and heading positions for a document, scanned once.

    Resolving these per chunk the direct way — slicing markdown[:start_index] and
    re-running both regexes over the prefix — is quadratic in document length. One
    forward pass plus a binary search per chunk is linear.
    """

    def __init__(self, markdown: str):
        self.page_positions: List[int] = []
        self.page_numbers: List[int] = []
        for match in re.finditer(r'\[Page (\d+)\]', markdown):
            # Keyed on the end of the marker: a page starts where its marker
            # finishes, which is also what _extract_page_content slices from.
            self.page_positions.append(match.end())
            self.page_numbers.append(int(match.group(1)))

        self.section_positions: List[int] = []
        self.section_prefixes: List[str] = []
        hierarchy: Dict[int, str] = {}
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', markdown, re.MULTILINE):
            level = len(match.group(1))
            # Remove inline markdown from heading titles (bold, italic, backticks)
            title = re.sub(r'[*_`]', '', match.group(2).strip()).strip()
            hierarchy[level] = title
            # Clear any deeper levels — they're no longer in scope
            for deeper in [lvl for lvl in hierarchy if lvl > level]:
                del hierarchy[deeper]
            self.section_positions.append(match.end())
            self.section_prefixes.append(
                "[" + "].[".join(hierarchy[lvl] for lvl in sorted(hierarchy)) + "]"
            )

    def page_at(self, position: int) -> int:
        """Page number in effect at a character offset (1 if none precedes it)."""
        idx = bisect_right(self.page_positions, position)
        return self.page_numbers[idx - 1] if idx else 1

    def section_at(self, position: int) -> str:
        """Heading hierarchy prefix in effect at a character offset."""
        idx = bisect_right(self.section_positions, position)
        return self.section_prefixes[idx - 1] if idx else ""


def _extract_page_content(markdown: str, page_number: int) -> str:
    """
    Extract the full text content of a specific page from markdown
    that contains [Page N] markers produced by PDFToMarkdownConverter.
    """
    page_markers = list(re.finditer(r'\[Page (\d+)\]', markdown))
    if not page_markers:
        return ""

    start_pos = None
    end_pos = None
    for i, marker in enumerate(page_markers):
        if int(marker.group(1)) == page_number:
            start_pos = marker.end()  # content starts right after the marker
            end_pos = page_markers[i + 1].start() if i + 1 < len(page_markers) else len(markdown)
            break

    if start_pos is None:
        return ""

    return markdown[start_pos:end_pos].strip()


class ChunkEmbeddingPipeline:
    """Complete pipeline for chunking documents and storing embeddings."""

    def __init__(self, db_params: Dict[str, str], embedding_model: str,
                 table_name: str):
        """
        Initialize the pipeline.
        Args:
            db_params: Database connection parameters
            embedding_model: SentenceTransformer model name
            table_name: Name of the chunks table
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore(db_params, table_name)

    @staticmethod
    async def parse_file(
        file_path: str,
        document_id: str,
        parse_backend: str = "",
    ) -> Dict[str, Any]:
        """
        Parse a document into raw text/markdown without chunking or embedding.

        Static because parsing needs no embedding model — the ingestion worker
        calls it on the class so the parse stage does not pay for loading one.

        Returns:
            Dict with parsed_text, parser_used, filename, file_type, file_size,
            and page_mapping (for non-PDF files).
        """
        file_path = Path(file_path)
        filename = file_path.name
        file_type = file_path.suffix.lower().replace('.', '')
        file_size = file_path.stat().st_size

        print(f"Parsing document: {filename} (ID: {document_id})")

        if file_path.suffix.lower() == '.pdf':
            from app.ingestion.processors.pdf_parser_factory import create_pdf_parser
            from app.config.app_config import AppSettings

            settings = AppSettings()
            backend = parse_backend or settings.pdf_parser_backend
            parser = create_pdf_parser(backend, settings)
            parsed_text = parser.parse_pdf(str(file_path), output_path=None)
            parser_used = parser.get_backend_name()
            page_mapping: List[Any] = []
        else:
            processor = get_processor_for_file(str(file_path))
            if not processor.validate_file(str(file_path)):
                raise ValueError(f"Invalid file: {file_path}")
            parsed_text, page_mapping = processor.extract_text(str(file_path))
            parser_used = processor.__class__.__name__

        logfire.info("Document parsed",
                     document_id=document_id,
                     parser_used=parser_used,
                     file_type=file_type,
                     file_size=file_size,
                     text_length=len(parsed_text))

        write_parsed_document(document_id, parsed_text, filename=filename)

        return {
            "parsed_text": parsed_text,
            "parser_used": parser_used,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "page_mapping": page_mapping,
        }

    @staticmethod
    def chunk_parsed_document(
        parsed_result: Dict[str, Any],
        chunk_size: int = 512,
        similarity_threshold: float = 0.5,
        chunker_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Chunk parsed text. For PDFs, uses markdown-aware chunking and page
        markers; for DOCX/TXT, uses the standard chunker factory.

        Static because chunking needs no embedding model — see parse_file.

        When `document_id` is given, the chunks are also dumped to
        `data/chunks/<document>/` for inspection. Without an id there is no
        per-document folder to write into, so the dump is skipped.
        """
        parsed_text = parsed_result["parsed_text"]
        file_type = parsed_result["file_type"]
        filename = parsed_result["filename"]

        if file_type == 'pdf':
            from app.ingestion.chunking.chunker_factory import chunk_markdown

            chunks = chunk_markdown(
                parsed_text,
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=50,
                similarity_threshold=similarity_threshold,
            )

            # Scanned once for the whole document, then queried per chunk.
            structure = _MarkdownStructure(parsed_text)

            page_content_cache: Dict[int, str] = {}
            last_section_prefix = ""
            for chunk in chunks:
                if hasattr(chunk, 'start_index') and chunk.start_index is not None:
                    chunk.page_number = structure.page_at(chunk.start_index)

                    section_prefix = structure.section_at(chunk.start_index)
                    if not section_prefix:
                        section_prefix = last_section_prefix
                    else:
                        last_section_prefix = section_prefix

                    chunk.section_path = section_prefix
                    if section_prefix:
                        chunk.text = f"{section_prefix} - {chunk.text}"
                else:
                    chunk.page_number = 1
                    chunk.section_path = last_section_prefix
                    if last_section_prefix:
                        chunk.text = f"{last_section_prefix} - {chunk.text}"

                pg = chunk.page_number
                if pg not in page_content_cache:
                    page_content_cache[pg] = _extract_page_content(parsed_text, pg)
                chunk.full_content = page_content_cache.get(pg, "")
        else:
            from app.ingestion.chunking.chunker_factory import get_chunker

            chunker = get_chunker(
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=50,
                similarity_threshold=similarity_threshold,
                text_length=len(parsed_text),
            )
            chunks = chunker.chunk(parsed_text)
            page_mapping = parsed_result.get("page_mapping", [])
            for chunk in chunks:
                if hasattr(chunk, 'start_index') and page_mapping:
                    chunk.page_number = get_page_number_for_position(chunk.start_index, page_mapping)
                else:
                    chunk.page_number = 1
                chunk.section_path = ""
                chunk.full_content = parsed_text

        print(f"Created {len(chunks)} chunks from {filename}")

        if document_id:
            write_chunk_artifacts(document_id, chunks, filename=filename)

        return chunks

    async def embed_chunks(
        self,
        chunks: List[Any],
        document_id: str,
        chunk_size: int,
        similarity_threshold: float,
        filename: str,
        file_type: str,
        file_size: int,
        parser_used: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Clean, embed, and store chunks in the vector DB.
        """
        valid_chunks = []
        invalid_chunks = 0

        for chunk in chunks:
            text = getattr(chunk, 'text', None)
            if text is None or (isinstance(text, str) and len(text.strip()) == 0):
                invalid_chunks += 1
                logfire.warn("Skipping chunk with None or empty text",
                             chunk_info=str(chunk)[:100])
            else:
                valid_chunks.append(chunk)

        if invalid_chunks > 0:
            logfire.warn("Filtered out invalid chunks during processing",
                         total_chunks=len(chunks),
                         invalid_chunks=invalid_chunks,
                         valid_chunks=len(valid_chunks))

        if not valid_chunks:
            raise ValueError(
                f"No valid chunks created from document {document_id}. "
                f"All {len(chunks)} chunks had None or empty text.")

        text_cleaner = TextCleaningPipeline()
        chunk_texts = []
        for chunk in valid_chunks:
            text = getattr(chunk, 'text', '')
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            elif not isinstance(text, str):
                text = str(text)
            text = text_cleaner.clean(text, log_steps=False)
            chunk.text = text
            chunk_texts.append(text)

        logfire.info("Extracted chunk texts for embedding",
                     num_chunks=len(chunk_texts),
                     sample_types=[type(t).__name__ for t in chunk_texts[:3]])

        logfire.info("Stage: generating embeddings", num_chunks=len(chunk_texts))
        print("Generating embeddings...")
        try:
            # Model inference is CPU/GPU-bound and synchronous — off the event loop.
            embeddings = await asyncio.to_thread(
                self.embedding_generator.embed_batch, chunk_texts
            )
        except Exception as e:
            tb = traceback.format_exc()
            logfire.error("Stage FAILED: embedding generation",
                          error_type=type(e).__name__, error=str(e), traceback=tb)
            print(f"[EMBED ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
            raise

        logfire.info("Stage: embeddings generated", num_embeddings=len(embeddings))
        chunks = valid_chunks

        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            page_number = getattr(chunk, 'page_number', 1)
            raw_page_content = getattr(chunk, 'full_content', '') or ''

            if len(raw_page_content) > 10000:
                logfire.warn("Large page_content in metadata",
                             page_number=page_number,
                             char_count=len(raw_page_content))

            chunk_metadata = {
                'chunk_index': i,
                'token_count': getattr(chunk, 'token_count', None),
                'start_index': getattr(chunk, 'start_index', None),
                'end_index': getattr(chunk, 'end_index', None),
                'page_number': page_number,
                'section_path': getattr(chunk, 'section_path', ''),
                # Stored once. This used to be written twice (as 'page_content' and
                # 'full_content'), so a page split into N chunks kept 2N copies of its
                # own text in JSONB — and both rode along in every API response.
                # Only 'page_content' is ever read (retrieval/search.py).
                'page_content': raw_page_content,
                'chunk_size': chunk_size,
                'similarity_threshold': similarity_threshold,
                'embedding_model': self.embedding_generator.model_name,
                'embedding_dimension': len(embedding),
                'filename': filename,
                'file_type': file_type,
                'file_size': file_size,
                'parser_used': parser_used,
            }

            if metadata:
                chunk_metadata.update(metadata)

            chunk_objects.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_metadata,
            ))

        logfire.info("Stage: inserting chunks into DB",
                     num_chunks=len(chunk_objects),
                     table_name=self.vector_store.table_name)
        print("Inserting chunks into database using pgvector...")
        try:
            await self.vector_store.add_chunks(chunk_objects)
        except Exception as e:
            tb = traceback.format_exc()
            logfire.error("Stage FAILED: DB insertion",
                          num_chunks=len(chunk_objects),
                          table_name=self.vector_store.table_name,
                          error_type=type(e).__name__, error=str(e), traceback=tb)
            print(f"[INSERT ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
            raise

        logfire.info("Stage: DB insertion complete",
                     num_chunks=len(chunk_objects),
                     document_id=document_id)

    async def ingest_document(self, file_path: str, chunk_size: int = 512,
                              similarity_threshold: float = 0.5,
                              document_id: str = None, metadata: Dict = None,
                              chunker_type: str = None,
                              parse_backend: str = "") -> str:
        """
        Backward-compatible wrapper: parse, chunk, embed, and store.

        Args:
            file_path: Path to the document file
            chunk_size: Maximum tokens per chunk
            similarity_threshold: Similarity threshold for chunking
            document_id: Optional document ID (if None, will generate one)
            metadata: Additional metadata for the document

        Returns:
            Document ID
        """
        file_path = Path(file_path)
        filename = file_path.name

        if document_id is None:
            document_id = str(uuid.uuid4())

        print(f"Processing document: {filename} (ID: {document_id})")

        parsed = await self.parse_file(
            file_path=str(file_path),
            document_id=document_id,
            parse_backend=parse_backend,
        )
        chunks = self.chunk_parsed_document(
            parsed,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            chunker_type=chunker_type,
            document_id=document_id,
        )
        await self.embed_chunks(
            chunks=chunks,
            document_id=document_id,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            filename=parsed["filename"],
            file_type=parsed["file_type"],
            file_size=parsed["file_size"],
            parser_used=parsed["parser_used"],
            metadata=metadata,
        )

        print(f"Successfully processed {filename} -> Document ID: {document_id}")
        return document_id

    async def search_documents(self, query: str, limit: int = 5, threshold: float = 0.5,
                                 document_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for relevant document chunks using pgvector.
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of relevant chunks
        """
        query_embedding = await asyncio.to_thread(
            self.embedding_generator.embed_text, query
        )
        return await self.vector_store.search_similar_chunks(
            query_embedding, limit, threshold, document_ids
        )

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.vector_store.delete_document_chunks(document_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return await self.vector_store.get_collection_stats()
