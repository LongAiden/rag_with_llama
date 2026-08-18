"""
Edge-case tests for the domain registry and doc_name feature.

Three categories of edge cases the plan document calls out:
1. Missing doc_name — pre-006 rows, NULL propagation, fallback to filename stem.
2. Empty chunks/embeddings — all chunks filtered, empty lists, zero-length text.
3. No metadata — None metadata, missing attributes, empty dicts.

No database and no model weights: everything external is mocked.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.embedding.chunk import Chunk
from app.models.schemas import RAGSource, RAGResponse, RAGResponseMetadata
from app.retrieval.reranking import RerankedResult
from app.retrieval.search import perform_document_search


_SENTINEL = object()


def _chunk_dict(chunk_id="c1", doc_name="Linear Algebra", similarity=0.9,
                metadata=_SENTINEL, text="some text"):
    if metadata is _SENTINEL:
        metadata = {"page_number": 12, "page_content": "", "section_path": ""}
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document_id": "doc-1",
        "doc_name": doc_name,
        "metadata": metadata,
        "similarity": similarity,
    }


@pytest.fixture
def pipeline():
    vector_store = MagicMock()
    vector_store.search_bm25 = AsyncMock(return_value=[])
    vector_store.get_chunks_by_section = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.embedding_generator = SimpleNamespace(
        model_name="all-MiniLM-L6-v2", embedding_dim=384
    )
    pipe.vector_store = vector_store
    pipe.search_documents = AsyncMock(return_value=[
        _chunk_dict("c1", "Linear Algebra", 0.91),
        _chunk_dict("c2", "Calculus", 0.72),
    ])
    return pipe


@pytest.fixture
def config():
    return SimpleNamespace(
        connection_string="postgresql://u:p@localhost:5432/db",
        reranker=None,
        pipeline=None,
        settings=SimpleNamespace(
            vector_search_limit=20,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_max_length=256,
            rerank_top_k=5,
        ),
    )


@pytest.fixture
def llm_response():
    from app.models.schemas import SimpleRAGResponse
    return SimpleRAGResponse(
        answer="A vector is an element of a vector space.",
        confidence=0.9,
        word_count=8,
        sources_used=2,
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        metadata={"method": "gemini", "model": "gemini-2.5-flash"},
    )


@pytest.fixture(autouse=True)
def no_telemetry():
    with patch("app.retrieval.search.log_interaction",
               new=AsyncMock(return_value=None)):
        yield


class TestMissingDocName:
    """doc_name is optional and additive — NULL must not break any path."""

    def test_chunk_with_none_doc_name(self):
        chunk = Chunk(id="c1", document_id="d1", text="t", embedding=[0.1],
                      doc_name=None)
        assert chunk.doc_name is None

    def test_chunk_omitting_doc_name_entirely(self):
        """Call sites that predate the column construct without it."""
        chunk = Chunk(id="c1", document_id="d1", text="t", embedding=[0.1])
        assert chunk.doc_name is None

    def test_rag_source_with_none_doc_name(self):
        source = RAGSource(chunk_id="c1", text="t", similarity=0.9,
                           document_id="d1", doc_name=None)
        assert source.doc_name is None

    def test_rag_source_omitting_doc_name(self):
        source = RAGSource(chunk_id="c1", text="t", similarity=0.9,
                           document_id="d1")
        assert source.doc_name is None

    @pytest.mark.asyncio
    async def test_search_returns_none_doc_name_when_db_has_null(
        self, pipeline, config, llm_response
    ):
        """Pre-006 chunk rows have doc_name IS NULL; the SELECT returns None."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", doc_name=None),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].doc_name is None

    @pytest.mark.asyncio
    async def test_context_omits_label_when_doc_name_is_none(
        self, pipeline, config, llm_response
    ):
        """Without a name the source label must not contain a dash separator."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", doc_name=None),
        ])
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        context = mock_llm.call_args[0][1]
        assert "Source 1 —" not in context
        assert "[Source 1 (Page 12)]" in context

    @pytest.mark.asyncio
    async def test_mixed_null_and_present_doc_names(
        self, pipeline, config, llm_response
    ):
        """A table with both pre-006 and post-006 chunks returns each correctly."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", doc_name="Linear Algebra", similarity=0.91),
            _chunk_dict("c2", doc_name=None, similarity=0.80),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].doc_name == "Linear Algebra"
        assert result.sources[1].doc_name is None

    @pytest.mark.asyncio
    async def test_rerank_preserves_none_doc_name(self, pipeline, config, llm_response):
        """Reranking a NULL doc_name must not fabricate one."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", doc_name=None, similarity=0.91),
        ])

        def _rerank(query, results, top_k=None):
            return [
                RerankedResult(
                    chunk_id=r["chunk_id"], text=r["text"],
                    document_id=r["document_id"], metadata=r["metadata"],
                    similarity=r["similarity"], rerank_score=9.0,
                    original_rank=0, new_rank=0,
                )
                for r in results[:top_k or len(results)]
            ]

        fake_reranker = SimpleNamespace(rerank=_rerank)
        with patch("app.retrieval.utils.get_reranker", return_value=fake_reranker), \
             patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config,
                enable_reranking=True, rerank_top_k=5,
            )

        assert result.sources[0].doc_name is None

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_pipeline")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_worker_falls_back_to_filename_stem_when_doc_name_is_null(
        self, mock_get_config, mock_get_repo, mock_get_pipeline
    ):
        """Pre-006 documents have no doc_name column; the stem fills the gap."""
        mock_config = MagicMock()
        mock_config.connection_string = "postgresql://u:p@localhost/db"
        mock_get_config.return_value = mock_config

        mock_repo = AsyncMock()
        mock_repo.claim_document = AsyncMock(return_value={
            "id": "doc-1",
            "file_name": "Building ML Systems.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
            "file_type": "pdf",
            "file_size": 1024,
            "content_type": "application/pdf",
            "metadata": {},
            "doc_name": None,
            "domain": None,
        })
        mock_repo.get_chunked = AsyncMock(return_value={
            "chunks": [{"text": "chunk 1", "page_number": 1}],
            "metadata": {"parser_used": "gemini-docling", "file_type": "pdf"},
        })
        mock_repo.transition_to_embedded = AsyncMock()
        mock_get_repo.return_value = mock_repo

        mock_pipeline = AsyncMock()
        mock_pipeline.embed_chunks = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline

        from app.worker.ingestion_tasks import _embed_document
        await _embed_document("doc-1")

        kwargs = mock_pipeline.embed_chunks.call_args.kwargs
        assert kwargs["doc_name"] == "Building ML Systems"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_pipeline")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_worker_uses_doc_name_when_present(
        self, mock_get_config, mock_get_repo, mock_get_pipeline
    ):
        """Post-006 documents carry doc_name; the filename stem is not used."""
        mock_config = MagicMock()
        mock_config.connection_string = "postgresql://u:p@localhost/db"
        mock_get_config.return_value = mock_config

        mock_repo = AsyncMock()
        mock_repo.claim_document = AsyncMock(return_value={
            "id": "doc-1",
            "file_name": "Linear_Algebra_v3.pdf",
            "target_table_name": "math",
            "chunk_size": 512,
            "file_type": "pdf",
            "file_size": 2048,
            "content_type": "application/pdf",
            "metadata": {},
            "doc_name": "Linear Algebra",
            "domain": "math",
        })
        mock_repo.get_chunked = AsyncMock(return_value={
            "chunks": [{"text": "chunk 1", "page_number": 1}],
            "metadata": {"parser_used": "gemini-docling", "file_type": "pdf"},
        })
        mock_repo.transition_to_embedded = AsyncMock()
        mock_get_repo.return_value = mock_repo

        mock_pipeline = AsyncMock()
        mock_pipeline.embed_chunks = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline

        from app.worker.ingestion_tasks import _embed_document
        await _embed_document("doc-1")

        kwargs = mock_pipeline.embed_chunks.call_args.kwargs
        assert kwargs["doc_name"] == "Linear Algebra"
        assert kwargs["metadata"]["domain"] == "math"

    def test_register_document_defaults_doc_name_to_filename_stem(self):
        """When doc_name is None, register_document uses Path(file_name).stem."""
        from pathlib import Path
        file_name = "My Complex File Name v2.pdf"
        expected = Path(file_name).stem
        assert expected == "My Complex File Name v2"


class TestEmptyChunksAndEmbeddings:
    """Empty or missing chunk data must be caught before hitting the DB."""

    @pytest.mark.asyncio
    async def test_embed_chunks_raises_when_all_chunks_have_empty_text(self):
        """A document that produces only whitespace chunks must fail loudly."""
        from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline

        pipeline = MagicMock(spec=ChunkEmbeddingPipeline)
        pipeline.embedding_generator = SimpleNamespace(
            model_name="test", embedding_dim=384
        )
        pipeline.vector_store = AsyncMock()

        empty_chunks = [
            SimpleNamespace(text=""),
            SimpleNamespace(text="   "),
            SimpleNamespace(text=None),
        ]

        with pytest.raises(ValueError, match="No valid chunks"):
            await ChunkEmbeddingPipeline.embed_chunks(
                pipeline,
                chunks=empty_chunks,
                document_id="doc-1",
                chunk_size=512,
                similarity_threshold=0.5,
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
                parser_used="test",
            )

    @pytest.mark.asyncio
    async def test_embed_chunks_filters_empty_text_chunks(self):
        """Valid chunks pass through; empty ones are skipped silently."""
        from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline

        pipeline = MagicMock(spec=ChunkEmbeddingPipeline)
        pipeline.embedding_generator = MagicMock()
        pipeline.embedding_generator.model_name = "test"
        pipeline.embedding_generator.embedding_dim = 384
        pipeline.embedding_generator.embed_batch = MagicMock(return_value=[[0.1] * 384])
        pipeline.vector_store = AsyncMock()

        chunks = [
            SimpleNamespace(text="valid text", page_number=1, full_content="",
                            token_count=10, start_index=0, end_index=10,
                            section_path=""),
            SimpleNamespace(text="", page_number=2, full_content="",
                            token_count=0, start_index=0, end_index=0,
                            section_path=""),
            SimpleNamespace(text=None, page_number=3, full_content="",
                            token_count=0, start_index=0, end_index=0,
                            section_path=""),
        ]

        with patch("app.ingestion.embedding.pipeline.TextCleaningPipeline") as mock_cleaner, \
             patch("app.ingestion.embedding.pipeline.asyncio.to_thread",
                   new=AsyncMock(return_value=[[0.1] * 384])), \
             patch("app.ingestion.embedding.pipeline.write_chunk_artifacts"):
            mock_cleaner.return_value.clean.return_value = "valid text"

            await ChunkEmbeddingPipeline.embed_chunks(
                pipeline,
                chunks=chunks,
                document_id="doc-1",
                chunk_size=512,
                similarity_threshold=0.5,
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
                parser_used="test",
                doc_name="Test Doc",
            )

        inserted = pipeline.vector_store.add_chunks.call_args[0][0]
        assert len(inserted) == 1
        assert inserted[0].text == "valid text"
        assert inserted[0].doc_name == "Test Doc"

    @pytest.mark.asyncio
    async def test_embed_chunks_raises_on_empty_chunk_list(self):
        """An empty chunk list is the same failure mode as all-invalid."""
        from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline

        pipeline = MagicMock(spec=ChunkEmbeddingPipeline)

        with pytest.raises(ValueError, match="No valid chunks"):
            await ChunkEmbeddingPipeline.embed_chunks(
                pipeline,
                chunks=[],
                document_id="doc-1",
                chunk_size=512,
                similarity_threshold=0.5,
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
                parser_used="test",
            )

    @pytest.mark.asyncio
    async def test_add_chunks_with_empty_list_is_noop(self):
        """add_chunks([]) should not call the DB at all."""
        from app.ingestion.embedding.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        vs._initialized = True
        vs.table_name = "test_table"
        vs.safe_table_name = '"test_table"'

        mock_conn = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        vs.connection = MagicMock(return_value=cm)

        await VectorStore.add_chunks(vs, [])

        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_chunks_stores_none_doc_name_as_null(self):
        """A chunk with doc_name=None must insert NULL, not a string."""
        from app.ingestion.embedding.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        vs._initialized = True
        vs.table_name = "test_table"
        vs.safe_table_name = '"test_table"'

        mock_conn = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        vs.connection = MagicMock(return_value=cm)

        chunks = [Chunk(id="c1", document_id="d1", text="t",
                        embedding=[0.1] * 384, doc_name=None)]
        await VectorStore.add_chunks(vs, chunks)

        batch = mock_conn.executemany.call_args[0][1]
        assert batch[0][5] is None

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_chunks_match(
        self, pipeline, config, llm_response
    ):
        """Zero results is a valid outcome, not an error."""
        pipeline.search_documents = AsyncMock(return_value=[])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="obscure topic", limit=5, threshold=0.99,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources == []
        assert "No relevant documents found" in result.answer

    @pytest.mark.asyncio
    async def test_hybrid_mode_empty_bm25_results(
        self, pipeline, config, llm_response
    ):
        """BM25 returning nothing while vector has results must not crash RRF."""
        pipeline.vector_store.search_bm25 = AsyncMock(return_value=[])
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", 0.91),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
                search_mode="hybrid",
            )

        assert len(result.sources) == 1
        assert result.sources[0].doc_name == "Linear Algebra"


class TestNoMetadata:
    """Chunks with None or empty metadata must not crash context building."""

    @pytest.mark.asyncio
    async def test_search_with_none_metadata(
        self, pipeline, config, llm_response
    ):
        """A chunk row whose metadata column is NULL must not crash the context."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", metadata=None),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].page_number is None
        assert result.sources[0].metadata == {}

    @pytest.mark.asyncio
    async def test_context_with_none_metadata_omits_page_number(
        self, pipeline, config, llm_response
    ):
        """No page_number in metadata means no '(Page N)' in the context label."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", metadata=None),
        ])
        mock_llm = AsyncMock(return_value=llm_response)
        with patch("app.retrieval.search.generate_llm_response", new=mock_llm):
            await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        context = mock_llm.call_args[0][1]
        assert "(Page" not in context
        assert "Source 1 — Linear Algebra" in context

    @pytest.mark.asyncio
    async def test_search_with_empty_metadata_dict(
        self, pipeline, config, llm_response
    ):
        """An empty dict is different from None — both must work."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", metadata={}),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].page_number is None
        assert result.sources[0].metadata == {}

    @pytest.mark.asyncio
    async def test_metadata_missing_page_number_but_has_section(
        self, pipeline, config, llm_response
    ):
        """page_number is optional; section_path can still be present."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra",
                        metadata={"section_path": "[Chapter 1]", "page_content": ""}),
        ])
        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].page_number is None

    @pytest.mark.asyncio
    async def test_rerank_with_none_metadata(self, pipeline, config, llm_response):
        """Reranking must not crash when metadata is None in the original result."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", metadata=None, similarity=0.91),
        ])

        def _rerank(query, results, top_k=None):
            return [
                RerankedResult(
                    chunk_id=r["chunk_id"], text=r["text"],
                    document_id=r["document_id"],
                    metadata=r.get("metadata"),
                    similarity=r["similarity"], rerank_score=9.0,
                    original_rank=0, new_rank=0,
                )
                for r in results[:top_k or len(results)]
            ]

        fake_reranker = SimpleNamespace(rerank=_rerank)
        with patch("app.retrieval.utils.get_reranker", return_value=fake_reranker), \
             patch("app.retrieval.search.generate_llm_response",
                   new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="what is a vector", limit=5, threshold=0.3,
                pipeline=pipeline, config=config,
                enable_reranking=True, rerank_top_k=5,
            )

        assert result.sources[0].doc_name == "Linear Algebra"

    @pytest.mark.asyncio
    async def test_add_chunks_stores_empty_dict_when_metadata_is_none(self):
        """VectorStore.add_chunks converts None metadata to {} for JSONB."""
        from app.ingestion.embedding.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        vs._initialized = True
        vs.table_name = "test_table"
        vs.safe_table_name = '"test_table"'

        mock_conn = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        vs.connection = MagicMock(return_value=cm)

        chunks = [Chunk(id="c1", document_id="d1", text="t",
                        embedding=[0.1] * 384, metadata=None)]
        await VectorStore.add_chunks(vs, chunks)

        batch = mock_conn.executemany.call_args[0][1]
        assert batch[0][4] == {}

    @pytest.mark.asyncio
    async def test_embed_chunks_writes_doc_name_into_metadata(self):
        """doc_name is denormalized into chunk metadata for index.json dumps."""
        from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline

        pipeline = MagicMock(spec=ChunkEmbeddingPipeline)
        pipeline.embedding_generator = MagicMock()
        pipeline.embedding_generator.model_name = "test"
        pipeline.embedding_generator.embedding_dim = 384
        pipeline.embedding_generator.embed_batch = MagicMock(return_value=[[0.1] * 384])
        pipeline.vector_store = AsyncMock()

        chunks = [
            SimpleNamespace(text="valid text", page_number=1, full_content="",
                            token_count=10, start_index=0, end_index=10,
                            section_path=""),
        ]

        with patch("app.ingestion.embedding.pipeline.TextCleaningPipeline") as mock_cleaner, \
             patch("app.ingestion.embedding.pipeline.asyncio.to_thread",
                   new=AsyncMock(return_value=[[0.1] * 384])), \
             patch("app.ingestion.embedding.pipeline.write_chunk_artifacts"):
            mock_cleaner.return_value.clean.return_value = "valid text"

            await ChunkEmbeddingPipeline.embed_chunks(
                pipeline,
                chunks=chunks,
                document_id="doc-1",
                chunk_size=512,
                similarity_threshold=0.5,
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
                parser_used="test",
                doc_name="My Document",
            )

        inserted = pipeline.vector_store.add_chunks.call_args[0][0]
        assert inserted[0].metadata["doc_name"] == "My Document"
        assert inserted[0].doc_name == "My Document"

    @pytest.mark.asyncio
    async def test_embed_chunks_none_doc_name_written_to_metadata(self):
        """When doc_name is None, metadata still carries it (as None)."""
        from app.ingestion.embedding.pipeline import ChunkEmbeddingPipeline

        pipeline = MagicMock(spec=ChunkEmbeddingPipeline)
        pipeline.embedding_generator = MagicMock()
        pipeline.embedding_generator.model_name = "test"
        pipeline.embedding_generator.embedding_dim = 384
        pipeline.embedding_generator.embed_batch = MagicMock(return_value=[[0.1] * 384])
        pipeline.vector_store = AsyncMock()

        chunks = [
            SimpleNamespace(text="valid text", page_number=1, full_content="",
                            token_count=10, start_index=0, end_index=10,
                            section_path=""),
        ]

        with patch("app.ingestion.embedding.pipeline.TextCleaningPipeline") as mock_cleaner, \
             patch("app.ingestion.embedding.pipeline.asyncio.to_thread",
                   new=AsyncMock(return_value=[[0.1] * 384])), \
             patch("app.ingestion.embedding.pipeline.write_chunk_artifacts"):
            mock_cleaner.return_value.clean.return_value = "valid text"

            await ChunkEmbeddingPipeline.embed_chunks(
                pipeline,
                chunks=chunks,
                document_id="doc-1",
                chunk_size=512,
                similarity_threshold=0.5,
                filename="test.pdf",
                file_type="pdf",
                file_size=1024,
                parser_used="test",
                doc_name=None,
            )

        inserted = pipeline.vector_store.add_chunks.call_args[0][0]
        assert inserted[0].metadata["doc_name"] is None
        assert inserted[0].doc_name is None

    @pytest.mark.asyncio
    async def test_sibling_expansion_with_none_metadata(
        self, pipeline, config, llm_response
    ):
        """Structural queries expand siblings; None metadata must not crash it."""
        pipeline.search_documents = AsyncMock(return_value=[
            _chunk_dict("c1", "Linear Algebra", metadata=None, similarity=0.91),
        ])
        pipeline.vector_store.get_chunks_by_section = AsyncMock(return_value=[
            {"text": "sibling text", "doc_name": "Linear Algebra"},
        ])

        with patch("app.retrieval.search.generate_llm_response",
                    new=AsyncMock(return_value=llm_response)):
            result = await perform_document_search(
                query="how many vectors are there", limit=5, threshold=0.3,
                pipeline=pipeline, config=config, enable_reranking=False,
            )

        assert result.sources[0].doc_name == "Linear Algebra"
