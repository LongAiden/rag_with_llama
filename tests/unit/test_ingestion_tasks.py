"""
Unit tests for the ingestion Celery tasks.

Tests the stage-based parse → chunk → embed pipeline tasks.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = MagicMock()
    config.db_params = {
        "host": "localhost",
        "port": "5432",
        "dbname": "rag_db",
        "user": "admin",
        "password": "admin",
    }
    config.connection_string = "postgresql://admin:admin@localhost/rag_db"
    return config


@pytest.fixture
def mock_repo():
    """Create a mock IngestionRepository."""
    return AsyncMock()


@pytest.fixture
def parsed_result():
    return {
        "parsed_text": "This is parsed text from a document.",
        "parser_used": "gemini-docling",
        "filename": "test.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "page_mapping": [{"page": 1, "start": 0}],
    }


@pytest.fixture
def chunk_objects():
    return [
        SimpleNamespace(
            text="chunk 1",
            token_count=100,
            start_index=0,
            end_index=50,
            page_number=1,
            section_path="[Chapter 1]",
            full_content="full page content",
        ),
        SimpleNamespace(
            text="chunk 2",
            token_count=100,
            start_index=50,
            end_index=100,
            page_number=2,
            section_path="[Chapter 2]",
            full_content="full page content",
        ),
    ]


@pytest.fixture
def mock_pipeline_cls(parsed_result, chunk_objects):
    """Patch ChunkEmbeddingPipeline where parse/chunk call it — on the class.

    parse_file and chunk_parsed_document are static: the parse and chunk stages
    must not construct a pipeline, because that loads a SentenceTransformer model
    they never use.
    """
    cls = MagicMock()
    cls.parse_file = AsyncMock(return_value=parsed_result)
    cls.chunk_parsed_document = MagicMock(return_value=chunk_objects)
    with patch("app.ingestion.embedding.pipeline.ChunkEmbeddingPipeline", cls):
        yield cls


@pytest.fixture
def mock_pipeline():
    """Create a mock embedding pipeline (embed stage only)."""
    pipeline = AsyncMock()
    pipeline.embed_chunks = AsyncMock()
    return pipeline


class TestParseDocumentTask:
    """Tests for parse_document_task."""

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_parse_document_success(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """Test successful document parsing."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "raw_storage_path": "/tmp/test.pdf",
            "target_table_name": "document_chunks",
            "parse_backend": "ollama",
        })
        mock_repo.transition_to_parsed = AsyncMock()

        from app.worker.ingestion_tasks import _parse_document
        result = await _parse_document("test-doc-id")

        assert result["status"] == "parsed"
        assert result["document_id"] == "test-doc-id"
        mock_pipeline_cls.parse_file.assert_called_once()
        mock_repo.transition_to_parsed.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_parse_claims_only_its_own_document(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """The claim is scoped to the dispatched id and its expected stage.

        Regression guard: claiming "the next registered row" moved an unrelated
        document into 'parsing' and abandoned it there for the claim timeout.
        """
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.claim_document = AsyncMock(return_value=None)

        from app.worker.ingestion_tasks import _parse_document
        await _parse_document("test-doc-id")

        kwargs = mock_repo.claim_document.call_args.kwargs
        assert kwargs["doc_id"] == "test-doc-id"
        assert kwargs["current_stage"] == "registered"
        assert kwargs["processing_stage"] == "parsing"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_parse_document_claim_fails(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo
    ):
        """Test parse task returns skipped when the document is not claimable."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.claim_document = AsyncMock(return_value=None)

        from app.worker.ingestion_tasks import _parse_document
        result = await _parse_document("test-doc-id")

        assert result["status"] == "skipped"
        assert result["stage"] == "parse"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_parse_document_error_recorded(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """Test parse errors are recorded in the status DB against the parse stage."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "raw_storage_path": "/tmp/test.pdf",
            "target_table_name": "document_chunks",
        })
        mock_pipeline_cls.parse_file = AsyncMock(side_effect=Exception("Parse failed"))
        mock_repo.record_error = AsyncMock()

        from app.worker.ingestion_tasks import _parse_document
        with pytest.raises(Exception, match="Parse failed"):
            await _parse_document("test-doc-id")

        mock_repo.record_error.assert_called_once()
        assert mock_repo.record_error.call_args.kwargs["stage"] == "parse"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_parse_persists_file_type_and_page_mapping(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """Parse must hand file_type and page_mapping forward to the chunk stage."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "raw_storage_path": "/tmp/test.pdf",
            "target_table_name": "document_chunks",
        })
        mock_repo.transition_to_parsed = AsyncMock()

        from app.worker.ingestion_tasks import _parse_document
        await _parse_document("test-doc-id")

        kwargs = mock_repo.transition_to_parsed.call_args.kwargs
        assert kwargs["file_type"] == "pdf"
        assert kwargs["metadata"]["page_mapping"] == [{"page": 1, "start": 0}]


class TestChunkDocumentTask:
    """Tests for chunk_document_task."""

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_chunk_document_success(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """Test successful document chunking."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
            "file_type": "pdf",
        })
        mock_repo.get_parsed = AsyncMock(return_value={
            "parsed_text": "This is parsed text.",
            "parser_used": "gemini-docling",
            "metadata": {"file_type": "pdf", "page_mapping": []},
        })
        mock_repo.transition_to_chunked = AsyncMock()

        from app.worker.ingestion_tasks import _chunk_document
        result = await _chunk_document("test-doc-id")

        assert result["status"] == "chunked"
        assert result["chunk_count"] == 2
        mock_pipeline_cls.chunk_parsed_document.assert_called_once()
        mock_repo.transition_to_chunked.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_chunk_passes_real_file_type(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """A PDF must reach the chunker as 'pdf'.

        Regression guard: file_type was read from a column that did not exist, so
        it was always "", and every PDF silently fell through to the generic
        chunker — losing page numbers, section paths and full_content.
        """
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
            "file_type": "pdf",
        })
        mock_repo.get_parsed = AsyncMock(return_value={
            "parsed_text": "# Heading\n[Page 1] text",
            "parser_used": "gemini-docling",
            "metadata": {"file_type": "pdf", "page_mapping": []},
        })
        mock_repo.transition_to_chunked = AsyncMock()

        from app.worker.ingestion_tasks import _chunk_document
        await _chunk_document("test-doc-id")

        parsed_arg = mock_pipeline_cls.chunk_parsed_document.call_args[0][0]
        assert parsed_arg["file_type"] == "pdf"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_chunk_falls_back_to_parsed_metadata_file_type(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """When the column is NULL, file_type comes from the parsed artifact."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": None,
            "file_type": None,
        })
        mock_repo.get_parsed = AsyncMock(return_value={
            "parsed_text": "text",
            "parser_used": "gemini-docling",
            "metadata": {"file_type": "pdf"},
        })
        mock_repo.transition_to_chunked = AsyncMock()

        from app.worker.ingestion_tasks import _chunk_document
        await _chunk_document("test-doc-id")

        parsed_arg = mock_pipeline_cls.chunk_parsed_document.call_args[0][0]
        assert parsed_arg["file_type"] == "pdf"
        # A NULL chunk_size column must not propagate as None.
        assert mock_pipeline_cls.chunk_parsed_document.call_args.kwargs["chunk_size"] == 512

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_chunk_document_missing_parsed_artifact(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo, mock_pipeline_cls
    ):
        """Test chunk task fails when parsed artifact is missing."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
        })
        mock_repo.get_parsed = AsyncMock(return_value=None)
        mock_repo.record_error = AsyncMock()

        from app.worker.ingestion_tasks import _chunk_document
        with pytest.raises(ValueError, match="Parsed artifact missing"):
            await _chunk_document("test-doc-id")

        assert mock_repo.record_error.call_args.kwargs["stage"] == "chunk"


class TestEmbedDocumentTask:
    """Tests for embed_document_task."""

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_pipeline")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_embed_document_success(
        self, mock_get_config, mock_get_repo, mock_get_pipeline, mock_config, mock_repo, mock_pipeline
    ):
        """Test successful document embedding."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_get_pipeline.return_value = mock_pipeline

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
            "file_type": "pdf",
            "file_size": 1024,
            "content_type": "application/pdf",
            "metadata": {},
        })
        mock_repo.get_chunked = AsyncMock(return_value={
            "chunks": [
                {"text": "chunk 1", "page_number": 1},
                {"text": "chunk 2", "page_number": 2},
            ],
            "metadata": {"parser_used": "gemini-docling", "file_type": "pdf"},
        })
        mock_repo.transition_to_embedded = AsyncMock()

        from app.worker.ingestion_tasks import _embed_document
        result = await _embed_document("test-doc-id")

        assert result["status"] == "embedded"
        assert result["chunk_count"] == 2
        mock_pipeline.embed_chunks.assert_called_once()
        mock_repo.transition_to_embedded.assert_called_once()

        kwargs = mock_pipeline.embed_chunks.call_args.kwargs
        assert kwargs["file_type"] == "pdf"
        assert kwargs["parser_used"] == "gemini-docling"

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_pipeline")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_embed_reads_chunks_as_objects(
        self, mock_get_config, mock_get_repo, mock_get_pipeline, mock_config, mock_repo, mock_pipeline
    ):
        """Chunks come back from JSONB as dicts and are rehydrated for the pipeline.

        This only works because the connection pool registers a jsonb codec; without
        it the artifact would decode as a raw string and unpacking would fail.
        """
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_get_pipeline.return_value = mock_pipeline

        mock_repo.claim_document = AsyncMock(return_value={
            "id": "test-doc-id",
            "file_name": "test.pdf",
            "target_table_name": "document_chunks",
            "chunk_size": 512,
            "file_type": "pdf",
            "file_size": 1024,
            "content_type": "application/pdf",
            "metadata": {"source": "upload"},
        })
        mock_repo.get_chunked = AsyncMock(return_value={
            "chunks": [{"text": "chunk 1", "page_number": 3, "section_path": "[A]"}],
            "metadata": {"parser_used": "gemini-docling"},
        })
        mock_repo.transition_to_embedded = AsyncMock()

        from app.worker.ingestion_tasks import _embed_document
        await _embed_document("test-doc-id")

        chunks = mock_pipeline.embed_chunks.call_args.kwargs["chunks"]
        assert chunks[0].text == "chunk 1"
        assert chunks[0].page_number == 3
        assert chunks[0].section_path == "[A]"


class TestBuildIngestionChain:
    """Tests for chain construction."""

    def test_chain_uses_immutable_signatures(self):
        """Every stage must receive only doc_id.

        Regression guard: a chain of mutable (.s) signatures prepends the previous
        task's return value, so chunk_document_task(result_dict, doc_id) raised a
        TypeError and no upload ever got past parse.
        """
        from app.worker.ingestion_tasks import build_ingestion_chain

        task_chain = build_ingestion_chain("doc-1", from_stage="registered")

        assert len(task_chain.tasks) == 3
        for signature in task_chain.tasks:
            assert signature.immutable is True
            assert signature.args == ("doc-1",)

    def test_chain_resumes_from_stage(self):
        """A partially processed document only re-runs the stages it still needs."""
        from app.worker.ingestion_tasks import build_ingestion_chain

        assert len(build_ingestion_chain("d", from_stage="registered").tasks) == 3
        assert len(build_ingestion_chain("d", from_stage="parsed").tasks) == 2
        assert len(build_ingestion_chain("d", from_stage="chunked").tasks) == 1

    def test_chain_is_none_for_terminal_stages(self):
        """Documents with no work left are not dispatched."""
        from app.worker.ingestion_tasks import build_ingestion_chain

        assert build_ingestion_chain("d", from_stage="embedded") is None
        assert build_ingestion_chain("d", from_stage="failed") is None

    def test_chain_sets_queue_on_every_task(self):
        """Queue is set per signature, not left to chain option propagation."""
        from app.worker.ingestion_tasks import build_ingestion_chain

        task_chain = build_ingestion_chain("doc-1", from_stage="registered", queue="upload")

        for signature in task_chain.tasks:
            assert signature.options["queue"] == "upload"


class TestRegisterAndDispatch:
    """Tests for register_and_dispatch_task (weekly scan)."""

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks.build_ingestion_chain")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    @patch("app.worker.ingestion_tasks.INPUT_RAW_DIR")
    async def test_register_and_dispatch_scans_files(
        self, mock_input_dir, mock_get_config, mock_get_repo, mock_build_chain,
        mock_config, mock_repo, tmp_path
    ):
        """Test the scan registers new files and dispatches chains."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.reset_stale_claims = AsyncMock(return_value=1)
        mock_repo.reset_error_documents = AsyncMock(return_value=2)
        mock_repo.is_path_registered = AsyncMock(return_value=False)
        mock_repo.is_file_registered = AsyncMock(return_value=False)
        mock_repo.register_document = AsyncMock()
        mock_repo.get_pending_doc_ids = AsyncMock(return_value=["doc-1"])
        mock_repo.get_document_status = AsyncMock(return_value={"stage": "registered"})

        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        mock_input_dir.iterdir = MagicMock(return_value=[test_file])
        mock_input_dir.mkdir = MagicMock()

        mock_task_chain = MagicMock()
        mock_build_chain.return_value = mock_task_chain

        from app.worker.ingestion_tasks import _register_and_dispatch
        result = await _register_and_dispatch()

        assert result["status"] == "ok"
        assert result["stale_reset"] == 1
        assert result["retried"] == 2
        assert result["registered"] == 1
        assert result["dispatched"] == 1
        mock_task_chain.apply_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    @patch("app.worker.ingestion_tasks.INPUT_RAW_DIR")
    async def test_scan_skips_files_already_registered_by_path(
        self, mock_input_dir, mock_get_config, mock_get_repo, mock_config, mock_repo, tmp_path
    ):
        """Uploaded files must not be re-registered by the scan.

        Regression guard: uploads land as '<uuid>_<name>' but register under
        '<name>', so a filename-only check registered a duplicate document every
        run and duplicated its chunks in the vector store.
        """
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo

        mock_repo.reset_stale_claims = AsyncMock(return_value=0)
        mock_repo.reset_error_documents = AsyncMock(return_value=0)
        mock_repo.is_path_registered = AsyncMock(return_value=True)
        mock_repo.is_file_registered = AsyncMock(return_value=False)
        mock_repo.register_document = AsyncMock()
        mock_repo.get_pending_doc_ids = AsyncMock(return_value=[])

        uploaded = tmp_path / "6f1e_report.pdf"
        uploaded.write_text("already ingested")
        mock_input_dir.iterdir = MagicMock(return_value=[uploaded])
        mock_input_dir.mkdir = MagicMock()

        from app.worker.ingestion_tasks import _register_and_dispatch
        result = await _register_and_dispatch()

        assert result["registered"] == 0
        mock_repo.register_document.assert_not_called()


class TestRecoverAndDispatch:
    """Tests for recover_and_dispatch_task."""

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks.build_ingestion_chain")
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_recovery_redispatches_reset_documents(
        self, mock_get_config, mock_get_repo, mock_build_chain, mock_config, mock_repo
    ):
        """Resetting a stale claim must also re-queue the work.

        Regression guard: the sweep reset stages but dispatched nothing, so a
        released document sat idle until the next weekly scan.
        """
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.reset_stale_claims = AsyncMock(return_value=5)
        mock_repo.reset_error_documents = AsyncMock(return_value=1)
        mock_repo.get_pending_doc_ids = AsyncMock(return_value=["doc-1", "doc-2"])
        mock_repo.get_document_status = AsyncMock(return_value={"stage": "parsed"})
        mock_build_chain.return_value = MagicMock()

        from app.worker.ingestion_tasks import _recover_and_dispatch
        result = await _recover_and_dispatch()

        assert result["status"] == "ok"
        assert result["stale_reset"] == 5
        assert result["retried"] == 1
        assert result["dispatched"] == 2

    @pytest.mark.asyncio
    @patch("app.worker.ingestion_tasks._get_repo")
    @patch("app.worker.ingestion_tasks._get_config")
    async def test_recovery_does_not_scan_input_dir(
        self, mock_get_config, mock_get_repo, mock_config, mock_repo
    ):
        """The frequent recovery pass must not do the full directory scan."""
        mock_get_config.return_value = mock_config
        mock_get_repo.return_value = mock_repo
        mock_repo.reset_stale_claims = AsyncMock(return_value=0)
        mock_repo.reset_error_documents = AsyncMock(return_value=0)
        mock_repo.get_pending_doc_ids = AsyncMock(return_value=[])

        from app.worker.ingestion_tasks import _recover_and_dispatch
        result = await _recover_and_dispatch()

        assert "registered" not in result
        mock_repo.register_document.assert_not_called()


class TestEventLoopReuse:
    """Tests for the worker's persistent event loop."""

    def test_run_reuses_one_loop_across_calls(self):
        """Tasks must share a loop.

        Regression guard: asyncio.run() per task closed the loop while the
        process-wide asyncpg pool still held connections bound to it, so the
        second task in a worker process failed with "Event loop is closed".
        """
        import app.worker.ingestion_tasks as tasks

        async def loop_id():
            import asyncio
            return id(asyncio.get_running_loop())

        first = tasks._run(loop_id())
        second = tasks._run(loop_id())

        assert first == second
        assert not tasks._LOOP.is_closed()
