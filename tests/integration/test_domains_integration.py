"""
Integration tests for the domain registry and `doc_name`.

Needs a running Postgres with migration 006 applied:

    docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \\
      < deploy/migrations/006_domains_and_doc_name.sql

Every test works in its own throwaway domain (and therefore its own chunk table),
so they neither see nor disturb real data.
"""
import uuid

import asyncpg
import pytest

from app.infra.db.domain_repository import DomainRepository
from app.infra.db.ingestion_repository import IngestionRepository


def _connection_string(db_params) -> str:
    return (
        f"postgresql://{db_params['user']}:{db_params['password']}"
        f"@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    )


@pytest.fixture
async def schema_ready(db_connection):
    """Skip the whole module if migration 006 has not been applied."""
    has_domains = await db_connection.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'domains')"
    )
    has_doc_name = await db_connection.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'documents' AND column_name = 'doc_name')"
    )
    if not (has_domains and has_doc_name):
        pytest.skip("migration 006 not applied to this database")


@pytest.fixture
async def domain_name(db_connection, schema_ready):
    """A unique domain, dropped along with its chunk table afterwards."""
    name = f"test_domain_{uuid.uuid4().hex[:8]}"
    yield name
    try:
        await db_connection.execute(f'DROP TABLE IF EXISTS "{name}" CASCADE')
        await db_connection.execute("DELETE FROM documents WHERE domain = $1", name)
        await db_connection.execute("DELETE FROM domains WHERE name = $1", name)
    except Exception as e:  # pragma: no cover - cleanup best effort
        print(f"Warning: could not clean up domain {name}: {e}")


@pytest.fixture
def domain_repo(db_params):
    return DomainRepository(connection_string=_connection_string(db_params))


@pytest.fixture
def ingestion_repo(db_params):
    return IngestionRepository(connection_string=_connection_string(db_params))


class TestDomainRegistry:
    @pytest.mark.asyncio
    async def test_create_then_list(self, domain_repo, domain_name):
        created = await domain_repo.create_domain(
            domain_name, display_name="Test Domain", description="scratch"
        )
        assert created["table_name"] == domain_name
        assert created["display_name"] == "Test Domain"
        assert created["document_count"] == 0

        listed = await domain_repo.list_domains(reconcile=False)
        assert domain_name in {d["name"] for d in listed}

    @pytest.mark.asyncio
    async def test_display_name_defaults_from_the_slug(self, domain_repo, db_connection):
        name = f"test_linear_algebra_{uuid.uuid4().hex[:6]}"
        try:
            created = await domain_repo.create_domain(name)
            assert created["display_name"] == name.replace("_", " ").title()
        finally:
            await db_connection.execute("DELETE FROM domains WHERE name = $1", name)

    @pytest.mark.asyncio
    async def test_create_is_idempotent(self, domain_repo, domain_name):
        first = await domain_repo.create_domain(domain_name, display_name="First")
        second = await domain_repo.create_domain(domain_name, display_name="Second")
        # ON CONFLICT DO NOTHING: the first write wins rather than being overwritten.
        assert second["display_name"] == first["display_name"] == "First"

    @pytest.mark.asyncio
    async def test_ensure_domain_creates_then_reuses(self, domain_repo, domain_name):
        created = await domain_repo.ensure_domain(domain_name)
        reused = await domain_repo.ensure_domain(domain_name)
        assert created["name"] == reused["name"] == domain_name

    @pytest.mark.asyncio
    async def test_reserved_names_are_rejected(self, domain_repo):
        with pytest.raises(ValueError, match="Reserved table name"):
            await domain_repo.create_domain("documents")

    @pytest.mark.asyncio
    async def test_delete_removes_the_registry_row(self, domain_repo, domain_name):
        await domain_repo.create_domain(domain_name)
        assert await domain_repo.delete_domain(domain_name) is True
        assert await domain_repo.get_domain(domain_name) is None
        assert await domain_repo.delete_domain(domain_name) is False


class TestDocumentMembership:
    @pytest.mark.asyncio
    async def test_registered_document_appears_in_its_domain(
        self, domain_repo, ingestion_repo, domain_name
    ):
        await domain_repo.ensure_domain(domain_name)
        doc_id = str(uuid.uuid4())
        await ingestion_repo.register_document(
            doc_id=doc_id,
            file_name="Linear_Algebra_v3.pdf",
            raw_storage_path=f"/tmp/{doc_id}_Linear_Algebra_v3.pdf",
            file_size=1234,
            target_table_name=domain_name,
            doc_name="Linear Algebra",
            domain=domain_name,
        )

        docs = await domain_repo.list_documents(domain_name)
        assert [d["doc_name"] for d in docs] == ["Linear Algebra"]
        assert docs[0]["stage"] == "registered"

        domain = await domain_repo.get_domain(domain_name)
        assert domain["document_count"] == 1

    @pytest.mark.asyncio
    async def test_doc_name_defaults_to_the_filename_stem(
        self, domain_repo, ingestion_repo, domain_name
    ):
        await domain_repo.ensure_domain(domain_name)
        doc_id = str(uuid.uuid4())
        row = await ingestion_repo.register_document(
            doc_id=doc_id,
            file_name="Calculus.pdf",
            raw_storage_path=f"/tmp/{doc_id}_Calculus.pdf",
            file_size=10,
            target_table_name=domain_name,
            domain=domain_name,
        )
        assert row["doc_name"] == "Calculus"

    @pytest.mark.asyncio
    async def test_two_documents_share_one_domain(
        self, domain_repo, ingestion_repo, domain_name
    ):
        await domain_repo.ensure_domain(domain_name)
        for file_name, doc_name in [("a.pdf", "Linear Algebra"), ("b.pdf", "Calculus")]:
            doc_id = str(uuid.uuid4())
            await ingestion_repo.register_document(
                doc_id=doc_id,
                file_name=file_name,
                raw_storage_path=f"/tmp/{doc_id}_{file_name}",
                file_size=10,
                target_table_name=domain_name,
                doc_name=doc_name,
                domain=domain_name,
            )

        docs = await domain_repo.list_documents(domain_name)
        assert sorted(d["doc_name"] for d in docs) == ["Calculus", "Linear Algebra"]

    @pytest.mark.asyncio
    async def test_same_name_twice_yields_two_ids(
        self, domain_repo, ingestion_repo, domain_name
    ):
        """doc_name is a label, not a key — which is why filters use document ids."""
        await domain_repo.ensure_domain(domain_name)
        for _ in range(2):
            doc_id = str(uuid.uuid4())
            await ingestion_repo.register_document(
                doc_id=doc_id,
                file_name="Linear_Algebra.pdf",
                raw_storage_path=f"/tmp/{doc_id}_Linear_Algebra.pdf",
                file_size=10,
                target_table_name=domain_name,
                doc_name="Linear Algebra",
                domain=domain_name,
            )

        ids = await domain_repo.find_document_ids_by_name("Linear Algebra", scope=domain_name)
        assert len(ids) == 2
        assert len(set(ids)) == 2

    @pytest.mark.asyncio
    async def test_unknown_domain_is_rejected_by_the_foreign_key(
        self, ingestion_repo, domain_name
    ):
        """documents.domain is an FK, so membership cannot point at nothing."""
        doc_id = str(uuid.uuid4())
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await ingestion_repo.register_document(
                doc_id=doc_id,
                file_name="orphan.pdf",
                raw_storage_path=f"/tmp/{doc_id}_orphan.pdf",
                file_size=10,
                target_table_name=domain_name,
                domain=f"never_created_{uuid.uuid4().hex[:8]}",
            )


class TestChunkTableDocName:
    @pytest.mark.asyncio
    async def test_vector_store_creates_and_returns_doc_name(
        self, db_params, db_connection, domain_name
    ):
        from app.ingestion.embedding.chunk import Chunk
        from app.ingestion.embedding.vector_store import VectorStore

        store = VectorStore(
            connection_params={
                "host": db_params["host"],
                "port": str(db_params["port"]),
                "dbname": db_params["database"],
                "user": db_params["user"],
                "password": db_params["password"],
            },
            table_name=domain_name,
        )
        doc_id = str(uuid.uuid4())
        await store.add_chunks([
            Chunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                text="Vectors are elements of a vector space.",
                embedding=[0.1] * 384,
                metadata={"page_number": 12},
                doc_name="Linear Algebra",
            )
        ])

        results = await store.search_similar_chunks(
            query_embedding=[0.1] * 384, limit=5, threshold=0.0
        )
        assert results
        assert results[0]["doc_name"] == "Linear Algebra"

        bm25 = await store.search_bm25(query="vectors", limit=5)
        assert bm25
        assert bm25[0]["doc_name"] == "Linear Algebra"

    @pytest.mark.asyncio
    async def test_pre_006_table_self_heals(self, db_params, db_connection, domain_name):
        """A chunk table created before 006 gains doc_name on first use."""
        from app.ingestion.embedding.vector_store import VectorStore

        await db_connection.execute(f"""
            CREATE TABLE "{domain_name}" (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector(384),
                metadata JSONB,
                entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        store = VectorStore(
            connection_params={
                "host": db_params["host"],
                "port": str(db_params["port"]),
                "dbname": db_params["database"],
                "user": db_params["user"],
                "password": db_params["password"],
            },
            table_name=domain_name,
        )
        await store._initialize_database()

        has_column = await db_connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_name = $1 AND column_name = 'doc_name')",
            domain_name,
        )
        assert has_column is True


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_a_chunk_table_without_a_registry_row_is_registered_on_read(
        self, domain_repo, db_connection, domain_name
    ):
        """Hand-created tables must not be invisible in the domain list."""
        await db_connection.execute(f"""
            CREATE TABLE "{domain_name}" (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector(384)
            )
        """)
        await db_connection.execute(
            f'INSERT INTO "{domain_name}" (id, document_id, text) VALUES ($1, $2, $3)',
            str(uuid.uuid4()), str(uuid.uuid4()), "some text",
        )

        listed = await domain_repo.list_domains(reconcile=True)
        assert domain_name in {d["name"] for d in listed}
