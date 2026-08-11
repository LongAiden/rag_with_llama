-- ============================================
-- Migration 006: domain registry + doc_name
--
-- A chunk table (math, history, technical) has always been an implicit topic
-- bucket: many documents share one table, and the only thing tying a chunk to
-- its source was an opaque document_id UUID. This makes the bucket explicit.
--
--   domains            registry, one row per domain, 1:1 with a chunk table
--   documents.domain   membership FK
--   documents.doc_name human-readable name from upload
--   <chunk>.doc_name   denormalized copy, so the search SELECT needs no join
--
-- The denormalized copy is a snapshot, not a mirror. documents.doc_name is
-- authoritative; renaming a document must update both or they drift. See
-- docs/plans/20260812_domains_and_doc_name.md section 7.3.
--
-- NOTE: migrations are mounted at /docker-entrypoint-initdb.d, which Postgres
-- only runs on an empty data directory. On an existing volume apply by hand:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     < deploy/migrations/006_domains_and_doc_name.sql
-- ============================================

CREATE TABLE IF NOT EXISTS domains (
    name         TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description  TEXT,
    table_name   TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT domains_name_is_identifier
        CHECK (name ~ '^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')
);

COMMENT ON TABLE  domains IS 'Registry of document domains. One domain maps 1:1 to one pgvector chunk table.';
COMMENT ON COLUMN domains.name IS 'Slug and API key. Constrained to a safe SQL identifier because it is also the chunk table name.';
COMMENT ON COLUMN domains.table_name IS 'Physical chunk table. Equals name today; kept separate so a domain could be renamed without a table rename.';

-- ============================================
-- documents: membership + human-readable name
-- ============================================
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain   TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_domain   ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_doc_name ON documents(doc_name);

COMMENT ON COLUMN documents.doc_name IS 'Human-readable document name from upload; defaults to the filename stem. Denormalized onto chunk rows - a rename must update both tables.';
COMMENT ON COLUMN documents.domain   IS 'Domain this document belongs to; FK to domains(name). NULL for pre-006 rows until backfilled.';

-- ============================================
-- Backfill: one domain row per existing chunk table, and doc_name on those
-- tables, so nothing disappears from the UI after this migration.
--
-- The chunk-table predicate matches CHUNK_TABLES_QUERY in
-- src/app/infra/db/table_repository.py: a public table with both a document_id
-- and an embedding column, excluding the graph tables.
-- ============================================
DO $$
DECLARE t TEXT;
BEGIN
    FOR t IN
        SELECT DISTINCT t1.table_name
        FROM information_schema.columns t1
        WHERE t1.table_schema = 'public'
          AND t1.column_name = 'document_id'
          AND EXISTS (
              SELECT 1 FROM information_schema.columns t2
              WHERE t2.table_name = t1.table_name
                AND t2.table_schema = 'public'
                AND t2.column_name = 'embedding'
          )
          AND t1.table_name NOT IN
              ('entities', 'relationships', 'entity_nodes', 'entity_edges')
    LOOP
        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS doc_name TEXT', t);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (doc_name)',
                       t || '_doc_name_idx', t);

        INSERT INTO domains (name, display_name, table_name)
        VALUES (t, initcap(replace(t, '_', ' ')), t)
        ON CONFLICT (name) DO NOTHING;
    END LOOP;
END $$;

-- Existing documents: name from the filename stem, domain from the table they
-- were ingested into.
UPDATE documents
SET doc_name = regexp_replace(file_name, '\.[^.]+$', '')
WHERE doc_name IS NULL;

UPDATE documents
SET domain = target_table_name
WHERE domain IS NULL
  AND target_table_name IN (SELECT name FROM domains);

-- FK added after the backfill so it cannot reject the migration's own rows.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_domain_fkey;
ALTER TABLE documents
    ADD CONSTRAINT documents_domain_fkey
    FOREIGN KEY (domain) REFERENCES domains(name) ON DELETE SET NULL;

-- ============================================
-- Deliberately NOT done here: backfilling doc_name onto existing chunk rows.
-- That is an UPDATE ... FROM documents per table, which rewrites every row.
-- Pre-006 chunks return doc_name NULL and the UI falls back to the document_id
-- prefix, which is exactly the pre-006 behaviour. Run per domain when wanted:
--
--   UPDATE math m
--   SET doc_name = d.doc_name
--   FROM documents d
--   WHERE m.document_id = d.id AND m.doc_name IS NULL;
-- ============================================
