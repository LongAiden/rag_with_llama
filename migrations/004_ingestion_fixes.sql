-- ============================================
-- Migration 004: Ingestion pipeline correctness fixes
--
-- 1. documents.file_type   — the chunker branches on this; without a real column
--                            every PDF fell through to the generic chunking path
--                            and lost page numbers / section hierarchy.
-- 2. documents.error_stage — remembers which stage failed so retries resume from
--                            the furthest completed artifact instead of re-parsing.
-- 3. UNIQUE (document_id) on the artifact tables — retries used to append a second
--                            row and get_parsed()/get_chunked() picked one at random.
--
-- NOTE: migrations are mounted at /docker-entrypoint-initdb.d, which Postgres only
-- runs on an empty data directory. On an existing volume apply this by hand:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     < migrations/004_ingestion_fixes.sql
-- ============================================

ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_stage TEXT;

COMMENT ON COLUMN documents.file_type IS 'Lowercased extension without dot (pdf, docx, txt); set when the document reaches the parsed stage';
COMMENT ON COLUMN documents.error_stage IS 'Stage that raised the last error, used to resume retries without redoing completed work';

-- ============================================
-- One artifact row per document, so retries overwrite rather than accumulate.
-- De-duplicate any existing rows first (keep the newest) or the index creation fails.
-- ============================================
DELETE FROM document_parsed a
    USING document_parsed b
    WHERE a.document_id = b.document_id
      AND (a.created_at, a.id) < (b.created_at, b.id);

DELETE FROM document_chunked a
    USING document_chunked b
    WHERE a.document_id = b.document_id
      AND (a.created_at, a.id) < (b.created_at, b.id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_parsed_document_id_unique
    ON document_parsed(document_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunked_document_id_unique
    ON document_chunked(document_id);

-- The plain (non-unique) indexes from migration 003 are now redundant.
DROP INDEX IF EXISTS idx_document_parsed_document_id;
DROP INDEX IF EXISTS idx_document_chunked_document_id;

-- ============================================
-- The weekly scan de-duplicates on the stored raw path, not the display filename.
-- ============================================
CREATE INDEX IF NOT EXISTS idx_documents_raw_storage_path
    ON documents(raw_storage_path);
