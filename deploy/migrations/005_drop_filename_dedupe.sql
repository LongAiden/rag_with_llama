-- ============================================
-- Migration 005: drop filename de-duplication
--
-- documents.file_name was globally UNIQUE (migration 003), which made the filename
-- the de-duplication key for the whole system. That had two bad consequences:
--
-- 1. The same file could never be ingested twice — not into a second chunk table,
--    and not again after DELETE /table/{name} dropped its chunks but left the
--    documents row behind. The file became permanently un-ingestable.
-- 2. Uploads were silently answered with status="duplicate" instead of being
--    processed, which is surprising when the user deliberately re-uploads.
--
-- Uploads are now always registered as new documents. Re-uploading the same file
-- creates a second document with its own id; deleting either one is independent.
-- The directory scan still avoids re-registering files by raw_storage_path, which
-- is about not reprocessing the same file on disk on every sweep — not about
-- content de-duplication.
--
-- NOTE: migrations are mounted at /docker-entrypoint-initdb.d, which Postgres only
-- runs on an empty data directory. On an existing volume apply this by hand:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     < migrations/005_drop_filename_dedupe.sql
-- ============================================

-- The constraint created by `file_name TEXT NOT NULL UNIQUE` in migration 003 is
-- named documents_file_name_key by Postgres.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_file_name_key;

-- Filename lookups are still used for reporting, so keep a non-unique index.
CREATE INDEX IF NOT EXISTS idx_documents_file_name
    ON documents(file_name);

COMMENT ON COLUMN documents.file_name IS 'Display filename. NOT unique: the same file may be uploaded more than once, and each upload is its own document.';
COMMENT ON COLUMN documents.content_hash IS 'Reserved for future hash-based deduplication. Filename-based dedupe was removed in migration 005.';
