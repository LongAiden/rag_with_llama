-- ============================================
-- Migration 003: Ingestion status and intermediate artifact tables
-- Tracks every input file through the pipeline stages and stores parsed
-- and chunked artifacts before embedding.
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- documents: ingestion status and claim source of truth
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    content_hash TEXT,
    raw_storage_path TEXT NOT NULL,
    file_size BIGINT,
    content_type TEXT,
    stage TEXT NOT NULL DEFAULT 'registered',
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_at TIMESTAMPTZ,
    claimed_by TEXT,
    parsed_id UUID,
    chunked_id UUID,
    chunk_count INTEGER,
    target_table_name TEXT,
    chunk_size INTEGER,
    parse_backend TEXT,
    last_error TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_stage_created
    ON documents(stage, created_at);

CREATE INDEX IF NOT EXISTS idx_documents_claimed_at
    ON documents(claimed_at);

COMMENT ON TABLE documents IS 'Ingestion status DB: one row per input file, tracks stage, claims, attempts and artifact links';
COMMENT ON COLUMN documents.stage IS 'Pipeline stage: registered, parsing, parsed, chunking, chunked, embedding, embedded, error, failed';
COMMENT ON COLUMN documents.content_hash IS 'Reserved for future hash-based deduplication (POC uses filename dedupe)';

-- ============================================
-- document_parsed: parsed text + metadata
-- ============================================
CREATE TABLE IF NOT EXISTS document_parsed (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parsed_text TEXT NOT NULL,
    parser_used TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_parsed_document_id
    ON document_parsed(document_id);

COMMENT ON TABLE document_parsed IS 'Intermediate parsed artifact storage (replaces input/markdown/ files)';

-- ============================================
-- document_chunked: chunk objects before embedding
-- ============================================
CREATE TABLE IF NOT EXISTS document_chunked (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunks JSONB NOT NULL,
    chunk_size INTEGER,
    chunk_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunked_document_id
    ON document_chunked(document_id);

COMMENT ON TABLE document_chunked IS 'Intermediate chunked artifact storage (pre-embedding)';
