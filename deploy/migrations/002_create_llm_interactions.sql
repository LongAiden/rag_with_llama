-- Migration 002: LLM Interaction Observability Table
-- Persists every query/answer cycle for local cost tracking and LLM-Ops monitoring.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS llm_interactions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id    TEXT,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    model         TEXT NOT NULL,
    backend       TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    total_tokens  INTEGER,
    latency_ms    INTEGER,
    sources_used  INTEGER,
    table_name    TEXT,
    rerank_method TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_created_at
    ON llm_interactions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_model
    ON llm_interactions (model, backend);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_session
    ON llm_interactions (session_id)
    WHERE session_id IS NOT NULL;
