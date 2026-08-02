-- ============================================
-- Graph Tables Migration for RAG + Knowledge Graph
-- Machine Learning / Deep Learning Domain
-- ============================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "postgis";  -- Required by pgrouting
CREATE EXTENSION IF NOT EXISTS "pgrouting";

-- ============================================
-- ENTITIES TABLE
-- Stores extracted entities (concepts, models, algorithms, datasets, etc.)
-- ============================================
CREATE TABLE IF NOT EXISTS entities (
    entity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- CONCEPT, ALGORITHM, MODEL, DATASET, etc.
    confidence FLOAT DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    embedding vector(384),  -- Match your SentenceTransformer dimension (all-MiniLM-L6-v2)
    metadata JSONB DEFAULT '{}',
    source_chunk_ids UUID[] DEFAULT ARRAY[]::UUID[],  -- Track source chunks
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for entities
CREATE INDEX IF NOT EXISTS idx_entities_entity_name ON entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_entities_entity_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_entities_metadata ON entities USING gin(metadata);

COMMENT ON TABLE entities IS 'ML/DL entities extracted from documents (models, algorithms, datasets, concepts, etc.)';
COMMENT ON COLUMN entities.entity_name IS 'Name of the entity as it appears in text';
COMMENT ON COLUMN entities.entity_type IS 'Type from ML/DL domain ontology (MODEL, ALGORITHM, DATASET, etc.)';
COMMENT ON COLUMN entities.confidence IS 'Extraction confidence score (0-1)';
COMMENT ON COLUMN entities.embedding IS 'Vector embedding for semantic search (384 dimensions)';

-- ============================================
-- RELATIONSHIPS TABLE
-- Stores relationships between entities
-- ============================================
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,  -- USES, IMPROVES, RELATED_TO, PART_OF, TRAINED_ON, etc.
    confidence FLOAT DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    weight FLOAT DEFAULT 1.0,  -- For graph algorithms (can be adjusted based on confidence)
    metadata JSONB DEFAULT '{}',
    source_chunk_ids UUID[] DEFAULT ARRAY[]::UUID[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT no_self_reference CHECK (source_entity_id != target_entity_id)
);

-- Indexes for relationships
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence);
CREATE INDEX IF NOT EXISTS idx_relationships_metadata ON relationships USING gin(metadata);

COMMENT ON TABLE relationships IS 'Relationships between entities (USES, IMPROVES, TRAINED_ON, etc.)';
COMMENT ON COLUMN relationships.relationship_type IS 'Type of relationship from ML/DL domain (USES, TRAINED_ON, IMPROVES, etc.)';
COMMENT ON COLUMN relationships.weight IS 'Weight for graph algorithms (default 1.0)';

-- ============================================
-- PGROUTING SUPPORT TABLES
-- pgRouting requires BIGINT node IDs, so we need a mapping
-- ============================================

-- Node ID mapping table (UUID to BIGINT)
CREATE TABLE IF NOT EXISTS entity_nodes (
    entity_id UUID PRIMARY KEY REFERENCES entities(entity_id) ON DELETE CASCADE,
    node_id BIGINT UNIQUE NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS entity_node_id_seq START WITH 1;
CREATE INDEX IF NOT EXISTS idx_entity_nodes_node_id ON entity_nodes(node_id);

COMMENT ON TABLE entity_nodes IS 'Mapping between entity UUIDs and pgRouting BIGINT node IDs';

-- Edges table for pgRouting (optimized format)
CREATE TABLE IF NOT EXISTS entity_edges (
    id SERIAL PRIMARY KEY,
    source BIGINT NOT NULL,  -- node_id from entity_nodes
    target BIGINT NOT NULL,  -- node_id from entity_nodes
    cost FLOAT NOT NULL DEFAULT 1.0,
    reverse_cost FLOAT NOT NULL DEFAULT 1.0,  -- For undirected graphs
    relationship_id UUID REFERENCES relationships(relationship_id) ON DELETE CASCADE,
    relationship_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source);
CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges(target);
CREATE INDEX IF NOT EXISTS idx_entity_edges_relationship ON entity_edges(relationship_id);

COMMENT ON TABLE entity_edges IS 'pgRouting-compatible edges table (BIGINT IDs)';
COMMENT ON COLUMN entity_edges.cost IS 'Cost calculated as (1.0 - confidence) * weight (lower = better)';

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get or create node ID for entity
CREATE OR REPLACE FUNCTION get_or_create_node_id(entity_uuid UUID)
RETURNS BIGINT AS $$
DECLARE
    node_id_val BIGINT;
BEGIN
    SELECT node_id INTO node_id_val FROM entity_nodes WHERE entity_id = entity_uuid;

    IF node_id_val IS NULL THEN
        node_id_val := nextval('entity_node_id_seq');
        INSERT INTO entity_nodes (entity_id, node_id) VALUES (entity_uuid, node_id_val);
    END IF;

    RETURN node_id_val;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_or_create_node_id IS 'Get or create pgRouting node ID for an entity UUID';

-- Function to sync relationships to entity_edges
CREATE OR REPLACE FUNCTION sync_entity_edges()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT' OR TG_OP = 'UPDATE') THEN
        -- Delete old edge if updating
        IF (TG_OP = 'UPDATE') THEN
            DELETE FROM entity_edges WHERE relationship_id = OLD.relationship_id;
        END IF;

        -- Calculate cost based on confidence (higher confidence = lower cost)
        INSERT INTO entity_edges (source, target, cost, reverse_cost, relationship_id, relationship_type)
        VALUES (
            get_or_create_node_id(NEW.source_entity_id),
            get_or_create_node_id(NEW.target_entity_id),
            (1.0 - NEW.confidence) * NEW.weight,
            (1.0 - NEW.confidence) * NEW.weight,
            NEW.relationship_id,
            NEW.relationship_type
        );

        RETURN NEW;
    ELSIF (TG_OP = 'DELETE') THEN
        DELETE FROM entity_edges WHERE relationship_id = OLD.relationship_id;
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION sync_entity_edges IS 'Automatically sync relationships table to entity_edges for pgRouting';

-- Trigger to automatically sync relationships to edges
DROP TRIGGER IF EXISTS sync_relationships_to_edges ON relationships;
CREATE TRIGGER sync_relationships_to_edges
    AFTER INSERT OR UPDATE OR DELETE ON relationships
    FOR EACH ROW EXECUTE FUNCTION sync_entity_edges();

COMMENT ON TRIGGER sync_relationships_to_edges ON relationships IS 'Auto-sync relationships to pgRouting edges';

-- ============================================
-- GRAPH QUERY FUNCTIONS
-- ============================================

-- Find shortest path between two entities
CREATE OR REPLACE FUNCTION find_entity_path(
    source_entity_uuid UUID,
    target_entity_uuid UUID
) RETURNS TABLE(
    seq INTEGER,
    entity_id UUID,
    entity_name TEXT,
    entity_type TEXT,
    edge_cost FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.seq,
        en.entity_id,
        e.entity_name,
        e.entity_type,
        p.cost as edge_cost
    FROM pgr_dijkstra(
        'SELECT id, source, target, cost FROM entity_edges',
        (SELECT node_id FROM entity_nodes WHERE entity_id = source_entity_uuid),
        (SELECT node_id FROM entity_nodes WHERE entity_id = target_entity_uuid),
        directed := false
    ) p
    JOIN entity_nodes en ON p.node = en.node_id
    JOIN entities e ON en.entity_id = e.entity_id
    ORDER BY p.seq;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_entity_path IS 'Find shortest path between two entities using Dijkstra algorithm';

-- Get connected entities within N hops
CREATE OR REPLACE FUNCTION get_connected_entities(
    source_entity_uuid UUID,
    max_hops INTEGER DEFAULT 2
) RETURNS TABLE(
    entity_id UUID,
    entity_name TEXT,
    entity_type TEXT,
    distance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        en.entity_id,
        e.entity_name,
        e.entity_type,
        dd.agg_cost as distance
    FROM pgr_drivingDistance(
        'SELECT id, source, target, cost FROM entity_edges',
        (SELECT node_id FROM entity_nodes WHERE entity_id = source_entity_uuid),
        max_hops::FLOAT,
        directed := false
    ) dd
    JOIN entity_nodes en ON dd.node = en.node_id
    JOIN entities e ON en.entity_id = e.entity_id
    WHERE en.entity_id != source_entity_uuid
    ORDER BY dd.agg_cost;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_connected_entities IS 'Get all entities connected within N hops using driving distance';

-- Calculate PageRank for entity importance
CREATE OR REPLACE FUNCTION calculate_entity_pagerank(
    damping_factor FLOAT DEFAULT 0.85,
    max_iterations INTEGER DEFAULT 100
) RETURNS TABLE(
-- NOTE: pgr_pageRank function does not exist in pgRouting
-- This function is disabled until a proper PageRank implementation is added
-- Consider using pg_rank extension or implementing PageRank in application code

-- ============================================
-- CHUNKS TABLE (Document Storage)
-- Create chunks table if not exists, then add entity references
-- ============================================

-- Create chunks table with entity support
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    entity_ids UUID[] DEFAULT ARRAY[]::UUID[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for chunks table
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_entity_ids
    ON chunks USING gin(entity_ids);

COMMENT ON TABLE chunks IS 'Document chunks with embeddings and entity references';
COMMENT ON COLUMN chunks.entity_ids IS 'Array of entity UUIDs extracted from this chunk';

-- ============================================
-- SAMPLE QUERIES FOR TESTING
-- ============================================

/*
-- Example 1: Find path between two entities
SELECT * FROM find_entity_path(
    'source-entity-uuid'::UUID,
    'target-entity-uuid'::UUID
);

-- Example 2: Get entities within 2 hops
SELECT * FROM get_connected_entities(
    'entity-uuid'::UUID,
    2
);

-- Example 3: Calculate PageRank
SELECT * FROM calculate_entity_pagerank();

-- Example 4: Get graph statistics
SELECT
    (SELECT COUNT(*) FROM entities) as total_entities,
    (SELECT COUNT(*) FROM relationships) as total_relationships,
    (SELECT COUNT(DISTINCT entity_type) FROM entities) as unique_entity_types,
    (SELECT COUNT(DISTINCT relationship_type) FROM relationships) as unique_relationship_types;

-- Example 5: Find most connected entities
SELECT
    e.entity_id,
    e.entity_name,
    e.entity_type,
    COUNT(*) as connection_count
FROM entities e
JOIN relationships r ON (e.entity_id = r.source_entity_id OR e.entity_id = r.target_entity_id)
GROUP BY e.entity_id, e.entity_name, e.entity_type
ORDER BY connection_count DESC
LIMIT 10;
*/

-- ============================================
-- VERIFICATION
-- ============================================

-- Verify extensions are installed
SELECT
    extname,
    extversion,
    CASE
        WHEN extname = 'pgrouting' THEN 'Graph algorithms enabled'
        WHEN extname = 'vector' THEN 'Vector similarity search enabled'
        WHEN extname = 'uuid-ossp' THEN 'UUID generation enabled'
    END as purpose
FROM pg_extension
WHERE extname IN ('pgrouting', 'vector', 'uuid-ossp')
ORDER BY extname;
