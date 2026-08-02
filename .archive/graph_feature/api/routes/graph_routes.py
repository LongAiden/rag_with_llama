"""
FastAPI routes for graph operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from uuid import UUID
import asyncpg

from models.graph_models import (
    EntityResponse,
    EntitySearchRequest,
    RelationshipWithNames,
    GraphPathResponse,
    ConnectedEntitiesResponse,
    GraphStatsResponse,
    PageRankResponse,
    SubgraphResponse,
    SubgraphRequest,
    ExtractEntitiesRequest,
    ExtractRelationshipsRequest,
    BatchExtractionRequest,
    BatchExtractionResponse,
    ExtractionResult,
)
from config.graph_config import get_graph_config
from graph_processing import (
    EntityExtractor,
    RelationshipExtractor,
    GraphService,
)

router = APIRouter(prefix="/graph", tags=["graph"])


# Dependency to get database pool (you'll need to implement this)
async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    # Import from your existing database setup
    from api.app import get_db_pool as get_pool
    return await get_pool()


async def get_entity_extractor(pool: asyncpg.Pool = Depends(get_db_pool)):
    """Get entity extractor instance."""
    import os
    from sentence_transformers import SentenceTransformer

    config = get_graph_config()
    gemini_api_key = config.gemini_api_key or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    gemini_model = config.gemini_model
    embedding_model = SentenceTransformer(config.entity_embedding_model)
    return EntityExtractor(pool, gemini_api_key, gemini_model, embedding_model)


async def get_relationship_extractor(pool: asyncpg.Pool = Depends(get_db_pool)):
    """Get relationship extractor instance."""
    import os

    config = get_graph_config()
    gemini_api_key = config.gemini_api_key or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    return RelationshipExtractor(pool, gemini_api_key, config.gemini_model)


async def get_graph_service(pool: asyncpg.Pool = Depends(get_db_pool)):
    """Get graph service instance."""
    return GraphService(pool)


# ============================================
# ENTITY EXTRACTION ENDPOINTS
# ============================================

@router.post("/extract/entities", response_model=List[EntityResponse])
async def extract_entities(
    request: ExtractEntitiesRequest,
    extractor: EntityExtractor = Depends(get_entity_extractor),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Extract entities from a specific chunk.

    This endpoint uses LLM to identify ML/DL entities (models, algorithms, datasets, etc.)
    from the text content of a chunk.
    """
    # Get chunk text from database
    async with pool.acquire() as conn:
        chunk = await conn.fetchrow(
            "SELECT chunk_text FROM chunks WHERE chunk_id = $1",
            request.chunk_id
        )
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

    # Extract entities
    entities = await extractor.extract_entities_from_chunk(
        chunk_id=request.chunk_id,
        chunk_text=chunk['chunk_text'],
        confidence_threshold=request.confidence_threshold
    )

    return entities


@router.post("/extract/relationships", response_model=List[RelationshipWithNames])
async def extract_relationships(
    request: ExtractRelationshipsRequest,
    entity_extractor: EntityExtractor = Depends(get_entity_extractor),
    rel_extractor: RelationshipExtractor = Depends(get_relationship_extractor),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Extract relationships between entities in a chunk.

    This endpoint identifies relationships (USES, IMPROVES, TRAINED_ON, etc.)
    between entities that have been extracted from the chunk.
    """
    # Get chunk text and entities
    async with pool.acquire() as conn:
        chunk = await conn.fetchrow(
            "SELECT chunk_text FROM chunks WHERE chunk_id = $1",
            request.chunk_id
        )
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

    # Get entities for this chunk
    entities = await entity_extractor.get_entities_by_chunk(request.chunk_id)

    if len(entities) < 2:
        return []

    # Extract relationships
    relationships = await rel_extractor.extract_relationships_from_chunk(
        chunk_id=request.chunk_id,
        chunk_text=chunk['chunk_text'],
        entities=entities,
        confidence_threshold=request.confidence_threshold
    )

    return relationships


@router.post("/extract/batch", response_model=BatchExtractionResponse)
async def batch_extract(
    request: BatchExtractionRequest,
    entity_extractor: EntityExtractor = Depends(get_entity_extractor),
    rel_extractor: RelationshipExtractor = Depends(get_relationship_extractor),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Extract entities and relationships from multiple chunks in batch.

    This is useful for processing an entire document after chunking.
    """
    results = []
    total_entities = 0
    total_relationships = 0
    successful = 0
    failed = 0

    for chunk_id in request.chunk_ids:
        try:
            # Get chunk text
            async with pool.acquire() as conn:
                chunk = await conn.fetchrow(
                    "SELECT chunk_text FROM chunks WHERE chunk_id = $1",
                    chunk_id
                )
                if not chunk:
                    results.append(ExtractionResult(
                        chunk_id=chunk_id,
                        entities_extracted=0,
                        relationships_extracted=0,
                        success=False,
                        error="Chunk not found"
                    ))
                    failed += 1
                    continue

            # Extract entities
            entities = await entity_extractor.extract_entities_from_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk['chunk_text'],
                confidence_threshold=request.entity_confidence_threshold
            )

            # Extract relationships
            relationships = []
            if len(entities) >= 2:
                relationships = await rel_extractor.extract_relationships_from_chunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk['chunk_text'],
                    entities=entities,
                    confidence_threshold=request.relationship_confidence_threshold
                )

            results.append(ExtractionResult(
                chunk_id=chunk_id,
                entities_extracted=len(entities),
                relationships_extracted=len(relationships),
                success=True
            ))

            total_entities += len(entities)
            total_relationships += len(relationships)
            successful += 1

        except Exception as e:
            results.append(ExtractionResult(
                chunk_id=chunk_id,
                entities_extracted=0,
                relationships_extracted=0,
                success=False,
                error=str(e)
            ))
            failed += 1

    return BatchExtractionResponse(
        results=results,
        total_chunks=len(request.chunk_ids),
        successful_chunks=successful,
        failed_chunks=failed,
        total_entities_extracted=total_entities,
        total_relationships_extracted=total_relationships
    )


# ============================================
# ENTITY QUERY ENDPOINTS
# ============================================

@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: UUID,
    extractor: EntityExtractor = Depends(get_entity_extractor)
):
    """Get entity details by ID."""
    entity = await extractor.get_entity_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.post("/entities/search", response_model=List[EntityResponse])
async def search_entities(
    request: EntitySearchRequest,
    extractor: EntityExtractor = Depends(get_entity_extractor)
):
    """
    Search entities by name or semantic similarity.

    Uses vector embeddings to find similar entities.
    """
    entities = await extractor.search_entities(
        query=request.query,
        entity_type=request.entity_type,
        limit=request.limit
    )
    return entities


@router.get("/entities/chunk/{chunk_id}", response_model=List[EntityResponse])
async def get_entities_by_chunk(
    chunk_id: UUID,
    extractor: EntityExtractor = Depends(get_entity_extractor)
):
    """Get all entities extracted from a specific chunk."""
    entities = await extractor.get_entities_by_chunk(chunk_id)
    return entities


# ============================================
# RELATIONSHIP QUERY ENDPOINTS
# ============================================

@router.get("/relationships/entity/{entity_id}", response_model=List[RelationshipWithNames])
async def get_entity_relationships(
    entity_id: UUID,
    relationship_type: str = Query(None, description="Filter by relationship type"),
    extractor: RelationshipExtractor = Depends(get_relationship_extractor)
):
    """Get all relationships for an entity."""
    relationships = await extractor.get_entity_relationships(
        entity_id=entity_id,
        relationship_type=relationship_type
    )
    return relationships


@router.get("/relationships/chunk/{chunk_id}", response_model=List[RelationshipWithNames])
async def get_relationships_by_chunk(
    chunk_id: UUID,
    extractor: RelationshipExtractor = Depends(get_relationship_extractor)
):
    """Get all relationships from a specific chunk."""
    relationships = await extractor.get_relationships_by_chunk(chunk_id)
    return relationships


# ============================================
# GRAPH ANALYSIS ENDPOINTS
# ============================================

@router.get("/path/{source_entity_id}/{target_entity_id}", response_model=GraphPathResponse)
async def find_shortest_path(
    source_entity_id: UUID,
    target_entity_id: UUID,
    service: GraphService = Depends(get_graph_service)
):
    """
    Find shortest path between two entities using Dijkstra's algorithm.

    This is useful for understanding how concepts are connected.
    Example: Find path from "BERT" to "ImageNet"
    """
    path = await service.find_shortest_path(source_entity_id, target_entity_id)
    return path


@router.get("/connected/{entity_id}", response_model=ConnectedEntitiesResponse)
async def get_connected_entities(
    entity_id: UUID,
    max_hops: int = Query(2, ge=1, le=5, description="Maximum hops to traverse"),
    service: GraphService = Depends(get_graph_service)
):
    """
    Get all entities connected within N hops.

    This helps discover related concepts. For example, all concepts
    within 2 hops of "ResNet" might include "CNN", "ImageNet", "Transfer Learning", etc.
    """
    connected = await service.get_connected_entities(entity_id, max_hops)

    return ConnectedEntitiesResponse(
        source_entity_id=str(entity_id),
        max_hops=max_hops,
        connected_entities=connected,
        total_connected=len(connected)
    )


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    service: GraphService = Depends(get_graph_service)
):
    """
    Get knowledge graph statistics.

    Provides overview of the graph: entity counts, relationship counts,
    most connected entities, and type distributions.
    """
    stats = await service.get_graph_stats()
    return stats


@router.get("/pagerank", response_model=PageRankResponse)
async def calculate_pagerank(
    service: GraphService = Depends(get_graph_service)
):
    """
    Calculate PageRank for all entities to find most important concepts.

    Entities with high PageRank scores are central to the knowledge base.
    """
    pagerank_entities = await service.calculate_pagerank()

    return PageRankResponse(
        entities=pagerank_entities,
        total_entities=len(pagerank_entities)
    )


@router.post("/subgraph", response_model=SubgraphResponse)
async def get_subgraph(
    request: SubgraphRequest,
    service: GraphService = Depends(get_graph_service)
):
    """
    Get a subgraph containing specified entities.

    Useful for visualizing relationships between a set of entities.
    """
    subgraph = await service.get_subgraph(
        entity_ids=request.entity_ids,
        include_intermediate=request.include_intermediate
    )
    return subgraph


# ============================================
# HEALTH CHECK
# ============================================

@router.get("/health")
async def graph_health_check(pool: asyncpg.Pool = Depends(get_db_pool)):
    """Check if graph tables and pgRouting extension are available."""
    async with pool.acquire() as conn:
        # Check if pgRouting is installed
        pgrouting_check = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'pgr_dijkstra')"
        )

        # Check if graph tables exist
        tables_check = await conn.fetchval(
            """
            SELECT COUNT(*) = 4 FROM information_schema.tables
            WHERE table_name IN ('entities', 'relationships', 'entity_nodes', 'entity_edges')
            """
        )

        return {
            "status": "healthy" if (pgrouting_check and tables_check) else "unhealthy",
            "pgrouting_installed": pgrouting_check,
            "graph_tables_exist": tables_check
        }
