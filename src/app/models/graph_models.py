"""
Pydantic models for graph-related API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID


# ============================================
# ENTITY MODELS
# ============================================

class EntityBase(BaseModel):
    """Base entity model."""
    entity_name: str
    entity_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = {}


class EntityCreate(EntityBase):
    """Create new entity."""
    chunk_id: UUID


class EntityResponse(EntityBase):
    """Entity response."""
    entity_id: UUID
    source_chunk_ids: List[UUID] = []
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class EntitySearchRequest(BaseModel):
    """Search entities request."""
    query: str
    entity_type: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


# ============================================
# RELATIONSHIP MODELS
# ============================================

class RelationshipBase(BaseModel):
    """Base relationship model."""
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    metadata: Optional[Dict[str, Any]] = {}


class RelationshipCreate(RelationshipBase):
    """Create new relationship."""
    chunk_id: UUID


class RelationshipResponse(RelationshipBase):
    """Relationship response."""
    relationship_id: UUID
    source_chunk_ids: List[UUID] = []
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class RelationshipWithNames(RelationshipResponse):
    """Relationship with entity names."""
    source_name: str
    source_type: str
    target_name: str
    target_type: str


# ============================================
# GRAPH PATH MODELS
# ============================================

class EntityInPath(BaseModel):
    """Entity in a path."""
    entity_id: str
    entity_name: str
    entity_type: str


class RelationshipInPath(BaseModel):
    """Relationship in a path."""
    relationship_id: str
    relationship_type: str
    confidence: float


class GraphPathResponse(BaseModel):
    """Response for shortest path query."""
    source_entity_id: str
    target_entity_id: str
    path_found: bool
    path_length: int
    path_cost: float
    path_entities: List[EntityInPath]
    path_relationships: List[RelationshipInPath]


# ============================================
# CONNECTED ENTITIES MODELS
# ============================================

class ConnectedEntity(BaseModel):
    """Connected entity with distance."""
    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float
    distance: float


class ConnectedEntitiesResponse(BaseModel):
    """Response for connected entities query."""
    source_entity_id: str
    max_hops: int
    connected_entities: List[ConnectedEntity]
    total_connected: int


# ============================================
# GRAPH STATS MODELS
# ============================================

class MostConnectedEntity(BaseModel):
    """Most connected entity details."""
    entity_id: str
    entity_name: str
    entity_type: str
    connection_count: int


class GraphStatsResponse(BaseModel):
    """Graph statistics response."""
    total_entities: int
    total_relationships: int
    avg_relationships_per_entity: float
    most_connected_entity: Optional[MostConnectedEntity]
    relationship_type_distribution: Dict[str, int]
    entity_type_distribution: Dict[str, int]


# ============================================
# PAGERANK MODELS
# ============================================

class EntityWithPageRank(BaseModel):
    """Entity with PageRank score."""
    entity_id: str
    entity_name: str
    entity_type: str
    pagerank_score: float


class PageRankResponse(BaseModel):
    """PageRank calculation response."""
    entities: List[EntityWithPageRank]
    total_entities: int


# ============================================
# SUBGRAPH MODELS
# ============================================

class SubgraphNode(BaseModel):
    """Node in subgraph."""
    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float


class SubgraphEdge(BaseModel):
    """Edge in subgraph."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    source_name: str
    target_name: str


class SubgraphResponse(BaseModel):
    """Subgraph response."""
    nodes: List[SubgraphNode]
    edges: List[SubgraphEdge]
    node_count: int
    edge_count: int


class SubgraphRequest(BaseModel):
    """Subgraph request."""
    entity_ids: List[UUID]
    include_intermediate: bool = True


# ============================================
# EXTRACTION REQUEST MODELS
# ============================================

class ExtractEntitiesRequest(BaseModel):
    """Request to extract entities from a chunk."""
    chunk_id: UUID
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class ExtractRelationshipsRequest(BaseModel):
    """Request to extract relationships from a chunk."""
    chunk_id: UUID
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class BatchExtractionRequest(BaseModel):
    """Request to extract entities and relationships from multiple chunks."""
    chunk_ids: List[UUID]
    entity_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    relationship_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


# ============================================
# BATCH RESPONSE MODELS
# ============================================

class ExtractionResult(BaseModel):
    """Result of extraction for a single chunk."""
    chunk_id: UUID
    entities_extracted: int
    relationships_extracted: int
    success: bool
    error: Optional[str] = None


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction."""
    results: List[ExtractionResult]
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    total_entities_extracted: int
    total_relationships_extracted: int
