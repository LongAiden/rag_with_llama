"""
Graph service for path finding and graph operations using pgRouting.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncpg

logger = logging.getLogger(__name__)


class GraphService:
    """Service for graph operations using pgRouting."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def find_shortest_path(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID
    ) -> Dict[str, Any]:
        """
        Find shortest path between two entities using pgRouting's Dijkstra algorithm.

        Args:
            source_entity_id: Starting entity UUID
            target_entity_id: Target entity UUID

        Returns:
            Dictionary with path information
        """
        async with self.db_pool.acquire() as conn:
            # Get node IDs
            source_node = await conn.fetchval(
                "SELECT node_id FROM entity_nodes WHERE entity_id = $1",
                source_entity_id
            )
            target_node = await conn.fetchval(
                "SELECT node_id FROM entity_nodes WHERE entity_id = $1",
                target_entity_id
            )

            if not source_node or not target_node:
                return self._empty_path_response(source_entity_id, target_entity_id)

            # Find shortest path using pgr_dijkstra
            query = """
                SELECT seq, node, edge, agg_cost
                FROM pgr_dijkstra(
                    'SELECT id, source, target, cost FROM entity_edges',
                    $1::bigint, $2::bigint, directed := false
                )
                ORDER BY seq
            """

            try:
                path_rows = await conn.fetch(query, source_node, target_node)

                if not path_rows:
                    return self._empty_path_response(source_entity_id, target_entity_id)

                # Convert nodes back to entity information
                path_entities = []
                for row in path_rows:
                    entity_info = await conn.fetchrow(
                        """
                        SELECT e.entity_id, e.entity_name, e.entity_type
                        FROM entity_nodes en
                        JOIN entities e ON en.entity_id = e.entity_id
                        WHERE en.node_id = $1
                        """,
                        row['node']
                    )
                    if entity_info:
                        path_entities.append({
                            'entity_id': str(entity_info['entity_id']),
                            'entity_name': entity_info['entity_name'],
                            'entity_type': entity_info['entity_type']
                        })

                # Get relationships in the path
                path_relationships = []
                for i in range(len(path_entities) - 1):
                    rel_info = await conn.fetchrow(
                        """
                        SELECT r.relationship_id, r.relationship_type, r.confidence
                        FROM relationships r
                        WHERE (r.source_entity_id = $1 AND r.target_entity_id = $2)
                           OR (r.source_entity_id = $2 AND r.target_entity_id = $1)
                        LIMIT 1
                        """,
                        UUID(path_entities[i]['entity_id']),
                        UUID(path_entities[i + 1]['entity_id'])
                    )
                    if rel_info:
                        path_relationships.append({
                            'relationship_id': str(rel_info['relationship_id']),
                            'relationship_type': rel_info['relationship_type'],
                            'confidence': float(rel_info['confidence'])
                        })

                return {
                    'source_entity_id': str(source_entity_id),
                    'target_entity_id': str(target_entity_id),
                    'path_found': True,
                    'path_length': len(path_entities) - 1,
                    'path_cost': float(path_rows[-1]['agg_cost']),
                    'path_entities': path_entities,
                    'path_relationships': path_relationships
                }

            except Exception as e:
                logger.error(f"Error finding path: {e}")
                return self._empty_path_response(source_entity_id, target_entity_id)

    async def get_connected_entities(
        self,
        entity_id: UUID,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get entities connected within max_hops using pgRouting's drivingDistance.

        Args:
            entity_id: Source entity UUID
            max_hops: Maximum number of hops (edges) to traverse

        Returns:
            List of connected entities with distance
        """
        async with self.db_pool.acquire() as conn:
            # Get node ID for the entity
            node_id = await conn.fetchval(
                "SELECT node_id FROM entity_nodes WHERE entity_id = $1",
                entity_id
            )

            if not node_id:
                return []

            # Use pgr_drivingDistance to find connected entities
            query = """
                SELECT DISTINCT node, agg_cost
                FROM pgr_drivingDistance(
                    'SELECT id, source, target, cost FROM entity_edges',
                    $1::bigint, $2::float, directed := false
                )
                WHERE node != $1
                ORDER BY agg_cost
            """

            try:
                connected_nodes = await conn.fetch(query, node_id, float(max_hops))

                # Convert node IDs back to entity information
                connected_entities = []
                for row in connected_nodes:
                    entity_info = await conn.fetchrow(
                        """
                        SELECT e.entity_id, e.entity_name, e.entity_type, e.confidence
                        FROM entity_nodes en
                        JOIN entities e ON en.entity_id = e.entity_id
                        WHERE en.node_id = $1
                        """,
                        row['node']
                    )
                    if entity_info:
                        connected_entities.append({
                            'entity_id': str(entity_info['entity_id']),
                            'entity_name': entity_info['entity_name'],
                            'entity_type': entity_info['entity_type'],
                            'confidence': float(entity_info['confidence']),
                            'distance': float(row['agg_cost'])
                        })

                return connected_entities

            except Exception as e:
                logger.error(f"Error getting connected entities: {e}")
                return []

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        async with self.db_pool.acquire() as conn:
            # Single query for basic stats
            stats_query = """
                SELECT
                    (SELECT COUNT(*) FROM entities) as entity_count,
                    (SELECT COUNT(*) FROM relationships) as relationship_count,
                    (SELECT source_entity_id FROM relationships
                     GROUP BY source_entity_id
                     ORDER BY COUNT(*) DESC LIMIT 1) as most_connected
            """

            stats = await conn.fetchrow(stats_query)

            # Get relationship type distribution
            type_dist = await conn.fetch(
                """
                SELECT relationship_type, COUNT(*) as count
                FROM relationships
                GROUP BY relationship_type
                ORDER BY count DESC
                """
            )

            # Get entity type distribution
            entity_dist = await conn.fetch(
                """
                SELECT entity_type, COUNT(*) as count
                FROM entities
                GROUP BY entity_type
                ORDER BY count DESC
                """
            )

            # Calculate average relationships per entity
            avg_relationships = 0.0
            if stats['entity_count'] > 0:
                avg_relationships = (stats['relationship_count'] * 2) / stats['entity_count']

            # Get most connected entity details
            most_connected_entity = None
            if stats['most_connected']:
                entity_info = await conn.fetchrow(
                    """
                    SELECT entity_id, entity_name, entity_type,
                           (SELECT COUNT(*) FROM relationships
                            WHERE source_entity_id = $1 OR target_entity_id = $1) as connection_count
                    FROM entities
                    WHERE entity_id = $1
                    """,
                    stats['most_connected']
                )
                if entity_info:
                    most_connected_entity = {
                        'entity_id': str(entity_info['entity_id']),
                        'entity_name': entity_info['entity_name'],
                        'entity_type': entity_info['entity_type'],
                        'connection_count': entity_info['connection_count']
                    }

            return {
                'total_entities': stats['entity_count'],
                'total_relationships': stats['relationship_count'],
                'avg_relationships_per_entity': round(avg_relationships, 2),
                'most_connected_entity': most_connected_entity,
                'relationship_type_distribution': {
                    row['relationship_type']: row['count']
                    for row in type_dist
                },
                'entity_type_distribution': {
                    row['entity_type']: row['count']
                    for row in entity_dist
                }
            }

    async def calculate_pagerank(self) -> List[Dict[str, Any]]:
        """
        Calculate PageRank scores for all entities to find important concepts.

        Returns:
            List of entities with PageRank scores (sorted by importance)
        """
        async with self.db_pool.acquire() as conn:
            try:
                query = """
                    SELECT
                        e.entity_id,
                        e.entity_name,
                        e.entity_type,
                        pr.pagerank
                    FROM pgr_pageRank(
                        'SELECT id, source, target, cost FROM entity_edges',
                        directed := false
                    ) pr
                    JOIN entity_nodes en ON pr.node = en.node_id
                    JOIN entities e ON en.entity_id = e.entity_id
                    ORDER BY pr.pagerank DESC
                    LIMIT 50
                """

                rows = await conn.fetch(query)

                return [
                    {
                        'entity_id': str(row['entity_id']),
                        'entity_name': row['entity_name'],
                        'entity_type': row['entity_type'],
                        'pagerank_score': float(row['pagerank'])
                    }
                    for row in rows
                ]

            except Exception as e:
                logger.error(f"Error calculating PageRank: {e}")
                return []

    async def get_subgraph(
        self,
        entity_ids: List[UUID],
        include_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Get a subgraph containing specified entities and optionally intermediate connections.

        Args:
            entity_ids: List of entity UUIDs to include
            include_intermediate: Whether to include entities on paths between specified entities

        Returns:
            Dictionary with nodes and edges of the subgraph
        """
        async with self.db_pool.acquire() as conn:
            # Get direct relationships between specified entities
            direct_rels = await conn.fetch(
                """
                SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                       r.relationship_type, r.confidence,
                       e1.entity_name as source_name, e1.entity_type as source_type,
                       e2.entity_name as target_name, e2.entity_type as target_type
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id = e1.entity_id
                JOIN entities e2 ON r.target_entity_id = e2.entity_id
                WHERE r.source_entity_id = ANY($1::uuid[])
                  AND r.target_entity_id = ANY($1::uuid[])
                """,
                entity_ids
            )

            nodes_set = set(entity_ids)
            edges = []

            for rel in direct_rels:
                edges.append({
                    'relationship_id': str(rel['relationship_id']),
                    'source_entity_id': str(rel['source_entity_id']),
                    'target_entity_id': str(rel['target_entity_id']),
                    'relationship_type': rel['relationship_type'],
                    'confidence': float(rel['confidence']),
                    'source_name': rel['source_name'],
                    'target_name': rel['target_name']
                })

            # Get node details
            nodes = []
            for entity_id in nodes_set:
                entity_info = await conn.fetchrow(
                    """
                    SELECT entity_id, entity_name, entity_type, confidence
                    FROM entities
                    WHERE entity_id = $1
                    """,
                    entity_id
                )
                if entity_info:
                    nodes.append({
                        'entity_id': str(entity_info['entity_id']),
                        'entity_name': entity_info['entity_name'],
                        'entity_type': entity_info['entity_type'],
                        'confidence': float(entity_info['confidence'])
                    })

            return {
                'nodes': nodes,
                'edges': edges,
                'node_count': len(nodes),
                'edge_count': len(edges)
            }

    def _empty_path_response(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID
    ) -> Dict[str, Any]:
        """Create empty path response when no path found."""
        return {
            'source_entity_id': str(source_entity_id),
            'target_entity_id': str(target_entity_id),
            'path_found': False,
            'path_length': 0,
            'path_cost': 0.0,
            'path_entities': [],
            'path_relationships': []
        }
