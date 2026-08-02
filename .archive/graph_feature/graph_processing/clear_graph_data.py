#!/usr/bin/env python3
"""
Utility script to clear entities and relationships from the graph database.

Usage:
    python scripts/clear_graph_data.py                    # Clear all graph data
    python scripts/clear_graph_data.py --document-id <id> # Clear data for specific document
    python scripts/clear_graph_data.py --confirm           # Skip confirmation prompt
"""

import asyncio
import asyncpg
import os
import sys
from uuid import UUID
import argparse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def get_db_pool() -> asyncpg.Pool:
    """Create database connection pool."""
    return await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "raguser"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "ragdb"),
        min_size=1,
        max_size=5
    )


async def get_graph_stats(pool: asyncpg.Pool) -> dict:
    """Get current graph statistics."""
    async with pool.acquire() as conn:
        entity_count = await conn.fetchval("SELECT COUNT(*) FROM entities")
        relationship_count = await conn.fetchval("SELECT COUNT(*) FROM relationships")
        entity_node_count = await conn.fetchval("SELECT COUNT(*) FROM entity_nodes")
        entity_edge_count = await conn.fetchval("SELECT COUNT(*) FROM entity_edges")

        # Get entity type distribution
        entity_types = await conn.fetch("""
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
            LIMIT 10
        """)

    return {
        'entities': entity_count,
        'relationships': relationship_count,
        'entity_nodes': entity_node_count,
        'entity_edges': entity_edge_count,
        'top_entity_types': entity_types
    }


async def clear_all_graph_data(pool: asyncpg.Pool, confirm: bool = True) -> dict:
    """
    Clear all entities and relationships from the graph.

    Args:
        pool: Database connection pool
        confirm: If True, ask for user confirmation

    Returns:
        Dict with deletion results
    """
    # Get stats before deletion
    stats_before = await get_graph_stats(pool)

    print("\n📊 Current Graph Statistics:")
    print(f"   Entities: {stats_before['entities']:,}")
    print(f"   Relationships: {stats_before['relationships']:,}")
    print(f"   Entity Nodes (pgRouting): {stats_before['entity_nodes']:,}")
    print(f"   Entity Edges (pgRouting): {stats_before['entity_edges']:,}")

    if stats_before['top_entity_types']:
        print("\n   Top Entity Types:")
        for row in stats_before['top_entity_types']:
            print(f"     - {row['entity_type']}: {row['count']:,}")

    if confirm and stats_before['entities'] > 0:
        print("\n⚠️  WARNING: This will permanently delete ALL graph data!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled.")
            return {'deleted': False, 'stats_before': stats_before}

    # Delete all graph data
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE entity_edges CASCADE")
        await conn.execute("TRUNCATE TABLE entity_nodes CASCADE")
        await conn.execute("TRUNCATE TABLE relationships CASCADE")
        await conn.execute("TRUNCATE TABLE entities CASCADE")

    # Get stats after deletion
    stats_after = await get_graph_stats(pool)

    print("\n✅ Graph data cleared successfully!")
    print(f"   Deleted {stats_before['entities']:,} entities")
    print(f"   Deleted {stats_before['relationships']:,} relationships")
    print(f"   Deleted {stats_before['entity_nodes']:,} entity nodes")
    print(f"   Deleted {stats_before['entity_edges']:,} entity edges")

    return {
        'deleted': True,
        'stats_before': stats_before,
        'stats_after': stats_after
    }


async def clear_document_graph_data(
    pool: asyncpg.Pool,
    document_id: UUID,
    confirm: bool = True
) -> dict:
    """
    Clear entities and relationships for a specific document.

    Args:
        pool: Database connection pool
        document_id: UUID of the document
        confirm: If True, ask for user confirmation

    Returns:
        Dict with deletion results
    """
    async with pool.acquire() as conn:
        # Get chunk IDs for this document
        chunks = await conn.fetch(
            "SELECT id FROM document_chunks WHERE document_id = $1",
            str(document_id)
        )
        chunk_ids = [row['id'] for row in chunks]

        if not chunk_ids:
            print(f"❌ No chunks found for document {document_id}")
            return {'deleted': False, 'reason': 'No chunks found'}

        # Count entities and relationships to be deleted
        entity_count = await conn.fetchval("""
            SELECT COUNT(*) FROM entities
            WHERE source_chunk_ids && $1::uuid[]
        """, chunk_ids)

        relationship_count = await conn.fetchval("""
            SELECT COUNT(*) FROM relationships
            WHERE source_entity_id IN (
                SELECT entity_id FROM entities
                WHERE source_chunk_ids && $1::uuid[]
            )
        """, chunk_ids)

        print(f"\n📊 Document {document_id} Graph Statistics:")
        print(f"   Chunks: {len(chunk_ids):,}")
        print(f"   Entities: {entity_count:,}")
        print(f"   Relationships: {relationship_count:,}")

        if confirm and entity_count > 0:
            print(f"\n⚠️  WARNING: This will delete graph data for document {document_id}")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Cancelled.")
                return {'deleted': False, 'reason': 'User cancelled'}

        # Delete relationships first (foreign key constraint)
        deleted_rels = await conn.execute("""
            DELETE FROM relationships
            WHERE source_entity_id IN (
                SELECT entity_id FROM entities
                WHERE source_chunk_ids && $1::uuid[]
            )
        """, chunk_ids)

        # Delete entities
        deleted_entities = await conn.execute("""
            DELETE FROM entities
            WHERE source_chunk_ids && $1::uuid[]
        """, chunk_ids)

        print(f"\n✅ Document graph data cleared!")
        print(f"   Deleted {entity_count:,} entities")
        print(f"   Deleted {relationship_count:,} relationships")

        return {
            'deleted': True,
            'document_id': str(document_id),
            'entities_deleted': entity_count,
            'relationships_deleted': relationship_count
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clear entities and relationships from the graph database'
    )
    parser.add_argument(
        '--document-id',
        type=str,
        help='Clear graph data for specific document only'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics, do not delete anything'
    )

    args = parser.parse_args()

    print("\n🗑️  Graph Data Cleanup Utility")
    print("=" * 50)

    try:
        pool = await get_db_pool()

        if args.stats_only:
            stats = await get_graph_stats(pool)
            print("\n📊 Current Graph Statistics:")
            print(f"   Entities: {stats['entities']:,}")
            print(f"   Relationships: {stats['relationships']:,}")
            print(f"   Entity Nodes: {stats['entity_nodes']:,}")
            print(f"   Entity Edges: {stats['entity_edges']:,}")
            if stats['top_entity_types']:
                print("\n   Top Entity Types:")
                for row in stats['top_entity_types']:
                    print(f"     - {row['entity_type']}: {row['count']:,}")

        elif args.document_id:
            try:
                document_id = UUID(args.document_id)
                await clear_document_graph_data(
                    pool,
                    document_id,
                    confirm=not args.confirm
                )
            except ValueError:
                print(f"❌ Invalid document ID: {args.document_id}")
                sys.exit(1)

        else:
            await clear_all_graph_data(pool, confirm=not args.confirm)

        await pool.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
