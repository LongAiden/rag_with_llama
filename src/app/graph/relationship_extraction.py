"""
Relationship extraction between entities using LLM.
"""

import logging
import logfire
import json
from typing import List, Dict, Any
from uuid import UUID
import asyncpg

from .entity_types import RelationshipType, RELATIONSHIP_TYPE_DESCRIPTIONS
from .json_utils import JSONParser
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extract relationships between entities using an LLM provider."""

    def __init__(self, db_pool: asyncpg.Pool, llm_provider: LLMProvider):
        self.db_pool = db_pool
        self.llm_provider = llm_provider

    async def extract_relationships_from_chunk(
        self,
        chunk_id: UUID,
        chunk_text: str,
        entities: List[Dict[str, Any]],
        confidence_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in a chunk.

        Args:
            chunk_id: UUID of the source chunk
            chunk_text: Text content
            entities: List of entities found in the chunk
            confidence_threshold: Minimum confidence score

        Returns:
            List of extracted relationships
        """
        if len(entities) < 2:
            logger.info(f"Not enough entities in chunk {chunk_id} for relationship extraction")
            return []

        # Create extraction prompt
        prompt = self._create_extraction_prompt(chunk_text, entities)

        try:
            response = await self.llm_provider.generate_content(prompt)
            relationships_text = self.llm_provider.extract_text_from_response(response)

            # Parse the response
            relationships = self._parse_relationships(
                relationships_text,
                entities,
                confidence_threshold
            )

            # Store relationships in database
            stored_relationships = []
            for rel_data in relationships:
                relationship_id = await self._store_relationship(
                    chunk_id=chunk_id,
                    source_entity_id=rel_data['source_entity_id'],
                    target_entity_id=rel_data['target_entity_id'],
                    relationship_type=rel_data['type'],
                    confidence=rel_data['confidence'],
                    metadata=rel_data.get('metadata', {})
                )
                stored_relationships.append({
                    'relationship_id': relationship_id,
                    **rel_data
                })

            logger.info(f"Extracted {len(stored_relationships)} relationships from chunk {chunk_id}")
            return stored_relationships

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    async def extract_relationships_for_batch(
        self,
        chunk_payloads: List[Dict[str, Any]],
        confidence_threshold: float = 0.6
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """
        Extract relationships for multiple chunks in a single Gemini request.

        Args:
            chunk_payloads: List of {
                'chunk_id': UUID,
                'chunk_text': str,
                'entities': List[Dict]
            }
            confidence_threshold: Minimum confidence score

        Returns:
            Dict mapping chunk_id to list of stored relationship dicts
        """
        if not chunk_payloads:
            return {}

        # Filter chunks with >= 2 entities
        valid_payloads = [
            p for p in chunk_payloads
            if len(p.get('entities', [])) >= 2
        ]

        if not valid_payloads:
            logger.info("No chunks with sufficient entities for batch relationship extraction")
            return {p['chunk_id']: [] for p in chunk_payloads}

        # Create batch prompt
        prompt = self._create_batch_extraction_prompt(valid_payloads)

        try:
            with logfire.span("batch_relationship_extraction",
                            num_chunks=len(valid_payloads)):
                response = await self.llm_provider.generate_content(prompt)
                batch_text = self.llm_provider.extract_text_from_response(response)

                logfire.info("LLM response received",
                           response_length=len(batch_text),
                           num_chunks=len(valid_payloads))

                # Parse batch response
                parsed_relationships = self._parse_batch_relationships(
                    batch_text,
                    valid_payloads,
                    confidence_threshold
                )

            # Store relationships for each chunk
            stored_results: Dict[UUID, List[Dict[str, Any]]] = {}

            for payload in chunk_payloads:
                chunk_id = payload['chunk_id']
                relationships = parsed_relationships.get(str(chunk_id), [])

                stored_rels = []
                for rel_data in relationships:
                    relationship_id = await self._store_relationship(
                        chunk_id=chunk_id,
                        source_entity_id=rel_data['source_entity_id'],
                        target_entity_id=rel_data['target_entity_id'],
                        relationship_type=rel_data['type'],
                        confidence=rel_data['confidence'],
                        metadata=rel_data.get('metadata', {})
                    )
                    stored_rels.append({
                        'relationship_id': relationship_id,
                        **rel_data
                    })

                stored_results[chunk_id] = stored_rels

            logger.info(f"Batch extracted relationships for {len(valid_payloads)} chunks")
            return stored_results

        except Exception as e:
            logger.error(f"Error in batch relationship extraction: {e}")
            # Return empty results for all chunks
            return {p['chunk_id']: [] for p in chunk_payloads}

    def _create_extraction_prompt(self, text: str, entities: List[Dict]) -> str:
        """Create prompt for relationship extraction."""
        relationship_types = "\n".join([
            f"- {rt.value}: {desc}"
            for rt, desc in RELATIONSHIP_TYPE_DESCRIPTIONS.items()
        ])

        # Format entities for the prompt
        entity_list = "\n".join([
            f"- {e['name']} ({e['type']}, ID: {e.get('entity_id', 'N/A')})"
            for e in entities
        ])

        prompt = f"""Extract ML/DL entity relationships as valid JSON only.

RELATIONSHIP TYPES:
{relationship_types}

ENTITIES:
{entity_list}

CRITICAL JSON RULES:
- Return ONLY JSON array starting with [ ending with ]
- NO explanations, NO markdown, NO extra text
- Use double quotes for strings
- Add commas between objects
- Keep descriptions under 50 characters
- No brackets in descriptions

REQUIRED FORMAT:
[
  {{"source": "ResNet", "target": "ImageNet", "type": "TRAINED_ON", "confidence": 0.9, "description": "ResNet trained on ImageNet"}},
  {{"source": "Dropout", "target": "Overfitting", "type": "MITIGATES", "confidence": 0.85, "description": "Dropout prevents overfitting"}}
]

TEXT:
{text}

JSON ARRAY (no other text):
```json
"""

        return prompt

    def _create_batch_extraction_prompt(
        self,
        chunk_payloads: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for batched relationship extraction."""
        relationship_types = "\n".join([
            f"- {rt.value}: {desc}"
            for rt, desc in RELATIONSHIP_TYPE_DESCRIPTIONS.items()
        ])

        # Create sections for each chunk
        chunk_sections = []
        for payload in chunk_payloads:
            chunk_id = payload['chunk_id']
            chunk_text = payload.get('chunk_text', '')
            entities = payload.get('entities', [])

            # Format entities for this chunk
            entity_list = "\n".join([
                f"  - {e['name']} ({e['type']}, ID: {e.get('entity_id', 'N/A')})"
                for e in entities
            ])

            chunk_sections.append(
                f"CHUNK ID: {chunk_id}\n"
                f"ENTITIES:\n{entity_list}\n"
                f"TEXT:\n{chunk_text}\n"
            )

        chunk_text_block = "\n".join(chunk_sections)

        prompt = f"""You are a JSON generator. Extract ML/DL relationships and return ONLY valid JSON.

RELATIONSHIP TYPES:
{relationship_types}

CRITICAL RULES - VALID JSON ONLY:
1. Return ONLY a JSON array, NO explanations, NO markdown, NO extra text
2. Use double quotes for ALL strings
3. Add commas between ALL objects and fields
4. Do NOT use brackets [ ] inside description strings
5. Keep descriptions under 50 characters to avoid errors
6. If unsure, return empty relationships array for that chunk

EXACT FORMAT REQUIRED:
[
  {{
    "chunk_id": "chunk-123",
    "relationships": [
      {{"source": "ResNet", "target": "ImageNet", "type": "TRAINED_ON", "confidence": 0.9, "description": "ResNet trained on ImageNet"}}
    ]
  }}
]

VALIDATION CHECKLIST:
✓ Starts with [ and ends with ]
✓ All strings use double quotes "
✓ Commas between all objects
✓ No trailing commas before }} or ]]
✓ No explanatory text before or after JSON

CHUNKS TO PROCESS:
{chunk_text_block}

RESPOND WITH VALID JSON ARRAY ONLY (no other text):
```json
"""
        return prompt

    def _parse_batch_relationships(
        self,
        batch_text: str,
        chunk_payloads: List[Dict[str, Any]],
        confidence_threshold: float
    ) -> Dict[str, List[Dict]]:
        """
        Parse batched relationship extraction response.

        Returns:
            Dict mapping chunk_id (as string) to list of relationship dicts
        """
        try:
            # Use robust JSON parser with multiple fallback strategies
            batch_data = JSONParser.parse_with_fallback(
                text=batch_text,
                expected_type="array",
                context="batch_relationships"
            )

            if not batch_data:
                logfire.warn("No relationships found in batch response",
                           response_length=len(batch_text))
                return {}

            logfire.info("Batch relationships parsed successfully",
                       num_chunks_in_response=len(batch_data) if isinstance(batch_data, list) else 0)

            # Create entity maps for each chunk
            chunk_entity_maps = {}
            for payload in chunk_payloads:
                chunk_id = str(payload['chunk_id'])
                entities = payload.get('entities', [])
                chunk_entity_maps[chunk_id] = {
                    e['name']: e.get('entity_id') for e in entities
                }

            # Parse relationships for each chunk
            results = {}
            for chunk_result in batch_data:
                chunk_id = str(chunk_result.get('chunk_id', ''))
                relationships = chunk_result.get('relationships', [])

                if chunk_id not in chunk_entity_maps:
                    continue

                entity_map = chunk_entity_maps[chunk_id]
                valid_relationships = []

                for rel in relationships:
                    confidence = rel.get('confidence', 0)
                    if confidence < confidence_threshold:
                        continue

                    # Validate relationship type
                    rel_type = rel.get('type', '').upper()
                    if rel_type not in [rt.value for rt in RelationshipType]:
                        continue

                    source_name = rel.get('source')
                    target_name = rel.get('target')

                    # Check if entities exist in this chunk's entity map
                    if source_name in entity_map and target_name in entity_map:
                        source_id = entity_map[source_name]
                        target_id = entity_map[target_name]

                        if source_id and target_id and source_id != target_id:
                            valid_relationships.append({
                                'source_entity_id': source_id,
                                'target_entity_id': target_id,
                                'type': rel_type,
                                'confidence': confidence,
                                'metadata': {
                                    'description': rel.get('description', ''),
                                    'source_name': source_name,
                                    'target_name': target_name
                                }
                            })

                results[chunk_id] = valid_relationships

            logfire.info("Batch relationship parsing completed",
                        chunks_with_relationships=len([r for r in results.values() if r]),
                        total_relationships=sum(len(r) for r in results.values()))

            return results

        except Exception as e:
            logfire.error("Error parsing batch relationships",
                        error=str(e),
                        error_type=type(e).__name__)
            logger.error(f"Error parsing batch relationships: {e}")
            return {}

    def _parse_relationships(
        self,
        relationships_text: str,
        entities: List[Dict],
        confidence_threshold: float
    ) -> List[Dict]:
        """Parse relationships from LLM response."""
        try:
            # Create entity name to ID mapping
            entity_map = {e['name']: e.get('entity_id') for e in entities}

            # Use robust JSON parser
            relationships = JSONParser.parse_with_fallback(
                text=relationships_text,
                expected_type="array",
                context="single_chunk_relationships"
            )

            if relationships:

                # Filter and validate relationships
                valid_relationships = []
                for rel in relationships:
                    if rel.get('confidence', 0) >= confidence_threshold:
                        # Validate relationship type
                        rel_type = rel.get('type', '').upper()
                        if rel_type in [rt.value for rt in RelationshipType]:
                            source_name = rel.get('source')
                            target_name = rel.get('target')

                            # Check if entities exist
                            if source_name in entity_map and target_name in entity_map:
                                source_id = entity_map[source_name]
                                target_id = entity_map[target_name]

                                if source_id and target_id and source_id != target_id:
                                    valid_relationships.append({
                                        'source_entity_id': source_id,
                                        'target_entity_id': target_id,
                                        'type': rel_type,
                                        'confidence': rel['confidence'],
                                        'metadata': {
                                            'description': rel.get('description', ''),
                                            'source_name': source_name,
                                            'target_name': target_name
                                        }
                                    })

                return valid_relationships

        except Exception as e:
            logger.error(f"Error parsing relationships: {e}")

        return []

    async def _store_relationship(
        self,
        chunk_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        relationship_type: str,
        confidence: float,
        metadata: Dict,
        weight: float = 1.0
    ) -> UUID:
        """Store or update relationship in database."""
        async with self.db_pool.acquire() as conn:
            # Check if relationship already exists
            existing = await conn.fetchrow(
                """
                SELECT relationship_id, source_chunk_ids
                FROM relationships
                WHERE source_entity_id = $1
                  AND target_entity_id = $2
                  AND relationship_type = $3
                """,
                source_entity_id, target_entity_id, relationship_type
            )

            if existing:
                # Update existing relationship
                relationship_id = existing['relationship_id']
                chunk_ids = list(existing['source_chunk_ids']) if existing['source_chunk_ids'] else []
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

                await conn.execute(
                    """
                    UPDATE relationships
                    SET confidence = GREATEST(confidence, $1),
                        source_chunk_ids = $2,
                        metadata = metadata || $3::jsonb,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE relationship_id = $4
                    """,
                    confidence, chunk_ids, json.dumps(metadata), relationship_id
                )
            else:
                # Insert new relationship (trigger will auto-create edge)
                relationship_id = await conn.fetchval(
                    """
                    INSERT INTO relationships
                    (source_entity_id, target_entity_id, relationship_type,
                     confidence, weight, metadata, source_chunk_ids)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING relationship_id
                    """,
                    source_entity_id, target_entity_id, relationship_type,
                    confidence, weight, json.dumps(metadata), [chunk_id]
                )

            return relationship_id

    async def get_relationships_by_chunk(self, chunk_id: UUID) -> List[Dict]:
        """Get all relationships from a chunk."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                       r.relationship_type, r.confidence, r.metadata,
                       e1.entity_name as source_name, e1.entity_type as source_type,
                       e2.entity_name as target_name, e2.entity_type as target_type
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id = e1.entity_id
                JOIN entities e2 ON r.target_entity_id = e2.entity_id
                WHERE $1 = ANY(r.source_chunk_ids)
                ORDER BY r.confidence DESC
                """,
                chunk_id
            )
            return [dict(row) for row in rows]

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        relationship_type: str = None
    ) -> List[Dict]:
        """Get all relationships for an entity."""
        async with self.db_pool.acquire() as conn:
            if relationship_type:
                rows = await conn.fetch(
                    """
                    SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                           r.relationship_type, r.confidence,
                           e1.entity_name as source_name,
                           e2.entity_name as target_name
                    FROM relationships r
                    JOIN entities e1 ON r.source_entity_id = e1.entity_id
                    JOIN entities e2 ON r.target_entity_id = e2.entity_id
                    WHERE (r.source_entity_id = $1 OR r.target_entity_id = $1)
                      AND r.relationship_type = $2
                    ORDER BY r.confidence DESC
                    """,
                    entity_id, relationship_type
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                           r.relationship_type, r.confidence,
                           e1.entity_name as source_name,
                           e2.entity_name as target_name
                    FROM relationships r
                    JOIN entities e1 ON r.source_entity_id = e1.entity_id
                    JOIN entities e2 ON r.target_entity_id = e2.entity_id
                    WHERE r.source_entity_id = $1 OR r.target_entity_id = $1
                    ORDER BY r.confidence DESC
                    """,
                    entity_id
                )

            return [dict(row) for row in rows]
