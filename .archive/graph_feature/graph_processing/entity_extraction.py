"""
Entity extraction from text chunks using LLM-based extraction.
"""

import json
import logging
import re
from typing import List, Dict, Any, Sequence
from uuid import UUID
import asyncpg

import asyncio
from .entity_types import EntityType, ENTITY_TYPE_DESCRIPTIONS
from .llm_provider import LLMProvider
from .entity_cache import EntityCache

logger = logging.getLogger(__name__)


class EntityExtractionError(Exception):
    """Raised when LLM entity extraction fails."""


class EntityExtractor:
    """Extract entities from text chunks using an LLM provider."""

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        llm_provider: LLMProvider,
        embedding_model,
    ):
        """
        Initialize EntityExtractor with an LLM provider.
        
        Args:
            db_pool: Database connection pool
            llm_provider: LLM provider instance (e.g., GeminiLLMProvider)
            embedding_model: Embedding model for entity embeddings
        """
        self.db_pool = db_pool
        self.llm_provider = llm_provider
        self.embedding_model = embedding_model
        self.valid_entity_types = {et.value for et in EntityType}

    async def _call_llm_with_retry(self, prompt: str) -> Any:
        """
        Call LLM API with retry logic (delegated to provider).

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            LLM response object

        Raises:
            EntityExtractionError: If the LLM call fails
        """
        try:
            return await self.llm_provider.generate_content(prompt)
        except Exception as e:
            raise EntityExtractionError(f"LLM generation failed: {str(e)}") from e


    async def extract_entities_from_chunk(
        self,
        chunk_id: UUID,
        chunk_text: str,
        confidence_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from a text chunk using the LLM provider.

        Args:
            chunk_id: UUID of the source chunk
            chunk_text: Text content to extract entities from
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            List of extracted entities with metadata
        """
        cached_entities = EntityCache.get(chunk_text)
        if cached_entities is not None:
            stored_entities = await self._store_entities(chunk_id, cached_entities)
            logger.info(f"Used cached entities for chunk {chunk_id}: {len(stored_entities)} entities")
            return stored_entities

        prompt = self._create_extraction_prompt(chunk_text)

        try:
            response = await self._call_llm_with_retry(prompt)
            entities_text = self.llm_provider.extract_text_from_response(response)
            entities = self._parse_entities(entities_text, confidence_threshold)

            EntityCache.set(chunk_text, entities)

            stored_entities = await self._store_entities(chunk_id, entities)

            logger.info(f"Extracted {len(stored_entities)} entities from chunk {chunk_id}")
            return stored_entities

        except Exception as exc:
            message = self._format_model_error(exc)
            logger.error(f"Error extracting entities for chunk {chunk_id}: {message}")
            raise EntityExtractionError(message) from exc

    async def extract_entities_for_batch(
        self,
        chunk_payloads: Sequence[Dict[str, Any]],
        confidence_threshold: float = 0.6
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """
        Extract entities for multiple chunks in a single LLM request.

        Args:
            chunk_payloads: Sequence of {'chunk_id': UUID, 'chunk_text': str}
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            Dict mapping chunk_id to list of stored entity dicts
        """
        if not chunk_payloads:
            return {}

        prompt = self._create_batch_prompt(chunk_payloads)

        try:
            response = await self._call_llm_with_retry(prompt)
            batch_text = self.llm_provider.extract_text_from_response(response)
        except Exception as exc:
            message = self._format_model_error(exc)
            logger.error(f"Batch entity extraction failed: {message}")
            raise EntityExtractionError(message) from exc

        parsed_entities = self._parse_batch_entities(batch_text, confidence_threshold)
        stored_results: Dict[UUID, List[Dict[str, Any]]] = {}

        for payload in chunk_payloads:
            chunk_id = payload["chunk_id"]
            chunk_entities = parsed_entities.get(str(chunk_id), [])
            stored_results[chunk_id] = await self._store_entities(chunk_id, chunk_entities)

        logger.info(f"Batch extracted entities for {len(chunk_payloads)} chunks")
        return stored_results

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for entity extraction."""
        entity_types = "\n".join([
            f"- {et.value}: {desc}"
            for et, desc in ENTITY_TYPE_DESCRIPTIONS.items()
        ])

        prompt = f"""You are an expert in Machine Learning and Deep Learning. Extract key entities from the following text.

ENTITY TYPES:
{entity_types}

INSTRUCTIONS:
1. Identify all important ML/DL entities in the text
2. For each entity, provide:
   - name: The entity name (as it appears in text)
   - type: One of the entity types listed above
   - confidence: Your confidence score (0.0 to 1.0)
   - description: Brief description of the entity

3. Return ONLY valid JSON array format:
[
  {{"name": "ResNet", "type": "MODEL", "confidence": 0.95, "description": "Residual neural network"}},
  {{"name": "ImageNet", "type": "DATASET", "confidence": 0.9, "description": "Large-scale image dataset"}}
]

TEXT TO ANALYZE:
{text}

EXTRACTED ENTITIES (JSON only):"""

        return prompt

    def _create_batch_prompt(self, chunks: Sequence[Dict[str, Any]]) -> str:
        """Create prompt for batched entity extraction."""
        entity_types = "\n".join([
            f"- {et.value}: {desc}"
            for et, desc in ENTITY_TYPE_DESCRIPTIONS.items()
        ])

        chunk_sections = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk.get("chunk_text") or ""
            chunk_sections.append(
                f"CHUNK ID: {chunk_id}\nTEXT:\n{chunk_text}\n"
            )

        chunk_text_block = "\n".join(chunk_sections)
        prompt = f"""You are an expert ML/DL entity extractor. Analyze each chunk independently and extract entities using the schema below.

ENTITY TYPES:
{entity_types}

INSTRUCTIONS:
1. Process each chunk separately. Do not merge information between chunks.
2. For each chunk, return a JSON object with:
   - chunk_id: The chunk ID provided
   - entities: List of entities following the single chunk schema (name, type, confidence, description)
3. Respond ONLY with valid JSON array. Example:
[
  {{
    "chunk_id": "chunk-123",
    "entities": [{{"name": "ResNet", "type": "MODEL", "confidence": 0.95, "description": "CNN architecture"}}]
  }},
  {{
    "chunk_id": "chunk-456",
    "entities": []
  }}
]

CHUNKS:
{chunk_text_block}

JSON RESPONSE:
"""
        return prompt

    def _parse_entities(self, entities_text: str, confidence_threshold: float) -> List[Dict]:
        """Parse entities from LLM response."""
        raw_entities = self._extract_json_array(entities_text)
        return self._filter_entities(raw_entities, confidence_threshold)

    def _parse_batch_entities(
        self,
        batch_text: str,
        confidence_threshold: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Parse batched entities keyed by chunk ID."""
        raw_entries = self._extract_json_array(batch_text)
        chunk_entities: Dict[str, List[Dict[str, Any]]] = {}

        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            chunk_id = entry.get("chunk_id") or entry.get("chunkId")
            if not chunk_id:
                continue
            entities = entry.get("entities", [])
            chunk_entities[str(chunk_id)] = self._filter_entities(entities, confidence_threshold)

        return chunk_entities

    def _extract_json_array(self, text: str) -> List[Any]:
        """Extract JSON array from Gemini response."""
        if not text:
            return []

        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            logger.warning("Gemini response did not contain JSON array")
            return []

        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, list):
                return data
            logger.warning("Expected JSON array in Gemini response but received %s", type(data))
        except json.JSONDecodeError as exc:
            logger.error(f"Unable to decode Gemini response as JSON: {exc}")

        return []

    def _filter_entities(
        self,
        entities: List[Dict[str, Any]],
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter and normalize entity objects."""
        if not entities:
            return []

        valid_entities: List[Dict[str, Any]] = []
        for entity in entities:
            name = entity.get('name')
            if not name:
                continue

            try:
                confidence = float(entity.get('confidence', 0))
            except (TypeError, ValueError):
                confidence = 0.0

            if confidence < confidence_threshold:
                continue

            entity_type = str(entity.get('type', '')).upper()
            if entity_type not in self.valid_entity_types:
                continue

            valid_entities.append({
                'name': name,
                'type': entity_type,
                'confidence': confidence,
                'metadata': {
                    'description': entity.get('description', '')
                }
            })

        return valid_entities

    async def _store_entities(
        self,
        chunk_id: UUID,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Persist entities for a chunk and attach IDs."""
        stored_entities: List[Dict[str, Any]] = []
        for entity_data in entities:
            entity_id = await self._store_entity(
                chunk_id=chunk_id,
                entity_name=entity_data['name'],
                entity_type=entity_data['type'],
                confidence=entity_data['confidence'],
                metadata=entity_data.get('metadata', {})
            )
            stored_entities.append({
                'entity_id': entity_id,
                **entity_data
            })
        return stored_entities

    async def _store_entity(
        self,
        chunk_id: UUID,
        entity_name: str,
        entity_type: str,
        confidence: float,
        metadata: Dict
    ) -> UUID:
        """Store or update entity in database."""
        async with self.db_pool.acquire() as conn:
            # Generate embedding for entity name (offload to thread as it is CPU bound)
            embedding = await asyncio.to_thread(self.embedding_model.encode, entity_name)
            embedding = embedding.tolist()
            # Convert embedding to pgvector format string
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            # Check if entity already exists (by name and type)
            existing = await conn.fetchrow(
                """
                SELECT entity_id, source_chunk_ids
                FROM entities
                WHERE entity_name = $1 AND entity_type = $2
                """,
                entity_name, entity_type
            )

            if existing:
                # Update existing entity (add chunk reference, update confidence if higher)
                entity_id = existing['entity_id']
                chunk_ids = list(existing['source_chunk_ids']) if existing['source_chunk_ids'] else []
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

                await conn.execute(
                    """
                    UPDATE entities
                    SET confidence = GREATEST(confidence, $1),
                        source_chunk_ids = $2,
                        metadata = metadata || $3::jsonb,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE entity_id = $4
                    """,
                    confidence, chunk_ids, json.dumps(metadata), entity_id
                )
            else:
                # Insert new entity
                entity_id = await conn.fetchval(
                    """
                    INSERT INTO entities
                    (entity_name, entity_type, confidence, embedding, metadata, source_chunk_ids)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING entity_id
                    """,
                    entity_name, entity_type, confidence, embedding_str,
                    json.dumps(metadata), [chunk_id]
                )

            # Update chunk with entity reference
            await conn.execute(
                """
                UPDATE document_chunks
                SET entity_ids = array_append(entity_ids, $1)
                WHERE id = $2 AND NOT ($1 = ANY(entity_ids))
                """,
                entity_id, chunk_id
            )

            return entity_id

    def _format_model_error(self, error: Exception) -> str:
        """Provide human readable Gemini error message."""
        message = str(error)
        lower_message = message.lower()

        quota_terms = ("quota", "exceed", "exhaust", "rate limit", "429")
        if any(term in lower_message for term in quota_terms):
            return f"Gemini quota or rate limit exceeded: {message}"

        return f"Gemini entity extraction failed: {message}"

    async def get_entities_by_chunk(self, chunk_id: UUID) -> List[Dict]:
        """Get all entities extracted from a chunk."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id, entity_name, entity_type, confidence, metadata
                FROM entities
                WHERE $1 = ANY(source_chunk_ids)
                ORDER BY confidence DESC
                """,
                chunk_id
            )
            return [dict(row) for row in rows]

    async def get_entity_by_id(self, entity_id: UUID) -> Dict:
        """Get entity details by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM entities WHERE entity_id = $1
                """,
                entity_id
            )
            return dict(row) if row else None

    async def search_entities(
        self,
        query: str,
        entity_type: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """Search entities by name or embedding similarity."""
        async with self.db_pool.acquire() as conn:
            # Generate query embedding (offload to thread)
            embedding_array = await asyncio.to_thread(self.embedding_model.encode, query)
            query_embedding = embedding_array.tolist()

            # Search by embedding similarity
            if entity_type:
                rows = await conn.fetch(
                    """
                    SELECT entity_id, entity_name, entity_type, confidence,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM entities
                    WHERE entity_type = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_embedding, entity_type, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT entity_id, entity_name, entity_type, confidence,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM entities
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    query_embedding, limit
                )

            return [dict(row) for row in rows]
