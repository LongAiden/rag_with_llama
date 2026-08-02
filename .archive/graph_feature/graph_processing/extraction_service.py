"""
Service for extracting entities and relationships from document chunks.

This module provides functionality to automatically extract knowledge graph
data when documents are uploaded and chunked.

Features:
- Async non-blocking LLM calls with semaphore-based concurrency control
- Adaptive rate limiting to respect LLM API quotas
- Inter-batch delays to prevent rate limit hits
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Optional, Any
from uuid import UUID
from sentence_transformers import SentenceTransformer

from config.graph_config import get_graph_config
from graph_processing.entity_extraction import EntityExtractor, EntityExtractionError
from graph_processing.relationship_extraction import RelationshipExtractor
from graph_processing.ollama_provider import OllamaLLMProvider
from repositories.table_repository import quote_ident

logger = logging.getLogger(__name__)


# Default concurrency limit for parallel API calls
DEFAULT_MAX_CONCURRENT_CALLS = 2

# Default delay between batches (seconds) to respect rate limits
DEFAULT_INTER_BATCH_DELAY = 1.0


class ExtractionService:
    """Service for extracting entities and relationships from chunks."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        llm_provider: OllamaLLMProvider,
        embedding_model: SentenceTransformer,
        entity_threshold: float = 0.6,
        relationship_threshold: float = 0.6,
        table_name: str = "document_chunks",
        batch_size: int = 10,
        max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
        inter_batch_delay: float = DEFAULT_INTER_BATCH_DELAY
    ):
        self.pool = pool

        self.entity_extractor = EntityExtractor(pool, llm_provider, embedding_model)
        self.rel_extractor = RelationshipExtractor(pool, llm_provider)
        self.entity_threshold = entity_threshold
        self.relationship_threshold = relationship_threshold
        self.table_name = table_name
        self.batch_size = max(1, batch_size)
        self.max_concurrent_calls = max(1, max_concurrent_calls)
        self.inter_batch_delay = max(0.0, inter_batch_delay)

        # Semaphore for controlling concurrent API calls
        self._api_semaphore = asyncio.Semaphore(self.max_concurrent_calls)

    @property
    def safe_table_name(self) -> str:
        return quote_ident(self.table_name)

    async def extract_from_chunk(
        self,
        chunk_id: UUID,
        chunk_text: str
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from a single chunk.

        Args:
            chunk_id: UUID of the chunk
            chunk_text: Text content of the chunk

        Returns:
            Dict with extraction results: {
                'success': bool,
                'entities_count': int,
                'relationships_count': int,
                'error': Optional[str]
            }
        """
        try:
            entities = await self.entity_extractor.extract_entities_from_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                confidence_threshold=self.entity_threshold
            )

            relationships = []
            if len(entities) >= 2:
                relationships = await self.rel_extractor.extract_relationships_from_chunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    entities=entities,
                    confidence_threshold=self.relationship_threshold
                )

            return {
                'success': True,
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'error': None
            }

        except EntityExtractionError as exc:
            return {
                'success': False,
                'entities_count': 0,
                'relationships_count': 0,
                'error': str(exc)
            }
        except Exception as exc:
            return {
                'success': False,
                'entities_count': 0,
                'relationships_count': 0,
                'error': str(exc)
            }

    async def _process_batch_with_semaphore(
        self,
        batch: List[Dict[str, Any]],
        batch_index: int,
        total_batches: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Process a single batch with semaphore-controlled concurrency.

        Args:
            batch: List of chunk dicts with 'id' and 'text'
            batch_index: Index of this batch (0-indexed)
            total_batches: Total number of batches
            verbose: Whether to print progress

        Returns:
            Dict with batch results
        """
        async with self._api_semaphore:
            chunk_payloads = [
                {'chunk_id': chunk['id'], 'chunk_text': chunk['text']}
                for chunk in batch
            ]

            batch_entities = 0
            batch_relationships = 0
            batch_successful = 0

            try:
                # Extract entities for this batch
                batch_results = await self.entity_extractor.extract_entities_for_batch(
                    chunk_payloads=chunk_payloads,
                    confidence_threshold=self.entity_threshold
                )

                # Prepare relationship extraction payloads
                relationship_payloads = []
                for chunk in batch:
                    chunk_id = chunk['id']
                    chunk_text = chunk['text']
                    entities = batch_results.get(chunk_id, [])
                    batch_entities += len(entities)

                    if len(entities) >= 2:
                        relationship_payloads.append({
                            'chunk_id': chunk_id,
                            'chunk_text': chunk_text,
                            'entities': entities
                        })

                # Extract relationships for this batch
                relationship_results = {}
                if relationship_payloads:
                    try:
                        relationship_results = await self.rel_extractor.extract_relationships_for_batch(
                            chunk_payloads=relationship_payloads,
                            confidence_threshold=self.relationship_threshold
                        )
                    except Exception as exc:
                        logger.error(f"Batch {batch_index + 1} relationship extraction failed: {exc}")

                # Count relationships
                for chunk in batch:
                    chunk_id = chunk['id']
                    relationships = relationship_results.get(chunk_id, [])
                    batch_relationships += len(relationships)
                    batch_successful += 1

                if verbose:
                    print(f"  [Batch {batch_index + 1}/{total_batches}] "
                          f"✓ {batch_entities} entities, {batch_relationships} relationships")

                return {
                    'successful': batch_successful,
                    'failed': 0,
                    'entities': batch_entities,
                    'relationships': batch_relationships,
                    'error': None
                }

            except EntityExtractionError as exc:
                # Propagate rate limit errors
                raise
            except Exception as exc:
                logger.error(f"Batch {batch_index + 1} processing failed: {exc}")
                return {
                    'successful': 0,
                    'failed': len(batch),
                    'entities': 0,
                    'relationships': 0,
                    'error': str(exc)
                }

    async def extract_from_chunks(
        self,
        chunk_ids: List[UUID],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from multiple chunks.

        Features:
        - Async non-blocking processing with semaphore-controlled concurrency
        - Inter-batch delays to respect Gemini API rate limits
        - Parallel batch processing when max_concurrent_calls > 1

        Args:
            chunk_ids: List of chunk UUIDs to process
            verbose: If True, print progress information

        Returns:
            Dict with summary: {
                'total_chunks': int,
                'successful': int,
                'failed': int,
                'total_entities': int,
                'total_relationships': int
            }
        """
        total_chunks = len(chunk_ids)
        if total_chunks == 0:
            return {
                'total_chunks': 0,
                'successful': 0,
                'failed': 0,
                'total_entities': 0,
                'total_relationships': 0
            }

        total_entities = 0
        total_relationships = 0
        successful = 0
        failed = 0

        if verbose:
            print(f"Extracting from {total_chunks} chunks "
                  f"(batch_size={self.batch_size}, concurrency={self.max_concurrent_calls})...")

        # Fetch all chunk texts at once
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, text FROM {self.safe_table_name} WHERE id = ANY($1::text[])",
                chunk_ids
            )

        # Preserve requested order and note missing chunks
        chunk_lookup = {
            str(row['id']): {'id': row['id'], 'text': row['text']}
            for row in rows
        }

        ordered_chunks: List[Dict[str, Any]] = []
        for requested_id in chunk_ids:
            chunk_data = chunk_lookup.get(str(requested_id))
            if chunk_data:
                ordered_chunks.append(chunk_data)
            else:
                failed += 1
                if verbose:
                    print(f"  [!] Chunk {requested_id} not found in table {self.table_name}")

        total_available = len(ordered_chunks)
        if total_available == 0:
            return {
                'total_chunks': total_chunks,
                'successful': successful,
                'failed': failed,
                'total_entities': total_entities,
                'total_relationships': total_relationships
            }

        # Create batches
        batches = []
        for start in range(0, total_available, self.batch_size):
            batches.append(ordered_chunks[start:start + self.batch_size])

        total_batches = len(batches)

        # Process batches with controlled concurrency
        # Use gather with inter-batch delays to respect rate limits
        batch_tasks = []
        for i, batch in enumerate(batches):
            # Add inter-batch delay to prevent rate limit hits
            if i > 0 and self.inter_batch_delay > 0:
                await asyncio.sleep(self.inter_batch_delay)

            task = self._process_batch_with_semaphore(
                batch=batch,
                batch_index=i,
                total_batches=total_batches,
                verbose=verbose
            )
            batch_tasks.append(task)

            # If we've queued up max_concurrent_calls tasks, await them
            if len(batch_tasks) >= self.max_concurrent_calls:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        if isinstance(result, EntityExtractionError):
                            raise result
                        logger.error(f"Batch failed with exception: {result}")
                        failed += self.batch_size
                    else:
                        successful += result['successful']
                        failed += result['failed']
                        total_entities += result['entities']
                        total_relationships += result['relationships']
                batch_tasks = []

        # Process remaining tasks
        if batch_tasks:
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    if isinstance(result, EntityExtractionError):
                        raise result
                    logger.error(f"Batch failed with exception: {result}")
                    failed += self.batch_size
                else:
                    successful += result['successful']
                    failed += result['failed']
                    total_entities += result['entities']
                    total_relationships += result['relationships']

        return {
            'total_chunks': total_chunks,
            'successful': successful,
            'failed': failed,
            'total_entities': total_entities,
            'total_relationships': total_relationships
        }

    async def extract_from_document(
        self,
        document_id: UUID,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Extract entities from all chunks of a document.

        Args:
            document_id: UUID of the document
            verbose: If True, print progress information

        Returns:
            Extraction summary dict
        """
        # Get all chunk IDs for this document
        async with self.pool.acquire() as conn:
            chunks = await conn.fetch(
                f"SELECT id FROM {self.safe_table_name} WHERE document_id = $1",
                str(document_id)
            )
            chunk_ids = [row['id'] for row in chunks]

        if not chunk_ids:
            return {
                'total_chunks': 0,
                'successful': 0,
                'failed': 0,
                'total_entities': 0,
                'total_relationships': 0
            }

        return await self.extract_from_chunks(chunk_ids, verbose=verbose)


async def create_extraction_service(
    pool: asyncpg.Pool,
    entity_threshold: Optional[float] = None,
    relationship_threshold: Optional[float] = None,
    table_name: str = "document_chunks",
    max_concurrent_calls: Optional[int] = None,
    inter_batch_delay: Optional[float] = None
) -> ExtractionService:
    """
    Factory function to create ExtractionService with default config.

    Args:
        pool: Database connection pool
        entity_threshold: Override default entity confidence threshold
        relationship_threshold: Override default relationship confidence threshold
        table_name: Name of the chunks table (default: document_chunks)
        max_concurrent_calls: Maximum concurrent API calls (default: 2)
        inter_batch_delay: Delay between batches in seconds (default: 1.0)

    Returns:
        Configured ExtractionService instance
    """
    config = get_graph_config()

    if config.llm_provider == "gemini":
        from graph_processing.gemini_provider import GeminiLLMProvider
        llm_provider = GeminiLLMProvider(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
        )
    else:
        llm_provider = OllamaLLMProvider(
            base_url=config.ollama_base_url,
            model_name=config.ollama_model,
        )

    entity_threshold = entity_threshold or config.entity_confidence_threshold
    relationship_threshold = relationship_threshold or config.relationship_confidence_threshold
    batch_size = config.batch_size
    max_concurrent = max_concurrent_calls or config.max_concurrent_api_calls
    batch_delay = inter_batch_delay if inter_batch_delay is not None else config.inter_batch_delay

    embedding_model = SentenceTransformer(config.entity_embedding_model)

    return ExtractionService(
        pool=pool,
        llm_provider=llm_provider,
        embedding_model=embedding_model,
        entity_threshold=entity_threshold,
        relationship_threshold=relationship_threshold,
        table_name=table_name,
        batch_size=batch_size,
        max_concurrent_calls=max_concurrent,
        inter_batch_delay=batch_delay
    )
