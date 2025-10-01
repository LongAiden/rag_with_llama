import os
import sys
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logfire
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from file_validator import FileValidator, FileValidationConfig
from embed_chunks_to_db import ChunkEmbeddingPipeline
from reranker import Reranker
from templates import (
    HOME_PAGE_HTML,
    SEARCH_RESULTS_HTML,
    SEARCH_ERROR_HTML,
    STATS_PAGE_HTML,
    STATS_ERROR_HTML,
    HEALTH_CHECK_HTML,
    HEALTH_ERROR_HTML
)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Configuration setup
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '../deployment/.env'))

# Add project root to path for models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.models import QueryRequest, UploadResponse, RAGResponse, SimpleRAGResponse, RAGSource, RAGResponseMetadata

# Constants
DEFAULT_TABLE_NAME = "document_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNKING_SIMILARITY = 0.5
ALLOWED_CONTENT_TYPES = [
    'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']

app = FastAPI(title="pgvector RAG API", version="1.0.0")

# Global configuration


class AppConfig:
    def __init__(self):
        # Initialize logfire with token from environment
        logfire_token = os.getenv('LOGFIRE_WRITE_TOKEN')
        if logfire_token:
            logfire.configure(token=logfire_token)
            print("✓ Logfire configured successfully")
        else:
            print("⚠️ LOGFIRE_WRITE_TOKEN not found, using default configuration")
            logfire.configure()

        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'rag_db'),
            'user': os.getenv('POSTGRES_USER', 'admin'),
            'password': os.getenv('POSTGRES_PASSWORD', 'admin')
        }
        self.file_validator = FileValidator(FileValidationConfig())
        self.agent = self._configure_pydantic_ai()
        self.pipeline = None
        self.reranker = None  # Lazy initialization to avoid startup overhead

    def _configure_pydantic_ai(self):
        gemini_key = os.getenv('GOOGLE_API_KEY')
        if gemini_key:
            try:
                # Configure Pydantic AI Agent with GoogleProvider
                provider = GoogleProvider(api_key=gemini_key)
                model = GoogleModel('gemini-2.5-flash', provider=provider)

                # Create agent with system prompt and output type
                agent = Agent(
                    model,
                    output_type=SimpleRAGResponse,
                    system_prompt="""You are a helpful RAG (Retrieval-Augmented Generation) assistant.
                    Based on the provided context from document search, provide comprehensive answers to user questions.

                    Instructions:
                    - Answer directly and accurately based on the provided context
                    - If the context doesn't fully answer the question, clearly state what information is available
                    - Cite specific sources when making claims, including page numbers when available (e.g., "according to Source 1, Page 5")
                    - Be concise but thorough
                    - Provide a confidence score (0-1) based on how well the context answers the question

                    Respond with:
                    - answer: Your comprehensive response with page references
                    - confidence: Float between 0-1 indicating confidence in the answer
                    - word_count: Number of words in your answer
                    - sources_used: Number of sources used (will be provided)
                    - metadata: Any additional relevant information"""
                )

                # Test the agent with a simple query
                print("✓ Pydantic AI Agent configured successfully")
                return agent
            except Exception as e:
                print(f"❌ Pydantic AI configuration failed: {e}")
                # Fallback to direct genai for backward compatibility
                try:
                    genai.configure(api_key=gemini_key)
                    print("✓ Fallback to direct Gemini API")
                except Exception as fallback_error:
                    print(f"❌ Gemini fallback also failed: {fallback_error}")
        return None


config = AppConfig()

def rerank_bm25(query:str,
                sources: List[Dict], 
                top_k:int):
    """
    Using bm25 to return top k sources from the inputs
    Args:
        sources: 
            - List of sources from postgresql
            - Format: {'chunk_id': '6250bf1e-e76e-4bb9-b33b-d05eb07aa77c',
                        'text': 'abc'
                        'metadata': {},
                        ...}
        top_k: top result
    """
    # Tokenize the text (simple word splitting)
    tokenized_corpus = [doc['text'].lower().split() for doc in sources]

    # Step 2: Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Step 3: Search with BM25
    tokenized_query = query.lower().split()

    # Get BM25 scores for all documents
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top k results
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # Retrieve top documents
    bm25_results = []
    for idx in top_indices:
        result = sources[idx].copy()
        result['bm25_score'] = float(bm25_scores[idx])
        bm25_results.append(result)

    return bm25_results


async def get_pipeline(table_name: str = DEFAULT_TABLE_NAME):
    if config.pipeline is None or config.pipeline.vector_store.table_name != table_name:
        config.pipeline = ChunkEmbeddingPipeline(
            db_params=config.db_params,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            table_name=table_name
        )
        # Initialize the database for the new pipeline
        await config.pipeline.vector_store._initialize_database()
    return config.pipeline


def get_reranker():
    """Get or initialize the reranker (lazy initialization)."""
    if config.reranker is None:
        # Get model from environment or use default
        rerank_model = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        config.reranker = Reranker(model_name=rerank_model)
        print(f"✓ Reranker initialized with model: {rerank_model}")
    return config.reranker


def validate_upload_params(chunk_size: int, content_type: str):
    """Validate upload parameters"""
    if not (128 <= chunk_size <= 2048):
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size must be between 128-2048"
        )
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files supported"
        )


async def generate_llm_response(query: str, context: str, results: list) -> SimpleRAGResponse:
    """Generate LLM response using Pydantic AI Agent or fallback"""

    # Calculate metadata
    sources_used = len(results)

    with logfire.span("llm_response_generation",
                     query=query[:100],  # Truncate long queries
                     sources_used=sources_used,
                     context_length=len(context)):

        logfire.info("Starting LLM response generation",
                    query_length=len(query),
                    context_length=len(context),
                    sources_used=sources_used)

        try:
            # Check if agent is configured
            if config.agent is None:
                raise Exception("Pydantic AI Agent is not configured - missing GOOGLE_API_KEY or configuration failed")

            # Use Pydantic AI Agent for structured response with proper user message
            user_message = f"""Context from documents:
{context}

User Question: {query}

Sources used: {sources_used}"""

            response = await config.agent.run(user_message)

            # The agent now returns a structured SimpleRAGResponse directly
            if hasattr(response, 'output') and isinstance(response.output, SimpleRAGResponse):
                # Update sources_used if not set correctly by the model
                if response.output.sources_used != sources_used:
                    response.output.sources_used = sources_used

                logfire.info("LLM response generated successfully",
                            word_count=response.output.word_count,
                            confidence=response.output.confidence,
                            method="pydantic_ai_agent")

                return response.output
            else:
                # Fallback if response structure is unexpected
                answer_text = response.output if hasattr(response, 'output') else str(response)

                logfire.warn("Unexpected response structure, using fallback",
                            response_type=type(response).__name__)

                return SimpleRAGResponse(
                    answer=answer_text,
                    confidence=0.8,
                    word_count=len(answer_text.split()),
                    sources_used=sources_used,
                    metadata={"method": "pydantic_ai_agent_fallback", "model": "gemini-2.5-flash"}
                )

        except Exception as llm_error:
            logfire.error("LLM generation failed, using fallback",
                         error=str(llm_error),
                         error_type=type(llm_error).__name__)

            print(f"Pydantic AI Agent failed: {llm_error}")
            # Fallback to basic structured response
            fallback_answer = f"LLM generation failed ({str(llm_error)}), but found {len(results)} relevant chunks:\n\n"
            for i, result in enumerate(results[:3]):
                fallback_answer += f"{i+1}. {result['text'][:300]}...\n\n"

            return SimpleRAGResponse(
                answer=fallback_answer,
                confidence=0.3,  # Low confidence due to fallback
                word_count=len(fallback_answer.split()),
                sources_used=sources_used,
                metadata={"fallback_reason": str(
                    llm_error), "method": "pydantic_ai_fallback"}
            )


async def perform_document_search(query: str, limit: int, threshold: float, document_ids=None,
                                 table_name=DEFAULT_TABLE_NAME):
    """Common document search logic with optional reranking"""
    pipeline = await get_pipeline(table_name)

    with logfire.span("document_search",
                     query=query[:100],  # Truncate long queries for logging
                     limit=limit,
                     threshold=threshold,
                     table_name=table_name):

        # Step 1: pgvector similarity search
        with logfire.span("embedding_generation_for_search"):
            logfire.info("Generating embeddings for search query",
                        query_length=len(query),
                        embedding_model=pipeline.embedding_generator.model_name)

            results = await pipeline.search_documents(
                query=query,
                limit=limit,
                threshold=threshold,
                document_ids=document_ids
            )

            logfire.info("Initial search completed",
                        results_found=len(results),
                        avg_similarity=sum(r['similarity'] for r in results) / len(results) if results else 0)

        # Step 1.5: Apply BM25 reranking if we have enough results
        avg_rerank_score = None

        if len(results) > 5:
            reranking_enabled = True
            with logfire.span("bm25_reranking"):
                # Rerank using BM25
                reranked_results = rerank_bm25(query=query, sources=results, top_k=5)

                # Add BM25 scores to results
                results = [{
                    'chunk_id': r['chunk_id'],
                    'text': r['text'],
                    'document_id': r['document_id'],
                    'metadata': r['metadata'],
                    'similarity': r['similarity'],
                    'rerank_score': r['bm25_score']
                } for r in reranked_results]

                avg_rerank_score = sum(r['rerank_score'] for r in results) / len(results) if results else None

                logfire.info("BM25 reranking completed",
                           final_results=len(results),
                           avg_rerank_score=avg_rerank_score)

        if not results:
            return RAGResponse(
                query=query,
                answer="No relevant documents found with the specified similarity threshold.",
                sources=[],
                search_stats=RAGResponseMetadata(
                    chunks_found=0,
                    avg_similarity=0.0,
                    search_method="pgvector_cosine" + ("_bm25" if reranking_enabled else ""),
                    threshold_used=threshold,
                    word_count=9,  # word count of the no-results message
                    confidence=0.0,
                    reranking_enabled=reranking_enabled,
                    avg_rerank_score=avg_rerank_score
                ),
                table_used=table_name
            )

        # Step 2: Build context from retrieved chunks with page numbers
        context_parts = []
        for i, result in enumerate(results):
            page_info = ""
            if result.get('metadata') and result['metadata'].get('page_number'):
                page_info = f" (Page {result['metadata']['page_number']})"
            context_parts.append(f"[Source {i+1}{page_info}]: {result['text']}")
        context = "\n\n".join(context_parts)

        # Step 3: Generate response with LLM
        llm_response = await generate_llm_response(query, context, results)

        # Calculate search statistics
        avg_similarity = sum(r['similarity'] for r in results) / len(results)

        # Create structured response
        return RAGResponse(
            query=query,
            answer=llm_response.answer,
            sources=[
                RAGSource(
                    chunk_id=r['chunk_id'],
                    text=r['text'],
                    similarity=round(r['similarity'], 3),
                    document_id=r['document_id'],
                    page_number=r.get('metadata', {}).get('page_number'),
                    metadata=r.get('metadata', {}),
                    rerank_score=round(r['rerank_score'], 3) if 'rerank_score' in r else None
                ) for r in results
            ],
            search_stats=RAGResponseMetadata(
                chunks_found=len(results),
                avg_similarity=round(avg_similarity, 3),
                search_method="pgvector_cosine" + ("_bm25" if reranking_enabled else ""),
                threshold_used=threshold,
                word_count=llm_response.word_count,
                confidence=llm_response.confidence,
                reranking_enabled=reranking_enabled,
                avg_rerank_score=round(avg_rerank_score, 3) if avg_rerank_score else None
            ),
            table_used=table_name
        )


@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_PAGE_HTML


@app.post("/upload", response_model=UploadResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    table_name: str = Form("document_chunks")
):
    """Upload and process document with comprehensive validation and pgvector storage"""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate parameters
    validate_upload_params(chunk_size, file.content_type)

    # Generate unique document ID
    document_id = str(uuid.uuid4())
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{document_id}_{file.filename}"

    try:
        # Write temporary file for validation
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Comprehensive file validation
        validation_result = config.file_validator.validate_file(str(temp_path))
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.error_message}"
            )

        # Process with pgvector pipeline using specified table
        pipeline = await get_pipeline(table_name)

        with logfire.span("document_insertion",
                         document_id=document_id,
                         filename=file.filename,
                         chunk_size=chunk_size,
                         table_name=table_name):
            logfire.info("Starting document processing",
                        document_id=document_id,
                        filename=file.filename,
                        file_size=validation_result.file_size)

            processed_id = await pipeline.process_document(
                file_path=str(temp_path),
                chunk_size=chunk_size,
                similarity_threshold=DEFAULT_CHUNKING_SIMILARITY,
                document_id=document_id,
                metadata={
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'file_size': validation_result.file_size,
                    'upload_timestamp': str(uuid.uuid1().time),
                    'validation_passed': True
                }
            )

            logfire.info("Document processing completed successfully",
                        document_id=processed_id,
                        filename=file.filename)

        return UploadResponse(
            status="success",
            document_id=processed_id,
            filename=file.filename,
            message=f"Document processed successfully",
            chunks_created=None
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)


@app.post("/query", response_model=RAGResponse)
async def query_documents(request: QueryRequest):
    """Query documents using pgvector similarity search + LLM generation with optional reranking"""
    try:
        result = await perform_document_search(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            document_ids=request.document_ids,
            table_name=DEFAULT_TABLE_NAME,
            enable_reranking=request.enable_reranking,
            rerank_top_k=request.rerank_top_k
        )
        # Return the structured RAGResponse directly
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query-form", response_class=HTMLResponse)
async def query_documents_form(
    query: str = Form(...),
    limit: int = Form(5),
    threshold: float = Form(0.7),
    table_name: str = Form(DEFAULT_TABLE_NAME),
    enable_reranking: bool = Form(False)
):
    """Query documents using form data (for HTML form submission) with optional reranking"""
    try:
        result = await perform_document_search(
            query=query,
            limit=limit,
            threshold=threshold,
            document_ids=None,
            table_name=table_name,
            enable_reranking=enable_reranking,
            rerank_top_k=None  # Use limit as default
        )

        # Build sources HTML with optional BM25 rerank scores
        sources_html = ''.join([f"""
        <div class="source-item">
            <strong>Source {i+1}</strong> (Similarity: {source.similarity:.1%}{"" if source.rerank_score is None else f", BM25: {source.rerank_score:.3f}"})<br>
            <em>Document: {source.document_id[:8]}... | Page: {source.page_number or 'N/A'}</em><br><br>
            {source.text}
        </div>
        """ for i, source in enumerate(result.sources)])

        # Use template with substitutions
        html_content = SEARCH_RESULTS_HTML.format(
            query=query,
            answer=result.answer.replace('\n', '<br>'),
            source_count=len(result.sources),
            sources_html=sources_html,
            chunks_found=result.search_stats.chunks_found,
            avg_similarity=f"{result.search_stats.avg_similarity:.1%}",
            search_method=result.search_stats.search_method,
            table_used=result.table_used,
            threshold_used=f"{result.search_stats.threshold_used:.1%}",
            confidence=f"{result.search_stats.confidence:.1%}" if result.search_stats.confidence else "N/A",
            word_count=result.search_stats.word_count or 0
        )

        return html_content

    except Exception as e:
        # Return error page using template
        return SEARCH_ERROR_HTML.format(error_message=str(e))


@app.get("/stats", response_class=HTMLResponse)
async def get_database_stats():
    """Get database statistics and collection information"""
    try:
        pipeline = await get_pipeline()
        stats = await pipeline.get_stats()

        # Use template with substitutions
        return STATS_PAGE_HTML.format(
            total_documents=f"{stats['total_documents']:,}",
            total_chunks=f"{stats['total_chunks']:,}",
            avg_text_length=f"{stats['avg_text_length']:.0f}",
            avg_chunks_per_doc=f"{stats['total_chunks'] // max(stats['total_documents'], 1):.0f}",
            embedding_model=pipeline.embedding_generator.model_name,
            embedding_dim=pipeline.embedding_generator.embedding_dim,
            table_name=pipeline.vector_store.table_name,
            earliest_chunk=stats['earliest_chunk'] or 'No documents yet',
            latest_chunk=stats['latest_chunk'] or 'No documents yet'
        )
    except Exception as e:
        return STATS_ERROR_HTML.format(error_message=str(e))


@app.get("/health", response_class=HTMLResponse)
async def health_check():
    """Health check endpoint to verify system status"""
    try:
        pipeline = await get_pipeline()
        stats = await pipeline.get_stats()

        # Check database connection
        db_status = "healthy" if stats['total_chunks'] >= 0 else "error"
        status_icon = "✅" if db_status == "healthy" else "❌"
        status_color = "#28a745" if db_status == "healthy" else "#dc3545"

        html_content = HEALTH_CHECK_HTML.format(
            status_color=status_color,
            status_icon=status_icon,
            db_status_upper=db_status.upper(),
            embedding_model=pipeline.embedding_generator.model_name,
            table_name=pipeline.vector_store.table_name,
            total_documents=f"{stats['total_documents']:,}",
            total_chunks=f"{stats['total_chunks']:,}",
            embedding_dim=pipeline.embedding_generator.embedding_dim,
            avg_text_length=f"{stats['avg_text_length']:.0f}",
            timestamp=str(uuid.uuid1().time)
        )
        return html_content

    except Exception as e:
        return HEALTH_ERROR_HTML.format(error_message=str(e))


@app.get("/supported-types")
async def get_supported_types():
    """Get information about supported file types and validation config"""
    return {
        "supported_extensions": config.file_validator.config.allowed_extensions,
        "max_file_size_mb": config.file_validator.config.max_file_size_mb,
        "supported_types": ["pdf", "docx", "txt"],
        "vector_store_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "database_backend": "PostgreSQL + pgvector",
            "chunking_method": "semantic_chunking_with_chonkie"
        }
    }


@app.delete("/table/{table_name}")
async def delete_table(table_name: str):
    """Delete a specific table from the database (optimized for speed)"""

    with logfire.span("table_deletion",
                     table_name=table_name):

        logfire.info("Starting table deletion",
                    table_name=table_name)

        try:
            pipeline_instance = await get_pipeline(table_name)

            # Get connection and delete table quickly
            conn = await pipeline_instance.vector_store._get_connection()
            row_count = 0

            with logfire.span("table_existence_check"):
                # Check if table exists and get approximate row count in one query
                result = await conn.fetchrow("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                    ) as table_exists,
                    COALESCE((
                        SELECT reltuples::bigint
                        FROM pg_catalog.pg_class
                        WHERE relname = $1
                    ), 0) as estimated_rows;
                """, table_name)

                table_exists = result['table_exists']
                row_count = result['estimated_rows']  # Approximate count (much faster)

                logfire.info("Table existence check completed",
                            table_exists=table_exists,
                            estimated_rows=row_count)

            if not table_exists:
                await conn.close()
                logfire.warn("Table deletion failed - table does not exist",
                            table_name=table_name)
                raise HTTPException(
                    status_code=404,
                    detail=f"Table '{table_name}' does not exist"
                )

            with logfire.span("table_data_deletion"):
                # Two-step ultra-fast deletion: TRUNCATE then DROP
                # Step 1: Instant data removal (no WAL overhead)
                await conn.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
                logfire.info("Table data truncated successfully",
                            table_name=table_name,
                            rows_deleted=row_count)

            with logfire.span("table_schema_deletion"):
                # Step 2: Clean schema removal
                await conn.execute(f"DROP TABLE {table_name} CASCADE;")
                logfire.info("Table schema dropped successfully",
                            table_name=table_name)

            await conn.close()

            # Reset pipeline if we deleted the current table
            if table_name == pipeline_instance.vector_store.table_name:
                config.pipeline = None
                logfire.info("Pipeline reset due to current table deletion",
                            table_name=table_name)

            logfire.info("Table deletion completed successfully",
                        table_name=table_name,
                        estimated_rows_deleted=row_count)

            return {
                "status": "success",
                "message": f"Table '{table_name}' deleted successfully",
                "table_name": table_name,
                "estimated_rows_deleted": row_count,
                "timestamp": str(uuid.uuid1().time)
            }

        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logfire.error("Table deletion failed with unexpected error",
                         table_name=table_name,
                         error=str(e),
                         error_type=type(e).__name__)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete table '{table_name}': {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
