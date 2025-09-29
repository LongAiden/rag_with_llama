from models.models import QueryRequest, UploadResponse, RAGResponse, SimpleRAGResponse, RAGSource, RAGResponseMetadata
import os
import sys
import uuid
from pathlib import Path

import google.generativeai as genai
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from file_validator import FileValidator, FileValidationConfig
from embed_chunks_to_db import ChunkEmbeddingPipeline
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


# Constants
DEFAULT_TABLE_NAME = "document_chunks"  # Changeable in webUI
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNKING_SIMILARITY = 0.5  # Default similarity threshold for chunking
CHUNK_SIZE_LIMITS = (128, 2048)
SIMILARITY_THRESHOLD_LIMITS = (0.1, 0.9)
ALLOWED_CONTENT_TYPES = [
    'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
ALLOWED_TABLES = ["document_chunks", 'bert', "test1", "test2"]

app = FastAPI(title="pgvector RAG API", version="1.0.0")

# Global configuration


class AppConfig:
    def __init__(self):
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

    def _configure_pydantic_ai(self):
        gemini_key = os.getenv('GOOGLE_API_KEY')
        if gemini_key:
            try:
                # Configure Pydantic AI Agent with GoogleProvider
                provider = GoogleProvider(api_key=gemini_key)
                model = GoogleModel('gemini-2.5-flash', provider=provider)

                # Create agent with system prompt and result type
                agent = Agent(
                    model,
                    result_type=SimpleRAGResponse,
                    system_prompt="""You are a helpful RAG (Retrieval-Augmented Generation) assistant.
                    Based on the provided context from document search, provide comprehensive answers to user questions.

                    Instructions:
                    - Answer directly and accurately based on the provided context
                    - If the context doesn't fully answer the question, clearly state what information is available
                    - Cite specific sources when making claims
                    - Be concise but thorough
                    - Provide a confidence score (0-1) based on how well the context answers the question

                    Respond with:
                    - answer: Your comprehensive response
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


def validate_upload_params(chunk_size: int, content_type: str):
    """Validate upload parameters"""
    if not (CHUNK_SIZE_LIMITS[0] <= chunk_size <= CHUNK_SIZE_LIMITS[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size must be between {CHUNK_SIZE_LIMITS[0]}-{CHUNK_SIZE_LIMITS[1]}"
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

    try:
        # Use Pydantic AI Agent for structured response with proper user message
        user_message = f"""Context from documents:
{context}

User Question: {query}

Sources used: {sources_used}"""

        response = await config.agent.run(user_message)

        # Ensure we always return SimpleRAGResponse type
        if isinstance(response, SimpleRAGResponse):
            # Update sources_used if not set correctly by the model
            if response.sources_used != sources_used:
                response.sources_used = sources_used
            return response
        else:
            # If the agent didn't return the expected type, create a proper response
            answer_text = str(response) if hasattr(
                response, '__str__') else "Response type error"
            return SimpleRAGResponse(
                answer=answer_text,
                confidence=0.5,
                word_count=len(answer_text.split()),
                sources_used=sources_used,
                metadata={"note": "Response type converted",
                          "method": "pydantic_ai_converted"}
            )

    except Exception as llm_error:
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


async def perform_document_search(query: str, limit: int, threshold: float, document_ids=None, table_name=DEFAULT_TABLE_NAME):
    """Common document search logic"""
    pipeline = await get_pipeline(table_name)

    # Step 1: pgvector similarity search
    results = await pipeline.search_documents(
        query=query,
        limit=limit,
        threshold=threshold,
        document_ids=document_ids
    )

    if not results:
        return RAGResponse(
            query=query,
            answer="No relevant documents found with the specified similarity threshold.",
            sources=[],
            search_stats=RAGResponseMetadata(
                chunks_found=0,
                avg_similarity=0.0,
                search_method="pgvector_cosine",
                threshold_used=threshold,
                word_count=9,  # word count of the no-results message
                confidence=0.0
            ),
            table_used=table_name
        )

    # Step 2: Build context from retrieved chunks
    context_parts = [
        f"[Source {i+1}]: {result['text']}" for i, result in enumerate(results)]
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
                similarity=round(r['similarity'], 3),
                document_id=r['document_id'],
                metadata=r.get('metadata', {})
            ) for r in results
        ],
        search_stats=RAGResponseMetadata(
            chunks_found=len(results),
            avg_similarity=round(avg_similarity, 3),
            search_method="pgvector_cosine",
            threshold_used=threshold,
            word_count=llm_response.word_count,
            confidence=llm_response.confidence
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

        return UploadResponse(
            status="success",
            document_id=processed_id,
            filename=file.filename,
            message=f"Document validated and processed successfully with pgvector storage",
            chunks_created=None  # Could query this if needed
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
    """Query documents using pgvector similarity search + LLM generation"""
    try:
        result = await perform_document_search(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            document_ids=request.document_ids,
            table_name=DEFAULT_TABLE_NAME
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
    table_name: str = Form(DEFAULT_TABLE_NAME)
):
    """Query documents using form data (for HTML form submission)"""
    try:
        result = await perform_document_search(
            query=query,
            limit=limit,
            threshold=threshold,
            document_ids=None,
            table_name=table_name
        )

        # Build sources HTML
        sources_html = ''.join([f"""
        <div class="source-item">
            <strong>Source {i+1}</strong> (Similarity: {source.similarity:.1%})<br>
            <em>Document: {source.document_id[:8]}... | Page: {source.metadata.get('page_number', 'N/A')}</em><br><br>
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
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(
            status_code=403,
            detail=f"Deletion of table '{table_name}' not allowed. Allowed tables: {ALLOWED_TABLES}"
        )

    try:
        pipeline_instance = await get_pipeline(table_name)

        # Get connection and delete table quickly
        conn = await pipeline_instance.vector_store._get_connection()
        row_count = 0

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

        if not table_exists:
            await conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' does not exist"
            )

        # Two-step ultra-fast deletion: TRUNCATE then DROP
        # Step 1: Instant data removal (no WAL overhead)
        await conn.execute(f"TRUNCATE TABLE {table_name} CASCADE;")

        # Step 2: Clean schema removal
        await conn.execute(f"DROP TABLE {table_name} CASCADE;")

        await conn.close()

        # Reset pipeline if we deleted the current table
        if table_name == pipeline_instance.vector_store.table_name:
            config.pipeline = None

        return {
            "status": "success",
            "message": f"Table '{table_name}' deleted successfully using fast TRUNCATE+DROP method",
            "table_name": table_name,
            "estimated_rows_deleted": row_count,
            "optimization": "Used TRUNCATE CASCADE + DROP CASCADE for ultra-fast deletion",
            "note": "Row count is estimated for performance",
            "timestamp": str(uuid.uuid1().time)
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete table '{table_name}': {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
