import os
import sys
import uuid
from pathlib import Path

import google.generativeai as genai
from file_validator import FileValidator, FileValidationConfig
from embed_chunks_to_db import ChunkEmbeddingPipeline
from templates import HOME_PAGE_HTML

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Configuration setup
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '../deployment/.env'))

# Add project root to path for models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.models import QueryRequest, UploadResponse

# Constants
DEFAULT_TABLE_NAME = "document_chunks" # Changeable in webUI 
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" #
CHUNK_SIZE_LIMITS = (128, 2048)
SIMILARITY_THRESHOLD_LIMITS = (0.1, 0.9)
ALLOWED_CONTENT_TYPES = ['application/pdf', 'text/plain']
ALLOWED_TABLES = ["document_chunks",'bert', "test1", "test2"]

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
        self.gemini_key = self._configure_gemini()
        self.pipeline = None

    def _configure_gemini(self):
        gemini_key = os.getenv('GOOGLE_API_KEY')
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                print("✓ Gemini configured successfully")
                return gemini_key
            except Exception as e:
                print(f"❌ Gemini configuration failed: {e}")
        return None

config = AppConfig()


def get_pipeline(table_name: str = DEFAULT_TABLE_NAME):
    if config.pipeline is None or config.pipeline.vector_store.table_name != table_name:
        config.pipeline = ChunkEmbeddingPipeline(
            db_params=config.db_params,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            table_name=table_name
        )
    return config.pipeline


def validate_upload_params(chunk_size: int, similarity_threshold: float, content_type: str):
    """Validate upload parameters"""
    if not (CHUNK_SIZE_LIMITS[0] <= chunk_size <= CHUNK_SIZE_LIMITS[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size must be between {CHUNK_SIZE_LIMITS[0]}-{CHUNK_SIZE_LIMITS[1]}"
        )
    if not (SIMILARITY_THRESHOLD_LIMITS[0] <= similarity_threshold <= SIMILARITY_THRESHOLD_LIMITS[1]):
        raise HTTPException(
            status_code=400,
            detail=f"Similarity threshold must be between {SIMILARITY_THRESHOLD_LIMITS[0]}-{SIMILARITY_THRESHOLD_LIMITS[1]}"
        )
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files supported"
        )


def generate_llm_response(query: str, context: str, results: list) -> str:
    """Generate LLM response using Gemini or fallback"""
    if config.gemini_key:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"""Based on the following context from document search, provide a comprehensive answer to the user's question.

Context from documents:
{context}

User Question: {query}

Instructions:
- Answer directly and accurately based on the provided context
- If the context doesn't fully answer the question, clearly state what information is available
- Cite specific sources when making claims
- Be concise but thorough

Answer:"""
            response = model.generate_content(prompt)
            return response.text
        except Exception as llm_error:
            # Fallback if LLM fails
            answer = f"LLM generation failed ({str(llm_error)}), but found {len(results)} relevant chunks:\n\n"
            for i, result in enumerate(results[:3]):
                answer += f"{i+1}. {result['text'][:300]}...\n\n"
            return answer
    else:
        # Fallback without LLM
        answer = f"Found {len(results)} relevant document chunks (LLM not configured):\n\n"
        for i, result in enumerate(results):
            similarity_pct = f"{result['similarity']*100:.1f}%"
            answer += f"**Source {i+1}** (similarity: {similarity_pct}):\n{result['text'][:400]}...\n\n"
        return answer


def perform_document_search(query: str, limit: int, threshold: float, document_ids=None, table_name=DEFAULT_TABLE_NAME):
    """Common document search logic"""
    pipeline = get_pipeline(table_name)

    # Step 1: pgvector similarity search
    results = pipeline.search_documents(
        query=query,
        limit=limit,
        threshold=threshold,
        document_ids=document_ids
    )

    if not results:
        return {
            "query": query,
            "answer": "No relevant documents found with the specified similarity threshold.",
            "sources": [],
            "table_used": table_name,
            "search_stats": {
                "chunks_found": 0,
                "avg_similarity": 0.0,
                "search_method": "pgvector_cosine",
                "threshold_used": threshold
            }
        }

    # Step 2: Build context from retrieved chunks
    context_parts = [f"[Source {i+1}]: {result['text']}" for i, result in enumerate(results)]
    context = "\n\n".join(context_parts)

    # Step 3: Generate response with LLM
    answer = generate_llm_response(query, context, results)

    # Calculate search statistics
    avg_similarity = sum(r['similarity'] for r in results) / len(results)

    return {
        "query": query,
        "answer": answer,
        "table_used": table_name,
        "sources": [
            {
                "chunk_id": r['chunk_id'],
                "text": r['text'][:300] + "..." if len(r['text']) > 300 else r['text'],
                "similarity": round(r['similarity'], 3),
                "document_id": r['document_id'],
                "metadata": r.get('metadata', {})
            } for r in results
        ],
        "search_stats": {
            "chunks_found": len(results),
            "avg_similarity": round(avg_similarity, 3),
            "search_method": "pgvector_cosine",
            "threshold_used": threshold
        }
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_PAGE_HTML


@app.post("/upload", response_model=UploadResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    similarity_threshold: float = Form(0.5),
    table_name: str = Form("document_chunks")
):
    """Upload and process document with comprehensive validation and pgvector storage"""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate parameters
    validate_upload_params(chunk_size, similarity_threshold, file.content_type)

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
        pipeline = get_pipeline(table_name)
        processed_id = pipeline.process_document(
            file_path=str(temp_path),
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
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


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents using pgvector similarity search + LLM generation"""
    try:
        result = perform_document_search(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            document_ids=request.document_ids,
            table_name=DEFAULT_TABLE_NAME
        )
        # Remove table_used from result for API consistency
        result.pop('table_used', None)
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
        result = perform_document_search(
            query=query,
            limit=limit,
            threshold=threshold,
            document_ids=None,
            table_name=table_name
        )

        # Create HTML response showing just the answer
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Search Results - pgvector RAG</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 900px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f8f9fa;
                    line-height: 1.6;
                }}
                .header {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .answer {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .sources {{
                    background: #e3f2fd;
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                }}
                .source-item {{
                    background: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }}
                .stats {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    font-size: 14px;
                    color: #666;
                }}
                button {{
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: all 0.2s;
                    text-decoration: none;
                    display: inline-block;
                }}
                button:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0,123,255,0.3);
                }}
                h1 {{ color: #2c3e50; margin-bottom: 10px; }}
                h2 {{ color: #34495e; margin-bottom: 15px; }}
                .query {{ font-style: italic; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Search Results</h1>
                <p class="query"><strong>Query:</strong> "{query}"</p>
                <a href="/"><button>← Back to Search</button></a>
            </div>

            <div class="answer">
                <h2>💡 Answer</h2>
                <p>{result['answer'].replace(chr(10), '<br>')}</p>
            </div>

            <div class="sources">
                <h2>📚 Sources ({len(result['sources'])} found)</h2>
                {''.join([f"""
                <div class="source-item">
                    <strong>Source {i+1}</strong> (Similarity: {source['similarity']:.1%})<br>
                    <em>Document: {source['document_id'][:8]}...</em><br><br>
                    {source['text']}
                </div>
                """ for i, source in enumerate(result['sources'])])}
            </div>

            <div class="stats">
                <strong>Search Statistics:</strong><br>
                • Chunks found: {result['search_stats']['chunks_found']}<br>
                • Average similarity: {result['search_stats']['avg_similarity']:.1%}<br>
                • Search method: {result['search_stats']['search_method']}<br>
                • Table used: {result['table_used']}<br>
                • Threshold: {result['search_stats']['threshold_used']:.1%}
            </div>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        # Return error page instead of JSON
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - pgvector RAG</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 600px;
                    margin: 100px auto;
                    padding: 20px;
                    text-align: center;
                }}
                .error {{
                    background: #ffebee;
                    padding: 30px;
                    border-radius: 12px;
                    border-left: 5px solid #f44336;
                }}
                button {{
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>❌ Search Failed</h2>
                <p>Sorry, there was an error processing your query:</p>
                <p><em>{str(e)}</em></p>
                <a href="/"><button>← Back to Search</button></a>
            </div>
        </body>
        </html>
        """
        return error_html


@app.get("/stats")
async def get_database_stats():
    """Get database statistics and collection information"""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        return {
            "status": "success",
            "database_stats": stats,
            "vector_store": {
                "embedding_model": pipeline.embedding_generator.model_name,
                "embedding_dimension": pipeline.embedding_generator.embedding_dim,
                "table_name": pipeline.vector_store.table_name
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status"""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()

        # Check database connection
        db_status = "healthy" if stats['total_chunks'] >= 0 else "error"

        return {
            "status": "healthy",
            "timestamp": str(uuid.uuid1().time),
            "components": {
                "database": db_status,
                "embedding_model": "loaded",
                "vector_store": "operational"
            },
            "metrics": {
                "total_documents": stats['total_documents'],
                "total_chunks": stats['total_chunks']
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(uuid.uuid1().time)
        }


@app.get("/supported-types")
async def get_supported_types():
    """Get information about supported file types and validation config"""
    return {
        "supported_extensions": config.file_validator.config.allowed_extensions,
        "max_file_size_mb": config.file_validator.config.max_file_size_mb,
        "supported_types": ["pdf", "txt"],
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
        pipeline_instance = get_pipeline(table_name)

        # Get connection and delete table quickly
        conn = pipeline_instance.vector_store._get_connection()
        row_count = 0

        with conn.cursor() as cur:
            # Check if table exists and get approximate row count in one query
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                ) as table_exists,
                COALESCE((
                    SELECT reltuples::bigint
                    FROM pg_catalog.pg_class
                    WHERE relname = %s
                ), 0) as estimated_rows;
            """, (table_name, table_name))

            result = cur.fetchone()
            table_exists = result[0]
            row_count = result[1]  # Approximate count (much faster)

            if not table_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Table '{table_name}' does not exist"
                )

            # Two-step ultra-fast deletion: TRUNCATE then DROP
            # Step 1: Instant data removal (no WAL overhead)
            cur.execute(f"TRUNCATE TABLE {table_name} CASCADE;")

            # Step 2: Clean schema removal
            cur.execute(f"DROP TABLE {table_name} CASCADE;")

        conn.commit()
        conn.close()

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


@app.post("/table/{table_name}/recreate")
async def recreate_table(table_name: str):
    """Recreate a specific table (delete and reinitialize)"""
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(
            status_code=403,
            detail=f"Recreation of table '{table_name}' not allowed. Allowed tables: {ALLOWED_TABLES}"
        )

    try:
        pipeline_instance = get_pipeline(table_name)

        # Get connection
        conn = pipeline_instance.vector_store._get_connection()
        old_row_count = 0

        with conn.cursor() as cur:
            # Check if table exists and get row count
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                );
            """, (table_name,))

            table_exists = cur.fetchone()[0]

            if table_exists:
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                old_row_count = cur.fetchone()[0]

                # Drop the existing table
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

            # Recreate table with proper structure
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector(384),  -- Adjust dimension as needed
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Create indexes
            cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
            ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)

            cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_document_id_idx
            ON {table_name} (document_id);
            """)

        conn.commit()
        conn.close()

        # Reset pipeline if we recreated the current table
        if table_name == pipeline_instance.vector_store.table_name:
            config.pipeline = None

        return {
            "status": "success",
            "message": f"Table '{table_name}' recreated successfully",
            "table_name": table_name,
            "previous_rows": old_row_count,
            "current_rows": 0,
            "timestamp": str(uuid.uuid1().time)
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recreate table '{table_name}': {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
