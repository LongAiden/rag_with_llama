import os
import uuid
from pathlib import Path
from typing import List, Optional
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '../deployment/.env'))

# Your existing components - updated with pgvector
from embed_chunks_to_db import ChunkEmbeddingPipeline
from file_validator import FileValidator, FileValidationConfig
import google.generativeai as genai

app = FastAPI(title="pgvector RAG API", version="1.0.0")

# Initialize once at startup
db_params = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('POSTGRES_DB', 'rag_db'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin')
}

# Initialize file validator
file_validator = FileValidator(FileValidationConfig())

# Configure Gemini with debugging
gemini_key = os.getenv('GOOGLE_API_KEY')

# Debug: Show what we loaded
print(f"🔧 Environment variables after loading:")
print(f"  POSTGRES_USER: {os.getenv('POSTGRES_USER', 'Not set')}")
print(f"  POSTGRES_DB: {os.getenv('POSTGRES_DB', 'Not set')}")
print(f"  GOOGLE_API_KEY: {'Set' if gemini_key else 'Not set'}")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        print("✓ Gemini configured successfully")
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
        gemini_key = None

# Lazy initialization for better startup time
pipeline = None
def get_pipeline(table_name: str = 'document_chunks'):
    global pipeline
    if pipeline is None:
        pipeline = ChunkEmbeddingPipeline(
            db_params=db_params,
            embedding_model='all-MiniLM-L6-v2',
            table_name=table_name
        )
    return pipeline

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = None

class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    message: str
    chunks_created: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>pgvector RAG System</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f8f9fa;
            }
            .section { 
                margin: 30px 0; 
                padding: 25px; 
                background: white;
                border-radius: 12px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            input, textarea { 
                margin: 10px 0; 
                padding: 12px; 
                width: 100%; 
                box-sizing: border-box; 
                border: 2px solid #e9ecef;
                border-radius: 6px;
                font-size: 14px;
            }
            input:focus, textarea:focus {
                outline: none;
                border-color: #007bff;
            }
            button { 
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer; 
                font-weight: 600;
                transition: all 0.2s;
            }
            button:hover { 
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,123,255,0.3);
            }
            .stats {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <h1>🚀 pgvector RAG System</h1>
        <div class="stats">
            <strong>Powered by:</strong> PostgreSQL + pgvector for high-performance similarity search
        </div>
        
        <div class="section">
            <h2>📤 Upload & Process Document</h2>
            <p>Supported formats: PDF, TXT. Documents are chunked semantically and stored with vector embeddings.</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,.txt" required>
                <br>
                <label>Table Name: <input type="text" name="table_name" value="document_chunks" placeholder="document_chunks"></label>
                <br>
                <label>Chunk Size: <input type="number" name="chunk_size" value="512" min="128" max="2048"></label>
                <label>Similarity Threshold: <input type="number" name="similarity_threshold" value="0.5" min="0.1" max="0.9" step="0.1"></label>
                <br>
                <button type="submit">Upload & Process</button>
            </form>
        </div>
        
        <div class="section">
            <h2>🔍 Query Documents</h2>
            <p>Semantic search powered by sentence embeddings and pgvector cosine similarity.</p>
            <form action="/query-form" method="post">
                <textarea name="query" placeholder="Ask a question about your documents..." required rows="3"></textarea>
                <br>
                <label>Table Name: <input type="text" name="table_name" value="document_chunks" placeholder="document_chunks"></label>
                <br>
                <label>Max Results: <input type="number" name="limit" value="5" min="1" max="10" style="width: 80px;"></label>
                <label>Similarity Threshold: <input type="number" name="threshold" value="0.7" min="0.5" max="0.95" step="0.05" style="width: 80px;"></label>
                <br>
                <button type="submit">Search</button>
            </form>
        </div>
        
        <div class="section">
            <h2>📊 System Status</h2>
            <a href="/stats" target="_blank"><button>View Database Statistics</button></a>
            <a href="/health" target="_blank"><button>Health Check</button></a>
        </div>
    </body>
    </html>
    """


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

    # Validate parameters first
    if not (128 <= chunk_size <= 2048):
        raise HTTPException(status_code=400, detail="Chunk size must be between 128-2048")
    if not (0.1 <= similarity_threshold <= 0.9):
        raise HTTPException(status_code=400, detail="Similarity threshold must be between 0.1-0.9")

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

        # Comprehensive file validation using FileValidator
        validation_result = file_validator.validate_file(str(temp_path))

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.error_message}"
            )

        # Additional content type check
        if file.content_type not in ['application/pdf', 'text/plain']:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files supported"
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
                'mime_type': validation_result.mime_type,
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
        pipeline = get_pipeline()

        # Step 1: pgvector similarity search
        results = pipeline.search_documents(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            document_ids=request.document_ids
        )

        if not results:
            return {
                "query": request.query,
                "answer": "No relevant documents found with the specified similarity threshold.",
                "sources": [],
                "search_stats": {
                    "chunks_found": 0,
                    "avg_similarity": 0.0,
                    "search_method": "pgvector_cosine"
                }
            }

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]: {result['text']}")

        context = "\n\n".join(context_parts)

        # Step 3: Generate response with LLM (if available)
        if gemini_key:
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"""Based on the following context from document search, provide a comprehensive answer to the user's question.
                            Context from documents:
                            {context}
                            User Question: {request.query}

                            Instructions:
                            - Answer directly and accurately based on the provided context
                            - If the context doesn't fully answer the question, clearly state what information is available
                            - Cite specific sources when making claims
                            - Be concise but thorough

                            Answer:"""

                response = model.generate_content(prompt)
                answer = response.text

            except Exception as llm_error:
                # Fallback if LLM fails
                answer = f"LLM generation failed ({str(llm_error)}), but found {len(results)} relevant chunks:\n\n"
                for i, result in enumerate(results[:3]):
                    answer += f"{i+1}. {result['text'][:300]}...\n\n"
        else:
            # Fallback without LLM
            answer = f"Found {len(results)} relevant document chunks (LLM not configured):\n\n"
            for i, result in enumerate(results):
                similarity_pct = f"{result['similarity']*100:.1f}%"
                answer += f"**Source {i+1}** (similarity: {similarity_pct}):\n{result['text'][:400]}...\n\n"

        # Calculate search statistics
        avg_similarity = sum(r['similarity'] for r in results) / len(results)

        return {
            "query": request.query,
            "answer": answer,
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
                "threshold_used": request.threshold
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/query-form")
async def query_documents_form(
    query: str = Form(...),
    limit: int = Form(5),
    threshold: float = Form(0.7),
    table_name: str = Form("document_chunks")
):
    """Query documents using form data (for HTML form submission)"""
    # Use the pipeline with specified table
    try:
        pipeline = get_pipeline(table_name)

        # Step 1: pgvector similarity search
        results = pipeline.search_documents(
            query=query,
            limit=limit,
            threshold=threshold,
            document_ids=None
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
                    "search_method": "pgvector_cosine"
                }
            }

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]: {result['text']}")

        context = "\n\n".join(context_parts)

        # Step 3: Generate response with LLM (if available)
        if gemini_key:
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
                answer = response.text

            except Exception as llm_error:
                # Fallback if LLM fails
                answer = f"LLM generation failed ({str(llm_error)}), but found {len(results)} relevant chunks:\n\n"
                for i, result in enumerate(results[:3]):
                    answer += f"{i+1}. {result['text'][:300]}...\n\n"
        else:
            # Fallback without LLM
            answer = f"Found {len(results)} relevant document chunks (LLM not configured):\n\n"
            for i, result in enumerate(results):
                similarity_pct = f"{result['similarity']*100:.1f}%"
                answer += f"**Source {i+1}** (similarity: {similarity_pct}):\n{result['text'][:400]}...\n\n"

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


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
        "supported_extensions": file_validator.config.allowed_extensions,
        "max_file_size_mb": file_validator.config.max_file_size_mb,
        "supported_types": ["pdf", "txt"],
        "mime_types": list(file_validator.mime_type_mapping.keys()),
        "vector_store_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "database_backend": "PostgreSQL + pgvector",
            "chunking_method": "semantic_chunking_with_chonkie"
        }
    }

@app.delete("/table/{table_name}")
async def delete_table(table_name: str):
    """Delete a specific table from the database (optimized for speed)"""
    # Security check - only allow deletion of document-related tables
    allowed_tables = ["document_chunks", "chunks", "documents"]

    if table_name not in allowed_tables:
        raise HTTPException(
            status_code=403,
            detail=f"Deletion of table '{table_name}' not allowed. Allowed tables: {allowed_tables}"
        )

    try:
        global pipeline
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
            pipeline = None

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
    # Security check - only allow recreation of document-related tables
    allowed_tables = ["document_chunks", "chunks", "documents"]

    if table_name not in allowed_tables:
        raise HTTPException(
            status_code=403,
            detail=f"Recreation of table '{table_name}' not allowed. Allowed tables: {allowed_tables}"
        )

    try:
        global pipeline
        pipeline_instance = get_pipeline()

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
            pipeline = None

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