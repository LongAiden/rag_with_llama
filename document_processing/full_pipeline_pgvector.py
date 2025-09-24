import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Your existing components - updated with pgvector
from embed_chunks_to_db import ChunkEmbeddingPipeline
import google.generativeai as genai

app = FastAPI(title="pgvector RAG API", version="1.0.0")

# Initialize once at startup
db_params = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('DB_NAME', 'rag_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Lazy initialization for better startup time
pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ChunkEmbeddingPipeline(
            db_params=db_params,
            embedding_model='all-MiniLM-L6-v2',
            table_name='document_chunks'
        )
    return pipeline

# Configure Gemini
gemini_key = os.getenv('GEMINI_API_KEY')
if gemini_key:
    genai.configure(api_key=gemini_key)

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
                <label>Chunk Size: <input type="number" name="chunk_size" value="512" min="128" max="2048"></label>
                <label>Similarity Threshold: <input type="number" name="similarity_threshold" value="0.5" min="0.1" max="0.9" step="0.1"></label>
                <br>
                <button type="submit">Upload & Process</button>
            </form>
        </div>
        
        <div class="section">
            <h2>🔍 Query Documents</h2>
            <p>Semantic search powered by sentence embeddings and pgvector cosine similarity.</p>
            <form action="/query" method="post">
                <textarea name="query" placeholder="Ask a question about your documents..." required rows="3"></textarea>
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
    similarity_threshold: float = Form(0.5)
):
    """Upload and process document with pgvector storage"""
    
    # Validate file type
    if file.content_type not in ['application/pdf', 'text/plain']:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF and TXT files supported"
        )
    
    # Validate parameters
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
        # Write temporary file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process with pgvector pipeline
        pipeline = get_pipeline()
        processed_id = pipeline.process_document(
            file_path=str(temp_path),
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            document_id=document_id,
            metadata={
                'filename': file.filename,
                'content_type': file.content_type,
                'upload_timestamp': str(uuid.uuid1().time)
            }
        )
        
        # Get chunk count for response
        stats = pipeline.get_stats()
        
        return UploadResponse(
            status="success",
            document_id=processed_id,
            filename=file.filename,
            message=f"Document processed and stored in pgvector database",
            chunks_created=None  # Could query this if needed
        )
        
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
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e