"""HTML templates for the RAG API"""

HOME_PAGE_HTML = """
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