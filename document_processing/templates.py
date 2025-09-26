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

        /* Success notification styles */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: white;
            font-weight: 600;
            z-index: 1000;
            opacity: 0;
            transform: translateX(400px);
            transition: all 0.3s ease;
        }
        .notification.success {
            background: linear-gradient(135deg, #28a745, #20c997);
        }
        .notification.error {
            background: linear-gradient(135deg, #dc3545, #e74c3c);
        }
        .notification.show {
            opacity: 1;
            transform: translateX(0);
        }
        .notification .close {
            float: right;
            margin-left: 15px;
            cursor: pointer;
            font-size: 18px;
            line-height: 1;
        }
    </style>
</head>
<body>
    <h1>🚀 pgvector RAG System</h1>
    <div class="stats">
        <strong>Powered by:</strong> PostgreSQL + pgvector for high-performance similarity search
    </div>

    <div class="section">
        <h2>📤 Upload & Process Document</h2>
        <p>Supported formats: PDF, DOCX, TXT. Documents are chunked semantically and stored with vector embeddings.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".pdf,.docx,.txt" required>
            <br>
            <label>Table Name: <input type="text" name="table_name" value="document_chunks" placeholder="document_chunks"></label>
            <br>
            <label>Chunk Size: <input type="number" name="chunk_size" value="512" min="128" max="2048"></label>
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

    <!-- Success/Error notification -->
    <div id="notification" class="notification">
        <span class="close" onclick="hideNotification()">×</span>
        <span id="notification-message"></span>
    </div>

    <script>
        // Show notification function
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            const messageElement = document.getElementById('notification-message');

            messageElement.textContent = message;
            notification.className = `notification ${type}`;

            // Show notification
            setTimeout(() => {
                notification.classList.add('show');
            }, 100);

            // Auto hide after 5 seconds
            setTimeout(() => {
                hideNotification();
            }, 5000);
        }

        // Hide notification function
        function hideNotification() {
            const notification = document.getElementById('notification');
            notification.classList.remove('show');
        }

        // Handle upload form submission with AJAX
        document.querySelector('form[action="/upload"]').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const submitButton = this.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;

            // Show loading state
            submitButton.textContent = 'Processing...';
            submitButton.disabled = true;

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    showNotification(`✅ Document "${result.filename}" uploaded and processed successfully! Document ID: ${result.document_id.substring(0,8)}...`, 'success');
                    this.reset(); // Clear the form
                } else {
                    showNotification(`❌ Upload failed: ${result.detail}`, 'error');
                }
            } catch (error) {
                showNotification(`❌ Upload failed: ${error.message}`, 'error');
            } finally {
                // Restore button
                submitButton.textContent = originalText;
                submitButton.disabled = false;
            }
        });

        // Check for URL parameters to show notifications (if redirected)
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('upload') === 'success') {
            showNotification('✅ Document uploaded and processed successfully!', 'success');
        } else if (urlParams.get('upload') === 'error') {
            showNotification('❌ Upload failed. Please try again.', 'error');
        }
    </script>
</body>
</html>
"""

SEARCH_RESULTS_HTML = """
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
        <p>{answer}</p>
    </div>

    <div class="sources">
        <h2>📚 Sources ({source_count} found)</h2>
        {sources_html}
    </div>

    <div class="stats">
        <strong>Search Statistics:</strong><br>
        • Chunks found: {chunks_found}<br>
        • Average similarity: {avg_similarity}<br>
        • Search method: {search_method}<br>
        • Table used: {table_used}<br>
        • Threshold: {threshold_used}
    </div>
</body>
</html>
"""

SEARCH_ERROR_HTML = """
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
        <p><em>{error_message}</em></p>
        <a href="/"><button>← Back to Search</button></a>
    </div>
</body>
</html>
"""

STATS_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Database Statistics - pgvector RAG</title>
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
            text-align: center;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .config-section {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        .config-item:last-child {{
            border-bottom: none;
        }}
        .config-label {{
            font-weight: 600;
            color: #333;
        }}
        .config-value {{
            color: #007bff;
            font-family: 'Courier New', monospace;
        }}
        button {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
        }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        h2 {{ color: #34495e; margin-bottom: 15px; }}
        .refresh-note {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Database Statistics</h1>
        <p>Real-time statistics from your pgvector RAG system</p>
        <a href="/"><button>← Back to Home</button></a>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_documents}</div>
            <div class="stat-label">Total Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_chunks}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_text_length}</div>
            <div class="stat-label">Avg Text Length</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_chunks_per_doc}</div>
            <div class="stat-label">Avg Chunks/Doc</div>
        </div>
    </div>

    <div class="config-section">
        <h2>⚙️ System Configuration</h2>
        <div class="config-item">
            <span class="config-label">Embedding Model</span>
            <span class="config-value">{embedding_model}</span>
        </div>
        <div class="config-item">
            <span class="config-label">Embedding Dimensions</span>
            <span class="config-value">{embedding_dim}</span>
        </div>
        <div class="config-item">
            <span class="config-label">Active Table</span>
            <span class="config-value">{table_name}</span>
        </div>
        <div class="config-item">
            <span class="config-label">Database Backend</span>
            <span class="config-value">PostgreSQL + pgvector</span>
        </div>
        <div class="config-item">
            <span class="config-label">Search Method</span>
            <span class="config-value">Cosine Similarity</span>
        </div>
    </div>

    <div class="config-section">
        <h2>📅 Timeline Information</h2>
        <div class="config-item">
            <span class="config-label">Earliest Document</span>
            <span class="config-value">{earliest_chunk}</span>
        </div>
        <div class="config-item">
            <span class="config-label">Latest Document</span>
            <span class="config-value">{latest_chunk}</span>
        </div>
    </div>

    <div class="refresh-note">
        <strong>📝 Note:</strong> Statistics are computed in real-time. Refresh this page to see updated numbers.
    </div>
</body>
</html>
"""

STATS_ERROR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Database Stats Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 100px auto; padding: 20px; text-align: center; }}
        .error {{ background: #ffebee; padding: 30px; border-radius: 12px; border-left: 5px solid #f44336; }}
        button {{ background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="error">
        <h2>❌ Failed to Load Statistics</h2>
        <p>Error: {error_message}</p>
        <a href="/"><button>← Back to Home</button></a>
    </div>
</body>
</html>
"""
