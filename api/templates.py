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
            background:
                linear-gradient(rgba(10, 20, 35, 0.50), rgba(10, 20, 35, 0.50)),
                url('/images/moutain_pexcel.jpeg') center center / cover fixed;
        }
        .section {
            margin: 30px 0;
            padding: 25px;
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-top: 2px solid rgba(59, 130, 246, 0.65);
            border-radius: 12px;
        }
        input, textarea {
            margin: 0;
            padding: 10px 12px;
            width: 100%;
            box-sizing: border-box;
            border: 1px solid rgba(255, 255, 255, 0.40);
            border-radius: 8px;
            font-size: 14px;
            background: rgba(255, 255, 255, 0.88);
            color: #1a1a2e;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: rgba(255, 255, 255, 0.80);
            background: rgba(255, 255, 255, 0.96);
        }
        input::placeholder, textarea::placeholder {
            color: #888;
        }
        .form-group {
            margin: 12px 0;
        }
        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.80);
            margin-bottom: 5px;
            letter-spacing: 0.02em;
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
        h1 { color: #ffffff; text-shadow: 0 2px 8px rgba(0,0,0,0.4); }
        h2 { color: #ffffff; text-shadow: 0 1px 4px rgba(0,0,0,0.35); margin-bottom: 15px; border-left: 3px solid #3b82f6; padding-left: 10px; }
        .section > p { color: rgba(255, 255, 255, 0.88); }
        .hero-card {
            background: rgba(255, 255, 255, 0.24);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.40);
            border-top: 3px solid #3b82f6;
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
        }
        .hero-card .stats {
            background: transparent;
            color: rgba(255, 255, 255, 0.9);
            padding: 10px 0 0 0;
            margin: 0;
            border: none;
        }
        .hero-card .stats strong { color: white; }

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

        /* Tab styles */
        .tab-bar {
            display: flex;
            gap: 8px;
            margin: 20px 0 12px 0;
        }
        .tab-btn {
            padding: 10px 28px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.18);
            color: rgba(255, 255, 255, 0.75);
            cursor: pointer;
            font-weight: 600;
            font-size: 15px;
            transition: background 0.2s, color 0.2s, border-color 0.2s, box-shadow 0.2s;
        }
        .tab-btn:hover {
            background: rgba(255, 255, 255, 0.28);
            color: #ffffff;
        }
        .tab-btn.active {
            background: rgba(59, 130, 246, 0.35);
            border-color: rgba(59, 130, 246, 0.70);
            color: #e0f0ff;
            box-shadow: 0 2px 12px rgba(59, 130, 246, 0.30);
        }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }
        .model-select {
            margin: 0;
            padding: 10px 12px;
            width: 100%;
            box-sizing: border-box;
            border: 1px solid rgba(255, 255, 255, 0.40);
            border-radius: 8px;
            font-size: 14px;
            background: rgba(255, 255, 255, 0.88);
            color: #1a1a2e;
            cursor: pointer;
        }
        .model-select:focus { outline: none; border-color: rgba(255,255,255,0.80); }

        /* Chat UI */
        #chat-messages {
            height: 480px;
            overflow-y: auto;
            display: none;
            flex-direction: column;
            gap: 12px;
            padding: 16px;
            background: rgba(59, 130, 246, 0.12);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .user-bubble {
            align-self: flex-end;
            max-width: 75%;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .ai-bubble {
            align-self: flex-start;
            max-width: 88%;
            background: white;
            border: 1px solid #e9ecef;
            padding: 16px;
            border-radius: 4px 18px 18px 18px;
            font-size: 14px;
            line-height: 1.6;
            word-wrap: break-word;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }
        .ai-answer-text { white-space: pre-wrap; margin-bottom: 12px; }
        .sources-section { border-top: 1px solid #e9ecef; padding-top: 10px; margin-top: 8px; }
        .sources-header { font-weight: 600; font-size: 13px; color: #495057; margin-bottom: 8px; }
        .source-item {
            background: #f8f9fa;
            border-left: 3px solid #007bff;
            padding: 8px 10px;
            margin: 6px 0;
            border-radius: 0 4px 4px 0;
            font-size: 12px;
            color: #555;
        }
        .source-meta { font-weight: 600; color: #333; margin-bottom: 3px; }
        .source-text {
            color: #666;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            white-space: pre-wrap;
        }
        .token-line {
            border-top: 1px solid #e9ecef;
            padding-top: 8px;
            margin-top: 10px;
            font-size: 11px;
            color: #888;
            font-family: 'Courier New', monospace;
        }
        .loading-bubble {
            align-self: flex-start;
            background: white;
            border: 1px solid #e9ecef;
            padding: 12px 16px;
            border-radius: 4px 18px 18px 18px;
            color: #888;
            font-size: 13px;
            font-style: italic;
        }
        .error-bubble {
            align-self: flex-start;
            max-width: 85%;
            background: #fff5f5;
            border: 1px solid #f5c6cb;
            padding: 12px 16px;
            border-radius: 4px 18px 18px 18px;
            color: #721c24;
            font-size: 13px;
        }
        #chat-input-bar { display: flex; gap: 8px; align-items: stretch; }
        #chat-input { flex: 1; margin: 0; resize: none; min-height: 44px; max-height: 120px; }
        #chat-send { white-space: nowrap; padding: 0 20px; flex-shrink: 0; align-self: stretch; }
        .settings-toggle {
            background: rgba(255, 255, 255, 0.28);
            border: 1px solid rgba(59, 130, 246, 0.45);
            color: rgba(255, 255, 255, 0.90);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            margin-bottom: 12px;
            backdrop-filter: blur(4px);
        }
        .settings-toggle:hover { background: rgba(255,255,255,0.40); transform: none; box-shadow: none; }
        #settings-panel {
            display: none;
            background: rgba(255, 255, 255, 0.34);
            border: 1px solid rgba(255, 255, 255, 0.30);
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
            font-size: 13px;
        }
        #settings-panel label { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-weight: 500; color: rgba(255,255,255,0.90); }
        #settings-panel input[type="number"],
        #settings-panel input[type="password"] { padding: 6px 10px; font-size: 13px; width: auto; }
        #settings-panel .model-select { padding: 6px 10px; font-size: 13px; }
        .btn-outline {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.45);
            color: rgba(255,255,255,0.85);
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 500;
        }
        .btn-outline:hover { background: rgba(255,255,255,0.28); transform: none; box-shadow: none; }
    </style>
</head>
<body>
    <div class="hero-card">
        <h1>🚀 pgvector RAG System</h1>
        <div class="stats">
            <strong>Powered by:</strong> PostgreSQL + pgvector for high-performance similarity search
        </div>
    </div>

    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('chat', this)">💬 Chat</button>
        <button class="tab-btn" onclick="switchTab('embed', this)">📤 Embed</button>
    </div>

    <div id="tab-chat" class="tab-panel active section">
        <h2>💬 Chat</h2>
        <p>Semantic search powered by sentence embeddings and pgvector cosine similarity.</p>
        <button class="settings-toggle" onclick="toggleSettings()">⚙️ Settings ▾</button>
        <div id="settings-panel">
            <label>Model:
                <select id="chat-model" class="model-select">
                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                    <option value="deepseek-r1:8b">DeepSeek R1 8B (Ollama)</option>
                    <option value="deepseek-r1:1.5b">DeepSeek R1 1.5B (Ollama)</option>
                    <option value="llama3.2:3b">Llama 3.2 3B (Ollama)</option>
                </select>
            </label>
            <label>Max Results: <input type="number" id="chat-limit" value="5" min="1" max="100" style="width:70px;"></label>
            <label>Threshold: <input type="number" id="chat-threshold" value="0.5" min="0.0" max="0.95" step="0.05" style="width:80px;"></label>
            <label>Table:
                <select id="chat-table" class="model-select" style="margin:0;padding:6px 10px;font-size:13px;">
                    <option value="document_chunks">document_chunks (loading...)</option>
                </select>
                <button type="button" onclick="loadTableList()" title="Refresh table list" style="margin-left:6px;padding:6px 10px;font-size:13px;cursor:pointer;">↻</button>
            </label>
            <label>Password: <input type="password" id="chat-password" placeholder="Required if configured"></label>
        </div>
        <div id="chat-messages"></div>
        <div id="chat-input-bar">
            <textarea id="chat-input" placeholder="Ask a question about your documents... (Ctrl+Enter to send)" rows="2"></textarea>
            <button id="chat-send" onclick="sendChat()">Send</button>
        </div>
        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <button type="button" class="btn-outline" onclick="clearChat()">🗑️ Clear</button>
            <a href="/stats" target="_blank"><button type="button" class="btn-outline">📊 Statistics</button></a>
            <a href="/health" target="_blank"><button type="button" class="btn-outline">🏥 Health</button></a>
        </div>
    </div>

    <div id="tab-embed" class="tab-panel section">
        <h2>📤 Upload & Process Document</h2>
        <p>Supported formats: PDF, DOCX, TXT. Documents are chunked semantically and stored with vector embeddings.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Document File</label>
                <input type="file" name="file" accept=".pdf,.docx,.txt" required>
            </div>
            <div class="form-group">
                <label>PDF Parsing Backend</label>
                <select name="parse_backend" class="model-select">
                    <option value="">Default (PyMuPDF only)</option>
                    <option value="gemini-docling">Gemini Vision (docling)</option>
                    <option value="ollama">Local LLM — qwen3.5:4b (Ollama)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Access Password</label>
                <input type="password" name="access_password" placeholder="Required if configured">
            </div>
            <div class="form-group">
                <label>Table Name</label>
                <input type="text" name="table_name" value="document_chunks">
            </div>
            <div class="form-group">
                <label>Chunk Size</label>
                <input type="number" name="chunk_size" value="512" min="128" max="2048">
            </div>
            <div style="margin-top: 18px;">
                <button type="submit">📤 Upload & Process</button>
            </div>
        </form>
    </div>

        <!-- Success/Error notification -->
    <div id="notification" class="notification">
        <span class="close" onclick="hideNotification()">x</span>
        <span id="notification-message"></span>
    </div>

    <script>
        function switchTab(tab, btn) {
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            btn.classList.add('active');
        }

        const PASSWORD_KEY = 'rag_access_password';

        function loadPassword() {
            try {
                return localStorage.getItem(PASSWORD_KEY) || '';
            } catch (e) {
                return '';
            }
        }

        function savePassword(value) {
            try {
                if (value) localStorage.setItem(PASSWORD_KEY, value);
            } catch (e) {
                // ignore storage issues
            }
        }

        function hydratePasswordInputs() {
            const saved = loadPassword();
            if (!saved) return;
            document.querySelectorAll('input[name="access_password"]').forEach((el) => {
                if (!el.value) el.value = saved;
            });
        }

        function cachePasswordFromForm(form) {
            const field = form.querySelector('input[name="access_password"]');
            if (field && field.value) savePassword(field.value);
        }

        hydratePasswordInputs();

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

            cachePasswordFromForm(this);
            hydratePasswordInputs();

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
                    showNotification(`Document "${result.filename}" uploaded and processed successfully! Document ID: ${result.document_id.substring(0,8)}...`, 'success');
                    this.reset(); // Clear the form
                    hydratePasswordInputs();
                } else {
                    showNotification(`Upload failed: ${result.detail}`, 'error');
                }
            } catch (error) {
                showNotification(`Upload failed: ${error.message}`, 'error');
            } finally {
                // Restore button
                submitButton.textContent = originalText;
                submitButton.disabled = false;
            }
        });

        // Check for URL parameters to show notifications (if redirected)
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('upload') === 'success') {
            showNotification('Document uploaded and processed successfully!', 'success');
        } else if (urlParams.get('upload') === 'error') {
            showNotification('Upload failed. Please try again.', 'error');
        }

        // ── Chat UI ──────────────────────────────────────────────────────────

        function toggleSettings() {
            const panel = document.getElementById('settings-panel');
            const btn = document.querySelector('.settings-toggle');
            if (panel.style.display === 'block') {
                panel.style.display = 'none';
                btn.textContent = '⚙️ Settings ▾';
            } else {
                panel.style.display = 'block';
                btn.textContent = '⚙️ Settings ▴';
                const saved = loadPassword();
                if (saved && !document.getElementById('chat-password').value) {
                    document.getElementById('chat-password').value = saved;
                }
            }
        }

        function escapeHtml(str) {
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }

        function clearChat() {
            const messages = document.getElementById('chat-messages');
            messages.innerHTML = '';
            messages.style.display = 'none';
        }

        function appendBubble(html, className) {
            const messages = document.getElementById('chat-messages');
            messages.style.display = 'flex';
            const div = document.createElement('div');
            div.className = className;
            div.innerHTML = html;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            return div;
        }

        async function sendChat() {
            const input = document.getElementById('chat-input');
            const query = input.value.trim();
            if (!query) return;

            const model = document.getElementById('chat-model').value;
            const limit = parseInt(document.getElementById('chat-limit').value) || 5;
            const threshold = parseFloat(document.getElementById('chat-threshold').value) || 0.5;
            const table_name = document.getElementById('chat-table').value || 'document_chunks';
            const password = document.getElementById('chat-password').value || loadPassword();
            if (password) savePassword(password);

            input.value = '';
            input.style.height = 'auto';
            const sendBtn = document.getElementById('chat-send');
            sendBtn.disabled = true;
            sendBtn.textContent = '...';

            appendBubble(escapeHtml(query), 'user-bubble');
            const loadingBubble = appendBubble('🤔 Thinking...', 'loading-bubble');

            try {
                const headers = { 'Content-Type': 'application/json' };
                if (password) headers['x-app-password'] = password;

                const res = await fetch('/query', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ query, limit, threshold, model, table_name })
                });

                loadingBubble.remove();

                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: res.statusText }));
                    appendBubble(`<strong>Error:</strong> ${escapeHtml(err.detail || 'Unknown error')}`, 'error-bubble');
                    return;
                }

                const data = await res.json();
                const stats = data.search_stats || {};

                // Build sources HTML
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    const items = data.sources.map((s, i) => {
                        const sim = (s.similarity * 100).toFixed(1);
                        const bm25 = s.bm25_score != null ? ` &nbsp;·&nbsp; BM25: ${s.bm25_score.toFixed(3)}` : '';
                        const rrf = s.rrf_score != null ? ` &nbsp;·&nbsp; RRF: ${s.rrf_score.toFixed(3)}` : '';
                        const rerank = s.rerank_score != null ? ` &nbsp;·&nbsp; Rerank: ${s.rerank_score.toFixed(3)}` : '';
                        const docId = s.document_id ? s.document_id.substring(0, 8) + '...' : 'N/A';
                        const page = s.page_number != null ? ` &nbsp;·&nbsp; Page ${s.page_number}` : '';
                        return `<div class="source-item">
                            <div class="source-meta">Source ${i + 1} &mdash; ${sim}% similarity${bm25}${rrf}${rerank} &nbsp;|&nbsp; Doc: ${escapeHtml(docId)}${page}</div>
                            <div class="source-text">${escapeHtml(s.text)}</div>
                        </div>`;
                    }).join('');
                    sourcesHtml = `<div class="sources-section">
                        <div class="sources-header">📚 Sources (${data.sources.length})</div>
                        ${items}
                    </div>`;
                }

                // Token line
                const inTok = stats.input_tokens;
                const outTok = stats.output_tokens;
                const totalTok = stats.total_tokens;
                const tokenLine = totalTok
                    ? `<div class="token-line">🪙 &uarr;&nbsp;${inTok != null ? inTok.toLocaleString() : '?'} in &nbsp;&middot;&nbsp; &darr;&nbsp;${outTok != null ? outTok.toLocaleString() : '?'} out &nbsp;&middot;&nbsp; &Sigma;&nbsp;${totalTok.toLocaleString()} total</div>`
                    : '';

                // Search stats line
                const chunksFound = stats.chunks_found != null ? stats.chunks_found : '?';
                const searchMethod = stats.search_method || '?';
                const tableUsed = data.table_used || '?';
                const statsLine = `<div class="token-line">🔍 ${chunksFound} chunks &nbsp;&middot;&nbsp; 📋 ${escapeHtml(tableUsed)} &nbsp;&middot;&nbsp; ⚙️ ${escapeHtml(searchMethod)}</div>`;

                appendBubble(
                    `<div class="ai-answer-text">${escapeHtml(data.answer || '')}</div>${sourcesHtml}${statsLine}${tokenLine}`,
                    'ai-bubble'
                );
            } catch (err) {
                loadingBubble.remove();
                appendBubble(`<strong>Error:</strong> ${escapeHtml(err.message)}`, 'error-bubble');
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }

        // Ctrl+Enter sends the message
        document.getElementById('chat-input').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                sendChat();
            }
        });

        // Hydrate password into chat panel on load
        (function() {
            const saved = loadPassword();
            if (saved) document.getElementById('chat-password').value = saved;
        })();

        // Populate table dropdown from /tables
        async function loadTableList() {
            try {
                const res = await fetch('/tables');
                if (!res.ok) return;
                const data = await res.json();
                const sel = document.getElementById('chat-table');
                const tables = data.tables || data.table_names || [];
                if (tables.length === 0) return;
                const current = sel.value;
                sel.innerHTML = tables.map(t => {
                    const name = typeof t === 'string' ? t : t.table_name;
                    return `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`;
                }).join('');
                // Restore previous selection if it still exists
                if (current && [...sel.options].some(o => o.value === current)) sel.value = current;
            } catch (_) { /* keep default option */ }
        }
        loadTableList();
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
        .answer h2 {{
            margin-top: 0;
            margin-bottom: 10px;
        }}
        .answer p {{
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.8;
            margin: 0;
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
            line-height: 1.8;
            word-wrap: break-word;
            overflow-wrap: break-word;
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
        <div class="stats" style="margin-top: 15px; margin-bottom: 0;">
            🪙 <strong>Tokens:</strong> {token_display}
        </div>
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
        • Threshold: {threshold_used}<br>
        • Response confidence: {confidence}<br>
        • Response word count: {word_count}<br>
        • Graph enriched: {graph_enriched}<br>
        • Tokens: {token_display}
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
            white-space: pre-wrap;
            word-wrap: break-word;
            text-align: left;
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
        <h2> Search Failed</h2>
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
            background:
                linear-gradient(rgba(10, 20, 35, 0.50), rgba(10, 20, 35, 0.50)),
                url('/images/moutain_pexcel.jpeg') center center / cover fixed;
            line-height: 1.6;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.24);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.40);
            border-top: 3px solid #3b82f6;
            padding: 20px;
            border-radius: 12px;
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
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-top: 2px solid rgba(59, 130, 246, 0.65);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #22d3ee;
            text-shadow: 0 0 14px rgba(34, 211, 238, 0.7);
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #ffffff;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .config-section {{
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-top: 2px solid rgba(59, 130, 246, 0.65);
            padding: 25px;
            border-radius: 12px;
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
            color: rgba(255, 255, 255, 0.85);
        }}
        .config-value {{
            color: #7dd3fc;
            text-shadow: 0 0 8px rgba(125, 211, 252, 0.5);
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
        h1 {{ color: #ffffff; text-shadow: 0 2px 8px rgba(0,0,0,0.4); margin-bottom: 10px; }}
        h2 {{ color: #ffffff; text-shadow: 0 1px 4px rgba(0,0,0,0.35); margin-bottom: 15px; border-left: 3px solid #3b82f6; padding-left: 10px; }}
        .stat-label {{ color: #ffffff; }}
        .refresh-note {{
            background: rgba(227, 242, 253, 0.15);
            border: 1px solid rgba(255,255,255,0.20);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: rgba(255,255,255,0.75);
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
        <div class="stat-card">
            <div class="stat-number">{total_tables}</div>
            <div class="stat-label">Tables</div>
        </div>
    </div>

    <div class="config-section">
        <h2>⚙️ System Configuration</h2>
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
        <h2> Failed to Load Statistics</h2>
        <p>Error: {error_message}</p>
        <a href="/"><button>← Back to Home</button></a>
    </div>
</body>
</html>
"""

HEALTH_CHECK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Check - pgvector RAG</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background:
                linear-gradient(rgba(10, 20, 35, 0.50), rgba(10, 20, 35, 0.50)),
                url('/images/moutain_pexcel.jpeg') center center / cover fixed;
            line-height: 1.6;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.24);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.40);
            border-top: 3px solid #3b82f6;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .status-main {{
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-left: 6px solid {status_color};
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .status-icon {{
            font-size: 4rem;
            margin-bottom: 15px;
        }}
        .status-text {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {status_color};
            text-shadow: 0 0 12px {status_color};
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .components-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .component-card {{
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-top: 2px solid rgba(59, 130, 246, 0.65);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        .component-status {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        .component-name {{
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
            margin-bottom: 5px;
        }}
        .component-detail {{
            color: #ffffff;
            font-size: 0.9rem;
        }}
        .metrics-section {{
            background: rgba(255, 255, 255, 0.32);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.44);
            border-top: 2px solid rgba(59, 130, 246, 0.65);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric-item {{
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.34);
            border-radius: 8px;
        }}
        .metric-number {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #22d3ee;
            text-shadow: 0 0 14px rgba(34, 211, 238, 0.7);
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #ffffff;
            font-size: 0.85rem;
            text-transform: uppercase;
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
            margin: 5px;
        }}
        h1 {{ color: #ffffff; text-shadow: 0 2px 8px rgba(0,0,0,0.4); margin-bottom: 10px; }}
        h2 {{ color: #ffffff; text-shadow: 0 1px 4px rgba(0,0,0,0.35); margin-bottom: 15px; border-left: 3px solid #3b82f6; padding-left: 10px; }}
        .timestamp {{
            background: rgba(227, 242, 253, 0.15);
            border: 1px solid rgba(255,255,255,0.20);
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            color: rgba(255,255,255,0.75);
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 System Health Check</h1>
        <p>Real-time status monitoring for your RAG system</p>
        <a href="/"><button>← Back to Home</button></a>
        <a href="/stats"><button>📊 View Statistics</button></a>
    </div>

    <div class="status-main">
        <div class="status-icon">{status_icon}</div>
        <div class="status-text">System {db_status_upper}</div>
    </div>

    <div class="components-grid">
        <div class="component-card">
            <div class="component-status">🗄️</div>
            <div class="component-name">Database</div>
            <div class="component-detail">PostgreSQL + pgvector</div>
            <div class="component-detail" style="color: {status_color}; font-weight: bold; text-shadow: 0 0 8px {status_color};">{db_status_upper}</div>
        </div>
        <div class="component-card">
            <div class="component-status">🧠</div>
            <div class="component-name">Embedding Model</div>
            <div class="component-detail">{embedding_model}</div>
            <div class="component-detail" style="color: #4ade80; font-weight: bold; text-shadow: 0 0 8px rgba(74,222,128,0.55);">LOADED</div>
        </div>
        <div class="component-card">
            <div class="component-status">🔍</div>
            <div class="component-name">Vector Store</div>
            <div class="component-detail">Table: {table_name}</div>
            <div class="component-detail" style="color: #4ade80; font-weight: bold; text-shadow: 0 0 8px rgba(74,222,128,0.55);">OPERATIONAL</div>
        </div>
    </div>

    <div class="metrics-section">
        <h2>📈 Quick Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-number">{total_documents}</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric-item">
                <div class="metric-number">{total_chunks}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-item">
                <div class="metric-number">{embedding_dim}</div>
                <div class="metric-label">Embedding Dim</div>
            </div>
            <div class="metric-item">
                <div class="metric-number">{avg_text_length}</div>
                <div class="metric-label">Avg Text Length</div>
            </div>
        </div>
    </div>

    <div class="timestamp">
        <strong>Last Updated:</strong> {timestamp} | <strong>Auto-refresh:</strong> Every 30 seconds
    </div>
</body>
</html>
"""

HEALTH_ERROR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Check Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 100px auto; padding: 20px; text-align: center; }}
        .error {{ background: #ffebee; padding: 30px; border-radius: 12px; border-left: 5px solid #f44336; }}
        .status-icon {{ font-size: 4rem; margin-bottom: 15px; }}
        button {{ background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="error">
        <div class="status-icon"></div>
        <h2>System Unhealthy</h2>
        <p><strong>Error:</strong> {error_message}</p>
        <p>The system is experiencing issues and may not function properly.</p>
        <a href="/"><button>← Back to Home</button></a>
    </div>
</body>
</html>
"""

