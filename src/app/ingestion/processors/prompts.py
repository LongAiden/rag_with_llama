# ── Gemini VLM prompts ────────────────────────────────────────────────────────

VLM_TABLE_PROMPT = """\
This is a cropped region from a PDF page containing a table.
Extract and render the table content.

Rules:
- Wrap the output in <table></table> tags (closing tag must be </table>).
- Optionally include <table_caption>Title</table_caption> before the table \
if a caption or title is visible above or below it.
- Use GitHub-flavoured markdown table syntax.
- Separator row must use ONLY dashes: |-|-| (no colons, no padding spaces).
- Do NOT use any HTML inside cells (no <br>, no <td>, no <tr>).
- If a cell contains multiple lines of text, join them with a single space \
in the SAME cell.
- Ensure every row has the same number of columns.
- If a cell is merged vertically (one value applies to multiple rows), keep \
the value in the top row only.
- If the table has multiple header rows, include all of them in order.
- Preserve all cell values exactly as they appear.
- Output only the <table>...</table> block — no commentary or preamble.
- Do NOT use code fences.
"""

VLM_IMAGE_PROMPT = """\
This is an image, chart, diagram, figure, or visual element cropped from a PDF page.
Extract and preserve ALL content inside <figure></figure> tags.

Rules:
- Transcribe ALL visible text EXACTLY as it appears, in its original language.
  Do NOT translate, paraphrase, or describe text — copy it verbatim.
- If the image contains Japanese text, output it in Japanese. NEVER translate it to English.
- Do NOT write any English description or summary of the diagram structure.
- Preserve logical reading order (top-to-bottom, left-to-right within each column).
- For diagrams or flowcharts: transcribe ONLY the visible text labels, captions,
  and annotations verbatim in the original language. No English descriptions.
- For charts/graphs: transcribe all axis labels, legend entries, and data values.
- For colored boxes, banners, or callout regions: include ALL text inside them.
- If the image also contains a table, render it as a GFM markdown table.
- ALL transcribed text must be placed inside the <figure>...</figure> block.
- Output only the <figure>...</figure> block — no commentary or preamble.
- Do NOT use markdown headings (#, ##, ###) anywhere in the output.
- Do NOT write any text before <figure> or after </figure>.
- Do NOT use code fences.
"""

# ── Ollama VLM prompts (simpler, fallback-friendly) ──────────────────────────

OLLAMA_IMAGE_PROMPT = """\
Look at this image from a PDF page.

Describe what you see inside <figure></figure> tags.

Rules:
- Start your output with <figure> and end with </figure>. Nothing outside these tags.
- On the first line inside <figure>, add: <figure_type>Chart|Diagram|Logo|Screenshot|Other</figure_type>
- Describe all visible content: axis labels, legend entries, data values, text annotations, flow steps.
- If the image shows a flowchart or process: describe the sequence (A → B → C).
- Do NOT use markdown headings (#, ##, ###) anywhere in the output.
- Do NOT write any text, title, or commentary before <figure> or after </figure>.
- Do NOT use code fences.
Output only the <figure>...</figure> block.
"""

OLLAMA_TABLE_PROMPT = """\
Look at this image from a PDF page. It contains a table.

Extract the table content as a GitHub-flavoured markdown table inside <table></table> tags.

Rules:
- Use | col1 | col2 | syntax.
- Separator row uses only dashes: |---|---|
- Join multi-line cell text with a single space.
- If the table is unreadable, respond with exactly: <table>[TABLE]</table>

Output only the <table>...</table> block. No extra commentary.
"""

# ── RAG generation prompt templates ──────────────────────────────────────────

OLLAMA_RAG_PROMPT_TEMPLATE = """\
You are a RAG assistant. Answer the question using ONLY the provided context below.

Context rules:
- Blocks labelled [Section context: ...] contain ALL chunks from a document section \
in order. Use them to answer structural questions (counts, lists, enumeration).
- Blocks labelled [Source N] are the top retrieved chunks with their page context.
- If a [Section context] block is present, prefer it over individual sources for \
counting or listing tasks.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Never make up information not present in the context.
- Cite page numbers when available (e.g. "Page 3").

Context:
{context}

Question: {query}

Answer:"""
