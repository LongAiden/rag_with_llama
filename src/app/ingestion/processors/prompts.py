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

# Bounded deliberately, second iteration. v1 halved output — mean 318 tok -> 160 tok,
# and 28 of 79 num_predict truncations -> 0. Its "Do NOT mention that something is
# absent" rule moved absence-reporting from 19 of 79 to 15 of 79 over the same
# 504-page run: real, but far short of the goal, and the rule kept the very thing
# that provokes it — an enumeration of "axis labels, legend entries, data values and
# text annotations" that the model echoes back item by item to say each is missing.
# (Measured with scripts/f21_census.py. An earlier count of 11 -> 12 came from a
# census that stopped at the first blank line and so only ever read paragraph 1.)
# "At most 3 short sentences (60 words)" produced a 95-word median, because a 0.8B
# model does not count words. v2 therefore deleted the checklist rather than negating
# it, and asked for a sentence count rather than a word count.
#
# v3 keeps v2's length win (mean 148 -> 105 tok over an 80-page probe) and fixes four
# things v2 broke or left broken, each measured on that probe:
#   * v2 said "The first names what the image is. The second states its single most
#     important content." The model recited that sentence back as output. Any rule
#     that describes the shape of the answer is recitable — so v3 states the bound
#     without narrating what each sentence should carry.
#   * descriptions landed on the </figure_type> line itself in 11 of 17 cases (F19: 2
#     of 79), so v3 demands the newline explicitly.
#   * 7 stray <p> tags appeared in 17 descriptions (F19: 2 in 79); nothing in v1 or v2
#     forbade HTML, only markdown headings and code fences.
#   * the page-1 cover became one fluent sentence of recalled knowledge about the book
#     — naming spaCy, which is not on the cover — with the whales and the banner
#     unmentioned. That is a grounding failure, not a length failure, so loosening the
#     sentence bound would not have fixed it. v3 forbids outside knowledge and adds
#     "Unclear image." as a licensed answer, so fabricating (an NLTK download dialog
#     became a spreadsheet of invented COPPER/CHROMOS/PHENOL columns) is no longer the
#     model's only way to fill the space.
# Latency is pure decode at ~42 tok/s, so output length IS latency — but num_predict
# only truncates, it never asks for brevity. That has to come from the prompt.
OLLAMA_IMAGE_PROMPT = """\
Look at this image from a PDF page.

Describe what you see inside <figure></figure> tags.

Rules:
- Start your output with <figure> and end with </figure>. Nothing outside these tags.
- On the first line inside <figure>, add: <figure_type>Chart|Diagram|Logo|Screenshot|Other</figure_type>
- Put the description on the NEXT line. Never on the <figure_type> line.
- Write at most two sentences, then stop.
- Describe only what is visible in this image. Do not use anything you know about the
  subject, the title, or the book from any other source.
- If you cannot read the image clearly, write exactly: Unclear image.
- If the image is a band of text, an equation, a rule, or a code listing — anything that
  is not a chart, diagram, logo or screenshot — output the visible text verbatim and
  nothing else: no sentences, no description of it.
- If the image shows a flowchart or process: describe the sequence (A → B → C).
- Output plain text. Do NOT use HTML tags (<p>, <div>, <span>), markdown headings
  (#, ##, ###), numbered lists, or code fences.
- Do NOT write any text, title, or commentary before <figure> or after </figure>.
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
- Summarize the relevant information in your own words. Do not copy sentences verbatim from the context.

Context:
{context}

Question: {query}

Answer:"""
