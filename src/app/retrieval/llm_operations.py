"""
LLM operations for the RAG application.
Handles LLM-based response generation with fallback mechanisms.
"""

import asyncio
import os
import logfire
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    langfuse_context = type("_Noop", (), {
        "update_current_trace": staticmethod(lambda **_: None),
        "update_current_observation": staticmethod(lambda **_: None),
    })()
    def observe(**__):
        def decorator(fn): return fn
        return decorator

from app.models.schemas import SimpleRAGResponse
from app.ingestion.processors.prompts import OLLAMA_RAG_PROMPT_TEMPLATE


class OllamaBackend:
    def __init__(self, model: str):
        self.model = model

    async def generate(self, query: str, context: str, results: list) -> SimpleRAGResponse:
        import httpx
        from app.config.app_config import AppSettings

        # Single source of truth: AppSettings also reads .env, which os.environ does not.
        ollama_base_url = AppSettings().ollama_base_url.rstrip("/")
        prompt = OLLAMA_RAG_PROMPT_TEMPLATE.format(context=context, query=query)
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ollama_base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            if resp.status_code != 200:
                body = resp.text
                print(f"Ollama error {resp.status_code}: {body}", flush=True)
                resp.raise_for_status()
            data = resp.json()
            answer = data["response"]
            input_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")
            total_tokens = (input_tokens or 0) + (output_tokens or 0) or None
        return SimpleRAGResponse(
            answer=answer,
            confidence=None,
            word_count=len(answer.split()),
            sources_used=len(results),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata={"method": "ollama", "model": self.model}
        )


class GeminiBackend:
    def __init__(self, model: str):
        self.model = model

    async def generate(self, query: str, context: str, results: list) -> SimpleRAGResponse:
        import google.generativeai as genai

        sources_used = len(results)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        genai.configure(api_key=api_key)

        prompt = f"""You are a RAG assistant. Answer the question using ONLY the provided context below.

Context rules:
- Blocks labelled [Section context: ...] contain ALL chunks from a document section in order. \
Use them to answer structural questions (counts, lists, enumeration).
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

        try:
            model = genai.GenerativeModel(self.model)
            # generate_content is synchronous and blocks for the full LLM latency.
            # Called directly it would stall the event loop and serialise every
            # concurrent request behind this one.
            response = await asyncio.to_thread(model.generate_content, prompt)
            answer = response.text

            usage = response.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', None)
            output_tokens = getattr(usage, 'candidates_token_count', None)
            # Calculate total from input + output to avoid Gemini's cached token inflation
            total_tokens = (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None

            logfire.info("Gemini response generated", model=self.model, sources_used=sources_used,
                        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)

            return SimpleRAGResponse(
                answer=answer,
                confidence=0.9,
                word_count=len(answer.split()),
                sources_used=sources_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                metadata={"method": "gemini", "model": self.model}
            )
        except Exception as llm_error:
            logfire.error("Gemini generation failed", error=str(llm_error), model=self.model)
            print(f"Gemini failed: {llm_error}", flush=True)
            raise


def _get_backend(model: str) -> OllamaBackend | GeminiBackend:
    if model.startswith("gemini-"):
        return GeminiBackend(model)
    return OllamaBackend(model)


async def generate_llm_response(
    query: str,
    context: str,
    results: list,
    model: str = "gemini-2.5-flash",
) -> SimpleRAGResponse:
    """Clean implementation — no Langfuse decorator here to avoid serializing heavy objects."""
    return await _traced_generate(query, context, results, model)


@observe(name="llm_generate", as_type="generation")
async def _traced_generate(
    query: str,
    context: str,
    results: list,
    model: str,
) -> SimpleRAGResponse:
    """Langfuse-traced LLM generation wrapper.

    Only receives serialisable primitives (str, list, int) so @observe
    never attempts to serialise a PyTorch model or Agent object.
    """
    backend = _get_backend(model)
    logfire.info("LLM request", model=model, backend=type(backend).__name__, results_count=len(results))
    response = await backend.generate(query, context, results)

    # Report usage to Langfuse with explicit input/output/total breakdown
    # Langfuse expects: {"input": int, "output": int, "total": int, "unit": "TOKENS"}
    usage_dict = {
        "input": response.input_tokens or 0,
        "output": response.output_tokens or 0,
        "total": response.total_tokens or 0,
        "unit": "TOKENS",
    }

    langfuse_context.update_current_observation(
        model=model,
        usage=usage_dict,
        metadata={
            "backend": type(backend).__name__.lower(),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        },
    )
    return response
