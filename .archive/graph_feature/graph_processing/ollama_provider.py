"""
Ollama LLM Provider implementation for graph processing.
"""

import httpx
import logging
from typing import Any

from .llm_provider import LLMProvider
from .retry_utils import retry_async_with_backoff
from config.graph_config import get_graph_config

logger = logging.getLogger(__name__)


class OllamaLLMProvider(LLMProvider):
    """
    Ollama implementation of the LLM provider.
    Calls a locally-hosted Ollama model via its HTTP API.
    """

    def __init__(self, base_url: str, model_name: str = "deepseek-r1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

        config = get_graph_config()
        self.timeout = config.gemini_request_timeout
        self.max_retries = config.gemini_max_retries
        self.retry_initial_delay = config.gemini_retry_initial_delay
        self.retry_max_delay = config.gemini_retry_max_delay
        self.retry_exponential_base = config.gemini_retry_exponential_base
        self.rate_limit_pause = config.gemini_rate_limit_pause

    async def generate_content(self, prompt: str) -> Any:
        @retry_async_with_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            exponential_base=self.retry_exponential_base,
            rate_limit_pause=self.rate_limit_pause
        )
        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                return response.json()

        return await _call()

    def extract_text_from_response(self, response: Any) -> str:
        return response["response"]

    def get_provider_name(self) -> str:
        return "ollama"
