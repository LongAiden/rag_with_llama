"""
Gemini LLM Provider implementation for graph processing.
"""

import logging
from typing import Any
from google import generativeai as genai

from .llm_provider import LLMProvider
from .retry_utils import retry_async_with_backoff
from config.graph_config import get_graph_config

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    """
    Gemini implementation of the LLM provider.
    
    Handles Gemini-specific API calls, retry logic, and response parsing.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini LLM provider.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name (default: gemini-2.5-flash)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Load retry configuration
        config = get_graph_config()
        self.max_retries = config.gemini_max_retries
        self.retry_initial_delay = config.gemini_retry_initial_delay
        self.retry_max_delay = config.gemini_retry_max_delay
        self.retry_exponential_base = config.gemini_retry_exponential_base
        self.rate_limit_pause = config.gemini_rate_limit_pause

    async def generate_content(self, prompt: str) -> Any:
        """
        Generate content using Gemini with retry logic.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Gemini response object
            
        Raises:
            Exception: If all retries are exhausted
        """
        @retry_async_with_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            exponential_base=self.retry_exponential_base,
            rate_limit_pause=self.rate_limit_pause
        )
        async def _call():
            return await self.model.generate_content_async(prompt)

        return await _call()

    def extract_text_from_response(self, response: Any) -> str:
        """
        Extract text from Gemini response.
        
        Args:
            response: Gemini response object
            
        Returns:
            Text content as string
        """
        return response.text

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "gemini"
