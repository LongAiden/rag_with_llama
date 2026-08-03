"""
LLM Provider abstraction for graph processing.
Decouples entity/relationship extraction from specific LLM implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers used in graph processing.
    
    This allows the graph processing module to work with different LLM providers
    (Gemini, OpenAI, Claude, local models) without changing the core logic.
    """

    @abstractmethod
    async def generate_content(self, prompt: str) -> Any:
        """
        Generate content from the LLM based on a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response object (provider-specific)
            
        Raises:
            Exception: If the LLM call fails
        """
        pass

    @abstractmethod
    def extract_text_from_response(self, response: Any) -> str:
        """
        Extract text content from the LLM response.
        
        Args:
            response: The response object from generate_content
            
        Returns:
            Text content as string
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the LLM provider.
        
        Returns:
            Provider name (e.g., "gemini", "openai", "claude")
        """
        pass
