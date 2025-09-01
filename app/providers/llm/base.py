from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
import time
import logging


class LLMMessage(BaseModel):
    """Standardized message format for LLM communication"""
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Standardized response from LLM providers"""
    content: str
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    response_time_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._setup_client(**kwargs)

    @abstractmethod
    def _setup_client(self, **kwargs) -> None:
        """Initialize the provider-specific client"""
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the provider is healthy and responding"""
        pass

    def _create_response(self, content: str, **metadata) -> LLMResponse:
        """Helper method to create standardized responses"""
        return LLMResponse(
            content=content,
            model=self.model,
            **metadata
        )

    def _measure_time(self, start_time: float) -> float:
        """Helper to measure response time in milliseconds"""
        return (time.time() - start_time) * 1000

    async def _handle_error(self, error: Exception, context: str = "") -> None:
        """Standardized error handling and logging"""
        self.logger.error(f"LLM Provider error in {context}: {str(error)}")
        raise error


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class LLMProviderRateLimitError(LLMProviderError):
    """Raised when hitting rate limits"""
    pass


class LLMProviderAuthError(LLMProviderError):
    """Raised when authentication fails"""
    pass


class LLMProviderAPIError(LLMProviderError):
    """Raised when API request fails"""
    pass
