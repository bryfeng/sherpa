from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import time
import logging


# =============================================================================
# Tool Calling Models
# =============================================================================

class ToolParameterType(str, Enum):
    """Supported parameter types for tool definitions"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a single tool parameter"""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the LLM"""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic's tool schema format"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


class ToolCall(BaseModel):
    """A tool call requested by the LLM"""
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool"""
    tool_call_id: str
    result: Any
    error: Optional[str] = None

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic's tool_result format"""
        content = self.error if self.error else self.result
        if not isinstance(content, str):
            import json
            content = json.dumps(content)
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": content,
            "is_error": self.error is not None,
        }


# =============================================================================
# Message Models
# =============================================================================

class LLMMessage(BaseModel):
    """Standardized message format for LLM communication"""
    role: str  # "system", "user", "assistant", "tool_result"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool use
    tool_result: Optional[ToolResult] = None      # For tool result messages


class LLMResponse(BaseModel):
    """Standardized response from LLM providers"""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None  # "end_turn", "tool_use", "max_tokens"
    response_time_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    # Class attribute indicating if provider supports native tool calling
    supports_tools: bool = False

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
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM

        Args:
            messages: List of messages in the conversation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and/or tool_calls
        """
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM

        Note: Streaming with tool calls may not be fully supported by all providers.
        """
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
