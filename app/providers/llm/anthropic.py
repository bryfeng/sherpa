import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import time
import logging

from .base import (
    LLMProvider, LLMMessage, LLMResponse,
    LLMProviderError, LLMProviderAPIError, LLMProviderAuthError, LLMProviderRateLimitError,
    ToolDefinition, ToolCall, ToolResult
)

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider implementation with native tool calling support"""

    supports_tools: bool = True

    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        if not anthropic:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        if not model:
            raise ValueError("AnthropicProvider requires a model to be specified")

        super().__init__(api_key, model, **kwargs)

    def _setup_client(self, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            self.client = AsyncAnthropic(api_key=self.api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            raise LLMProviderAuthError(f"Failed to initialize Anthropic client: {e}")

    def _convert_message_to_anthropic(self, msg: LLMMessage) -> Optional[Dict[str, Any]]:
        """Convert a single LLMMessage to Anthropic format"""
        if msg.role == "system":
            return None  # System messages handled separately

        if msg.role == "tool_result" and msg.tool_result:
            # Tool result message
            return {
                "role": "user",
                "content": [msg.tool_result.to_anthropic_format()]
            }

        if msg.role == "assistant" and msg.tool_calls:
            # Assistant message with tool calls
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments
                })
            return {"role": "assistant", "content": content}

        # Regular text message
        return {
            "role": msg.role,
            "content": msg.content or ""
        }

    async def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from Claude with optional tool calling"""
        start_time = time.time()

        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    converted = self._convert_message_to_anthropic(msg)
                    if converted:
                        anthropic_messages.append(converted)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or 4000,
            }

            if system_message:
                request_params["system"] = system_message

            if temperature is not None:
                request_params["temperature"] = temperature

            # Add tools if provided
            if tools:
                request_params["tools"] = [t.to_anthropic_format() for t in tools]

            # Add any additional parameters (but not 'tools' again)
            kwargs.pop('tools', None)
            request_params.update(kwargs)

            # Make the API call
            response = await self.client.messages.create(**request_params)

            # Extract response content and tool calls
            content = ""
            tool_calls = []

            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                    elif hasattr(block, 'type') and block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if hasattr(block, 'input') else {}
                        ))

            response_time = self._measure_time(start_time)

            # Determine finish reason
            finish_reason = response.stop_reason if hasattr(response, 'stop_reason') else None
            if finish_reason == "tool_use":
                finish_reason = "tool_use"
            elif finish_reason == "end_turn":
                finish_reason = "end_turn"

            return LLMResponse(
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else None,
                model=self.model,
                finish_reason=finish_reason,
                response_time_ms=response_time
            )

        except anthropic.AuthenticationError as e:
            await self._handle_error(LLMProviderAuthError(f"Authentication failed: {e}"), "generate_response")
        except anthropic.RateLimitError as e:
            await self._handle_error(LLMProviderRateLimitError(f"Rate limit exceeded: {e}"), "generate_response")
        except anthropic.APIError as e:
            await self._handle_error(LLMProviderAPIError(f"API error: {e}"), "generate_response")
        except Exception as e:
            await self._handle_error(LLMProviderError(f"Unexpected error: {e}"), "generate_response")

    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Claude

        Note: Streaming with tool calls has limited support - tool calls may not be
        properly streamed. Use generate_response for full tool calling support.
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    converted = self._convert_message_to_anthropic(msg)
                    if converted:
                        anthropic_messages.append(converted)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or 4000,
            }

            if system_message:
                request_params["system"] = system_message

            if temperature is not None:
                request_params["temperature"] = temperature

            # Add tools if provided
            if tools:
                request_params["tools"] = [t.to_anthropic_format() for t in tools]

            kwargs.pop('tools', None)
            request_params.update(kwargs)

            # Make the streaming API call
            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        yield event.delta.text

        except anthropic.AuthenticationError as e:
            await self._handle_error(LLMProviderAuthError(f"Authentication failed: {e}"), "generate_streaming_response")
        except anthropic.RateLimitError as e:
            await self._handle_error(LLMProviderRateLimitError(f"Rate limit exceeded: {e}"), "generate_streaming_response")
        except anthropic.APIError as e:
            await self._handle_error(LLMProviderAPIError(f"API error: {e}"), "generate_streaming_response")
        except Exception as e:
            await self._handle_error(LLMProviderError(f"Unexpected error: {e}"), "generate_streaming_response")

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Anthropic doesn't provide a direct token counting API
        # Using a rough approximation: ~4 characters per token
        return len(text) // 4

    async def health_check(self) -> Dict[str, Any]:
        """Check if Anthropic API is healthy"""
        try:
            start_time = time.time()
            
            # Simple test message to check API connectivity
            test_messages = [
                LLMMessage(role="user", content="Hello")
            ]
            
            response = await self.generate_response(
                messages=test_messages,
                max_tokens=10,
                temperature=0
            )
            
            response_time = self._measure_time(start_time)
            
            return {
                "status": "healthy",
                "provider": "anthropic",
                "model": self.model,
                "response_time_ms": response_time,
                "test_response_length": len(response.content)
            }
            
        except LLMProviderAuthError:
            return {
                "status": "error",
                "provider": "anthropic",
                "model": self.model,
                "error": "Authentication failed"
            }
        except LLMProviderRateLimitError:
            return {
                "status": "error",
                "provider": "anthropic",
                "model": self.model,
                "error": "Rate limit exceeded"
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": "anthropic",
                "model": self.model,
                "error": str(e)
            }
