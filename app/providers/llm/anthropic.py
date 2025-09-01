import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import time
import logging

from .base import LLMProvider, LLMMessage, LLMResponse, LLMProviderError, LLMProviderAPIError, LLMProviderAuthError, LLMProviderRateLimitError

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider implementation"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        if not anthropic:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        super().__init__(api_key, model, **kwargs)

    def _setup_client(self, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            self.client = AsyncAnthropic(api_key=self.api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            raise LLMProviderAuthError(f"Failed to initialize Anthropic client: {e}")

    async def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from Claude"""
        start_time = time.time()
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    # Anthropic handles system messages separately
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
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
            
            # Add any additional parameters
            request_params.update(kwargs)
            
            # Make the API call
            response = await self.client.messages.create(**request_params)
            
            # Extract response content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            response_time = self._measure_time(start_time)
            
            return self._create_response(
                content=content,
                tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else None,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else None,
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
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Claude"""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or 4000,
                "stream": True,
            }
            
            if system_message:
                request_params["system"] = system_message
            
            if temperature is not None:
                request_params["temperature"] = temperature
            
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
