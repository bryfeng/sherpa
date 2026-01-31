"""Async LLM provider integration for Z AI chat completion API."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

import httpx

from .base import (
    LLMMessage,
    LLMProvider,
    LLMProviderAPIError,
    LLMProviderAuthError,
    LLMProviderError,
    LLMProviderRateLimitError,
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class ZAIProvider(LLMProvider):
    """Z AI chat completion provider with function/tool calling support.

    GLM-4 models support OpenAI-compatible function calling.
    """

    supports_tools: bool = True

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        *,
        base_url: str | None = None,
        timeout: float = 40.0,
        **kwargs: Any,
    ) -> None:
        self.base_url = (base_url or "https://api.z.ai").rstrip("/")
        self.timeout = timeout
        self._chat_completions_path = "/api/paas/v4/chat/completions"
        if not model:
            raise ValueError("ZAIProvider requires a model to be specified")
        super().__init__(api_key, model, **kwargs)

    def _setup_client(self, **kwargs: Any) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )

    async def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await self._client.post(path, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            message = exc.response.text
            if status in (401, 403):
                raise LLMProviderAuthError(f"Z AI authentication failed: {message}") from exc
            if status == 429:
                raise LLMProviderRateLimitError("Z AI rate limit exceeded") from exc
            raise LLMProviderAPIError(f"Z AI API error ({status}): {message}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderAPIError(f"Z AI request error: {exc}") from exc

    async def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.time()

        payload = self._build_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            extra=kwargs,
        )

        if tools:
            self.logger.info(f"ZAI request with {len(tools)} tools: {[t.name for t in tools]}")

        data = await self._post(self._chat_completions_path, json=payload)

        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError("Z AI response missing choices")

        choice = choices[0]
        message = choice.get("message") or {}
        content = self._normalize_content(message.get("content"))
        reasoning = self._normalize_content(message.get("reasoning_content"))
        combined_segments = [segment for segment in (content, reasoning) if segment]
        combined_content = "\n".join(combined_segments)

        # Extract tool calls if present (OpenAI-compatible format)
        tool_calls_data = message.get("tool_calls") or []
        tool_calls = []
        for tc in tool_calls_data:
            if tc.get("type") == "function":
                func = tc.get("function", {})
                # Parse arguments - they come as a JSON string
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse tool arguments: {args_str}")
                    args = {}

                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{len(tool_calls)}"),
                    name=func.get("name", ""),
                    arguments=args,
                ))

        if tool_calls:
            self.logger.info(f"ZAI returned {len(tool_calls)} tool call(s): {[tc.name for tc in tool_calls]}")

        usage = data.get("usage", {})
        response_time = self._measure_time(start_time)

        finish_reason = choice.get("finish_reason")
        if tool_calls:
            finish_reason = "tool_use"

        return LLMResponse(
            content=combined_content or content or None,
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=usage.get("total_tokens"),
            model=self.model,
            finish_reason=finish_reason,
            response_time_ms=response_time,
        )

    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        # Note: Streaming with tool calls may not work properly
        # Tool calls are better handled through generate_response
        payload = self._build_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            extra=kwargs,
            stream=True,
        )

        async with self._client.stream("POST", self._chat_completions_path, json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                message = exc.response.text
                if status in (401, 403):
                    raise LLMProviderAuthError(f"Z AI authentication failed: {message}") from exc
                if status == 429:
                    raise LLMProviderRateLimitError("Z AI rate limit exceeded") from exc
                raise LLMProviderAPIError(f"Z AI API error ({status}): {message}") from exc

            buffer = ""

            async for raw_line in response.aiter_lines():
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                # Server-Sent Events prefix
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    # Non-default event types not used currently
                    continue

                data_line = line
                if line.startswith("data:"):
                    data_line = line[len("data:") :].strip()

                if not data_line:
                    continue

                if data_line == "[DONE]":
                    break

                buffer += data_line
                try:
                    payload_json = json.loads(buffer)
                    buffer = ""
                except json.JSONDecodeError:
                    # Wait for the rest of the chunk
                    buffer += ""
                    continue

                choices = payload_json.get("choices") or []
                if not choices:
                    continue

                for choice in choices:
                    delta = choice.get("delta") or {}
                    for segment in self._iter_text_segments(delta):
                        if segment:
                            yield segment

                    message_obj = choice.get("message")
                    if message_obj:
                        for segment in self._iter_text_segments(message_obj):
                            if segment:
                                yield segment

    def count_tokens(self, text: str) -> int:
        # Approximate: 4 characters per token
        return max(1, len(text) // 4)

    async def health_check(self) -> Dict[str, Any]:
        try:
            response = await self.generate_response(
                messages=[LLMMessage(role="user", content="ping")],
                max_tokens=4,
                temperature=0.0,
            )
            return {
                "status": "healthy",
                "provider": "zai",
                "model": self.model,
                "response_preview": response.content[:32],
            }
        except LLMProviderAuthError as exc:
            return {
                "status": "error",
                "provider": "zai",
                "model": self.model,
                "error": str(exc),
            }
        except LLMProviderRateLimitError:
            return {
                "status": "degraded",
                "provider": "zai",
                "model": self.model,
                "error": "rate_limited",
            }
        except Exception as exc:
            return {
                "status": "error",
                "provider": "zai",
                "model": self.model,
                "error": str(exc),
            }

    async def __aenter__(self) -> "ZAIProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.aclose()

    async def close(self) -> None:
        await self._client.aclose()

    def _build_payload(
        self,
        *,
        messages: List[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        tools: Optional[List[ToolDefinition]] = None,
        extra: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        # Convert messages to OpenAI-compatible format
        converted_messages = []
        for msg in messages:
            if msg.role == "tool_result" and msg.tool_result:
                # Tool result message
                content = msg.tool_result.error if msg.tool_result.error else msg.tool_result.result
                if not isinstance(content, str):
                    content = json.dumps(content)
                converted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_result.tool_call_id,
                    "content": content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                tool_calls_formatted = []
                for tc in msg.tool_calls:
                    tool_calls_formatted.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    })
                converted_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls_formatted,
                })
            else:
                # Regular message
                converted_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                })

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add tools in OpenAI-compatible format
        if tools:
            payload["tools"] = [self._convert_tool_to_openai_format(t) for t in tools]
            # Disable thinking when using tools to avoid confusion
            payload["thinking"] = {"type": "disabled"}
        else:
            payload.setdefault("thinking", {"type": "enabled"})

        if extra:
            # Allow callers to override defaults but avoid mutating caller dict
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value

        if stream:
            payload["stream"] = True

        return payload

    def _convert_tool_to_openai_format(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert ToolDefinition to OpenAI-compatible function format."""
        properties = {}
        required = []

        for param in tool.parameters:
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
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _iter_text_segments(self, payload: Dict[str, Any]) -> Iterable[str]:
        if not isinstance(payload, dict):
            return []

        segments: List[str] = []
        for key in ("content", "reasoning_content"):
            if key not in payload:
                continue
            normalized = self._normalize_content(payload.get(key))
            if normalized:
                segments.append(normalized)
        return segments

    def _normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item and item["text"] is not None:
                        parts.append(str(item["text"]))
                    elif "content" in item and item["content"] is not None:
                        parts.append(str(item["content"]))
            return "".join(parts)
        return str(content)
