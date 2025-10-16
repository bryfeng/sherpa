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
)


class ZAIProvider(LLMProvider):
    """Z AI chat completion provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.6",
        *,
        base_url: str | None = None,
        timeout: float = 40.0,
        **kwargs: Any,
    ) -> None:
        self.base_url = (base_url or "https://api.z.ai").rstrip("/")
        self.timeout = timeout
        self._chat_completions_path = "/api/paas/v4/chat/completions"
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
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.time()

        payload = self._build_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=kwargs,
        )

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

        usage = data.get("usage", {})
        response_time = self._measure_time(start_time)
        return self._create_response(
            content=combined_content or content,
            tokens_used=usage.get("total_tokens"),
            finish_reason=choice.get("finish_reason"),
            response_time_ms=response_time,
        )

    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
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
        extra: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        if extra:
            # Allow callers to override defaults but avoid mutating caller dict
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value

        payload.setdefault("thinking", {"type": "enabled"})

        if stream:
            payload["stream"] = True

        return payload

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
