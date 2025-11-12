"""Shared cache helpers with Redis optional support."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from ..cache import cache as local_cache
from ..config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None  # type: ignore


def _default_serializer(value: Any) -> str:
    def _encode(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return obj

    return json.dumps(value, default=_encode)


def _default_deserializer(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


class HistoryCache:
    def __init__(self) -> None:
        self._ttl = settings.cache_ttl_seconds
        redis_url = getattr(settings, "redis_url", "")
        self._client = None
        if redis and redis_url:
            try:
                self._client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize Redis client", exc_info=exc)
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def get(self, key: str) -> Any:
        if self._client is not None:
            payload = await self._client.get(key)
            if payload:
                return _default_deserializer(payload)
        return await local_cache.get(key)

    async def set(self, key: str, value: Any, *, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._ttl
        if self._client is not None:
            payload = _default_serializer(value)
            await self._client.set(key, payload, ex=ttl)
        await local_cache.set(key, value, ttl=ttl)

    async def clear(self) -> None:
        if self._client is not None:
            await self._client.flushdb()
        await local_cache.clear()


history_cache = HistoryCache()


def history_cache_key(address: str, chain: str, start: str, end: str) -> str:
    return f"history:{chain}:{address}:{start}:{end}"


__all__ = ["history_cache", "history_cache_key"]
