"""Service that caches and serves trending EVM token information."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..providers.coingecko import CoingeckoProvider


class TrendingTokenService:
    """Fetch and cache trending EVM tokens from Coingecko."""

    def __init__(
        self,
        *,
        provider: Optional[CoingeckoProvider] = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        self._provider = provider or CoingeckoProvider()
        self._cache_ttl = timedelta(seconds=max(10, cache_ttl_seconds))
        self._cache: Optional[tuple[datetime, List[Dict[str, Any]]]] = None
        self._lock = asyncio.Lock()

    async def get_trending_tokens(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Return cached trending tokens, refreshing as needed."""

        if limit <= 0:
            return []

        async with self._lock:
            now = datetime.now(timezone.utc)
            if self._cache and now - self._cache[0] < self._cache_ttl:
                data = self._cache[1]
            else:
                data = await self._provider.get_trending_evm_tokens(limit=limit + 5)
                self._cache = (now, data)

        return data[:limit]


trending_token_service = TrendingTokenService()


async def get_trending_tokens(*, limit: int = 10) -> Sequence[Dict[str, Any]]:
    """Module-level helper that proxies to the singleton service."""

    return await trending_token_service.get_trending_tokens(limit=limit)


__all__ = [
    'TrendingTokenService',
    'trending_token_service',
    'get_trending_tokens',
]

