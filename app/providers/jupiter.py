"""
Jupiter Token List Provider for Solana.

Jupiter provides the most comprehensive token list for Solana, including:
- Strict list: ~1,500 curated, verified tokens
- All tokens: 20,000+ tokens including unverified

This provider caches the token list in memory for fast lookups.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .base import Provider
from ..config import settings


@dataclass
class JupiterToken:
    """Parsed Jupiter token metadata."""

    address: str  # Mint address (Base58)
    symbol: str
    name: str
    decimals: int
    logo_uri: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    coingecko_id: Optional[str] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "JupiterToken":
        """Parse a token from Jupiter API response."""
        extensions = data.get("extensions") or {}
        return cls(
            address=data.get("address", ""),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            decimals=data.get("decimals", 9),
            logo_uri=data.get("logoURI"),
            tags=data.get("tags") or [],
            coingecko_id=extensions.get("coingeckoId"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "symbol": self.symbol,
            "name": self.name,
            "decimals": self.decimals,
            "logo_uri": self.logo_uri,
            "tags": self.tags,
            "coingecko_id": self.coingecko_id,
        }


class JupiterProvider(Provider):
    """
    Jupiter token list and price provider for Solana.

    Provides:
    - Token metadata lookup by mint address
    - Token search by symbol
    - Price data via Jupiter Price API

    No API key required. Token list is cached in memory with configurable TTL.
    """

    name = "jupiter"
    timeout_s = 15

    STRICT_LIST_URL = "https://token.jup.ag/strict"
    ALL_LIST_URL = "https://token.jup.ag/all"
    PRICE_API_URL = "https://price.jup.ag/v6/price"

    def __init__(self) -> None:
        # Token caches
        self._strict_cache: Dict[str, JupiterToken] = {}  # address -> token
        self._symbol_index: Dict[str, List[JupiterToken]] = {}  # symbol.lower() -> [tokens]
        self._name_index: Dict[str, List[JupiterToken]] = {}  # name.lower() -> [tokens]

        # Cache metadata
        self._cache_loaded: bool = False
        self._cache_lock = asyncio.Lock()
        self._last_refresh: float = 0
        self._cache_ttl_seconds: int = getattr(settings, "jupiter_cache_ttl_seconds", 3600)

    async def ready(self) -> bool:
        """Jupiter API requires no authentication."""
        return getattr(settings, "enable_jupiter", True)

    async def health_check(self) -> Dict[str, Any]:
        """Check if Jupiter API is reachable."""
        if not await self.ready():
            return {"status": "disabled", "reason": "Jupiter provider disabled"}

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Quick check with limit=1
                resp = await client.get(
                    self.STRICT_LIST_URL,
                    timeout=5,
                )
                resp.raise_for_status()
                return {
                    "status": "healthy",
                    "latency_ms": int(resp.elapsed.total_seconds() * 1000),
                    "cached_tokens": len(self._strict_cache),
                }
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def _ensure_cache(self) -> None:
        """Load token list into memory if not cached or expired."""
        async with self._cache_lock:
            now = time.time()
            if self._cache_loaded and (now - self._last_refresh) < self._cache_ttl_seconds:
                return

            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.get(self.STRICT_LIST_URL)
                    resp.raise_for_status()
                    tokens_data = resp.json()

                # Clear existing caches
                self._strict_cache.clear()
                self._symbol_index.clear()
                self._name_index.clear()

                # Build indexes
                for item in tokens_data:
                    token = JupiterToken.from_api(item)
                    if not token.address:
                        continue

                    # Primary lookup by address
                    self._strict_cache[token.address] = token

                    # Symbol index (case-insensitive)
                    symbol_lower = token.symbol.lower()
                    if symbol_lower not in self._symbol_index:
                        self._symbol_index[symbol_lower] = []
                    self._symbol_index[symbol_lower].append(token)

                    # Name index (case-insensitive)
                    name_lower = token.name.lower()
                    if name_lower not in self._name_index:
                        self._name_index[name_lower] = []
                    self._name_index[name_lower].append(token)

                self._cache_loaded = True
                self._last_refresh = now

            except Exception:
                # If cache load fails but we have stale data, keep using it
                if not self._cache_loaded:
                    raise

    async def get_token_by_mint(self, mint_address: str) -> Optional[JupiterToken]:
        """
        Look up token by mint address.

        First checks the strict (curated) list, then falls back to the all list
        for unverified tokens.

        Args:
            mint_address: Solana token mint address (Base58).

        Returns:
            JupiterToken if found, None otherwise.
        """
        await self._ensure_cache()

        # Check strict cache first (fast path)
        if mint_address in self._strict_cache:
            return self._strict_cache[mint_address]

        # Fallback: query all list for specific address
        # This is slower but covers unverified tokens
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(self.ALL_LIST_URL)
                resp.raise_for_status()
                all_tokens = resp.json()

                for item in all_tokens:
                    if item.get("address") == mint_address:
                        token = JupiterToken.from_api(item)
                        # Cache this token for future lookups
                        self._strict_cache[mint_address] = token
                        return token
        except Exception:
            pass

        return None

    async def search_by_symbol(
        self,
        symbol: str,
        *,
        limit: int = 10,
        exact_match: bool = False,
    ) -> List[JupiterToken]:
        """
        Find tokens by symbol.

        Args:
            symbol: Token symbol to search for (case-insensitive).
            limit: Maximum number of results.
            exact_match: If True, only return exact symbol matches.

        Returns:
            List of matching JupiterToken objects.
        """
        await self._ensure_cache()

        symbol_lower = symbol.lower()

        if exact_match:
            # Only exact matches
            matches = self._symbol_index.get(symbol_lower, [])
        else:
            # Include prefix matches
            matches = []
            for sym, tokens in self._symbol_index.items():
                if sym == symbol_lower or sym.startswith(symbol_lower):
                    matches.extend(tokens)

        # Sort by relevance: exact matches first, then by name length
        def sort_key(t: JupiterToken) -> tuple:
            is_exact = t.symbol.lower() == symbol_lower
            return (not is_exact, len(t.name))

        matches.sort(key=sort_key)
        return matches[:limit]

    async def search_by_name(self, name: str, *, limit: int = 10) -> List[JupiterToken]:
        """
        Find tokens by name (partial match).

        Args:
            name: Token name to search for (case-insensitive).
            limit: Maximum number of results.

        Returns:
            List of matching JupiterToken objects.
        """
        await self._ensure_cache()

        name_lower = name.lower()
        matches = []

        for token_name, tokens in self._name_index.items():
            if name_lower in token_name:
                matches.extend(tokens)

        # Sort by relevance: exact matches first, then by name length
        def sort_key(t: JupiterToken) -> tuple:
            is_exact = t.name.lower() == name_lower
            return (not is_exact, len(t.name))

        matches.sort(key=sort_key)
        return matches[:limit]

    async def get_token_price(
        self,
        mint_address: str,
        *,
        vs_token: str = "So11111111111111111111111111111111111111112",
    ) -> Optional[float]:
        """
        Get current price from Jupiter Price API.

        Args:
            mint_address: Token mint address.
            vs_token: Quote token (default: wrapped SOL).

        Returns:
            Price as float, or None if unavailable.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(
                    self.PRICE_API_URL,
                    params={"ids": mint_address, "vsToken": vs_token},
                )
                resp.raise_for_status()
                data = resp.json()

                price_data = data.get("data", {}).get(mint_address, {})
                return price_data.get("price")
        except Exception:
            return None

    async def get_multiple_prices(
        self,
        mint_addresses: List[str],
    ) -> Dict[str, Optional[float]]:
        """
        Get prices for multiple tokens in a single request.

        Args:
            mint_addresses: List of token mint addresses.

        Returns:
            Dict mapping mint address to price (or None if unavailable).
        """
        if not mint_addresses:
            return {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(
                    self.PRICE_API_URL,
                    params={"ids": ",".join(mint_addresses)},
                )
                resp.raise_for_status()
                data = resp.json()

                prices = {}
                for addr in mint_addresses:
                    price_data = data.get("data", {}).get(addr, {})
                    prices[addr] = price_data.get("price")
                return prices
        except Exception:
            return {addr: None for addr in mint_addresses}

    def get_cached_token_count(self) -> int:
        """Return the number of tokens in the cache."""
        return len(self._strict_cache)

    def clear_cache(self) -> None:
        """Clear the token cache (useful for testing)."""
        self._strict_cache.clear()
        self._symbol_index.clear()
        self._name_index.clear()
        self._cache_loaded = False
        self._last_refresh = 0


__all__ = [
    "JupiterProvider",
    "JupiterToken",
]
