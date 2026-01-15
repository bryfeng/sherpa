"""
Chain configuration service - queries the Convex chains table.

This service provides chain metadata including Alchemy slugs for multi-chain support.
"""

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache
import time

from app.db.convex_client import get_convex_client, ConvexQueryError


@dataclass
class ChainConfig:
    """Chain configuration from Convex."""
    chain_id: int
    name: str
    aliases: List[str]
    alchemy_slug: Optional[str]
    alchemy_verified: bool
    native_symbol: str
    native_decimals: int
    is_testnet: bool
    is_enabled: bool
    explorer_url: Optional[str]

    @classmethod
    def from_convex(cls, data: Dict[str, Any]) -> "ChainConfig":
        """Create ChainConfig from Convex response."""
        return cls(
            chain_id=data["chainId"],
            name=data["name"],
            aliases=data.get("aliases", []),
            alchemy_slug=data.get("alchemySlug"),
            alchemy_verified=data.get("alchemyVerified", False),
            native_symbol=data["nativeSymbol"],
            native_decimals=data["nativeDecimals"],
            is_testnet=data.get("isTestnet", False),
            is_enabled=data.get("isEnabled", True),
            explorer_url=data.get("explorerUrl"),
        )


class ChainService:
    """
    Service for fetching chain configurations from Convex.

    Implements caching to avoid repeated Convex queries.
    """

    # Cache TTL in seconds (5 minutes)
    CACHE_TTL = 300

    def __init__(self):
        self._cache: Dict[str, ChainConfig] = {}
        self._cache_by_id: Dict[int, ChainConfig] = {}
        self._all_chains: List[ChainConfig] = []
        self._cache_timestamp: float = 0
        self._lock = asyncio.Lock()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (
            self._cache_timestamp > 0
            and (time.time() - self._cache_timestamp) < self.CACHE_TTL
        )

    async def _refresh_cache(self) -> None:
        """Refresh the chain cache from Convex."""
        async with self._lock:
            # Double-check after acquiring lock
            if self._is_cache_valid():
                return

            try:
                client = get_convex_client()
                chains_data = await client.query("chains:listEnabled", {})

                # Clear existing cache
                self._cache.clear()
                self._cache_by_id.clear()
                self._all_chains.clear()

                for chain_data in chains_data or []:
                    config = ChainConfig.from_convex(chain_data)
                    self._all_chains.append(config)
                    self._cache_by_id[config.chain_id] = config

                    # Index by all aliases (lowercase)
                    for alias in config.aliases:
                        self._cache[alias.lower()] = config
                    # Also index by name
                    self._cache[config.name.lower()] = config

                self._cache_timestamp = time.time()

            except ConvexQueryError as e:
                # Log error but don't crash - use stale cache if available
                print(f"Failed to refresh chain cache: {e}")
                if not self._all_chains:
                    raise

    async def get_by_chain_id(self, chain_id: int) -> Optional[ChainConfig]:
        """Get chain config by chain ID."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return self._cache_by_id.get(chain_id)

    async def resolve_alias(self, alias: str) -> Optional[ChainConfig]:
        """Resolve a chain alias to its config."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return self._cache.get(alias.lower())

    async def get_alchemy_slug(self, chain: str | int) -> Optional[str]:
        """
        Get the Alchemy API slug for a chain.

        Args:
            chain: Chain ID (int) or alias (str)

        Returns:
            Alchemy slug like "eth-mainnet" or None if not supported
        """
        if isinstance(chain, int):
            config = await self.get_by_chain_id(chain)
        else:
            config = await self.resolve_alias(chain)

        if not config:
            return None

        # Only return slug if verified or if we want to try unverified
        return config.alchemy_slug

    async def get_alchemy_url(self, chain: str | int, api_key: str) -> Optional[str]:
        """
        Get the full Alchemy API URL for a chain.

        Args:
            chain: Chain ID (int) or alias (str)
            api_key: Alchemy API key

        Returns:
            Full Alchemy URL like "https://eth-mainnet.g.alchemy.com/v2/{key}"
        """
        slug = await self.get_alchemy_slug(chain)
        if not slug:
            return None
        return f"https://{slug}.g.alchemy.com/v2/{api_key}"

    async def is_alchemy_supported(self, chain: str | int) -> bool:
        """Check if Alchemy supports this chain (verified)."""
        if isinstance(chain, int):
            config = await self.get_by_chain_id(chain)
        else:
            config = await self.resolve_alias(chain)

        if not config:
            return False

        return bool(config.alchemy_slug and config.alchemy_verified)

    async def list_enabled(self) -> List[ChainConfig]:
        """List all enabled chains."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return self._all_chains.copy()

    async def list_alchemy_supported(self) -> List[ChainConfig]:
        """List chains with verified Alchemy support."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return [c for c in self._all_chains if c.alchemy_slug and c.alchemy_verified]

    async def get_native_symbol(self, chain: str | int) -> str:
        """Get the native token symbol for a chain."""
        if isinstance(chain, int):
            config = await self.get_by_chain_id(chain)
        else:
            config = await self.resolve_alias(chain)

        return config.native_symbol if config else "ETH"

    async def get_explorer_url(self, chain: str | int) -> Optional[str]:
        """Get the block explorer URL for a chain."""
        if isinstance(chain, int):
            config = await self.get_by_chain_id(chain)
        else:
            config = await self.resolve_alias(chain)

        return config.explorer_url if config else None


# Singleton instance
_chain_service: Optional[ChainService] = None


def get_chain_service() -> ChainService:
    """Get the singleton ChainService instance."""
    global _chain_service
    if _chain_service is None:
        _chain_service = ChainService()
    return _chain_service
