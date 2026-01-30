"""
Token configuration service - queries the Convex tokenCatalog table.

This service provides token metadata for swap operations including symbol resolution,
alias matching, and multi-chain support (EVM + Solana).

Follows the ChainService pattern with:
- 5-minute TTL cache
- Async lock for thread-safety
- Graceful degradation on Convex errors
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Union

from app.db.convex_client import get_convex_client, ConvexQueryError


# Type alias for chain IDs: int for EVM, "solana" for Solana
ChainId = Union[int, Literal["solana"]]

# Solana constants
SOLANA_CHAIN_ID: Literal["solana"] = "solana"
NATIVE_SOL_MINT = "So11111111111111111111111111111111111111112"
NATIVE_PLACEHOLDER = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"


def is_solana_chain(chain_id: ChainId) -> bool:
    """Check if a chain ID represents Solana."""
    if isinstance(chain_id, str):
        return chain_id.lower() == "solana"
    return False


def is_evm_chain(chain_id: ChainId) -> bool:
    """Check if a chain ID represents an EVM chain."""
    return isinstance(chain_id, int)


@dataclass
class TokenConfig:
    """Token configuration from Convex tokenCatalog."""

    chain_id: ChainId
    address: str
    symbol: str
    name: str
    decimals: int
    aliases: List[str] = field(default_factory=list)
    is_native: bool = False
    is_enabled: bool = True
    coingecko_id: Optional[str] = None
    logo_url: Optional[str] = None

    @classmethod
    def from_convex(cls, data: Dict[str, Any]) -> "TokenConfig":
        """Create TokenConfig from Convex response."""
        return cls(
            chain_id=data["chainId"],
            address=data["address"],
            symbol=data["symbol"],
            name=data["name"],
            decimals=data["decimals"],
            aliases=data.get("aliases") or [],
            is_native=data.get("isNative", False),
            is_enabled=data.get("isEnabled", True) if data.get("isEnabled") is not None else True,
            coingecko_id=data.get("coingeckoId"),
            logo_url=data.get("logoUrl"),
        )

    @property
    def canonical_id(self) -> str:
        """Unique identifier: chain_id:address."""
        addr = self.address.lower() if is_evm_chain(self.chain_id) else self.address
        return f"{self.chain_id}:{addr}"

    def matches_query(self, query: str) -> bool:
        """Check if this token matches a search query."""
        query_lower = query.lower()

        # Match by symbol
        if self.symbol.lower() == query_lower:
            return True

        # Match by alias
        if query_lower in [a.lower() for a in self.aliases]:
            return True

        # Match by address
        if is_evm_chain(self.chain_id):
            if self.address.lower() == query_lower:
                return True
        else:
            if self.address == query:
                return True

        return False


class TokenService:
    """
    Service for fetching token configurations from Convex.

    Implements caching to avoid repeated Convex queries.
    On-demand caching: tokens resolved from fallback sources are persisted to Convex.
    """

    # Cache TTL in seconds (5 minutes - matches ChainService)
    CACHE_TTL = 300

    def __init__(self):
        # All tokens by chain
        self._tokens_by_chain: Dict[ChainId, List[TokenConfig]] = {}

        # Indexes for fast lookup
        self._by_chain_symbol: Dict[ChainId, Dict[str, TokenConfig]] = {}
        self._by_chain_address: Dict[ChainId, Dict[str, TokenConfig]] = {}
        self._alias_index: Dict[ChainId, Dict[str, str]] = {}  # alias -> symbol

        self._cache_timestamp: float = 0
        self._lock = asyncio.Lock()

        # Track tokens being persisted to avoid duplicate writes
        self._pending_persists: Set[str] = set()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (
            self._cache_timestamp > 0
            and (time.time() - self._cache_timestamp) < self.CACHE_TTL
        )

    async def _refresh_cache(self) -> None:
        """Refresh the token cache from Convex."""
        async with self._lock:
            # Double-check after acquiring lock
            if self._is_cache_valid():
                return

            try:
                client = get_convex_client()
                tokens_data = await client.query("tokenCatalog:listEnabledForSwaps", {})

                # Clear existing cache
                self._tokens_by_chain.clear()
                self._by_chain_symbol.clear()
                self._by_chain_address.clear()
                self._alias_index.clear()

                for token_data in tokens_data or []:
                    config = TokenConfig.from_convex(token_data)
                    chain_id = config.chain_id

                    # Initialize chain dicts if needed
                    if chain_id not in self._tokens_by_chain:
                        self._tokens_by_chain[chain_id] = []
                        self._by_chain_symbol[chain_id] = {}
                        self._by_chain_address[chain_id] = {}
                        self._alias_index[chain_id] = {}

                    self._tokens_by_chain[chain_id].append(config)

                    # Index by symbol (case-insensitive)
                    self._by_chain_symbol[chain_id][config.symbol.lower()] = config

                    # Index by address
                    addr_key = config.address.lower() if is_evm_chain(chain_id) else config.address
                    self._by_chain_address[chain_id][addr_key] = config

                    # Index by aliases
                    for alias in config.aliases:
                        self._alias_index[chain_id][alias.lower()] = config.symbol

                self._cache_timestamp = time.time()

            except ConvexQueryError as e:
                # Log error but don't crash - use stale cache if available
                print(f"Failed to refresh token cache: {e}")
                if not self._tokens_by_chain:
                    raise

    async def _persist_token(self, config: TokenConfig) -> None:
        """
        Persist a token to Convex (fire-and-forget).

        Called when a token is resolved from a fallback source (portfolio, API).
        Adds it to the Convex tokenCatalog so future lookups are faster.
        """
        canonical_id = config.canonical_id

        # Skip if already pending or already in cache
        if canonical_id in self._pending_persists:
            return

        # Check if already in cache
        addr_key = config.address.lower() if is_evm_chain(config.chain_id) else config.address
        if config.chain_id in self._by_chain_address:
            if addr_key in self._by_chain_address[config.chain_id]:
                return

        self._pending_persists.add(canonical_id)
        try:
            client = get_convex_client()
            # Build token data, omitting None values for optional fields
            token_data: Dict[str, Any] = {
                "address": config.address,
                "chainId": config.chain_id,
                "symbol": config.symbol,
                "name": config.name,
                "decimals": config.decimals,
                "aliases": config.aliases,
                "isNative": config.is_native,
                "isEnabled": config.is_enabled,
            }
            # Only include optional fields if they have values
            if config.coingecko_id:
                token_data["coingeckoId"] = config.coingecko_id
            if config.logo_url:
                token_data["logoUrl"] = config.logo_url

            await client.mutation(
                "tokenCatalog:upsertBatchForSwaps",
                {"tokens": [token_data]},
            )
            # Add to local cache immediately
            self._add_to_cache(config)
        except Exception as e:
            # Non-critical - just log and continue
            print(f"Failed to persist token {config.symbol} on chain {config.chain_id}: {e}")
        finally:
            self._pending_persists.discard(canonical_id)

    def _add_to_cache(self, config: TokenConfig) -> None:
        """Add a token to the local cache without refreshing from Convex."""
        chain_id = config.chain_id

        # Initialize chain dicts if needed
        if chain_id not in self._tokens_by_chain:
            self._tokens_by_chain[chain_id] = []
            self._by_chain_symbol[chain_id] = {}
            self._by_chain_address[chain_id] = {}
            self._alias_index[chain_id] = {}

        # Check if already exists (avoid duplicates)
        addr_key = config.address.lower() if is_evm_chain(chain_id) else config.address
        if addr_key in self._by_chain_address[chain_id]:
            return

        self._tokens_by_chain[chain_id].append(config)
        self._by_chain_symbol[chain_id][config.symbol.lower()] = config
        self._by_chain_address[chain_id][addr_key] = config

        for alias in config.aliases:
            self._alias_index[chain_id][alias.lower()] = config.symbol

    def _schedule_persist(self, config: TokenConfig) -> None:
        """Schedule a fire-and-forget persist task."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_token(config))
        except RuntimeError:
            # No running loop - skip persist
            pass

    async def resolve_token(
        self,
        chain_id: ChainId,
        query: str,
        portfolio_tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[TokenConfig]:
        """
        Resolve a token query to a TokenConfig.

        Resolution priority:
        1. Portfolio tokens (user's holdings)
        2. Alias match
        3. Symbol match
        4. Address match

        Args:
            chain_id: The chain to search on
            query: Token symbol, alias, or address
            portfolio_tokens: Optional list of user's portfolio tokens

        Returns:
            TokenConfig if found, None otherwise
        """
        if not self._is_cache_valid():
            await self._refresh_cache()

        query_normalized = query.strip()
        query_lower = query_normalized.lower()

        # 1. Check portfolio tokens first (highest priority)
        if portfolio_tokens:
            for pt in portfolio_tokens:
                pt_chain = pt.get("chain_id")
                if pt_chain != chain_id:
                    continue

                pt_symbol = str(pt.get("symbol", "")).lower()
                pt_address = str(pt.get("address", ""))

                if query_lower == pt_symbol:
                    # Found in portfolio, try to get full config from cache
                    config = self._by_chain_symbol.get(chain_id, {}).get(pt_symbol)
                    if config:
                        return config

                    # Create config from portfolio and persist for future lookups
                    fallback_config = TokenConfig(
                        chain_id=chain_id,
                        address=pt_address,
                        symbol=pt.get("symbol", "").upper(),
                        name=pt.get("name", pt.get("symbol", "")),
                        decimals=pt.get("decimals", 18),
                        is_native=pt.get("is_native", False),
                        logo_url=pt.get("logo_url") or pt.get("logoUrl"),
                    )
                    # Fire-and-forget persist to Convex
                    self._schedule_persist(fallback_config)
                    return fallback_config

        # 2. Check alias index
        chain_aliases = self._alias_index.get(chain_id, {})
        if query_lower in chain_aliases:
            symbol = chain_aliases[query_lower]
            return self._by_chain_symbol.get(chain_id, {}).get(symbol.lower())

        # 3. Check symbol index
        chain_symbols = self._by_chain_symbol.get(chain_id, {})
        if query_lower in chain_symbols:
            return chain_symbols[query_lower]

        # 4. Check address index
        chain_addresses = self._by_chain_address.get(chain_id, {})
        addr_key = query_normalized.lower() if is_evm_chain(chain_id) else query_normalized
        if addr_key in chain_addresses:
            return chain_addresses[addr_key]

        return None

    async def list_by_chain(self, chain_id: ChainId) -> List[TokenConfig]:
        """List all enabled tokens for a chain."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return self._tokens_by_chain.get(chain_id, []).copy()

    async def get_supported_symbols(self, chain_id: ChainId) -> List[str]:
        """Get list of supported token symbols for a chain."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return list(self._by_chain_symbol.get(chain_id, {}).keys())

    async def get_supported_chains(self) -> List[ChainId]:
        """Get list of chains with registered tokens."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return list(self._tokens_by_chain.keys())

    def get_supported_symbols_sync(self, chain_id: ChainId) -> List[str]:
        """Synchronous version - uses cached data only."""
        return list(self._by_chain_symbol.get(chain_id, {}).keys())

    def get_supported_chains_sync(self) -> List[ChainId]:
        """Synchronous version - uses cached data only."""
        return list(self._tokens_by_chain.keys())

    def resolve_token_sync(
        self,
        chain_id: ChainId,
        query: str,
        portfolio_tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[TokenConfig]:
        """
        Synchronous token resolution using cached data only.

        Use this for fast resolution when async is not available.
        """
        query_normalized = query.strip()
        query_lower = query_normalized.lower()

        # 1. Check portfolio tokens first
        if portfolio_tokens:
            for pt in portfolio_tokens:
                pt_chain = pt.get("chain_id")
                if pt_chain != chain_id:
                    continue

                pt_symbol = str(pt.get("symbol", "")).lower()
                if query_lower == pt_symbol:
                    config = self._by_chain_symbol.get(chain_id, {}).get(pt_symbol)
                    if config:
                        return config
                    # Create config from portfolio and persist for future lookups
                    fallback_config = TokenConfig(
                        chain_id=chain_id,
                        address=pt.get("address", ""),
                        symbol=pt.get("symbol", "").upper(),
                        name=pt.get("name", pt.get("symbol", "")),
                        decimals=pt.get("decimals", 18),
                        is_native=pt.get("is_native", False),
                        logo_url=pt.get("logo_url") or pt.get("logoUrl"),
                    )
                    # Fire-and-forget persist to Convex
                    self._schedule_persist(fallback_config)
                    return fallback_config

        # 2. Check alias index
        chain_aliases = self._alias_index.get(chain_id, {})
        if query_lower in chain_aliases:
            symbol = chain_aliases[query_lower]
            return self._by_chain_symbol.get(chain_id, {}).get(symbol.lower())

        # 3. Check symbol index
        chain_symbols = self._by_chain_symbol.get(chain_id, {})
        if query_lower in chain_symbols:
            return chain_symbols[query_lower]

        # 4. Check address index
        chain_addresses = self._by_chain_address.get(chain_id, {})
        addr_key = query_normalized.lower() if is_evm_chain(chain_id) else query_normalized
        if addr_key in chain_addresses:
            return chain_addresses[addr_key]

        return None

    async def is_supported(self, chain_id: ChainId) -> bool:
        """Check if a chain has registered tokens."""
        if not self._is_cache_valid():
            await self._refresh_cache()
        return chain_id in self._tokens_by_chain


# Singleton instance
_token_service: Optional[TokenService] = None


def get_token_service() -> TokenService:
    """Get the singleton TokenService instance."""
    global _token_service
    if _token_service is None:
        _token_service = TokenService()
    return _token_service


__all__ = [
    "TokenConfig",
    "TokenService",
    "get_token_service",
    "ChainId",
    "SOLANA_CHAIN_ID",
    "NATIVE_SOL_MINT",
    "NATIVE_PLACEHOLDER",
    "is_solana_chain",
    "is_evm_chain",
]
