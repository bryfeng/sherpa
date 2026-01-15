"""Dynamic chain registry that fetches supported chains from Relay API."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...providers.relay import RelayProvider

# Type alias for chain IDs (int for EVM, str for non-EVM like Solana)
ChainId = Union[int, str]


class ChainRegistry:
    """Dynamic chain metadata from Relay API with caching and alias generation.

    Instead of hardcoding chain metadata, this registry fetches the list of
    supported chains from Relay's /chains endpoint and builds lookup maps
    dynamically. This ensures new chains (like Ink) are automatically supported.

    Usage:
        registry = ChainRegistry()
        await registry.ensure_loaded()

        chain_id = registry.get_chain_id("ink")  # Returns 57073
        name = registry.get_chain_name(57073)    # Returns "Ink"
    """

    # Cache TTL in seconds (1 hour default - chains don't change often)
    DEFAULT_CACHE_TTL = 3600

    # Common alias patterns to generate from chain names
    ALIAS_EXPANSIONS = {
        "ethereum": ["eth", "mainnet", "main net", "layer1", "layer 1", "l1"],
        "arbitrum": ["arb"],
        "optimism": ["op"],
        "polygon": ["matic"],
        "solana": ["sol"],
    }

    def __init__(
        self,
        *,
        relay_provider: Optional[RelayProvider] = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._relay = relay_provider or RelayProvider()
        self._cache_ttl = cache_ttl_seconds
        self._logger = logger or logging.getLogger(__name__)

        # Chain data storage
        self._chains: Dict[ChainId, Dict[str, Any]] = {}
        self._alias_to_id: Dict[str, ChainId] = {}
        self._last_refresh: Optional[datetime] = None
        self._loading = False

    async def ensure_loaded(self) -> bool:
        """Ensure chain data is loaded and fresh. Returns True if data is available."""
        if self._needs_refresh():
            try:
                await self._refresh()
                return True
            except Exception as exc:
                self._logger.warning("Failed to refresh chain registry: %s", exc)
                # Return True if we have stale data, False if no data at all
                return bool(self._chains)
        return bool(self._chains)

    def _needs_refresh(self) -> bool:
        """Check if cache is stale or empty."""
        if not self._chains:
            return True
        if self._last_refresh is None:
            return True
        elapsed = (datetime.now() - self._last_refresh).total_seconds()
        return elapsed > self._cache_ttl

    async def _refresh(self) -> None:
        """Fetch fresh chain data from Relay API."""
        if self._loading:
            return
        self._loading = True

        try:
            chains_data = await self._relay.get_chains()
            self._process_chains(chains_data)
            self._last_refresh = datetime.now()
            self._logger.info(
                "Chain registry refreshed: %d chains, %d aliases",
                len(self._chains),
                len(self._alias_to_id),
            )
        finally:
            self._loading = False

    def _process_chains(self, chains_data: List[Dict[str, Any]]) -> None:
        """Process raw chain data into lookup structures."""
        new_chains: Dict[ChainId, Dict[str, Any]] = {}
        new_aliases: Dict[str, ChainId] = {}

        for chain in chains_data:
            chain_id = chain.get("id")
            if chain_id is None:
                continue

            # Skip disabled chains
            if chain.get("disabled", False):
                continue

            new_chains[chain_id] = chain

            # Generate aliases for this chain
            aliases = self._generate_aliases(chain)
            for alias in aliases:
                # Don't overwrite existing aliases (first wins)
                if alias not in new_aliases:
                    new_aliases[alias] = chain_id

        self._chains = new_chains
        self._alias_to_id = new_aliases

    def _generate_aliases(self, chain: Dict[str, Any]) -> Set[str]:
        """Generate all possible aliases for a chain."""
        aliases: Set[str] = set()
        chain_id = chain.get("id")

        # Primary names
        name = (chain.get("name") or "").lower().strip()
        display_name = (chain.get("displayName") or "").lower().strip()

        if name:
            aliases.add(name)
            # Handle multi-word names: "arbitrum one" -> "arbitrum", "arbitrumone"
            words = name.split()
            if len(words) > 1:
                aliases.add(words[0])  # First word
                aliases.add("".join(words))  # Concatenated

        if display_name and display_name != name:
            aliases.add(display_name)
            words = display_name.split()
            if len(words) > 1:
                aliases.add(words[0])
                aliases.add("".join(words))

        # Add chain ID as string alias (useful for "chain 57073")
        if chain_id is not None:
            aliases.add(str(chain_id))

        # Expand common aliases
        for base_name, expansions in self.ALIAS_EXPANSIONS.items():
            if base_name in name or base_name in display_name:
                aliases.update(expansions)

        # Handle "mainnet" suffix: "base mainnet" -> "base"
        for alias in list(aliases):
            if alias.endswith(" mainnet"):
                aliases.add(alias.replace(" mainnet", ""))
            elif alias.endswith("mainnet"):
                aliases.add(alias.replace("mainnet", ""))

        # Remove empty strings
        aliases.discard("")

        return aliases

    # ─────────────────────────────────────────────────────────────────────────
    # Public lookup methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_chain_id(self, alias: str) -> Optional[ChainId]:
        """Look up chain ID by alias. Returns None if not found."""
        normalized = alias.lower().strip()
        return self._alias_to_id.get(normalized)

    def get_chain(self, chain_id: ChainId) -> Optional[Dict[str, Any]]:
        """Get full chain metadata by ID."""
        return self._chains.get(chain_id)

    def get_chain_name(self, chain_id: ChainId) -> str:
        """Get human-readable chain name."""
        chain = self._chains.get(chain_id)
        if chain:
            return chain.get("displayName") or chain.get("name") or f"Chain {chain_id}"
        return f"Chain {chain_id}"

    def get_native_symbol(self, chain_id: ChainId) -> str:
        """Get native token symbol (e.g., 'ETH', 'MATIC')."""
        chain = self._chains.get(chain_id)
        if chain:
            currency = chain.get("currency") or {}
            return currency.get("symbol", "ETH")
        return "ETH"

    def get_native_decimals(self, chain_id: ChainId) -> int:
        """Get native token decimals."""
        chain = self._chains.get(chain_id)
        if chain:
            currency = chain.get("currency") or {}
            try:
                return int(currency.get("decimals", 18))
            except (TypeError, ValueError):
                return 18
        return 18

    def get_native_token_address(self, chain_id: ChainId) -> str:
        """Get native token address (usually zero address for ETH)."""
        chain = self._chains.get(chain_id)
        if chain:
            currency = chain.get("currency") or {}
            return currency.get("address", "0x0000000000000000000000000000000000000000")
        return "0x0000000000000000000000000000000000000000"

    def is_chain_supported(self, chain_id: ChainId) -> bool:
        """Check if a chain ID is supported."""
        return chain_id in self._chains

    def get_all_aliases(self) -> Dict[str, ChainId]:
        """Get all alias -> chain_id mappings."""
        return dict(self._alias_to_id)

    def get_supported_chain_names(self, limit: int = 20) -> List[str]:
        """Get list of supported chain names for error messages."""
        names = []
        for chain in self._chains.values():
            name = chain.get("displayName") or chain.get("name")
            if name and name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return sorted(names)

    def detect_chain_in_text(self, text: str) -> Optional[ChainId]:
        """Detect a chain reference anywhere in text."""
        text_lower = text.lower()
        # Check longer aliases first to avoid partial matches
        sorted_aliases = sorted(self._alias_to_id.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            # Use word boundary matching
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, text_lower):
                return self._alias_to_id[alias]
        return None

    def detect_chain_with_preposition(
        self, text: str, prepositions: List[str]
    ) -> Optional[ChainId]:
        """Detect chain following specific prepositions (e.g., 'to base', 'from ethereum')."""
        text_lower = text.lower()
        for alias, chain_id in self._alias_to_id.items():
            escaped = re.escape(alias)
            for prep in prepositions:
                pattern = rf"\b{prep}\s+{escaped}\b"
                if re.search(pattern, text_lower):
                    return chain_id
        return None

    @property
    def chain_count(self) -> int:
        """Number of supported chains."""
        return len(self._chains)

    @property
    def is_loaded(self) -> bool:
        """Whether chain data has been loaded."""
        return bool(self._chains)


# Module-level singleton for convenience
_default_registry: Optional[ChainRegistry] = None


async def get_chain_registry() -> ChainRegistry:
    """Get or create the default chain registry singleton (async).

    This ensures the registry is loaded before returning.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ChainRegistry()
    await _default_registry.ensure_loaded()
    return _default_registry


def get_registry_sync() -> ChainRegistry:
    """Get the chain registry synchronously.

    Returns the registry if it has been initialized, otherwise returns
    an empty registry that will be populated on first async access.

    For modules that need sync access, this allows reading cached data.
    The registry should be initialized at app startup via init_chain_registry().
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ChainRegistry()
    return _default_registry


async def init_chain_registry() -> ChainRegistry:
    """Initialize the chain registry at application startup.

    Call this from FastAPI's startup event to ensure chain data is
    available before handling requests.

    Example:
        @app.on_event("startup")
        async def startup():
            await init_chain_registry()
    """
    return await get_chain_registry()


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible exports for legacy code
# These provide dict-like access matching the old CHAIN_METADATA interface
# ─────────────────────────────────────────────────────────────────────────────

def get_chain_alias_to_id() -> Dict[str, ChainId]:
    """Get alias -> chain_id mapping (backwards compatible).

    Equivalent to the old CHAIN_ALIAS_TO_ID constant.
    """
    return get_registry_sync().get_all_aliases()


def get_chain_metadata() -> Dict[ChainId, Dict[str, Any]]:
    """Get chain_id -> metadata mapping (backwards compatible).

    Equivalent to the old CHAIN_METADATA constant.
    Returns raw chain data from the registry.
    """
    registry = get_registry_sync()
    return {chain_id: registry.get_chain(chain_id) or {} for chain_id in registry._chains}


def resolve_chain_id_from_alias(alias: str) -> Optional[ChainId]:
    """Resolve a chain alias to chain ID (backwards compatible).

    Equivalent to checking CHAIN_ALIAS_TO_ID.get(alias).
    """
    return get_registry_sync().get_chain_id(alias)
