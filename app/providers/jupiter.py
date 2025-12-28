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


# =============================================================================
# Swap Quote and Transaction Building
# =============================================================================

QUOTE_API_URL = "https://quote-api.jup.ag/v6"

# Well-known token mints
NATIVE_SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


@dataclass
class RoutePlanStep:
    """A single step in the swap route."""
    swap_info: Dict[str, Any]
    percent: int  # Percentage of input going through this route


@dataclass
class JupiterQuote:
    """Quote response from Jupiter."""
    input_mint: str
    output_mint: str
    in_amount: int                              # In smallest units (lamports)
    out_amount: int                             # In smallest units
    other_amount_threshold: int                 # Minimum output (with slippage)
    swap_mode: str                              # "ExactIn" or "ExactOut"
    slippage_bps: int
    price_impact_pct: float
    route_plan: List[RoutePlanStep]

    # Token metadata
    input_token: Optional[JupiterToken] = None
    output_token: Optional[JupiterToken] = None

    # For transaction building
    quote_response: Optional[Dict[str, Any]] = None

    # Timing
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """Check if quote is still valid (within 30 seconds)."""
        return (time.time() - self.fetched_at) < 30


@dataclass
class JupiterSwapResult:
    """Result of building a swap transaction."""
    swap_transaction: str                       # Base64 encoded transaction
    last_valid_block_height: int
    priority_fee_lamports: int
    compute_unit_limit: int


class JupiterQuoteError(Exception):
    """Failed to get a quote from Jupiter."""
    pass


class JupiterSwapError(Exception):
    """Failed to build swap transaction."""
    pass


class JupiterSwapProvider:
    """
    Jupiter swap provider for Solana token swaps.

    Extends the base JupiterProvider with quote and swap transaction building.

    Usage:
        provider = JupiterSwapProvider()

        # Get a quote
        quote = await provider.get_swap_quote(
            input_mint=NATIVE_SOL_MINT,
            output_mint=USDC_MINT,
            amount=1_000_000_000,  # 1 SOL in lamports
            slippage_bps=50,
        )

        # Build swap transaction
        swap = await provider.build_swap_transaction(
            quote=quote,
            user_public_key="...",
        )

        # Sign and send the transaction via wallet
    """

    def __init__(
        self,
        token_provider: Optional[JupiterProvider] = None,
        timeout_s: float = 30.0,
    ):
        self._token_provider = token_provider or JupiterProvider()
        self._timeout_s = timeout_s

    async def get_swap_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        max_accounts: Optional[int] = None,
    ) -> JupiterQuote:
        """
        Get a swap quote from Jupiter.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest units (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
            swap_mode: "ExactIn" or "ExactOut"
            only_direct_routes: Only use direct routes (no multi-hop)
            as_legacy_transaction: Use legacy transaction format
            max_accounts: Maximum accounts in transaction

        Returns:
            JupiterQuote with route and amounts
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": slippage_bps,
            "swapMode": swap_mode,
            "onlyDirectRoutes": str(only_direct_routes).lower(),
            "asLegacyTransaction": str(as_legacy_transaction).lower(),
        }

        if max_accounts:
            params["maxAccounts"] = max_accounts

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.get(
                    f"{QUOTE_API_URL}/quote",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

            if "error" in data:
                raise JupiterQuoteError(f"Jupiter quote error: {data['error']}")

            # Parse route plan
            route_plan = []
            for step in data.get("routePlan", []):
                route_plan.append(RoutePlanStep(
                    swap_info=step.get("swapInfo", {}),
                    percent=step.get("percent", 100),
                ))

            # Get token metadata
            input_token = await self._token_provider.get_token_by_mint(input_mint)
            output_token = await self._token_provider.get_token_by_mint(output_mint)

            quote = JupiterQuote(
                input_mint=data["inputMint"],
                output_mint=data["outputMint"],
                in_amount=int(data["inAmount"]),
                out_amount=int(data["outAmount"]),
                other_amount_threshold=int(data.get("otherAmountThreshold", data["outAmount"])),
                swap_mode=data.get("swapMode", swap_mode),
                slippage_bps=slippage_bps,
                price_impact_pct=float(data.get("priceImpactPct", 0)),
                route_plan=route_plan,
                input_token=input_token,
                output_token=output_token,
                quote_response=data,
            )

            return quote

        except httpx.HTTPStatusError as e:
            raise JupiterQuoteError(f"HTTP error: {e.response.status_code}")
        except JupiterQuoteError:
            raise
        except Exception as e:
            raise JupiterQuoteError(str(e))

    async def build_swap_transaction(
        self,
        quote: JupiterQuote,
        user_public_key: str,
        wrap_and_unwrap_sol: bool = True,
        use_shared_accounts: bool = True,
        priority_level: str = "medium",  # "low", "medium", "high", "veryHigh"
        as_legacy_transaction: bool = False,
    ) -> JupiterSwapResult:
        """
        Build a swap transaction from a quote.

        Args:
            quote: The quote to build a transaction for
            user_public_key: User's Solana wallet public key
            wrap_and_unwrap_sol: Automatically wrap/unwrap SOL
            use_shared_accounts: Use shared accounts for efficiency
            priority_level: Priority level for automatic fee estimation
            as_legacy_transaction: Use legacy transaction format

        Returns:
            JupiterSwapResult with base64 encoded transaction
        """
        if not quote.quote_response:
            raise JupiterSwapError("Quote response required for swap transaction")

        if not quote.is_valid:
            raise JupiterSwapError("Quote has expired, please get a new quote")

        payload = {
            "quoteResponse": quote.quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": wrap_and_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "asLegacyTransaction": as_legacy_transaction,
            "prioritizationFeeLamports": {
                "priorityLevelWithMaxLamports": {
                    "maxLamports": 10_000_000,  # 0.01 SOL max
                    "priorityLevel": priority_level,
                }
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{QUOTE_API_URL}/swap",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            if "error" in data:
                raise JupiterSwapError(f"Jupiter swap error: {data['error']}")

            return JupiterSwapResult(
                swap_transaction=data["swapTransaction"],
                last_valid_block_height=data.get("lastValidBlockHeight", 0),
                priority_fee_lamports=data.get("prioritizationFeeLamports", 0),
                compute_unit_limit=data.get("computeUnitLimit", 200_000),
            )

        except httpx.HTTPStatusError as e:
            raise JupiterSwapError(f"HTTP error: {e.response.status_code}")
        except JupiterSwapError:
            raise
        except Exception as e:
            raise JupiterSwapError(str(e))


# Singleton instances
_jupiter_provider: Optional[JupiterProvider] = None
_jupiter_swap_provider: Optional[JupiterSwapProvider] = None


def get_jupiter_provider() -> JupiterProvider:
    """Get the singleton Jupiter token provider."""
    global _jupiter_provider
    if _jupiter_provider is None:
        _jupiter_provider = JupiterProvider()
    return _jupiter_provider


def get_jupiter_swap_provider() -> JupiterSwapProvider:
    """Get the singleton Jupiter swap provider."""
    global _jupiter_swap_provider
    if _jupiter_swap_provider is None:
        _jupiter_swap_provider = JupiterSwapProvider(
            token_provider=get_jupiter_provider()
        )
    return _jupiter_swap_provider


__all__ = [
    "JupiterProvider",
    "JupiterToken",
    "JupiterSwapProvider",
    "JupiterQuote",
    "JupiterSwapResult",
    "JupiterQuoteError",
    "JupiterSwapError",
    "RoutePlanStep",
    "get_jupiter_provider",
    "get_jupiter_swap_provider",
    "NATIVE_SOL_MINT",
    "USDC_MINT",
    "USDT_MINT",
]
