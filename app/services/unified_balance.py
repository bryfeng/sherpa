"""
Unified Balance Service.

Provides multi-chain token balance aggregation via Rhinestone Omni Account.
Users see their total holdings across all supported chains with a single query.

Features:
- Aggregate balances across EVM chains
- USD value calculation
- Caching for performance
- Chain-specific breakdown
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config import settings
from ..providers.rhinestone import (
    get_rhinestone_provider,
    RhinestoneError,
    UnifiedBalance,
    ChainBalance,
)

logger = logging.getLogger(__name__)


# Supported EVM chains for unified balances
SUPPORTED_CHAINS: Dict[int, str] = {
    1: "Ethereum",
    10: "Optimism",
    137: "Polygon",
    8453: "Base",
    42161: "Arbitrum",
    43114: "Avalanche",
    56: "BSC",
}


@dataclass
class TokenBalance:
    """Balance of a single token across chains."""
    symbol: str
    name: Optional[str] = None
    total_balance: Decimal = Decimal("0")
    total_usd_value: Decimal = Decimal("0")
    chains: List["ChainTokenBalance"] = field(default_factory=list)
    logo_uri: Optional[str] = None


@dataclass
class ChainTokenBalance:
    """Token balance on a specific chain."""
    chain_id: int
    chain_name: str
    balance: Decimal
    balance_formatted: str
    usd_value: Decimal
    token_address: str


@dataclass
class UnifiedBalanceResult:
    """Result of unified balance query."""
    success: bool
    account_address: str
    total_usd_value: Decimal = Decimal("0")
    tokens: List[TokenBalance] = field(default_factory=list)
    chains_queried: List[int] = field(default_factory=list)
    cached: bool = False
    timestamp: int = 0
    error: Optional[str] = None


class UnifiedBalanceService:
    """
    Service for fetching and aggregating multi-chain balances.

    Uses Rhinestone's unified balance API when available, falls back to
    individual chain queries if needed.

    Usage:
        service = get_unified_balance_service()

        # Get all balances
        result = await service.get_unified_balances("0x...")

        # Get balances for specific chains
        result = await service.get_unified_balances("0x...", chains=[8453, 42161])

        # Get balance for specific token
        token_balance = await service.get_token_balance("0x...", "USDC")
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 30,
    ) -> None:
        self._rhinestone = get_rhinestone_provider()
        self._cache: Dict[str, UnifiedBalanceResult] = {}
        self._cache_ttl = cache_ttl_seconds

    async def get_unified_balances(
        self,
        account_address: str,
        chains: Optional[List[int]] = None,
        force_refresh: bool = False,
    ) -> UnifiedBalanceResult:
        """
        Get unified token balances across all chains.

        Args:
            account_address: Smart Account address
            chains: Optional list of chain IDs (defaults to all supported)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            UnifiedBalanceResult with aggregated balances
        """
        # Normalize address
        account_address = account_address.lower()
        chains = chains or list(SUPPORTED_CHAINS.keys())

        # Check cache
        cache_key = f"{account_address}:{','.join(str(c) for c in sorted(chains))}"
        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        try:
            # Check if Rhinestone is enabled
            if not await self._rhinestone.ready():
                return UnifiedBalanceResult(
                    success=False,
                    account_address=account_address,
                    error="Rhinestone not enabled",
                )

            # Fetch unified balances from Rhinestone
            rhinestone_balances = await self._rhinestone.get_unified_balances(
                account_address=account_address,
                chains=chains,
            )

            # Transform to our format
            tokens: List[TokenBalance] = []
            total_usd = Decimal("0")

            for rb in rhinestone_balances:
                chain_balances = [
                    ChainTokenBalance(
                        chain_id=cb.chain_id,
                        chain_name=cb.chain_name or SUPPORTED_CHAINS.get(cb.chain_id, f"Chain {cb.chain_id}"),
                        balance=Decimal(cb.balance) if cb.balance else Decimal("0"),
                        balance_formatted=cb.balance_formatted or "0",
                        usd_value=Decimal(str(cb.usd_value)) if cb.usd_value else Decimal("0"),
                        token_address=cb.token_address,
                    )
                    for cb in rb.chain_balances
                ]

                token_total_usd = Decimal(str(rb.total_usd_value)) if rb.total_usd_value else Decimal("0")
                total_usd += token_total_usd

                tokens.append(TokenBalance(
                    symbol=rb.token_symbol,
                    total_balance=Decimal(rb.total_balance) if rb.total_balance else Decimal("0"),
                    total_usd_value=token_total_usd,
                    chains=chain_balances,
                ))

            # Sort by USD value descending
            tokens.sort(key=lambda t: t.total_usd_value, reverse=True)

            result = UnifiedBalanceResult(
                success=True,
                account_address=account_address,
                total_usd_value=total_usd,
                tokens=tokens,
                chains_queried=chains,
                timestamp=int(time.time()),
            )

            # Cache result
            self._cache[cache_key] = result

            return result

        except RhinestoneError as e:
            logger.error(f"Failed to fetch unified balances: {e}")
            return UnifiedBalanceResult(
                success=False,
                account_address=account_address,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching unified balances: {e}")
            return UnifiedBalanceResult(
                success=False,
                account_address=account_address,
                error=str(e),
            )

    async def get_token_balance(
        self,
        account_address: str,
        token_symbol: str,
        chains: Optional[List[int]] = None,
    ) -> Optional[TokenBalance]:
        """
        Get balance for a specific token across chains.

        Args:
            account_address: Smart Account address
            token_symbol: Token symbol (e.g., "USDC", "ETH")
            chains: Optional list of chain IDs

        Returns:
            TokenBalance if found, None otherwise
        """
        result = await self.get_unified_balances(account_address, chains)
        if not result.success:
            return None

        token_symbol_upper = token_symbol.upper()
        for token in result.tokens:
            if token.symbol.upper() == token_symbol_upper:
                return token

        return None

    async def get_chain_breakdown(
        self,
        account_address: str,
    ) -> Dict[int, Decimal]:
        """
        Get total USD value breakdown by chain.

        Args:
            account_address: Smart Account address

        Returns:
            Dict mapping chain_id to total USD value on that chain
        """
        result = await self.get_unified_balances(account_address)
        if not result.success:
            return {}

        breakdown: Dict[int, Decimal] = {}
        for token in result.tokens:
            for chain_balance in token.chains:
                if chain_balance.chain_id not in breakdown:
                    breakdown[chain_balance.chain_id] = Decimal("0")
                breakdown[chain_balance.chain_id] += chain_balance.usd_value

        return breakdown

    async def find_best_source_chain(
        self,
        account_address: str,
        token_symbol: str,
        amount_usd: Decimal,
    ) -> Optional[int]:
        """
        Find the best chain to source funds from for an intent.

        Considers:
        - Token availability
        - Sufficient balance
        - Gas costs (prefer L2s)

        Args:
            account_address: Smart Account address
            token_symbol: Token to source
            amount_usd: Required USD value

        Returns:
            Best chain_id, or None if insufficient balance
        """
        token = await self.get_token_balance(account_address, token_symbol)
        if not token:
            return None

        # Filter chains with sufficient balance
        viable_chains = [
            cb for cb in token.chains
            if cb.usd_value >= amount_usd
        ]

        if not viable_chains:
            return None

        # Prefer L2s (lower gas costs)
        l2_preference = [8453, 42161, 10, 137]  # Base, Arbitrum, Optimism, Polygon

        for preferred_chain in l2_preference:
            for cb in viable_chains:
                if cb.chain_id == preferred_chain:
                    return cb.chain_id

        # Fall back to first viable chain
        return viable_chains[0].chain_id

    def _get_cached(self, cache_key: str) -> Optional[UnifiedBalanceResult]:
        """Get cached result if still valid."""
        if cache_key not in self._cache:
            return None

        cached = self._cache[cache_key]
        if time.time() - cached.timestamp > self._cache_ttl:
            del self._cache[cache_key]
            return None

        # Mark as cached
        cached.cached = True
        return cached

    def clear_cache(self, account_address: Optional[str] = None) -> None:
        """
        Clear cached balances.

        Args:
            account_address: Clear only for this address, or all if None
        """
        if account_address:
            account_address = account_address.lower()
            keys_to_delete = [
                k for k in self._cache
                if k.startswith(account_address)
            ]
            for key in keys_to_delete:
                del self._cache[key]
        else:
            self._cache.clear()


# Singleton instance
_unified_balance_service: Optional[UnifiedBalanceService] = None


def get_unified_balance_service() -> UnifiedBalanceService:
    """Get the singleton UnifiedBalance service instance."""
    global _unified_balance_service
    if _unified_balance_service is None:
        _unified_balance_service = UnifiedBalanceService()
    return _unified_balance_service
