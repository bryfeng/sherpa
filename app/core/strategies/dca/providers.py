"""
DCA Provider Adapters

Thin adapter classes that bridge the DCAExecutor's expected provider interfaces
to the existing raw providers (Relay, Coingecko, Alchemy RPC, PolicyEngine).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

import httpx

from app.config import settings
from app.providers.coingecko import CoingeckoProvider
from app.providers.relay import RelayProvider
from app.core.policy.engine import PolicyEngine
from app.core.policy.models import ActionContext, PolicyResult

logger = logging.getLogger(__name__)

# ── Chain ID → Alchemy RPC slug mapping (matches websocket_monitor.py) ──────
_ALCHEMY_SLUG: Dict[int, str] = {
    1: "eth-mainnet",
    10: "opt-mainnet",
    137: "polygon-mainnet",
    42161: "arb-mainnet",
    8453: "base-mainnet",
}


class DCASwapProvider:
    """Wraps RelayProvider to expose the get_quote interface expected by DCAExecutor."""

    def __init__(self) -> None:
        self._relay = RelayProvider()

    async def get_quote(
        self,
        from_token: str,
        to_token: str,
        amount: str,
        chain_id: int,
        slippage_bps: int = 100,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a swap quote via the Relay v2 API.

        Returns a dict with at least:
          - toAmount (str): output amount in smallest units
          - priceImpactBps (int): estimated price impact
          - routeDescription (str): human-readable route
        """
        try:
            # Relay v2 quote payload
            # Uses a zero address as placeholder user; the actual smart account
            # address is injected later during intent-based execution.
            payload = {
                "user": "0x0000000000000000000000000000000000000000",
                "originChainId": chain_id,
                "destinationChainId": chain_id,
                "originCurrency": from_token,
                "destinationCurrency": to_token,
                "amount": amount,
                "tradeType": "EXACT_INPUT",
            }

            data = await self._relay.quote(payload)

            # Normalize to the shape DCAExecutor expects
            steps = data.get("steps", [])
            to_amount = "0"
            route_desc = "Relay"

            if steps:
                # Walk through steps to find output amount
                for step in steps:
                    items = step.get("items", [])
                    for item in items:
                        item_data = item.get("data", {})
                        if "amountOut" in item_data:
                            to_amount = item_data["amountOut"]
                        elif "toAmount" in item_data:
                            to_amount = item_data["toAmount"]

            # Relay v2 may return details.currencyOut.amountFormatted at top level
            details = data.get("details", {})
            currency_out = details.get("currencyOut", {})
            if currency_out.get("amount"):
                to_amount = currency_out["amount"]

            return {
                "toAmount": to_amount,
                "priceImpactBps": 0,  # Relay doesn't expose price impact directly
                "routeDescription": route_desc,
                "_raw": data,
            }
        except Exception as e:
            logger.error(f"DCASwapProvider.get_quote failed: {e}")
            return None


class DCAPricingProvider:
    """Wraps CoingeckoProvider to expose the pricing interface expected by DCAExecutor."""

    def __init__(self) -> None:
        self._cg = CoingeckoProvider()

    async def get_price(self, address: str, chain_id: int) -> Decimal:
        """Get current USD price for a token by contract address."""
        try:
            prices = await self._cg.get_token_prices([address])
            addr_lower = address.lower()
            if addr_lower in prices and "price_usd" in prices[addr_lower]:
                return Decimal(str(prices[addr_lower]["price_usd"]))
        except Exception as e:
            logger.error(f"DCAPricingProvider.get_price failed: {e}")

        return Decimal("0")

    async def get_eth_price(self, chain_id: int = 1) -> Decimal:
        """Get current ETH price in USD."""
        try:
            data = await self._cg.get_eth_price()
            if "price_usd" in data:
                return Decimal(str(data["price_usd"]))
        except Exception as e:
            logger.error(f"DCAPricingProvider.get_eth_price failed: {e}")

        return Decimal("0")

    async def get_sol_price(self) -> Decimal:
        """Get current SOL price in USD."""
        try:
            headers: Dict[str, str] = {}
            if settings.coingecko_api_key:
                headers["X-CG-Demo-API-Key"] = settings.coingecko_api_key

            params = {
                "ids": "solana",
                "vs_currencies": "usd",
            }

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    headers=headers,
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if "solana" in data and "usd" in data["solana"]:
                    return Decimal(str(data["solana"]["usd"]))
        except Exception as e:
            logger.error(f"DCAPricingProvider.get_sol_price failed: {e}")

        return Decimal("0")


class DCAGasProvider:
    """Uses public Alchemy RPC eth_gasPrice to get current gas prices."""

    async def get_gas_price(self, chain_id: int) -> Decimal:
        """
        Get current gas price in gwei for the given EVM chain.

        Returns Decimal("0") on failure (executor will still proceed but
        gas cost estimate will be zero).
        """
        slug = _ALCHEMY_SLUG.get(chain_id)
        if not slug or not settings.alchemy_api_key:
            logger.warning(
                f"DCAGasProvider: no Alchemy RPC for chain {chain_id}, returning 0"
            )
            return Decimal("0")

        rpc_url = f"https://{slug}.g.alchemy.com/v2/{settings.alchemy_api_key}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    rpc_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_gasPrice",
                        "params": [],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                hex_price = data.get("result", "0x0")
                wei = int(hex_price, 16)
                gwei = Decimal(wei) / Decimal("1000000000")
                return gwei
        except Exception as e:
            logger.error(f"DCAGasProvider.get_gas_price failed for chain {chain_id}: {e}")
            return Decimal("0")


class DCASessionManager:
    """
    Thin wrapper for the executor's legacy session-key validation path.

    Smart Session validation is handled directly via Convex queries inside
    DCAExecutor._validate_smart_session, so this adapter only needs to exist
    so the DCAService constructor sees a truthy session_manager arg.
    """

    def __init__(self, convex_client: Any) -> None:
        self._convex = convex_client


class DCAPolicyEngine:
    """
    Wraps the sync PolicyEngine.evaluate() behind an async interface
    expected by DCAExecutor._validate_policy.
    """

    def __init__(self) -> None:
        self._engine = PolicyEngine()

    async def evaluate(self, context: ActionContext) -> PolicyResult:
        """Evaluate an action context against the policy engine."""
        return self._engine.evaluate(context)
