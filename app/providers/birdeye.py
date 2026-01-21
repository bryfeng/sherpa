"""
Birdeye API Provider

Provides access to Birdeye's DeFi analytics API for:
- Top traders by token
- Wallet portfolio and PnL
- Token information

Docs: https://docs.birdeye.so/reference
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class BirdeyeProvider:
    """Birdeye API provider for trader discovery and wallet analytics."""

    name = "birdeye"
    timeout_s = 30
    base_url = "https://public-api.birdeye.so"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "birdeye_api_key", None)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                headers=self._build_headers(),
            )
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers

    async def ready(self) -> bool:
        """Check if provider is ready."""
        return bool(self.api_key)

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # Top Traders
    # =========================================================================

    async def get_top_traders_by_token(
        self,
        token_address: str,
        chain: str = "solana",
        time_frame: str = "24h",
        sort_by: str = "pnl",
        sort_type: str = "desc",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get top traders for a specific token.

        Args:
            token_address: Token mint address
            chain: Blockchain (default: solana)
            time_frame: Time frame (24h, 7d, 30d)
            sort_by: Sort field (pnl, volume, trades)
            sort_type: Sort direction (asc, desc)
            limit: Max results (default: 20, max: 100)
            offset: Pagination offset

        Returns:
            Dict with traders list and pagination info
        """
        client = await self._get_client()

        params = {
            "address": token_address,
            "time_frame": time_frame,
            "sort_by": sort_by,
            "sort_type": sort_type,
            "offset": offset,
            "limit": min(limit, 100),
        }

        try:
            response = await client.get(
                f"{self.base_url}/defi/v2/tokens/top_traders",
                params=params,
                headers={"x-chain": chain, **self._build_headers()},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                logger.warning(f"Birdeye API error: {data.get('message')}")
                return {"traders": [], "total": 0, "error": data.get("message")}

            traders = data.get("data", {}).get("items", [])

            return {
                "traders": [self._parse_trader(t) for t in traders],
                "total": data.get("data", {}).get("total", len(traders)),
                "token_address": token_address,
                "time_frame": time_frame,
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Birdeye API HTTP error: {e}")
            return {"traders": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.error(f"Birdeye API error: {e}", exc_info=True)
            return {"traders": [], "total": 0, "error": str(e)}

    def _parse_trader(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trader data from API response."""
        return {
            "address": data.get("owner") or data.get("address"),
            "pnl_usd": Decimal(str(data.get("pnl", 0))),
            "pnl_percent": data.get("pnlPercent"),
            "volume_usd": Decimal(str(data.get("volume", 0))),
            "trade_count": data.get("tradeCount") or data.get("trades"),
            "buy_count": data.get("buyCount"),
            "sell_count": data.get("sellCount"),
            "avg_buy_price": data.get("avgBuyPrice"),
            "avg_sell_price": data.get("avgSellPrice"),
            "win_rate": data.get("winRate"),
            "last_active_at": data.get("lastActiveAt"),
        }

    # =========================================================================
    # Wallet Analytics
    # =========================================================================

    async def get_wallet_portfolio(
        self,
        wallet_address: str,
        chain: str = "solana",
    ) -> Dict[str, Any]:
        """
        Get wallet portfolio with token holdings.

        Args:
            wallet_address: Wallet address
            chain: Blockchain (default: solana)

        Returns:
            Portfolio with holdings and total value
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.base_url}/v1/wallet/token_list",
                params={"wallet": wallet_address},
                headers={"x-chain": chain, **self._build_headers()},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                return {"holdings": [], "total_value_usd": Decimal("0"), "error": data.get("message")}

            items = data.get("data", {}).get("items", [])
            total_value = sum(Decimal(str(item.get("valueUsd", 0))) for item in items)

            return {
                "address": wallet_address,
                "chain": chain,
                "holdings": [
                    {
                        "token_address": item.get("address"),
                        "symbol": item.get("symbol"),
                        "name": item.get("name"),
                        "balance": Decimal(str(item.get("uiAmount", 0))),
                        "value_usd": Decimal(str(item.get("valueUsd", 0))),
                        "price_usd": Decimal(str(item.get("priceUsd", 0))),
                    }
                    for item in items
                ],
                "total_value_usd": total_value,
                "token_count": len(items),
            }

        except Exception as e:
            logger.error(f"Error fetching wallet portfolio: {e}")
            return {"holdings": [], "total_value_usd": Decimal("0"), "error": str(e)}

    async def get_wallet_trade_history(
        self,
        wallet_address: str,
        chain: str = "solana",
        limit: int = 50,
        before_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get wallet's recent trade history.

        Args:
            wallet_address: Wallet address
            chain: Blockchain
            limit: Max trades to return
            before_time: Unix timestamp for pagination

        Returns:
            List of recent trades
        """
        client = await self._get_client()

        params: Dict[str, Any] = {
            "wallet": wallet_address,
            "limit": min(limit, 100),
        }
        if before_time:
            params["before_time"] = before_time

        try:
            response = await client.get(
                f"{self.base_url}/v1/wallet/tx_list",
                params=params,
                headers={"x-chain": chain, **self._build_headers()},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                return {"trades": [], "error": data.get("message")}

            items = data.get("data", {}).get("items", [])

            # Filter for swap transactions
            swaps = [
                {
                    "tx_hash": item.get("txHash"),
                    "timestamp": datetime.fromtimestamp(
                        item.get("blockUnixTime", 0), tz=timezone.utc
                    ),
                    "from_token": item.get("from", {}).get("symbol"),
                    "from_amount": item.get("from", {}).get("uiAmount"),
                    "from_value_usd": item.get("from", {}).get("nearestPrice"),
                    "to_token": item.get("to", {}).get("symbol"),
                    "to_amount": item.get("to", {}).get("uiAmount"),
                    "to_value_usd": item.get("to", {}).get("nearestPrice"),
                    "source": item.get("source"),
                }
                for item in items
                if item.get("txType") == "swap"
            ]

            return {
                "address": wallet_address,
                "trades": swaps,
                "total": len(swaps),
            }

        except Exception as e:
            logger.error(f"Error fetching wallet trades: {e}")
            return {"trades": [], "error": str(e)}

    async def get_wallet_pnl(
        self,
        wallet_address: str,
        chain: str = "solana",
    ) -> Dict[str, Any]:
        """
        Get wallet's overall PnL metrics.

        Args:
            wallet_address: Wallet address
            chain: Blockchain

        Returns:
            PnL metrics including realized/unrealized gains
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.base_url}/v1/wallet/pnl",
                params={"wallet": wallet_address},
                headers={"x-chain": chain, **self._build_headers()},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                return {"error": data.get("message")}

            pnl_data = data.get("data", {})

            return {
                "address": wallet_address,
                "chain": chain,
                "total_pnl_usd": Decimal(str(pnl_data.get("totalPnl", 0))),
                "realized_pnl_usd": Decimal(str(pnl_data.get("realizedPnl", 0))),
                "unrealized_pnl_usd": Decimal(str(pnl_data.get("unrealizedPnl", 0))),
                "total_invested_usd": Decimal(str(pnl_data.get("totalInvested", 0))),
                "total_sold_usd": Decimal(str(pnl_data.get("totalSold", 0))),
                "win_rate": pnl_data.get("winRate"),
                "trade_count": pnl_data.get("tradeCount"),
            }

        except Exception as e:
            logger.error(f"Error fetching wallet PnL: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Token Information
    # =========================================================================

    async def get_token_info(
        self,
        token_address: str,
        chain: str = "solana",
    ) -> Dict[str, Any]:
        """
        Get detailed token information.

        Args:
            token_address: Token mint address
            chain: Blockchain

        Returns:
            Token metadata and market info
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.base_url}/defi/token_overview",
                params={"address": token_address},
                headers={"x-chain": chain, **self._build_headers()},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                return {"error": data.get("message")}

            token_data = data.get("data", {})

            return {
                "address": token_address,
                "symbol": token_data.get("symbol"),
                "name": token_data.get("name"),
                "decimals": token_data.get("decimals"),
                "price_usd": Decimal(str(token_data.get("price", 0))),
                "price_change_24h": token_data.get("priceChange24h"),
                "volume_24h_usd": Decimal(str(token_data.get("v24hUSD", 0))),
                "market_cap_usd": Decimal(str(token_data.get("mc", 0))),
                "liquidity_usd": Decimal(str(token_data.get("liquidity", 0))),
                "holder_count": token_data.get("holder"),
                "trade_24h": token_data.get("trade24h"),
            }

        except Exception as e:
            logger.error(f"Error fetching token info: {e}")
            return {"error": str(e)}


# Singleton instance
_provider_instance: Optional[BirdeyeProvider] = None


def get_birdeye_provider() -> BirdeyeProvider:
    """Get the singleton Birdeye provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = BirdeyeProvider()
    return _provider_instance
