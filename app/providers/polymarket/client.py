"""
Polymarket API Client

Client for interacting with Polymarket's APIs:
- CLOB API (clob.polymarket.com) - Trading, orderbooks, prices
- Gamma API (gamma-api.polymarket.com) - Market/event discovery
- Data API (data-api.polymarket.com) - User positions, activity
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx

from .models import (
    Market,
    Event,
    Outcome,
    Position,
    OrderBook,
    OrderBookLevel,
    Order,
    Trade,
    OrderSide,
    MarketCategory,
)

logger = logging.getLogger(__name__)


class PolymarketClient:
    """
    Client for Polymarket prediction markets.

    Provides access to:
    - Market discovery and search
    - Order book and pricing data
    - User positions and trade history
    - Order placement (requires authentication)
    """

    # API Base URLs
    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    DATA_URL = "https://data-api.polymarket.com"

    def __init__(
        self,
        timeout: float = 30.0,
    ):
        """
        Initialize Polymarket client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    # =========================================================================
    # Market Discovery (Gamma API)
    # =========================================================================

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Market]:
        """
        Get list of markets.

        Args:
            active: Include active markets
            closed: Include closed markets
            limit: Max results to return
            offset: Pagination offset

        Returns:
            List of markets
        """
        client = await self._get_client()

        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }

        try:
            response = await client.get(f"{self.GAMMA_URL}/markets", params=params)
            response.raise_for_status()
            data = response.json()

            markets = []
            for item in data:
                try:
                    market = self._parse_market(item)
                    markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
                    continue

            return markets

        except httpx.HTTPError as e:
            logger.error(f"Failed to get markets: {e}")
            raise

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """
        Get a single market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            Market or None if not found
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.GAMMA_URL}/markets/{condition_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data)

        except httpx.HTTPError as e:
            logger.error(f"Failed to get market {condition_id}: {e}")
            raise

    async def search_markets(
        self,
        query: str,
        limit: int = 50,
    ) -> List[Market]:
        """
        Search markets by question text.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching markets
        """
        # Gamma API doesn't have dedicated search, so we filter client-side
        # For production, consider caching markets or using a search index
        markets = await self.get_markets(active=True, limit=500)

        query_lower = query.lower()
        matches = [
            m for m in markets
            if query_lower in m.question.lower()
            or query_lower in m.description.lower()
            or any(query_lower in tag.lower() for tag in m.tags)
        ]

        return matches[:limit]

    async def get_markets_by_category(
        self,
        category: MarketCategory,
        limit: int = 50,
    ) -> List[Market]:
        """
        Get markets filtered by category.

        Args:
            category: Market category
            limit: Max results

        Returns:
            List of markets in category
        """
        markets = await self.get_markets(active=True, limit=500)

        filtered = [m for m in markets if m.category == category]
        return filtered[:limit]

    async def get_events(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Event]:
        """
        Get list of events (groups of related markets).

        Args:
            active: Include active events
            limit: Max results
            offset: Pagination offset

        Returns:
            List of events
        """
        client = await self._get_client()

        params = {
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
        }

        try:
            response = await client.get(f"{self.GAMMA_URL}/events", params=params)
            response.raise_for_status()
            data = response.json()

            events = []
            for item in data:
                try:
                    event = self._parse_event(item)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event: {e}")
                    continue

            return events

        except httpx.HTTPError as e:
            logger.error(f"Failed to get events: {e}")
            raise

    async def get_event(self, event_id: str) -> Optional[Event]:
        """
        Get a single event by ID.

        Args:
            event_id: Event ID or slug

        Returns:
            Event or None
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.GAMMA_URL}/events/{event_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return self._parse_event(data)

        except httpx.HTTPError as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            raise

    async def get_trending_markets(
        self,
        limit: int = 20,
    ) -> List[Market]:
        """
        Get trending markets by 24h volume.

        Args:
            limit: Max results

        Returns:
            List of trending markets
        """
        markets = await self.get_markets(active=True, limit=200)

        # Sort by 24h volume
        sorted_markets = sorted(
            markets,
            key=lambda m: float(m.volume_24h or 0),
            reverse=True,
        )

        return sorted_markets[:limit]

    async def get_closing_soon(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> List[Market]:
        """
        Get markets closing within timeframe.

        Args:
            hours: Hours from now
            limit: Max results

        Returns:
            List of markets closing soon
        """
        from datetime import timedelta

        markets = await self.get_markets(active=True, limit=500)

        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours)

        closing = [
            m for m in markets
            if m.end_date and now < m.end_date <= cutoff
        ]

        # Sort by end date (soonest first)
        closing.sort(key=lambda m: m.end_date or datetime.max)

        return closing[:limit]

    # =========================================================================
    # Pricing & Order Book (CLOB API)
    # =========================================================================

    async def get_price(
        self,
        token_id: str,
        side: OrderSide = OrderSide.BUY,
    ) -> Optional[Decimal]:
        """
        Get current price for a token.

        Args:
            token_id: Token ID
            side: BUY or SELL price

        Returns:
            Current price or None
        """
        client = await self._get_client()

        try:
            params = {"token_id": token_id, "side": side.value}
            response = await client.get(f"{self.CLOB_URL}/price", params=params)
            response.raise_for_status()
            data = response.json()
            return Decimal(str(data.get("price", 0)))

        except httpx.HTTPError as e:
            logger.error(f"Failed to get price for {token_id}: {e}")
            return None

    async def get_midpoint(self, token_id: str) -> Optional[Decimal]:
        """
        Get midpoint price for a token.

        Args:
            token_id: Token ID

        Returns:
            Midpoint price or None
        """
        client = await self._get_client()

        try:
            params = {"token_id": token_id}
            response = await client.get(f"{self.CLOB_URL}/midpoint", params=params)
            response.raise_for_status()
            data = response.json()
            return Decimal(str(data.get("mid", 0)))

        except httpx.HTTPError as e:
            logger.error(f"Failed to get midpoint for {token_id}: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """
        Get order book for a token.

        Args:
            token_id: Token ID

        Returns:
            Order book or None
        """
        client = await self._get_client()

        try:
            params = {"token_id": token_id}
            response = await client.get(f"{self.CLOB_URL}/book", params=params)
            response.raise_for_status()
            data = response.json()

            return OrderBook(
                tokenId=token_id,
                bids=[
                    OrderBookLevel(price=Decimal(str(b["price"])), size=Decimal(str(b["size"])))
                    for b in data.get("bids", [])
                ],
                asks=[
                    OrderBookLevel(price=Decimal(str(a["price"])), size=Decimal(str(a["size"])))
                    for a in data.get("asks", [])
                ],
                timestamp=datetime.utcnow(),
            )

        except httpx.HTTPError as e:
            logger.error(f"Failed to get orderbook for {token_id}: {e}")
            return None

    async def get_spread(self, token_id: str) -> Optional[Dict[str, Decimal]]:
        """
        Get spread data for a token.

        Args:
            token_id: Token ID

        Returns:
            Dict with bid, ask, spread, mid
        """
        client = await self._get_client()

        try:
            params = {"token_id": token_id}
            response = await client.get(f"{self.CLOB_URL}/spread", params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "bid": Decimal(str(data.get("bid", 0))),
                "ask": Decimal(str(data.get("ask", 0))),
                "spread": Decimal(str(data.get("spread", 0))),
            }

        except httpx.HTTPError as e:
            logger.error(f"Failed to get spread for {token_id}: {e}")
            return None

    # =========================================================================
    # User Data (Data API)
    # =========================================================================

    async def get_positions(self, address: str) -> List[Position]:
        """
        Get user's positions.

        Args:
            address: User's wallet address

        Returns:
            List of positions
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.DATA_URL}/positions", params={"user": address})
            response.raise_for_status()
            data = response.json()

            positions = []
            for item in data:
                try:
                    position = self._parse_position(item)
                    positions.append(position)
                except Exception as e:
                    logger.warning(f"Failed to parse position: {e}")
                    continue

            return positions

        except httpx.HTTPError as e:
            logger.error(f"Failed to get positions for {address}: {e}")
            return []

    async def get_trades(
        self,
        address: str,
        limit: int = 100,
    ) -> List[Trade]:
        """
        Get user's trade history.

        Args:
            address: User's wallet address
            limit: Max results

        Returns:
            List of trades
        """
        client = await self._get_client()

        try:
            params = {"user": address, "limit": limit}
            response = await client.get(f"{self.DATA_URL}/trades", params=params)
            response.raise_for_status()
            data = response.json()

            trades = []
            for item in data:
                try:
                    trade = self._parse_trade(item)
                    trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to parse trade: {e}")
                    continue

            return trades

        except httpx.HTTPError as e:
            logger.error(f"Failed to get trades for {address}: {e}")
            return []

    async def get_open_orders(self, address: str) -> List[Order]:
        """
        Get user's open orders.

        Args:
            address: User's wallet address

        Returns:
            List of open orders
        """
        client = await self._get_client()

        try:
            response = await client.get(f"{self.CLOB_URL}/orders", params={"maker_address": address})
            response.raise_for_status()
            data = response.json()

            orders = []
            for item in data:
                try:
                    order = self._parse_order(item)
                    orders.append(order)
                except Exception as e:
                    logger.warning(f"Failed to parse order: {e}")
                    continue

            return orders

        except httpx.HTTPError as e:
            logger.error(f"Failed to get orders for {address}: {e}")
            return []

    # =========================================================================
    # Parsing Helpers
    # =========================================================================

    def _parse_market(self, data: Dict[str, Any]) -> Market:
        """Parse market from API response."""
        # Parse outcomes into tokens
        tokens = []
        outcomes = data.get("outcomes", [])
        outcome_prices = data.get("outcomePrices", [])
        clob_token_ids = data.get("clobTokenIds", [])

        for i, outcome_name in enumerate(outcomes):
            price = Decimal(outcome_prices[i]) if i < len(outcome_prices) else Decimal("0")
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
            tokens.append(Outcome(
                tokenId=token_id,
                outcome=outcome_name,
                price=price,
            ))

        return Market(
            conditionId=data.get("conditionId", data.get("condition_id", "")),
            questionId=data.get("questionId", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            outcomes=outcomes,
            outcomePrices=[str(p) for p in outcome_prices],
            tokens=tokens,
            volume=Decimal(str(data.get("volume", 0))),
            volume24hr=Decimal(str(data.get("volume24hr", data.get("volumeNum24hr", 0)))),
            liquidity=Decimal(str(data.get("liquidity", data.get("liquidityNum", 0)))),
            active=data.get("active", True),
            closed=data.get("closed", False),
            archived=data.get("archived", False),
            acceptingOrders=data.get("acceptingOrders", True),
            endDate=self._parse_datetime(data.get("endDate") or data.get("end_date_iso")),
            resolved=data.get("resolved", False),
            resolutionSource=data.get("resolutionSource"),
            slug=data.get("slug", ""),
            image=data.get("image"),
            icon=data.get("icon"),
            tags=data.get("tags", []),
            eventSlug=data.get("eventSlug"),
        )

    def _parse_event(self, data: Dict[str, Any]) -> Event:
        """Parse event from API response."""
        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(self._parse_market(m))
            except Exception:
                continue

        return Event(
            id=data.get("id", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            markets=markets,
            active=data.get("active", True),
            closed=data.get("closed", False),
            archived=data.get("archived", False),
            volume=Decimal(str(data.get("volume", 0))),
            liquidity=Decimal(str(data.get("liquidity", 0))),
            competitive=data.get("competitive", False),
            startDate=self._parse_datetime(data.get("startDate")),
            endDate=self._parse_datetime(data.get("endDate")),
            createdAt=self._parse_datetime(data.get("createdAt")),
            image=data.get("image"),
            icon=data.get("icon"),
            tags=data.get("tags", []),
        )

    def _parse_position(self, data: Dict[str, Any]) -> Position:
        """Parse position from API response."""
        return Position(
            conditionId=data.get("conditionId", data.get("condition_id", "")),
            outcomeIndex=data.get("outcomeIndex", data.get("outcome_index", 0)),
            tokenId=data.get("tokenId", data.get("token_id", "")),
            size=Decimal(str(data.get("size", 0))),
            avgPrice=Decimal(str(data.get("avgPrice", data.get("avg_price", 0)))),
            currentPrice=Decimal(str(data.get("currentPrice", data.get("current_price", 0)))),
            realizedPnl=Decimal(str(data.get("realizedPnl", data.get("realized_pnl", 0)))),
            unrealizedPnl=Decimal(str(data.get("unrealizedPnl", data.get("unrealized_pnl", 0)))),
            costBasis=Decimal(str(data.get("costBasis", data.get("cost_basis", 0)))),
            currentValue=Decimal(str(data.get("currentValue", data.get("current_value", 0)))),
        )

    def _parse_order(self, data: Dict[str, Any]) -> Order:
        """Parse order from API response."""
        from .models import OrderStatus, OrderType

        return Order(
            id=data.get("id", ""),
            conditionId=data.get("conditionId", data.get("asset_id", "")),
            tokenId=data.get("tokenId", data.get("token_id", "")),
            outcome=data.get("outcome", ""),
            side=OrderSide(data.get("side", "BUY")),
            type=OrderType(data.get("type", "GTC")),
            status=OrderStatus(data.get("status", "LIVE")),
            price=Decimal(str(data.get("price", 0))),
            originalSize=Decimal(str(data.get("originalSize", data.get("original_size", 0)))),
            sizeMatched=Decimal(str(data.get("sizeMatched", data.get("size_matched", 0)))),
            makerAddress=data.get("makerAddress", data.get("maker_address", "")),
            createdAt=self._parse_datetime(data.get("createdAt")),
            expiration=self._parse_datetime(data.get("expiration")),
        )

    def _parse_trade(self, data: Dict[str, Any]) -> Trade:
        """Parse trade from API response."""
        return Trade(
            id=data.get("id", ""),
            conditionId=data.get("conditionId", data.get("condition_id", "")),
            tokenId=data.get("tokenId", data.get("token_id", "")),
            side=OrderSide(data.get("side", "BUY")),
            price=Decimal(str(data.get("price", 0))),
            size=Decimal(str(data.get("size", 0))),
            fee=Decimal(str(data.get("fee", 0))),
            makerAddress=data.get("makerAddress", data.get("maker_address", "")),
            takerAddress=data.get("takerAddress", data.get("taker_address")),
            timestamp=self._parse_datetime(data.get("timestamp")) or datetime.utcnow(),
            transactionHash=data.get("transactionHash", data.get("transaction_hash")),
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            # Unix timestamp (seconds or milliseconds)
            if value > 1e12:
                value = value / 1000
            return datetime.utcfromtimestamp(value)
        if isinstance(value, str):
            try:
                # ISO format
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return None


# Singleton instance
_client_instance: Optional[PolymarketClient] = None


def get_polymarket_client() -> PolymarketClient:
    """Get singleton Polymarket client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = PolymarketClient()
    return _client_instance
