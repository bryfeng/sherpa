"""
Polymarket Provider

Client for interacting with Polymarket prediction markets.
Uses CLOB API for trading, Gamma API for market discovery, Data API for positions.
"""

from .models import (
    Market,
    Event,
    Outcome,
    Position,
    OrderBook,
    OrderBookLevel,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Trade,
    MarketCategory,
)
from .client import PolymarketClient, get_polymarket_client

__all__ = [
    # Models
    "Market",
    "Event",
    "Outcome",
    "Position",
    "OrderBook",
    "OrderBookLevel",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Trade",
    "MarketCategory",
    # Client
    "PolymarketClient",
    "get_polymarket_client",
]
