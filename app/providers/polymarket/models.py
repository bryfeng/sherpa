"""
Polymarket Data Models

Models for markets, positions, orders, and trades on Polymarket.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MarketCategory(str, Enum):
    """Polymarket market categories."""

    POLITICS = "politics"
    CRYPTO = "crypto"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    ECONOMICS = "economics"
    WEATHER = "weather"
    CURRENT_EVENTS = "current_events"
    OTHER = "other"


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    GTC = "GTC"  # Good Till Cancelled
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class OrderStatus(str, Enum):
    """Order status."""

    LIVE = "LIVE"
    MATCHED = "MATCHED"
    CANCELLED = "CANCELLED"


class Outcome(BaseModel):
    """A market outcome (e.g., 'Yes' or 'No')."""

    token_id: str = Field(..., alias="tokenId", description="Token ID for this outcome")
    outcome: str = Field(..., description="Outcome name (e.g., 'Yes', 'No')")
    price: Decimal = Field(..., description="Current price (0.00 to 1.00)")
    winner: Optional[bool] = Field(None, description="True if this outcome won")

    @property
    def implied_probability(self) -> float:
        """Price as percentage probability."""
        return float(self.price) * 100

    class Config:
        populate_by_name = True


class Market(BaseModel):
    """A Polymarket prediction market."""

    condition_id: str = Field(..., alias="conditionId", description="Market condition ID")
    question_id: str = Field("", alias="questionId", description="Question ID")
    question: str = Field(..., description="Market question")
    description: str = Field("", description="Market description")

    # Outcomes
    outcomes: List[str] = Field(default_factory=list, description="Outcome names")
    outcome_prices: List[str] = Field(
        default_factory=list, alias="outcomePrices", description="Outcome prices as strings"
    )
    tokens: List[Outcome] = Field(default_factory=list, description="Token details per outcome")

    # Market state
    volume: Decimal = Field(Decimal("0"), description="Total volume in USDC")
    volume_24h: Decimal = Field(Decimal("0"), alias="volume24hr", description="24h volume")
    liquidity: Decimal = Field(Decimal("0"), description="Available liquidity")
    active: bool = Field(True, description="Whether market is active for trading")
    closed: bool = Field(False, description="Whether market is closed")
    archived: bool = Field(False, description="Whether market is archived")
    accepting_orders: bool = Field(True, alias="acceptingOrders")

    # Resolution
    end_date: Optional[datetime] = Field(None, alias="endDate", description="Market end date")
    resolved: bool = Field(False, description="Whether market has resolved")
    resolution_source: Optional[str] = Field(None, alias="resolutionSource")

    # Metadata
    slug: str = Field("", description="URL slug")
    image: Optional[str] = Field(None, description="Market image URL")
    icon: Optional[str] = Field(None, description="Market icon URL")
    tags: List[str] = Field(default_factory=list, description="Market tags")

    # Event relationship
    event_slug: Optional[str] = Field(None, alias="eventSlug")

    class Config:
        populate_by_name = True

    @property
    def category(self) -> MarketCategory:
        """Infer category from tags."""
        tag_lower = [t.lower() for t in self.tags]
        if any(t in tag_lower for t in ["politics", "election", "trump", "biden"]):
            return MarketCategory.POLITICS
        if any(t in tag_lower for t in ["crypto", "bitcoin", "ethereum", "btc", "eth"]):
            return MarketCategory.CRYPTO
        if any(t in tag_lower for t in ["sports", "nfl", "nba", "soccer", "football"]):
            return MarketCategory.SPORTS
        if any(t in tag_lower for t in ["entertainment", "movies", "tv", "oscars"]):
            return MarketCategory.ENTERTAINMENT
        return MarketCategory.OTHER


class Event(BaseModel):
    """A Polymarket event (group of related markets)."""

    id: str = Field(..., description="Event ID")
    slug: str = Field(..., description="URL slug")
    title: str = Field(..., description="Event title")
    description: str = Field("", description="Event description")

    # Markets
    markets: List[Market] = Field(default_factory=list, description="Markets in this event")

    # State
    active: bool = Field(True)
    closed: bool = Field(False)
    archived: bool = Field(False)

    # Metrics
    volume: Decimal = Field(Decimal("0"), description="Total event volume")
    liquidity: Decimal = Field(Decimal("0"), description="Total liquidity")
    competitive: bool = Field(False, description="Whether event is competitive")

    # Timestamps
    start_date: Optional[datetime] = Field(None, alias="startDate")
    end_date: Optional[datetime] = Field(None, alias="endDate")
    created_at: Optional[datetime] = Field(None, alias="createdAt")

    # Metadata
    image: Optional[str] = Field(None)
    icon: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class OrderBookLevel(BaseModel):
    """A single level in the order book."""

    price: Decimal = Field(..., description="Price level")
    size: Decimal = Field(..., description="Size at this level")


class OrderBook(BaseModel):
    """Order book for a market outcome."""

    token_id: str = Field(..., alias="tokenId")
    market: Optional[str] = Field(None, description="Market condition ID")

    bids: List[OrderBookLevel] = Field(default_factory=list, description="Buy orders")
    asks: List[OrderBookLevel] = Field(default_factory=list, description="Sell orders")

    timestamp: Optional[datetime] = Field(None)

    class Config:
        populate_by_name = True

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Highest bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Lowest ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid-market price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class Position(BaseModel):
    """User position in a market."""

    market_id: str = Field(..., alias="conditionId", description="Market condition ID")
    outcome_index: int = Field(..., alias="outcomeIndex", description="Outcome index (0 or 1)")
    token_id: str = Field(..., alias="tokenId")

    # Position details
    size: Decimal = Field(..., description="Number of shares")
    avg_price: Decimal = Field(Decimal("0"), alias="avgPrice", description="Average entry price")
    current_price: Decimal = Field(Decimal("0"), alias="currentPrice")

    # P&L
    realized_pnl: Decimal = Field(Decimal("0"), alias="realizedPnl")
    unrealized_pnl: Decimal = Field(Decimal("0"), alias="unrealizedPnl")
    cost_basis: Decimal = Field(Decimal("0"), alias="costBasis")
    current_value: Decimal = Field(Decimal("0"), alias="currentValue")

    # Market info (populated separately)
    market_question: Optional[str] = Field(None, alias="marketQuestion")
    outcome_name: Optional[str] = Field(None, alias="outcomeName")

    class Config:
        populate_by_name = True

    @property
    def pnl_percent(self) -> Optional[float]:
        """P&L as percentage of cost basis."""
        if self.cost_basis and self.cost_basis > 0:
            return float(self.unrealized_pnl / self.cost_basis) * 100
        return None


class Order(BaseModel):
    """A Polymarket order."""

    id: str = Field(..., description="Order ID")
    market_id: str = Field(..., alias="conditionId", description="Market condition ID")
    token_id: str = Field(..., alias="tokenId")
    outcome: str = Field("", description="Outcome name")

    # Order details
    side: OrderSide = Field(..., description="BUY or SELL")
    type: OrderType = Field(OrderType.GTC, description="Order type")
    status: OrderStatus = Field(OrderStatus.LIVE)

    # Pricing
    price: Decimal = Field(..., description="Limit price")
    size: Decimal = Field(..., alias="originalSize", description="Original order size")
    size_matched: Decimal = Field(Decimal("0"), alias="sizeMatched", description="Size filled")

    # User
    maker_address: str = Field(..., alias="makerAddress")

    # Timestamps
    created_at: Optional[datetime] = Field(None, alias="createdAt")
    expiration: Optional[datetime] = Field(None)

    class Config:
        populate_by_name = True

    @property
    def size_remaining(self) -> Decimal:
        """Remaining unfilled size."""
        return self.size - self.size_matched

    @property
    def fill_percent(self) -> float:
        """Percentage filled."""
        if self.size > 0:
            return float(self.size_matched / self.size) * 100
        return 0.0


class Trade(BaseModel):
    """A completed trade."""

    id: str = Field(..., description="Trade ID")
    market_id: str = Field(..., alias="conditionId")
    token_id: str = Field(..., alias="tokenId")

    # Trade details
    side: OrderSide = Field(...)
    price: Decimal = Field(...)
    size: Decimal = Field(...)
    fee: Decimal = Field(Decimal("0"))

    # Parties
    maker_address: str = Field(..., alias="makerAddress")
    taker_address: Optional[str] = Field(None, alias="takerAddress")

    # Timestamps
    timestamp: datetime = Field(...)
    transaction_hash: Optional[str] = Field(None, alias="transactionHash")

    class Config:
        populate_by_name = True

    @property
    def value_usd(self) -> Decimal:
        """Trade value in USDC."""
        return self.price * self.size


class OrderRequest(BaseModel):
    """Request to create an order."""

    token_id: str = Field(..., alias="tokenId")
    side: OrderSide = Field(...)
    price: Decimal = Field(..., ge=0, le=1, description="Price 0.00-1.00")
    size: Decimal = Field(..., gt=0, description="Number of shares")
    order_type: OrderType = Field(OrderType.GTC, alias="orderType")
    expiration: Optional[int] = Field(None, description="Unix timestamp for GTD orders")

    class Config:
        populate_by_name = True


class MarketOrderRequest(BaseModel):
    """Request to create a market order."""

    token_id: str = Field(..., alias="tokenId")
    side: OrderSide = Field(...)
    amount: Decimal = Field(..., gt=0, description="Amount in USDC")

    class Config:
        populate_by_name = True
