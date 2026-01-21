"""
Polymarket Trading Models

Models for portfolio, analysis, and trade execution.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.providers.polymarket.models import Market, Position, OrderSide


class PortfolioPosition(BaseModel):
    """Enhanced position with market context."""

    # From Position
    market_id: str = Field(..., alias="marketId")
    token_id: str = Field(..., alias="tokenId")
    outcome_index: int = Field(..., alias="outcomeIndex")

    # Position data
    size: Decimal = Field(..., description="Number of shares")
    avg_price: Decimal = Field(..., alias="avgPrice")
    current_price: Decimal = Field(..., alias="currentPrice")
    cost_basis: Decimal = Field(..., alias="costBasis")
    current_value: Decimal = Field(..., alias="currentValue")

    # P&L
    unrealized_pnl: Decimal = Field(..., alias="unrealizedPnl")
    unrealized_pnl_pct: Optional[float] = Field(None, alias="unrealizedPnlPct")
    realized_pnl: Decimal = Field(Decimal("0"), alias="realizedPnl")

    # Market context
    market_question: str = Field(..., alias="marketQuestion")
    outcome_name: str = Field(..., alias="outcomeName")
    market_end_date: Optional[datetime] = Field(None, alias="marketEndDate")
    market_resolved: bool = Field(False, alias="marketResolved")

    class Config:
        populate_by_name = True


class PolymarketPortfolio(BaseModel):
    """User's complete Polymarket portfolio."""

    address: str = Field(..., description="Wallet address")

    # Positions
    positions: List[PortfolioPosition] = Field(default_factory=list)

    # Aggregates
    total_value: Decimal = Field(Decimal("0"), alias="totalValue")
    total_cost_basis: Decimal = Field(Decimal("0"), alias="totalCostBasis")
    total_unrealized_pnl: Decimal = Field(Decimal("0"), alias="totalUnrealizedPnl")
    total_realized_pnl: Decimal = Field(Decimal("0"), alias="totalRealizedPnl")
    total_pnl: Decimal = Field(Decimal("0"), alias="totalPnl")
    total_pnl_pct: Optional[float] = Field(None, alias="totalPnlPct")

    # Stats
    open_positions_count: int = Field(0, alias="openPositionsCount")
    winning_positions: int = Field(0, alias="winningPositions")
    losing_positions: int = Field(0, alias="losingPositions")

    # Timestamp
    updated_at: datetime = Field(default_factory=datetime.utcnow, alias="updatedAt")

    class Config:
        populate_by_name = True


class MarketAnalysis(BaseModel):
    """AI-generated analysis of a market."""

    market_id: str = Field(..., alias="marketId")
    question: str = Field(...)

    # Probabilities
    current_yes_price: Decimal = Field(..., alias="currentYesPrice")
    current_no_price: Decimal = Field(..., alias="currentNoPrice")

    # Analysis
    summary: str = Field(..., description="Brief summary of the market")
    key_factors: List[str] = Field(default_factory=list, alias="keyFactors")
    sentiment: str = Field("neutral", description="bullish, bearish, or neutral")
    confidence: float = Field(0.5, description="AI confidence in analysis 0-1")

    # Recommendation
    recommended_side: Optional[str] = Field(None, alias="recommendedSide")
    recommended_reason: Optional[str] = Field(None, alias="recommendedReason")

    # Context
    recent_news: List[str] = Field(default_factory=list, alias="recentNews")
    volume_trend: str = Field("stable", alias="volumeTrend")  # increasing, decreasing, stable

    # Meta
    analyzed_at: datetime = Field(default_factory=datetime.utcnow, alias="analyzedAt")

    class Config:
        populate_by_name = True


class TradeQuote(BaseModel):
    """Quote for a potential trade."""

    market_id: str = Field(..., alias="marketId")
    token_id: str = Field(..., alias="tokenId")
    side: OrderSide = Field(...)
    outcome_name: str = Field(..., alias="outcomeName")

    # Trade details
    amount_usd: Decimal = Field(..., alias="amountUsd", description="Input amount in USDC")
    shares: Decimal = Field(..., description="Number of shares to receive/sell")
    price: Decimal = Field(..., description="Effective price per share")
    avg_price: Decimal = Field(..., alias="avgPrice", description="Average fill price")

    # Fees & slippage
    estimated_fee: Decimal = Field(Decimal("0"), alias="estimatedFee")
    price_impact_pct: float = Field(0, alias="priceImpactPct")

    # If buying: potential payout
    max_payout: Optional[Decimal] = Field(None, alias="maxPayout")
    potential_profit: Optional[Decimal] = Field(None, alias="potentialProfit")
    potential_profit_pct: Optional[float] = Field(None, alias="potentialProfitPct")

    # Execution
    requires_approval: bool = Field(True, alias="requiresApproval")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")

    class Config:
        populate_by_name = True


class TradeResult(BaseModel):
    """Result of executing a trade."""

    success: bool = Field(...)
    trade_id: Optional[str] = Field(None, alias="tradeId")
    transaction_hash: Optional[str] = Field(None, alias="transactionHash")

    # Trade details
    market_id: str = Field(..., alias="marketId")
    token_id: str = Field(..., alias="tokenId")
    side: OrderSide = Field(...)
    outcome_name: str = Field(..., alias="outcomeName")

    # Execution
    shares_filled: Decimal = Field(Decimal("0"), alias="sharesFilled")
    avg_fill_price: Decimal = Field(Decimal("0"), alias="avgFillPrice")
    total_cost: Decimal = Field(Decimal("0"), alias="totalCost")
    fees_paid: Decimal = Field(Decimal("0"), alias="feesPaid")

    # Error
    error_message: Optional[str] = Field(None, alias="errorMessage")

    # Timestamps
    executed_at: datetime = Field(default_factory=datetime.utcnow, alias="executedAt")

    class Config:
        populate_by_name = True
