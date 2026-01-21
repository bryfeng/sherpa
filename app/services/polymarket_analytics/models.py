"""
Polymarket Trader Analytics Models

Models for tracking trader performance, positions, and trade history.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class TraderPosition(BaseModel):
    """A trader's position in a market."""

    market_id: str = Field(..., alias="marketId")
    market_question: str = Field(..., alias="marketQuestion")
    outcome: str
    token_id: str = Field(..., alias="tokenId")

    # Position details
    shares: Decimal
    avg_entry_price: Decimal = Field(..., alias="avgEntryPrice")
    current_price: Decimal = Field(..., alias="currentPrice")
    cost_basis: Decimal = Field(..., alias="costBasis")
    current_value: Decimal = Field(..., alias="currentValue")

    # P&L
    unrealized_pnl: Decimal = Field(..., alias="unrealizedPnl")
    unrealized_pnl_pct: Optional[float] = Field(None, alias="unrealizedPnlPct")

    # Market info
    market_end_date: Optional[datetime] = Field(None, alias="marketEndDate")
    market_resolved: bool = Field(False, alias="marketResolved")
    market_category: Optional[str] = Field(None, alias="marketCategory")

    # Timestamps
    opened_at: Optional[datetime] = Field(None, alias="openedAt")

    class Config:
        populate_by_name = True


class HistoricalTrade(BaseModel):
    """A historical trade made by a trader."""

    trade_id: str = Field(..., alias="tradeId")
    market_id: str = Field(..., alias="marketId")
    market_question: str = Field(..., alias="marketQuestion")
    outcome: str
    token_id: str = Field(..., alias="tokenId")

    # Trade details
    side: str  # BUY or SELL
    shares: Decimal
    price: Decimal
    value_usd: Decimal = Field(..., alias="valueUsd")
    fee: Decimal = Field(Decimal("0"))

    # Result (for closed positions)
    realized_pnl: Optional[Decimal] = Field(None, alias="realizedPnl")
    realized_pnl_pct: Optional[float] = Field(None, alias="realizedPnlPct")
    was_winner: Optional[bool] = Field(None, alias="wasWinner")

    # Resolution (if market resolved)
    market_resolved: bool = Field(False, alias="marketResolved")
    winning_outcome: Optional[str] = Field(None, alias="winningOutcome")
    payout_received: Optional[Decimal] = Field(None, alias="payoutReceived")

    # Timestamps
    timestamp: datetime
    transaction_hash: Optional[str] = Field(None, alias="transactionHash")

    class Config:
        populate_by_name = True


class PerformanceMetrics(BaseModel):
    """Performance metrics for a trader."""

    # Volume
    total_volume_usd: Decimal = Field(..., alias="totalVolumeUsd")
    volume_30d_usd: Decimal = Field(Decimal("0"), alias="volume30dUsd")

    # P&L
    total_pnl_usd: Decimal = Field(..., alias="totalPnlUsd")
    realized_pnl_usd: Decimal = Field(Decimal("0"), alias="realizedPnlUsd")
    unrealized_pnl_usd: Decimal = Field(Decimal("0"), alias="unrealizedPnlUsd")
    roi_pct: float = Field(..., alias="roiPct")

    # Win rate
    total_trades: int = Field(..., alias="totalTrades")
    winning_trades: int = Field(..., alias="winningTrades")
    losing_trades: int = Field(..., alias="losingTrades")
    win_rate: float = Field(..., alias="winRate")

    # Prediction quality (Brier score - lower is better)
    brier_score: Optional[float] = Field(None, alias="brierScore")
    calibration_score: Optional[float] = Field(None, alias="calibrationScore")

    # Streaks
    current_streak: int = Field(0, alias="currentStreak")  # Positive = winning, negative = losing
    best_streak: int = Field(0, alias="bestStreak")
    worst_streak: int = Field(0, alias="worstStreak")

    # Best/worst trades
    best_trade_pnl_usd: Optional[Decimal] = Field(None, alias="bestTradePnlUsd")
    worst_trade_pnl_usd: Optional[Decimal] = Field(None, alias="worstTradePnlUsd")

    class Config:
        populate_by_name = True


class PolymarketTraderProfile(BaseModel):
    """Comprehensive profile of a Polymarket trader."""

    address: str

    # Performance
    metrics: PerformanceMetrics

    # Current state
    active_positions: int = Field(..., alias="activePositions")
    total_exposure_usd: Decimal = Field(..., alias="totalExposureUsd")

    # Position sizing
    avg_position_size_usd: Decimal = Field(..., alias="avgPositionSizeUsd")
    max_position_size_usd: Decimal = Field(..., alias="maxPositionSizeUsd")
    max_single_bet_pct: float = Field(..., alias="maxSingleBetPct")

    # Categories
    preferred_categories: List[str] = Field(default_factory=list, alias="preferredCategories")
    category_performance: dict = Field(default_factory=dict, alias="categoryPerformance")

    # Timing
    avg_hold_time_days: float = Field(0, alias="avgHoldTimeDays")
    trades_per_week: float = Field(0, alias="tradesPerWeek")

    # Risk profile
    diversification_score: float = Field(0.5, alias="diversificationScore")  # 0-1, higher = more diversified
    risk_score: float = Field(0.5, alias="riskScore")  # 0-1, higher = riskier
    uses_leverage: bool = Field(False, alias="usesLeverage")

    # Social
    follower_count: int = Field(0, alias="followerCount")
    total_copied_volume_usd: Decimal = Field(Decimal("0"), alias="totalCopiedVolumeUsd")

    # Activity
    first_trade_at: Optional[datetime] = Field(None, alias="firstTradeAt")
    last_trade_at: Optional[datetime] = Field(None, alias="lastTradeAt")
    last_analyzed_at: Optional[datetime] = Field(None, alias="lastAnalyzedAt")

    # Data quality
    data_quality_score: float = Field(0.5, alias="dataQualityScore")
    trade_count_for_analysis: int = Field(0, alias="tradeCountForAnalysis")

    class Config:
        populate_by_name = True

    @property
    def is_experienced(self) -> bool:
        """Whether trader has enough history for reliable analysis."""
        return self.trade_count_for_analysis >= 20

    @property
    def is_profitable(self) -> bool:
        """Whether trader is net profitable."""
        return self.metrics.total_pnl_usd > 0


class LeaderboardEntry(BaseModel):
    """Entry in the trader leaderboard."""

    rank: int
    address: str
    display_name: Optional[str] = Field(None, alias="displayName")

    # Key metrics
    total_pnl_usd: Decimal = Field(..., alias="totalPnlUsd")
    roi_pct: float = Field(..., alias="roiPct")
    win_rate: float = Field(..., alias="winRate")
    total_volume_usd: Decimal = Field(..., alias="totalVolumeUsd")

    # Activity
    active_positions: int = Field(..., alias="activePositions")
    total_trades: int = Field(..., alias="totalTrades")

    # Social
    follower_count: int = Field(0, alias="followerCount")

    class Config:
        populate_by_name = True
