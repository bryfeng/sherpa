"""
Polymarket API Endpoints

REST API for Polymarket prediction market discovery, trading, and portfolio management.
"""

from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.polymarket import (
    PolymarketTradingService,
    get_polymarket_trading_service,
    PolymarketPortfolio,
    MarketAnalysis,
    TradeQuote,
    PolymarketCopyManager,
    get_polymarket_copy_manager,
    PolymarketCopyConfig,
    PolymarketCopyRelationship,
    PolymarketCopyExecution,
    PMSizingMode,
    PMCopyExecutionStatus,
)
from app.providers.polymarket import (
    Market,
    Event,
    OrderSide,
    MarketCategory,
)
from app.services.polymarket_analytics import (
    PolymarketLeaderboard,
    get_leaderboard,
    PolymarketTraderTracker,
    get_trader_tracker,
    PolymarketTraderProfile,
    LeaderboardEntry,
)

router = APIRouter(prefix="/polymarket", tags=["Polymarket"])


# =============================================================================
# Request/Response Models
# =============================================================================


class MarketResponse(BaseModel):
    """Market response."""

    market_id: str = Field(..., alias="marketId")
    question: str
    description: str
    outcomes: List[str]
    prices: Dict[str, float]
    volume_usd: float = Field(..., alias="volumeUsd")
    volume_24h_usd: float = Field(..., alias="volume24hUsd")
    liquidity_usd: float = Field(..., alias="liquidityUsd")
    end_date: Optional[datetime] = Field(None, alias="endDate")
    active: bool
    resolved: bool
    tags: List[str]

    class Config:
        populate_by_name = True


class MarketDetailResponse(BaseModel):
    """Detailed market response with orderbook."""

    market: MarketResponse
    orderbook_depth: Dict[str, Dict[str, Any]] = Field(..., alias="orderbookDepth")

    class Config:
        populate_by_name = True


class EventResponse(BaseModel):
    """Event response."""

    id: str
    slug: str
    title: str
    description: str
    market_count: int = Field(..., alias="marketCount")
    volume_usd: float = Field(..., alias="volumeUsd")
    active: bool

    class Config:
        populate_by_name = True


class PositionResponse(BaseModel):
    """Position response."""

    market_id: str = Field(..., alias="marketId")
    market_question: str = Field(..., alias="marketQuestion")
    outcome: str
    shares: float
    avg_price: float = Field(..., alias="avgPrice")
    current_price: float = Field(..., alias="currentPrice")
    value_usd: float = Field(..., alias="valueUsd")
    cost_basis_usd: float = Field(..., alias="costBasisUsd")
    pnl_usd: float = Field(..., alias="pnlUsd")
    pnl_pct: Optional[float] = Field(None, alias="pnlPct")

    class Config:
        populate_by_name = True


class PortfolioResponse(BaseModel):
    """Portfolio response."""

    address: str
    total_value_usd: float = Field(..., alias="totalValueUsd")
    total_cost_basis_usd: float = Field(..., alias="totalCostBasisUsd")
    total_pnl_usd: float = Field(..., alias="totalPnlUsd")
    total_pnl_pct: Optional[float] = Field(None, alias="totalPnlPct")
    open_positions: int = Field(..., alias="openPositions")
    winning_positions: int = Field(..., alias="winningPositions")
    losing_positions: int = Field(..., alias="losingPositions")
    positions: List[PositionResponse]

    class Config:
        populate_by_name = True


class QuoteRequest(BaseModel):
    """Request for a trade quote."""

    market_id: str = Field(..., alias="marketId")
    outcome: str
    side: str = Field(..., description="BUY or SELL")
    amount_usd: float = Field(..., alias="amountUsd")

    class Config:
        populate_by_name = True


class QuoteResponse(BaseModel):
    """Trade quote response."""

    market_id: str = Field(..., alias="marketId")
    outcome: str
    side: str
    amount_usd: float = Field(..., alias="amountUsd")
    shares: float
    avg_price: float = Field(..., alias="avgPrice")
    price_impact_pct: float = Field(..., alias="priceImpactPct")
    max_payout_usd: Optional[float] = Field(None, alias="maxPayoutUsd")
    potential_profit_usd: Optional[float] = Field(None, alias="potentialProfitUsd")
    potential_profit_pct: Optional[float] = Field(None, alias="potentialProfitPct")

    class Config:
        populate_by_name = True


class AnalysisResponse(BaseModel):
    """Market analysis response."""

    market_id: str = Field(..., alias="marketId")
    question: str
    current_yes_price: float = Field(..., alias="currentYesPrice")
    current_no_price: float = Field(..., alias="currentNoPrice")
    summary: str
    key_factors: List[str] = Field(..., alias="keyFactors")
    sentiment: str
    confidence: float
    volume_trend: str = Field(..., alias="volumeTrend")
    recommended_side: Optional[str] = Field(None, alias="recommendedSide")
    recommended_reason: Optional[str] = Field(None, alias="recommendedReason")
    analyzed_at: datetime = Field(..., alias="analyzedAt")

    class Config:
        populate_by_name = True


# =============================================================================
# Dependencies
# =============================================================================


def get_trading_service() -> PolymarketTradingService:
    """Get the Polymarket trading service."""
    return get_polymarket_trading_service()


# =============================================================================
# Helper Functions
# =============================================================================


def _market_to_response(market: Market) -> MarketResponse:
    """Convert Market to response model."""
    return MarketResponse(
        marketId=market.condition_id,
        question=market.question,
        description=market.description,
        outcomes=market.outcomes,
        prices={
            t.outcome: float(t.price)
            for t in market.tokens
        },
        volumeUsd=float(market.volume),
        volume24hUsd=float(market.volume_24h),
        liquidityUsd=float(market.liquidity),
        endDate=market.end_date,
        active=market.active,
        resolved=market.resolved,
        tags=market.tags,
    )


def _portfolio_to_response(portfolio: PolymarketPortfolio) -> PortfolioResponse:
    """Convert Portfolio to response model."""
    return PortfolioResponse(
        address=portfolio.address,
        totalValueUsd=float(portfolio.total_value),
        totalCostBasisUsd=float(portfolio.total_cost_basis),
        totalPnlUsd=float(portfolio.total_pnl),
        totalPnlPct=portfolio.total_pnl_pct,
        openPositions=portfolio.open_positions_count,
        winningPositions=portfolio.winning_positions,
        losingPositions=portfolio.losing_positions,
        positions=[
            PositionResponse(
                marketId=p.market_id,
                marketQuestion=p.market_question,
                outcome=p.outcome_name,
                shares=float(p.size),
                avgPrice=float(p.avg_price),
                currentPrice=float(p.current_price),
                valueUsd=float(p.current_value),
                costBasisUsd=float(p.cost_basis),
                pnlUsd=float(p.unrealized_pnl),
                pnlPct=p.unrealized_pnl_pct,
            )
            for p in portfolio.positions
        ],
    )


def _analysis_to_response(analysis: MarketAnalysis) -> AnalysisResponse:
    """Convert MarketAnalysis to response model."""
    return AnalysisResponse(
        marketId=analysis.market_id,
        question=analysis.question,
        currentYesPrice=float(analysis.current_yes_price),
        currentNoPrice=float(analysis.current_no_price),
        summary=analysis.summary,
        keyFactors=analysis.key_factors,
        sentiment=analysis.sentiment,
        confidence=analysis.confidence,
        volumeTrend=analysis.volume_trend,
        recommendedSide=analysis.recommended_side,
        recommendedReason=analysis.recommended_reason,
        analyzedAt=analysis.analyzed_at,
    )


def _quote_to_response(quote: TradeQuote) -> QuoteResponse:
    """Convert TradeQuote to response model."""
    return QuoteResponse(
        marketId=quote.market_id,
        outcome=quote.outcome_name,
        side=quote.side.value,
        amountUsd=float(quote.amount_usd),
        shares=float(quote.shares),
        avgPrice=float(quote.avg_price),
        priceImpactPct=quote.price_impact_pct,
        maxPayoutUsd=float(quote.max_payout) if quote.max_payout else None,
        potentialProfitUsd=float(quote.potential_profit) if quote.potential_profit else None,
        potentialProfitPct=quote.potential_profit_pct,
    )


# =============================================================================
# Market Discovery Endpoints
# =============================================================================


@router.get("/markets", response_model=List[MarketResponse])
async def get_markets(
    category: Optional[str] = Query(None, description="Filter by category"),
    query: Optional[str] = Query(None, description="Search query"),
    trending: bool = Query(False, description="Get trending markets"),
    closing_soon_hours: Optional[int] = Query(None, alias="closingSoonHours"),
    limit: int = Query(50, ge=1, le=200),
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """
    Get Polymarket prediction markets.

    Can filter by:
    - category: politics, crypto, sports, entertainment, science, economics
    - query: Search by question text
    - trending: Get markets by 24h volume
    - closingSoonHours: Markets closing within N hours
    """
    try:
        markets = await service.get_markets(
            category=category,
            query=query,
            trending=trending,
            closing_soon_hours=closing_soon_hours,
            limit=limit,
        )
        return [_market_to_response(m) for m in markets]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets/{market_id}", response_model=MarketDetailResponse)
async def get_market(
    market_id: str,
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """Get detailed market information including orderbook depth."""
    try:
        details = await service.get_market_details(market_id)
        if not details:
            raise HTTPException(status_code=404, detail="Market not found")

        market = details["market"]
        orderbooks = details.get("orderbooks", {})

        return MarketDetailResponse(
            market=_market_to_response(market),
            orderbookDepth={
                outcome: {
                    "bestBid": float(ob.best_bid) if ob.best_bid else None,
                    "bestAsk": float(ob.best_ask) if ob.best_ask else None,
                    "spread": float(ob.spread) if ob.spread else None,
                    "bidLevels": len(ob.bids),
                    "askLevels": len(ob.asks),
                }
                for outcome, ob in orderbooks.items()
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending", response_model=List[MarketResponse])
async def get_trending_markets(
    limit: int = Query(20, ge=1, le=100),
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """Get trending markets by 24h volume."""
    try:
        markets = await service.get_markets(trending=True, limit=limit)
        return [_market_to_response(m) for m in markets]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/closing-soon", response_model=List[MarketResponse])
async def get_closing_soon(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(20, ge=1, le=100),
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """Get markets closing within specified hours."""
    try:
        markets = await service.get_markets(closing_soon_hours=hours, limit=limit)
        return [_market_to_response(m) for m in markets]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories():
    """Get available market categories."""
    return {
        "categories": [
            {"id": "politics", "name": "Politics", "description": "Elections, policy, government"},
            {"id": "crypto", "name": "Crypto", "description": "Bitcoin, Ethereum, DeFi"},
            {"id": "sports", "name": "Sports", "description": "NFL, NBA, Soccer, Olympics"},
            {"id": "entertainment", "name": "Entertainment", "description": "Movies, TV, Oscars"},
            {"id": "science", "name": "Science", "description": "Space, technology, research"},
            {"id": "economics", "name": "Economics", "description": "Markets, Fed, economic data"},
            {"id": "weather", "name": "Weather", "description": "Climate, natural events"},
            {"id": "current_events", "name": "Current Events", "description": "News, world events"},
        ]
    }


# =============================================================================
# Portfolio Endpoints
# =============================================================================


@router.get("/portfolio/{address}", response_model=PortfolioResponse)
async def get_portfolio(
    address: str,
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """Get user's Polymarket portfolio with positions and P&L."""
    try:
        portfolio = await service.get_portfolio(address)
        return _portfolio_to_response(portfolio)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trading Endpoints
# =============================================================================


@router.post("/quote", response_model=QuoteResponse)
async def get_quote(
    request: QuoteRequest,
    address: str = Query(..., description="User's wallet address"),
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """
    Get a quote for buying or selling shares.

    For BUY: amount_usd is the USDC to spend
    For SELL: amount_usd is the number of shares to sell

    Returns estimated shares, price, and potential profit.
    Does NOT execute the trade.
    """
    try:
        side = OrderSide.BUY if request.side.upper() == "BUY" else OrderSide.SELL

        if side == OrderSide.BUY:
            quote = await service.get_buy_quote(
                market_id=request.market_id,
                outcome=request.outcome,
                amount_usd=Decimal(str(request.amount_usd)),
            )
        else:
            quote = await service.get_sell_quote(
                market_id=request.market_id,
                outcome=request.outcome,
                shares=Decimal(str(request.amount_usd)),
                address=address,
            )

        if not quote:
            raise HTTPException(status_code=400, detail="Could not generate quote")

        return _quote_to_response(quote)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analysis Endpoints
# =============================================================================


@router.get("/analyze/{market_id}", response_model=AnalysisResponse)
async def analyze_market(
    market_id: str,
    service: PolymarketTradingService = Depends(get_trading_service),
):
    """Get AI-powered analysis of a prediction market."""
    try:
        analysis = await service.analyze_market(market_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Market not found")

        return _analysis_to_response(analysis)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trader Analytics Response Models
# =============================================================================


class TraderMetricsResponse(BaseModel):
    """Trader performance metrics."""

    total_volume_usd: float = Field(..., alias="totalVolumeUsd")
    volume_30d_usd: Optional[float] = Field(None, alias="volume30dUsd")
    total_pnl_usd: float = Field(..., alias="totalPnlUsd")
    realized_pnl_usd: Optional[float] = Field(None, alias="realizedPnlUsd")
    unrealized_pnl_usd: Optional[float] = Field(None, alias="unrealizedPnlUsd")
    roi_pct: float = Field(..., alias="roiPct")
    total_trades: int = Field(..., alias="totalTrades")
    winning_trades: int = Field(..., alias="winningTrades")
    losing_trades: int = Field(..., alias="losingTrades")
    win_rate: float = Field(..., alias="winRate")
    brier_score: Optional[float] = Field(None, alias="brierScore")
    current_streak: Optional[int] = Field(None, alias="currentStreak")
    best_streak: Optional[int] = Field(None, alias="bestStreak")
    worst_streak: Optional[int] = Field(None, alias="worstStreak")

    class Config:
        populate_by_name = True


class TraderProfileResponse(BaseModel):
    """Full trader profile."""

    address: str
    metrics: TraderMetricsResponse
    active_positions: int = Field(..., alias="activePositions")
    total_exposure_usd: float = Field(..., alias="totalExposureUsd")
    avg_position_size_usd: float = Field(..., alias="avgPositionSizeUsd")
    max_position_size_usd: float = Field(..., alias="maxPositionSizeUsd")
    max_single_bet_pct: float = Field(..., alias="maxSingleBetPct")
    preferred_categories: List[str] = Field(..., alias="preferredCategories")
    avg_hold_time_days: float = Field(..., alias="avgHoldTimeDays")
    trades_per_week: float = Field(..., alias="tradesPerWeek")
    diversification_score: float = Field(..., alias="diversificationScore")
    risk_score: float = Field(..., alias="riskScore")
    is_experienced: bool = Field(..., alias="isExperienced")
    follower_count: int = Field(..., alias="followerCount")
    first_trade_at: Optional[datetime] = Field(None, alias="firstTradeAt")
    last_trade_at: Optional[datetime] = Field(None, alias="lastTradeAt")
    data_quality_score: float = Field(..., alias="dataQualityScore")

    class Config:
        populate_by_name = True


class LeaderboardEntryResponse(BaseModel):
    """Leaderboard entry."""

    rank: int
    address: str
    total_pnl_usd: float = Field(..., alias="totalPnlUsd")
    roi_pct: float = Field(..., alias="roiPct")
    win_rate: float = Field(..., alias="winRate")
    total_volume_usd: float = Field(..., alias="totalVolumeUsd")
    active_positions: int = Field(..., alias="activePositions")
    total_trades: int = Field(..., alias="totalTrades")
    follower_count: int = Field(..., alias="followerCount")

    class Config:
        populate_by_name = True


# =============================================================================
# Copy Trading Response Models
# =============================================================================


class CopyConfigRequest(BaseModel):
    """Request to start copy trading."""

    leader_address: str = Field(..., alias="leaderAddress")
    sizing_mode: str = Field("percentage", alias="sizingMode")
    size_value: float = Field(100.0, alias="sizeValue")
    min_position_usd: Optional[float] = Field(None, alias="minPositionUsd")
    max_position_usd: Optional[float] = Field(None, alias="maxPositionUsd")
    category_whitelist: Optional[List[str]] = Field(None, alias="categoryWhitelist")
    category_blacklist: Optional[List[str]] = Field(None, alias="categoryBlacklist")
    copy_new_positions: bool = Field(True, alias="copyNewPositions")
    copy_exits: bool = Field(True, alias="copyExits")
    sync_existing: bool = Field(False, alias="syncExisting")
    max_positions: Optional[int] = Field(None, alias="maxPositions")
    max_exposure_usd: Optional[float] = Field(None, alias="maxExposureUsd")
    max_slippage_pct: float = Field(2.0, alias="maxSlippagePct")
    approval_timeout_minutes: int = Field(60, alias="approvalTimeoutMinutes")

    class Config:
        populate_by_name = True


class CopyRelationshipResponse(BaseModel):
    """Copy relationship response."""

    id: str
    user_id: str = Field(..., alias="userId")
    follower_address: str = Field(..., alias="followerAddress")
    leader_address: str = Field(..., alias="leaderAddress")
    config: Dict[str, Any]
    status: str
    total_copied_trades: int = Field(..., alias="totalCopiedTrades")
    total_copied_volume_usd: float = Field(..., alias="totalCopiedVolumeUsd")
    total_pnl_usd: float = Field(..., alias="totalPnlUsd")
    skipped_trades: int = Field(..., alias="skippedTrades")
    created_at: datetime = Field(..., alias="createdAt")
    last_copy_at: Optional[datetime] = Field(None, alias="lastCopyAt")

    class Config:
        populate_by_name = True


class CopyExecutionResponse(BaseModel):
    """Copy execution response."""

    id: str
    relationship_id: str = Field(..., alias="relationshipId")
    leader_trade_id: str = Field(..., alias="leaderTradeId")
    market_id: str = Field(..., alias="marketId")
    market_question: str = Field(..., alias="marketQuestion")
    outcome: str
    side: str
    leader_shares: float = Field(..., alias="leaderShares")
    leader_price: float = Field(..., alias="leaderPrice")
    follower_shares: Optional[float] = Field(None, alias="followerShares")
    follower_price: Optional[float] = Field(None, alias="followerPrice")
    follower_amount_usd: Optional[float] = Field(None, alias="followerAmountUsd")
    status: str
    skip_reason: Optional[str] = Field(None, alias="skipReason")
    created_at: datetime = Field(..., alias="createdAt")
    approved_at: Optional[datetime] = Field(None, alias="approvedAt")
    executed_at: Optional[datetime] = Field(None, alias="executedAt")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")

    class Config:
        populate_by_name = True


# =============================================================================
# Dependencies for Copy Trading
# =============================================================================


def get_copy_manager() -> PolymarketCopyManager:
    """Get the Polymarket copy manager."""
    return get_polymarket_copy_manager()


def get_tracker() -> PolymarketTraderTracker:
    """Get the trader tracker."""
    return get_trader_tracker()


def get_lb() -> PolymarketLeaderboard:
    """Get the leaderboard service."""
    return get_leaderboard()


# =============================================================================
# Helper Functions for Copy Trading
# =============================================================================


def _profile_to_response(profile: PolymarketTraderProfile) -> TraderProfileResponse:
    """Convert trader profile to response."""
    return TraderProfileResponse(
        address=profile.address,
        metrics=TraderMetricsResponse(
            totalVolumeUsd=float(profile.metrics.total_volume_usd),
            volume30dUsd=float(profile.metrics.volume_30d_usd) if profile.metrics.volume_30d_usd else None,
            totalPnlUsd=float(profile.metrics.total_pnl_usd),
            realizedPnlUsd=float(profile.metrics.realized_pnl_usd) if profile.metrics.realized_pnl_usd else None,
            unrealizedPnlUsd=float(profile.metrics.unrealized_pnl_usd) if profile.metrics.unrealized_pnl_usd else None,
            roiPct=profile.metrics.roi_pct,
            totalTrades=profile.metrics.total_trades,
            winningTrades=profile.metrics.winning_trades,
            losingTrades=profile.metrics.losing_trades,
            winRate=profile.metrics.win_rate,
            brierScore=profile.metrics.brier_score,
            currentStreak=profile.metrics.current_streak,
            bestStreak=profile.metrics.best_streak,
            worstStreak=profile.metrics.worst_streak,
        ),
        activePositions=profile.active_positions,
        totalExposureUsd=float(profile.total_exposure_usd),
        avgPositionSizeUsd=float(profile.avg_position_size_usd),
        maxPositionSizeUsd=float(profile.max_position_size_usd),
        maxSingleBetPct=profile.max_single_bet_pct,
        preferredCategories=profile.preferred_categories,
        avgHoldTimeDays=profile.avg_hold_time_days,
        tradesPerWeek=profile.trades_per_week,
        diversificationScore=profile.diversification_score,
        riskScore=profile.risk_score,
        isExperienced=profile.is_experienced,
        followerCount=profile.follower_count,
        firstTradeAt=profile.first_trade_at,
        lastTradeAt=profile.last_trade_at,
        dataQualityScore=profile.data_quality_score,
    )


def _leaderboard_entry_to_response(entry: LeaderboardEntry) -> LeaderboardEntryResponse:
    """Convert leaderboard entry to response."""
    return LeaderboardEntryResponse(
        rank=entry.rank,
        address=entry.address,
        totalPnlUsd=float(entry.total_pnl_usd),
        roiPct=entry.roi_pct,
        winRate=entry.win_rate,
        totalVolumeUsd=float(entry.total_volume_usd),
        activePositions=entry.active_positions,
        totalTrades=entry.total_trades,
        followerCount=entry.follower_count,
    )


def _relationship_to_response(rel: PolymarketCopyRelationship) -> CopyRelationshipResponse:
    """Convert copy relationship to response."""
    return CopyRelationshipResponse(
        id=rel.id,
        userId=rel.user_id,
        followerAddress=rel.follower_address,
        leaderAddress=rel.leader_address,
        config=rel.config.model_dump() if rel.config else {},
        status=rel.status,
        totalCopiedTrades=rel.total_copied_trades,
        totalCopiedVolumeUsd=float(rel.total_copied_volume_usd),
        totalPnlUsd=float(rel.total_pnl_usd),
        skippedTrades=rel.skipped_trades,
        createdAt=rel.created_at,
        lastCopyAt=rel.last_copy_at,
    )


def _execution_to_response(ex: PolymarketCopyExecution) -> CopyExecutionResponse:
    """Convert copy execution to response."""
    return CopyExecutionResponse(
        id=ex.id,
        relationshipId=ex.relationship_id,
        leaderTradeId=ex.leader_trade_id,
        marketId=ex.market_id,
        marketQuestion=ex.market_question,
        outcome=ex.outcome,
        side=ex.side,
        leaderShares=float(ex.leader_shares),
        leaderPrice=float(ex.leader_price),
        followerShares=float(ex.follower_shares) if ex.follower_shares else None,
        followerPrice=float(ex.follower_price) if ex.follower_price else None,
        followerAmountUsd=float(ex.follower_amount_usd) if ex.follower_amount_usd else None,
        status=ex.status.value,
        skipReason=ex.skip_reason,
        createdAt=ex.created_at,
        approvedAt=ex.approved_at,
        executedAt=ex.executed_at,
        expiresAt=ex.expires_at,
    )


# =============================================================================
# Trader Analytics Endpoints
# =============================================================================


@router.get("/traders/leaderboard", response_model=List[LeaderboardEntryResponse])
async def get_trader_leaderboard(
    sort_by: str = Query("total_pnl", description="Sort by: total_pnl, roi, win_rate, volume"),
    limit: int = Query(50, ge=1, le=100),
    min_trades: int = Query(10, ge=1, alias="minTrades"),
    leaderboard: PolymarketLeaderboard = Depends(get_lb),
):
    """Get the Polymarket trader leaderboard."""
    try:
        entries = await leaderboard.get_leaderboard(
            sort_by=sort_by,
            limit=limit,
            min_trades=min_trades,
        )
        return [_leaderboard_entry_to_response(e) for e in entries]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traders/{address}", response_model=TraderProfileResponse)
async def get_trader_profile(
    address: str,
    refresh: bool = Query(False, description="Force refresh from API"),
    tracker: PolymarketTraderTracker = Depends(get_tracker),
):
    """Get a trader's full profile with performance metrics."""
    try:
        profile = await tracker.get_trader_profile(address, refresh=refresh)
        return _profile_to_response(profile)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traders/search/{query}", response_model=List[LeaderboardEntryResponse])
async def search_traders(
    query: str,
    limit: int = Query(20, ge=1, le=50),
    leaderboard: PolymarketLeaderboard = Depends(get_lb),
):
    """Search for traders by address."""
    try:
        entries = await leaderboard.search_traders(query, limit=limit)
        return [_leaderboard_entry_to_response(e) for e in entries]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Copy Trading Endpoints
# =============================================================================


@router.post("/copy/relationships", response_model=CopyRelationshipResponse)
async def start_copying(
    request: CopyConfigRequest,
    user_id: str = Query(..., alias="userId"),
    follower_address: str = Query(..., alias="followerAddress"),
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Start copy trading a Polymarket trader."""
    try:
        config = PolymarketCopyConfig(
            leader_address=request.leader_address,
            sizing_mode=PMSizingMode(request.sizing_mode),
            size_value=request.size_value,
            min_position_usd=Decimal(str(request.min_position_usd)) if request.min_position_usd else None,
            max_position_usd=Decimal(str(request.max_position_usd)) if request.max_position_usd else None,
            category_whitelist=request.category_whitelist,
            category_blacklist=request.category_blacklist,
            copy_new_positions=request.copy_new_positions,
            copy_exits=request.copy_exits,
            sync_existing=request.sync_existing,
            max_positions=request.max_positions,
            max_exposure_usd=Decimal(str(request.max_exposure_usd)) if request.max_exposure_usd else None,
            max_slippage_pct=request.max_slippage_pct,
            approval_timeout_minutes=request.approval_timeout_minutes,
        )

        relationship = await manager.start_copying(
            user_id=user_id,
            follower_address=follower_address,
            config=config,
        )
        return _relationship_to_response(relationship)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/copy/relationships/user/{user_id}", response_model=List[CopyRelationshipResponse])
async def list_copy_relationships(
    user_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """List all copy trading relationships for a user."""
    try:
        relationships = await manager.get_relationships(user_id)
        return [_relationship_to_response(r) for r in relationships]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/copy/relationships/{relationship_id}")
async def stop_copying(
    relationship_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Stop copy trading."""
    try:
        success = await manager.stop_copying(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        return {"status": "stopped", "relationshipId": relationship_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy/relationships/{relationship_id}/pause")
async def pause_copying(
    relationship_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Pause copy trading (can be resumed later)."""
    try:
        success = await manager.pause_copying(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        return {"status": "paused", "relationshipId": relationship_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy/relationships/{relationship_id}/resume")
async def resume_copying(
    relationship_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Resume paused copy trading."""
    try:
        success = await manager.resume_copying(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        return {"status": "active", "relationshipId": relationship_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Copy Execution Endpoints
# =============================================================================


@router.get("/copy/executions/pending/{user_id}", response_model=List[CopyExecutionResponse])
async def get_pending_executions(
    user_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Get pending copy executions that need user approval."""
    try:
        executions = await manager.get_pending_approvals(user_id)
        return [_execution_to_response(e) for e in executions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy/executions/{execution_id}/approve", response_model=CopyExecutionResponse)
async def approve_execution(
    execution_id: str,
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Approve a copy execution (generates quote for user to sign)."""
    try:
        execution = await manager.approve_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found or expired")
        return _execution_to_response(execution)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy/executions/{execution_id}/confirm", response_model=CopyExecutionResponse)
async def confirm_execution(
    execution_id: str,
    tx_hash: str = Query(..., alias="txHash"),
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """
    Confirm that a copy trade was executed.

    Called by frontend after user signs and submits the trade.
    """
    try:
        execution = await manager.confirm_execution(execution_id, tx_hash)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return _execution_to_response(execution)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy/executions/{execution_id}/reject")
async def reject_execution(
    execution_id: str,
    reason: Optional[str] = Query(None),
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Reject a pending copy execution."""
    try:
        success = await manager.reject_execution(execution_id, reason)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found")
        return {"status": "rejected", "executionId": execution_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/copy/executions/history/{user_id}", response_model=List[CopyExecutionResponse])
async def get_execution_history(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter by status"),
    manager: PolymarketCopyManager = Depends(get_copy_manager),
):
    """Get copy execution history for a user."""
    try:
        executions = await manager.get_execution_history(
            user_id,
            limit=limit,
            status=PMCopyExecutionStatus(status) if status else None,
        )
        return [_execution_to_response(e) for e in executions]

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
