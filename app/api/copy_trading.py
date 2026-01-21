"""
Copy Trading API Endpoints

REST API for managing wallet copy trading relationships and leader analytics.
"""

from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.copy_trading import (
    CopyConfig,
    CopyRelationship,
    CopyExecution,
    CopyExecutionStatus,
    CopyTradingManager,
    CopyExecutor,
    LeaderProfile,
    SizingMode,
    TradeSignal,
)
from app.core.copy_trading.analytics import LeaderAnalytics
from app.core.copy_trading.models import SkipReason, CopyTradingStats
from app.db.convex_client import get_convex_client

router = APIRouter(prefix="/copy-trading", tags=["Copy Trading"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateCopyConfigRequest(BaseModel):
    """Request to create a copy trading configuration."""

    user_id: str = Field(..., alias="userId", description="User ID")
    follower_address: str = Field(..., alias="followerAddress", description="Follower wallet address")
    follower_chain: str = Field(..., alias="followerChain", description="Follower wallet chain")

    # Leader info
    leader_address: str = Field(..., alias="leaderAddress", description="Leader wallet address")
    leader_chain: str = Field(..., alias="leaderChain", description="Leader wallet chain")
    leader_label: Optional[str] = Field(None, alias="leaderLabel", description="Label for the leader")

    # Sizing
    sizing_mode: SizingMode = Field(SizingMode.PERCENTAGE, alias="sizingMode")
    size_value: float = Field(5.0, alias="sizeValue", description="Size value based on mode")

    # Filters
    min_trade_usd: float = Field(10.0, alias="minTradeUsd")
    max_trade_usd: Optional[float] = Field(None, alias="maxTradeUsd")
    token_whitelist: Optional[List[str]] = Field(None, alias="tokenWhitelist")
    token_blacklist: Optional[List[str]] = Field(None, alias="tokenBlacklist")
    allowed_actions: List[str] = Field(default=["swap"], alias="allowedActions")

    # Timing
    delay_seconds: int = Field(0, alias="delaySeconds")
    max_delay_seconds: int = Field(300, alias="maxDelaySeconds")

    # Risk controls
    max_slippage_bps: int = Field(100, alias="maxSlippageBps")
    max_daily_trades: int = Field(20, alias="maxDailyTrades")
    max_daily_volume_usd: float = Field(10000, alias="maxDailyVolumeUsd")

    class Config:
        populate_by_name = True


class UpdateCopyConfigRequest(BaseModel):
    """Request to update copy trading configuration."""

    sizing_mode: Optional[SizingMode] = Field(None, alias="sizingMode")
    size_value: Optional[float] = Field(None, alias="sizeValue")
    min_trade_usd: Optional[float] = Field(None, alias="minTradeUsd")
    max_trade_usd: Optional[float] = Field(None, alias="maxTradeUsd")
    token_whitelist: Optional[List[str]] = Field(None, alias="tokenWhitelist")
    token_blacklist: Optional[List[str]] = Field(None, alias="tokenBlacklist")
    allowed_actions: Optional[List[str]] = Field(None, alias="allowedActions")
    delay_seconds: Optional[int] = Field(None, alias="delaySeconds")
    max_delay_seconds: Optional[int] = Field(None, alias="maxDelaySeconds")
    max_slippage_bps: Optional[int] = Field(None, alias="maxSlippageBps")
    max_daily_trades: Optional[int] = Field(None, alias="maxDailyTrades")
    max_daily_volume_usd: Optional[float] = Field(None, alias="maxDailyVolumeUsd")

    class Config:
        populate_by_name = True


class ActivateCopyRequest(BaseModel):
    """Request to activate copy trading with a session key."""

    session_key_id: str = Field(..., alias="sessionKeyId")

    class Config:
        populate_by_name = True


class CopyRelationshipResponse(BaseModel):
    """Copy relationship response."""

    id: str
    user_id: str = Field(..., alias="userId")
    follower_address: str = Field(..., alias="followerAddress")
    follower_chain: str = Field(..., alias="followerChain")
    leader_address: str = Field(..., alias="leaderAddress")
    leader_chain: str = Field(..., alias="leaderChain")
    leader_label: Optional[str] = Field(None, alias="leaderLabel")

    # Config summary
    sizing_mode: str = Field(..., alias="sizingMode")
    size_value: float = Field(..., alias="sizeValue")
    max_slippage_bps: int = Field(..., alias="maxSlippageBps")

    # Status
    is_active: bool = Field(..., alias="isActive")
    is_paused: bool = Field(..., alias="isPaused")
    pause_reason: Optional[str] = Field(None, alias="pauseReason")

    # Stats
    daily_trade_count: int = Field(..., alias="dailyTradeCount")
    daily_volume_usd: float = Field(..., alias="dailyVolumeUsd")
    total_trades: int = Field(..., alias="totalTrades")
    successful_trades: int = Field(..., alias="successfulTrades")
    failed_trades: int = Field(..., alias="failedTrades")
    skipped_trades: int = Field(..., alias="skippedTrades")
    total_volume_usd: float = Field(..., alias="totalVolumeUsd")
    total_pnl_usd: Optional[float] = Field(None, alias="totalPnlUsd")

    # Timestamps
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    last_copy_at: Optional[datetime] = Field(None, alias="lastCopyAt")

    class Config:
        populate_by_name = True


class LeaderProfileResponse(BaseModel):
    """Leader profile response."""

    address: str
    chain: str
    label: Optional[str] = None
    notes: Optional[str] = None

    # Performance
    total_trades: int = Field(..., alias="totalTrades")
    win_rate: Optional[float] = Field(None, alias="winRate")
    avg_trade_pnl_pct: Optional[float] = Field(None, alias="avgTradePnlPct")
    total_pnl_usd: Optional[float] = Field(None, alias="totalPnlUsd")
    sharpe_ratio: Optional[float] = Field(None, alias="sharpeRatio")
    max_drawdown_pct: Optional[float] = Field(None, alias="maxDrawdownPct")

    # Activity
    avg_trades_per_day: Optional[float] = Field(None, alias="avgTradesPerDay")
    avg_hold_time_hours: Optional[float] = Field(None, alias="avgHoldTimeHours")
    most_traded_tokens: List[str] = Field(default=[], alias="mostTradedTokens")
    preferred_sectors: List[str] = Field(default=[], alias="preferredSectors")

    # Risk
    avg_position_size_pct: Optional[float] = Field(None, alias="avgPositionSizePct")
    max_position_size_pct: Optional[float] = Field(None, alias="maxPositionSizePct")
    uses_leverage: bool = Field(False, alias="usesLeverage")
    risk_score: float = Field(..., alias="riskScore")

    # Social
    follower_count: int = Field(..., alias="followerCount")
    total_copied_volume_usd: float = Field(..., alias="totalCopiedVolumeUsd")

    # Status
    is_active: bool = Field(..., alias="isActive")
    data_quality_score: float = Field(..., alias="dataQualityScore")
    first_seen_at: datetime = Field(..., alias="firstSeenAt")
    last_active_at: datetime = Field(..., alias="lastActiveAt")
    last_analyzed_at: Optional[datetime] = Field(None, alias="lastAnalyzedAt")

    class Config:
        populate_by_name = True


class CopyExecutionResponse(BaseModel):
    """Copy execution response."""

    id: str
    relationship_id: str = Field(..., alias="relationshipId")
    status: str
    skip_reason: Optional[str] = Field(None, alias="skipReason")

    # Signal info
    leader_address: str = Field(..., alias="leaderAddress")
    leader_tx_hash: str = Field(..., alias="leaderTxHash")
    action: str
    token_in_symbol: Optional[str] = Field(None, alias="tokenInSymbol")
    token_out_symbol: Optional[str] = Field(None, alias="tokenOutSymbol")
    signal_value_usd: Optional[float] = Field(None, alias="signalValueUsd")

    # Execution
    calculated_size_usd: Optional[float] = Field(None, alias="calculatedSizeUsd")
    actual_size_usd: Optional[float] = Field(None, alias="actualSizeUsd")
    tx_hash: Optional[str] = Field(None, alias="txHash")
    slippage_bps: Optional[int] = Field(None, alias="slippageBps")
    gas_cost_usd: Optional[float] = Field(None, alias="gasCostUsd")
    error_message: Optional[str] = Field(None, alias="errorMessage")

    # Timing
    signal_received_at: datetime = Field(..., alias="signalReceivedAt")
    execution_started_at: Optional[datetime] = Field(None, alias="executionStartedAt")
    execution_completed_at: Optional[datetime] = Field(None, alias="executionCompletedAt")
    execution_delay_seconds: Optional[float] = Field(None, alias="executionDelaySeconds")

    class Config:
        populate_by_name = True


class CopyTradingStatsResponse(BaseModel):
    """Aggregated copy trading stats."""

    user_id: str = Field(..., alias="userId")
    active_relationships: int = Field(..., alias="activeRelationships")
    total_relationships: int = Field(..., alias="totalRelationships")
    total_copy_trades: int = Field(..., alias="totalCopyTrades")
    successful_trades: int = Field(..., alias="successfulTrades")
    failed_trades: int = Field(..., alias="failedTrades")
    skipped_trades: int = Field(..., alias="skippedTrades")
    total_volume_usd: float = Field(..., alias="totalVolumeUsd")
    today_volume_usd: float = Field(..., alias="todayVolumeUsd")
    total_pnl_usd: Optional[float] = Field(None, alias="totalPnlUsd")
    last_copy_at: Optional[datetime] = Field(None, alias="lastCopyAt")

    class Config:
        populate_by_name = True


class EstimateExecutionRequest(BaseModel):
    """Request to estimate a copy trade execution."""

    leader_address: str = Field(..., alias="leaderAddress")
    leader_chain: str = Field(..., alias="leaderChain")
    token_in_address: str = Field(..., alias="tokenInAddress")
    token_out_address: str = Field(..., alias="tokenOutAddress")
    value_usd: float = Field(..., alias="valueUsd")
    follower_chain: str = Field(..., alias="followerChain")
    size_usd: float = Field(..., alias="sizeUsd")
    max_slippage_bps: int = Field(100, alias="maxSlippageBps")

    class Config:
        populate_by_name = True


# =============================================================================
# Dependencies
# =============================================================================


_manager: Optional[CopyTradingManager] = None
_analytics: Optional[LeaderAnalytics] = None


def get_copy_trading_manager() -> CopyTradingManager:
    """Get or create the copy trading manager."""
    global _manager
    if _manager is None:
        convex = get_convex_client()
        # Use executor with real providers (Jupiter + Relay)
        executor = CopyExecutor.create_with_providers()
        _manager = CopyTradingManager(
            convex_client=convex,
            executor=executor,
        )
    return _manager


def get_leader_analytics() -> LeaderAnalytics:
    """Get or create the leader analytics service."""
    global _analytics
    if _analytics is None:
        convex = get_convex_client()
        _analytics = LeaderAnalytics(convex_client=convex)
    return _analytics


# =============================================================================
# Helper Functions
# =============================================================================


def _relationship_to_response(rel: CopyRelationship) -> CopyRelationshipResponse:
    """Convert CopyRelationship to response model."""
    return CopyRelationshipResponse(
        id=rel.id,
        userId=rel.user_id,
        followerAddress=rel.follower_address,
        followerChain=rel.follower_chain,
        leaderAddress=rel.config.leader_address,
        leaderChain=rel.config.leader_chain,
        leaderLabel=rel.config.leader_label,
        sizingMode=rel.config.sizing_mode.value,
        sizeValue=float(rel.config.size_value),
        maxSlippageBps=rel.config.max_slippage_bps,
        isActive=rel.is_active,
        isPaused=rel.is_paused,
        pauseReason=rel.pause_reason,
        dailyTradeCount=rel.daily_trade_count,
        dailyVolumeUsd=float(rel.daily_volume_usd),
        totalTrades=rel.total_trades,
        successfulTrades=rel.successful_trades,
        failedTrades=rel.failed_trades,
        skippedTrades=rel.skipped_trades,
        totalVolumeUsd=float(rel.total_volume_usd),
        totalPnlUsd=float(rel.total_pnl_usd) if rel.total_pnl_usd else None,
        createdAt=rel.created_at,
        updatedAt=rel.updated_at,
        lastCopyAt=rel.last_copy_at,
    )


def _profile_to_response(profile: LeaderProfile) -> LeaderProfileResponse:
    """Convert LeaderProfile to response model."""
    return LeaderProfileResponse(
        address=profile.address,
        chain=profile.chain,
        label=profile.label,
        notes=profile.notes,
        totalTrades=profile.total_trades,
        winRate=profile.win_rate,
        avgTradePnlPct=profile.avg_trade_pnl_pct,
        totalPnlUsd=float(profile.total_pnl_usd) if profile.total_pnl_usd else None,
        sharpeRatio=profile.sharpe_ratio,
        maxDrawdownPct=profile.max_drawdown_pct,
        avgTradesPerDay=profile.avg_trades_per_day,
        avgHoldTimeHours=profile.avg_hold_time_hours,
        mostTradedTokens=profile.most_traded_tokens,
        preferredSectors=profile.preferred_sectors,
        avgPositionSizePct=profile.avg_position_size_pct,
        maxPositionSizePct=profile.max_position_size_pct,
        usesLeverage=profile.uses_leverage,
        riskScore=profile.risk_score,
        followerCount=profile.follower_count,
        totalCopiedVolumeUsd=float(profile.total_copied_volume_usd),
        isActive=profile.is_active,
        dataQualityScore=profile.data_quality_score,
        firstSeenAt=profile.first_seen_at,
        lastActiveAt=profile.last_active_at,
        lastAnalyzedAt=profile.last_analyzed_at,
    )


def _execution_to_response(execution: CopyExecution) -> CopyExecutionResponse:
    """Convert CopyExecution to response model."""
    return CopyExecutionResponse(
        id=execution.id,
        relationshipId=execution.relationship_id,
        status=execution.status.value,
        skipReason=execution.skip_reason.value if execution.skip_reason else None,
        leaderAddress=execution.signal.leader_address,
        leaderTxHash=execution.signal.tx_hash,
        action=execution.signal.action,
        tokenInSymbol=execution.signal.token_in_symbol,
        tokenOutSymbol=execution.signal.token_out_symbol,
        signalValueUsd=float(execution.signal.value_usd) if execution.signal.value_usd else None,
        calculatedSizeUsd=float(execution.calculated_size_usd) if execution.calculated_size_usd else None,
        actualSizeUsd=float(execution.actual_size_usd) if execution.actual_size_usd else None,
        txHash=execution.tx_hash,
        slippageBps=execution.slippage_bps,
        gasCostUsd=float(execution.gas_cost_usd) if execution.gas_cost_usd else None,
        errorMessage=execution.error_message,
        signalReceivedAt=execution.signal_received_at,
        executionStartedAt=execution.execution_started_at,
        executionCompletedAt=execution.execution_completed_at,
        executionDelaySeconds=execution.execution_delay_seconds,
    )


def _stats_to_response(stats: CopyTradingStats) -> CopyTradingStatsResponse:
    """Convert CopyTradingStats to response model."""
    return CopyTradingStatsResponse(
        userId=stats.user_id,
        activeRelationships=stats.active_relationships,
        totalRelationships=stats.total_relationships,
        totalCopyTrades=stats.total_copy_trades,
        successfulTrades=stats.successful_trades,
        failedTrades=stats.failed_trades,
        skippedTrades=stats.skipped_trades,
        totalVolumeUsd=float(stats.total_volume_usd),
        todayVolumeUsd=float(stats.today_volume_usd),
        totalPnlUsd=float(stats.total_pnl_usd) if stats.total_pnl_usd else None,
        lastCopyAt=stats.last_copy_at,
    )


# =============================================================================
# Copy Relationship Endpoints
# =============================================================================


@router.post("/relationships", response_model=CopyRelationshipResponse)
async def create_copy_relationship(
    request: CreateCopyConfigRequest,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Create a new copy trading relationship."""
    try:
        config = CopyConfig(
            leader_address=request.leader_address,
            leader_chain=request.leader_chain,
            leader_label=request.leader_label,
            sizing_mode=request.sizing_mode,
            size_value=Decimal(str(request.size_value)),
            min_trade_usd=Decimal(str(request.min_trade_usd)),
            max_trade_usd=Decimal(str(request.max_trade_usd)) if request.max_trade_usd else None,
            token_whitelist=request.token_whitelist,
            token_blacklist=request.token_blacklist,
            allowed_actions=request.allowed_actions,
            delay_seconds=request.delay_seconds,
            max_delay_seconds=request.max_delay_seconds,
            max_slippage_bps=request.max_slippage_bps,
            max_daily_trades=request.max_daily_trades,
            max_daily_volume_usd=Decimal(str(request.max_daily_volume_usd)),
        )

        relationship = await manager.start_copying(
            user_id=request.user_id,
            follower_address=request.follower_address,
            follower_chain=request.follower_chain,
            config=config,
        )

        return _relationship_to_response(relationship)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/{relationship_id}", response_model=CopyRelationshipResponse)
async def get_copy_relationship(
    relationship_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Get a copy trading relationship by ID."""
    relationship = manager.get_relationship(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")
    return _relationship_to_response(relationship)


@router.get("/relationships/user/{user_id}", response_model=List[CopyRelationshipResponse])
async def list_user_relationships(
    user_id: str,
    active_only: bool = Query(False, alias="activeOnly"),
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """List all copy trading relationships for a user."""
    relationships = await manager.get_relationships_for_user(user_id)
    if active_only:
        relationships = [r for r in relationships if r.is_active and not r.is_paused]
    return [_relationship_to_response(r) for r in relationships]


@router.get("/relationships/follower/{follower_address}", response_model=List[CopyRelationshipResponse])
async def list_follower_relationships(
    follower_address: str,
    follower_chain: Optional[str] = None,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """List all copy trading relationships for a follower wallet."""
    relationships = await manager.get_relationships_for_follower(
        follower_address=follower_address,
        follower_chain=follower_chain,
    )
    return [_relationship_to_response(r) for r in relationships]


@router.post("/relationships/{relationship_id}/activate", response_model=CopyRelationshipResponse)
async def activate_copy_relationship(
    relationship_id: str,
    request: ActivateCopyRequest,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Activate a copy trading relationship with a session key."""
    try:
        relationship = await manager.activate_relationship(
            relationship_id=relationship_id,
            session_key_id=request.session_key_id,
        )
        return _relationship_to_response(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/relationships/{relationship_id}/pause", response_model=CopyRelationshipResponse)
async def pause_copy_relationship(
    relationship_id: str,
    reason: Optional[str] = None,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Pause a copy trading relationship."""
    try:
        relationship = await manager.pause_copying(relationship_id, reason)
        return _relationship_to_response(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/relationships/{relationship_id}/resume", response_model=CopyRelationshipResponse)
async def resume_copy_relationship(
    relationship_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Resume a paused copy trading relationship."""
    try:
        relationship = await manager.resume_copying(relationship_id)
        return _relationship_to_response(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/relationships/{relationship_id}", response_model=CopyRelationshipResponse)
async def stop_copy_relationship(
    relationship_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Stop and delete a copy trading relationship."""
    try:
        relationship = await manager.stop_copying(relationship_id)
        return _relationship_to_response(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/relationships/{relationship_id}", response_model=CopyRelationshipResponse)
async def update_copy_relationship(
    relationship_id: str,
    request: UpdateCopyConfigRequest,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Update copy trading configuration."""
    try:
        updates = request.model_dump(exclude_unset=True, by_alias=False)
        relationship = await manager.update_config(relationship_id, **updates)
        return _relationship_to_response(relationship)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/relationships/{relationship_id}/executions", response_model=List[CopyExecutionResponse])
async def get_relationship_executions(
    relationship_id: str,
    limit: int = Query(50, ge=1, le=500),
    status: Optional[CopyExecutionStatus] = None,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Get execution history for a copy relationship."""
    executions = await manager.get_executions(
        relationship_id=relationship_id,
        limit=limit,
        status=status,
    )
    return [_execution_to_response(e) for e in executions]


# =============================================================================
# Leader Analytics Endpoints
# =============================================================================


@router.get("/leaders/{chain}/{address}", response_model=LeaderProfileResponse)
async def get_leader_profile(
    chain: str,
    address: str,
    refresh: bool = Query(False, description="Force recalculation of metrics"),
    analytics: LeaderAnalytics = Depends(get_leader_analytics),
):
    """Get a leader's profile with performance metrics."""
    profile = await analytics.get_profile(
        address=address,
        chain=chain,
        refresh=refresh,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Leader not found")
    return _profile_to_response(profile)


@router.get("/leaderboard", response_model=List[LeaderProfileResponse])
async def get_leaderboard(
    chain: Optional[str] = None,
    sort_by: str = Query("total_pnl_usd", alias="sortBy"),
    limit: int = Query(50, ge=1, le=100),
    min_trades: int = Query(10, alias="minTrades"),
    analytics: LeaderAnalytics = Depends(get_leader_analytics),
):
    """Get ranked leaderboard of top performing leaders."""
    profiles = await analytics.get_leaderboard(
        chain=chain,
        sort_by=sort_by,
        limit=limit,
        min_trades=min_trades,
    )
    return [_profile_to_response(p) for p in profiles]


@router.post("/leaders/{chain}/{address}/analyze", response_model=LeaderProfileResponse)
async def analyze_leader(
    chain: str,
    address: str,
    analytics: LeaderAnalytics = Depends(get_leader_analytics),
):
    """Trigger a fresh analysis of a leader's performance."""
    profile = await analytics.get_profile(
        address=address,
        chain=chain,
        refresh=True,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Leader not found")
    return _profile_to_response(profile)


# =============================================================================
# Stats Endpoints
# =============================================================================


@router.get("/stats/user/{user_id}", response_model=CopyTradingStatsResponse)
async def get_user_copy_trading_stats(
    user_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Get aggregated copy trading stats for a user."""
    stats = await manager.get_user_stats(user_id)
    return _stats_to_response(stats)


# =============================================================================
# Estimation Endpoints
# =============================================================================


@router.post("/estimate")
async def estimate_copy_execution(
    request: EstimateExecutionRequest,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Estimate a copy trade execution without executing."""
    try:
        signal = TradeSignal(
            leader_address=request.leader_address,
            leader_chain=request.leader_chain,
            tx_hash="0x" + "0" * 64,  # Dummy hash
            block_number=0,
            timestamp=datetime.now(),
            action="swap",
            token_in_address=request.token_in_address,
            token_in_amount=Decimal("0"),
            token_out_address=request.token_out_address,
            value_usd=Decimal(str(request.value_usd)),
        )

        estimate = await manager.executor.estimate_execution(
            signal=signal,
            size_usd=Decimal(str(request.size_usd)),
            follower_chain=request.follower_chain,
            max_slippage_bps=request.max_slippage_bps,
        )

        return estimate

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Manual Approval Flow Endpoints
# =============================================================================


class ApproveExecutionResponse(BaseModel):
    """Response when approving a copy execution."""

    execution_id: str = Field(..., alias="executionId")
    status: str
    unsigned_transaction: Optional[str] = Field(None, alias="unsignedTransaction")
    transaction_data: Optional[Dict[str, Any]] = Field(None, alias="transactionData")
    quote_response: Optional[Dict[str, Any]] = Field(None, alias="quoteResponse")
    input_amount: Optional[float] = Field(None, alias="inputAmount")
    output_amount: Optional[float] = Field(None, alias="outputAmount")
    price_impact_pct: Optional[float] = Field(None, alias="priceImpactPct")
    last_valid_block_height: Optional[int] = Field(None, alias="lastValidBlockHeight")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")
    chain: str
    error_message: Optional[str] = Field(None, alias="errorMessage")

    class Config:
        populate_by_name = True


class ConfirmExecutionRequest(BaseModel):
    """Request to confirm a copy execution after frontend signing."""

    tx_hash: str = Field(..., alias="txHash", description="Signed transaction hash")

    class Config:
        populate_by_name = True


class RejectExecutionRequest(BaseModel):
    """Request to reject a pending copy execution."""

    reason: Optional[str] = None

    class Config:
        populate_by_name = True


@router.get("/executions/pending/{user_id}", response_model=List[CopyExecutionResponse])
async def get_pending_executions(
    user_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Get all pending approval copy executions for a user."""
    executions = await manager.get_pending_approvals(user_id)
    return [_execution_to_response(e) for e in executions]


@router.post("/executions/{execution_id}/approve", response_model=ApproveExecutionResponse)
async def approve_execution(
    execution_id: str,
    user_id: str = Query(..., alias="userId"),
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """
    Approve a pending copy execution and get unsigned transaction.

    Returns the unsigned transaction for the frontend to sign and submit.
    After frontend signs, call POST /executions/{execution_id}/confirm with the tx hash.
    """
    try:
        # Get the execution
        execution = await manager._load_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        # Verify ownership via relationship
        relationship = manager.get_relationship(execution.relationship_id)
        if not relationship or relationship.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to approve this execution")

        # Check status
        if execution.status != CopyExecutionStatus.PENDING_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=f"Execution is not pending approval (status: {execution.status.value})"
            )

        # Get quote using executor
        quote_result = await manager.executor.get_quote(
            signal=execution.signal,
            size_usd=execution.calculated_size_usd or Decimal("0"),
            follower_address=relationship.follower_address,
            follower_chain=relationship.follower_chain,
            max_slippage_bps=relationship.config.max_slippage_bps,
        )

        if not quote_result.success:
            # Update execution status to failed
            execution.status = CopyExecutionStatus.FAILED
            execution.error_message = quote_result.error_message
            await manager._save_execution(execution)

            return ApproveExecutionResponse(
                executionId=execution_id,
                status="failed",
                chain=relationship.follower_chain,
                errorMessage=quote_result.error_message,
            )

        # Update execution to QUEUED (waiting for signed tx)
        execution.status = CopyExecutionStatus.QUEUED
        execution.execution_started_at = datetime.now()
        await manager._save_execution(execution)

        return ApproveExecutionResponse(
            executionId=execution_id,
            status="queued",
            unsignedTransaction=quote_result.unsigned_transaction,
            transactionData=quote_result.transaction_data,
            quoteResponse=quote_result.quote_response,
            inputAmount=float(quote_result.input_amount) if quote_result.input_amount else None,
            outputAmount=float(quote_result.output_amount) if quote_result.output_amount else None,
            priceImpactPct=quote_result.price_impact_pct,
            lastValidBlockHeight=quote_result.last_valid_block_height,
            expiresAt=quote_result.expires_at,
            chain=relationship.follower_chain,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executions/{execution_id}/confirm", response_model=CopyExecutionResponse)
async def confirm_execution(
    execution_id: str,
    request: ConfirmExecutionRequest,
    user_id: str = Query(..., alias="userId"),
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """
    Confirm a copy execution after frontend has signed and submitted the transaction.

    Call this after receiving the unsigned transaction from /approve,
    signing it with the user's wallet, and submitting it to the chain.
    """
    try:
        # Get the execution
        execution = await manager._load_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        # Verify ownership via relationship
        relationship = manager.get_relationship(execution.relationship_id)
        if not relationship or relationship.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to confirm this execution")

        # Check status
        if execution.status != CopyExecutionStatus.QUEUED:
            raise HTTPException(
                status_code=400,
                detail=f"Execution is not queued (status: {execution.status.value})"
            )

        # Update execution with tx hash
        execution.status = CopyExecutionStatus.COMPLETED
        execution.tx_hash = request.tx_hash
        execution.execution_completed_at = datetime.now()

        # Calculate delay
        if execution.signal_received_at and execution.execution_completed_at:
            delay = (execution.execution_completed_at - execution.signal_received_at).total_seconds()
            execution.execution_delay_seconds = delay

        await manager._save_execution(execution)

        # Update relationship stats
        relationship.successful_trades += 1
        relationship.total_trades += 1
        if execution.actual_size_usd:
            relationship.total_volume_usd += execution.actual_size_usd
            relationship.daily_volume_usd += execution.actual_size_usd
        relationship.daily_trade_count += 1
        relationship.last_copy_at = datetime.now()
        await manager._save_relationship(relationship)

        return _execution_to_response(execution)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executions/{execution_id}/reject", response_model=CopyExecutionResponse)
async def reject_execution(
    execution_id: str,
    request: RejectExecutionRequest,
    user_id: str = Query(..., alias="userId"),
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Reject a pending copy execution."""
    try:
        execution = await manager.reject_execution(
            execution_id=execution_id,
            user_id=user_id,
            reason=request.reason,
        )
        return _execution_to_response(execution)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Key Requirements
# =============================================================================


@router.get("/relationships/{relationship_id}/session-requirements")
async def get_session_requirements(
    relationship_id: str,
    manager: CopyTradingManager = Depends(get_copy_trading_manager),
):
    """Get session key requirements for a copy trading relationship."""
    relationship = manager.get_relationship(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")

    config = relationship.config

    return {
        "permissions": ["swap"],
        "valuePerTxUsd": float(config.max_trade_usd) if config.max_trade_usd else 1000,
        "totalValueUsd": float(config.max_daily_volume_usd),
        "tokenAllowlist": config.token_whitelist or [],
        "chainAllowlist": [relationship.follower_chain],
        "durationDays": 30,
        "maxDailyTrades": config.max_daily_trades,
        "maxSlippageBps": config.max_slippage_bps,
    }
