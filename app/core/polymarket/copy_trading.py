"""
Polymarket Copy Trading Manager

Enable users to follow and automatically copy Polymarket traders.
Uses manual approval flow - no autonomous execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from app.providers.polymarket import (
    PolymarketClient,
    get_polymarket_client,
    OrderSide,
)
from app.services.polymarket_analytics import (
    PolymarketTraderTracker,
    get_trader_tracker,
    TraderPosition,
)
from .trading import PolymarketTradingService, get_polymarket_trading_service
from .models import TradeQuote

logger = logging.getLogger(__name__)


class PMSizingMode(str, Enum):
    """Position sizing mode for Polymarket copy trading."""

    PERCENTAGE = "percentage"  # % of leader's position size
    FIXED = "fixed"  # Fixed USD amount per trade
    PROPORTIONAL = "proportional"  # Proportional to portfolio size


class PMCopyExecutionStatus(str, Enum):
    """Status of a Polymarket copy execution."""

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PolymarketCopyConfig(BaseModel):
    """Configuration for copying a Polymarket trader."""

    leader_address: str = Field(..., alias="leaderAddress")

    # Sizing
    sizing_mode: PMSizingMode = Field(PMSizingMode.PERCENTAGE, alias="sizingMode")
    size_value: Decimal = Field(Decimal("10"), alias="sizeValue")  # e.g., 10%

    # Filters
    min_position_usd: Decimal = Field(Decimal("5"), alias="minPositionUsd")
    max_position_usd: Optional[Decimal] = Field(None, alias="maxPositionUsd")
    category_whitelist: Optional[List[str]] = Field(None, alias="categoryWhitelist")
    category_blacklist: Optional[List[str]] = Field(None, alias="categoryBlacklist")

    # What to copy
    copy_new_positions: bool = Field(True, alias="copyNewPositions")
    copy_exits: bool = Field(True, alias="copyExits")
    sync_existing: bool = Field(False, alias="syncExisting")  # Copy existing positions on start

    # Risk limits
    max_positions: int = Field(20, alias="maxPositions")
    max_exposure_usd: Decimal = Field(Decimal("1000"), alias="maxExposureUsd")
    max_slippage_pct: float = Field(2.0, alias="maxSlippagePct")

    # Timing
    approval_timeout_minutes: int = Field(60, alias="approvalTimeoutMinutes")

    class Config:
        populate_by_name = True


class PolymarketCopyRelationship(BaseModel):
    """A copy trading relationship with a Polymarket trader."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., alias="userId")
    follower_address: str = Field(..., alias="followerAddress")

    # Config
    config: PolymarketCopyConfig

    # Status
    is_active: bool = Field(True, alias="isActive")
    is_paused: bool = Field(False, alias="isPaused")
    pause_reason: Optional[str] = Field(None, alias="pauseReason")

    # Stats
    total_copied_positions: int = Field(0, alias="totalCopiedPositions")
    successful_copies: int = Field(0, alias="successfulCopies")
    failed_copies: int = Field(0, alias="failedCopies")
    skipped_copies: int = Field(0, alias="skippedCopies")
    total_volume_usd: Decimal = Field(Decimal("0"), alias="totalVolumeUsd")
    total_pnl_usd: Optional[Decimal] = Field(None, alias="totalPnlUsd")

    # Current state
    current_exposure_usd: Decimal = Field(Decimal("0"), alias="currentExposureUsd")
    current_positions: int = Field(0, alias="currentPositions")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="createdAt")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="updatedAt")
    last_copy_at: Optional[datetime] = Field(None, alias="lastCopyAt")
    last_synced_at: Optional[datetime] = Field(None, alias="lastSyncedAt")

    @property
    def status(self) -> str:
        """Get relationship status."""
        if not self.is_active:
            return "stopped"
        if self.is_paused:
            return "paused"
        return "active"

    @property
    def leader_address(self) -> str:
        """Get leader address from config."""
        return self.config.leader_address

    @property
    def total_copied_trades(self) -> int:
        """Alias for total_copied_positions."""
        return self.total_copied_positions

    @property
    def total_copied_volume_usd(self) -> Decimal:
        """Alias for total_volume_usd."""
        return self.total_volume_usd

    @property
    def total_pnl_usd(self) -> Decimal:
        """Get total P&L (or 0 if not calculated)."""
        return self._total_pnl_usd or Decimal("0")

    @total_pnl_usd.setter
    def total_pnl_usd(self, value: Optional[Decimal]):
        """Set total P&L."""
        self._total_pnl_usd = value

    @property
    def skipped_trades(self) -> int:
        """Alias for skipped_copies."""
        return self.skipped_copies

    class Config:
        populate_by_name = True


class PolymarketCopyExecution(BaseModel):
    """A single copy trade execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relationship_id: str = Field(..., alias="relationshipId")
    user_id: str = Field(..., alias="userId")
    leader_trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="leaderTradeId")

    # Status
    status: PMCopyExecutionStatus
    skip_reason: Optional[str] = Field(None, alias="skipReason")
    error_message: Optional[str] = Field(None, alias="errorMessage")

    # Leader's action
    leader_address: str = Field(..., alias="leaderAddress")
    leader_action: str = Field(..., alias="leaderAction")  # BUY or SELL
    market_id: str = Field(..., alias="marketId")
    market_question: str = Field(..., alias="marketQuestion")
    outcome: str
    leader_shares: Decimal = Field(..., alias="leaderShares")
    leader_price: Decimal = Field(..., alias="leaderPrice")
    leader_value_usd: Decimal = Field(..., alias="leaderValueUsd")

    @property
    def side(self) -> str:
        """Get side (alias for leader_action)."""
        return self.leader_action

    @property
    def follower_shares(self) -> Optional[Decimal]:
        """Get follower shares (alias for calculated_shares)."""
        return self.calculated_shares

    @property
    def follower_price(self) -> Optional[Decimal]:
        """Get follower price (alias for actual_price or quote avg_price)."""
        return self.actual_price or (self.quote.avg_price if self.quote else None)

    @property
    def follower_amount_usd(self) -> Optional[Decimal]:
        """Get follower amount (alias for calculated_value_usd)."""
        return self.calculated_value_usd

    @property
    def created_at(self) -> datetime:
        """Alias for detected_at."""
        return self.detected_at

    # Calculated copy trade
    calculated_shares: Optional[Decimal] = Field(None, alias="calculatedShares")
    calculated_value_usd: Optional[Decimal] = Field(None, alias="calculatedValueUsd")

    # Quote (from approval)
    quote: Optional[TradeQuote] = None

    # Execution result
    actual_shares: Optional[Decimal] = Field(None, alias="actualShares")
    actual_value_usd: Optional[Decimal] = Field(None, alias="actualValueUsd")
    actual_price: Optional[Decimal] = Field(None, alias="actualPrice")
    tx_hash: Optional[str] = Field(None, alias="txHash")

    # Timestamps
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="detectedAt")
    approved_at: Optional[datetime] = Field(None, alias="approvedAt")
    executed_at: Optional[datetime] = Field(None, alias="executedAt")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")

    class Config:
        populate_by_name = True


class PolymarketCopyManager:
    """
    Manage Polymarket copy trading relationships.

    Uses manual approval flow:
    1. Detect leader position change
    2. Create pending execution
    3. User approves (gets quote)
    4. User signs and submits
    5. Confirm execution
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
        tracker: Optional[PolymarketTraderTracker] = None,
        trading_service: Optional[PolymarketTradingService] = None,
        convex_client: Optional[Any] = None,
    ):
        """Initialize copy manager."""
        self.client = client or get_polymarket_client()
        self.tracker = tracker or get_trader_tracker()
        self.trading_service = trading_service or get_polymarket_trading_service()
        self.convex = convex_client

        # In-memory storage (would be Convex in production)
        self._relationships: Dict[str, PolymarketCopyRelationship] = {}
        self._executions: Dict[str, PolymarketCopyExecution] = {}
        self._user_relationships: Dict[str, List[str]] = {}  # user_id -> relationship_ids
        self._leader_relationships: Dict[str, List[str]] = {}  # leader_address -> relationship_ids

        # Callbacks
        self._on_pending_approval: List[Callable] = []

    # =========================================================================
    # Relationship Management
    # =========================================================================

    async def start_copying(
        self,
        user_id: str,
        follower_address: str,
        config: PolymarketCopyConfig,
    ) -> PolymarketCopyRelationship:
        """
        Start copying a Polymarket trader.

        Args:
            user_id: User ID
            follower_address: User's wallet address
            config: Copy configuration

        Returns:
            New copy relationship
        """
        # Check if already copying this leader
        existing = await self._find_relationship(
            user_id=user_id,
            leader_address=config.leader_address,
        )

        if existing:
            logger.info(f"Already copying {config.leader_address}, updating config")
            existing.config = config
            existing.updated_at = datetime.now(timezone.utc)
            await self._save_relationship(existing)
            return existing

        # Create new relationship
        relationship = PolymarketCopyRelationship(
            userId=user_id,
            followerAddress=follower_address,
            config=config,
        )

        await self._save_relationship(relationship)

        # If sync_existing is enabled, sync leader's current positions
        if config.sync_existing:
            await self._sync_leader_positions(relationship)

        logger.info(f"Started copying {config.leader_address} for user {user_id}")
        return relationship

    async def stop_copying(
        self,
        relationship_id: str,
    ) -> PolymarketCopyRelationship:
        """Stop copying a trader."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        relationship.is_active = False
        relationship.updated_at = datetime.now(timezone.utc)
        await self._save_relationship(relationship)

        logger.info(f"Stopped copying {relationship.config.leader_address}")
        return relationship

    async def pause_copying(
        self,
        relationship_id: str,
        reason: Optional[str] = None,
    ) -> PolymarketCopyRelationship:
        """Pause a copy relationship."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        relationship.is_paused = True
        relationship.pause_reason = reason
        relationship.updated_at = datetime.now(timezone.utc)
        await self._save_relationship(relationship)

        return relationship

    async def resume_copying(
        self,
        relationship_id: str,
    ) -> PolymarketCopyRelationship:
        """Resume a paused copy relationship."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        relationship.is_paused = False
        relationship.pause_reason = None
        relationship.updated_at = datetime.now(timezone.utc)
        await self._save_relationship(relationship)

        return relationship

    async def get_relationships_for_user(
        self,
        user_id: str,
    ) -> List[PolymarketCopyRelationship]:
        """Get all copy relationships for a user."""
        relationship_ids = self._user_relationships.get(user_id, [])
        return [
            self._relationships[rid]
            for rid in relationship_ids
            if rid in self._relationships
        ]

    async def get_relationships(
        self,
        user_id: str,
    ) -> List[PolymarketCopyRelationship]:
        """Alias for get_relationships_for_user."""
        return await self.get_relationships_for_user(user_id)

    async def get_relationships_for_leader(
        self,
        leader_address: str,
    ) -> List[PolymarketCopyRelationship]:
        """Get all relationships following a leader."""
        relationship_ids = self._leader_relationships.get(leader_address.lower(), [])
        return [
            self._relationships[rid]
            for rid in relationship_ids
            if rid in self._relationships and self._relationships[rid].is_active
        ]

    # =========================================================================
    # Trade Detection & Execution
    # =========================================================================

    async def handle_leader_trade(
        self,
        leader_address: str,
        market_id: str,
        outcome: str,
        action: str,  # BUY or SELL
        shares: Decimal,
        price: Decimal,
        value_usd: Decimal,
    ) -> List[PolymarketCopyExecution]:
        """
        Handle a detected trade from a leader.

        Creates pending executions for all followers.

        Args:
            leader_address: Leader's wallet address
            market_id: Market condition ID
            outcome: Outcome name
            action: BUY or SELL
            shares: Number of shares
            price: Trade price
            value_usd: Trade value in USD

        Returns:
            List of created executions
        """
        relationships = await self.get_relationships_for_leader(leader_address)

        if not relationships:
            return []

        # Get market details
        market = await self.client.get_market(market_id)
        market_question = market.question if market else market_id

        executions = []
        for relationship in relationships:
            if relationship.is_paused:
                continue

            # Check filters
            if not self._passes_filters(relationship, action, value_usd, market):
                execution = await self._create_skipped_execution(
                    relationship=relationship,
                    leader_address=leader_address,
                    action=action,
                    market_id=market_id,
                    market_question=market_question,
                    outcome=outcome,
                    shares=shares,
                    price=price,
                    value_usd=value_usd,
                    reason="Filtered out by config",
                )
                executions.append(execution)
                continue

            # Calculate copy size
            calculated_value = self._calculate_copy_size(relationship, value_usd)

            if calculated_value < relationship.config.min_position_usd:
                execution = await self._create_skipped_execution(
                    relationship=relationship,
                    leader_address=leader_address,
                    action=action,
                    market_id=market_id,
                    market_question=market_question,
                    outcome=outcome,
                    shares=shares,
                    price=price,
                    value_usd=value_usd,
                    reason=f"Below minimum position size (${calculated_value} < ${relationship.config.min_position_usd})",
                )
                executions.append(execution)
                continue

            # Check exposure limits
            if relationship.current_exposure_usd + calculated_value > relationship.config.max_exposure_usd:
                execution = await self._create_skipped_execution(
                    relationship=relationship,
                    leader_address=leader_address,
                    action=action,
                    market_id=market_id,
                    market_question=market_question,
                    outcome=outcome,
                    shares=shares,
                    price=price,
                    value_usd=value_usd,
                    reason="Would exceed max exposure",
                )
                executions.append(execution)
                continue

            # Create pending execution
            execution = PolymarketCopyExecution(
                relationshipId=relationship.id,
                userId=relationship.user_id,
                status=PMCopyExecutionStatus.PENDING_APPROVAL,
                leaderAddress=leader_address,
                leaderAction=action,
                marketId=market_id,
                marketQuestion=market_question,
                outcome=outcome,
                leaderShares=shares,
                leaderPrice=price,
                leaderValueUsd=value_usd,
                calculatedValueUsd=calculated_value,
                expiresAt=datetime.now(timezone.utc) + timedelta(
                    minutes=relationship.config.approval_timeout_minutes
                ),
            )

            await self._save_execution(execution)
            executions.append(execution)

            # Notify about pending approval
            await self._notify_pending_approval(execution, relationship)

        return executions

    async def get_pending_approvals(
        self,
        user_id: str,
    ) -> List[PolymarketCopyExecution]:
        """Get all pending approval executions for a user."""
        pending = []
        for execution in self._executions.values():
            if (
                execution.user_id == user_id
                and execution.status == PMCopyExecutionStatus.PENDING_APPROVAL
            ):
                # Check if expired
                if execution.expires_at and datetime.now(timezone.utc) > execution.expires_at:
                    execution.status = PMCopyExecutionStatus.EXPIRED
                    await self._save_execution(execution)
                else:
                    pending.append(execution)

        return pending

    async def get_execution_history(
        self,
        user_id: str,
        limit: int = 50,
        status: Optional[PMCopyExecutionStatus] = None,
    ) -> List[PolymarketCopyExecution]:
        """Get execution history for a user."""
        executions = []
        for execution in self._executions.values():
            if execution.user_id != user_id:
                continue
            if status and execution.status != status:
                continue
            executions.append(execution)

        # Sort by detected_at descending
        executions.sort(key=lambda e: e.detected_at, reverse=True)
        return executions[:limit]

    async def approve_execution(
        self,
        execution_id: str,
        user_id: Optional[str] = None,
    ) -> PolymarketCopyExecution:
        """
        Approve a pending execution and get quote.

        Args:
            execution_id: Execution ID
            user_id: Optional user ID (for verification)

        Returns:
            Updated execution with quote
        """
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        if user_id and execution.user_id != user_id:
            raise ValueError("Not authorized to approve this execution")

        if execution.status != PMCopyExecutionStatus.PENDING_APPROVAL:
            raise ValueError(f"Execution is not pending approval (status: {execution.status})")

        # Check expiry
        if execution.expires_at and datetime.now(timezone.utc) > execution.expires_at:
            execution.status = PMCopyExecutionStatus.EXPIRED
            await self._save_execution(execution)
            raise ValueError("Execution has expired")

        # Get quote
        side = OrderSide.BUY if execution.leader_action == "BUY" else OrderSide.SELL
        relationship = self._relationships.get(execution.relationship_id)

        if side == OrderSide.BUY:
            quote = await self.trading_service.get_buy_quote(
                market_id=execution.market_id,
                outcome=execution.outcome,
                amount_usd=execution.calculated_value_usd or Decimal("0"),
            )
        else:
            quote = await self.trading_service.get_sell_quote(
                market_id=execution.market_id,
                outcome=execution.outcome,
                shares=execution.calculated_shares,
                address=relationship.follower_address if relationship else None,
            )

        if not quote:
            execution.status = PMCopyExecutionStatus.FAILED
            execution.error_message = "Failed to get quote"
            await self._save_execution(execution)
            raise ValueError("Failed to get quote")

        execution.quote = quote
        execution.calculated_shares = quote.shares
        execution.status = PMCopyExecutionStatus.APPROVED
        execution.approved_at = datetime.now(timezone.utc)
        await self._save_execution(execution)

        return execution

    async def confirm_execution(
        self,
        execution_id: str,
        tx_hash: str,
        user_id: Optional[str] = None,
    ) -> PolymarketCopyExecution:
        """
        Confirm execution after user has signed and submitted.

        Args:
            execution_id: Execution ID
            tx_hash: Transaction hash
            user_id: Optional user ID (for verification)

        Returns:
            Updated execution
        """
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        if user_id and execution.user_id != user_id:
            raise ValueError("Not authorized to confirm this execution")

        if execution.status != PMCopyExecutionStatus.APPROVED:
            raise ValueError(f"Execution is not approved (status: {execution.status})")

        execution.status = PMCopyExecutionStatus.COMPLETED
        execution.tx_hash = tx_hash
        execution.executed_at = datetime.now(timezone.utc)
        execution.actual_shares = execution.calculated_shares
        execution.actual_value_usd = execution.calculated_value_usd
        execution.actual_price = execution.quote.avg_price if execution.quote else None
        await self._save_execution(execution)

        # Update relationship stats
        relationship = self._relationships.get(execution.relationship_id)
        if relationship:
            relationship.successful_copies += 1
            relationship.total_copied_positions += 1
            relationship.total_volume_usd += execution.actual_value_usd or Decimal("0")
            relationship.current_exposure_usd += execution.actual_value_usd or Decimal("0")
            relationship.current_positions += 1
            relationship.last_copy_at = datetime.now(timezone.utc)
            relationship.updated_at = datetime.now(timezone.utc)
            await self._save_relationship(relationship)

        return execution

    async def reject_execution(
        self,
        execution_id: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> PolymarketCopyExecution:
        """Reject a pending execution."""
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        if user_id and execution.user_id != user_id:
            raise ValueError("Not authorized to reject this execution")

        execution.status = PMCopyExecutionStatus.REJECTED
        execution.skip_reason = reason or "User rejected"
        await self._save_execution(execution)

        # Update relationship stats
        relationship = self._relationships.get(execution.relationship_id)
        if relationship:
            relationship.skipped_copies += 1
            await self._save_relationship(relationship)

        return execution

    # =========================================================================
    # Sync & Polling
    # =========================================================================

    async def sync_leader_positions(
        self,
        relationship_id: str,
    ) -> List[PolymarketCopyExecution]:
        """
        Sync with leader's current positions.

        Creates pending executions for positions user doesn't have.
        """
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        return await self._sync_leader_positions(relationship)

    async def _sync_leader_positions(
        self,
        relationship: PolymarketCopyRelationship,
    ) -> List[PolymarketCopyExecution]:
        """Internal sync implementation."""
        leader_positions = await self.tracker.get_trader_positions(
            relationship.config.leader_address
        )

        # Get user's positions to compare
        user_positions = await self.client.get_positions(relationship.follower_address)
        user_markets = {p.market_id for p in user_positions}

        executions = []
        for leader_pos in leader_positions:
            # Skip if user already has position in this market
            if leader_pos.market_id in user_markets:
                continue

            # Create pending execution
            execution = await self.handle_leader_trade(
                leader_address=relationship.config.leader_address,
                market_id=leader_pos.market_id,
                outcome=leader_pos.outcome,
                action="BUY",
                shares=leader_pos.shares,
                price=leader_pos.avg_entry_price,
                value_usd=leader_pos.cost_basis,
            )
            executions.extend(execution)

        relationship.last_synced_at = datetime.now(timezone.utc)
        await self._save_relationship(relationship)

        return executions

    # =========================================================================
    # Helpers
    # =========================================================================

    def _passes_filters(
        self,
        relationship: PolymarketCopyRelationship,
        action: str,
        value_usd: Decimal,
        market: Optional[Any],
    ) -> bool:
        """Check if trade passes config filters."""
        config = relationship.config

        # Check action type
        if action == "BUY" and not config.copy_new_positions:
            return False
        if action == "SELL" and not config.copy_exits:
            return False

        # Check position count limit
        if relationship.current_positions >= config.max_positions and action == "BUY":
            return False

        # Check max position size
        if config.max_position_usd and value_usd > config.max_position_usd:
            return False

        # Check category filters (would need market category)
        # This is simplified - full implementation would check market tags

        return True

    def _calculate_copy_size(
        self,
        relationship: PolymarketCopyRelationship,
        leader_value_usd: Decimal,
    ) -> Decimal:
        """Calculate the copy trade size based on sizing mode."""
        config = relationship.config

        if config.sizing_mode == PMSizingMode.FIXED:
            return config.size_value

        elif config.sizing_mode == PMSizingMode.PERCENTAGE:
            return leader_value_usd * config.size_value / Decimal("100")

        elif config.sizing_mode == PMSizingMode.PROPORTIONAL:
            # Would need to compare portfolio sizes
            # For now, use percentage
            return leader_value_usd * config.size_value / Decimal("100")

        return config.size_value

    async def _create_skipped_execution(
        self,
        relationship: PolymarketCopyRelationship,
        leader_address: str,
        action: str,
        market_id: str,
        market_question: str,
        outcome: str,
        shares: Decimal,
        price: Decimal,
        value_usd: Decimal,
        reason: str,
    ) -> PolymarketCopyExecution:
        """Create a skipped execution record."""
        execution = PolymarketCopyExecution(
            relationshipId=relationship.id,
            userId=relationship.user_id,
            status=PMCopyExecutionStatus.SKIPPED,
            skipReason=reason,
            leaderAddress=leader_address,
            leaderAction=action,
            marketId=market_id,
            marketQuestion=market_question,
            outcome=outcome,
            leaderShares=shares,
            leaderPrice=price,
            leaderValueUsd=value_usd,
        )

        await self._save_execution(execution)

        relationship.skipped_copies += 1
        await self._save_relationship(relationship)

        return execution

    async def _find_relationship(
        self,
        user_id: str,
        leader_address: str,
    ) -> Optional[PolymarketCopyRelationship]:
        """Find existing relationship."""
        for rel in self._relationships.values():
            if (
                rel.user_id == user_id
                and rel.config.leader_address.lower() == leader_address.lower()
            ):
                return rel
        return None

    async def _save_relationship(
        self,
        relationship: PolymarketCopyRelationship,
    ):
        """Save relationship (in-memory or Convex)."""
        self._relationships[relationship.id] = relationship

        # Update indices
        if relationship.user_id not in self._user_relationships:
            self._user_relationships[relationship.user_id] = []
        if relationship.id not in self._user_relationships[relationship.user_id]:
            self._user_relationships[relationship.user_id].append(relationship.id)

        leader_key = relationship.config.leader_address.lower()
        if leader_key not in self._leader_relationships:
            self._leader_relationships[leader_key] = []
        if relationship.id not in self._leader_relationships[leader_key]:
            self._leader_relationships[leader_key].append(relationship.id)

    async def _save_execution(self, execution: PolymarketCopyExecution):
        """Save execution (in-memory or Convex)."""
        self._executions[execution.id] = execution

    async def _notify_pending_approval(
        self,
        execution: PolymarketCopyExecution,
        relationship: PolymarketCopyRelationship,
    ):
        """Notify user about pending approval."""
        for callback in self._on_pending_approval:
            try:
                await callback(execution, relationship)
            except Exception as e:
                logger.error(f"Error in pending approval callback: {e}")

    def on_pending_approval(self, callback: Callable):
        """Register callback for pending approval notifications."""
        self._on_pending_approval.append(callback)


# Singleton instance
_copy_manager_instance: Optional[PolymarketCopyManager] = None


def get_polymarket_copy_manager() -> PolymarketCopyManager:
    """Get singleton copy manager instance."""
    global _copy_manager_instance
    if _copy_manager_instance is None:
        _copy_manager_instance = PolymarketCopyManager()
    return _copy_manager_instance
