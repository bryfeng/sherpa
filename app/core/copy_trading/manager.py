"""
Copy Trading Manager

Main orchestration layer for copy trading. Manages relationships,
processes trade signals, and coordinates execution.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from .models import (
    CopyConfig,
    CopyRelationship,
    CopyExecution,
    CopyExecutionStatus,
    CopyTradingStats,
    TradeSignal,
    LeaderProfile,
    SkipReason,
    SizingMode,
)
from .sizing import SizingStrategy, get_sizing_strategy
from ..wallet.models import SessionKey

logger = logging.getLogger(__name__)


class CopyTradingManager:
    """
    Manages copy trading relationships and execution.

    Responsibilities:
    - Create/update/delete copy relationships
    - Process trade signals from leaders
    - Coordinate with executor for trade execution
    - Track stats and performance
    """

    def __init__(
        self,
        convex_client: Optional[Any] = None,
        executor: Optional[Any] = None,  # CopyExecutor
        portfolio_provider: Optional[Any] = None,
    ):
        self.convex_client = convex_client
        self.executor = executor
        self.portfolio_provider = portfolio_provider

        # In-memory caches
        self._relationships: Dict[str, CopyRelationship] = {}
        self._leaders_by_address: Dict[str, List[str]] = {}  # address -> relationship_ids

        # Callbacks
        self._on_copy_executed: List[Callable] = []
        self._on_copy_skipped: List[Callable] = []
        self._on_copy_failed: List[Callable] = []

    # =========================================================================
    # Relationship Management
    # =========================================================================

    async def start_copying(
        self,
        user_id: str,
        follower_address: str,
        follower_chain: str,
        config: CopyConfig,
    ) -> CopyRelationship:
        """
        Start copying a wallet.

        Args:
            user_id: ID of the user starting the copy
            follower_address: Address of the follower wallet
            follower_chain: Chain of the follower wallet
            config: Copy configuration

        Returns:
            New copy relationship
        """
        # Check if already copying this leader
        existing = await self._find_relationship(
            user_id=user_id,
            leader_address=config.leader_address,
            leader_chain=config.leader_chain,
        )

        if existing:
            logger.info(f"Already copying {config.leader_address}, updating config")
            existing.config = config
            existing.updated_at = datetime.now(timezone.utc)
            await self._save_relationship(existing)
            return existing

        # Create new relationship
        relationship = CopyRelationship(
            user_id=user_id,
            follower_address=follower_address,
            follower_chain=follower_chain,
            config=config,
        )

        # Save to storage
        await self._save_relationship(relationship)

        # Update caches
        self._relationships[relationship.id] = relationship
        leader_key = f"{config.leader_chain}:{config.leader_address.lower()}"
        if leader_key not in self._leaders_by_address:
            self._leaders_by_address[leader_key] = []
        self._leaders_by_address[leader_key].append(relationship.id)

        logger.info(
            f"Started copying {config.leader_address} for user {user_id} "
            f"(relationship {relationship.id})"
        )

        return relationship

    async def stop_copying(
        self,
        relationship_id: str,
    ) -> CopyRelationship:
        """Stop copying a wallet."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            relationship = await self._load_relationship(relationship_id)

        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        # Deactivate
        relationship.is_active = False
        relationship.updated_at = datetime.now(timezone.utc)

        await self._save_relationship(relationship)

        # Remove from caches
        leader_key = f"{relationship.config.leader_chain}:{relationship.config.leader_address.lower()}"
        if leader_key in self._leaders_by_address:
            self._leaders_by_address[leader_key] = [
                rid for rid in self._leaders_by_address[leader_key]
                if rid != relationship.id
            ]

        logger.info(f"Stopped copying for relationship {relationship_id}")

        return relationship

    async def pause_copying(
        self,
        relationship_id: str,
        reason: Optional[str] = None,
    ) -> CopyRelationship:
        """Temporarily pause a copy relationship."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            relationship = await self._load_relationship(relationship_id)

        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        relationship.is_paused = True
        relationship.pause_reason = reason
        relationship.updated_at = datetime.now(timezone.utc)

        await self._save_relationship(relationship)

        logger.info(f"Paused relationship {relationship_id}: {reason}")

        return relationship

    async def resume_copying(self, relationship_id: str) -> CopyRelationship:
        """Resume a paused copy relationship."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            relationship = await self._load_relationship(relationship_id)

        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        relationship.is_paused = False
        relationship.pause_reason = None
        relationship.updated_at = datetime.now(timezone.utc)

        await self._save_relationship(relationship)

        logger.info(f"Resumed relationship {relationship_id}")

        return relationship

    async def update_config(
        self,
        relationship_id: str,
        **kwargs,
    ) -> CopyRelationship:
        """Update copy configuration for a relationship with partial updates."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            relationship = await self._load_relationship(relationship_id)

        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        # Apply partial updates to config
        config = relationship.config
        for key, value in kwargs.items():
            if value is not None:
                if key == "sizing_mode" and isinstance(value, str):
                    value = SizingMode(value)
                if key in ["size_value", "min_trade_usd", "max_trade_usd", "max_daily_volume_usd"]:
                    value = Decimal(str(value)) if value is not None else None
                if hasattr(config, key):
                    setattr(config, key, value)

        relationship.config = config
        relationship.updated_at = datetime.now(timezone.utc)

        await self._save_relationship(relationship)

        return relationship

    def get_relationship(self, relationship_id: str) -> Optional[CopyRelationship]:
        """Get a relationship by ID from cache."""
        return self._relationships.get(relationship_id)

    async def get_relationship_async(self, relationship_id: str) -> Optional[CopyRelationship]:
        """Get a relationship by ID, loading from storage if needed."""
        if relationship_id in self._relationships:
            return self._relationships[relationship_id]
        return await self._load_relationship(relationship_id)

    async def get_relationships_for_user(self, user_id: str) -> List[CopyRelationship]:
        """Get all relationships for a user."""
        # Check cache first
        cached = [r for r in self._relationships.values() if r.user_id == user_id]

        if cached:
            return cached

        # Load from storage
        if self.convex_client:
            try:
                data = await self.convex_client.query(
                    "copyTrading:listByUser",
                    {"userId": user_id},
                )
                return [self._dict_to_relationship(d) for d in data]
            except Exception as e:
                logger.error(f"Failed to load relationships for user {user_id}: {e}")

        return []

    async def get_relationships_for_leader(
        self,
        leader_address: str,
        leader_chain: str,
    ) -> List[CopyRelationship]:
        """Get all active relationships following a specific leader."""
        leader_key = f"{leader_chain}:{leader_address.lower()}"

        # Check cache
        relationship_ids = self._leaders_by_address.get(leader_key, [])
        if relationship_ids:
            return [
                self._relationships[rid]
                for rid in relationship_ids
                if rid in self._relationships and self._relationships[rid].is_active
            ]

        # Load from storage
        if self.convex_client:
            try:
                data = await self.convex_client.query(
                    "copyTrading:listByLeader",
                    {
                        "leaderAddress": leader_address.lower(),
                        "leaderChain": leader_chain,
                    },
                )
                relationships = [self._dict_to_relationship(d) for d in data]

                # Update cache
                for r in relationships:
                    self._relationships[r.id] = r
                self._leaders_by_address[leader_key] = [r.id for r in relationships]

                return relationships
            except Exception as e:
                logger.error(f"Failed to load relationships for leader: {e}")

        return []

    async def get_relationships_for_follower(
        self,
        follower_address: str,
        follower_chain: Optional[str] = None,
    ) -> List[CopyRelationship]:
        """Get all relationships for a follower wallet."""
        # Check cache first
        cached = [
            r for r in self._relationships.values()
            if r.follower_address.lower() == follower_address.lower()
            and (follower_chain is None or r.follower_chain == follower_chain)
        ]

        if cached:
            return cached

        # Load from storage
        if self.convex_client:
            try:
                data = await self.convex_client.query(
                    "copyTrading:listByFollower",
                    {
                        "followerAddress": follower_address.lower(),
                        "followerChain": follower_chain,
                    },
                )
                relationships = [self._dict_to_relationship(d) for d in data]

                # Update cache
                for r in relationships:
                    self._relationships[r.id] = r

                return relationships
            except Exception as e:
                logger.error(f"Failed to load relationships for follower: {e}")

        return []

    async def activate_relationship(
        self,
        relationship_id: str,
        session_key_id: str,
    ) -> CopyRelationship:
        """Activate a copy relationship with a session key."""
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            relationship = await self._load_relationship(relationship_id)

        if not relationship:
            raise ValueError(f"Relationship {relationship_id} not found")

        # Set session key
        relationship.config.session_key_id = session_key_id
        relationship.is_active = True
        relationship.is_paused = False
        relationship.pause_reason = None
        relationship.updated_at = datetime.now(timezone.utc)

        await self._save_relationship(relationship)

        # Update leader cache
        leader_key = f"{relationship.config.leader_chain}:{relationship.config.leader_address.lower()}"
        if leader_key not in self._leaders_by_address:
            self._leaders_by_address[leader_key] = []
        if relationship.id not in self._leaders_by_address[leader_key]:
            self._leaders_by_address[leader_key].append(relationship.id)

        logger.info(f"Activated relationship {relationship_id} with session key")

        return relationship

    async def get_executions(
        self,
        relationship_id: str,
        limit: int = 50,
        status: Optional[CopyExecutionStatus] = None,
    ) -> List[CopyExecution]:
        """Get execution history for a relationship."""
        if not self.convex_client:
            return []

        try:
            data = await self.convex_client.query(
                "copyTrading:listExecutions",
                {
                    "relationshipId": relationship_id,
                    "limit": limit,
                    "status": status.value if status else None,
                },
            )
            return [self._dict_to_execution(d) for d in data]
        except Exception as e:
            logger.error(f"Failed to load executions: {e}")
            return []

    async def get_user_stats(self, user_id: str) -> CopyTradingStats:
        """Get aggregated copy trading stats for a user."""
        relationships = await self.get_relationships_for_user(user_id)

        stats = CopyTradingStats(user_id=user_id)
        stats.total_relationships = len(relationships)
        stats.active_relationships = sum(1 for r in relationships if r.is_active and not r.is_paused)

        for r in relationships:
            stats.total_copy_trades += r.total_trades
            stats.successful_trades += r.successful_trades
            stats.failed_trades += r.failed_trades
            stats.skipped_trades += r.skipped_trades
            stats.total_volume_usd += r.total_volume_usd
            stats.today_volume_usd += r.daily_volume_usd

            if r.total_pnl_usd is not None:
                if stats.total_pnl_usd is None:
                    stats.total_pnl_usd = Decimal("0")
                stats.total_pnl_usd += r.total_pnl_usd

            if r.last_copy_at:
                if stats.last_copy_at is None or r.last_copy_at > stats.last_copy_at:
                    stats.last_copy_at = r.last_copy_at

        return stats

    async def get_pending_approvals(
        self,
        user_id: str,
    ) -> List[CopyExecution]:
        """Get all executions pending user approval."""
        if not self.convex_client:
            return []

        try:
            data = await self.convex_client.query(
                "copyTrading:listPendingApprovals",
                {"userId": user_id},
            )
            return [self._dict_to_execution(d) for d in data]
        except Exception as e:
            logger.error(f"Failed to load pending approvals: {e}")
            return []

    async def approve_execution(
        self,
        execution_id: str,
        user_id: str,
    ) -> CopyExecution:
        """
        Approve and execute a pending copy trade.

        Args:
            execution_id: ID of the execution to approve
            user_id: User approving (for authorization)

        Returns:
            Updated execution
        """
        # Load execution
        execution = await self._load_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        # Check status
        if execution.status != CopyExecutionStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Execution is not pending approval (status: {execution.status})"
            )

        # Load relationship
        relationship = await self.get_relationship_async(execution.relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {execution.relationship_id} not found")

        # Verify user owns this relationship
        if relationship.user_id != user_id:
            raise ValueError("User not authorized to approve this execution")

        # Check if trade is too old
        config = relationship.config
        age_seconds = (datetime.now(timezone.utc) - execution.signal.timestamp).total_seconds()
        if age_seconds > config.max_delay_seconds:
            execution.status = CopyExecutionStatus.EXPIRED
            execution.error_message = f"Trade expired after {age_seconds:.0f}s (max: {config.max_delay_seconds}s)"
            await self._save_execution(execution)
            return execution

        # Execute the trade
        return await self._execute_copy(execution, relationship)

    async def reject_execution(
        self,
        execution_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> CopyExecution:
        """
        Reject/cancel a pending copy trade.

        Args:
            execution_id: ID of the execution to reject
            user_id: User rejecting (for authorization)
            reason: Optional rejection reason

        Returns:
            Updated execution
        """
        # Load execution
        execution = await self._load_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        # Check status
        if execution.status != CopyExecutionStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Execution is not pending approval (status: {execution.status})"
            )

        # Load relationship to verify authorization
        relationship = await self.get_relationship_async(execution.relationship_id)
        if not relationship:
            raise ValueError(f"Relationship {execution.relationship_id} not found")

        if relationship.user_id != user_id:
            raise ValueError("User not authorized to reject this execution")

        # Update status
        execution.status = CopyExecutionStatus.CANCELLED
        execution.error_message = reason or "Rejected by user"
        execution.execution_completed_at = datetime.now(timezone.utc)

        # Update relationship stats
        relationship.record_trade(success=False, volume_usd=Decimal("0"), skipped=True)

        await self._save_execution(execution)
        await self._save_relationship(relationship)

        logger.info(f"Rejected execution {execution_id}: {reason}")

        return execution

    # =========================================================================
    # Signal Processing
    # =========================================================================

    async def handle_trade_signal(
        self,
        signal: TradeSignal,
    ) -> List[CopyExecution]:
        """
        Process a trade signal from a leader.

        Args:
            signal: Trade signal from the leader

        Returns:
            List of copy executions (one per follower)
        """
        # Get all followers of this leader
        relationships = await self.get_relationships_for_leader(
            leader_address=signal.leader_address,
            leader_chain=signal.leader_chain,
        )

        if not relationships:
            logger.debug(f"No followers for {signal.leader_address}")
            return []

        logger.info(
            f"Processing signal from {signal.leader_address}: "
            f"{signal.action} {signal.token_in_symbol} -> {signal.token_out_symbol}"
        )

        # Process each relationship
        executions: List[CopyExecution] = []

        for relationship in relationships:
            execution = await self._process_signal_for_relationship(
                signal=signal,
                relationship=relationship,
            )
            if execution:
                executions.append(execution)

        return executions

    async def _process_signal_for_relationship(
        self,
        signal: TradeSignal,
        relationship: CopyRelationship,
    ) -> Optional[CopyExecution]:
        """Process a signal for a specific relationship."""
        config = relationship.config

        # Create execution record
        execution = CopyExecution(
            relationship_id=relationship.id,
            signal=signal,
        )

        # Check if action is allowed
        if not config.is_action_allowed(signal.action):
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.ACTION_NOT_ALLOWED
            await self._notify_skipped(execution, relationship)
            return execution

        # Check token allowlist/blacklist
        if not config.is_token_allowed(signal.token_in_address):
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.TOKEN_BLACKLISTED
            await self._notify_skipped(execution, relationship)
            return execution

        if not config.is_token_allowed(signal.token_out_address):
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.TOKEN_BLACKLISTED
            await self._notify_skipped(execution, relationship)
            return execution

        # Check if trade is too old
        age_seconds = (datetime.now(timezone.utc) - signal.timestamp).total_seconds()
        if age_seconds > config.max_delay_seconds:
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.TRADE_TOO_OLD
            await self._notify_skipped(execution, relationship)
            return execution

        # Check daily limits
        can_execute, skip_reason = relationship.can_execute_trade(
            signal.value_usd or Decimal("0")
        )
        if not can_execute:
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = skip_reason
            await self._notify_skipped(execution, relationship)
            return execution

        # Calculate size
        try:
            size_usd = await self._calculate_trade_size(
                signal=signal,
                relationship=relationship,
            )
        except Exception as e:
            logger.error(f"Error calculating trade size: {e}")
            execution.status = CopyExecutionStatus.FAILED
            execution.error_message = str(e)
            await self._notify_failed(execution, relationship)
            return execution

        if size_usd <= 0:
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.VALUE_TOO_LOW
            await self._notify_skipped(execution, relationship)
            return execution

        execution.calculated_size_usd = size_usd

        # Check if session key is configured (autonomous mode)
        if config.session_key_id:
            # Apply delay if configured
            if config.delay_seconds > 0:
                execution.status = CopyExecutionStatus.QUEUED
                await self._save_execution(execution)

                # Schedule delayed execution
                asyncio.create_task(
                    self._delayed_execute(execution, relationship, config.delay_seconds)
                )
                return execution

            # Execute immediately (autonomous mode with session key)
            return await self._execute_copy(execution, relationship)

        # Manual approval flow - set to PENDING_APPROVAL and notify user
        execution.status = CopyExecutionStatus.PENDING_APPROVAL
        await self._save_execution(execution)
        await self._notify_pending_approval(execution, relationship)

        return execution

    async def _delayed_execute(
        self,
        execution: CopyExecution,
        relationship: CopyRelationship,
        delay_seconds: int,
    ):
        """Execute a copy trade after a delay."""
        await asyncio.sleep(delay_seconds)

        # Re-check if still valid
        age_seconds = (datetime.now(timezone.utc) - execution.signal.timestamp).total_seconds()
        if age_seconds > relationship.config.max_delay_seconds:
            execution.status = CopyExecutionStatus.SKIPPED
            execution.skip_reason = SkipReason.TRADE_TOO_OLD
            await self._save_execution(execution)
            return

        await self._execute_copy(execution, relationship)

    async def _execute_copy(
        self,
        execution: CopyExecution,
        relationship: CopyRelationship,
    ) -> CopyExecution:
        """Execute a copy trade."""
        execution.status = CopyExecutionStatus.EXECUTING
        execution.execution_started_at = datetime.now(timezone.utc)

        if not self.executor:
            execution.status = CopyExecutionStatus.FAILED
            execution.error_message = "No executor configured"
            relationship.record_trade(success=False, volume_usd=Decimal("0"))
            await self._save_execution(execution)
            await self._save_relationship(relationship)
            return execution

        try:
            # Get session key if configured
            session_key = None
            if relationship.config.session_key_id:
                session_key = await self._get_session_key(relationship.config.session_key_id)
                if not session_key:
                    execution.status = CopyExecutionStatus.SKIPPED
                    execution.skip_reason = SkipReason.SESSION_EXPIRED
                    relationship.record_trade(success=False, volume_usd=Decimal("0"), skipped=True)
                    await self._save_execution(execution)
                    await self._save_relationship(relationship)
                    return execution

            # Execute the trade
            result = await self.executor.execute(
                signal=execution.signal,
                size_usd=execution.calculated_size_usd,
                follower_address=relationship.follower_address,
                follower_chain=relationship.follower_chain,
                max_slippage_bps=relationship.config.max_slippage_bps,
                session_key=session_key,
            )

            # Update execution with result
            execution.status = CopyExecutionStatus.COMPLETED if result.success else CopyExecutionStatus.FAILED
            execution.tx_hash = result.tx_hash
            execution.actual_size_usd = result.actual_value_usd
            execution.token_out_amount = result.token_out_amount
            execution.slippage_bps = result.slippage_bps
            execution.gas_used = result.gas_used
            execution.gas_price_gwei = result.gas_price_gwei
            execution.gas_cost_usd = result.gas_cost_usd
            execution.error_message = result.error_message
            execution.execution_completed_at = datetime.now(timezone.utc)

            # Update relationship stats
            relationship.record_trade(
                success=result.success,
                volume_usd=result.actual_value_usd or Decimal("0"),
            )

            # Notify
            if result.success:
                await self._notify_executed(execution, relationship)
            else:
                await self._notify_failed(execution, relationship)

        except Exception as e:
            logger.error(f"Copy execution failed: {e}", exc_info=True)
            execution.status = CopyExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.execution_completed_at = datetime.now(timezone.utc)
            relationship.record_trade(success=False, volume_usd=Decimal("0"))
            await self._notify_failed(execution, relationship)

        await self._save_execution(execution)
        await self._save_relationship(relationship)

        return execution

    async def _calculate_trade_size(
        self,
        signal: TradeSignal,
        relationship: CopyRelationship,
    ) -> Decimal:
        """Calculate the trade size for a copy."""
        config = relationship.config

        # Get follower portfolio value
        follower_portfolio_value = Decimal("10000")  # Default
        if self.portfolio_provider:
            try:
                portfolio = await self.portfolio_provider.get_portfolio(
                    address=relationship.follower_address,
                    chain=relationship.follower_chain,
                )
                follower_portfolio_value = Decimal(str(portfolio.total_value_usd))
            except Exception as e:
                logger.warning(f"Could not get follower portfolio: {e}")

        # Get leader portfolio value (for proportional sizing)
        leader_portfolio_value = None
        if config.sizing_mode.value == "proportional" and self.portfolio_provider:
            try:
                portfolio = await self.portfolio_provider.get_portfolio(
                    address=config.leader_address,
                    chain=config.leader_chain,
                )
                leader_portfolio_value = Decimal(str(portfolio.total_value_usd))
            except Exception as e:
                logger.warning(f"Could not get leader portfolio: {e}")

        # Get sizing strategy and calculate
        strategy = get_sizing_strategy(config.sizing_mode)
        size_usd = strategy.calculate_size(
            signal=signal,
            config=config,
            follower_portfolio_value_usd=follower_portfolio_value,
            leader_portfolio_value_usd=leader_portfolio_value,
        )

        return size_usd

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_executed(self, callback: Callable):
        """Register callback for successful copy executions."""
        self._on_copy_executed.append(callback)

    def on_skipped(self, callback: Callable):
        """Register callback for skipped copy trades."""
        self._on_copy_skipped.append(callback)

    def on_failed(self, callback: Callable):
        """Register callback for failed copy trades."""
        self._on_copy_failed.append(callback)

    def on_pending_approval(self, callback: Callable):
        """Register callback for trades pending user approval."""
        if not hasattr(self, "_on_pending_approval"):
            self._on_pending_approval = []
        self._on_pending_approval.append(callback)

    async def _notify_pending_approval(self, execution: CopyExecution, relationship: CopyRelationship):
        """Notify listeners of trade pending approval."""
        if not hasattr(self, "_on_pending_approval"):
            self._on_pending_approval = []
        for cb in self._on_pending_approval:
            try:
                await cb(execution, relationship)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _notify_executed(self, execution: CopyExecution, relationship: CopyRelationship):
        """Notify listeners of successful execution."""
        for cb in self._on_copy_executed:
            try:
                await cb(execution, relationship)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _notify_skipped(self, execution: CopyExecution, relationship: CopyRelationship):
        """Notify listeners of skipped trade."""
        for cb in self._on_copy_skipped:
            try:
                await cb(execution, relationship)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _notify_failed(self, execution: CopyExecution, relationship: CopyRelationship):
        """Notify listeners of failed execution."""
        for cb in self._on_copy_failed:
            try:
                await cb(execution, relationship)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # =========================================================================
    # Storage Helpers
    # =========================================================================

    async def _save_relationship(self, relationship: CopyRelationship):
        """Save relationship to storage."""
        self._relationships[relationship.id] = relationship

        if self.convex_client:
            try:
                await self.convex_client.mutation(
                    "copyTrading:upsertRelationship",
                    self._relationship_to_dict(relationship),
                )
            except Exception as e:
                logger.error(f"Failed to save relationship: {e}")

    async def _load_relationship(self, relationship_id: str) -> Optional[CopyRelationship]:
        """Load relationship from storage."""
        if not self.convex_client:
            return None

        try:
            data = await self.convex_client.query(
                "copyTrading:getRelationship",
                {"id": relationship_id},
            )
            if data:
                relationship = self._dict_to_relationship(data)
                self._relationships[relationship.id] = relationship
                return relationship
        except Exception as e:
            logger.error(f"Failed to load relationship: {e}")

        return None

    async def _find_relationship(
        self,
        user_id: str,
        leader_address: str,
        leader_chain: str,
    ) -> Optional[CopyRelationship]:
        """Find a relationship by user and leader."""
        # Check cache
        for r in self._relationships.values():
            if (
                r.user_id == user_id
                and r.config.leader_address.lower() == leader_address.lower()
                and r.config.leader_chain == leader_chain
            ):
                return r

        # Check storage
        if self.convex_client:
            try:
                data = await self.convex_client.query(
                    "copyTrading:findRelationship",
                    {
                        "userId": user_id,
                        "leaderAddress": leader_address.lower(),
                        "leaderChain": leader_chain,
                    },
                )
                if data:
                    return self._dict_to_relationship(data)
            except Exception as e:
                logger.error(f"Failed to find relationship: {e}")

        return None

    async def _save_execution(self, execution: CopyExecution):
        """Save execution to storage."""
        if self.convex_client:
            try:
                await self.convex_client.mutation(
                    "copyTrading:upsertExecution",
                    self._execution_to_dict(execution),
                )
            except Exception as e:
                logger.error(f"Failed to save execution: {e}")

    async def _load_execution(self, execution_id: str) -> Optional[CopyExecution]:
        """Load execution from storage."""
        if not self.convex_client:
            return None

        try:
            data = await self.convex_client.query(
                "copyTrading:getExecution",
                {"id": execution_id},
            )
            if data:
                return self._dict_to_execution(data)
        except Exception as e:
            logger.error(f"Failed to load execution: {e}")

        return None

    async def _get_session_key(self, session_key_id: str) -> Optional[SessionKey]:
        """Get a session key by ID."""
        if not self.convex_client:
            return None

        try:
            data = await self.convex_client.query(
                "sessionKeys:getById",
                {"id": session_key_id},
            )
            if data and data.get("status") == "active":
                # Check if expired
                if data.get("expiresAt", 0) > datetime.now(timezone.utc).timestamp() * 1000:
                    return SessionKey(**data)
        except Exception as e:
            logger.error(f"Failed to get session key: {e}")

        return None

    def _relationship_to_dict(self, r: CopyRelationship) -> Dict[str, Any]:
        """Convert relationship to dict for storage."""
        return {
            "id": r.id,
            "userId": r.user_id,
            "followerAddress": r.follower_address,
            "followerChain": r.follower_chain,
            "config": {
                "leaderAddress": r.config.leader_address,
                "leaderChain": r.config.leader_chain,
                "leaderLabel": r.config.leader_label,
                "sizingMode": r.config.sizing_mode.value,
                "sizeValue": str(r.config.size_value),
                "minTradeUsd": str(r.config.min_trade_usd),
                "maxTradeUsd": str(r.config.max_trade_usd) if r.config.max_trade_usd else None,
                "tokenWhitelist": r.config.token_whitelist,
                "tokenBlacklist": r.config.token_blacklist,
                "allowedActions": r.config.allowed_actions,
                "delaySeconds": r.config.delay_seconds,
                "maxDelaySeconds": r.config.max_delay_seconds,
                "maxSlippageBps": r.config.max_slippage_bps,
                "maxDailyTrades": r.config.max_daily_trades,
                "maxDailyVolumeUsd": str(r.config.max_daily_volume_usd),
                "sessionKeyId": r.config.session_key_id,
            },
            "isActive": r.is_active,
            "isPaused": r.is_paused,
            "pauseReason": r.pause_reason,
            "dailyTradeCount": r.daily_trade_count,
            "dailyVolumeUsd": str(r.daily_volume_usd),
            "dailyResetAt": int(r.daily_reset_at.timestamp() * 1000),
            "totalTrades": r.total_trades,
            "successfulTrades": r.successful_trades,
            "failedTrades": r.failed_trades,
            "skippedTrades": r.skipped_trades,
            "totalVolumeUsd": str(r.total_volume_usd),
            "totalPnlUsd": str(r.total_pnl_usd) if r.total_pnl_usd else None,
            "createdAt": int(r.created_at.timestamp() * 1000),
            "updatedAt": int(r.updated_at.timestamp() * 1000),
            "lastCopyAt": int(r.last_copy_at.timestamp() * 1000) if r.last_copy_at else None,
        }

    def _dict_to_relationship(self, d: Dict[str, Any]) -> CopyRelationship:
        """Convert dict from storage to relationship."""
        config_data = d.get("config", {})
        config = CopyConfig(
            leader_address=config_data.get("leaderAddress", ""),
            leader_chain=config_data.get("leaderChain", ""),
            leader_label=config_data.get("leaderLabel"),
            sizing_mode=SizingMode(config_data.get("sizingMode", "percentage")),
            size_value=Decimal(config_data.get("sizeValue", "5")),
            min_trade_usd=Decimal(config_data.get("minTradeUsd", "10")),
            max_trade_usd=Decimal(config_data["maxTradeUsd"]) if config_data.get("maxTradeUsd") else None,
            token_whitelist=config_data.get("tokenWhitelist"),
            token_blacklist=config_data.get("tokenBlacklist"),
            allowed_actions=config_data.get("allowedActions", ["swap"]),
            delay_seconds=config_data.get("delaySeconds", 0),
            max_delay_seconds=config_data.get("maxDelaySeconds", 300),
            max_slippage_bps=config_data.get("maxSlippageBps", 100),
            max_daily_trades=config_data.get("maxDailyTrades", 20),
            max_daily_volume_usd=Decimal(config_data.get("maxDailyVolumeUsd", "10000")),
            session_key_id=config_data.get("sessionKeyId"),
        )

        return CopyRelationship(
            id=d["id"],
            user_id=d["userId"],
            follower_address=d["followerAddress"],
            follower_chain=d["followerChain"],
            config=config,
            is_active=d.get("isActive", True),
            is_paused=d.get("isPaused", False),
            pause_reason=d.get("pauseReason"),
            daily_trade_count=d.get("dailyTradeCount", 0),
            daily_volume_usd=Decimal(d.get("dailyVolumeUsd", "0")),
            daily_reset_at=datetime.fromtimestamp(d["dailyResetAt"] / 1000, tz=timezone.utc) if d.get("dailyResetAt") else datetime.now(timezone.utc),
            total_trades=d.get("totalTrades", 0),
            successful_trades=d.get("successfulTrades", 0),
            failed_trades=d.get("failedTrades", 0),
            skipped_trades=d.get("skippedTrades", 0),
            total_volume_usd=Decimal(d.get("totalVolumeUsd", "0")),
            total_pnl_usd=Decimal(d["totalPnlUsd"]) if d.get("totalPnlUsd") else None,
            created_at=datetime.fromtimestamp(d["createdAt"] / 1000, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(d["updatedAt"] / 1000, tz=timezone.utc),
            last_copy_at=datetime.fromtimestamp(d["lastCopyAt"] / 1000, tz=timezone.utc) if d.get("lastCopyAt") else None,
        )

    def _execution_to_dict(self, e: CopyExecution) -> Dict[str, Any]:
        """Convert execution to dict for storage."""
        return {
            "id": e.id,
            "relationshipId": e.relationship_id,
            "signal": {
                "leaderAddress": e.signal.leader_address,
                "leaderChain": e.signal.leader_chain,
                "txHash": e.signal.tx_hash,
                "blockNumber": e.signal.block_number,
                "timestamp": int(e.signal.timestamp.timestamp() * 1000),
                "action": e.signal.action,
                "tokenInAddress": e.signal.token_in_address,
                "tokenInSymbol": e.signal.token_in_symbol,
                "tokenInAmount": str(e.signal.token_in_amount),
                "tokenOutAddress": e.signal.token_out_address,
                "tokenOutSymbol": e.signal.token_out_symbol,
                "tokenOutAmount": str(e.signal.token_out_amount) if e.signal.token_out_amount else None,
                "valueUsd": str(e.signal.value_usd) if e.signal.value_usd else None,
                "dex": e.signal.dex,
            },
            "status": e.status.value,
            "skipReason": e.skip_reason.value if e.skip_reason else None,
            "calculatedSizeUsd": str(e.calculated_size_usd) if e.calculated_size_usd else None,
            "actualSizeUsd": str(e.actual_size_usd) if e.actual_size_usd else None,
            "txHash": e.tx_hash,
            "gasUsed": e.gas_used,
            "gasPriceGwei": str(e.gas_price_gwei) if e.gas_price_gwei else None,
            "gasCostUsd": str(e.gas_cost_usd) if e.gas_cost_usd else None,
            "tokenOutAmount": str(e.token_out_amount) if e.token_out_amount else None,
            "slippageBps": e.slippage_bps,
            "errorMessage": e.error_message,
            "signalReceivedAt": int(e.signal_received_at.timestamp() * 1000),
            "executionStartedAt": int(e.execution_started_at.timestamp() * 1000) if e.execution_started_at else None,
            "executionCompletedAt": int(e.execution_completed_at.timestamp() * 1000) if e.execution_completed_at else None,
        }

    def _dict_to_execution(self, d: Dict[str, Any]) -> CopyExecution:
        """Convert dict from storage to execution."""
        signal_data = d.get("signal", {})
        signal = TradeSignal(
            leader_address=signal_data.get("leaderAddress", ""),
            leader_chain=signal_data.get("leaderChain", ""),
            tx_hash=signal_data.get("txHash", ""),
            block_number=signal_data.get("blockNumber", 0),
            timestamp=datetime.fromtimestamp(signal_data.get("timestamp", 0) / 1000, tz=timezone.utc),
            action=signal_data.get("action", "swap"),
            token_in_address=signal_data.get("tokenInAddress", ""),
            token_in_symbol=signal_data.get("tokenInSymbol"),
            token_in_amount=Decimal(signal_data.get("tokenInAmount", "0")),
            token_out_address=signal_data.get("tokenOutAddress", ""),
            token_out_symbol=signal_data.get("tokenOutSymbol"),
            token_out_amount=Decimal(signal_data["tokenOutAmount"]) if signal_data.get("tokenOutAmount") else None,
            value_usd=Decimal(signal_data["valueUsd"]) if signal_data.get("valueUsd") else None,
            dex=signal_data.get("dex"),
        )

        return CopyExecution(
            id=d["id"],
            relationship_id=d["relationshipId"],
            signal=signal,
            status=CopyExecutionStatus(d.get("status", "pending")),
            skip_reason=SkipReason(d["skipReason"]) if d.get("skipReason") else None,
            calculated_size_usd=Decimal(d["calculatedSizeUsd"]) if d.get("calculatedSizeUsd") else None,
            actual_size_usd=Decimal(d["actualSizeUsd"]) if d.get("actualSizeUsd") else None,
            tx_hash=d.get("txHash"),
            gas_used=d.get("gasUsed"),
            gas_price_gwei=Decimal(d["gasPriceGwei"]) if d.get("gasPriceGwei") else None,
            gas_cost_usd=Decimal(d["gasCostUsd"]) if d.get("gasCostUsd") else None,
            token_out_amount=Decimal(d["tokenOutAmount"]) if d.get("tokenOutAmount") else None,
            slippage_bps=d.get("slippageBps"),
            error_message=d.get("errorMessage"),
            signal_received_at=datetime.fromtimestamp(d["signalReceivedAt"] / 1000, tz=timezone.utc),
            execution_started_at=datetime.fromtimestamp(d["executionStartedAt"] / 1000, tz=timezone.utc) if d.get("executionStartedAt") else None,
            execution_completed_at=datetime.fromtimestamp(d["executionCompletedAt"] / 1000, tz=timezone.utc) if d.get("executionCompletedAt") else None,
        )
