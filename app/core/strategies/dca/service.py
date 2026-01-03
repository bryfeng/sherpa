"""
DCA Service

High-level service for managing DCA strategies.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .models import (
    DCAStrategy,
    DCAConfig,
    DCAExecution,
    DCAStatus,
    TokenInfo,
    SessionKeyRequirements,
)
from .scheduler import DCAScheduler
from .executor import DCAExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class DCAService:
    """
    Service for managing DCA strategies.

    Provides high-level operations for:
    - Creating and configuring strategies
    - Activating with session keys
    - Pausing/resuming/stopping
    - Viewing stats and history
    """

    def __init__(
        self,
        convex_client: Any,
        swap_provider: Any = None,
        pricing_provider: Any = None,
        gas_provider: Any = None,
        session_manager: Any = None,
        policy_engine: Any = None,
    ):
        """
        Initialize DCA service.

        Args:
            convex_client: Convex client for database operations
            swap_provider: Optional swap provider for execution
            pricing_provider: Optional pricing provider
            gas_provider: Optional gas price provider
            session_manager: Optional session key manager
            policy_engine: Optional policy engine
        """
        self._convex = convex_client
        self._swap = swap_provider
        self._pricing = pricing_provider
        self._gas = gas_provider
        self._sessions = session_manager
        self._policy = policy_engine

        # Create executor if we have required providers
        self._executor = None
        if all([swap_provider, pricing_provider, gas_provider, session_manager, policy_engine]):
            self._executor = DCAExecutor(
                convex_client=convex_client,
                swap_provider=swap_provider,
                pricing_provider=pricing_provider,
                gas_provider=gas_provider,
                session_manager=session_manager,
                policy_engine=policy_engine,
            )

    # =========================================================================
    # Strategy CRUD
    # =========================================================================

    async def create_strategy(
        self,
        user_id: str,
        wallet_id: str,
        wallet_address: str,
        name: str,
        config: DCAConfig,
        description: Optional[str] = None,
    ) -> DCAStrategy:
        """
        Create a new DCA strategy (in draft status).

        Args:
            user_id: Convex user ID
            wallet_id: Convex wallet ID
            wallet_address: Ethereum wallet address
            name: Strategy name
            config: DCA configuration
            description: Optional description

        Returns:
            Created DCAStrategy

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate schedule
        schedule_error = DCAScheduler.validate_schedule(config)
        if schedule_error:
            raise ValueError(schedule_error)

        # Validate amounts
        if config.amount_per_execution_usd <= 0:
            raise ValueError("Amount per execution must be positive")

        if config.max_slippage_bps < 0 or config.max_slippage_bps > 1000:
            raise ValueError("Slippage must be between 0 and 1000 bps (10%)")

        # Create in Convex
        strategy_id = await self._convex.mutation(
            "dca:create",
            {
                "userId": user_id,
                "walletId": wallet_id,
                "walletAddress": wallet_address.lower(),
                "name": name,
                "description": description,
                "fromToken": config.from_token.to_dict(),
                "toToken": config.to_token.to_dict(),
                "amountPerExecutionUsd": float(config.amount_per_execution_usd),
                "frequency": config.frequency.value,
                "cronExpression": config.cron_expression,
                "executionHourUtc": config.execution_hour_utc,
                "executionDayOfWeek": config.execution_day_of_week,
                "executionDayOfMonth": config.execution_day_of_month,
                "maxSlippageBps": config.max_slippage_bps,
                "maxGasUsd": float(config.max_gas_usd),
                "skipIfGasAboveUsd": float(config.skip_if_gas_above_usd) if config.skip_if_gas_above_usd else None,
                "pauseIfPriceAboveUsd": float(config.pause_if_price_above_usd) if config.pause_if_price_above_usd else None,
                "pauseIfPriceBelowUsd": float(config.pause_if_price_below_usd) if config.pause_if_price_below_usd else None,
                "maxTotalSpendUsd": float(config.max_total_spend_usd) if config.max_total_spend_usd else None,
                "maxExecutions": config.max_executions,
                "endDate": int(config.end_date.timestamp() * 1000) if config.end_date else None,
            },
        )

        logger.info(f"Created DCA strategy {strategy_id}: {name}")

        # Fetch and return the created strategy
        return await self.get_strategy(strategy_id)

    async def get_strategy(self, strategy_id: str) -> Optional[DCAStrategy]:
        """Get a strategy by ID."""
        data = await self._convex.query("dca:get", {"id": strategy_id})
        if not data:
            return None
        return DCAStrategy.from_convex(data)

    async def list_strategies(
        self,
        user_id: Optional[str] = None,
        wallet_address: Optional[str] = None,
        status: Optional[DCAStatus] = None,
    ) -> List[DCAStrategy]:
        """
        List strategies filtered by user, wallet, or status.

        Args:
            user_id: Filter by user ID
            wallet_address: Filter by wallet address
            status: Filter by status

        Returns:
            List of matching strategies
        """
        if user_id:
            data = await self._convex.query(
                "dca:listByUser",
                {
                    "userId": user_id,
                    "status": status.value if status else None,
                },
            )
        elif wallet_address:
            data = await self._convex.query(
                "dca:listByWallet",
                {
                    "walletAddress": wallet_address.lower(),
                    "status": status.value if status else None,
                },
            )
        else:
            raise ValueError("Either user_id or wallet_address must be provided")

        return [DCAStrategy.from_convex(d) for d in data]

    async def update_config(
        self,
        strategy_id: str,
        **updates,
    ) -> DCAStrategy:
        """
        Update strategy configuration.

        Only allowed when strategy is paused or draft.

        Args:
            strategy_id: Strategy ID
            **updates: Configuration updates

        Returns:
            Updated strategy

        Raises:
            ValueError: If strategy cannot be updated
        """
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError("Strategy not found")

        if strategy.status not in [DCAStatus.DRAFT, DCAStatus.PAUSED]:
            raise ValueError(
                f"Cannot update strategy in {strategy.status.value} status. "
                "Pause it first."
            )

        # Map Python field names to Convex field names
        field_mapping = {
            "amount_per_execution_usd": "amountPerExecutionUsd",
            "frequency": "frequency",
            "cron_expression": "cronExpression",
            "execution_hour_utc": "executionHourUtc",
            "execution_day_of_week": "executionDayOfWeek",
            "execution_day_of_month": "executionDayOfMonth",
            "max_slippage_bps": "maxSlippageBps",
            "max_gas_usd": "maxGasUsd",
            "skip_if_gas_above_usd": "skipIfGasAboveUsd",
            "pause_if_price_above_usd": "pauseIfPriceAboveUsd",
            "pause_if_price_below_usd": "pauseIfPriceBelowUsd",
            "max_total_spend_usd": "maxTotalSpendUsd",
            "max_executions": "maxExecutions",
            "end_date": "endDate",
        }

        convex_updates = {"strategyId": strategy_id}
        for key, value in updates.items():
            if key in field_mapping and value is not None:
                convex_key = field_mapping[key]
                if isinstance(value, Decimal):
                    value = float(value)
                elif isinstance(value, datetime):
                    value = int(value.timestamp() * 1000)
                elif hasattr(value, "value"):  # Enum
                    value = value.value
                convex_updates[convex_key] = value

        await self._convex.mutation("dca:updateConfig", convex_updates)

        return await self.get_strategy(strategy_id)

    # =========================================================================
    # Strategy Lifecycle
    # =========================================================================

    async def activate(
        self,
        strategy_id: str,
        session_key_id: str,
    ) -> DCAStrategy:
        """
        Activate a strategy with a session key.

        Args:
            strategy_id: Strategy to activate
            session_key_id: Session key ID for autonomous execution

        Returns:
            Activated strategy

        Raises:
            ValueError: If strategy cannot be activated
        """
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError("Strategy not found")

        if strategy.status not in [DCAStatus.DRAFT, DCAStatus.PENDING_SESSION]:
            raise ValueError(f"Cannot activate strategy in {strategy.status.value} status")

        # Validate session key
        session_key = await self._convex.query(
            "sessionKeys:get",
            {"sessionId": session_key_id},
        )
        if not session_key:
            raise ValueError("Session key not found")
        if session_key.get("status") != "active":
            raise ValueError(f"Session key is {session_key.get('status')}")

        # Calculate first execution time
        next_execution = DCAScheduler.get_next_execution(strategy.config)

        await self._convex.mutation(
            "dca:activate",
            {
                "strategyId": strategy_id,
                "sessionKeyId": session_key_id,
                "nextExecutionAt": int(next_execution.timestamp() * 1000),
            },
        )

        logger.info(
            f"Activated DCA strategy {strategy_id}, "
            f"next execution at {next_execution}"
        )

        return await self.get_strategy(strategy_id)

    async def pause(
        self,
        strategy_id: str,
        reason: Optional[str] = None,
    ) -> DCAStrategy:
        """Pause an active strategy."""
        await self._convex.mutation(
            "dca:pause",
            {
                "strategyId": strategy_id,
                "reason": reason,
            },
        )

        logger.info(f"Paused DCA strategy {strategy_id}: {reason}")

        return await self.get_strategy(strategy_id)

    async def resume(
        self,
        strategy_id: str,
    ) -> DCAStrategy:
        """Resume a paused strategy."""
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError("Strategy not found")

        # Calculate next execution time
        next_execution = DCAScheduler.get_next_execution(strategy.config)

        await self._convex.mutation(
            "dca:resume",
            {
                "strategyId": strategy_id,
                "nextExecutionAt": int(next_execution.timestamp() * 1000),
            },
        )

        logger.info(f"Resumed DCA strategy {strategy_id}")

        return await self.get_strategy(strategy_id)

    async def stop(self, strategy_id: str) -> DCAStrategy:
        """Stop/complete a strategy."""
        await self._convex.mutation("dca:stop", {"strategyId": strategy_id})

        logger.info(f"Stopped DCA strategy {strategy_id}")

        return await self.get_strategy(strategy_id)

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute_now(
        self,
        strategy_id: str,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute a strategy immediately.

        Args:
            strategy_id: Strategy to execute
            dry_run: If True, simulate but don't execute

        Returns:
            Execution result

        Raises:
            ValueError: If executor not configured or strategy not found
        """
        if not self._executor:
            raise ValueError("Executor not configured")

        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError("Strategy not found")

        if strategy.status != DCAStatus.ACTIVE:
            raise ValueError(f"Cannot execute strategy in {strategy.status.value} status")

        return await self._executor.execute(strategy, dry_run=dry_run)

    async def get_executions(
        self,
        strategy_id: str,
        limit: int = 50,
    ) -> List[DCAExecution]:
        """Get execution history for a strategy."""
        data = await self._convex.query(
            "dca:getExecutions",
            {
                "strategyId": strategy_id,
                "limit": limit,
            },
        )
        return [DCAExecution.from_convex(d) for d in data]

    # =========================================================================
    # Session Key Helpers
    # =========================================================================

    def get_session_key_requirements(
        self,
        config: DCAConfig,
    ) -> SessionKeyRequirements:
        """
        Get session key requirements for a DCA configuration.

        Returns requirements that can be used to create/validate a session key.
        """
        executions_estimate = DCAScheduler.get_estimated_executions_per_year(config)
        return SessionKeyRequirements.for_dca_strategy(config, executions_estimate)

    def format_schedule_description(self, config: DCAConfig) -> str:
        """Get human-readable schedule description."""
        return DCAScheduler.format_schedule_description(config)

    # =========================================================================
    # Stats and Reporting
    # =========================================================================

    async def get_stats(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy statistics."""
        data = await self._convex.query("dca:getStats", {"strategyId": strategy_id})
        return data

    async def calculate_performance(
        self,
        strategy_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate performance metrics for a strategy.

        Returns metrics like ROI, vs buy-and-hold, etc.
        """
        strategy = await self.get_strategy(strategy_id)
        if not strategy or strategy.stats.total_executions == 0:
            return None

        stats = strategy.stats

        # Get current token price
        current_price = Decimal("0")
        if self._pricing:
            current_price = Decimal(
                str(
                    await self._pricing.get_price(
                        strategy.config.to_token.address,
                        strategy.config.to_token.chain_id,
                    )
                )
            )

        # Calculate metrics
        total_invested = stats.total_amount_spent_usd
        tokens_held = stats.total_tokens_acquired
        current_value = tokens_held * current_price if current_price else Decimal("0")

        roi = ((current_value - total_invested) / total_invested * 100) if total_invested > 0 else Decimal("0")

        return {
            "totalInvestedUsd": float(total_invested),
            "tokensAcquired": str(tokens_held),
            "averageBuyPriceUsd": float(stats.average_price_usd) if stats.average_price_usd else None,
            "currentPriceUsd": float(current_price),
            "currentValueUsd": float(current_value),
            "roiPercent": float(roi),
            "totalExecutions": stats.total_executions,
            "successRate": (
                stats.successful_executions / stats.total_executions * 100
                if stats.total_executions > 0
                else 0
            ),
        }
