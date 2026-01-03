"""
DCA Executor

Executes DCA strategy buys with constraint checking and session key validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from .models import (
    DCAStrategy,
    DCAExecution,
    ExecutionStatus,
    MarketConditions,
    ExecutionQuote,
    SkipReason,
)
from .scheduler import DCAScheduler

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a DCA execution attempt."""
    success: bool
    status: ExecutionStatus
    skip_reason: Optional[SkipReason] = None
    tx_hash: Optional[str] = None
    input_amount: Optional[Decimal] = None
    output_amount: Optional[Decimal] = None
    price_usd: Optional[Decimal] = None
    gas_usd: Optional[Decimal] = None
    error_message: Optional[str] = None
    next_execution_at: Optional[datetime] = None


class DCAExecutor:
    """
    Executes DCA strategy buys.

    Responsibilities:
    1. Validate session key is still active
    2. Check market constraints (gas, price limits)
    3. Get swap quote
    4. Execute swap via transaction executor
    5. Update strategy stats
    """

    def __init__(
        self,
        convex_client: Any,
        swap_provider: Any,
        pricing_provider: Any,
        gas_provider: Any,
        session_manager: Any,
        policy_engine: Any,
        tx_executor: Optional[Any] = None,
    ):
        """
        Initialize DCA executor.

        Args:
            convex_client: Convex client for database operations
            swap_provider: Swap quote provider (1inch, Jupiter)
            pricing_provider: Token price provider
            gas_provider: Gas price provider
            session_manager: Session key manager
            policy_engine: Policy engine for validation
            tx_executor: Transaction executor (optional, for actual execution)
        """
        self._convex = convex_client
        self._swap = swap_provider
        self._pricing = pricing_provider
        self._gas = gas_provider
        self._sessions = session_manager
        self._policy = policy_engine
        self._tx_executor = tx_executor

    async def execute(
        self,
        strategy: DCAStrategy,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute a single DCA buy.

        Args:
            strategy: DCA strategy to execute
            dry_run: If True, simulate but don't execute

        Returns:
            ExecutionResult with status and details
        """
        logger.info(f"Executing DCA strategy {strategy.id}: {strategy.name}")

        config = strategy.config
        execution_number = strategy.stats.total_executions + 1

        # Create execution record
        execution_id = await self._create_execution_record(
            strategy.id,
            execution_number,
            config.from_token.chain_id,
        )

        try:
            # 1. Validate session key
            session_valid, session_error = await self._validate_session_key(strategy)
            if not session_valid:
                return await self._skip_execution(
                    execution_id,
                    strategy,
                    SkipReason.SESSION_EXPIRED,
                    session_error,
                )

            # 2. Get market conditions
            market_conditions = await self._get_market_conditions(config)
            logger.info(
                f"Market conditions: price=${market_conditions.token_price_usd}, "
                f"gas={market_conditions.gas_gwei}gwei (${market_conditions.estimated_gas_usd})"
            )

            # 3. Check constraints
            skip_reason = self._check_constraints(config, market_conditions)
            if skip_reason:
                return await self._skip_execution(
                    execution_id,
                    strategy,
                    skip_reason,
                    f"Constraint check failed: {skip_reason.value}",
                    market_conditions,
                )

            # 4. Get swap quote
            quote = await self._get_swap_quote(config, market_conditions)
            if not quote:
                return await self._fail_execution(
                    execution_id,
                    strategy,
                    "Failed to get swap quote",
                )

            # Mark as running with quote
            await self._mark_running(execution_id, market_conditions, quote)

            # 5. Validate against policy engine
            policy_result = await self._validate_policy(strategy, quote)
            if not policy_result.approved:
                return await self._fail_execution(
                    execution_id,
                    strategy,
                    f"Policy rejected: {policy_result.violations}",
                )

            # 6. Execute swap (or simulate for dry run)
            if dry_run:
                logger.info(f"DRY RUN: Would execute swap for ${config.amount_per_execution_usd}")
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.COMPLETED,
                    input_amount=quote.input_amount,
                    output_amount=quote.expected_output_amount,
                    price_usd=market_conditions.token_price_usd,
                    gas_usd=market_conditions.estimated_gas_usd,
                    next_execution_at=DCAScheduler.get_next_execution(config),
                )

            tx_result = await self._execute_swap(strategy, quote)
            if not tx_result.success:
                return await self._fail_execution(
                    execution_id,
                    strategy,
                    tx_result.error or "Swap execution failed",
                )

            # 7. Record success
            return await self._complete_execution(
                execution_id,
                strategy,
                tx_result,
                market_conditions,
            )

        except Exception as e:
            logger.exception(f"DCA execution error: {e}")
            return await self._fail_execution(
                execution_id,
                strategy,
                str(e),
            )

    async def _validate_session_key(
        self,
        strategy: DCAStrategy,
    ) -> Tuple[bool, Optional[str]]:
        """Validate the strategy's session key is still active."""
        if not strategy.session_key_id:
            return False, "No session key configured"

        try:
            # Check session key status
            session_key = await self._convex.query(
                "sessionKeys:get",
                {"sessionId": strategy.session_key_id},
            )

            if not session_key:
                return False, "Session key not found"

            if session_key.get("status") != "active":
                return False, f"Session key is {session_key.get('status')}"

            # Check expiry
            expires_at = session_key.get("expiresAt", 0)
            if expires_at < datetime.utcnow().timestamp() * 1000:
                return False, "Session key expired"

            # Check value limits
            value_limits = session_key.get("valueLimits", {})
            total_used = Decimal(value_limits.get("totalValueUsedUsd", "0"))
            max_total = Decimal(value_limits.get("maxTotalValueUsd", "0"))

            if max_total > 0 and total_used >= max_total:
                return False, "Session key value limit exhausted"

            return True, None

        except Exception as e:
            logger.error(f"Session key validation error: {e}")
            return False, str(e)

    async def _get_market_conditions(
        self,
        config: "DCAConfig",
    ) -> MarketConditions:
        """Fetch current market conditions."""
        # Get token price
        token_price = await self._pricing.get_price(
            config.to_token.address,
            config.to_token.chain_id,
        )

        # Get gas price
        gas_price = await self._gas.get_gas_price(config.from_token.chain_id)

        # Estimate gas cost in USD
        # Typical swap uses ~150k gas
        estimated_gas_units = 150_000
        eth_price = await self._pricing.get_eth_price(config.from_token.chain_id)
        estimated_gas_usd = (
            Decimal(str(gas_price)) *
            Decimal(str(estimated_gas_units)) *
            Decimal(str(eth_price)) /
            Decimal("1e9")  # Convert gwei to ETH
        )

        return MarketConditions(
            token_price_usd=Decimal(str(token_price)),
            gas_gwei=Decimal(str(gas_price)),
            estimated_gas_usd=estimated_gas_usd,
        )

    def _check_constraints(
        self,
        config: "DCAConfig",
        conditions: MarketConditions,
    ) -> Optional[SkipReason]:
        """Check if constraints allow execution."""
        # Check gas limit
        if config.skip_if_gas_above_usd:
            if conditions.estimated_gas_usd > config.skip_if_gas_above_usd:
                logger.info(
                    f"Skipping: gas ${conditions.estimated_gas_usd} > limit ${config.skip_if_gas_above_usd}"
                )
                return SkipReason.GAS_TOO_HIGH

        # Check price upper bound
        if config.pause_if_price_above_usd:
            if conditions.token_price_usd > config.pause_if_price_above_usd:
                logger.info(
                    f"Skipping: price ${conditions.token_price_usd} > limit ${config.pause_if_price_above_usd}"
                )
                return SkipReason.PRICE_ABOVE_LIMIT

        # Check price lower bound
        if config.pause_if_price_below_usd:
            if conditions.token_price_usd < config.pause_if_price_below_usd:
                logger.info(
                    f"Skipping: price ${conditions.token_price_usd} < limit ${config.pause_if_price_below_usd}"
                )
                return SkipReason.PRICE_BELOW_LIMIT

        return None

    async def _get_swap_quote(
        self,
        config: "DCAConfig",
        conditions: MarketConditions,
    ) -> Optional[ExecutionQuote]:
        """Get swap quote for the DCA amount."""
        try:
            # Calculate input amount in token units
            # For stablecoin input, amount_per_execution_usd is the amount
            input_amount = config.amount_per_execution_usd

            # Get quote from swap provider
            quote = await self._swap.get_quote(
                from_token=config.from_token.address,
                to_token=config.to_token.address,
                amount=str(int(input_amount * (10 ** config.from_token.decimals))),
                chain_id=config.from_token.chain_id,
                slippage_bps=config.max_slippage_bps,
            )

            if not quote:
                return None

            # Calculate minimum output with slippage
            expected_output = Decimal(quote.get("toAmount", "0")) / (10 ** config.to_token.decimals)
            min_output = expected_output * (1 - Decimal(config.max_slippage_bps) / 10000)

            return ExecutionQuote(
                input_amount=input_amount,
                expected_output_amount=expected_output,
                minimum_output_amount=min_output,
                price_impact_bps=quote.get("priceImpactBps", 0),
                route=quote.get("routeDescription"),
            )

        except Exception as e:
            logger.error(f"Failed to get swap quote: {e}")
            return None

    async def _validate_policy(
        self,
        strategy: DCAStrategy,
        quote: ExecutionQuote,
    ) -> Any:
        """Validate execution against policy engine."""
        from app.core.policy import ActionContext, PolicyEngine

        context = ActionContext(
            action_type="swap",
            wallet_address=strategy.wallet_address,
            chain_id=strategy.config.from_token.chain_id,
            value_usd=float(quote.input_amount),
            from_token=strategy.config.from_token.symbol,
            to_token=strategy.config.to_token.symbol,
            session_key_id=strategy.session_key_id,
        )

        return await self._policy.evaluate(context)

    async def _execute_swap(
        self,
        strategy: DCAStrategy,
        quote: ExecutionQuote,
    ) -> "SwapResult":
        """Execute the actual swap transaction."""
        if not self._tx_executor:
            # Return mock success if no executor
            return SwapResult(
                success=True,
                tx_hash="0x" + "0" * 64,
                input_amount=quote.input_amount,
                output_amount=quote.expected_output_amount,
            )

        # Build and execute swap transaction
        config = strategy.config

        result = await self._tx_executor.execute_swap(
            from_token=config.from_token.address,
            to_token=config.to_token.address,
            amount=str(int(quote.input_amount * (10 ** config.from_token.decimals))),
            min_amount_out=str(int(quote.minimum_output_amount * (10 ** config.to_token.decimals))),
            wallet_address=strategy.wallet_address,
            chain_id=config.from_token.chain_id,
        )

        return result

    async def _create_execution_record(
        self,
        strategy_id: str,
        execution_number: int,
        chain_id: int,
    ) -> str:
        """Create a new execution record in Convex."""
        execution_id = await self._convex.mutation(
            "dca:createExecution",
            {
                "strategyId": strategy_id,
                "executionNumber": execution_number,
                "scheduledAt": int(datetime.utcnow().timestamp() * 1000),
                "chainId": chain_id,
            },
        )
        return execution_id

    async def _mark_running(
        self,
        execution_id: str,
        conditions: MarketConditions,
        quote: ExecutionQuote,
    ):
        """Mark execution as running with quote details."""
        await self._convex.mutation(
            "dca:markExecutionRunning",
            {
                "executionId": execution_id,
                "marketConditions": conditions.to_dict(),
                "quote": quote.to_dict(),
            },
        )

    async def _skip_execution(
        self,
        execution_id: str,
        strategy: DCAStrategy,
        reason: SkipReason,
        message: str,
        conditions: Optional[MarketConditions] = None,
    ) -> ExecutionResult:
        """Record a skipped execution."""
        logger.info(f"Skipping execution: {reason.value} - {message}")

        await self._convex.mutation(
            "dca:markExecutionSkipped",
            {
                "executionId": execution_id,
                "skipReason": reason.value,
                "marketConditions": conditions.to_dict() if conditions else None,
            },
        )

        # Calculate next execution
        next_execution = DCAScheduler.get_next_execution(strategy.config)
        await self._convex.mutation(
            "dca:updateNextExecution",
            {
                "strategyId": strategy.id,
                "nextExecutionAt": int(next_execution.timestamp() * 1000),
            },
        )

        return ExecutionResult(
            success=False,
            status=ExecutionStatus.SKIPPED,
            skip_reason=reason,
            error_message=message,
            next_execution_at=next_execution,
        )

    async def _fail_execution(
        self,
        execution_id: str,
        strategy: DCAStrategy,
        error: str,
    ) -> ExecutionResult:
        """Record a failed execution."""
        logger.error(f"Execution failed: {error}")

        await self._convex.mutation(
            "dca:markExecutionFailed",
            {
                "executionId": execution_id,
                "errorMessage": error,
            },
        )

        # Calculate next execution (still schedule next one)
        next_execution = DCAScheduler.get_next_execution(strategy.config)
        await self._convex.mutation(
            "dca:updateNextExecution",
            {
                "strategyId": strategy.id,
                "nextExecutionAt": int(next_execution.timestamp() * 1000),
            },
        )

        return ExecutionResult(
            success=False,
            status=ExecutionStatus.FAILED,
            error_message=error,
            next_execution_at=next_execution,
        )

    async def _complete_execution(
        self,
        execution_id: str,
        strategy: DCAStrategy,
        tx_result: "SwapResult",
        conditions: MarketConditions,
    ) -> ExecutionResult:
        """Record a successful execution."""
        logger.info(f"Execution completed: tx={tx_result.tx_hash}")

        await self._convex.mutation(
            "dca:markExecutionCompleted",
            {
                "executionId": execution_id,
                "txHash": tx_result.tx_hash,
                "actualInputAmount": str(tx_result.input_amount),
                "actualOutputAmount": str(tx_result.output_amount),
                "actualPriceUsd": float(conditions.token_price_usd),
                "gasUsed": tx_result.gas_used or 0,
                "gasPriceGwei": float(conditions.gas_gwei),
                "gasUsd": float(conditions.estimated_gas_usd),
            },
        )

        # Calculate next execution
        next_execution = DCAScheduler.get_next_execution(strategy.config)
        await self._convex.mutation(
            "dca:updateNextExecution",
            {
                "strategyId": strategy.id,
                "nextExecutionAt": int(next_execution.timestamp() * 1000),
            },
        )

        return ExecutionResult(
            success=True,
            status=ExecutionStatus.COMPLETED,
            tx_hash=tx_result.tx_hash,
            input_amount=tx_result.input_amount,
            output_amount=tx_result.output_amount,
            price_usd=conditions.token_price_usd,
            gas_usd=conditions.estimated_gas_usd,
            next_execution_at=next_execution,
        )


@dataclass
class SwapResult:
    """Result from swap execution."""
    success: bool
    tx_hash: Optional[str] = None
    input_amount: Optional[Decimal] = None
    output_amount: Optional[Decimal] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None
