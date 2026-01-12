"""
Generic Strategy Executor - Phase 13.5

Executes approved strategies from the generic strategies table.
For Phase 1 (manual approval), this prepares transactions for user signing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status values."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    AWAITING_SIGNATURE = "awaiting_signature"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of a strategy execution attempt."""
    success: bool
    status: ExecutionStatus
    execution_id: str
    strategy_id: str

    # Transaction details (for awaiting_signature)
    transaction: Optional[Dict[str, Any]] = None
    quote: Optional[Dict[str, Any]] = None

    # Completion details
    tx_hash: Optional[str] = None
    input_amount: Optional[str] = None
    output_amount: Optional[str] = None
    gas_used: Optional[str] = None

    # Error details
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    recoverable: bool = True

    # Metadata
    executed_at: Optional[datetime] = None


@dataclass
class SwapParams:
    """Parameters for a swap execution."""
    from_token: str
    to_token: str
    amount: str
    chain_id: int
    wallet_address: str
    max_slippage_bps: int = 100  # 1% default


class GenericStrategyExecutor:
    """
    Executes generic strategies from the strategies table.

    For Phase 1 (manual approval flow):
    1. Gets quote for the swap
    2. Builds unsigned transaction
    3. Returns transaction for frontend to sign
    4. After signing, monitors and records result

    For Phase 2 (session key flow):
    - Uses session key to sign and submit automatically
    """

    def __init__(
        self,
        convex_client: Any,
        swap_provider: Optional[Any] = None,
        pricing_provider: Optional[Any] = None,
        tx_builder: Optional[Any] = None,
    ):
        """
        Initialize generic executor.

        Args:
            convex_client: Convex client for database operations
            swap_provider: Swap quote provider (optional)
            pricing_provider: Token price provider (optional)
            tx_builder: Transaction builder (optional)
        """
        self._convex = convex_client
        self._swap = swap_provider
        self._pricing = pricing_provider
        self._tx_builder = tx_builder

    async def prepare_execution(
        self,
        execution_id: str,
    ) -> ExecutionResult:
        """
        Prepare a strategy execution for user signing.

        This is called after the user approves an execution in chat.
        It gets a fresh quote and builds the transaction for signing.

        Args:
            execution_id: The execution ID to prepare

        Returns:
            ExecutionResult with transaction ready for signing
        """
        logger.info(f"Preparing execution: {execution_id}")

        try:
            # 1. Get execution and strategy details
            execution = await self._convex.query(
                "strategyExecutions:get",
                {"executionId": execution_id},
            )

            if not execution:
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    execution_id=execution_id,
                    strategy_id="",
                    error_message="Execution not found",
                    error_code="EXECUTION_NOT_FOUND",
                )

            strategy = execution.get("strategy", {})
            strategy_id = str(execution.get("strategyId", ""))
            wallet_address = execution.get("walletAddress", "")
            config = strategy.get("config", {})

            # 2. Extract swap parameters from strategy config
            swap_params = self._extract_swap_params(config, wallet_address)

            if not swap_params:
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    execution_id=execution_id,
                    strategy_id=strategy_id,
                    error_message="Could not extract swap parameters from strategy config",
                    error_code="INVALID_CONFIG",
                )

            # 3. Get quote (if swap provider available)
            quote = None
            if self._swap:
                try:
                    quote = await self._get_quote(swap_params)
                except Exception as e:
                    logger.warning(f"Quote fetch failed: {e}")
                    # Continue without quote - frontend can fetch

            # 4. Build transaction (if builder available)
            transaction = None
            if self._tx_builder and quote:
                try:
                    transaction = await self._build_transaction(swap_params, quote)
                except Exception as e:
                    logger.warning(f"Transaction build failed: {e}")

            # 5. Update execution state to indicate ready for signing
            await self._convex.mutation(
                "strategyExecutions:transitionState",
                {
                    "executionId": execution_id,
                    "toState": "executing",
                    "trigger": "prepared_for_signing",
                    "context": {
                        "quote": quote,
                        "swapParams": {
                            "fromToken": swap_params.from_token,
                            "toToken": swap_params.to_token,
                            "amount": swap_params.amount,
                            "chainId": swap_params.chain_id,
                        },
                    },
                },
            )

            return ExecutionResult(
                success=True,
                status=ExecutionStatus.AWAITING_SIGNATURE,
                execution_id=execution_id,
                strategy_id=strategy_id,
                transaction=transaction,
                quote=quote,
            )

        except Exception as e:
            logger.error(f"Error preparing execution {execution_id}: {e}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                execution_id=execution_id,
                strategy_id="",
                error_message=str(e),
                error_code="PREPARATION_ERROR",
            )

    async def record_completion(
        self,
        execution_id: str,
        tx_hash: str,
        output_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Record successful completion of an execution.

        Called after the frontend confirms transaction success.

        Args:
            execution_id: The execution ID
            tx_hash: The transaction hash
            output_data: Optional additional output data

        Returns:
            ExecutionResult with completion status
        """
        logger.info(f"Recording completion for execution {execution_id}: tx={tx_hash}")

        try:
            await self._convex.mutation(
                "strategyExecutions:complete",
                {
                    "executionId": execution_id,
                    "txHash": tx_hash,
                    "outputData": output_data,
                },
            )

            return ExecutionResult(
                success=True,
                status=ExecutionStatus.COMPLETED,
                execution_id=execution_id,
                strategy_id="",
                tx_hash=tx_hash,
                executed_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error recording completion for {execution_id}: {e}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                execution_id=execution_id,
                strategy_id="",
                error_message=str(e),
                error_code="COMPLETION_RECORD_ERROR",
            )

    async def record_failure(
        self,
        execution_id: str,
        error_message: str,
        error_code: Optional[str] = None,
        recoverable: bool = False,
    ) -> ExecutionResult:
        """
        Record failure of an execution.

        Args:
            execution_id: The execution ID
            error_message: Error description
            error_code: Optional error code
            recoverable: Whether the error is recoverable

        Returns:
            ExecutionResult with failure status
        """
        logger.warning(f"Recording failure for execution {execution_id}: {error_message}")

        try:
            await self._convex.mutation(
                "strategyExecutions:fail",
                {
                    "executionId": execution_id,
                    "errorMessage": error_message,
                    "errorCode": error_code,
                    "recoverable": recoverable,
                },
            )

            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                execution_id=execution_id,
                strategy_id="",
                error_message=error_message,
                error_code=error_code,
                recoverable=recoverable,
            )

        except Exception as e:
            logger.error(f"Error recording failure for {execution_id}: {e}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                execution_id=execution_id,
                strategy_id="",
                error_message=str(e),
                error_code="FAILURE_RECORD_ERROR",
            )

    def _extract_swap_params(
        self,
        config: Dict[str, Any],
        wallet_address: str,
    ) -> Optional[SwapParams]:
        """
        Extract swap parameters from strategy config.

        Handles different strategy types:
        - DCA: fromToken, toToken, amountPerExecution
        - Swap: fromToken, toToken, amount
        - Limit Order: fromToken, toToken, amount, triggerPrice
        """
        try:
            # Get token addresses/symbols
            from_token_config = config.get("fromToken", {})
            to_token_config = config.get("toToken", {})

            from_token = (
                from_token_config.get("address")
                or from_token_config.get("symbol")
                or config.get("fromTokenAddress")
            )
            to_token = (
                to_token_config.get("address")
                or to_token_config.get("symbol")
                or config.get("toTokenAddress")
            )

            if not from_token or not to_token:
                logger.warning("Missing from/to token in config")
                return None

            # Get amount
            amount = (
                config.get("amountPerExecution")
                or config.get("amountPerExecutionUsd")
                or config.get("amount")
                or config.get("amountUsd")
            )

            if not amount:
                logger.warning("Missing amount in config")
                return None

            # Get chain ID
            chain_id = (
                from_token_config.get("chainId")
                or config.get("chainId")
                or 1  # Default to Ethereum mainnet
            )

            # Get slippage
            max_slippage_bps = config.get("maxSlippageBps", 100)

            return SwapParams(
                from_token=str(from_token),
                to_token=str(to_token),
                amount=str(amount),
                chain_id=int(chain_id),
                wallet_address=wallet_address,
                max_slippage_bps=int(max_slippage_bps),
            )

        except Exception as e:
            logger.warning(f"Error extracting swap params: {e}")
            return None

    async def _get_quote(
        self,
        params: SwapParams,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a swap quote from the provider.

        Returns quote data or None if unavailable.
        """
        if not self._swap:
            return None

        # This would call the actual swap provider
        # For now, return a placeholder structure
        return {
            "fromToken": params.from_token,
            "toToken": params.to_token,
            "inputAmount": params.amount,
            "chainId": params.chain_id,
            "slippage": params.max_slippage_bps / 10000,
        }

    async def _build_transaction(
        self,
        params: SwapParams,
        quote: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Build an unsigned transaction from the quote.

        Returns transaction data for frontend signing.
        """
        if not self._tx_builder:
            return None

        # This would use the actual transaction builder
        # For now, return a placeholder structure
        return {
            "to": quote.get("routerAddress", ""),
            "data": quote.get("callData", ""),
            "value": "0",
            "chainId": params.chain_id,
            "from": params.wallet_address,
        }


# Convenience function to get executor instance
def get_generic_executor() -> GenericStrategyExecutor:
    """Get a configured GenericStrategyExecutor instance."""
    from ...db import get_convex_client

    return GenericStrategyExecutor(
        convex_client=get_convex_client(),
        # Providers can be added later
        swap_provider=None,
        pricing_provider=None,
        tx_builder=None,
    )
