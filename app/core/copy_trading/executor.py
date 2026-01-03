"""
Copy Executor

Executes copy trades on various chains and DEXs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

from .models import TradeSignal
from ..wallet.models import SessionKey

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a copy trade execution."""

    success: bool
    tx_hash: Optional[str] = None
    actual_value_usd: Optional[Decimal] = None
    token_out_amount: Optional[Decimal] = None
    slippage_bps: Optional[int] = None
    gas_used: Optional[int] = None
    gas_price_gwei: Optional[Decimal] = None
    gas_cost_usd: Optional[Decimal] = None
    error_message: Optional[str] = None


class CopyExecutor:
    """
    Execute copy trades.

    Supports multiple chains and DEXs:
    - Solana: Jupiter
    - EVM: Uniswap, 1inch, 0x
    """

    def __init__(
        self,
        jupiter_provider: Optional[Any] = None,
        evm_swap_provider: Optional[Any] = None,
        session_key_service: Optional[Any] = None,
    ):
        self.jupiter = jupiter_provider
        self.evm_swap = evm_swap_provider
        self.session_key_service = session_key_service

    async def execute(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int = 100,
        session_key: Optional[SessionKey] = None,
    ) -> ExecutionResult:
        """
        Execute a copy trade.

        Args:
            signal: Original trade signal from the leader
            size_usd: USD amount to trade
            follower_address: Address of the follower wallet
            follower_chain: Chain of the follower wallet
            max_slippage_bps: Maximum slippage in basis points
            session_key: Session key for autonomous execution

        Returns:
            Execution result
        """
        logger.info(
            f"Executing copy trade: {signal.action} {signal.token_in_symbol} -> "
            f"{signal.token_out_symbol}, size=${size_usd}"
        )

        try:
            # Determine execution path based on chain
            if follower_chain.lower() == "solana":
                return await self._execute_solana(
                    signal=signal,
                    size_usd=size_usd,
                    follower_address=follower_address,
                    max_slippage_bps=max_slippage_bps,
                    session_key=session_key,
                )
            else:
                return await self._execute_evm(
                    signal=signal,
                    size_usd=size_usd,
                    follower_address=follower_address,
                    follower_chain=follower_chain,
                    max_slippage_bps=max_slippage_bps,
                    session_key=session_key,
                )

        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )

    async def _execute_solana(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        max_slippage_bps: int,
        session_key: Optional[SessionKey] = None,
    ) -> ExecutionResult:
        """Execute a swap on Solana via Jupiter."""
        if not self.jupiter:
            return ExecutionResult(
                success=False,
                error_message="Jupiter provider not configured",
            )

        try:
            # Get quote from Jupiter
            quote = await self.jupiter.get_quote(
                input_mint=signal.token_in_address,
                output_mint=signal.token_out_address,
                amount_usd=float(size_usd),
                slippage_bps=max_slippage_bps,
            )

            if not quote:
                return ExecutionResult(
                    success=False,
                    error_message="Failed to get Jupiter quote",
                )

            # Check if session key allows this transaction
            if session_key and self.session_key_service:
                is_valid = await self.session_key_service.validate_transaction(
                    session_key=session_key,
                    action="swap",
                    value_usd=size_usd,
                    token_address=signal.token_out_address,
                )
                if not is_valid:
                    return ExecutionResult(
                        success=False,
                        error_message="Session key validation failed",
                    )

            # Build and send transaction
            tx_result = await self.jupiter.execute_swap(
                quote=quote,
                user_address=follower_address,
                # In production, this would use the session key to sign
            )

            if tx_result.get("success"):
                return ExecutionResult(
                    success=True,
                    tx_hash=tx_result.get("txHash"),
                    actual_value_usd=Decimal(str(tx_result.get("valueUsd", size_usd))),
                    token_out_amount=Decimal(str(tx_result.get("outputAmount", 0))),
                    slippage_bps=tx_result.get("actualSlippageBps"),
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message=tx_result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"Solana execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )

    async def _execute_evm(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int,
        session_key: Optional[SessionKey] = None,
    ) -> ExecutionResult:
        """Execute a swap on EVM chains."""
        if not self.evm_swap:
            return ExecutionResult(
                success=False,
                error_message="EVM swap provider not configured",
            )

        try:
            # Map chain to chain ID
            chain_id = self._chain_to_id(follower_chain)

            # Get quote
            quote = await self.evm_swap.get_quote(
                chain_id=chain_id,
                token_in=signal.token_in_address,
                token_out=signal.token_out_address,
                amount_usd=float(size_usd),
                slippage_bps=max_slippage_bps,
            )

            if not quote:
                return ExecutionResult(
                    success=False,
                    error_message="Failed to get EVM swap quote",
                )

            # Check session key
            if session_key and self.session_key_service:
                is_valid = await self.session_key_service.validate_transaction(
                    session_key=session_key,
                    action="swap",
                    value_usd=size_usd,
                    token_address=signal.token_out_address,
                    chain_id=chain_id,
                )
                if not is_valid:
                    return ExecutionResult(
                        success=False,
                        error_message="Session key validation failed",
                    )

            # Execute swap
            tx_result = await self.evm_swap.execute_swap(
                quote=quote,
                user_address=follower_address,
                chain_id=chain_id,
            )

            if tx_result.get("success"):
                return ExecutionResult(
                    success=True,
                    tx_hash=tx_result.get("txHash"),
                    actual_value_usd=Decimal(str(tx_result.get("valueUsd", size_usd))),
                    token_out_amount=Decimal(str(tx_result.get("outputAmount", 0))),
                    slippage_bps=tx_result.get("actualSlippageBps"),
                    gas_used=tx_result.get("gasUsed"),
                    gas_price_gwei=Decimal(str(tx_result.get("gasPriceGwei", 0))),
                    gas_cost_usd=Decimal(str(tx_result.get("gasCostUsd", 0))),
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message=tx_result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"EVM execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )

    def _chain_to_id(self, chain: str) -> int:
        """Convert chain name to chain ID."""
        chain_map = {
            "ethereum": 1,
            "polygon": 137,
            "arbitrum": 42161,
            "optimism": 10,
            "base": 8453,
            "avalanche": 43114,
            "bsc": 56,
        }
        return chain_map.get(chain.lower(), 1)

    async def estimate_execution(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_chain: str,
        max_slippage_bps: int = 100,
    ) -> Dict[str, Any]:
        """
        Estimate execution details without executing.

        Returns quote information, expected output, gas costs, etc.
        """
        try:
            if follower_chain.lower() == "solana":
                if not self.jupiter:
                    return {"error": "Jupiter not configured"}

                quote = await self.jupiter.get_quote(
                    input_mint=signal.token_in_address,
                    output_mint=signal.token_out_address,
                    amount_usd=float(size_usd),
                    slippage_bps=max_slippage_bps,
                )

                if quote:
                    return {
                        "input_amount": quote.get("inputAmount"),
                        "output_amount": quote.get("outputAmount"),
                        "price_impact_bps": quote.get("priceImpactBps"),
                        "routes": quote.get("routes", []),
                        "estimated_fees": quote.get("fees"),
                    }

            else:
                if not self.evm_swap:
                    return {"error": "EVM swap not configured"}

                chain_id = self._chain_to_id(follower_chain)
                quote = await self.evm_swap.get_quote(
                    chain_id=chain_id,
                    token_in=signal.token_in_address,
                    token_out=signal.token_out_address,
                    amount_usd=float(size_usd),
                    slippage_bps=max_slippage_bps,
                )

                if quote:
                    return {
                        "input_amount": quote.get("inputAmount"),
                        "output_amount": quote.get("outputAmount"),
                        "price_impact_bps": quote.get("priceImpactBps"),
                        "gas_estimate": quote.get("gasEstimate"),
                        "gas_cost_usd": quote.get("gasCostUsd"),
                    }

            return {"error": "Failed to get quote"}

        except Exception as e:
            logger.error(f"Estimation error: {e}")
            return {"error": str(e)}


class MockCopyExecutor(CopyExecutor):
    """
    Mock executor for testing.

    Simulates successful executions without actually trading.
    """

    def __init__(self, success_rate: float = 0.9):
        super().__init__()
        self.success_rate = success_rate
        self.executions: list = []

    async def execute(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int = 100,
        session_key: Optional[SessionKey] = None,
    ) -> ExecutionResult:
        """Mock execution."""
        import random

        success = random.random() < self.success_rate

        result = ExecutionResult(
            success=success,
            tx_hash=f"0x{''.join(random.choices('0123456789abcdef', k=64))}" if success else None,
            actual_value_usd=size_usd if success else None,
            token_out_amount=signal.token_out_amount if success else None,
            slippage_bps=random.randint(10, max_slippage_bps) if success else None,
            gas_used=random.randint(100000, 300000) if success and follower_chain != "solana" else None,
            gas_price_gwei=Decimal(str(random.randint(20, 50))) if success and follower_chain != "solana" else None,
            error_message="Mock execution failed" if not success else None,
        )

        self.executions.append({
            "signal": signal,
            "size_usd": size_usd,
            "result": result,
        })

        return result
