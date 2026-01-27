"""
Intent Builder Service.

Constructs Rhinestone intent payloads for various DeFi operations:
- Token swaps
- DCA executions
- Copy trading
- Bridge operations

Intents are submitted to the Rhinestone Orchestrator for execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from eth_utils import to_checksum_address

from ..providers.rhinestone import (
    IntentRequest,
    IntentResult,
    RhinestoneError,
    get_rhinestone_provider,
)
from ..config import settings

logger = logging.getLogger(__name__)


# Well-known token addresses (checksummed)
USDC_ADDRESSES: Dict[int, str] = {
    1: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",      # Ethereum
    10: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",     # Optimism
    137: "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",    # Polygon
    8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # Base
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # Arbitrum
}

WETH_ADDRESSES: Dict[int, str] = {
    1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",      # Ethereum
    10: "0x4200000000000000000000000000000000000006",     # Optimism
    137: "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",    # Polygon
    8453: "0x4200000000000000000000000000000000000006",   # Base
    42161: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # Arbitrum
}

# Chain metadata
CHAIN_NAMES: Dict[int, str] = {
    1: "ethereum",
    10: "optimism",
    137: "polygon",
    8453: "base",
    42161: "arbitrum",
}


@dataclass
class SwapIntent:
    """Intent for a token swap."""
    account_address: str
    source_chain: int
    target_chain: int
    token_in: str
    token_out: str
    amount_in: str  # Raw amount (wei)
    min_amount_out: Optional[str] = None  # Minimum output (slippage protection)
    recipient: Optional[str] = None  # Defaults to account_address


@dataclass
class DCAExecutionIntent:
    """Intent for a DCA strategy execution."""
    account_address: str
    strategy_id: str
    source_chains: List[int]  # Where to pull funds from
    target_chain: int  # Where swap happens
    token_in: str
    token_out: str
    amount: str  # Raw amount per execution
    session_id: Optional[str] = None  # Smart Session for permission validation


@dataclass
class CopyTradeIntent:
    """Intent to copy a trade from a leader."""
    account_address: str
    leader_address: str
    source_chains: List[int]
    target_chain: int
    token_in: str
    token_out: str
    amount: str  # Calculated based on sizing strategy
    session_id: Optional[str] = None


@dataclass
class IntentBuilderResult:
    """Result from intent building."""
    success: bool
    intent_id: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class IntentBuilderService:
    """
    Builds and submits intents to Rhinestone.

    Usage:
        service = get_intent_builder_service()

        # Build and submit a swap intent
        result = await service.build_swap_intent(
            account_address="0x...",
            source_chain=8453,
            target_chain=8453,
            token_in="0x...",  # USDC
            token_out="0x...",  # WETH
            amount_in="1000000000",  # 1000 USDC
        )
    """

    def __init__(self) -> None:
        self._provider = get_rhinestone_provider()

    async def build_swap_intent(
        self,
        account_address: str,
        source_chain: int,
        target_chain: int,
        token_in: str,
        token_out: str,
        amount_in: str,
        min_amount_out: Optional[str] = None,
        recipient: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> IntentBuilderResult:
        """
        Build and submit a swap intent.

        Args:
            account_address: User's Smart Account address
            source_chain: Chain ID to source funds from
            target_chain: Chain ID where swap executes
            token_in: Input token address
            token_out: Output token address
            amount_in: Raw input amount
            min_amount_out: Minimum output amount (slippage)
            recipient: Output recipient (defaults to account_address)
            session_id: Smart Session ID for autonomous execution

        Returns:
            IntentBuilderResult with intent_id if successful
        """
        try:
            # Normalize addresses
            account_address = to_checksum_address(account_address)
            token_in = to_checksum_address(token_in)
            token_out = to_checksum_address(token_out)
            recipient = to_checksum_address(recipient) if recipient else account_address

            # Build swap call data
            # The Rhinestone Orchestrator handles the actual DEX routing
            # We just specify the intent parameters
            calls = [
                {
                    "to": token_out,  # Target token (Orchestrator routes to DEX)
                    "value": "0",
                    "data": "",  # Orchestrator fills this
                    "intent": {
                        "type": "swap",
                        "tokenIn": token_in,
                        "tokenOut": token_out,
                        "amountIn": amount_in,
                        "minAmountOut": min_amount_out or "0",
                        "recipient": recipient,
                    },
                }
            ]

            token_requests = [
                {
                    "address": token_in,
                    "amount": amount_in,
                }
            ]

            # Submit to Rhinestone
            result = await self._provider.submit_intent(
                account_address=account_address,
                source_chains=[source_chain],
                target_chain=target_chain,
                calls=calls,
                token_requests=token_requests,
                session_id=session_id,
            )

            logger.info(
                f"Swap intent submitted: {result.intent_id} "
                f"({token_in} -> {token_out} on chain {target_chain})"
            )

            return IntentBuilderResult(
                success=True,
                intent_id=result.intent_id,
                details={
                    "source_chain": source_chain,
                    "target_chain": target_chain,
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount_in": amount_in,
                },
            )

        except RhinestoneError as e:
            logger.error(f"Failed to build swap intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error building swap intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))

    async def build_dca_execution_intent(
        self,
        account_address: str,
        strategy_id: str,
        source_chains: List[int],
        target_chain: int,
        token_in: str,
        token_out: str,
        amount: str,
        session_id: Optional[str] = None,
    ) -> IntentBuilderResult:
        """
        Build and submit a DCA execution intent.

        This is called by the DCA scheduler when it's time to execute
        a DCA strategy.

        Args:
            account_address: User's Smart Account
            strategy_id: DCA strategy ID (for tracking)
            source_chains: Chains to source funds from
            target_chain: Chain where swap executes
            token_in: Input token
            token_out: Output token
            amount: Amount per DCA execution
            session_id: Smart Session for autonomous execution

        Returns:
            IntentBuilderResult
        """
        try:
            account_address = to_checksum_address(account_address)
            token_in = to_checksum_address(token_in)
            token_out = to_checksum_address(token_out)

            calls = [
                {
                    "to": token_out,
                    "value": "0",
                    "data": "",
                    "intent": {
                        "type": "swap",
                        "tokenIn": token_in,
                        "tokenOut": token_out,
                        "amountIn": amount,
                        "minAmountOut": "0",  # DCA typically uses market orders
                        "recipient": account_address,
                    },
                    "metadata": {
                        "strategyId": strategy_id,
                        "strategyType": "dca",
                    },
                }
            ]

            token_requests = [{"address": token_in, "amount": amount}]

            result = await self._provider.submit_intent(
                account_address=account_address,
                source_chains=source_chains,
                target_chain=target_chain,
                calls=calls,
                token_requests=token_requests,
                session_id=session_id,
            )

            logger.info(
                f"DCA intent submitted: {result.intent_id} "
                f"(strategy={strategy_id}, amount={amount})"
            )

            return IntentBuilderResult(
                success=True,
                intent_id=result.intent_id,
                details={
                    "strategy_id": strategy_id,
                    "source_chains": source_chains,
                    "target_chain": target_chain,
                    "amount": amount,
                },
            )

        except RhinestoneError as e:
            logger.error(f"Failed to build DCA intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error building DCA intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))

    async def build_copy_trade_intent(
        self,
        account_address: str,
        leader_address: str,
        source_chains: List[int],
        target_chain: int,
        token_in: str,
        token_out: str,
        amount: str,
        original_tx_hash: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> IntentBuilderResult:
        """
        Build and submit a copy trade intent.

        Args:
            account_address: Copier's Smart Account
            leader_address: Leader wallet being copied
            source_chains: Chains to source funds from
            target_chain: Chain where trade executes
            token_in: Input token
            token_out: Output token
            amount: Copy trade amount (based on sizing strategy)
            original_tx_hash: Leader's original transaction
            session_id: Smart Session for autonomous execution

        Returns:
            IntentBuilderResult
        """
        try:
            account_address = to_checksum_address(account_address)
            leader_address = to_checksum_address(leader_address)
            token_in = to_checksum_address(token_in)
            token_out = to_checksum_address(token_out)

            calls = [
                {
                    "to": token_out,
                    "value": "0",
                    "data": "",
                    "intent": {
                        "type": "swap",
                        "tokenIn": token_in,
                        "tokenOut": token_out,
                        "amountIn": amount,
                        "minAmountOut": "0",
                        "recipient": account_address,
                    },
                    "metadata": {
                        "strategyType": "copy_trade",
                        "leaderAddress": leader_address,
                        "originalTxHash": original_tx_hash,
                    },
                }
            ]

            token_requests = [{"address": token_in, "amount": amount}]

            result = await self._provider.submit_intent(
                account_address=account_address,
                source_chains=source_chains,
                target_chain=target_chain,
                calls=calls,
                token_requests=token_requests,
                session_id=session_id,
            )

            logger.info(
                f"Copy trade intent submitted: {result.intent_id} "
                f"(copying {leader_address})"
            )

            return IntentBuilderResult(
                success=True,
                intent_id=result.intent_id,
                details={
                    "leader_address": leader_address,
                    "target_chain": target_chain,
                    "amount": amount,
                    "original_tx_hash": original_tx_hash,
                },
            )

        except RhinestoneError as e:
            logger.error(f"Failed to build copy trade intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error building copy trade intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))

    async def build_bridge_intent(
        self,
        account_address: str,
        source_chain: int,
        target_chain: int,
        token: str,
        amount: str,
        recipient: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> IntentBuilderResult:
        """
        Build and submit a bridge intent.

        Args:
            account_address: User's Smart Account
            source_chain: Source chain ID
            target_chain: Destination chain ID
            token: Token to bridge
            amount: Amount to bridge
            recipient: Destination recipient (defaults to account)
            session_id: Smart Session for autonomous execution

        Returns:
            IntentBuilderResult
        """
        try:
            account_address = to_checksum_address(account_address)
            token = to_checksum_address(token)
            recipient = to_checksum_address(recipient) if recipient else account_address

            calls = [
                {
                    "to": recipient,
                    "value": "0",
                    "data": "",
                    "intent": {
                        "type": "bridge",
                        "token": token,
                        "amount": amount,
                        "recipient": recipient,
                    },
                }
            ]

            token_requests = [{"address": token, "amount": amount}]

            result = await self._provider.submit_intent(
                account_address=account_address,
                source_chains=[source_chain],
                target_chain=target_chain,
                calls=calls,
                token_requests=token_requests,
                session_id=session_id,
            )

            logger.info(
                f"Bridge intent submitted: {result.intent_id} "
                f"({source_chain} -> {target_chain})"
            )

            return IntentBuilderResult(
                success=True,
                intent_id=result.intent_id,
                details={
                    "source_chain": source_chain,
                    "target_chain": target_chain,
                    "token": token,
                    "amount": amount,
                },
            )

        except RhinestoneError as e:
            logger.error(f"Failed to build bridge intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error building bridge intent: {e}")
            return IntentBuilderResult(success=False, error=str(e))

    async def get_intent_status(self, intent_id: str) -> IntentResult:
        """
        Get the status of a submitted intent.

        Args:
            intent_id: Intent ID from build_*_intent

        Returns:
            IntentResult with current status
        """
        return await self._provider.get_intent_status(intent_id)

    async def wait_for_intent(
        self,
        intent_id: str,
        timeout_s: float = 120.0,
    ) -> IntentResult:
        """
        Wait for an intent to complete.

        Args:
            intent_id: Intent ID
            timeout_s: Maximum wait time

        Returns:
            Final IntentResult
        """
        return await self._provider.wait_for_intent(intent_id, timeout_s=timeout_s)


# Singleton instance
_intent_builder_service: Optional[IntentBuilderService] = None


def get_intent_builder_service() -> IntentBuilderService:
    """Get the singleton IntentBuilder service instance."""
    global _intent_builder_service
    if _intent_builder_service is None:
        _intent_builder_service = IntentBuilderService()
    return _intent_builder_service
