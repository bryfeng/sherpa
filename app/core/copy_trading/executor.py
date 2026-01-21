"""
Copy Executor

Executes copy trades on various chains and DEXs.

For manual approval flow:
- Returns unsigned transactions for frontend signing
- Does NOT execute transactions directly

For autonomous flow (with session keys):
- Would execute via session key signing (future)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import TradeSignal
from ..wallet.models import SessionKey

if TYPE_CHECKING:
    from ...providers.jupiter import JupiterSwapProvider, JupiterQuote
    from ...providers.relay import RelayProvider

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

    # For manual approval - unsigned transaction data
    requires_signature: bool = False
    unsigned_transaction: Optional[str] = None  # Base64/hex encoded
    transaction_data: Optional[Dict[str, Any]] = None  # Full TX data for frontend
    quote_data: Optional[Dict[str, Any]] = None  # Quote details for display


@dataclass
class QuoteResult:
    """Result of getting a swap quote (before execution)."""

    success: bool
    input_amount: Optional[Decimal] = None
    input_amount_raw: Optional[int] = None  # In smallest units
    output_amount: Optional[Decimal] = None
    output_amount_raw: Optional[int] = None
    price_impact_pct: Optional[float] = None
    estimated_fee_usd: Optional[Decimal] = None
    route_info: Optional[str] = None  # Description of route
    unsigned_transaction: Optional[str] = None
    transaction_data: Optional[Dict[str, Any]] = None
    quote_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Solana-specific
    last_valid_block_height: Optional[int] = None

    # EVM-specific
    gas_estimate: Optional[int] = None
    steps: Optional[List[Dict[str, Any]]] = None


class CopyExecutor:
    """
    Execute copy trades.

    Supports multiple chains and DEXs:
    - Solana: Jupiter
    - EVM: Relay

    For manual approval flow, returns unsigned transactions.
    Frontend signs and submits, then confirms execution.
    """

    def __init__(
        self,
        jupiter_provider: Optional["JupiterSwapProvider"] = None,
        relay_provider: Optional["RelayProvider"] = None,
        session_key_service: Optional[Any] = None,
    ):
        self.jupiter = jupiter_provider
        self.relay = relay_provider
        self.session_key_service = session_key_service

    @classmethod
    def create_with_providers(cls) -> "CopyExecutor":
        """Factory method to create executor with real providers."""
        from ...providers.jupiter import get_jupiter_swap_provider
        from ...providers.relay import RelayProvider

        return cls(
            jupiter_provider=get_jupiter_swap_provider(),
            relay_provider=RelayProvider(),
        )

    async def get_quote(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int = 100,
    ) -> QuoteResult:
        """
        Get a swap quote with unsigned transaction for manual approval.

        Args:
            signal: Original trade signal from the leader
            size_usd: USD amount to trade
            follower_address: Address of the follower wallet
            follower_chain: Chain of the follower wallet
            max_slippage_bps: Maximum slippage in basis points

        Returns:
            QuoteResult with unsigned transaction
        """
        logger.info(
            f"Getting copy quote: {signal.token_in_symbol} -> "
            f"{signal.token_out_symbol}, size=${size_usd} on {follower_chain}"
        )

        try:
            if follower_chain.lower() == "solana":
                return await self._get_solana_quote(
                    signal=signal,
                    size_usd=size_usd,
                    follower_address=follower_address,
                    max_slippage_bps=max_slippage_bps,
                )
            else:
                return await self._get_evm_quote(
                    signal=signal,
                    size_usd=size_usd,
                    follower_address=follower_address,
                    follower_chain=follower_chain,
                    max_slippage_bps=max_slippage_bps,
                )
        except Exception as e:
            logger.error(f"Quote error: {e}", exc_info=True)
            return QuoteResult(success=False, error_message=str(e))

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

        For manual approval flow:
        - Returns unsigned transaction in ExecutionResult
        - Frontend signs and submits
        - Frontend calls confirm_execution() with tx hash

        For autonomous flow (with session key):
        - Would sign and submit directly (future)

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
            # Get quote with unsigned transaction
            quote = await self.get_quote(
                signal=signal,
                size_usd=size_usd,
                follower_address=follower_address,
                follower_chain=follower_chain,
                max_slippage_bps=max_slippage_bps,
            )

            if not quote.success:
                return ExecutionResult(
                    success=False,
                    error_message=quote.error_message or "Failed to get quote",
                )

            # If no session key, return unsigned transaction for manual signing
            if not session_key:
                return ExecutionResult(
                    success=True,
                    requires_signature=True,
                    unsigned_transaction=quote.unsigned_transaction,
                    transaction_data=quote.transaction_data,
                    quote_data=quote.quote_response,
                    token_out_amount=quote.output_amount,
                    actual_value_usd=size_usd,
                    slippage_bps=max_slippage_bps,
                )

            # TODO: Autonomous execution with session key
            # For now, still return for manual signing
            logger.warning("Session key execution not yet implemented, returning for manual signing")
            return ExecutionResult(
                success=True,
                requires_signature=True,
                unsigned_transaction=quote.unsigned_transaction,
                transaction_data=quote.transaction_data,
                quote_data=quote.quote_response,
                token_out_amount=quote.output_amount,
                actual_value_usd=size_usd,
                slippage_bps=max_slippage_bps,
            )

        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )

    async def _get_solana_quote(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        max_slippage_bps: int,
    ) -> QuoteResult:
        """Get a swap quote on Solana via Jupiter."""
        if not self.jupiter:
            return QuoteResult(
                success=False,
                error_message="Jupiter provider not configured",
            )

        try:
            # Get price of input token to convert USD to lamports
            input_price = await self._get_token_price_solana(signal.token_in_address)
            if not input_price or input_price <= 0:
                return QuoteResult(
                    success=False,
                    error_message=f"Could not get price for {signal.token_in_symbol}",
                )

            # Get token decimals
            input_token = await self.jupiter._token_provider.get_token_by_mint(
                signal.token_in_address
            )
            input_decimals = input_token.decimals if input_token else 9

            # Convert USD to token amount in smallest units
            token_amount = size_usd / Decimal(str(input_price))
            amount_raw = int(token_amount * (Decimal(10) ** input_decimals))

            logger.info(
                f"Solana quote: ${size_usd} = {token_amount} {signal.token_in_symbol} "
                f"= {amount_raw} raw units"
            )

            # Get quote from Jupiter
            quote = await self.jupiter.get_swap_quote(
                input_mint=signal.token_in_address,
                output_mint=signal.token_out_address,
                amount=amount_raw,
                slippage_bps=max_slippage_bps,
            )

            # Get output token decimals
            output_token = await self.jupiter._token_provider.get_token_by_mint(
                signal.token_out_address
            )
            output_decimals = output_token.decimals if output_token else 9

            # Build swap transaction
            swap_result = await self.jupiter.build_swap_transaction(
                quote=quote,
                user_public_key=follower_address,
            )

            # Calculate human-readable amounts
            output_amount = Decimal(str(quote.out_amount)) / (Decimal(10) ** output_decimals)

            return QuoteResult(
                success=True,
                input_amount=token_amount,
                input_amount_raw=amount_raw,
                output_amount=output_amount,
                output_amount_raw=quote.out_amount,
                price_impact_pct=quote.price_impact_pct,
                unsigned_transaction=swap_result.swap_transaction,
                last_valid_block_height=swap_result.last_valid_block_height,
                transaction_data={
                    "type": "solana",
                    "swap_transaction": swap_result.swap_transaction,
                    "last_valid_block_height": swap_result.last_valid_block_height,
                    "priority_fee_lamports": swap_result.priority_fee_lamports,
                    "compute_unit_limit": swap_result.compute_unit_limit,
                },
                quote_response=quote.quote_response,
                route_info=self._format_route_info(quote),
            )

        except Exception as e:
            logger.error(f"Solana quote error: {e}", exc_info=True)
            return QuoteResult(
                success=False,
                error_message=str(e),
            )

    async def _get_token_price_solana(self, mint_address: str) -> Optional[float]:
        """Get token price in USD from Jupiter."""
        if not self.jupiter:
            return None
        try:
            return await self.jupiter._token_provider.get_token_price(mint_address)
        except Exception:
            return None

    def _format_route_info(self, quote: "JupiterQuote") -> str:
        """Format route info for display."""
        if not quote.route_plan:
            return "Direct swap"

        route_parts = []
        for step in quote.route_plan:
            swap_info = step.swap_info
            amm_key = swap_info.get("ammKey", "")[:8] if swap_info.get("ammKey") else "?"
            route_parts.append(f"{swap_info.get('label', 'AMM')}({amm_key})")

        return " → ".join(route_parts) if route_parts else "Multi-hop route"

    async def _get_evm_quote(
        self,
        signal: TradeSignal,
        size_usd: Decimal,
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int,
    ) -> QuoteResult:
        """Get a swap quote on EVM chains via Relay."""
        if not self.relay:
            return QuoteResult(
                success=False,
                error_message="Relay provider not configured",
            )

        try:
            # Map chain to chain ID
            chain_id = self._chain_to_id(follower_chain)

            # Get token info for decimals
            from ..bridge.chain_registry import get_chain_registry

            registry = get_chain_registry()
            if not registry:
                return QuoteResult(
                    success=False,
                    error_message="Chain registry not initialized",
                )

            # Get token decimals (default to 18 for EVM)
            input_decimals = 18
            token_meta = registry.get_token_by_address(chain_id, signal.token_in_address)
            if token_meta:
                input_decimals = token_meta.get("decimals", 18)

            # Get token price to convert USD to token amount
            input_price = await self._get_token_price_evm(
                chain_id, signal.token_in_address
            )
            if not input_price or input_price <= 0:
                return QuoteResult(
                    success=False,
                    error_message=f"Could not get price for {signal.token_in_symbol}",
                )

            # Convert USD to token amount in smallest units
            token_amount = size_usd / Decimal(str(input_price))
            amount_raw = int(token_amount * (Decimal(10) ** input_decimals))

            logger.info(
                f"EVM quote: ${size_usd} = {token_amount} {signal.token_in_symbol} "
                f"= {amount_raw} raw units on chain {chain_id}"
            )

            # Build Relay quote request
            relay_payload = {
                "user": follower_address,
                "originChainId": chain_id,
                "destinationChainId": chain_id,  # Same chain swap
                "originCurrency": signal.token_in_address,
                "destinationCurrency": signal.token_out_address,
                "tradeType": "EXACT_INPUT",
                "amount": str(amount_raw),
                "referrer": "sherpa.chat",
            }

            quote = await self.relay.quote(relay_payload)

            # Parse quote response
            details = quote.get("details", {})
            currency_out = details.get("currencyOut", {})
            output_amount_raw = int(currency_out.get("amount", "0"))
            output_decimals = currency_out.get("decimals", 18)
            output_amount = Decimal(str(output_amount_raw)) / (Decimal(10) ** output_decimals)

            # Get fees and gas estimate
            total_fee_usd = Decimal(str(details.get("totalFeeUsd", "0")))

            # Get steps/transactions
            steps = quote.get("steps", [])

            return QuoteResult(
                success=True,
                input_amount=token_amount,
                input_amount_raw=amount_raw,
                output_amount=output_amount,
                output_amount_raw=output_amount_raw,
                estimated_fee_usd=total_fee_usd,
                steps=steps,
                transaction_data={
                    "type": "evm",
                    "chain_id": chain_id,
                    "steps": steps,
                    "quote": quote,
                },
                quote_response=quote,
                route_info=f"Relay swap on {follower_chain}",
            )

        except Exception as e:
            logger.error(f"EVM quote error: {e}", exc_info=True)
            return QuoteResult(
                success=False,
                error_message=str(e),
            )

    async def _get_token_price_evm(
        self, chain_id: int, token_address: str
    ) -> Optional[float]:
        """Get token price in USD from Coingecko or other source."""
        try:
            from ...providers.coingecko import CoingeckoProvider

            cg = CoingeckoProvider()
            prices = await cg.get_token_prices([token_address.lower()])
            price_data = prices.get(token_address.lower(), {})
            return price_data.get("price_usd")
        except Exception:
            # Fallback: try to infer from common tokens
            return None

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
        follower_address: str,
        follower_chain: str,
        max_slippage_bps: int = 100,
    ) -> Dict[str, Any]:
        """
        Estimate execution details without building transaction.

        Returns quote information, expected output, gas costs, etc.
        """
        quote = await self.get_quote(
            signal=signal,
            size_usd=size_usd,
            follower_address=follower_address,
            follower_chain=follower_chain,
            max_slippage_bps=max_slippage_bps,
        )

        if not quote.success:
            return {"error": quote.error_message}

        return {
            "success": True,
            "input_amount": str(quote.input_amount) if quote.input_amount else None,
            "output_amount": str(quote.output_amount) if quote.output_amount else None,
            "price_impact_pct": quote.price_impact_pct,
            "estimated_fee_usd": str(quote.estimated_fee_usd) if quote.estimated_fee_usd else None,
            "route_info": quote.route_info,
            "chain": follower_chain,
        }


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
