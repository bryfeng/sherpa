"""Dollar Cost Averaging (DCA) Strategy Implementation.

This module implements the DCA strategy which:
- Periodically swaps a fixed amount of source token
- Distributes across target tokens according to allocation weights
- Skips execution if source balance is insufficient

Example:
    config = {
        "source_token": "USDC",
        "target_tokens": ["ETH", "BTC"],
        "amount_per_execution": 100,
        "allocation": {"ETH": 0.6, "BTC": 0.4},
        "default_chain": 1,
    }

    strategy = DCAStrategy()
    intents = strategy.evaluate(context)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config import AgentConfig, DCAStrategyParams
from ..models import TokenReference
from ..protocol import (
    ActionType,
    AmountSpec,
    AmountUnit,
    BaseStrategy,
    StrategyContext,
    TradeIntent,
)

logger = logging.getLogger(__name__)


class DCAStrategy:
    """Dollar Cost Averaging strategy implementation.

    Periodically accumulates target tokens by swapping from a source token.
    Designed for long-term accumulation with consistent execution.

    Attributes:
        id: Strategy identifier
        version: Strategy version
    """

    id = "dca"
    version = "1.0"

    def __init__(self, params: Optional[DCAStrategyParams] = None):
        """Initialize DCA strategy.

        Args:
            params: Optional strategy parameters (can also come from context)
        """
        self._params = params

    def evaluate(self, ctx: StrategyContext) -> List[TradeIntent]:
        """Evaluate current conditions and generate trade intents.

        The DCA strategy:
        1. Checks if we have sufficient source token balance
        2. Creates swap intents for each target token based on allocation
        3. Returns intents (never executes directly)

        Args:
            ctx: Strategy context with portfolio and config

        Returns:
            List of TradeIntent objects (may be empty if conditions not met)
        """
        intents: List[TradeIntent] = []

        # Get parameters from config or init
        params = self._get_params(ctx)
        if not params:
            logger.warning("DCA strategy: No parameters configured")
            return intents

        # Validate allocation
        validation_errors = params.validate_allocation()
        if validation_errors:
            for error in validation_errors:
                logger.warning(f"DCA strategy: {error}")
            return intents

        # Check source token balance
        source_balance = ctx.portfolio.get_usd_value(params.source_token)
        if source_balance < params.amount_per_execution:
            logger.info(
                f"DCA strategy: Insufficient {params.source_token} balance "
                f"({source_balance} < {params.amount_per_execution})"
            )
            return intents

        # Resolve source token
        source_token = self._resolve_token_from_portfolio(
            params.source_token, ctx, params.default_chain
        )
        if not source_token:
            logger.warning(f"DCA strategy: Could not resolve source token {params.source_token}")
            return intents

        # Create intents for each target token
        for target_symbol in params.target_tokens:
            weight = params.allocation.get(target_symbol, 0)
            if weight <= 0:
                continue

            # Calculate amount for this target
            amount_usd = params.amount_per_execution * Decimal(str(weight))

            # Resolve target token
            target_token = self._resolve_token_from_portfolio(
                target_symbol, ctx, params.default_chain
            )
            if not target_token:
                logger.warning(f"DCA strategy: Could not resolve target token {target_symbol}")
                continue

            # Create intent
            intent = TradeIntent(
                action_type=ActionType.SWAP,
                chain_id=params.default_chain,
                token_in=source_token,
                token_out=target_token,
                amount=AmountSpec(value=amount_usd, unit=AmountUnit.USD),
                confidence=1.0,  # DCA is deterministic
                reasoning=f"DCA: Allocating {weight * 100:.1f}% (${amount_usd}) to {target_symbol}",
                metadata={
                    "strategy": "dca",
                    "weight": weight,
                    "target_symbol": target_symbol,
                },
            )
            intents.append(intent)

        logger.info(f"DCA strategy: Generated {len(intents)} trade intents")
        return intents

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate DCA strategy configuration.

        Args:
            config: Strategy configuration dict

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Required fields
        if "target_tokens" not in config:
            errors.append("target_tokens is required")
        elif not isinstance(config["target_tokens"], list) or len(config["target_tokens"]) == 0:
            errors.append("target_tokens must be a non-empty list")

        if "amount_per_execution" not in config:
            errors.append("amount_per_execution is required")
        else:
            try:
                amount = Decimal(str(config["amount_per_execution"]))
                if amount <= 0:
                    errors.append("amount_per_execution must be positive")
            except Exception:
                errors.append("amount_per_execution must be a valid number")

        if "allocation" not in config:
            errors.append("allocation is required")
        elif not isinstance(config["allocation"], dict):
            errors.append("allocation must be a dict mapping token symbols to weights")
        else:
            # Validate allocation weights
            total = sum(config["allocation"].values())
            if abs(total - 1.0) > 0.001:
                errors.append(f"allocation weights must sum to 1.0, got {total}")

            # Check all target tokens have allocation
            target_tokens = set(config.get("target_tokens", []))
            allocation_tokens = set(config["allocation"].keys())
            missing = target_tokens - allocation_tokens
            if missing:
                errors.append(f"Missing allocation for tokens: {missing}")

        return errors

    def _get_params(self, ctx: StrategyContext) -> Optional[DCAStrategyParams]:
        """Get strategy parameters from init or context."""
        if self._params:
            return self._params

        # Try to get from agent config
        if hasattr(ctx, "agent_config") and ctx.agent_config:
            strategy_params = ctx.agent_config.strategy_params
            if strategy_params:
                try:
                    return DCAStrategyParams(**strategy_params)
                except Exception as e:
                    logger.warning(f"Failed to parse DCA params: {e}")

        return None

    def _resolve_token_from_portfolio(
        self,
        symbol: str,
        ctx: StrategyContext,
        default_chain: int,
    ) -> Optional[TokenReference]:
        """Resolve a token from portfolio or create a reference.

        Args:
            symbol: Token symbol
            ctx: Strategy context
            default_chain: Default chain ID

        Returns:
            TokenReference or None
        """
        # In a full implementation, this would use TokenResolutionService
        # For now, create a basic reference
        symbol_upper = symbol.upper()

        # Common token addresses on Ethereum mainnet
        known_tokens: Dict[str, Dict[str, Any]] = {
            "USDC": {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "decimals": 6,
            },
            "ETH": {
                "address": "0x0000000000000000000000000000000000000000",
                "decimals": 18,
            },
            "WETH": {
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "decimals": 18,
            },
            "BTC": {
                "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
                "decimals": 8,
            },
            "WBTC": {
                "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                "decimals": 8,
            },
        }

        if symbol_upper in known_tokens:
            token_info = known_tokens[symbol_upper]
            return TokenReference(
                chain_id=default_chain,
                address=token_info["address"],
                symbol=symbol_upper,
                decimals=token_info["decimals"],
                name=symbol_upper,
                confidence=1.0,
            )

        # Fallback: return a reference with unknown address
        return TokenReference(
            chain_id=default_chain,
            address="",
            symbol=symbol_upper,
            decimals=18,
            confidence=0.5,
        )
