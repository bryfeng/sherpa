"""Strategy Protocol and Intent objects for the planning system.

This module defines the core abstractions for strategy logic:
- TradeIntent: Immutable object representing what a strategy wants to do
- StrategyContext: Read-only context passed to strategy evaluation
- BaseStrategy: Protocol that all strategies must implement

Key Design Principle:
    Strategies produce intents, not executions. The intent is then validated
    against policy constraints and executed by the PlanExecutor. This separation
    enables:
    1. Policy validation before execution
    2. Audit trails for all decisions
    3. Future onchain validation of intents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Union

if TYPE_CHECKING:
    from .config import AgentConfig
    from .models import TokenReference


class ActionType(str, Enum):
    """Types of actions that can be included in an intent."""

    SWAP = "swap"
    BRIDGE = "bridge"
    STAKE = "stake"
    UNSTAKE = "unstake"
    APPROVE = "approve"
    CLAIM = "claim"


class AmountUnit(str, Enum):
    """Units for specifying amounts."""

    TOKEN = "token"  # Amount in token units
    USD = "usd"  # Amount in USD value
    PERCENT = "percent"  # Percentage of balance


@dataclass(frozen=True)
class AmountSpec:
    """Specification for an amount, supporting multiple units."""

    value: Decimal
    unit: AmountUnit = AmountUnit.TOKEN

    def __post_init__(self) -> None:
        # Validate that value is positive
        if self.value <= 0:
            raise ValueError("Amount must be positive")

    @classmethod
    def from_tokens(cls, value: Union[int, float, Decimal, str]) -> AmountSpec:
        return cls(value=Decimal(str(value)), unit=AmountUnit.TOKEN)

    @classmethod
    def from_usd(cls, value: Union[int, float, Decimal, str]) -> AmountSpec:
        return cls(value=Decimal(str(value)), unit=AmountUnit.USD)

    @classmethod
    def from_percent(cls, value: Union[int, float, Decimal, str]) -> AmountSpec:
        return cls(value=Decimal(str(value)), unit=AmountUnit.PERCENT)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": str(self.value), "unit": self.unit.value}


@dataclass(frozen=True)
class TradeIntent:
    """Immutable intent object representing what a strategy wants to do.

    TradeIntents are produced by strategies and then validated against
    PolicyConstraints before being converted to executable Actions.

    Attributes:
        action_type: The type of action (swap, bridge, etc.)
        chain_id: The chain to execute on (int for EVM, 'solana' for Solana)
        token_in: Token to spend/sell
        token_out: Token to receive/buy
        amount: Amount specification
        confidence: Strategy's confidence in this trade (0.0 to 1.0)
        reasoning: Human-readable explanation for audit trail
        metadata: Strategy-specific data (e.g., feature scores)
        created_at: Timestamp when intent was created
    """

    action_type: ActionType
    chain_id: Union[int, str]  # int for EVM, 'solana' for Solana
    token_in: TokenReference
    token_out: TokenReference
    amount: AmountSpec
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict for storage/transmission."""
        return {
            "action_type": self.action_type.value,
            "chain_id": self.chain_id,
            "token_in": self.token_in.to_dict() if hasattr(self.token_in, "to_dict") else str(self.token_in),
            "token_out": self.token_out.to_dict() if hasattr(self.token_out, "to_dict") else str(self.token_out),
            "amount": self.amount.to_dict(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PortfolioSnapshot:
    """Read-only snapshot of portfolio state for strategy evaluation."""

    wallet_address: str
    chain_id: Union[int, str]
    balances: Dict[str, Decimal]  # symbol -> balance in token units
    usd_values: Dict[str, Decimal]  # symbol -> balance in USD
    total_usd: Decimal
    timestamp: datetime

    def get_balance(self, symbol: str) -> Decimal:
        """Get balance for a token symbol, returns 0 if not found."""
        return self.balances.get(symbol.upper(), Decimal("0"))

    def get_usd_value(self, symbol: str) -> Decimal:
        """Get USD value for a token, returns 0 if not found."""
        return self.usd_values.get(symbol.upper(), Decimal("0"))


@dataclass
class StrategyContext:
    """Read-only context passed to strategy evaluation.

    Contains all the information a strategy needs to make decisions,
    without giving it the ability to execute anything directly.

    Attributes:
        agent_config: The agent's configuration
        portfolio: Current portfolio snapshot
        market_data: Additional market data (prices, features, etc.)
        timestamp: Current evaluation timestamp
        previous_decisions: Recent decision history for continuity
    """

    agent_config: AgentConfig
    portfolio: PortfolioSnapshot
    market_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    previous_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a token from market data."""
        prices = self.market_data.get("prices", {})
        return prices.get(symbol.upper())

    def get_feature(self, name: str) -> Optional[Any]:
        """Get a computed feature from market data."""
        features = self.market_data.get("features", {})
        return features.get(name)


class BaseStrategy(Protocol):
    """Protocol that all strategies must implement.

    Strategies are responsible for:
    1. Evaluating market conditions
    2. Producing TradeIntent objects
    3. Validating their own configuration

    Strategies are NOT responsible for:
    - Executing trades (that's PlanExecutor's job)
    - Validating against policy (that's PolicyConstraints' job)
    - Persisting state (that's PlanStore's job)

    Example:
        class DCAStrategy:
            id = "dca"
            version = "1.0"

            def evaluate(self, ctx: StrategyContext) -> List[TradeIntent]:
                # Check balance, create intents
                return [TradeIntent(...)]

            def validate_config(self, config: Dict[str, Any]) -> List[str]:
                errors = []
                if 'target_tokens' not in config:
                    errors.append("target_tokens is required")
                return errors
    """

    id: str
    version: str

    def evaluate(self, ctx: StrategyContext) -> List[TradeIntent]:
        """Evaluate current conditions and return trade intents.

        This is the ONLY method that contains strategy logic.
        Strategies NEVER execute trades directly - they return intents.

        Args:
            ctx: Read-only context with portfolio, market data, config

        Returns:
            List of TradeIntent objects (may be empty if no action needed)
        """
        ...

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate strategy-specific configuration.

        Args:
            config: Strategy parameters from AgentConfig.strategy_params

        Returns:
            List of error messages (empty if valid)
        """
        ...
