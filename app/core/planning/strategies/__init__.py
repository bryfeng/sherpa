"""Strategy implementations for autonomous agents.

This module provides concrete strategy implementations that follow
the BaseStrategy protocol. Each strategy:
- Receives a StrategyContext with portfolio and market data
- Returns a list of TradeIntent objects
- Never executes trades directly

Available Strategies:
- DCAStrategy: Dollar Cost Averaging
- (Future) MomentumStrategy: Momentum-based trading
- (Future) YieldStrategy: Yield rotation

Usage:
    from app.core.planning.strategies import get_strategy, DCAStrategy

    # Get strategy by type
    strategy = get_strategy("dca")

    # Or instantiate directly
    dca = DCAStrategy(config)
    intents = dca.evaluate(context)
"""

from typing import Dict, Optional, Type

from ..protocol import BaseStrategy
from .dca import DCAStrategy

# Registry of strategy implementations
_STRATEGY_REGISTRY: Dict[str, Type] = {
    "dca": DCAStrategy,
}


def get_strategy(strategy_type: str) -> Optional[BaseStrategy]:
    """Get a strategy implementation by type.

    Args:
        strategy_type: Strategy type (e.g., 'dca', 'momentum')

    Returns:
        Strategy instance or None if not found
    """
    strategy_class = _STRATEGY_REGISTRY.get(strategy_type.lower())
    if strategy_class:
        return strategy_class()
    return None


def register_strategy(strategy_type: str, strategy_class: Type) -> None:
    """Register a custom strategy implementation.

    Args:
        strategy_type: Strategy type identifier
        strategy_class: Strategy class implementing BaseStrategy
    """
    _STRATEGY_REGISTRY[strategy_type.lower()] = strategy_class


def list_strategies() -> Dict[str, Type]:
    """List all registered strategies."""
    return _STRATEGY_REGISTRY.copy()


__all__ = [
    "DCAStrategy",
    "get_strategy",
    "register_strategy",
    "list_strategies",
]
