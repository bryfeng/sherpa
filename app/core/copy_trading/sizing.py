"""
Position Sizing Strategies

Different strategies for calculating copy trade sizes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Optional

from .models import CopyConfig, TradeSignal, SizingMode


class SizingStrategy(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate the trade size for a copy trade.

        Args:
            signal: The trade signal from the leader
            config: Copy configuration with sizing parameters
            follower_portfolio_value_usd: Total value of follower's portfolio
            leader_portfolio_value_usd: Total value of leader's portfolio (for proportional)

        Returns:
            Trade size in USD
        """
        pass

    def apply_bounds(
        self,
        size_usd: Decimal,
        config: CopyConfig,
    ) -> Decimal:
        """Apply min/max bounds to the calculated size."""
        # Apply minimum
        if size_usd < config.min_trade_usd:
            return Decimal("0")  # Too small, skip trade

        # Apply maximum
        if config.max_trade_usd and size_usd > config.max_trade_usd:
            size_usd = config.max_trade_usd

        return size_usd


class PercentageSizing(SizingStrategy):
    """
    Size as a percentage of the follower's portfolio.

    Example: If size_value=5 and portfolio=$10,000, trade size = $500
    """

    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
    ) -> Decimal:
        # Calculate percentage of portfolio
        percentage = config.size_value / Decimal("100")
        raw_size = follower_portfolio_value_usd * percentage

        return self.apply_bounds(raw_size, config)


class FixedSizing(SizingStrategy):
    """
    Use a fixed USD amount for each trade.

    Example: If size_value=100, every copy trade is $100
    """

    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
    ) -> Decimal:
        raw_size = config.size_value

        return self.apply_bounds(raw_size, config)


class ProportionalSizing(SizingStrategy):
    """
    Size proportional to the leader's trade relative to their portfolio.

    Example: If leader traded 5% of their portfolio and multiplier=1,
    follower trades 5% of their portfolio.

    The size_value acts as a multiplier (0.5 = half, 1.0 = same, 2.0 = double)
    """

    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
    ) -> Decimal:
        if not leader_portfolio_value_usd or leader_portfolio_value_usd <= 0:
            # Fallback to fixed sizing if we don't know leader's portfolio
            return self.apply_bounds(config.size_value, config)

        if not signal.value_usd or signal.value_usd <= 0:
            return Decimal("0")

        # Calculate what percentage of their portfolio the leader traded
        leader_trade_pct = signal.value_usd / leader_portfolio_value_usd

        # Apply the same percentage to follower's portfolio, with multiplier
        raw_size = follower_portfolio_value_usd * leader_trade_pct * config.size_value

        return self.apply_bounds(raw_size, config)


class ScaledProportionalSizing(SizingStrategy):
    """
    Similar to proportional, but with portfolio-size-based scaling.

    Larger followers don't need to match leader's percentage exactly.
    Scales down proportionally for larger portfolios.
    """

    def __init__(
        self,
        base_portfolio_usd: Decimal = Decimal("10000"),
        scaling_factor: Decimal = Decimal("0.5"),
    ):
        """
        Args:
            base_portfolio_usd: Portfolio size at which scaling = 1
            scaling_factor: How aggressively to scale (0-1)
        """
        self.base_portfolio = base_portfolio_usd
        self.scaling_factor = scaling_factor

    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
    ) -> Decimal:
        if not leader_portfolio_value_usd or leader_portfolio_value_usd <= 0:
            return self.apply_bounds(config.size_value, config)

        if not signal.value_usd or signal.value_usd <= 0:
            return Decimal("0")

        # Calculate leader's trade percentage
        leader_trade_pct = signal.value_usd / leader_portfolio_value_usd

        # Calculate scaling factor based on portfolio size
        # Larger portfolios get scaled down
        if follower_portfolio_value_usd > self.base_portfolio:
            ratio = follower_portfolio_value_usd / self.base_portfolio
            scale = Decimal("1") / (Decimal("1") + (ratio - Decimal("1")) * self.scaling_factor)
        else:
            scale = Decimal("1")

        # Apply scaled percentage
        raw_size = follower_portfolio_value_usd * leader_trade_pct * config.size_value * scale

        return self.apply_bounds(raw_size, config)


class RiskAdjustedSizing(SizingStrategy):
    """
    Adjust size based on trade risk factors.

    Reduces size for:
    - High volatility tokens
    - Low liquidity
    - New/unknown tokens
    """

    def __init__(
        self,
        base_strategy: SizingStrategy,
        volatility_factor: Decimal = Decimal("0.3"),
        liquidity_factor: Decimal = Decimal("0.3"),
        newness_factor: Decimal = Decimal("0.2"),
    ):
        self.base_strategy = base_strategy
        self.volatility_factor = volatility_factor
        self.liquidity_factor = liquidity_factor
        self.newness_factor = newness_factor

    def calculate_size(
        self,
        signal: TradeSignal,
        config: CopyConfig,
        follower_portfolio_value_usd: Decimal,
        leader_portfolio_value_usd: Optional[Decimal] = None,
        risk_data: Optional[Dict[str, Any]] = None,
    ) -> Decimal:
        # Get base size
        base_size = self.base_strategy.calculate_size(
            signal, config, follower_portfolio_value_usd, leader_portfolio_value_usd
        )

        if not risk_data:
            return base_size

        # Apply risk adjustments
        risk_multiplier = Decimal("1")

        # Volatility adjustment
        volatility = risk_data.get("volatility_24h", 0)
        if volatility > 20:  # High volatility
            risk_multiplier -= self.volatility_factor * Decimal(str(min(volatility / 100, 1)))

        # Liquidity adjustment
        liquidity_usd = risk_data.get("liquidity_usd", 0)
        if liquidity_usd < 100000:  # Low liquidity
            risk_multiplier -= self.liquidity_factor * (1 - Decimal(str(min(liquidity_usd / 100000, 1))))

        # Token age adjustment
        token_age_days = risk_data.get("token_age_days", 365)
        if token_age_days < 30:  # New token
            risk_multiplier -= self.newness_factor * (1 - Decimal(str(token_age_days / 30)))

        # Ensure multiplier doesn't go below 0.1 (90% reduction max)
        risk_multiplier = max(risk_multiplier, Decimal("0.1"))

        adjusted_size = base_size * risk_multiplier

        return self.apply_bounds(adjusted_size, config)


def get_sizing_strategy(mode: SizingMode) -> SizingStrategy:
    """Get the appropriate sizing strategy for a mode."""
    strategies = {
        SizingMode.PERCENTAGE: PercentageSizing(),
        SizingMode.FIXED: FixedSizing(),
        SizingMode.PROPORTIONAL: ProportionalSizing(),
    }
    return strategies.get(mode, PercentageSizing())
