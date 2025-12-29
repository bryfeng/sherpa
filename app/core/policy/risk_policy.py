"""
Risk Policy

Evaluates actions against user-configurable risk preferences.
Checks position limits, daily limits, slippage, gas costs, and concentration.
"""

from decimal import Decimal
from typing import List, Optional

from .models import (
    ActionContext,
    PolicyType,
    PolicyViolation,
    RiskLevel,
    RiskPolicyConfig,
    ViolationSeverity,
)


class RiskPolicy:
    """
    Evaluates actions against user risk preferences.

    Risk policies are user-configurable and focus on:
    - Position limits (max % in single asset)
    - Daily trading limits (volume, loss)
    - Transaction size limits
    - Slippage tolerance
    - Gas cost limits
    """

    def __init__(self, config: RiskPolicyConfig):
        self.config = config

    def evaluate(self, context: ActionContext) -> List[PolicyViolation]:
        """
        Evaluate an action against user risk preferences.

        Returns a list of violations (may include warnings).
        """
        if not self.config.enabled:
            return []

        violations: List[PolicyViolation] = []

        # Check transaction size limits
        tx_violations = self._check_transaction_limits(context.value_usd)
        violations.extend(tx_violations)

        # Check position limits
        if self.config.check_position_limits:
            position_violations = self._check_position_limits(context)
            violations.extend(position_violations)

        # Check daily limits
        if self.config.check_daily_limits:
            daily_violations = self._check_daily_limits(context)
            violations.extend(daily_violations)

        # Check slippage
        if self.config.check_slippage and context.slippage_percent is not None:
            slippage_violations = self._check_slippage(context.slippage_percent)
            violations.extend(slippage_violations)

        # Check gas costs
        if self.config.check_gas and context.estimated_gas_usd is not None:
            gas_violations = self._check_gas_costs(
                context.estimated_gas_usd,
                context.value_usd,
            )
            violations.extend(gas_violations)

        return violations

    def calculate_risk_score(self, context: ActionContext) -> tuple[float, RiskLevel]:
        """
        Calculate a risk score for the action.

        Returns (score, level) where score is 0.0 to 1.0.
        """
        score = 0.0
        factors = []

        # Transaction size factor (0-0.3)
        if context.value_usd > Decimal("0"):
            size_ratio = float(context.value_usd / self.config.max_single_tx_usd)
            size_score = min(size_ratio * 0.3, 0.3)
            score += size_score
            factors.append(("transaction_size", size_score))

        # Position concentration factor (0-0.3)
        if context.current_position_percent is not None:
            position_ratio = context.current_position_percent / self.config.max_position_percent
            position_score = min(position_ratio * 0.3, 0.3)
            score += position_score
            factors.append(("position_concentration", position_score))

        # Slippage factor (0-0.2)
        if context.slippage_percent is not None:
            slippage_ratio = context.slippage_percent / self.config.max_slippage_percent
            slippage_score = min(slippage_ratio * 0.2, 0.2)
            score += slippage_score
            factors.append(("slippage", slippage_score))

        # Gas cost factor (0-0.1)
        if context.estimated_gas_usd and context.value_usd > Decimal("0"):
            gas_percent = float(context.estimated_gas_usd / context.value_usd * 100)
            gas_ratio = gas_percent / self.config.max_gas_percent
            gas_score = min(gas_ratio * 0.1, 0.1)
            score += gas_score
            factors.append(("gas_cost", gas_score))

        # Daily usage factor (0-0.1)
        if context.daily_volume_usd and self.config.max_daily_volume_usd > Decimal("0"):
            daily_ratio = float(context.daily_volume_usd / self.config.max_daily_volume_usd)
            daily_score = min(daily_ratio * 0.1, 0.1)
            score += daily_score
            factors.append(("daily_usage", daily_score))

        # Determine risk level
        if score < 0.25:
            level = RiskLevel.LOW
        elif score < 0.5:
            level = RiskLevel.MEDIUM
        elif score < 0.75:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        return min(score, 1.0), level

    def requires_approval(self, context: ActionContext) -> tuple[bool, Optional[str]]:
        """
        Check if this action requires human approval.

        Returns (requires_approval, reason).
        """
        if context.value_usd > self.config.require_approval_above_usd:
            return True, f"Transaction value ${context.value_usd} exceeds approval threshold ${self.config.require_approval_above_usd}"

        # High risk actions require approval
        _, risk_level = self.calculate_risk_score(context)
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return True, f"Action has {risk_level.value} risk level"

        return False, None

    def _check_transaction_limits(self, value_usd: Decimal) -> List[PolicyViolation]:
        """Check transaction size limits."""
        violations: List[PolicyViolation] = []

        if value_usd > self.config.max_single_tx_usd:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="max_single_tx",
                severity=ViolationSeverity.BLOCK,
                message=f"Transaction ${value_usd} exceeds your max single transaction limit of ${self.config.max_single_tx_usd}",
                details={
                    "requestedValue": str(value_usd),
                    "maxAllowed": str(self.config.max_single_tx_usd),
                },
                suggestion=f"Reduce transaction to ${self.config.max_single_tx_usd} or update your risk settings",
            ))

        return violations

    def _check_position_limits(self, context: ActionContext) -> List[PolicyViolation]:
        """Check position concentration limits."""
        violations: List[PolicyViolation] = []

        # Check absolute position value
        if context.value_usd > self.config.max_position_value_usd:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="max_position_value",
                severity=ViolationSeverity.WARN,
                message=f"This would create a position larger than your ${self.config.max_position_value_usd} limit",
                details={
                    "requestedValue": str(context.value_usd),
                    "maxPositionValue": str(self.config.max_position_value_usd),
                },
                suggestion="Consider splitting into smaller positions",
            ))

        # Check portfolio percentage (if portfolio context available)
        if context.portfolio_value_usd and context.portfolio_value_usd > Decimal("0"):
            new_position_percent = float(context.value_usd / context.portfolio_value_usd * 100)

            # Add existing position if provided
            if context.current_position_percent:
                new_position_percent += context.current_position_percent

            if new_position_percent > self.config.max_position_percent:
                violations.append(PolicyViolation(
                    policy_type=PolicyType.RISK,
                    policy_name="max_position_percent",
                    severity=ViolationSeverity.WARN,
                    message=f"Position would be {new_position_percent:.1f}% of portfolio (limit: {self.config.max_position_percent}%)",
                    details={
                        "newPositionPercent": new_position_percent,
                        "currentPositionPercent": context.current_position_percent or 0,
                        "maxPositionPercent": self.config.max_position_percent,
                        "portfolioValueUsd": str(context.portfolio_value_usd),
                    },
                    suggestion=f"Keep position under {self.config.max_position_percent}% for diversification",
                ))

        return violations

    def _check_daily_limits(self, context: ActionContext) -> List[PolicyViolation]:
        """Check daily trading limits."""
        violations: List[PolicyViolation] = []

        # Check daily volume limit
        if context.daily_volume_usd is not None:
            new_daily_volume = context.daily_volume_usd + context.value_usd
            if new_daily_volume > self.config.max_daily_volume_usd:
                remaining = self.config.max_daily_volume_usd - context.daily_volume_usd
                violations.append(PolicyViolation(
                    policy_type=PolicyType.RISK,
                    policy_name="daily_volume_limit",
                    severity=ViolationSeverity.WARN,
                    message=f"Transaction would exceed your daily volume limit. Remaining: ${max(remaining, Decimal('0'))}",
                    details={
                        "currentDailyVolume": str(context.daily_volume_usd),
                        "requestedValue": str(context.value_usd),
                        "maxDailyVolume": str(self.config.max_daily_volume_usd),
                        "remaining": str(max(remaining, Decimal("0"))),
                    },
                    suggestion="Consider waiting until tomorrow or adjusting your daily limits",
                ))

        # Check daily loss limit
        if context.daily_loss_usd is not None:
            if context.daily_loss_usd >= self.config.max_daily_loss_usd:
                violations.append(PolicyViolation(
                    policy_type=PolicyType.RISK,
                    policy_name="daily_loss_limit",
                    severity=ViolationSeverity.BLOCK,
                    message=f"Daily loss limit of ${self.config.max_daily_loss_usd} has been reached",
                    details={
                        "currentDailyLoss": str(context.daily_loss_usd),
                        "maxDailyLoss": str(self.config.max_daily_loss_usd),
                    },
                    suggestion="Trading is paused for today. Limits reset at midnight UTC.",
                ))

        return violations

    def _check_slippage(self, slippage_percent: float) -> List[PolicyViolation]:
        """Check slippage tolerance."""
        violations: List[PolicyViolation] = []

        if slippage_percent > self.config.max_slippage_percent:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="max_slippage",
                severity=ViolationSeverity.BLOCK,
                message=f"Slippage {slippage_percent}% exceeds your maximum tolerance of {self.config.max_slippage_percent}%",
                details={
                    "requestedSlippage": slippage_percent,
                    "maxSlippage": self.config.max_slippage_percent,
                },
                suggestion="Wait for better liquidity or increase your slippage tolerance",
            ))
        elif slippage_percent > self.config.warn_slippage_percent:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="slippage_warning",
                severity=ViolationSeverity.WARN,
                message=f"Slippage {slippage_percent}% is above recommended {self.config.warn_slippage_percent}%",
                details={
                    "requestedSlippage": slippage_percent,
                    "warnThreshold": self.config.warn_slippage_percent,
                    "maxSlippage": self.config.max_slippage_percent,
                },
                suggestion="Consider waiting for better liquidity conditions",
            ))

        return violations

    def _check_gas_costs(
        self,
        gas_usd: Decimal,
        value_usd: Decimal,
    ) -> List[PolicyViolation]:
        """Check gas cost as percentage of transaction value."""
        violations: List[PolicyViolation] = []

        if value_usd <= Decimal("0"):
            return violations

        gas_percent = float(gas_usd / value_usd * 100)

        if gas_percent > self.config.max_gas_percent:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="max_gas_percent",
                severity=ViolationSeverity.WARN,
                message=f"Gas cost ${gas_usd} is {gas_percent:.1f}% of transaction (limit: {self.config.max_gas_percent}%)",
                details={
                    "gasUsd": str(gas_usd),
                    "gasPercent": gas_percent,
                    "maxGasPercent": self.config.max_gas_percent,
                    "transactionValue": str(value_usd),
                },
                suggestion="Consider waiting for lower gas prices or increasing transaction size",
            ))
        elif gas_percent > self.config.warn_gas_percent:
            violations.append(PolicyViolation(
                policy_type=PolicyType.RISK,
                policy_name="gas_warning",
                severity=ViolationSeverity.INFO,
                message=f"Gas cost is {gas_percent:.1f}% of transaction value",
                details={
                    "gasUsd": str(gas_usd),
                    "gasPercent": gas_percent,
                    "warnThreshold": self.config.warn_gas_percent,
                },
            ))

        return violations
