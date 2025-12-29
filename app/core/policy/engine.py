"""
Policy Engine

Unified policy enforcement system that evaluates all policies for an action.
Combines session, risk, and system policies into a single evaluation.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from ..wallet.models import SessionKey
from .models import (
    ActionContext,
    PolicyResult,
    PolicyViolation,
    RiskLevel,
    RiskPolicyConfig,
    SystemPolicyConfig,
    ViolationSeverity,
)
from .risk_policy import RiskPolicy
from .session_policy import SessionPolicy
from .system_policy import SystemPolicy


class PolicyEngine:
    """
    Unified policy evaluation engine.

    Evaluates actions against all policy layers:
    1. System Policy - Platform-wide rules (checked first)
    2. Session Policy - Session key constraints
    3. Risk Policy - User risk preferences

    Each layer can block or warn about actions.
    """

    def __init__(
        self,
        session_key: Optional[SessionKey] = None,
        risk_config: Optional[RiskPolicyConfig] = None,
        system_config: Optional[SystemPolicyConfig] = None,
    ):
        """
        Initialize the policy engine.

        Args:
            session_key: The session key to validate against (optional)
            risk_config: User risk preferences (defaults to sensible values)
            system_config: Platform system config (defaults to permissive)
        """
        self.session_policy = SessionPolicy(session_key) if session_key else None
        self.risk_policy = RiskPolicy(risk_config or RiskPolicyConfig())
        self.system_policy = SystemPolicy(system_config or SystemPolicyConfig())

    def evaluate(self, context: ActionContext) -> PolicyResult:
        """
        Evaluate an action against all policies.

        Returns a PolicyResult with approval status, violations, and risk assessment.

        Evaluation order:
        1. System policy (platform-wide blocks)
        2. Session policy (session key constraints)
        3. Risk policy (user preferences)
        """
        start_time = time.perf_counter()

        all_violations: list[PolicyViolation] = []
        all_warnings: list[PolicyViolation] = []

        # 1. System policy first (emergency stops, blocked contracts)
        system_violations = self.system_policy.evaluate(context)
        self._categorize_violations(system_violations, all_violations, all_warnings)

        # If system blocks, return immediately
        if any(v.severity == ViolationSeverity.BLOCK for v in system_violations):
            return self._build_result(
                all_violations,
                all_warnings,
                context,
                start_time,
            )

        # 2. Session policy (if session key provided)
        if self.session_policy:
            session_violations = self.session_policy.evaluate(context)
            self._categorize_violations(session_violations, all_violations, all_warnings)

            # If session blocks, return
            if any(v.severity == ViolationSeverity.BLOCK for v in session_violations):
                return self._build_result(
                    all_violations,
                    all_warnings,
                    context,
                    start_time,
                )

        # 3. Risk policy (user preferences)
        risk_violations = self.risk_policy.evaluate(context)
        self._categorize_violations(risk_violations, all_violations, all_warnings)

        return self._build_result(
            all_violations,
            all_warnings,
            context,
            start_time,
        )

    def is_operational(self) -> tuple[bool, Optional[str]]:
        """
        Check if the system is operational.

        Returns (is_operational, reason_if_not).
        """
        return self.system_policy.is_operational()

    def _categorize_violations(
        self,
        violations: list[PolicyViolation],
        all_violations: list[PolicyViolation],
        all_warnings: list[PolicyViolation],
    ) -> None:
        """Categorize violations into blocking and warnings."""
        for v in violations:
            if v.severity == ViolationSeverity.BLOCK:
                all_violations.append(v)
            else:
                all_warnings.append(v)

    def _build_result(
        self,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
        context: ActionContext,
        start_time: float,
    ) -> PolicyResult:
        """Build the final policy result."""
        # Calculate risk score
        risk_score, risk_level = self.risk_policy.calculate_risk_score(context)

        # Check if approval is required
        requires_approval, approval_reason = self.risk_policy.requires_approval(context)

        # If there are blocking violations, don't require approval
        # (it's already blocked)
        if violations:
            requires_approval = False
            approval_reason = None

        # Calculate evaluation time
        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        return PolicyResult(
            approved=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
            risk_level=risk_level,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
            evaluated_at=datetime.now(timezone.utc),
            evaluation_time_ms=evaluation_time_ms,
        )


# Convenience function for quick evaluation
async def evaluate_action(
    context: ActionContext,
    session_key: Optional[SessionKey] = None,
    risk_config: Optional[RiskPolicyConfig] = None,
    system_config: Optional[SystemPolicyConfig] = None,
) -> PolicyResult:
    """
    Evaluate an action against all policies.

    This is a convenience function that creates a PolicyEngine and evaluates
    the action in one call.
    """
    engine = PolicyEngine(
        session_key=session_key,
        risk_config=risk_config,
        system_config=system_config,
    )
    return engine.evaluate(context)
