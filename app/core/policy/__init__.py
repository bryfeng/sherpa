"""
Policy Engine Module

Unified policy enforcement for the Sherpa platform.
"""

from .engine import PolicyEngine, evaluate_action
from .models import (
    ActionContext,
    FeePolicyConfig,
    PolicyResult,
    PolicyType,
    PolicyViolation,
    RiskLevel,
    RiskPolicyConfig,
    SystemPolicyConfig,
    ViolationSeverity,
)
from .fee_policy import FeePolicy
from .risk_policy import RiskPolicy
from .session_policy import SessionPolicy
from .system_policy import SystemPolicy

__all__ = [
    # Engine
    "PolicyEngine",
    "evaluate_action",
    # Policies
    "SessionPolicy",
    "RiskPolicy",
    "SystemPolicy",
    "FeePolicy",
    # Models
    "ActionContext",
    "FeePolicyConfig",
    "PolicyResult",
    "PolicyType",
    "PolicyViolation",
    "RiskLevel",
    "RiskPolicyConfig",
    "SystemPolicyConfig",
    "ViolationSeverity",
]
