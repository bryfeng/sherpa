from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ...config import settings


@dataclass
class PolicyLimits:
    max_leverage: float = settings.default_max_leverage
    max_notional_usd: float = settings.default_max_position_notional_usd
    max_daily_loss_usd: float = settings.default_max_daily_loss_usd
    per_trade_risk_cap_usd: float = settings.default_per_trade_risk_cap_usd
    allowed_markets: List[str] = field(default_factory=lambda: ["ETH-PERP", "BTC-PERP", "SOL-PERP"])
    max_open_positions: Optional[int] = 3


@dataclass
class PolicyEvaluation:
    ok: bool
    violations: List[str]
    limits: PolicyLimits


class PolicyManager:
    def __init__(self, context_manager=None) -> None:
        self.context_manager = context_manager

    def _load_from_context(self, session_id: Optional[str]) -> Optional[PolicyLimits]:
        if not self.context_manager or not session_id:
            return None
        convo = getattr(self.context_manager, "_conversations", {}).get(session_id)
        if not convo:
            return None
        data = getattr(convo, "user_preferences", {}).get("perps_policy")
        if not data:
            return None
        try:
            return PolicyLimits(**data)
        except Exception:
            return None

    def get_limits(self, session_id: Optional[str], user_id: Optional[str]):
        return {"session_id": session_id, "limits": self._load_from_context(session_id) or PolicyLimits()}

    def evaluate(self, context, symbol: str, leverage: float, notional: float, var_95: float) -> PolicyEvaluation:
        limits: PolicyLimits = context["limits"]
        violations: List[str] = []
        if limits.allowed_markets and symbol.upper() not in {m.upper() for m in limits.allowed_markets}:
            violations.append(f"Market {symbol} not in policy allowlist")
        if leverage > limits.max_leverage:
            violations.append(f"Leverage {leverage:.2f}x exceeds max {limits.max_leverage:.2f}x")
        if notional > limits.max_notional_usd:
            violations.append(f"Notional ${notional:,.2f} exceeds max ${limits.max_notional_usd:,.2f}")
        cap = min(limits.max_daily_loss_usd, limits.per_trade_risk_cap_usd)
        if cap > 0 and var_95 > cap:
            violations.append(f"Risk ${var_95:,.2f} breaches cap ${cap:,.2f}")
        return PolicyEvaluation(ok=not violations, violations=violations, limits=limits)
