from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, Optional, Literal

from pydantic import BaseModel, model_validator

from ...config import settings
from .policies import PolicyManager
from .providers import ProviderRegistry
from .providers.cex_proxy import CexProxyProvider
from .providers.gmx_v2 import GMXV2Provider
from .providers.perennial import PerennialProvider
from .risk import calculate_liquidation_price, compute_risk_metrics, suggest_position_size

logger = logging.getLogger(__name__)
_VOL_PRIORS = ("BTC", 0.45), ("ETH", 0.55), ("SOL", 0.75)


class SimulationRequest(BaseModel):
    symbol: str
    side: Literal["LONG", "SHORT"]
    notional_usd: Optional[float] = None
    quantity: Optional[float] = None
    max_leverage: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon_days: float = 1.0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    entry_price: Optional[float] = None

    model_config = dict(extra="forbid")


class SimulationResult(BaseModel):
    entry_price: float
    est_funding_apr: float
    liq_price: float
    fee_estimate: float
    var_95: float
    es_95: float
    max_drawdown_est: float
    position_size_suggestion: float
    policy_ok: bool
    policy_violations: list[str]
    notes: list[str]
    explainability: Dict[str, object]


class PerpsSimulator:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry
        self.var_conf = settings.default_var_conf
        self.kelly_cap = settings.default_kelly_cap
        self.maintenance_margin_rate = 0.005

    async def simulate(self, request: SimulationRequest, policy_manager: PolicyManager) -> SimulationResult:
        t0 = perf_counter()
        if not (request.notional_usd or request.quantity):
            raise ValueError("Provide notional_usd or quantity")
        snapshot = await self.registry.get_snapshot(request.symbol)
        entry = request.entry_price or snapshot.index_price
        notional = max(request.notional_usd or entry * (request.quantity or 0.0), 0.0)
        if request.max_leverage is not None and request.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if request.time_horizon_days <= 0:
            raise ValueError("time_horizon_days must be positive")
        leverage = max(1.0, min(request.max_leverage or settings.default_max_leverage, settings.default_max_leverage))
        context = policy_manager.get_limits(request.session_id, request.user_id)
        vol = next((vol for prefix, vol in _VOL_PRIORS if request.symbol.upper().startswith(prefix)), 0.6)
        risk = compute_risk_metrics(notional, vol, request.time_horizon_days, self.var_conf)
        policy = policy_manager.evaluate(context, request.symbol, leverage, notional, risk.var_95)
        funding = notional * snapshot.funding_apr_est * (request.time_horizon_days / 365.0)
        fee = notional * (snapshot.taker_fee_bps / 10_000)
        expected_return = funding - fee
        effective_cap = max(0.0, min(policy.limits.max_daily_loss_usd, policy.limits.per_trade_risk_cap_usd) + expected_return)
        suggestion, risk_pct = suggest_position_size(
            requested_notional=notional,
            limits_max_notional=policy.limits.max_notional_usd,
            limits_risk_cap=effective_cap,
            sigma=risk.sigma,
            z_score=risk.z_score,
            expected_edge=expected_return / notional if notional else 0.0,
            kelly_cap=self.kelly_cap,
        )
        liq_price = calculate_liquidation_price(entry, leverage, self.maintenance_margin_rate, request.side)
        es_adjusted = max(risk.es_95 - expected_return, 0.0)
        is_long = request.side == "LONG"
        tp = request.take_profit or entry * (1 + (0.02 if is_long else -0.02))
        sl = request.stop_loss or entry * (0.96 if is_long else 1.04)
        liq_diff_pct = (entry - liq_price) / entry * 100 if entry else 0.0
        logger.info(
            "perps.simulate.requests=1 perps.policy.blocked_count=%d perps.simulate.latency_ms=%.2f perps.risk.var95_usd=%.2f perps.risk.es95_usd=%.2f",
            0 if policy.ok else 1,
            (perf_counter() - t0) * 1000,
            risk.var_95,
            es_adjusted,
        )
        return SimulationResult(
            entry_price=entry,
            est_funding_apr=snapshot.funding_apr_est,
            liq_price=liq_price,
            fee_estimate=fee,
            var_95=risk.var_95,
            es_95=es_adjusted,
            max_drawdown_est=max(es_adjusted, risk.var_95),
            position_size_suggestion=suggestion,
            policy_ok=policy.ok,
            policy_violations=policy.violations,
            notes=["Simulation only – funding impact included"],
            explainability={
                "reasons": [
                    f"Liq price {abs(liq_diff_pct):.1f}% {'below' if liq_diff_pct >= 0 else 'above'} entry with {leverage:.1f}x",
                    f"Funding {snapshot.funding_apr_est*100:.2f}% APR → ${funding:.2f}",
                    f"VaR95 ${risk.var_95:,.2f}; ES95 ${es_adjusted:,.2f}",
                ],
                "risk_buckets": dict(zip(
                    ("market", "leverage", "funding", "liquidity"),
                    [
                        round(min(1.0, risk.sigma / 0.25), 3),
                        round(min(1.0, leverage / max(settings.default_max_leverage, 1e-6)), 3),
                        round(max(0.0, min(1.0, 0.5 - snapshot.funding_apr_est)), 3),
                        round(min(1.0, snapshot.est_impact_bps / 50.0), 3),
                    ],
                )),
                "suggestion": {
                    "size_usd": suggestion,
                    "leverage": leverage,
                    "tp_level": tp,
                    "sl_level": sl,
                    "daily_risk_budget_used_pct": risk_pct,
                },
            },
        )


def build_simulator() -> PerpsSimulator:
    providers = [p for p in (
        GMXV2Provider() if settings.enable_gmx else None,
        PerennialProvider() if settings.enable_perennial else None,
        CexProxyProvider(enabled=True) if settings.enable_cex_proxy else None,
    ) if p]
    return PerpsSimulator(ProviderRegistry(providers, settings.feature_flag_fake_perps))
