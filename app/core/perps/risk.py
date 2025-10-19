from __future__ import annotations

from collections import namedtuple
from statistics import NormalDist

RiskMetrics = namedtuple("RiskMetrics", "var_95 es_95 sigma z_score")


def compute_risk_metrics(notional_usd: float, annualized_vol: float, horizon_days: float, confidence: float) -> RiskMetrics:
    nd = NormalDist()
    z = nd.inv_cdf(confidence)
    sigma = abs(annualized_vol) * (horizon_days / 365.0) ** 0.5
    notional = max(notional_usd, 0.0)
    var = max(notional * sigma * z, 0.0)
    es = max(notional * sigma * nd.pdf(z) / (1 - confidence), 0.0)
    return RiskMetrics(var, es, sigma, z)


def calculate_liquidation_price(entry_price: float, leverage: float, maintenance_margin_rate: float, side: str) -> float:
    entry = max(entry_price, 1e-8)
    lev = max(leverage, 1e-6)
    mmr = max(maintenance_margin_rate, 0.0)
    return max(entry * (1 - 1 / lev + mmr) if side.upper() == "LONG" else entry * (1 + 1 / lev - mmr), 0.0)


def suggest_position_size(requested_notional: float, limits_max_notional: float, limits_risk_cap: float, sigma: float, z_score: float, expected_edge: float, kelly_cap: float):
    base = min(limits_max_notional or requested_notional or 0.0, requested_notional or limits_max_notional or 0.0)
    sigma = max(sigma, 1e-8)
    z = max(z_score, 1e-8)
    risk_cap = max(limits_risk_cap, 0.0)
    safe = base if risk_cap <= 0 else min(base, risk_cap / (sigma * z))
    variance = sigma ** 2
    if variance > 0 and expected_edge > 0:
        safe = min(safe, max(0.0, min(expected_edge / variance, kelly_cap)) * (limits_max_notional or safe))
    if expected_edge < 0:
        safe *= max(0.0, 1 + max(-0.5, expected_edge * 500))
    suggested = max(0.0, safe)
    if requested_notional:
        suggested = min(suggested, requested_notional)
    used_pct = 0.0
    if risk_cap > 0 and suggested > 0:
        used_pct = min(100.0, (suggested * sigma * z) / risk_cap * 100)
    return suggested, used_pct
