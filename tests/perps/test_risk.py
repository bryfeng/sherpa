from app.core.perps.risk import calculate_liquidation_price, compute_risk_metrics, suggest_position_size


def test_liquidation_behaviour():
    long_liq = calculate_liquidation_price(2000.0, leverage=3.0, maintenance_margin_rate=0.005, side="LONG")
    short_liq = calculate_liquidation_price(2000.0, leverage=3.0, maintenance_margin_rate=0.005, side="SHORT")
    assert 0 < long_liq < 2000.0
    assert short_liq > 2000.0


def test_var_es_scaling():
    short = compute_risk_metrics(1000.0, annualized_vol=0.3, horizon_days=1, confidence=0.95)
    long = compute_risk_metrics(1000.0, annualized_vol=0.6, horizon_days=7, confidence=0.95)
    assert long.var_95 > short.var_95
    assert long.es_95 > short.es_95


def test_sizing_clamps_and_budget():
    suggested, used = suggest_position_size(
        requested_notional=1500.0,
        limits_max_notional=5000.0,
        limits_risk_cap=300.0,
        sigma=0.1,
        z_score=1.65,
        expected_edge=0.2,
        kelly_cap=0.5,
    )
    assert suggested > 0
    assert suggested <= 300.0 / (0.1 * 1.65) + 1e-6
    assert used <= 100.0
