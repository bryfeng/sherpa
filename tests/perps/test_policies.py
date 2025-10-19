from app.core.perps.policies import PolicyManager


def test_policy_detects_market_leverage_and_risk():
    manager = PolicyManager()
    context = manager.get_limits(None, None)
    context["limits"].allowed_markets = ["ETH-PERP"]
    evaluation = manager.evaluate(context, symbol="BTC-PERP", leverage=2.0, notional=1000.0, var_95=50.0)
    assert not evaluation.ok and any("allowlist" in v for v in evaluation.violations)

    context = manager.get_limits(None, None)
    limits = context["limits"]
    limits.max_leverage = 2.0
    limits.max_notional_usd = 1000.0
    limits.max_daily_loss_usd = 200.0
    limits.per_trade_risk_cap_usd = 150.0
    evaluation = manager.evaluate(context, symbol="ETH-PERP", leverage=3.0, notional=1500.0, var_95=250.0)
    assert not evaluation.ok
    assert any("Leverage" in v for v in evaluation.violations)
    assert any("Risk" in v for v in evaluation.violations)
