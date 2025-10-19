import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.perps.providers import _RAW
from app.main import app

client = TestClient(app)


def sim(payload: dict) -> dict:
    resp = client.post("/perps/simulate", json=payload)
    assert resp.status_code == 200, resp.json()
    return resp.json()


@pytest.fixture(autouse=True)
def enable_mocks():
    original = settings.feature_flag_fake_perps
    settings.feature_flag_fake_perps = True
    yield
    settings.feature_flag_fake_perps = original


def test_simulate_policy_horizon_and_funding(monkeypatch):
    base = sim({"symbol": "ETH-PERP", "side": "LONG", "notional_usd": 1500.0, "time_horizon_days": 1, "session_id": "metrics"})
    blocked = sim({"symbol": "ETH-PERP", "side": "LONG", "notional_usd": 20_000.0, "time_horizon_days": 1, "session_id": "metrics"})
    assert base["policy_ok"] and base["var_95"] > 0
    assert not blocked["policy_ok"] and blocked["policy_violations"]

    short, long = (
        sim({"symbol": "BTC-PERP", "side": "LONG", "notional_usd": 2000.0, "time_horizon_days": horizon, "session_id": "horizon"})
        for horizon in (1, 7)
    )
    assert long["var_95"] > short["var_95"]
    assert long["position_size_suggestion"] < short["position_size_suggestion"]

    original = _RAW["SOL-PERP"]
    positive = sim({"symbol": "SOL-PERP", "side": "LONG", "notional_usd": 1200.0, "time_horizon_days": 1, "session_id": "funding"})
    monkeypatch.setitem(_RAW, "SOL-PERP", (
        original[0],
        -abs(original[1]),
        *original[2:],
    ))
    negative = sim({"symbol": "SOL-PERP", "side": "LONG", "notional_usd": 1200.0, "time_horizon_days": 1, "session_id": "funding"})
    monkeypatch.setitem(_RAW, "SOL-PERP", original)
    assert negative["position_size_suggestion"] < positive["position_size_suggestion"]
    assert negative["es_95"] >= positive["es_95"]
