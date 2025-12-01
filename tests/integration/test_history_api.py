from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(app)


def test_history_window_is_capped(monkeypatch):
    captured = {}

    async def fake_snapshot(**kwargs):
        captured.update(kwargs)
        return {
            "walletAddress": kwargs["address"],
            "chain": kwargs["chain"],
            "timeWindow": {
                "start": kwargs["start"].isoformat(),
                "end": kwargs["end"].isoformat(),
            },
            "bucketSize": "day",
            "totals": {"inflow": 0, "outflow": 0, "inflowUsd": 0, "outflowUsd": 0, "feeUsd": 0},
            "notableEvents": [],
            "buckets": [],
            "exportRefs": [],
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "cached": False,
        }, []

    monkeypatch.setattr("app.api.history_summary.get_history_snapshot", fake_snapshot)

    async def _fake_exports(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.api.history_summary.export_worker.list_exports_for_address", _fake_exports)

    resp = client.get(
        "/wallets/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/history-summary",
        params={
            "chain": "ethereum",
            "start": (datetime.utcnow() - timedelta(days=365)).isoformat(),
            "end": datetime.utcnow().isoformat(),
        },
    )
    assert resp.status_code == 200
    assert "totals" in resp.json()
    assert (captured["end"] - captured["start"]) <= timedelta(days=90)


def test_history_defaults_to_limit(monkeypatch):
    captured = {}

    async def fake_snapshot(**kwargs):
        captured.update(kwargs)
        return {
            "walletAddress": kwargs["address"],
            "chain": kwargs["chain"],
            "timeWindow": {
                "start": datetime.utcnow().isoformat(),
                "end": datetime.utcnow().isoformat(),
            },
            "bucketSize": "day",
            "totals": {"inflow": 0, "outflow": 0, "inflowUsd": 0, "outflowUsd": 0, "feeUsd": 0},
            "notableEvents": [],
            "buckets": [],
            "exportRefs": [],
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "cached": False,
        }, []

    monkeypatch.setattr("app.api.history_summary.get_history_snapshot", fake_snapshot)

    async def _fake_exports(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.api.history_summary.export_worker.list_exports_for_address", _fake_exports)

    resp = client.get(
        "/wallets/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/history-summary",
        params={"chain": "ethereum"},
    )
    assert resp.status_code == 200
    assert captured.get("limit") == settings.history_summary_default_limit
