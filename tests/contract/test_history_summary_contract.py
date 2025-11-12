from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


async def _fake_snapshot(*_, **__):
    snapshot = {
        "walletAddress": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "chain": "ethereum",
        "timeWindow": {
            "start": datetime(2024, 10, 1, tzinfo=timezone.utc).isoformat(),
            "end": datetime(2024, 10, 31, tzinfo=timezone.utc).isoformat(),
        },
        "bucketSize": "week",
        "totals": {
            "inflow": 1.0,
            "outflow": 0.5,
            "inflowUsd": 1000.0,
            "outflowUsd": 500.0,
            "feeUsd": 5.0,
        },
        "notableEvents": [],
        "buckets": [],
        "exportRefs": [],
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cached": False,
    }
    events = [
        {
            "tx_hash": "0xabc",
            "timestamp": datetime(2024, 10, 15, tzinfo=timezone.utc).isoformat(),
            "direction": "inflow",
            "symbol": "ETH",
            "native_amount": 1.0,
            "usd_value": 1000.0,
            "counterparty": "0x123",
            "protocol": "external",
            "fee_native": 0.0,
            "chain": "ethereum",
        }
    ]
    return snapshot, events


def test_history_summary_contract(monkeypatch):
    monkeypatch.setattr("app.api.history_summary.get_history_snapshot", _fake_snapshot)

    async def _fake_exports(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.api.history_summary.export_worker.list_exports_for_address", _fake_exports)

    resp = client.get(
        "/wallets/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/history-summary",
        params={
            "chain": "ethereum",
            "start": datetime(2024, 10, 1, tzinfo=timezone.utc).isoformat(),
            "end": datetime(2024, 10, 31, tzinfo=timezone.utc).isoformat(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain"] == "ethereum"
    assert "totals" in body and "inflowUsd" in body["totals"]
