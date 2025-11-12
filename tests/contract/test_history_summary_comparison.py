from datetime import datetime, timezone, timedelta

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _window(days: int) -> tuple[datetime, datetime]:
    end = datetime(2024, 10, 1, tzinfo=timezone.utc)
    return end - timedelta(days=days), end


async def _fake_report(*_, **__):
    return {
        "baselineWindow": {"start": "2024-09-01T00:00:00Z", "end": "2024-09-30T00:00:00Z"},
        "comparisonWindow": {"start": "2024-10-01T00:00:00Z", "end": "2024-10-31T00:00:00Z"},
        "metricDeltas": [
            {
                "metric": "inflowUsd",
                "baselineValueUsd": 1000,
                "comparisonValueUsd": 1200,
                "deltaPct": 0.2,
                "direction": "up",
            }
        ],
        "thresholdFlags": [],
        "supportingTables": [],
    }


def test_history_comparison_contract(monkeypatch):
    monkeypatch.setattr("app.api.history_summary.generate_comparison_report", _fake_report)
    baseline_start, baseline_end = _window(30)
    comparison_start, comparison_end = _window(30)

    resp = client.post(
        "/wallets/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/history-summary/comparisons",
        json=
        {
            "chain": "ethereum",
            "baseline": {"start": baseline_start.isoformat(), "end": baseline_end.isoformat()},
            "comparison": {"start": comparison_start.isoformat(), "end": comparison_end.isoformat()},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metricDeltas"][0]["metric"] == "inflowUsd"
