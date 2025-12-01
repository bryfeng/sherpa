import asyncio
from datetime import datetime, timezone

import pytest

from app.services import portfolio_activity as pa


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummyClient:
    last_request = None

    def __init__(self, *_, **__):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        _DummyClient.last_request = {"url": url, "json": json}
        transfer = {
            "hash": "0x1",
            "from": json["params"][0]["fromAddress"],
            "to": "0xabc",
            "asset": "ETH",
            "value": 1,
            "category": "external",
            "rawContract": {"value": "0xde0b6b3a7640000", "address": None, "decimal": "0x12"},
            "metadata": {"blockTimestamp": "2025-01-01T00:00:00.000Z"},
        }
        return _DummyResponse({"result": {"transfers": [transfer]}})


@pytest.mark.asyncio
async def test_fetch_activity_defaults_to_limit_of_ten(monkeypatch):
    captured = {}

    async def fake_fetch(address, chain, start, end, limit):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(pa, "_fetch_evm_activity", fake_fetch)
    await pa.fetch_activity(
        address="0x123",
        chain="ethereum",
        start=None,
        end=None,
        limit=None,
    )
    assert captured["limit"] == 10


@pytest.mark.asyncio
async def test_fetch_evm_activity_uses_from_only(monkeypatch):
    pa.settings.alchemy_api_key = "test"
    monkeypatch.setattr(pa.httpx, "AsyncClient", _DummyClient)

    events = await pa._fetch_evm_activity(
        address="0x123",
        chain="ethereum",
        start=None,
        end=None,
        limit=10,
    )

    assert _DummyClient.last_request is not None
    request_filter = _DummyClient.last_request["json"]["params"][0]
    assert "fromAddress" in request_filter
    assert "toAddress" not in request_filter
    assert request_filter["maxCount"] == hex(10)
    assert len(events) == 1
    assert events[0].timestamp == datetime(2025, 1, 1, tzinfo=timezone.utc)
