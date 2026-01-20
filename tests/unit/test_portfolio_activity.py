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
    last_requests = []

    def __init__(self, *_, **__):
        _DummyClient.last_requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        _DummyClient.last_requests.append({"url": url, "json": json})
        params = json["params"][0]
        # Handle both outgoing (fromAddress) and incoming (toAddress) requests
        if "fromAddress" in params:
            transfer = {
                "hash": "0x1",
                "from": params["fromAddress"],
                "to": "0xabc",
                "asset": "ETH",
                "value": 1,
                "category": "external",
                "rawContract": {"value": "0xde0b6b3a7640000", "address": None, "decimal": "0x12"},
                "metadata": {"blockTimestamp": "2025-01-01T00:00:00.000Z"},
            }
            return _DummyResponse({"result": {"transfers": [transfer]}})
        elif "toAddress" in params:
            transfer = {
                "hash": "0x2",
                "from": "0xdef",
                "to": params["toAddress"],
                "asset": "ETH",
                "value": 2,
                "category": "external",
                "rawContract": {"value": "0xde0b6b3a7640000", "address": None, "decimal": "0x12"},
                "metadata": {"blockTimestamp": "2025-01-02T00:00:00.000Z"},
            }
            return _DummyResponse({"result": {"transfers": [transfer]}})
        return _DummyResponse({"result": {"transfers": []}})


@pytest.mark.asyncio
async def test_fetch_activity_defaults_to_limit_of_ten(monkeypatch):
    captured = {}

    async def fake_fetch(address, chain, start, end, limit):
        captured["limit"] = limit
        return []

    # Mock the chain service to return a valid chain config
    class FakeChainConfig:
        alchemy_slug = "eth-mainnet"

    class FakeChainService:
        async def resolve_alias(self, chain):
            return FakeChainConfig()

    monkeypatch.setattr(pa, "get_chain_service", lambda: FakeChainService())
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
async def test_fetch_evm_activity_fetches_both_directions(monkeypatch):
    """Test that EVM activity fetches both outgoing and incoming transfers."""
    pa.settings.alchemy_api_key = "test"

    # Mock the Alchemy URL resolution to avoid Convex calls
    async def fake_resolve_url(chain):
        return "https://eth-mainnet.g.alchemy.com/v2/test"

    monkeypatch.setattr(pa, "_resolve_alchemy_url", fake_resolve_url)
    monkeypatch.setattr(pa.httpx, "AsyncClient", _DummyClient)

    events = await pa._fetch_evm_activity(
        address="0x123",
        chain="ethereum",
        start=None,
        end=None,
        limit=10,
    )

    # Should make 2 requests: one for outgoing (fromAddress), one for incoming (toAddress)
    assert len(_DummyClient.last_requests) == 2

    outgoing_req = _DummyClient.last_requests[0]["json"]["params"][0]
    incoming_req = _DummyClient.last_requests[1]["json"]["params"][0]

    assert "fromAddress" in outgoing_req
    assert "toAddress" in incoming_req
    assert outgoing_req["maxCount"] == hex(10)

    # Should return deduplicated events from both directions
    assert len(events) == 2
    timestamps = {e.timestamp for e in events}
    assert datetime(2025, 1, 1, tzinfo=timezone.utc) in timestamps
    assert datetime(2025, 1, 2, tzinfo=timezone.utc) in timestamps
