from __future__ import annotations

from collections import namedtuple
from typing import Sequence

_FIELDS = (
    "index_price",
    "funding_apr_est",
    "taker_fee_bps",
    "maker_fee_bps",
    "min_tick",
    "min_qty",
    "est_impact_bps",
    "oracle_source",
)
_RAW = {
    "ETH-PERP": (3200.0, 0.05, 5.0, 2.5, 0.1, 0.01, 6.0, "mock:coingecko"),
    "BTC-PERP": (65000.0, 0.03, 4.0, 2.0, 1.0, 0.001, 4.0, "mock:coingecko"),
    "SOL-PERP": (175.0, 0.08, 6.0, 3.0, 0.01, 0.1, 8.0, "mock:coingecko"),
}
MarketSnapshot = namedtuple("MarketSnapshot", ("symbol",) + _FIELDS)


class PerpsMarketDataProvider:
    name = "perps"
    enabled = True

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:  # pragma: no cover
        raise NotImplementedError


def _mock_snapshot(symbol: str) -> MarketSnapshot:
    key = symbol.upper()
    data = _RAW.get(key)
    if not data:
        base = _RAW["ETH-PERP"]
        data = (
            base[0] * 0.5,
            base[1] * 0.8,
            base[2],
            base[3],
            base[4],
            base[5],
            base[6] * 1.2,
            base[7],
        )
        _RAW[key] = data
    return MarketSnapshot(key, *data)


class ProviderRegistry:
    def __init__(self, providers: Sequence[PerpsMarketDataProvider], use_mocks: bool) -> None:
        self.providers = [p for p in providers if getattr(p, "enabled", True)]
        self.use_mocks = use_mocks

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        if self.use_mocks:
            return _mock_snapshot(symbol)
        for provider in self.providers:
            try:
                snapshot = await provider.get_market_snapshot(symbol)
                if snapshot:
                    return snapshot
            except Exception:  # pragma: no cover
                continue
        return _mock_snapshot(symbol)
