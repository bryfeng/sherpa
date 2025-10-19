from . import MarketSnapshot, PerpsMarketDataProvider, _mock_snapshot


class PerennialProvider(PerpsMarketDataProvider):
    name = "perennial"

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return _mock_snapshot(symbol)
