from . import MarketSnapshot, PerpsMarketDataProvider, _mock_snapshot


class GMXV2Provider(PerpsMarketDataProvider):
    name = "gmx_v2"

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return _mock_snapshot(symbol)
