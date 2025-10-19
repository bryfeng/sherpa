from . import MarketSnapshot, PerpsMarketDataProvider, _mock_snapshot


class CexProxyProvider(PerpsMarketDataProvider):
    name = "cex_proxy"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return _mock_snapshot(symbol)
