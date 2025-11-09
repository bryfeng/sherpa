from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..cache import cache
from ..providers.coingecko import CoingeckoProvider
from ..services.address import normalize_chain, is_supported_chain, is_valid_address_for_chain


@dataclass(frozen=True)
class RangeSetting:
    key: str
    days_param: str
    ohlc_param: Union[str, int]
    interval: Optional[str]


_RANGE_SETTINGS: Dict[str, RangeSetting] = {
    "1d": RangeSetting(key="1d", days_param="1", ohlc_param=1, interval="hourly"),
    "7d": RangeSetting(key="7d", days_param="7", ohlc_param=7, interval="hourly"),
    "30d": RangeSetting(key="30d", days_param="30", ohlc_param=30, interval="daily"),
    "90d": RangeSetting(key="90d", days_param="90", ohlc_param=90, interval="daily"),
    "180d": RangeSetting(key="180d", days_param="180", ohlc_param=180, interval="daily"),
    "365d": RangeSetting(key="365d", days_param="365", ohlc_param=365, interval="daily"),
    "max": RangeSetting(key="max", days_param="max", ohlc_param="max", interval="daily"),
}

_DEFAULT_RANGE = _RANGE_SETTINGS["7d"]


def _normalize_range(range_key: str) -> RangeSetting:
    if not range_key:
        return _DEFAULT_RANGE
    normalized = range_key.lower()
    return _RANGE_SETTINGS.get(normalized, _DEFAULT_RANGE)


def _normalize_price_points(raw: List[List[Any]], value_key: str) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for entry in raw or []:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        timestamp, value = entry[0], entry[1]
        try:
            ts = int(timestamp)
            val = float(value)
        except (TypeError, ValueError):
            continue
        points.append({"time": ts, value_key: val})
    return points


def _normalize_candles(raw: List[List[Any]]) -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    for entry in raw or []:
        if not isinstance(entry, list) or len(entry) < 5:
            continue
        try:
            ts = int(entry[0])
            open_p = float(entry[1])
            high_p = float(entry[2])
            low_p = float(entry[3])
            close_p = float(entry[4])
        except (TypeError, ValueError):
            continue
        candles.append(
            {
                "time": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
            }
        )
    return candles


def _compute_stats(price_points: List[Dict[str, float]]) -> Dict[str, Any]:
    if not price_points:
        return {}

    open_price = price_points[0]["price"]
    close_price = price_points[-1]["price"]
    high_point = max(price_points, key=lambda p: p["price"])
    low_point = min(price_points, key=lambda p: p["price"])

    change_abs = close_price - open_price
    change_pct = (change_abs / open_price * 100.0) if open_price else None

    return {
        "open": open_price,
        "close": close_price,
        "latest": close_price,
        "change_abs": change_abs,
        "change_pct": change_pct,
        "high": high_point["price"],
        "high_time": high_point["time"],
        "low": low_point["price"],
        "low_time": low_point["time"],
        "samples": len(price_points),
        "range_start": price_points[0]["time"],
        "range_end": price_points[-1]["time"],
    }


class TokenChartService:
    """Fetch and cache token price chart data from CoinGecko."""

    def __init__(
        self,
        *,
        provider: Optional[CoingeckoProvider] = None,
        cache_ttl_seconds: int = 180,
    ) -> None:
        self._provider = provider or CoingeckoProvider()
        self._cache_ttl = max(30, cache_ttl_seconds)
        self._lock = asyncio.Lock()

    async def get_token_chart(
        self,
        *,
        coin_id: Optional[str] = None,
        symbol: Optional[str] = None,
        contract_address: Optional[str] = None,
        chain: str = "ethereum",
        range_key: str = "7d",
        vs_currency: str = "usd",
        include_candles: bool = True,
    ) -> Dict[str, Any]:
        if not await self._provider.ready():
            raise RuntimeError("CoinGecko provider unavailable")

        if not coin_id and not contract_address and not symbol:
            raise ValueError("Provide coin_id, symbol, or contract_address")

        normalized_vs = vs_currency.lower()
        range_setting = _normalize_range(range_key)

        cache_key = (
            f"token_chart:{(coin_id or symbol or '').lower()}:{(contract_address or '').lower()}"
            f":{normalized_vs}:{range_setting.key}:{'ohlc' if include_candles else 'line'}"
        )

        cached_payload = await cache.get(cache_key)
        if cached_payload:
            payload = copy.deepcopy(cached_payload)
            payload["cached"] = True
            return payload

        async with self._lock:
            cached_payload = await cache.get(cache_key)
            if cached_payload:
                payload = copy.deepcopy(cached_payload)
                payload["cached"] = True
                return payload

            payload = await self._build_chart_payload(
                coin_id=coin_id,
                symbol=symbol,
                contract_address=contract_address,
                chain=chain,
                range_setting=range_setting,
                vs_currency=normalized_vs,
                include_candles=include_candles,
            )
            await cache.set(cache_key, payload, ttl=self._cache_ttl)

        result = copy.deepcopy(payload)
        result["cached"] = False
        return result

    async def _build_chart_payload(
        self,
        *,
        coin_id: Optional[str],
        symbol: Optional[str],
        contract_address: Optional[str],
        chain: str,
        range_setting: RangeSetting,
        vs_currency: str,
        include_candles: bool,
    ) -> Dict[str, Any]:
        metadata, resolved_coin_id = await self._resolve_metadata(
            coin_id=coin_id,
            symbol=symbol,
            contract_address=contract_address,
            chain=chain,
        )
        if not resolved_coin_id:
            raise ValueError("Unable to resolve CoinGecko identifier for token")

        market_chart = await self._provider.get_coin_market_chart(
            resolved_coin_id,
            vs_currency=vs_currency,
            days=range_setting.days_param,
            interval=range_setting.interval,
        )
        if not market_chart or "prices" not in market_chart:
            raise ValueError("CoinGecko did not return market chart data")

        price_points = _normalize_price_points(market_chart.get("prices", []), "price")
        if not price_points:
            raise ValueError("Price series is empty for requested range")

        market_caps = _normalize_price_points(market_chart.get("market_caps", []), "market_cap")
        total_volumes = _normalize_price_points(market_chart.get("total_volumes", []), "volume")

        candles: List[Dict[str, float]] = []
        if include_candles and range_setting.ohlc_param:
            ohlc_data = await self._provider.get_coin_ohlc(
                resolved_coin_id,
                vs_currency=vs_currency,
                days=range_setting.ohlc_param,
            )
            candles = _normalize_candles(ohlc_data)

        stats = _compute_stats(price_points)

        payload = {
            "success": True,
            "metadata": metadata,
            "coin_id": resolved_coin_id,
            "range": range_setting.key,
            "vs_currency": vs_currency,
            "series": {
                "prices": price_points,
                "market_caps": market_caps,
                "total_volumes": total_volumes,
            },
            "candles": candles,
            "stats": stats,
            "sources": [
                {"label": "CoinGecko", "href": "https://www.coingecko.com"},
            ],
            "interval": range_setting.interval,
        }
        return payload

    async def _resolve_metadata(
        self,
        *,
        coin_id: Optional[str],
        symbol: Optional[str],
        contract_address: Optional[str],
        chain: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        if contract_address:
            normalized_chain = normalize_chain(chain)
            if not is_supported_chain(normalized_chain):
                raise ValueError(f"Unsupported chain '{chain}'")
            if not is_valid_address_for_chain(contract_address, normalized_chain):
                raise ValueError("Invalid contract address for specified chain")
            if normalized_chain != "ethereum":
                raise ValueError("Contract lookups currently supported on ethereum only")

            token_info = await self._provider.get_token_info(contract_address)
            if not token_info:
                raise ValueError("Token not found on CoinGecko for provided address")

            resolved_coin_id = token_info.get("id")
            if not resolved_coin_id:
                raise ValueError("CoinGecko response missing token id")

            metadata = {
                "symbol": token_info.get("symbol"),
                "name": token_info.get("name"),
                "image": token_info.get("image"),
                "contract_address": token_info.get("contract_address") or contract_address.lower(),
                "chain": normalized_chain,
                "decimals": token_info.get("decimals"),
                "platforms": token_info.get("platforms") or {},
            }
            return metadata, resolved_coin_id

        normalized_coin_id: Optional[str]
        if coin_id:
            normalized_coin_id = coin_id.lower()
        elif symbol:
            normalized_coin_id = await self._lookup_coin_id_by_symbol(symbol)
        else:
            normalized_coin_id = None

        if not normalized_coin_id:
            raise ValueError("Coin not found on CoinGecko")

        coin_metadata = await self._provider.get_coin_metadata(normalized_coin_id)
        if not coin_metadata:
            raise ValueError("Coin not found on CoinGecko")

        metadata = {
            "symbol": coin_metadata.get("symbol"),
            "name": coin_metadata.get("name"),
            "image": coin_metadata.get("image"),
            "contract_address": coin_metadata.get("contract_address"),
            "chain": "ethereum" if coin_metadata.get("contract_address") else None,
            "decimals": None,
            "platforms": coin_metadata.get("platforms") if "platforms" in coin_metadata else {},
            "market_cap_rank": coin_metadata.get("market_cap_rank"),
        }
        resolved_coin_id = coin_metadata.get("id", normalized_coin_id)
        return metadata, resolved_coin_id

    async def _lookup_coin_id_by_symbol(self, symbol: str) -> Optional[str]:
        query = symbol.strip()
        if not query:
            return None

        results = await self._provider.search_coins(query, limit=10)
        if not results:
            return None

        symbol_lower = query.lower()
        for entry in results:
            entry_symbol = str(entry.get("symbol", "")).lower()
            if entry_symbol == symbol_lower:
                coin_id = entry.get("id")
                if coin_id:
                    return str(coin_id).lower()

        first_id = results[0].get("id")
        return str(first_id).lower() if first_id else None


token_chart_service = TokenChartService()


async def get_token_chart(
    *,
    coin_id: Optional[str] = None,
    symbol: Optional[str] = None,
    contract_address: Optional[str] = None,
    chain: str = "ethereum",
    range_key: str = "7d",
    vs_currency: str = "usd",
    include_candles: bool = True,
) -> Dict[str, Any]:
    return await token_chart_service.get_token_chart(
        coin_id=coin_id,
        symbol=symbol,
        contract_address=contract_address,
        chain=chain,
        range_key=range_key,
        vs_currency=vs_currency,
        include_candles=include_candles,
    )


__all__ = [
    "TokenChartService",
    "token_chart_service",
    "get_token_chart",
]
