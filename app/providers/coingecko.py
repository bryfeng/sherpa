import asyncio
import math
import httpx
from typing import Any, Dict, List, Optional
from ..config import settings
from .base import PriceProvider
from ..services.evm import chain_id_from_coingecko_platform, is_evm_chain


class CoingeckoProvider(PriceProvider):
    """Coingecko API provider for token prices"""
    
    name = "coingecko"
    timeout_s = 15
    
    def __init__(self):
        self.api_key = settings.coingecko_api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self._platform_cache: Dict[str, Dict[str, str]] = {}
        self._coin_cache: Dict[str, Any] = {}
        
    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key
        return headers

    async def ready(self) -> bool:
        return settings.enable_coingecko  # API key is optional for basic tier
    
    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {
                "status": "unavailable",
                "reason": "Provider disabled"
            }
        
        try:
            headers = {}
            if self.api_key:
                headers["X-CG-Demo-API-Key"] = self.api_key
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/ping",
                    headers=headers,
                    timeout=self.timeout_s
                )
                response.raise_for_status()
                return {"status": "healthy", "latency_ms": int(response.elapsed.total_seconds() * 1000)}
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    async def get_token_prices(self, token_addresses: List[str], vs_currency: str = "usd") -> Dict[str, Any]:
        """Get current prices for multiple tokens by contract address"""
        if not token_addresses:
            return {}
        
        # Coingecko expects comma-separated addresses
        addresses_param = ",".join(token_addresses)
        
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key
        
        params = {
            "contract_addresses": addresses_param,
            "vs_currencies": vs_currency,
            "include_market_cap": "false",
            "include_24hr_vol": "false",
            "include_24hr_change": "false"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/simple/token_price/ethereum",
                headers=headers,
                params=params,
                timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
            
            # Transform to our format
            prices = {}
            for address, price_data in data.items():
                if vs_currency in price_data:
                    prices[address.lower()] = {
                        "price_usd": price_data[vs_currency],
                        "_source": {"name": "coingecko", "url": "https://coingecko.com"}
                    }
            
            return prices
    
    async def get_eth_price(self) -> Dict[str, Any]:
        """Get ETH price specifically"""
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key
        
        params = {
            "ids": "ethereum",
            "vs_currencies": "usd",
            "include_market_cap": "false",
            "include_24hr_vol": "false",
            "include_24hr_change": "false"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/simple/price",
                headers=headers,
                params=params,
                timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
            
            if "ethereum" in data and "usd" in data["ethereum"]:
                return {
                    "price_usd": data["ethereum"]["usd"],
                    "_source": {"name": "coingecko", "url": "https://coingecko.com"}
                }
            
            return {}
    
    async def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get token metadata from Coingecko"""
        headers = self._build_headers()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/coins/ethereum/contract/{token_address}",
                headers=headers,
                timeout=self.timeout_s
            )
            
            if response.status_code == 404:
                return {}  # Token not found
            
            response.raise_for_status()
            data = response.json()
            
            return {
                "id": data.get("id"),
                "symbol": data.get("symbol", "").upper(),
                "name": data.get("name", ""),
                "decimals": data.get("detail_platforms", {}).get("ethereum", {}).get("decimal_place", 18),
                "contract_address": data.get("contract_address") or token_address.lower(),
                "image": (data.get("image") or {}).get("small"),
                "platforms": data.get("platforms") or {},
                "_source": {"name": "coingecko", "url": "https://coingecko.com"}
            }

    async def _get_coin_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        if not coin_id:
            return None

        cached = self._coin_cache.get(coin_id)
        if cached is not None:
            return cached or None

        headers = self._build_headers()
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        }

        if not self.api_key:
            await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            for attempt in range(2):
                try:
                    resp = await client.get(
                        f"{self.base_url}/coins/{coin_id}",
                        headers=headers,
                        params=params,
                        timeout=self.timeout_s,
                    )
                    if resp.status_code == 404:
                        self._coin_cache[coin_id] = {}
                        return None
                    resp.raise_for_status()
                    data = resp.json()
                    self._coin_cache[coin_id] = data
                    return data
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and attempt == 0:
                        await asyncio.sleep(2)
                        continue
                    raise
        return None

    async def get_coin_metadata(self, coin_id: str) -> Dict[str, Any]:
        data = await self._get_coin_data(coin_id)
        if not data:
            return {}

        image = data.get("image") or {}
        platforms_raw = data.get("platforms") or {}
        normalized_platforms = {k: str(v).lower() for k, v in platforms_raw.items() if v}
        if normalized_platforms:
            self._platform_cache[coin_id] = normalized_platforms

        return {
            "id": data.get("id", coin_id),
            "symbol": str(data.get("symbol", "")).upper(),
            "name": data.get("name"),
            "platforms": normalized_platforms,
            "image": image.get("small") or image.get("thumb"),
            "market_cap_rank": data.get("market_cap_rank"),
            "contract_address": normalized_platforms.get("ethereum"),
            "_source": {"name": "coingecko", "url": "https://coingecko.com"},
        }

    async def search_coins(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []

        headers = self._build_headers()
        params = {"query": query}

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/search",
                headers=headers,
                params=params,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()

        coins = payload.get("coins") or []
        if limit and limit > 0:
            return coins[:limit]
        return coins

    async def get_coin_market_chart(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: str = "7",
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not coin_id:
            return {}

        headers = self._build_headers()
        params: Dict[str, Any] = {
            "vs_currency": vs_currency,
            "days": days,
        }
        if interval:
            params["interval"] = interval

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/coins/{coin_id}/market_chart",
                    headers=headers,
                    params=params,
                    timeout=self.timeout_s,
                )
                if resp.status_code == 404:
                    return {}
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401 and "interval" in params:
                    fallback_params = dict(params)
                    fallback_params.pop("interval", None)
                    retry = await client.get(
                        f"{self.base_url}/coins/{coin_id}/market_chart",
                        headers=headers,
                        params=fallback_params,
                        timeout=self.timeout_s,
                    )
                    if retry.status_code == 404:
                        return {}
                    retry.raise_for_status()
                    return retry.json()
                raise

    async def get_coin_ohlc(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: str | int = "7",
    ) -> List[List[float]]:
        if not coin_id:
            return []

        headers = self._build_headers()
        params: Dict[str, Any] = {
            "vs_currency": vs_currency,
            "days": days,
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/coins/{coin_id}/ohlc",
                    headers=headers,
                    params=params,
                    timeout=self.timeout_s,
                )
                if resp.status_code == 404:
                    return []
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    return []
                raise

    async def get_top_coins(self, limit: int = 5, exclude_stable: bool = True) -> List[Dict[str, Any]]:
        """Get top coins by market cap, optionally excluding stablecoins."""
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": max(5, min(50, limit + 5)),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }

        data = await self._fetch_market_trending(params)
        if data is None:
            await asyncio.sleep(2)
            data = await self._fetch_market_trending(params)
            if data is None:
                return []

        # Simple stablecoin filter by common symbols/names
        stable_symbols = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDD", "GUSD", "FRAX", "LUSD", "USDE", "PYUSD"}
        results: List[Dict[str, Any]] = []
        for item in data:
            sym = str(item.get("symbol", "")).upper()
            name = str(item.get("name", ""))
            if exclude_stable and (sym in stable_symbols or "stable" in name.lower()):
                continue
            results.append({
                "id": item.get("id"),
                "symbol": sym,
                "name": name,
                "price_usd": item.get("current_price"),
                "change_24h": item.get("price_change_percentage_24h"),
                "market_cap": item.get("market_cap"),
                "_source": {"name": "coingecko", "url": "https://coingecko.com"}
            })
            if len(results) >= limit:
                break
        return results

    async def get_trending_evm_tokens(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return trending tokens constrained to EVM-compatible chains."""

        if not await self.ready():
            return []

        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key

        search_payload = await self._fetch_search_trending() or []
        if not search_payload:
            return []

        trending: List[Dict[str, Any]] = []
        stable_symbols = {
            "USDT",
            "USDC",
            "BUSD",
            "DAI",
            "TUSD",
            "USDD",
            "GUSD",
            "FRAX",
            "LUSD",
            "USDE",
            "PYUSD",
            "FDUSD",
            "USD+",
            "USDP",
        }

        def _parse_number(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.replace("$", "").replace(",", "").strip()
                if cleaned:
                    try:
                        return float(cleaned)
                    except ValueError:
                        return None
            return None

        for entry in search_payload:
            item = entry.get("item") or {}
            symbol = str(item.get("symbol", "")).upper()
            if not symbol or symbol in stable_symbols:
                continue

            coin_id = str(item.get("id", ""))
            if not coin_id:
                continue

            platforms = await self._get_coin_platforms(coin_id)
            matching_chain_id: Optional[int] = None
            contract_address: Optional[str] = None
            platform_name: Optional[str] = None

            for platform, address in platforms.items():
                chain_id = chain_id_from_coingecko_platform(platform)
                if chain_id is not None and address and is_evm_chain(chain_id):
                    matching_chain_id = chain_id
                    contract_address = str(address).lower()
                    platform_name = platform
                    break

            if matching_chain_id is None and platforms:
                platform_name, address = next(iter(platforms.items()))
                contract_address = str(address).lower() if address else None

            if matching_chain_id is None and not platforms:
                platform_name = None

            data_blob = item.get("data") or {}
            price_usd = _parse_number(data_blob.get("price"))
            if price_usd is None or not math.isfinite(price_usd) or price_usd <= 0:
                continue

            change_dict = data_blob.get("price_change_percentage_24h") or {}
            change_24h = _parse_number(change_dict.get("usd")) if isinstance(change_dict, dict) else None
            volume_24h = _parse_number(data_blob.get("total_volume"))
            market_cap = _parse_number(data_blob.get("market_cap"))

            trending.append(
                {
                    "id": coin_id,
                    "symbol": symbol,
                    "name": item.get("name"),
                    "price_usd": price_usd,
                    "change_1h": None,
                    "change_24h": change_24h,
                    "volume_24h": volume_24h,
                    "market_cap": market_cap,
                    "platform": platform_name,
                    "contract_address": contract_address,
                    "chain_id": matching_chain_id,
                    "_source": {"name": "coingecko", "url": "https://coingecko.com"},
                }
            )

            if len(trending) >= limit:
                break

        return trending

    async def _get_coin_platforms(self, coin_id: str) -> Dict[str, str]:
        cached = self._platform_cache.get(coin_id)
        if cached is not None:
            return cached

        data = await self._get_coin_data(coin_id)
        if not data:
            return {}

        platforms: Dict[str, Any] = data.get("platforms") or {}
        normalized = {k: str(v).lower() for k, v in platforms.items() if v}
        self._platform_cache[coin_id] = normalized
        return normalized

    async def _fetch_market_trending(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/coins/markets",
                    headers=headers,
                    params=params,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    return None
                raise
            return resp.json()

    async def _fetch_search_trending(self) -> Optional[List[Dict[str, Any]]]:
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/search/trending",
                    headers=headers,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    return None
                raise
            payload = resp.json()
        return payload.get("coins", [])
