import httpx
from typing import Any, Dict, List
from ..config import settings
from .base import PriceProvider


class CoingeckoProvider(PriceProvider):
    """Coingecko API provider for token prices"""
    
    name = "coingecko"
    timeout_s = 15
    
    def __init__(self):
        self.api_key = settings.coingecko_api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        
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
        headers = {}
        if self.api_key:
            headers["X-CG-Demo-API-Key"] = self.api_key
        
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
                "symbol": data.get("symbol", "").upper(),
                "name": data.get("name", ""),
                "decimals": data.get("detail_platforms", {}).get("ethereum", {}).get("decimal_place", 18),
                "_source": {"name": "coingecko", "url": "https://coingecko.com"}
            }
