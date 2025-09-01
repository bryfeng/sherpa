import httpx
from typing import Any, Dict, List
from ..config import settings
from .base import IndexerProvider


class AlchemyProvider(IndexerProvider):
    """Alchemy API provider for Ethereum indexing"""
    
    name = "alchemy"
    timeout_s = 30
    
    def __init__(self):
        self.api_key = settings.alchemy_api_key
        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        
    async def ready(self) -> bool:
        return bool(self.api_key) and settings.enable_alchemy
    
    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {
                "status": "unavailable", 
                "reason": "API key not configured or provider disabled"
            }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}",
                    params={"method": "eth_chainId"},
                    timeout=self.timeout_s
                )
                response.raise_for_status()
                return {"status": "healthy", "latency_ms": int(response.elapsed.total_seconds() * 1000)}
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    async def get_native_balance(self, address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Get ETH balance for address"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy error: {data['error']}")
            
            balance_wei = int(data["result"], 16)
            return {
                "symbol": "ETH",
                "name": "Ethereum",
                "address": None,  # Native token
                "decimals": 18,
                "balance_wei": str(balance_wei),
                "balance_formatted": f"{balance_wei / 10**18:.6f}",
                "_source": {"name": "alchemy", "url": "https://alchemy.com"}
            }
    
    async def get_token_balances(self, address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Get all ERC-20 token balances for address"""
        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address],
            "id": 1
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy error: {data['error']}")
            
            tokens = []
            for token_data in data["result"]["tokenBalances"]:
                if int(token_data["tokenBalance"], 16) > 0:  # Only include non-zero balances
                    tokens.append({
                        "address": token_data["contractAddress"],
                        "balance_wei": str(int(token_data["tokenBalance"], 16)),
                        "_source": {"name": "alchemy", "url": "https://alchemy.com"}
                    })
            
            return {"tokens": tokens}
    
    async def get_token_metadata(self, token_addresses: List[str]) -> Dict[str, Any]:
        """Get metadata for multiple tokens"""
        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenMetadata",
            "params": [{"contractAddress": addr} for addr in token_addresses],
            "id": 1
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy error: {data['error']}")
            
            metadata = {}
            if isinstance(data.get("result"), list):
                for i, addr in enumerate(token_addresses):
                    if i < len(data["result"]):
                        token_info = data["result"][i]
                        metadata[addr] = {
                            "symbol": token_info.get("symbol", "UNKNOWN"),
                            "name": token_info.get("name", "Unknown Token"),
                            "decimals": token_info.get("decimals", 18),
                            "_source": {"name": "alchemy", "url": "https://alchemy.com"}
                        }
            
            return metadata
