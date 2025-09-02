import httpx
from typing import Any, Dict, List
from ..config import settings
from .base import IndexerProvider
from .token_list import get_fallback_token_metadata, is_likely_spam_token


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
        metadata = {}
        
        # Call metadata for each token individually
        for addr in token_addresses:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "alchemy_getTokenMetadata",
                    "params": [addr],
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
                        print(f"Alchemy metadata error for {addr}: {data['error']}")
                        # Try fallback token metadata
                        fallback_metadata = get_fallback_token_metadata(addr)
                        metadata[addr] = fallback_metadata
                        continue
                    
                    token_info = data.get("result", {})
                    
                    # Validate token info and set defaults
                    symbol_raw = token_info.get("symbol")
                    name_raw = token_info.get("name")
                    
                    symbol = (symbol_raw or "").strip() if symbol_raw else "UNKNOWN"
                    name = (name_raw or "").strip() if name_raw else "Unknown Token"
                    
                    if not symbol:
                        symbol = "UNKNOWN"
                    if not name:
                        name = "Unknown Token"
                    decimals = token_info.get("decimals")
                    
                    # Handle decimals validation
                    if decimals is None:
                        decimals = 18
                    elif isinstance(decimals, str):
                        try:
                            decimals = int(decimals)
                        except ValueError:
                            decimals = 18
                    
                    # If Alchemy returns UNKNOWN, try fallback
                    if symbol == "UNKNOWN" or name == "Unknown Token":
                        fallback_metadata = get_fallback_token_metadata(addr)
                        if fallback_metadata["symbol"] != "UNKNOWN":
                            # Use fallback data if it's better than what Alchemy returned
                            symbol = fallback_metadata["symbol"]
                            name = fallback_metadata["name"]
                            decimals = fallback_metadata["decimals"]
                    
                    # Check for spam tokens and mark them
                    if is_likely_spam_token(symbol, name, addr):
                        symbol = f"[SPAM] {symbol}"
                        name = f"[SPAM] {name}"
                    
                    metadata[addr] = {
                        "symbol": symbol,
                        "name": name,
                        "decimals": decimals,
                        "_source": {"name": "alchemy", "url": "https://alchemy.com"}
                    }
                    
            except Exception as e:
                print(f"Failed to get metadata for {addr}: {str(e)}")
                # Try fallback token metadata for failed requests
                fallback_metadata = get_fallback_token_metadata(addr)
                metadata[addr] = fallback_metadata
        
        return metadata
