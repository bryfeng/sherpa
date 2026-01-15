import httpx
from typing import Any, Dict, List, Optional
from ..config import settings
from .base import IndexerProvider
from .token_list import get_fallback_token_metadata, is_likely_spam_token


class UnsupportedChainError(Exception):
    """Raised when a chain is not supported by Alchemy."""
    pass


class AlchemyProvider(IndexerProvider):
    """Alchemy API provider for multi-chain indexing"""

    name = "alchemy"
    timeout_s = 30

    # Fallback slugs for when Convex is unavailable
    _FALLBACK_SLUGS = {
        "ethereum": "eth-mainnet",
        "eth": "eth-mainnet",
        "mainnet": "eth-mainnet",
        "polygon": "polygon-mainnet",
        "matic": "polygon-mainnet",
        "base": "base-mainnet",
        "arbitrum": "arb-mainnet",
        "arb": "arb-mainnet",
        "optimism": "opt-mainnet",
        "op": "opt-mainnet",
    }

    # Native token symbols per chain
    _NATIVE_SYMBOLS = {
        "ethereum": "ETH",
        "eth": "ETH",
        "mainnet": "ETH",
        "polygon": "MATIC",
        "matic": "MATIC",
        "base": "ETH",
        "arbitrum": "ETH",
        "arb": "ETH",
        "optimism": "ETH",
        "op": "ETH",
        "ink": "ETH",
        "avalanche": "AVAX",
        "avax": "AVAX",
        "bnb": "BNB",
        "bsc": "BNB",
    }

    def __init__(self):
        self.api_key = settings.alchemy_api_key
        # Default URL for backwards compatibility
        self._default_base_url = f"https://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        self._chain_service = None

    def _get_chain_service(self):
        """Lazy-load chain service to avoid circular imports."""
        if self._chain_service is None:
            from ..services.chains import get_chain_service
            self._chain_service = get_chain_service()
        return self._chain_service

    async def _get_base_url(self, chain: str = "ethereum") -> str:
        """
        Get the Alchemy base URL for a specific chain.

        Tries to fetch from Convex chain registry first, falls back to hardcoded slugs.
        """
        normalized = chain.lower().strip()

        # Try to get from chain service (Convex)
        try:
            chain_service = self._get_chain_service()
            url = await chain_service.get_alchemy_url(normalized, self.api_key)
            if url:
                return url
        except Exception as e:
            # Log but continue with fallback
            print(f"Chain service unavailable, using fallback: {e}")

        # Fallback to hardcoded slugs
        slug = self._FALLBACK_SLUGS.get(normalized)
        if slug:
            return f"https://{slug}.g.alchemy.com/v2/{self.api_key}"

        raise UnsupportedChainError(f"Chain '{chain}' is not supported by Alchemy")

    def _get_native_symbol(self, chain: str = "ethereum") -> str:
        """Get the native token symbol for a chain."""
        return self._NATIVE_SYMBOLS.get(chain.lower(), "ETH")
        
    async def ready(self) -> bool:
        return bool(self.api_key) and settings.enable_alchemy

    async def health_check(self, chain: str = "ethereum") -> Dict[str, Any]:
        if not await self.ready():
            return {
                "status": "unavailable",
                "reason": "API key not configured or provider disabled"
            }

        try:
            base_url = await self._get_base_url(chain)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    base_url,
                    params={"method": "eth_chainId"},
                    timeout=self.timeout_s
                )
                response.raise_for_status()
                return {"status": "healthy", "chain": chain, "latency_ms": int(response.elapsed.total_seconds() * 1000)}
        except UnsupportedChainError as e:
            return {"status": "unsupported", "reason": str(e)}
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    async def get_native_balance(self, address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Get native token balance for address on any supported chain."""
        base_url = await self._get_base_url(chain)
        native_symbol = self._get_native_symbol(chain)

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                base_url,
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
                "symbol": native_symbol,
                "name": f"{native_symbol} ({chain})",
                "address": None,  # Native token
                "decimals": 18,
                "balance_wei": str(balance_wei),
                "balance_formatted": f"{balance_wei / 10**18:.6f}",
                "chain": chain,
                "_source": {"name": "alchemy", "url": "https://alchemy.com"}
            }
    
    async def get_token_balances(self, address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Get all ERC-20 token balances for address on any supported chain."""
        base_url = await self._get_base_url(chain)

        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address],
            "id": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                base_url,
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
                        "chain": chain,
                        "_source": {"name": "alchemy", "url": "https://alchemy.com"}
                    })

            return {"tokens": tokens, "chain": chain}
    
    async def get_token_metadata(self, token_addresses: List[str], chain: str = "ethereum") -> Dict[str, Any]:
        """Get metadata for multiple tokens on any supported chain."""
        base_url = await self._get_base_url(chain)
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
                        base_url,
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

    async def get_token_balance_for_contract(self, address: str, token_address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Get balance for a specific ERC-20 contract on any supported chain."""
        base_url = await self._get_base_url(chain)

        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, [token_address]],
            "id": 1,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise Exception(f"Alchemy error: {data['error']}")

            token_balances = data.get("result", {}).get("tokenBalances", [])
            if not token_balances:
                return {"contractAddress": token_address, "tokenBalance": "0x0", "chain": chain}

            entry = token_balances[0]
            return {
                "contractAddress": entry.get("contractAddress", token_address),
                "tokenBalance": entry.get("tokenBalance", "0x0"),
                "chain": chain,
            }

    async def get_owned_nfts(
        self,
        address: str,
        contract_address: str,
        chain: str = "ethereum",
        page_size: int = 25,
    ) -> Dict[str, Any]:
        """Fetch NFT ownership details for a specific contract on any supported chain."""
        base_url = await self._get_base_url(chain)

        params = [
            ("owner", address),
            ("withMetadata", "false"),
            ("pageSize", str(page_size)),
            ("contractAddresses[]", contract_address),
        ]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/getNFTs/",
                params=params,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise Exception(f"Alchemy error: {data['error']}")

            return {
                "total": data.get("totalCount", 0),
                "owned_nfts": data.get("ownedNfts", []),
                "pageKey": data.get("pageKey"),
                "chain": chain,
            }
