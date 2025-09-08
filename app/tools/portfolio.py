import asyncio
from decimal import Decimal
from datetime import datetime
from typing import List, Dict, Any
from ..types import ToolEnvelope, Source, TokenBalance, Portfolio
from ..providers.alchemy import AlchemyProvider
from ..providers.coingecko import CoingeckoProvider
from ..cache import cache


async def get_portfolio(address: str, chain: str = "ethereum") -> ToolEnvelope:
    """Get complete portfolio for an address including prices"""
    
    # Check cache first
    cache_key = f"portfolio:{chain}:{address.lower()}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        cached_result.cached = True
        return cached_result
    
    start_time = datetime.now()
    sources = []
    warnings = []
    
    try:
        # Initialize providers
        alchemy = AlchemyProvider()
        coingecko = CoingeckoProvider()
        
        # Check provider readiness
        if not await alchemy.ready():
            raise Exception("Alchemy provider not available")
        
        if not await coingecko.ready():
            warnings.append("Coingecko provider unavailable - prices may be missing")
        
        # Get native ETH balance
        eth_balance_data = await alchemy.get_native_balance(address, chain)
        sources.append(Source(
            name=eth_balance_data["_source"]["name"],
            url=eth_balance_data["_source"]["url"]
        ))
        
        # Get ERC-20 token balances
        token_balances_data = await alchemy.get_token_balances(address, chain)
        
        # Collect all token addresses for metadata and price lookups
        token_addresses = [token["address"] for token in token_balances_data["tokens"]]
        
        # Get token metadata in parallel
        metadata_tasks = []
        if token_addresses:
            # Split into batches to avoid hitting API limits
            batch_size = 10
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i:i + batch_size]
                metadata_tasks.append(alchemy.get_token_metadata(batch))
        
        metadata_results = await asyncio.gather(*metadata_tasks, return_exceptions=True)
        
        # Combine metadata
        all_metadata = {}
        for result in metadata_results:
            if not isinstance(result, Exception):
                all_metadata.update(result)
        
        # Get prices
        prices = {}
        if await coingecko.ready():
            try:
                # Get ETH price
                eth_price_data = await coingecko.get_eth_price()
                if eth_price_data:
                    prices["ETH"] = eth_price_data["price_usd"]
                    sources.append(Source(
                        name=eth_price_data["_source"]["name"],
                        url=eth_price_data["_source"]["url"]
                    ))
                
                # Get token prices
                if token_addresses:
                    token_prices_data = await coingecko.get_token_prices(token_addresses)
                    for addr, price_data in token_prices_data.items():
                        prices[addr] = price_data["price_usd"]
            except Exception as e:
                warnings.append(f"Failed to fetch prices: {str(e)}")
        
        # Build token list starting with ETH
        tokens = []
        
        # Add ETH
        eth_balance_wei = int(eth_balance_data["balance_wei"])
        eth_balance_formatted = f"{eth_balance_wei / 10**18:.6f}"
        eth_price = Decimal(str(prices.get("ETH", 0)))
        eth_value = Decimal(eth_balance_formatted) * eth_price
        
        tokens.append(TokenBalance(
            symbol="ETH",
            name="Ethereum",
            address=None,
            decimals=18,
            balance_wei=str(eth_balance_wei),
            balance_formatted=eth_balance_formatted,
            price_usd=eth_price if eth_price > 0 else None,
            value_usd=eth_value if eth_price > 0 else None
        ))
        
        # Add ERC-20 tokens
        for token_data in token_balances_data["tokens"]:
            token_addr = token_data["address"]
            balance_wei = int(token_data["balance_wei"])
            
            # Get metadata
            metadata = all_metadata.get(token_addr, {})
            symbol = metadata.get("symbol", "UNKNOWN")
            name = metadata.get("name", "Unknown Token")
            decimals = metadata.get("decimals", 18)
            
            # Calculate formatted balance
            balance_formatted = f"{balance_wei / 10**decimals:.6f}"
            
            # Get price and value
            price = Decimal(str(prices.get(token_addr.lower(), 0)))
            value = Decimal(balance_formatted) * price if price > 0 else None
            
            tokens.append(TokenBalance(
                symbol=symbol,
                name=name,
                address=token_addr,
                decimals=decimals,
                balance_wei=str(balance_wei),
                balance_formatted=balance_formatted,
                price_usd=price if price > 0 else None,
                value_usd=value
            ))

        # Filter out spam and zero-value tokens, then sort by USD value desc
        def _is_spam(tok: TokenBalance) -> bool:
            return tok.symbol.startswith("[SPAM]") or tok.name.startswith("[SPAM]")

        filtered_tokens = [
            t for t in tokens
            if (t.value_usd is not None and t.value_usd > 0) and not _is_spam(t)
        ]

        filtered_tokens.sort(key=lambda t: (t.value_usd or Decimal("0")), reverse=True)

        # Calculate total portfolio value from filtered tokens only
        total_value = sum((t.value_usd or Decimal("0")) for t in filtered_tokens)

        # Build portfolio
        portfolio = Portfolio(
            address=address,
            chain=chain,
            total_value_usd=total_value,
            token_count=len(filtered_tokens),
            tokens=filtered_tokens
        )
        
        # Calculate latency
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create result envelope
        result = ToolEnvelope(
            data=portfolio,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            latency_ms=latency_ms,
            warnings=warnings
        )
        
        # Cache the result
        await cache.set(cache_key, result, ttl=300)  # 5 minute cache
        
        return result
        
    except Exception as e:
        return ToolEnvelope(
            data=None,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            warnings=[f"Failed to fetch portfolio: {str(e)}"]
        )
