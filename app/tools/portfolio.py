from __future__ import annotations

import asyncio
import re
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..cache import cache
from ..providers.alchemy import AlchemyProvider, UnsupportedChainError
from ..providers.coingecko import CoingeckoProvider
from ..providers.solana import SolanaProvider
from ..services.address import normalize_chain
from ..services.chains import get_chain_service
from ..types import Portfolio, Source, TokenBalance, ToolEnvelope

CACHE_TTL_SECONDS = 300

# Stablecoins pegged to $1 USD - includes base symbols and common bridged variants
# Pattern: Base symbols + common bridge suffixes (.e, .b, bridged, etc.)
USD_STABLECOIN_SYMBOLS = {
    "USDC", "USDT", "DAI", "FRAX", "LUSD", "BUSD", "TUSD", "USDP",
    "GUSD", "USDD", "MIM", "FDUSD", "PYUSD", "CRVUSD", "GHO", "SUSD",
    "DOLA", "MAI", "CUSD", "USDbC", "USDe",
}

# Regex pattern to match bridged stablecoin variants (e.g., USDC.e, USDT.b, etc.)
BRIDGED_STABLECOIN_PATTERN = re.compile(
    r"^(USDC|USDT|DAI|FRAX|BUSD|TUSD)[.\-_]?[eEbB]?$|"
    r"^(USDC|USDT|DAI)\.e$|"  # Stargate bridged
    r"^(USDC|USDT)\.b$|"  # Other bridges
    r"^Bridged (USDC|USDT|DAI)",  # Name pattern
    re.IGNORECASE
)


def _is_usd_stablecoin(symbol: str, name: Optional[str] = None) -> bool:
    """Check if a token is a USD-pegged stablecoin."""
    symbol_upper = symbol.upper().strip()

    # Direct symbol match
    if symbol_upper in USD_STABLECOIN_SYMBOLS:
        return True

    # Check bridged variants via regex
    if BRIDGED_STABLECOIN_PATTERN.match(symbol_upper):
        return True

    # Check name for bridged stablecoins
    if name:
        name_lower = name.lower()
        if any(stable.lower() in name_lower for stable in ["usdc", "usdt", "dai"]):
            if any(bridge in name_lower for bridge in ["bridged", "stargate", "wormhole", "multichain"]):
                return True

    return False


async def get_portfolio(address: str, chain: str = "ethereum") -> ToolEnvelope:
    """Return portfolio data for the requested wallet/chain pair."""

    normalized_chain = normalize_chain(chain)
    cache_key = f"portfolio:{normalized_chain}:{address.lower()}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        cached_result.cached = True
        return cached_result

    start_time = datetime.now()

    if normalized_chain == "solana":
        result = await _solana_portfolio(address, start_time)
    else:
        # Try EVM chains - the chain service will validate if supported
        result = await _evm_portfolio(address, normalized_chain, start_time)

    if result.data is not None:
        await cache.set(cache_key, result, ttl=CACHE_TTL_SECONDS)

    return result


async def _evm_portfolio(address: str, chain: str, start_time: datetime) -> ToolEnvelope:
    sources: List[Source] = []
    warnings: List[str] = []

    try:
        alchemy = AlchemyProvider()
        coingecko = CoingeckoProvider()
        chain_service = get_chain_service()

        if not await alchemy.ready():
            raise RuntimeError("Alchemy provider not available")

        # Get chain config for native token info
        chain_config = await chain_service.resolve_alias(chain)
        if chain_config:
            native_symbol = chain_config.native_symbol
            native_name = f"{native_symbol} ({chain_config.name})"
        else:
            native_symbol = "ETH"
            native_name = f"Native Token ({chain})"

        coingecko_ready = await coingecko.ready()
        if not coingecko_ready:
            warnings.append("Coingecko provider unavailable - prices may be missing")

        native_balance = await alchemy.get_native_balance(address, chain)
        sources.append(Source(**native_balance["_source"]))

        token_balances = await alchemy.get_token_balances(address, chain)
        token_addresses = [token["address"] for token in token_balances["tokens"]]

        metadata_tasks = []
        if token_addresses:
            batch_size = 10
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i : i + batch_size]
                metadata_tasks.append(alchemy.get_token_metadata(batch, chain))

        metadata_results = await asyncio.gather(*metadata_tasks, return_exceptions=True)
        all_metadata: Dict[str, Dict[str, Any]] = {}
        for result in metadata_results:
            if not isinstance(result, Exception):
                all_metadata.update(result)

        prices: Dict[str, Decimal] = {}
        if coingecko_ready:
            try:
                # Get native token price (ETH for most chains)
                eth_price = await coingecko.get_eth_price()
                if eth_price:
                    prices[native_symbol] = Decimal(str(eth_price["price_usd"]))
                    sources.append(Source(**eth_price["_source"]))

                if token_addresses:
                    token_prices = await coingecko.get_token_prices(token_addresses)
                    for addr, price_data in token_prices.items():
                        prices[addr.lower()] = Decimal(str(price_data["price_usd"]))
            except Exception as exc:  # pragma: no cover - network errors are non-deterministic
                warnings.append(f"Failed to fetch prices: {exc}")

        tokens: List[TokenBalance] = []

        native_balance_wei = int(native_balance["balance_wei"])
        native_balance_formatted = f"{native_balance_wei / 10**18:.6f}"
        native_price = prices.get(native_symbol)
        native_value = (Decimal(native_balance_formatted) * native_price) if native_price else None
        tokens.append(
            TokenBalance(
                symbol=native_symbol,
                name=native_name,
                address=None,
                decimals=18,
                balance_wei=str(native_balance_wei),
                balance_formatted=native_balance_formatted,
                price_usd=native_price,
                value_usd=native_value,
            )
        )

        for token_data in token_balances["tokens"]:
            token_addr = token_data["address"]
            balance_wei = int(token_data["balance_wei"])

            metadata = all_metadata.get(token_addr, {})
            symbol = metadata.get("symbol", "UNKNOWN")
            name = metadata.get("name", "Unknown Token")
            decimals = metadata.get("decimals", 18)
            balance_formatted = f"{balance_wei / 10**decimals:.6f}"

            # Get price from CoinGecko, or use $1 peg for stablecoins
            price = prices.get(token_addr.lower())
            if price is None and _is_usd_stablecoin(symbol, name):
                price = Decimal("1.0")

            value = (Decimal(balance_formatted) * price) if price else None

            tokens.append(
                TokenBalance(
                    symbol=symbol,
                    name=name,
                    address=token_addr,
                    decimals=decimals,
                    balance_wei=str(balance_wei),
                    balance_formatted=balance_formatted,
                    price_usd=price,
                    value_usd=value,
                )
            )

        filtered_tokens = _filter_and_rank_tokens(tokens)
        total_value = sum((t.value_usd or Decimal("0")) for t in filtered_tokens)

        portfolio = Portfolio(
            address=address,
            chain=chain,
            total_value_usd=total_value,
            token_count=len(filtered_tokens),
            tokens=filtered_tokens,
        )

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return ToolEnvelope(
            data=portfolio,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            latency_ms=latency_ms,
            warnings=warnings,
        )

    except UnsupportedChainError as exc:
        return ToolEnvelope(
            data=None,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            warnings=[f"Chain not supported: {exc}"],
        )
    except Exception as exc:
        return ToolEnvelope(
            data=None,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            warnings=[f"Failed to fetch EVM portfolio: {exc}"],
        )


async def _solana_portfolio(address: str, start_time: datetime) -> ToolEnvelope:
    sources: List[Source] = []
    warnings: List[str] = []

    provider = SolanaProvider()
    if not await provider.ready():
        warnings.append("Solana provider not configured")
        return ToolEnvelope(
            data=None,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            warnings=warnings,
        )

    try:
        native_balance = await provider.get_native_balance(address)
        token_balances = await provider.get_token_balances(address)

        source_payload = native_balance.get("_source")
        if isinstance(source_payload, dict):
            sources.append(Source(**source_payload))

        tokens: List[TokenBalance] = []
        tokens.append(
            TokenBalance(
                symbol=native_balance.get("symbol", "SOL"),
                name=native_balance.get("name", "Solana"),
                address=None,
                decimals=native_balance.get("decimals", 9),
                balance_wei=str(native_balance.get("balance_wei", "0")),
                balance_formatted=native_balance.get("balance_formatted", "0"),
                price_usd=_coerce_decimal(native_balance.get("price_usd")),
                value_usd=_coerce_decimal(native_balance.get("value_usd")),
            )
        )

        for token in token_balances.get("tokens", []):
            tokens.append(
                TokenBalance(
                    symbol=token.get("symbol", "UNKNOWN"),
                    name=token.get("name", "Unknown Token"),
                    address=token.get("address"),
                    decimals=token.get("decimals", 0),
                    balance_wei=str(token.get("balance_wei", "0")),
                    balance_formatted=token.get("balance_formatted", "0"),
                    price_usd=_coerce_decimal(token.get("price_usd")),
                    value_usd=_coerce_decimal(token.get("value_usd")),
                )
            )

        filtered_tokens = _filter_and_rank_tokens(tokens)
        total_value = sum((t.value_usd or Decimal("0")) for t in filtered_tokens)

        portfolio = Portfolio(
            address=address,
            chain="solana",
            total_value_usd=total_value,
            token_count=len(filtered_tokens),
            tokens=filtered_tokens,
        )

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return ToolEnvelope(
            data=portfolio,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            latency_ms=latency_ms,
            warnings=warnings,
        )

    except Exception as exc:
        return ToolEnvelope(
            data=None,
            sources=sources,
            fetched_at=start_time,
            cached=False,
            warnings=[f"Failed to fetch Solana portfolio: {exc}"],
        )


def _filter_and_rank_tokens(tokens: List[TokenBalance]) -> List[TokenBalance]:
    def _is_spam(tok: TokenBalance) -> bool:
        return tok.symbol.startswith("[SPAM]") or tok.name.startswith("[SPAM]")

    filtered = [
        t
        for t in tokens
        if (t.value_usd is not None and t.value_usd > 0)
        and not _is_spam(t)
    ]
    filtered.sort(key=lambda t: (t.value_usd or Decimal("0")), reverse=True)
    return filtered


def _coerce_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return None
