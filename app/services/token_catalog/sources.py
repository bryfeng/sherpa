"""
Token Catalog Data Sources

External data sources for token enrichment:
- CoinGecko: Token metadata, market data, categories
- DefiLlama: Protocol mappings, TVL data
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import httpx
import logging

from .models import (
    EnrichedToken,
    TokenTaxonomy,
    MarketCapTier,
    TokenSector,
    TokenSubsector,
    RelatedToken,
)

logger = logging.getLogger(__name__)

# Chain ID to CoinGecko platform mapping
CHAIN_TO_COINGECKO_PLATFORM = {
    1: "ethereum",
    10: "optimistic-ethereum",
    137: "polygon-pos",
    8453: "base",
    42161: "arbitrum-one",
    43114: "avalanche",
    56: "binance-smart-chain",
}

# CoinGecko category to our taxonomy mapping
COINGECKO_CATEGORY_MAP = {
    # DeFi
    "decentralized-exchange": (TokenSector.DEFI, TokenSubsector.DEX, ["defi", "dex"]),
    "decentralized-exchange-token-dex": (TokenSector.DEFI, TokenSubsector.DEX, ["defi", "dex"]),
    "lending-borrowing": (TokenSector.DEFI, TokenSubsector.LENDING, ["defi", "lending"]),
    "decentralized-derivatives": (TokenSector.DEFI, TokenSubsector.DERIVATIVES, ["defi", "derivatives"]),
    "yield-farming": (TokenSector.DEFI, TokenSubsector.YIELD, ["defi", "yield"]),
    "yield-aggregator": (TokenSector.DEFI, TokenSubsector.YIELD, ["defi", "yield"]),
    "stablecoins": (TokenSector.DEFI, TokenSubsector.STABLECOIN, ["stablecoin"]),
    "insurance": (TokenSector.DEFI, TokenSubsector.INSURANCE, ["defi", "insurance"]),
    "liquid-staking-governance-tokens": (TokenSector.DEFI, TokenSubsector.YIELD, ["defi", "staking"]),

    # Infrastructure
    "layer-1": (TokenSector.INFRASTRUCTURE, TokenSubsector.L1, ["infrastructure", "l1"]),
    "layer-2": (TokenSector.INFRASTRUCTURE, TokenSubsector.L2, ["infrastructure", "l2"]),
    "ethereum-layer-2": (TokenSector.INFRASTRUCTURE, TokenSubsector.L2, ["infrastructure", "l2", "ethereum"]),
    "oracles": (TokenSector.INFRASTRUCTURE, TokenSubsector.ORACLE, ["infrastructure", "oracle"]),
    "cross-chain": (TokenSector.INFRASTRUCTURE, TokenSubsector.BRIDGE, ["infrastructure", "bridge"]),
    "storage": (TokenSector.INFRASTRUCTURE, TokenSubsector.STORAGE, ["infrastructure", "storage"]),
    "privacy-coins": (TokenSector.INFRASTRUCTURE, TokenSubsector.PRIVACY, ["infrastructure", "privacy"]),

    # Gaming
    "gaming": (TokenSector.GAMING, TokenSubsector.GAMING, ["gaming"]),
    "play-to-earn": (TokenSector.GAMING, TokenSubsector.GAMING, ["gaming", "p2e"]),
    "metaverse": (TokenSector.GAMING, TokenSubsector.METAVERSE, ["gaming", "metaverse"]),
    "virtual-reality": (TokenSector.GAMING, TokenSubsector.METAVERSE, ["gaming", "vr"]),
    "nft-index": (TokenSector.GAMING, TokenSubsector.NFT_INFRA, ["nft"]),

    # Social
    "social-money": (TokenSector.SOCIAL, TokenSubsector.SOCIAL_NETWORK, ["social"]),
    "socialfi": (TokenSector.SOCIAL, TokenSubsector.SOCIAL_NETWORK, ["social"]),
    "fan-token": (TokenSector.SOCIAL, TokenSubsector.CREATOR, ["social", "fan"]),

    # AI/Data
    "artificial-intelligence": (TokenSector.AI_DATA, TokenSubsector.AI, ["ai"]),
    "big-data": (TokenSector.AI_DATA, TokenSubsector.DATA_INDEX, ["data"]),
    "internet-of-things-iot": (TokenSector.AI_DATA, TokenSubsector.DATA_INDEX, ["iot", "data"]),

    # Meme
    "meme-token": (TokenSector.MEME, TokenSubsector.MEME_TOKEN, ["meme"]),

    # Governance
    "governance": (TokenSector.DEFI, TokenSubsector.GOVERNANCE, ["governance"]),
}


@dataclass
class CoinGeckoTokenInfo:
    """Raw token info from CoinGecko."""
    id: str
    symbol: str
    name: str
    categories: List[str]
    market_cap: Optional[float]
    image: Optional[str]
    description: Optional[str]
    links: Dict[str, Any]
    platforms: Dict[str, str]  # platform -> contract_address


class CoinGeckoSource:
    """
    CoinGecko API data source for token metadata.

    Provides:
    - Token info by contract address
    - Token search by symbol/name
    - Category information
    - Market cap data
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko source.

        Args:
            api_key: Optional CoinGecko Pro API key for higher rate limits.
        """
        self._api_key = api_key
        self._base_url = "https://pro-api.coingecko.com/api/v3" if api_key else self.BASE_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["x-cg-pro-api-key"] = self._api_key
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers=headers,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_token_by_contract(
        self,
        address: str,
        chain_id: int,
    ) -> Optional[CoinGeckoTokenInfo]:
        """
        Get token info by contract address.

        Args:
            address: Contract address
            chain_id: EVM chain ID

        Returns:
            CoinGeckoTokenInfo or None if not found
        """
        platform = CHAIN_TO_COINGECKO_PLATFORM.get(chain_id)
        if not platform:
            return None

        client = await self._get_client()

        try:
            url = f"{self._base_url}/coins/{platform}/contract/{address.lower()}"
            response = await client.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return CoinGeckoTokenInfo(
                id=data.get("id", ""),
                symbol=data.get("symbol", "").upper(),
                name=data.get("name", ""),
                categories=data.get("categories", []) or [],
                market_cap=data.get("market_data", {}).get("market_cap", {}).get("usd"),
                image=data.get("image", {}).get("small"),
                description=data.get("description", {}).get("en", "")[:500],
                links=data.get("links", {}),
                platforms=data.get("platforms", {}),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
            else:
                logger.error(f"CoinGecko API error: {e}")
            return None
        except Exception as e:
            logger.error(f"CoinGecko error: {e}")
            return None

    async def get_token_by_id(self, coingecko_id: str) -> Optional[CoinGeckoTokenInfo]:
        """Get token info by CoinGecko ID."""
        client = await self._get_client()

        try:
            url = f"{self._base_url}/coins/{coingecko_id}"
            response = await client.get(url, params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
            })

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return CoinGeckoTokenInfo(
                id=data.get("id", ""),
                symbol=data.get("symbol", "").upper(),
                name=data.get("name", ""),
                categories=data.get("categories", []) or [],
                market_cap=data.get("market_data", {}).get("market_cap", {}).get("usd"),
                image=data.get("image", {}).get("small"),
                description=data.get("description", {}).get("en", "")[:500],
                links=data.get("links", {}),
                platforms=data.get("platforms", {}),
            )
        except Exception as e:
            logger.error(f"CoinGecko error for {coingecko_id}: {e}")
            return None

    async def search_tokens(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search tokens by symbol or name.

        Returns list of search results with id, symbol, name, market_cap_rank.
        """
        client = await self._get_client()

        try:
            url = f"{self._base_url}/search"
            response = await client.get(url, params={"query": query})
            response.raise_for_status()

            data = response.json()
            coins = data.get("coins", [])[:limit]

            return [
                {
                    "id": c.get("id"),
                    "symbol": c.get("symbol", "").upper(),
                    "name": c.get("name"),
                    "market_cap_rank": c.get("market_cap_rank"),
                    "thumb": c.get("thumb"),
                }
                for c in coins
            ]
        except Exception as e:
            logger.error(f"CoinGecko search error: {e}")
            return []

    def map_categories_to_taxonomy(
        self,
        categories: List[str],
    ) -> TokenTaxonomy:
        """
        Map CoinGecko categories to our taxonomy.

        Returns TokenTaxonomy with sector, subsector, and categories.
        """
        taxonomy = TokenTaxonomy()
        all_tags: set = set()

        for category in categories:
            if not category:
                continue

            category_lower = category.lower().replace(" ", "-")
            mapping = COINGECKO_CATEGORY_MAP.get(category_lower)

            if mapping:
                sector, subsector, tags = mapping
                if taxonomy.sector is None:
                    taxonomy.sector = sector
                if taxonomy.subsector is None:
                    taxonomy.subsector = subsector
                all_tags.update(tags)

            # Detect stablecoins
            if "stablecoin" in category_lower:
                taxonomy.is_stablecoin = True

            # Detect governance tokens
            if "governance" in category_lower:
                taxonomy.is_governance_token = True

        taxonomy.categories = list(all_tags)
        return taxonomy

    def calculate_market_cap_tier(self, market_cap: Optional[float]) -> Optional[MarketCapTier]:
        """Calculate market cap tier from USD value."""
        if market_cap is None:
            return None

        if market_cap >= 10_000_000_000:  # $10B+
            return MarketCapTier.MEGA
        elif market_cap >= 1_000_000_000:  # $1B+
            return MarketCapTier.LARGE
        elif market_cap >= 100_000_000:  # $100M+
            return MarketCapTier.MID
        elif market_cap >= 10_000_000:  # $10M+
            return MarketCapTier.SMALL
        else:
            return MarketCapTier.MICRO


@dataclass
class DefiLlamaProtocol:
    """Protocol info from DefiLlama."""
    id: str
    name: str
    slug: str
    category: str
    chains: List[str]
    tvl: Optional[float]
    symbol: Optional[str]
    twitter: Optional[str]
    url: Optional[str]
    gecko_id: Optional[str]
    address: Optional[str]


class DefiLlamaSource:
    """
    DefiLlama API data source for protocol mappings.

    Provides:
    - Protocol info by name/slug
    - Protocol TVL data
    - Protocol-to-token mappings
    """

    BASE_URL = "https://api.llama.fi"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._protocols_cache: Optional[List[DefiLlamaProtocol]] = None
        self._cache_timestamp: float = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_all_protocols(self, force_refresh: bool = False) -> List[DefiLlamaProtocol]:
        """
        Get all protocols from DefiLlama.

        Caches results for 1 hour.
        """
        import time

        # Check cache (1 hour TTL)
        if not force_refresh and self._protocols_cache and (time.time() - self._cache_timestamp < 3600):
            return self._protocols_cache

        client = await self._get_client()

        try:
            url = f"{self.BASE_URL}/protocols"
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            protocols = []

            for p in data:
                protocols.append(DefiLlamaProtocol(
                    id=p.get("id", ""),
                    name=p.get("name", ""),
                    slug=p.get("slug", ""),
                    category=p.get("category", ""),
                    chains=p.get("chains", []),
                    tvl=p.get("tvl"),
                    symbol=p.get("symbol"),
                    twitter=p.get("twitter"),
                    url=p.get("url"),
                    gecko_id=p.get("gecko_id"),
                    address=p.get("address"),
                ))

            self._protocols_cache = protocols
            self._cache_timestamp = time.time()

            return protocols
        except Exception as e:
            logger.error(f"DefiLlama protocols error: {e}")
            return self._protocols_cache or []

    async def get_protocol(self, slug: str) -> Optional[DefiLlamaProtocol]:
        """Get a specific protocol by slug."""
        client = await self._get_client()

        try:
            url = f"{self.BASE_URL}/protocol/{slug}"
            response = await client.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return DefiLlamaProtocol(
                id=data.get("id", ""),
                name=data.get("name", ""),
                slug=slug,
                category=data.get("category", ""),
                chains=list(data.get("chainTvls", {}).keys()),
                tvl=data.get("tvl"),
                symbol=data.get("symbol"),
                twitter=data.get("twitter"),
                url=data.get("url"),
                gecko_id=data.get("gecko_id"),
                address=data.get("address"),
            )
        except Exception as e:
            logger.error(f"DefiLlama protocol error for {slug}: {e}")
            return None

    async def find_protocol_for_token(
        self,
        symbol: str,
        coingecko_id: Optional[str] = None,
    ) -> Optional[DefiLlamaProtocol]:
        """
        Find a protocol that matches a token.

        Matches by symbol or CoinGecko ID.
        """
        protocols = await self.get_all_protocols()

        symbol_upper = symbol.upper()

        for protocol in protocols:
            # Match by symbol
            if protocol.symbol and protocol.symbol.upper() == symbol_upper:
                return protocol

            # Match by CoinGecko ID
            if coingecko_id and protocol.gecko_id == coingecko_id:
                return protocol

        return None

    def map_category_to_taxonomy(self, category: str) -> Tuple[Optional[TokenSector], Optional[TokenSubsector]]:
        """Map DefiLlama category to our taxonomy."""
        category_lower = category.lower()

        mapping = {
            "dexes": (TokenSector.DEFI, TokenSubsector.DEX),
            "lending": (TokenSector.DEFI, TokenSubsector.LENDING),
            "derivatives": (TokenSector.DEFI, TokenSubsector.DERIVATIVES),
            "yield": (TokenSector.DEFI, TokenSubsector.YIELD),
            "yield aggregator": (TokenSector.DEFI, TokenSubsector.YIELD),
            "liquid staking": (TokenSector.DEFI, TokenSubsector.YIELD),
            "cdp": (TokenSector.DEFI, TokenSubsector.LENDING),
            "bridge": (TokenSector.INFRASTRUCTURE, TokenSubsector.BRIDGE),
            "cross chain": (TokenSector.INFRASTRUCTURE, TokenSubsector.BRIDGE),
            "oracle": (TokenSector.INFRASTRUCTURE, TokenSubsector.ORACLE),
            "chain": (TokenSector.INFRASTRUCTURE, TokenSubsector.L1),
            "gaming": (TokenSector.GAMING, TokenSubsector.GAMING),
            "nft marketplace": (TokenSector.GAMING, TokenSubsector.NFT_INFRA),
            "insurance": (TokenSector.DEFI, TokenSubsector.INSURANCE),
            "privacy": (TokenSector.INFRASTRUCTURE, TokenSubsector.PRIVACY),
        }

        return mapping.get(category_lower, (None, None))
