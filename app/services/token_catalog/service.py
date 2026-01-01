"""
Token Catalog Service

Main service for enriched token metadata and portfolio analysis.

Features:
- Multi-source token enrichment (CoinGecko, DefiLlama)
- Automatic classification and taxonomy
- Convex persistence with caching
- Portfolio profile calculation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    EnrichedToken,
    TokenTaxonomy,
    PortfolioProfile,
    RiskProfile,
    RelatedToken,
    MarketCapTier,
    TokenSector,
)
from .sources import CoinGeckoSource, DefiLlamaSource, CoinGeckoTokenInfo
from .classifier import TokenClassifier

logger = logging.getLogger(__name__)

# Cache TTL in seconds
DEFAULT_CACHE_TTL = 24 * 60 * 60  # 24 hours
STALE_THRESHOLD = 7 * 24 * 60 * 60  # 7 days

# Current enrichment schema version
ENRICHMENT_VERSION = 1


@dataclass
class TokenHolding:
    """User token holding for portfolio analysis."""
    address: str
    chain_id: int
    symbol: str
    name: str
    decimals: int
    balance: float
    value_usd: Optional[float] = None


class TokenCatalogService:
    """
    Service for maintaining enriched token metadata.

    Provides:
    - Token enrichment from multiple sources
    - Automatic classification into taxonomy
    - Convex persistence with caching
    - Portfolio profile analysis
    """

    def __init__(
        self,
        convex_client: Optional[Any] = None,
        coingecko_api_key: Optional[str] = None,
    ):
        """
        Initialize the token catalog service.

        Args:
            convex_client: Optional Convex client for persistence
            coingecko_api_key: Optional CoinGecko Pro API key
        """
        self._convex = convex_client
        self._coingecko = CoinGeckoSource(api_key=coingecko_api_key)
        self._defillama = DefiLlamaSource()
        self._classifier = TokenClassifier()

    async def close(self):
        """Close data source connections."""
        await self._coingecko.close()
        await self._defillama.close()

    # =========================================================================
    # Token Enrichment
    # =========================================================================

    async def enrich_token(
        self,
        address: str,
        chain_id: int,
        *,
        symbol: Optional[str] = None,
        name: Optional[str] = None,
        decimals: int = 18,
        force_refresh: bool = False,
    ) -> EnrichedToken:
        """
        Enrich a token with metadata from multiple sources.

        Args:
            address: Token contract address
            chain_id: Chain ID
            symbol: Optional known symbol
            name: Optional known name
            decimals: Token decimals
            force_refresh: Force refresh even if cached

        Returns:
            EnrichedToken with full metadata
        """
        address = address.lower()

        # 1. Check Convex cache first
        if self._convex and not force_refresh:
            cached = await self._get_cached_token(chain_id, address)
            if cached and not self._is_stale(cached):
                return EnrichedToken.from_dict(cached)

        # 2. Fetch from CoinGecko
        coingecko_info = await self._coingecko.get_token_by_contract(address, chain_id)

        # 3. Create base token
        enriched = self._create_base_token(
            address=address,
            chain_id=chain_id,
            symbol=symbol or (coingecko_info.symbol if coingecko_info else "UNKNOWN"),
            name=name or (coingecko_info.name if coingecko_info else "Unknown Token"),
            decimals=decimals,
        )

        # 4. Apply CoinGecko data
        if coingecko_info:
            enriched = self._apply_coingecko_data(enriched, coingecko_info)

        # 5. Try DefiLlama for protocol info
        if enriched.coingecko_id or enriched.symbol:
            defillama_protocol = await self._defillama.find_protocol_for_token(
                enriched.symbol,
                enriched.coingecko_id,
            )
            if defillama_protocol:
                enriched = self._apply_defillama_data(enriched, defillama_protocol)

        # 6. Classify token
        enriched.taxonomy = self._classifier.classify(
            symbol=enriched.symbol,
            name=enriched.name,
            categories=enriched.taxonomy.categories,
            coingecko_categories=coingecko_info.categories if coingecko_info else None,
            defillama_category=enriched.defillama_id,  # We stored category here temporarily
            market_cap=coingecko_info.market_cap if coingecko_info else None,
        )

        # 7. Find related tokens
        enriched.related_tokens = self._classifier.find_related_tokens(
            enriched.symbol,
            enriched.project_slug,
            chain_id,
        )

        # 8. Update metadata
        enriched.last_updated = datetime.now(timezone.utc)
        enriched.data_source = "coingecko" if coingecko_info else "manual"
        enriched.enrichment_version = ENRICHMENT_VERSION

        # 9. Persist to Convex
        if self._convex:
            await self._persist_token(enriched)

        return enriched

    async def enrich_tokens_batch(
        self,
        tokens: List[Tuple[str, int]],  # (address, chain_id) pairs
        force_refresh: bool = False,
    ) -> List[EnrichedToken]:
        """
        Enrich multiple tokens in batch.

        Args:
            tokens: List of (address, chain_id) tuples
            force_refresh: Force refresh all tokens

        Returns:
            List of EnrichedTokens
        """
        # Process in batches to avoid rate limits
        batch_size = 5
        results: List[EnrichedToken] = []

        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.enrich_token(addr, chain_id, force_refresh=force_refresh)
                for addr, chain_id in batch
            ], return_exceptions=True)

            for result in batch_results:
                if isinstance(result, EnrichedToken):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Token enrichment failed: {result}")

            # Rate limit pause between batches
            if i + batch_size < len(tokens):
                await asyncio.sleep(1.0)

        return results

    async def get_token(
        self,
        address: str,
        chain_id: int,
        enrich_if_missing: bool = True,
    ) -> Optional[EnrichedToken]:
        """
        Get an enriched token from the catalog.

        Args:
            address: Token address
            chain_id: Chain ID
            enrich_if_missing: Enrich the token if not in catalog

        Returns:
            EnrichedToken or None
        """
        address = address.lower()

        if self._convex:
            cached = await self._get_cached_token(chain_id, address)
            if cached:
                return EnrichedToken.from_dict(cached)

        if enrich_if_missing:
            return await self.enrich_token(address, chain_id)

        return None

    async def search_tokens(self, query: str, limit: int = 10) -> List[EnrichedToken]:
        """
        Search tokens in the catalog.

        Args:
            query: Search query (symbol, name, project)
            limit: Max results

        Returns:
            List of matching EnrichedTokens
        """
        if self._convex:
            results = await self._convex.query(
                "tokenCatalog:search",
                {"query": query, "limit": limit},
            )
            return [EnrichedToken.from_dict(r) for r in results]

        # Fallback to CoinGecko search
        search_results = await self._coingecko.search_tokens(query, limit)
        tokens: List[EnrichedToken] = []

        for result in search_results[:3]:  # Limit API calls
            if result.get("id"):
                info = await self._coingecko.get_token_by_id(result["id"])
                if info:
                    # Create minimal enriched token from search
                    token = self._create_base_token(
                        address="",
                        chain_id=1,
                        symbol=info.symbol,
                        name=info.name,
                        decimals=18,
                    )
                    token = self._apply_coingecko_data(token, info)
                    tokens.append(token)

        return tokens

    # =========================================================================
    # Portfolio Profile Analysis
    # =========================================================================

    async def get_portfolio_profile(
        self,
        holdings: List[TokenHolding],
        wallet_address: Optional[str] = None,
    ) -> PortfolioProfile:
        """
        Analyze portfolio composition and generate profile.

        Args:
            holdings: List of token holdings
            wallet_address: Optional wallet address for caching

        Returns:
            PortfolioProfile with sector allocation and risk assessment
        """
        if not holdings:
            return self._empty_portfolio_profile(wallet_address or "unknown")

        # 1. Enrich all holdings
        enriched_holdings: List[Tuple[TokenHolding, EnrichedToken]] = []

        for holding in holdings:
            try:
                enriched = await self.get_token(
                    holding.address,
                    holding.chain_id,
                    enrich_if_missing=True,
                )
                if enriched:
                    enriched_holdings.append((holding, enriched))
            except Exception as e:
                logger.warning(f"Failed to enrich {holding.symbol}: {e}")

        if not enriched_holdings:
            return self._empty_portfolio_profile(wallet_address or "unknown")

        # 2. Calculate total value
        total_value = sum(
            h.value_usd or 0
            for h, _ in enriched_holdings
        )

        # 3. Calculate sector allocation
        sector_allocation = self._calculate_sector_allocation(enriched_holdings, total_value)

        # 4. Calculate category exposure
        category_exposure = self._calculate_category_exposure(enriched_holdings, total_value)

        # 5. Calculate risk profile
        risk_profile = self._calculate_risk_profile(enriched_holdings, total_value)

        # 6. Count tokens by tier
        tokens_by_tier = self._count_tokens_by_tier(enriched_holdings)

        profile = PortfolioProfile(
            wallet_address=wallet_address or "unknown",
            sector_allocation=sector_allocation,
            category_exposure=category_exposure,
            risk_profile=risk_profile,
            tokens_by_tier=tokens_by_tier,
            total_value_usd=total_value,
            token_count=len(enriched_holdings),
            calculated_at=datetime.now(timezone.utc),
        )

        # 7. Persist profile if we have wallet address
        if self._convex and wallet_address:
            await self._persist_portfolio_profile(profile)

        return profile

    def _calculate_sector_allocation(
        self,
        holdings: List[Tuple[TokenHolding, EnrichedToken]],
        total_value: float,
    ) -> Dict[str, float]:
        """Calculate portfolio allocation by sector."""
        if total_value == 0:
            return {}

        sector_values: Dict[str, float] = {}

        for holding, enriched in holdings:
            value = holding.value_usd or 0
            sector = enriched.taxonomy.sector or TokenSector.UNKNOWN
            sector_name = sector.value

            sector_values[sector_name] = sector_values.get(sector_name, 0) + value

        return {
            sector: round((value / total_value) * 100, 2)
            for sector, value in sector_values.items()
        }

    def _calculate_category_exposure(
        self,
        holdings: List[Tuple[TokenHolding, EnrichedToken]],
        total_value: float,
    ) -> Dict[str, float]:
        """Calculate portfolio exposure by category."""
        if total_value == 0:
            return {}

        category_values: Dict[str, float] = {}

        for holding, enriched in holdings:
            value = holding.value_usd or 0
            for category in enriched.taxonomy.categories:
                category_values[category] = category_values.get(category, 0) + value

        return {
            category: round((value / total_value) * 100, 2)
            for category, value in category_values.items()
        }

    def _calculate_risk_profile(
        self,
        holdings: List[Tuple[TokenHolding, EnrichedToken]],
        total_value: float,
    ) -> RiskProfile:
        """Calculate portfolio risk profile."""
        if total_value == 0:
            return RiskProfile(
                diversification_score=0,
                stablecoin_percent=0,
                meme_percent=0,
                concentration_risk=100,
            )

        # Calculate stablecoin percentage
        stablecoin_value = sum(
            h.value_usd or 0
            for h, e in holdings
            if e.taxonomy.is_stablecoin
        )
        stablecoin_percent = (stablecoin_value / total_value) * 100

        # Calculate meme percentage
        meme_value = sum(
            h.value_usd or 0
            for h, e in holdings
            if e.taxonomy.sector == TokenSector.MEME
        )
        meme_percent = (meme_value / total_value) * 100

        # Calculate concentration risk (top holding %)
        holding_values = sorted(
            [h.value_usd or 0 for h, _ in holdings],
            reverse=True,
        )
        concentration_risk = (holding_values[0] / total_value) * 100 if holding_values else 0

        # Calculate diversification score
        # Higher score = more diversified
        # Penalize: high concentration, high meme %, low token count
        token_count = len(holdings)
        unique_sectors = len(set(e.taxonomy.sector for _, e in holdings))

        diversification_score = min(100, max(0,
            50  # Base score
            + min(20, token_count * 2)  # Up to 20 points for token count
            + min(15, unique_sectors * 5)  # Up to 15 points for sector diversity
            - concentration_risk * 0.3  # Penalize concentration
            - meme_percent * 0.2  # Penalize meme exposure
        ))

        return RiskProfile(
            diversification_score=round(diversification_score, 1),
            stablecoin_percent=round(stablecoin_percent, 2),
            meme_percent=round(meme_percent, 2),
            concentration_risk=round(concentration_risk, 2),
        )

    def _count_tokens_by_tier(
        self,
        holdings: List[Tuple[TokenHolding, EnrichedToken]],
    ) -> Dict[str, int]:
        """Count tokens by market cap tier."""
        tier_counts: Dict[str, int] = {
            "mega": 0, "large": 0, "mid": 0, "small": 0, "micro": 0, "unknown": 0,
        }

        for _, enriched in holdings:
            tier = enriched.taxonomy.market_cap_tier
            tier_name = tier.value if tier else "unknown"
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1

        return tier_counts

    def _empty_portfolio_profile(self, wallet_address: str) -> PortfolioProfile:
        """Create an empty portfolio profile."""
        return PortfolioProfile(
            wallet_address=wallet_address,
            sector_allocation={},
            category_exposure={},
            risk_profile=RiskProfile(
                diversification_score=0,
                stablecoin_percent=0,
                meme_percent=0,
                concentration_risk=0,
            ),
            tokens_by_tier={},
            total_value_usd=0,
            token_count=0,
            calculated_at=datetime.now(timezone.utc),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_base_token(
        self,
        address: str,
        chain_id: int,
        symbol: str,
        name: str,
        decimals: int,
    ) -> EnrichedToken:
        """Create a base EnrichedToken with minimal data."""
        return EnrichedToken(
            address=address.lower(),
            chain_id=chain_id,
            symbol=symbol.upper(),
            name=name,
            decimals=decimals,
            taxonomy=TokenTaxonomy(),
        )

    def _apply_coingecko_data(
        self,
        token: EnrichedToken,
        info: CoinGeckoTokenInfo,
    ) -> EnrichedToken:
        """Apply CoinGecko data to token."""
        token.coingecko_id = info.id
        token.logo_url = info.image
        token.description = info.description

        # Apply links
        links = info.links
        if links:
            token.website = links.get("homepage", [None])[0] if links.get("homepage") else None
            token.twitter = links.get("twitter_screen_name")
            token.discord = links.get("chat_url", [None])[0] if links.get("chat_url") else None
            token.github = links.get("repos_url", {}).get("github", [None])[0] if links.get("repos_url") else None

        # Apply market cap tier
        if info.market_cap:
            token.taxonomy.market_cap_tier = self._coingecko.calculate_market_cap_tier(info.market_cap)

        return token

    def _apply_defillama_data(self, token: EnrichedToken, protocol: Any) -> EnrichedToken:
        """Apply DefiLlama protocol data to token."""
        token.defillama_id = protocol.slug
        token.project_name = protocol.name
        token.project_slug = protocol.slug

        if protocol.twitter:
            token.twitter = protocol.twitter
        if protocol.url:
            token.website = protocol.url

        # Store category for classifier
        if protocol.category:
            if "defi" not in token.taxonomy.categories:
                token.taxonomy.categories.append("defi")

        return token

    async def _get_cached_token(self, chain_id: int, address: str) -> Optional[Dict]:
        """Get cached token from Convex."""
        if not self._convex:
            return None

        try:
            return await self._convex.query(
                "tokenCatalog:get",
                {"chainId": chain_id, "address": address},
            )
        except Exception as e:
            logger.warning(f"Failed to get cached token: {e}")
            return None

    def _is_stale(self, cached: Dict) -> bool:
        """Check if cached token is stale."""
        last_updated = cached.get("lastUpdated", 0)
        age = (datetime.now(timezone.utc).timestamp() * 1000) - last_updated
        return age > (STALE_THRESHOLD * 1000)

    async def _persist_token(self, token: EnrichedToken) -> None:
        """Persist token to Convex."""
        if not self._convex:
            return

        try:
            await self._convex.mutation(
                "tokenCatalog:upsert",
                token.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to persist token: {e}")

    async def _persist_portfolio_profile(self, profile: PortfolioProfile) -> None:
        """Persist portfolio profile to Convex."""
        if not self._convex:
            return

        try:
            # This would need the wallet ID, skipping for now
            pass
        except Exception as e:
            logger.error(f"Failed to persist portfolio profile: {e}")
