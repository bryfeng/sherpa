"""
Relevance Service

High-level service that integrates relevance scoring with
the token catalog and news fetcher services.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .models import (
    ContentContext,
    PortfolioContext,
    RelevanceScore,
    TokenHolding,
)
from .scorer import RelevanceScorer

logger = logging.getLogger(__name__)


class RelevanceService:
    """
    High-level service for portfolio-aware content relevance.

    Integrates:
    - Token catalog (for enriching portfolio)
    - News fetcher (for content)
    - Relevance scorer (for scoring)
    """

    def __init__(
        self,
        convex_client: Any = None,
        token_catalog_service: Any = None,
        scorer: Optional[RelevanceScorer] = None,
    ):
        """
        Initialize the relevance service.

        Args:
            convex_client: Convex client for data access
            token_catalog_service: Token catalog for enrichment
            scorer: Custom scorer (uses default if None)
        """
        self._convex = convex_client
        self._token_catalog = token_catalog_service
        self._scorer = scorer or RelevanceScorer()

    async def build_portfolio_context(
        self,
        wallet_address: str,
        holdings: Optional[List[Dict[str, Any]]] = None,
    ) -> PortfolioContext:
        """
        Build portfolio context from wallet holdings.

        Args:
            wallet_address: User's wallet address
            holdings: Optional pre-fetched holdings (fetches if None)

        Returns:
            PortfolioContext ready for scoring
        """
        # Fetch holdings if not provided
        if holdings is None and self._convex:
            # Try to get cached portfolio profile first
            profile = await self._convex.query(
                "tokenCatalog:getPortfolioProfile",
                {"walletAddress": wallet_address.lower()},
            )
            if profile:
                return self._profile_to_context(profile)

        # Build from raw holdings
        if holdings is None:
            holdings = []

        token_holdings: List[TokenHolding] = []
        total_value = Decimal("0")

        # Calculate total value first
        for h in holdings:
            value = Decimal(str(h.get("valueUsd", 0)))
            total_value += value

        # Build token holdings with enrichment
        for h in holdings:
            symbol = h.get("symbol", "").upper()
            address = h.get("address", "").lower() if h.get("address") else None
            chain_id = h.get("chainId", 1)
            value_usd = Decimal(str(h.get("valueUsd", 0)))

            # Calculate percentage
            percentage = float(value_usd / total_value * 100) if total_value > 0 else 0

            # Get enriched data from token catalog
            enriched = None
            if self._token_catalog and address:
                try:
                    enriched = await self._token_catalog.get_token(address, chain_id)
                except Exception as e:
                    logger.warning(f"Could not enrich token {symbol}: {e}")

            # Build holding
            holding = TokenHolding(
                symbol=symbol,
                address=address,
                chain_id=chain_id,
                value_usd=value_usd,
                percentage=percentage,
                sector=enriched.get("sector") if enriched else None,
                subsector=enriched.get("subsector") if enriched else None,
                categories=enriched.get("categories", []) if enriched else [],
                project_slug=enriched.get("projectSlug") if enriched else None,
                related_tokens=[
                    r.get("address", "") for r in enriched.get("relatedTokens", [])
                ] if enriched else [],
            )
            token_holdings.append(holding)

        return PortfolioContext(
            holdings=token_holdings,
            total_value_usd=total_value,
        )

    def _profile_to_context(self, profile: Dict[str, Any]) -> PortfolioContext:
        """Convert a cached portfolio profile to context."""
        # This assumes the profile has pre-aggregated data
        holdings: List[TokenHolding] = []

        # Extract holdings from profile if available
        # For now, create minimal context from aggregated data
        sector_allocation = profile.get("sectorAllocation", {})
        category_exposure = profile.get("categoryExposure", {})

        # Create context with pre-set lookups
        # Set symbols to sentinel to prevent __post_init__ from rebuilding
        context = PortfolioContext(
            holdings=holdings,
            symbols={"__PROFILE_LOADED__"},  # Sentinel to skip _build_lookups
            sectors=sector_allocation,
            categories=category_exposure,
            total_value_usd=Decimal(str(profile.get("portfolioValueUsd", 0))),
        )
        # Remove sentinel
        context.symbols = set()

        return context

    async def get_personalized_news(
        self,
        wallet_address: str,
        holdings: Optional[List[Dict[str, Any]]] = None,
        limit: int = 20,
        min_relevance: float = 0.2,
        hours_back: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get news personalized for a portfolio.

        Args:
            wallet_address: User's wallet address
            holdings: Optional pre-fetched holdings
            limit: Max items to return
            min_relevance: Minimum relevance score
            hours_back: How far back to look for news

        Returns:
            List of news items with relevance scores, sorted by relevance
        """
        # Build portfolio context
        portfolio = await self.build_portfolio_context(wallet_address, holdings)

        # Fetch recent news
        if not self._convex:
            return []

        import time
        since_timestamp = int((time.time() - hours_back * 3600) * 1000)

        news_items = await self._convex.query(
            "news:getRecent",
            {"limit": 100, "sinceTimestamp": since_timestamp},
        )

        if not news_items:
            return []

        # Score each news item
        scored_items: List[Dict[str, Any]] = []

        for news in news_items:
            content = ContentContext.from_processed_news(news)
            score = self._scorer.score(content, portfolio)

            if score.score >= min_relevance:
                scored_items.append({
                    **news,
                    "relevance": score.to_dict(),
                })

        # Sort by relevance score (highest first)
        scored_items.sort(key=lambda x: x["relevance"]["score"], reverse=True)

        return scored_items[:limit]

    async def score_news_item(
        self,
        news: Dict[str, Any],
        portfolio: PortfolioContext,
    ) -> RelevanceScore:
        """
        Score a single news item against a portfolio.

        Args:
            news: Processed news item dict
            portfolio: Portfolio context

        Returns:
            RelevanceScore with breakdown
        """
        content = ContentContext.from_processed_news(news)
        return self._scorer.score(content, portfolio)

    async def score_news_batch(
        self,
        news_items: List[Dict[str, Any]],
        portfolio: PortfolioContext,
    ) -> List[Dict[str, Any]]:
        """
        Score multiple news items against a portfolio.

        Args:
            news_items: List of processed news items
            portfolio: Portfolio context

        Returns:
            News items with relevance scores added
        """
        results = []

        for news in news_items:
            content = ContentContext.from_processed_news(news)
            score = self._scorer.score(content, portfolio)
            results.append({
                **news,
                "relevance": score.to_dict(),
            })

        return results

    def score_content(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceScore:
        """
        Score arbitrary content against a portfolio.

        Args:
            content: Content context
            portfolio: Portfolio context

        Returns:
            RelevanceScore
        """
        return self._scorer.score(content, portfolio)

    async def get_relevant_tokens_for_news(
        self,
        news: Dict[str, Any],
        wallet_address: str,
        holdings: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get which held tokens are relevant to a news item and why.

        Args:
            news: News item
            wallet_address: User's wallet
            holdings: Optional pre-fetched holdings

        Returns:
            List of relevant holdings with explanations
        """
        portfolio = await self.build_portfolio_context(wallet_address, holdings)
        content = ContentContext.from_processed_news(news)
        score = self._scorer.score(content, portfolio)

        relevant_holdings = []

        # Check direct matches
        for token in content.tokens:
            holding = portfolio.get_holding(token)
            if holding:
                relevant_holdings.append({
                    "symbol": holding.symbol,
                    "percentage": holding.percentage,
                    "reason": "directly_mentioned",
                    "explanation": f"This news directly mentions {holding.symbol}",
                })

        # Check sector matches
        for sector in content.sectors:
            if portfolio.has_sector(sector):
                for holding in portfolio.holdings:
                    if holding.sector == sector and holding.symbol not in [r["symbol"] for r in relevant_holdings]:
                        relevant_holdings.append({
                            "symbol": holding.symbol,
                            "percentage": holding.percentage,
                            "reason": "sector_exposure",
                            "explanation": f"{holding.symbol} is in the {sector} sector",
                        })

        return relevant_holdings[:5]  # Limit to top 5


async def get_personalized_feed(
    wallet_address: str,
    holdings: List[Dict[str, Any]],
    convex_client: Any = None,
    token_catalog: Any = None,
    limit: int = 20,
    min_relevance: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Convenience function to get personalized news feed.

    Args:
        wallet_address: User's wallet
        holdings: Portfolio holdings
        convex_client: Convex client
        token_catalog: Token catalog service
        limit: Max items
        min_relevance: Minimum score

    Returns:
        Personalized news feed
    """
    service = RelevanceService(
        convex_client=convex_client,
        token_catalog_service=token_catalog,
    )

    return await service.get_personalized_news(
        wallet_address=wallet_address,
        holdings=holdings,
        limit=limit,
        min_relevance=min_relevance,
    )
