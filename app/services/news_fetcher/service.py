"""
News Fetcher Service

Main service that orchestrates news fetching, processing, and storage.
Designed for cron-based execution every 15 minutes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .models import (
    DEFAULT_RSS_SOURCES,
    NewsItem,
    NewsSource,
    NewsSourceType,
    ProcessedNews,
)
from .sources import (
    CoinGeckoNewsSource,
    DefiLlamaNewsSource,
    RSSSource,
    fetch_all_sources,
)
from .processor import NewsProcessor

logger = logging.getLogger(__name__)


class NewsFetcherService:
    """
    Main news fetcher service.

    Orchestrates:
    - Fetching from multiple sources
    - LLM processing for classification/sentiment
    - Storing to Convex database
    - Deduplication and cleanup
    """

    def __init__(
        self,
        convex_client: Any = None,
        llm_provider: Any = None,
        sources: Optional[List[NewsSource]] = None,
        fetch_interval_minutes: int = 15,
        news_retention_days: int = 7,
    ):
        """
        Initialize the news fetcher service.

        Args:
            convex_client: Convex client for database operations
            llm_provider: LLM provider for classification/sentiment
            sources: Custom news sources (uses defaults if not provided)
            fetch_interval_minutes: Interval between fetches (for dedup)
            news_retention_days: How long to keep news items
        """
        self._convex = convex_client
        self._llm_provider = llm_provider
        self._fetch_interval = fetch_interval_minutes
        self._retention_days = news_retention_days

        # Initialize sources
        self._sources = sources or self._get_default_sources()

        # Initialize fetchers
        self._rss_fetcher = RSSSource()
        self._coingecko_fetcher = CoinGeckoNewsSource()
        self._defillama_fetcher = DefiLlamaNewsSource()

        # Initialize processor
        self._processor = NewsProcessor(
            llm_provider=llm_provider,
            batch_size=5,
            max_concurrent=3,
        )

    def _get_default_sources(self) -> List[NewsSource]:
        """Get default news sources."""
        sources = list(DEFAULT_RSS_SOURCES)

        # Add API sources
        sources.append(NewsSource(
            name="coingecko",
            type=NewsSourceType.API,
            url="https://api.coingecko.com/api/v3",
            config={"rate_limit": "50/min"},
        ))

        sources.append(NewsSource(
            name="defillama",
            type=NewsSourceType.API,
            url="https://api.llama.fi",
            config={"rate_limit": "100/min"},
        ))

        return sources

    async def run_fetch_cycle(self) -> Dict[str, Any]:
        """
        Run a complete fetch cycle.

        This is the main entry point for cron execution.

        Returns:
            Statistics about the fetch cycle
        """
        start_time = datetime.now(timezone.utc)
        stats = {
            "started_at": start_time.isoformat(),
            "sources_attempted": 0,
            "sources_successful": 0,
            "items_fetched": 0,
            "items_new": 0,
            "items_processed": 0,
            "items_stored": 0,
            "errors": [],
        }

        try:
            # 1. Fetch from all sources
            logger.info("Starting news fetch cycle")
            items = await self._fetch_all()
            stats["items_fetched"] = len(items)
            stats["sources_attempted"] = len([s for s in self._sources if s.enabled])

            if not items:
                logger.info("No new items fetched")
                return stats

            # 2. Deduplicate against existing items
            new_items = await self._deduplicate(items)
            stats["items_new"] = len(new_items)

            if not new_items:
                logger.info("No new items after deduplication")
                return stats

            # 3. Process with LLM
            processed = await self._process_items(new_items)
            stats["items_processed"] = len(processed)

            # 4. Store to database
            stored = await self._store_items(processed)
            stats["items_stored"] = stored

            # 5. Cleanup expired items
            await self._cleanup_expired()

            # 6. Update source statuses
            await self._update_source_statuses()

            stats["sources_successful"] = len([s for s in self._sources if s.last_success_at])

        except Exception as e:
            logger.error(f"Error in fetch cycle: {e}")
            stats["errors"].append(str(e))

        finally:
            end_time = datetime.now(timezone.utc)
            stats["ended_at"] = end_time.isoformat()
            stats["duration_seconds"] = (end_time - start_time).total_seconds()

            logger.info(
                f"Fetch cycle complete: {stats['items_stored']} items stored "
                f"({stats['items_new']} new, {stats['items_processed']} processed) "
                f"in {stats['duration_seconds']:.1f}s"
            )

        return stats

    async def _fetch_all(self) -> List[NewsItem]:
        """Fetch news from all enabled sources."""
        return await fetch_all_sources(
            sources=self._sources,
            rss_fetcher=self._rss_fetcher,
            coingecko_fetcher=self._coingecko_fetcher,
            defillama_fetcher=self._defillama_fetcher,
        )

    async def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove items that already exist in the database."""
        if not self._convex:
            # No database, return all items
            return items

        # Check each item against the database
        new_items: List[NewsItem] = []

        for item in items:
            try:
                existing = await self._convex.query(
                    "news:get",
                    {"source": item.source, "sourceId": item.source_id},
                )

                if existing is None:
                    new_items.append(item)

            except Exception as e:
                logger.warning(f"Error checking for existing item: {e}")
                # Include item if we can't check (safer)
                new_items.append(item)

        logger.info(f"Deduplication: {len(items)} -> {len(new_items)} items")
        return new_items

    async def _process_items(self, items: List[NewsItem]) -> List[ProcessedNews]:
        """Process items with LLM for classification and sentiment."""
        return await self._processor.process_batch(items)

    async def _store_items(self, items: List[ProcessedNews]) -> int:
        """Store processed items to the database."""
        if not self._convex:
            logger.warning("No Convex client configured, skipping storage")
            return 0

        stored = 0
        batch_size = 10  # Convex batch limit

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            try:
                # Convert to Convex format
                convex_items = [item.to_dict() for item in batch]

                # Calculate expiry time
                expiry = datetime.now(timezone.utc) + timedelta(days=self._retention_days)
                for item in convex_items:
                    item["expiresAt"] = int(expiry.timestamp() * 1000)

                result = await self._convex.mutation(
                    "news:insertBatch",
                    {"items": convex_items},
                )

                # Handle None or list result
                if result is None:
                    stored += len(convex_items)  # Assume success if no error
                elif isinstance(result, list):
                    stored += sum(1 for r in result if r.get("created", False))
                else:
                    stored += len(convex_items)

            except Exception as e:
                logger.error(f"Error storing batch: {e}")

        return stored

    async def _cleanup_expired(self) -> int:
        """Clean up expired news items."""
        if not self._convex:
            return 0

        try:
            result = await self._convex.mutation(
                "news:deleteExpired",
                {"batchSize": 100},
            )
            deleted = result.get("deleted", 0)

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired news items")

            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up expired items: {e}")
            return 0

    async def _update_source_statuses(self):
        """Update source statuses in the database."""
        if not self._convex:
            return

        for source in self._sources:
            try:
                # Only update if we have status info
                if source.last_fetched_at:
                    await self._convex.mutation(
                        "news:updateSourceStatus",
                        {
                            "name": source.name,
                            "success": source.last_error is None,
                            "error": source.last_error,
                        },
                    )
            except Exception as e:
                logger.warning(f"Error updating source status for {source.name}: {e}")

    async def get_personalized_news(
        self,
        portfolio_tokens: List[Dict[str, Any]],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get news personalized for a user's portfolio.

        Args:
            portfolio_tokens: List of tokens with symbol, sector, categories, valueWeight
            limit: Maximum number of news items to return

        Returns:
            Ranked list of relevant news items
        """
        if not self._convex:
            return []

        try:
            return await self._convex.query(
                "news:getForPortfolio",
                {"tokens": portfolio_tokens, "limit": limit},
            )
        except Exception as e:
            logger.error(f"Error getting personalized news: {e}")
            return []

    async def get_recent_news(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        since_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get recent news items.

        Args:
            category: Optional category filter
            limit: Maximum items to return
            since_hours: Only return news from the last N hours

        Returns:
            List of news items
        """
        if not self._convex:
            return []

        since_timestamp = int(
            (datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp() * 1000
        )

        try:
            params: Dict[str, Any] = {
                "limit": limit,
                "sinceTimestamp": since_timestamp,
            }
            if category:
                params["category"] = category

            return await self._convex.query("news:getRecent", params)
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []

    async def get_token_news(
        self,
        symbols: List[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get news related to specific tokens.

        Args:
            symbols: Token symbols to search for
            limit: Maximum items to return

        Returns:
            List of relevant news items
        """
        if not self._convex:
            return []

        try:
            return await self._convex.query(
                "news:getByTokens",
                {"symbols": symbols, "limit": limit},
            )
        except Exception as e:
            logger.error(f"Error getting token news: {e}")
            return []

    async def initialize_sources(self):
        """Initialize news sources in the database."""
        if not self._convex:
            return

        for source in self._sources:
            try:
                await self._convex.mutation(
                    "news:upsertSource",
                    source.to_dict(),
                )
            except Exception as e:
                logger.warning(f"Error initializing source {source.name}: {e}")

    async def close(self):
        """Clean up resources."""
        await self._rss_fetcher.close()
        await self._coingecko_fetcher.close()
        await self._defillama_fetcher.close()


# Convenience function for cron job
async def run_news_fetch_cron(
    convex_client: Any = None,
    llm_provider: Any = None,
) -> Dict[str, Any]:
    """
    Run news fetch as a cron job.

    This function is designed to be called by a scheduler every 15 minutes.

    Args:
        convex_client: Convex client instance
        llm_provider: LLM provider instance

    Returns:
        Fetch cycle statistics
    """
    service = NewsFetcherService(
        convex_client=convex_client,
        llm_provider=llm_provider,
    )

    try:
        return await service.run_fetch_cycle()
    finally:
        await service.close()
