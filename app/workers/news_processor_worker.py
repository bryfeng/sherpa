"""
News Processor Worker

Background worker that processes unprocessed news items using LLM
for classification, sentiment analysis, and token extraction.

Designed to be run as a scheduled task (cron) or triggered on-demand.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.news_fetcher.batch_processor import (
    BatchNewsProcessor,
    BatchProcessingConfig,
    ProcessingStats,
)
from app.services.news_fetcher.models import NewsItem, ProcessedNews

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for the news processor worker."""
    batch_size: int = 10  # Items per batch
    max_items_per_run: int = 50  # Max items to process per run
    min_interval_seconds: int = 60  # Min time between runs
    max_daily_tokens: int = 100_000  # Daily LLM token budget
    enable_llm: bool = True  # Set to False to use only rule-based


@dataclass
class WorkerResult:
    """Result from a worker run."""
    started_at: datetime
    ended_at: datetime
    items_fetched: int
    items_processed: int
    items_updated: int
    items_failed: int
    llm_calls: int
    tokens_used: int
    errors: List[str]

    @property
    def duration_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "startedAt": self.started_at.isoformat(),
            "endedAt": self.ended_at.isoformat(),
            "durationSeconds": self.duration_seconds,
            "itemsFetched": self.items_fetched,
            "itemsProcessed": self.items_processed,
            "itemsUpdated": self.items_updated,
            "itemsFailed": self.items_failed,
            "llmCalls": self.llm_calls,
            "tokensUsed": self.tokens_used,
            "errors": self.errors,
        }


class NewsProcessorWorker:
    """
    Background worker for processing unprocessed news items.

    Fetches items from Convex, processes them with LLM batch processor,
    and updates Convex with the results.
    """

    def __init__(
        self,
        convex_client: Any,
        llm_provider: Any = None,
        config: Optional[WorkerConfig] = None,
    ):
        """
        Initialize the worker.

        Args:
            convex_client: Convex client for database operations
            llm_provider: LLM provider for classification
            config: Worker configuration
        """
        self._convex = convex_client
        self._llm = llm_provider
        self._config = config or WorkerConfig()

        # Initialize batch processor
        batch_config = BatchProcessingConfig(
            batch_size=self._config.batch_size,
            max_daily_tokens=self._config.max_daily_tokens,
        )
        self._processor = BatchNewsProcessor(
            llm_provider=llm_provider if self._config.enable_llm else None,
            config=batch_config,
        )

        self._last_run: Optional[datetime] = None

    async def run(self) -> WorkerResult:
        """
        Run one processing cycle.

        Returns:
            WorkerResult with statistics
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        items_updated = 0
        items_failed = 0
        stats = ProcessingStats()

        try:
            # Check minimum interval
            if self._last_run:
                elapsed = (started_at - self._last_run).total_seconds()
                if elapsed < self._config.min_interval_seconds:
                    logger.info(
                        f"Skipping run, only {elapsed:.0f}s since last run "
                        f"(min: {self._config.min_interval_seconds}s)"
                    )
                    return WorkerResult(
                        started_at=started_at,
                        ended_at=datetime.utcnow(),
                        items_fetched=0,
                        items_processed=0,
                        items_updated=0,
                        items_failed=0,
                        llm_calls=0,
                        tokens_used=0,
                        errors=[],
                    )

            # 1. Fetch unprocessed items from Convex
            logger.info("Fetching unprocessed news items...")
            raw_items = await self._fetch_unprocessed()

            if not raw_items:
                logger.info("No unprocessed items found")
                self._last_run = datetime.utcnow()
                return WorkerResult(
                    started_at=started_at,
                    ended_at=datetime.utcnow(),
                    items_fetched=0,
                    items_processed=0,
                    items_updated=0,
                    items_failed=0,
                    llm_calls=0,
                    tokens_used=0,
                    errors=[],
                )

            logger.info(f"Found {len(raw_items)} unprocessed items")

            # 2. Convert to NewsItem objects
            news_items = self._convert_to_news_items(raw_items)

            # 3. Process with batch processor
            logger.info("Processing items with batch processor...")
            processed, stats = await self._processor.process_items(news_items)

            logger.info(
                f"Processed {len(processed)} items "
                f"(LLM calls: {stats.llm_calls}, tokens: {stats.tokens_used})"
            )

            # 4. Update Convex with results
            logger.info("Updating Convex with processed results...")
            for i, (raw, proc) in enumerate(zip(raw_items, processed)):
                try:
                    await self._update_processed(raw, proc)
                    items_updated += 1
                except Exception as e:
                    logger.error(f"Error updating item {i}: {e}")
                    errors.append(f"Update error: {str(e)}")
                    items_failed += 1

            self._last_run = datetime.utcnow()

        except Exception as e:
            logger.error(f"Worker error: {e}")
            errors.append(str(e))

        ended_at = datetime.utcnow()

        result = WorkerResult(
            started_at=started_at,
            ended_at=ended_at,
            items_fetched=len(raw_items) if 'raw_items' in locals() else 0,
            items_processed=stats.processed_items,
            items_updated=items_updated,
            items_failed=items_failed,
            llm_calls=stats.llm_calls,
            tokens_used=stats.tokens_used,
            errors=errors + stats.errors,
        )

        logger.info(
            f"Worker completed: {items_updated} updated, {items_failed} failed "
            f"in {result.duration_seconds:.1f}s"
        )

        return result

    async def _fetch_unprocessed(self) -> List[Dict[str, Any]]:
        """Fetch unprocessed items from Convex.

        Raises:
            Exception: If query fails (propagated to caller for error tracking)
        """
        items = await self._convex.query(
            "news:getUnprocessed",
            {"limit": self._config.max_items_per_run},
        )
        return items or []

    def _convert_to_news_items(self, raw_items: List[Dict[str, Any]]) -> List[NewsItem]:
        """Convert Convex documents to NewsItem objects."""
        news_items: List[NewsItem] = []

        for item in raw_items:
            try:
                # Parse published timestamp
                published_ms = item.get("publishedAt", 0)
                published_at = datetime.fromtimestamp(published_ms / 1000)

                news_items.append(NewsItem(
                    source_id=item.get("sourceId", ""),
                    source=item.get("source", ""),
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    published_at=published_at,
                    summary=item.get("summary"),
                    raw_content=item.get("rawContent"),
                    image_url=item.get("imageUrl"),
                ))
            except Exception as e:
                logger.warning(f"Error converting item: {e}")
                continue

        return news_items

    async def _update_processed(
        self,
        raw_item: Dict[str, Any],
        processed: ProcessedNews,
    ):
        """Update a processed item in Convex."""
        await self._convex.mutation(
            "news:updateProcessing",
            {
                "source": raw_item.get("source"),
                "sourceId": raw_item.get("sourceId"),
                "category": processed.category.value,
                "sentiment": processed.sentiment.to_dict(),
                "summary": processed.summary,
                "relatedTokens": [t.to_dict() for t in processed.related_tokens],
                "relatedSectors": processed.related_sectors,
                "relatedCategories": processed.related_categories,
                "importance": processed.importance.to_dict(),
            },
        )


async def run_news_processor_worker(
    convex_client: Any,
    llm_provider: Any = None,
    config: Optional[WorkerConfig] = None,
) -> WorkerResult:
    """
    Run the news processor worker once.

    Convenience function for scheduled execution.

    Args:
        convex_client: Convex client
        llm_provider: LLM provider (optional)
        config: Worker configuration

    Returns:
        WorkerResult with statistics
    """
    worker = NewsProcessorWorker(
        convex_client=convex_client,
        llm_provider=llm_provider,
        config=config,
    )
    return await worker.run()


async def run_news_processor_loop(
    convex_client: Any,
    llm_provider: Any = None,
    config: Optional[WorkerConfig] = None,
    interval_seconds: int = 300,  # 5 minutes
    max_iterations: Optional[int] = None,
):
    """
    Run the news processor worker in a continuous loop.

    Useful for background processing in development or simple deployments.

    Args:
        convex_client: Convex client
        llm_provider: LLM provider (optional)
        config: Worker configuration
        interval_seconds: Seconds between runs
        max_iterations: Max iterations (None for infinite)
    """
    worker = NewsProcessorWorker(
        convex_client=convex_client,
        llm_provider=llm_provider,
        config=config,
    )

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            result = await worker.run()
            logger.info(f"Worker iteration {iterations + 1}: {result.items_updated} items processed")
        except Exception as e:
            logger.error(f"Worker iteration {iterations + 1} failed: {e}")

        iterations += 1

        if max_iterations is None or iterations < max_iterations:
            logger.info(f"Sleeping {interval_seconds}s until next run...")
            await asyncio.sleep(interval_seconds)
