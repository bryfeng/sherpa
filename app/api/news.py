"""
News API Endpoints

Provides endpoints for fetching and managing news.
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import logging

from ..config import settings
from ..db import get_convex_client
from ..services.news_fetcher.service import NewsFetcherService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/news")


# =============================================================================
# Response Models
# =============================================================================


class NewsItem(BaseModel):
    """News item response."""
    title: str
    summary: Optional[str] = None
    url: str
    source: str
    published_at: int
    category: str = "general"
    sentiment: Optional[dict] = None
    related_tokens: List[str] = []
    importance: float = 0.5


class NewsResponse(BaseModel):
    """Response containing news items."""
    items: List[NewsItem]
    count: int
    category: Optional[str] = None


class FetchResult(BaseModel):
    """Result of a news fetch cycle."""
    success: bool
    fetched: int = 0
    new: int = 0
    processed: int = 0
    errors: int = 0
    message: Optional[str] = None


# =============================================================================
# Public Endpoints
# =============================================================================


@router.get("", response_model=NewsResponse)
async def get_recent_news(
    category: Optional[str] = None,
    limit: int = 20,
    hours_back: int = 24,
):
    """Get recent news items."""
    try:
        convex = get_convex_client()
        service = NewsFetcherService(convex_client=convex)

        items = await service.get_recent_news(
            category=category,
            limit=limit,
            since_hours=hours_back,
        )

        # Format response
        formatted = []
        for item in items:
            formatted.append(NewsItem(
                title=item.get("title", ""),
                summary=item.get("summary"),
                url=item.get("url", ""),
                source=item.get("source", ""),
                published_at=item.get("publishedAt", 0),
                category=item.get("category", "general"),
                sentiment=item.get("sentiment"),
                related_tokens=[t.get("symbol", "") for t in item.get("relatedTokens", [])],
                importance=item.get("importance", {}).get("score", 0.5),
            ))

        return NewsResponse(
            items=formatted,
            count=len(formatted),
            category=category,
        )
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token/{symbol}", response_model=NewsResponse)
async def get_token_news(
    symbol: str,
    limit: int = 20,
):
    """Get news for a specific token."""
    try:
        convex = get_convex_client()
        service = NewsFetcherService(convex_client=convex)

        items = await service.get_token_news(
            symbols=[symbol.upper()],
            limit=limit,
        )

        formatted = []
        for item in items:
            formatted.append(NewsItem(
                title=item.get("title", ""),
                summary=item.get("summary"),
                url=item.get("url", ""),
                source=item.get("source", ""),
                published_at=item.get("publishedAt", 0),
                category=item.get("category", "general"),
                sentiment=item.get("sentiment"),
                related_tokens=[t.get("symbol", "") for t in item.get("relatedTokens", [])],
                importance=item.get("importance", {}).get("score", 0.5),
            ))

        return NewsResponse(
            items=formatted,
            count=len(formatted),
        )
    except Exception as e:
        logger.error(f"Error fetching token news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Internal Endpoints (called by Convex cron)
# =============================================================================


@router.post("/internal/fetch", response_model=FetchResult)
async def trigger_news_fetch(
    x_internal_key: str = Header(..., alias="X-Internal-Key"),
):
    """
    Trigger a news fetch cycle.

    Called by Convex cron job every 15 minutes.
    """
    # Validate internal API key
    if x_internal_key != settings.convex_internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")

    try:
        convex = get_convex_client()

        # Run the fetch cycle
        service = NewsFetcherService(convex_client=convex)
        stats = await service.run_fetch_cycle()

        return FetchResult(
            success=True,
            fetched=stats.get("fetched", 0),
            new=stats.get("new", 0),
            processed=stats.get("processed", 0),
            errors=stats.get("errors", 0),
        )
    except Exception as e:
        logger.error(f"Error in news fetch cycle: {e}")
        return FetchResult(
            success=False,
            message=str(e),
        )


@router.post("/internal/process", response_model=FetchResult)
async def trigger_news_processing(
    x_internal_key: str = Header(..., alias="X-Internal-Key"),
    limit: int = 50,
):
    """
    Trigger LLM processing of unprocessed news items.

    Called by Convex cron job every 5 minutes.
    """
    from ..workers.news_processor_worker import run_news_processor_worker, WorkerConfig

    # Validate internal API key
    if x_internal_key != settings.convex_internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")

    try:
        convex = get_convex_client()

        # Run the processor with config
        config = WorkerConfig(max_items_per_run=limit)
        result = await run_news_processor_worker(
            convex_client=convex,
            config=config,
        )

        return FetchResult(
            success=result.items_failed == 0,
            processed=result.items_processed,
            errors=result.items_failed,
        )
    except Exception as e:
        logger.error(f"Error in news processing: {e}")
        return FetchResult(
            success=False,
            message=str(e),
        )
