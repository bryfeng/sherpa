"""
News Fetcher Service

Aggregates crypto news from multiple sources, processes with LLM
for classification and sentiment, and stores for portfolio-relevant delivery.
"""

from .models import (
    NewsCategory,
    SentimentLabel,
    NewsItem,
    ProcessedNews,
    NewsSource,
    NewsSourceType,
    Sentiment,
    RelatedToken,
    Importance,
)
from .sources import RSSSource, CoinGeckoNewsSource, DefiLlamaNewsSource
from .processor import NewsProcessor
from .service import NewsFetcherService

__all__ = [
    # Models
    "NewsCategory",
    "SentimentLabel",
    "NewsItem",
    "ProcessedNews",
    "NewsSource",
    "NewsSourceType",
    "Sentiment",
    "RelatedToken",
    "Importance",
    # Sources
    "RSSSource",
    "CoinGeckoNewsSource",
    "DefiLlamaNewsSource",
    # Processor
    "NewsProcessor",
    # Service
    "NewsFetcherService",
]
