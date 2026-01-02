"""
News Fetcher Data Models

Defines the data structures for news items, sources, and processing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class NewsCategory(str, Enum):
    """News classification categories."""
    REGULATORY = "regulatory"      # Laws, regulations, government actions
    TECHNICAL = "technical"        # Protocol updates, technical developments
    PARTNERSHIP = "partnership"    # Partnerships, integrations, collaborations
    TOKENOMICS = "tokenomics"      # Token burns, airdrops, supply changes
    MARKET = "market"              # Price movements, trading volume, market analysis
    HACK = "hack"                  # Security incidents, exploits, hacks
    UPGRADE = "upgrade"            # Network upgrades, hard forks, migrations
    GENERAL = "general"            # General news that doesn't fit other categories


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

    @classmethod
    def from_score(cls, score: float) -> "SentimentLabel":
        """Convert a -1 to 1 score to a sentiment label."""
        if score <= -0.6:
            return cls.VERY_NEGATIVE
        elif score <= -0.2:
            return cls.NEGATIVE
        elif score <= 0.2:
            return cls.NEUTRAL
        elif score <= 0.6:
            return cls.POSITIVE
        else:
            return cls.VERY_POSITIVE


class NewsSourceType(str, Enum):
    """Types of news sources."""
    RSS = "rss"
    API = "api"
    SCRAPER = "scraper"


@dataclass
class Sentiment:
    """Sentiment analysis result."""
    score: float  # -1 to 1
    label: SentimentLabel
    confidence: float  # 0 to 1

    def __post_init__(self):
        if not -1 <= self.score <= 1:
            raise ValueError(f"Sentiment score must be between -1 and 1, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @classmethod
    def from_score(cls, score: float, confidence: float = 0.8) -> "Sentiment":
        """Create sentiment from score, auto-assigning label."""
        return cls(
            score=max(-1, min(1, score)),
            label=SentimentLabel.from_score(score),
            confidence=max(0, min(1, confidence)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "score": self.score,
            "label": self.label.value,
            "confidence": self.confidence,
        }


@dataclass
class RelatedToken:
    """A token related to a news item."""
    symbol: str
    relevance_score: float  # 0 to 1
    address: Optional[str] = None
    chain_id: Optional[int] = None

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        if not 0 <= self.relevance_score <= 1:
            raise ValueError(f"Relevance score must be between 0 and 1, got {self.relevance_score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "symbol": self.symbol,
            "relevanceScore": self.relevance_score,
        }
        if self.address:
            result["address"] = self.address
        if self.chain_id:
            result["chainId"] = self.chain_id
        return result


@dataclass
class Importance:
    """Importance scoring for a news item."""
    score: float  # 0 to 1
    factors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0 <= self.score <= 1:
            raise ValueError(f"Importance score must be between 0 and 1, got {self.score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "score": self.score,
            "factors": self.factors,
        }


@dataclass
class NewsItem:
    """Raw news item from a source before LLM processing."""
    source_id: str  # Unique ID within source
    source: str  # Source identifier (e.g., "rss:coindesk")
    title: str
    url: str
    published_at: datetime
    raw_content: Optional[str] = None
    summary: Optional[str] = None
    image_url: Optional[str] = None

    def __post_init__(self):
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.title:
            raise ValueError("title is required")
        if not self.url:
            raise ValueError("url is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Convex insertion."""
        result = {
            "sourceId": self.source_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "publishedAt": int(self.published_at.timestamp() * 1000),
            "relatedTokens": [],
            "relatedSectors": [],
            "relatedCategories": [],
            "processingVersion": 1,
        }
        if self.raw_content:
            result["rawContent"] = self.raw_content[:10000]  # Limit size
        if self.summary:
            result["summary"] = self.summary[:1000]
        if self.image_url:
            result["imageUrl"] = self.image_url
        return result


@dataclass
class ProcessedNews:
    """Fully processed news item with LLM-derived metadata."""
    source_id: str
    source: str
    title: str
    url: str
    published_at: datetime

    # LLM-derived fields
    category: NewsCategory
    sentiment: Sentiment
    summary: str
    related_tokens: List[RelatedToken]
    related_sectors: List[str]
    related_categories: List[str]
    importance: Importance

    # Optional fields
    image_url: Optional[str] = None
    raw_content: Optional[str] = None
    processed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Convex insertion."""
        result = {
            "sourceId": self.source_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "publishedAt": int(self.published_at.timestamp() * 1000),
            "category": self.category.value,
            "sentiment": self.sentiment.to_dict(),
            "summary": self.summary[:1000],
            "relatedTokens": [t.to_dict() for t in self.related_tokens],
            "relatedSectors": self.related_sectors,
            "relatedCategories": self.related_categories,
            "importance": self.importance.to_dict(),
            "processingVersion": 1,
        }
        if self.image_url:
            result["imageUrl"] = self.image_url
        if self.raw_content:
            result["rawContent"] = self.raw_content[:10000]
        if self.processed_at:
            result["processedAt"] = int(self.processed_at.timestamp() * 1000)
        return result

    @classmethod
    def from_news_item(
        cls,
        item: NewsItem,
        category: NewsCategory,
        sentiment: Sentiment,
        summary: str,
        related_tokens: List[RelatedToken],
        related_sectors: List[str],
        related_categories: List[str],
        importance: Importance,
    ) -> "ProcessedNews":
        """Create ProcessedNews from a NewsItem and processing results."""
        return cls(
            source_id=item.source_id,
            source=item.source,
            title=item.title,
            url=item.url,
            published_at=item.published_at,
            image_url=item.image_url,
            raw_content=item.raw_content,
            category=category,
            sentiment=sentiment,
            summary=summary,
            related_tokens=related_tokens,
            related_sectors=related_sectors,
            related_categories=related_categories,
            importance=importance,
            processed_at=datetime.utcnow(),
        )


@dataclass
class NewsSource:
    """Configuration for a news source."""
    name: str
    type: NewsSourceType
    url: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime status (not persisted)
    last_fetched_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Convex storage."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "url": self.url,
            "enabled": self.enabled,
            "errorCount": self.error_count,
        }
        if self.config:
            result["config"] = self.config
        if self.last_fetched_at:
            result["lastFetchedAt"] = int(self.last_fetched_at.timestamp() * 1000)
        if self.last_success_at:
            result["lastSuccessAt"] = int(self.last_success_at.timestamp() * 1000)
        if self.last_error:
            result["lastError"] = self.last_error
        return result


# Default news sources (free, no API key required)
DEFAULT_RSS_SOURCES = [
    NewsSource(
        name="coindesk",
        type=NewsSourceType.RSS,
        url="https://www.coindesk.com/arc/outboundfeeds/rss/",
        config={"category_hint": "general"},
    ),
    NewsSource(
        name="cointelegraph",
        type=NewsSourceType.RSS,
        url="https://cointelegraph.com/rss",
        config={"category_hint": "general"},
    ),
    NewsSource(
        name="theblock",
        type=NewsSourceType.RSS,
        url="https://www.theblock.co/rss.xml",
        config={"category_hint": "general"},
    ),
    NewsSource(
        name="decrypt",
        type=NewsSourceType.RSS,
        url="https://decrypt.co/feed",
        config={"category_hint": "general"},
    ),
    NewsSource(
        name="bitcoinmagazine",
        type=NewsSourceType.RSS,
        url="https://bitcoinmagazine.com/feed",
        config={"category_hint": "general"},
    ),
]
