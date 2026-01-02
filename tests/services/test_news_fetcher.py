"""
Tests for News Fetcher Service

Tests the models, sources, processor, and main service.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.news_fetcher.models import (
    NewsCategory,
    SentimentLabel,
    Sentiment,
    RelatedToken,
    Importance,
    NewsItem,
    ProcessedNews,
    NewsSource,
    NewsSourceType,
    DEFAULT_RSS_SOURCES,
)
from app.services.news_fetcher.sources import (
    RSSSource,
    CoinGeckoNewsSource,
    DefiLlamaNewsSource,
    NewsSourceFetcher,
)
from app.services.news_fetcher.processor import NewsProcessor
from app.services.news_fetcher.service import NewsFetcherService


# =============================================================================
# Model Tests
# =============================================================================

class TestNewsCategory:
    """Tests for NewsCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        expected = ["regulatory", "technical", "partnership", "tokenomics",
                    "market", "hack", "upgrade", "general"]
        for cat in expected:
            assert NewsCategory(cat) is not None

    def test_category_values(self):
        """Test category string values."""
        assert NewsCategory.REGULATORY.value == "regulatory"
        assert NewsCategory.HACK.value == "hack"
        assert NewsCategory.GENERAL.value == "general"


class TestSentimentLabel:
    """Tests for SentimentLabel enum."""

    def test_from_score_very_negative(self):
        """Test very negative sentiment detection."""
        assert SentimentLabel.from_score(-0.8) == SentimentLabel.VERY_NEGATIVE
        assert SentimentLabel.from_score(-1.0) == SentimentLabel.VERY_NEGATIVE

    def test_from_score_negative(self):
        """Test negative sentiment detection."""
        assert SentimentLabel.from_score(-0.4) == SentimentLabel.NEGATIVE
        assert SentimentLabel.from_score(-0.3) == SentimentLabel.NEGATIVE

    def test_from_score_neutral(self):
        """Test neutral sentiment detection."""
        assert SentimentLabel.from_score(0.0) == SentimentLabel.NEUTRAL
        assert SentimentLabel.from_score(0.1) == SentimentLabel.NEUTRAL
        assert SentimentLabel.from_score(-0.1) == SentimentLabel.NEUTRAL

    def test_from_score_positive(self):
        """Test positive sentiment detection."""
        assert SentimentLabel.from_score(0.4) == SentimentLabel.POSITIVE
        assert SentimentLabel.from_score(0.5) == SentimentLabel.POSITIVE

    def test_from_score_very_positive(self):
        """Test very positive sentiment detection."""
        assert SentimentLabel.from_score(0.8) == SentimentLabel.VERY_POSITIVE
        assert SentimentLabel.from_score(1.0) == SentimentLabel.VERY_POSITIVE


class TestSentiment:
    """Tests for Sentiment dataclass."""

    def test_from_score(self):
        """Test creating sentiment from score."""
        sentiment = Sentiment.from_score(0.5, 0.9)
        assert sentiment.score == 0.5
        assert sentiment.label == SentimentLabel.POSITIVE
        assert sentiment.confidence == 0.9

    def test_score_clamping(self):
        """Test that scores are clamped to valid range."""
        sentiment = Sentiment.from_score(1.5, 1.2)
        assert sentiment.score == 1.0
        assert sentiment.confidence == 1.0

        sentiment2 = Sentiment.from_score(-1.5, -0.1)
        assert sentiment2.score == -1.0
        assert sentiment2.confidence == 0.0

    def test_to_dict(self):
        """Test sentiment serialization."""
        sentiment = Sentiment.from_score(0.3, 0.7)
        d = sentiment.to_dict()

        assert d["score"] == 0.3
        assert d["label"] == "positive"
        assert d["confidence"] == 0.7

    def test_invalid_score_raises(self):
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError):
            Sentiment(score=2.0, label=SentimentLabel.POSITIVE, confidence=0.5)

    def test_invalid_confidence_raises(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            Sentiment(score=0.5, label=SentimentLabel.POSITIVE, confidence=1.5)


class TestRelatedToken:
    """Tests for RelatedToken dataclass."""

    def test_symbol_uppercase(self):
        """Test that symbols are uppercased."""
        token = RelatedToken(symbol="eth", relevance_score=0.8)
        assert token.symbol == "ETH"

    def test_to_dict(self):
        """Test token serialization."""
        token = RelatedToken(
            symbol="BTC",
            relevance_score=0.9,
            address="0x123",
            chain_id=1,
        )
        d = token.to_dict()

        assert d["symbol"] == "BTC"
        assert d["relevanceScore"] == 0.9
        assert d["address"] == "0x123"
        assert d["chainId"] == 1

    def test_invalid_relevance_raises(self):
        """Test that invalid relevance raises ValueError."""
        with pytest.raises(ValueError):
            RelatedToken(symbol="ETH", relevance_score=1.5)


class TestImportance:
    """Tests for Importance dataclass."""

    def test_basic_creation(self):
        """Test basic importance creation."""
        imp = Importance(score=0.7, factors=["hack", "major_protocol"])
        assert imp.score == 0.7
        assert "hack" in imp.factors

    def test_to_dict(self):
        """Test importance serialization."""
        imp = Importance(score=0.8, factors=["regulatory"])
        d = imp.to_dict()

        assert d["score"] == 0.8
        assert d["factors"] == ["regulatory"]

    def test_invalid_score_raises(self):
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError):
            Importance(score=1.5, factors=[])


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_basic_creation(self):
        """Test basic news item creation."""
        item = NewsItem(
            source_id="abc123",
            source="rss:coindesk",
            title="Bitcoin Hits New High",
            url="https://example.com/news/1",
            published_at=datetime.now(timezone.utc),
        )
        assert item.title == "Bitcoin Hits New High"
        assert item.source == "rss:coindesk"

    def test_to_dict(self):
        """Test news item serialization."""
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        item = NewsItem(
            source_id="abc123",
            source="rss:coindesk",
            title="Test News",
            url="https://example.com/news/1",
            published_at=now,
            summary="This is a test",
            raw_content="Full content here",
        )
        d = item.to_dict()

        assert d["sourceId"] == "abc123"
        assert d["source"] == "rss:coindesk"
        assert d["title"] == "Test News"
        assert d["publishedAt"] == int(now.timestamp() * 1000)
        assert d["summary"] == "This is a test"

    def test_missing_required_fields_raises(self):
        """Test that missing required fields raise ValueError."""
        with pytest.raises(ValueError):
            NewsItem(
                source_id="",
                source="test",
                title="Test",
                url="https://example.com",
                published_at=datetime.now(timezone.utc),
            )


class TestProcessedNews:
    """Tests for ProcessedNews dataclass."""

    def test_from_news_item(self):
        """Test creating processed news from raw item."""
        item = NewsItem(
            source_id="abc123",
            source="rss:coindesk",
            title="Test News",
            url="https://example.com",
            published_at=datetime.now(timezone.utc),
        )

        processed = ProcessedNews.from_news_item(
            item=item,
            category=NewsCategory.MARKET,
            sentiment=Sentiment.from_score(0.5),
            summary="Test summary",
            related_tokens=[RelatedToken("BTC", 0.9)],
            related_sectors=["Infrastructure"],
            related_categories=["l1"],
            importance=Importance(0.6, ["market_news"]),
        )

        assert processed.category == NewsCategory.MARKET
        assert processed.sentiment.score == 0.5
        assert len(processed.related_tokens) == 1
        assert processed.processed_at is not None


class TestNewsSource:
    """Tests for NewsSource dataclass."""

    def test_default_sources_exist(self):
        """Test that default RSS sources are defined."""
        assert len(DEFAULT_RSS_SOURCES) > 0
        for source in DEFAULT_RSS_SOURCES:
            assert source.type == NewsSourceType.RSS
            assert source.enabled is True

    def test_to_dict(self):
        """Test source serialization."""
        source = NewsSource(
            name="test",
            type=NewsSourceType.RSS,
            url="https://example.com/rss",
        )
        d = source.to_dict()

        assert d["name"] == "test"
        assert d["type"] == "rss"
        assert d["enabled"] is True


# =============================================================================
# Source Tests
# =============================================================================

class TestNewsSourceFetcher:
    """Tests for NewsSourceFetcher base class."""

    def test_generate_source_id(self):
        """Test source ID generation."""
        url1 = "https://example.com/news/1"
        url2 = "https://example.com/news/2"

        id1 = NewsSourceFetcher.generate_source_id(url1)
        id2 = NewsSourceFetcher.generate_source_id(url2)

        assert id1 != id2
        assert len(id1) == 16
        # Same URL should give same ID
        assert NewsSourceFetcher.generate_source_id(url1) == id1

    def test_clean_html(self):
        """Test HTML cleaning."""
        html = "<p>Hello <b>world</b>!</p>"
        clean = NewsSourceFetcher.clean_html(html)
        assert clean == "Hello world !"

        # Test with complex HTML
        complex_html = "<div class='test'><a href='#'>Link</a> text</div>"
        clean = NewsSourceFetcher.clean_html(complex_html)
        assert "Link" in clean
        assert "text" in clean
        assert "<" not in clean


class TestRSSSource:
    """Tests for RSS source fetcher."""

    @pytest.fixture
    def rss_fetcher(self):
        return RSSSource()

    def test_parse_date_rfc2822(self, rss_fetcher):
        """Test parsing RFC 2822 date format."""
        date_str = "Mon, 15 Jan 2024 12:00:00 +0000"
        dt = rss_fetcher._parse_date(date_str)

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_date_iso8601(self, rss_fetcher):
        """Test parsing ISO 8601 date format."""
        date_str = "2024-01-15T12:00:00Z"
        dt = rss_fetcher._parse_date(date_str)

        assert dt.year == 2024
        assert dt.month == 1

    def test_parse_date_invalid_returns_now(self, rss_fetcher):
        """Test that invalid dates return current time."""
        dt = rss_fetcher._parse_date("not a date")
        now = datetime.now(timezone.utc)

        # Should be close to now
        assert abs((dt - now).total_seconds()) < 5


class TestCoinGeckoNewsSource:
    """Tests for CoinGecko news source."""

    @pytest.fixture
    def cg_fetcher(self):
        return CoinGeckoNewsSource()

    def test_parse_date(self, cg_fetcher):
        """Test parsing CoinGecko date format."""
        date_str = "2024-01-15T12:00:00.000Z"
        dt = cg_fetcher._parse_date(date_str)

        assert dt.year == 2024
        assert dt.month == 1


class TestDefiLlamaNewsSource:
    """Tests for DefiLlama news source."""

    @pytest.fixture
    def dl_fetcher(self):
        return DefiLlamaNewsSource()

    @pytest.mark.asyncio
    async def test_fetcher_initialization(self, dl_fetcher):
        """Test fetcher initializes correctly."""
        assert dl_fetcher._client is None
        assert dl_fetcher._last_tvl_snapshot == {}


# =============================================================================
# Processor Tests
# =============================================================================

class TestNewsProcessor:
    """Tests for news processor."""

    @pytest.fixture
    def processor(self):
        return NewsProcessor(llm_provider=None)

    def test_rule_based_classification_hack(self, processor):
        """Test rule-based classification detects hacks."""
        content = "Title: Major DeFi Protocol Hacked\n\nContent: Millions stolen in exploit"
        result = processor._rule_based_classification(content)

        assert result["category"] == "hack"
        assert result["sentiment"]["score"] < 0
        assert result["importance"]["score"] >= 0.8

    def test_rule_based_classification_regulatory(self, processor):
        """Test rule-based classification detects regulatory news."""
        content = "Title: SEC Files Lawsuit Against Crypto Exchange\n\nContent: Regulation news"
        result = processor._rule_based_classification(content)

        assert result["category"] == "regulatory"

    def test_rule_based_classification_partnership(self, processor):
        """Test rule-based classification detects partnerships."""
        content = "Title: Protocol Announces Partnership with Tech Company\n\nContent: New integration launching soon"
        result = processor._rule_based_classification(content)

        assert result["category"] == "partnership"

    def test_rule_based_token_extraction(self, processor):
        """Test token extraction from content."""
        content = "Title: ETH and BTC See Major Gains\n\nContent: SOL also rises"
        result = processor._rule_based_classification(content)

        symbols = [t["symbol"] for t in result["related_tokens"]]
        assert "ETH" in symbols
        assert "BTC" in symbols

    def test_rule_based_sector_detection(self, processor):
        """Test sector detection."""
        content = "Title: New DEX Launch\n\nContent: DeFi protocol offers yield farming"
        result = processor._rule_based_classification(content)

        assert "DeFi" in result["sectors"]

    def test_parse_category_valid(self, processor):
        """Test parsing valid category."""
        assert processor._parse_category("hack") == NewsCategory.HACK
        assert processor._parse_category("REGULATORY") == NewsCategory.REGULATORY

    def test_parse_category_invalid_returns_general(self, processor):
        """Test parsing invalid category returns general."""
        assert processor._parse_category("invalid") == NewsCategory.GENERAL

    def test_parse_sentiment(self, processor):
        """Test parsing sentiment data."""
        sentiment = processor._parse_sentiment({"score": 0.5, "confidence": 0.8})

        assert sentiment.score == 0.5
        assert sentiment.confidence == 0.8
        assert sentiment.label == SentimentLabel.POSITIVE

    def test_parse_related_tokens(self, processor):
        """Test parsing related tokens."""
        tokens = processor._parse_related_tokens([
            {"symbol": "eth", "relevance": 0.9},
            {"symbol": "btc", "relevance": 0.7},
        ])

        assert len(tokens) == 2
        assert tokens[0].symbol == "ETH"
        assert tokens[0].relevance_score == 0.9

    def test_create_default_processed(self, processor):
        """Test creating default processed news."""
        item = NewsItem(
            source_id="test123",
            source="test",
            title="Test Hack News",
            url="https://example.com",
            published_at=datetime.now(timezone.utc),
        )

        processed = processor._create_default_processed(item)

        assert processed.source_id == "test123"
        assert processed.category is not None
        assert processed.sentiment is not None

    @pytest.mark.asyncio
    async def test_process_batch_without_llm(self, processor):
        """Test batch processing without LLM uses rule-based."""
        items = [
            NewsItem(
                source_id="1",
                source="test",
                title="Bitcoin Price Surges",
                url="https://example.com/1",
                published_at=datetime.now(timezone.utc),
            ),
            NewsItem(
                source_id="2",
                source="test",
                title="Protocol Hacked for $10M",
                url="https://example.com/2",
                published_at=datetime.now(timezone.utc),
            ),
        ]

        processed = await processor.process_batch(items)

        assert len(processed) == 2
        # Second item should be classified as hack
        assert processed[1].category == NewsCategory.HACK


# =============================================================================
# Service Tests
# =============================================================================

class TestNewsFetcherService:
    """Tests for main news fetcher service."""

    @pytest.fixture
    def service(self):
        return NewsFetcherService(convex_client=None, llm_provider=None)

    def test_default_sources_initialized(self, service):
        """Test that default sources are initialized."""
        assert len(service._sources) > 0

        # Should have RSS sources
        rss_sources = [s for s in service._sources if s.type == NewsSourceType.RSS]
        assert len(rss_sources) > 0

        # Should have API sources
        api_sources = [s for s in service._sources if s.type == NewsSourceType.API]
        assert len(api_sources) >= 2  # CoinGecko and DefiLlama

    def test_custom_sources(self):
        """Test service with custom sources."""
        custom = [NewsSource(
            name="custom",
            type=NewsSourceType.RSS,
            url="https://example.com/rss",
        )]

        service = NewsFetcherService(sources=custom)
        assert len(service._sources) == 1
        assert service._sources[0].name == "custom"

    @pytest.mark.asyncio
    async def test_deduplicate_without_convex(self, service):
        """Test deduplication returns all items when no Convex."""
        items = [
            NewsItem(
                source_id="1",
                source="test",
                title="Test",
                url="https://example.com",
                published_at=datetime.now(timezone.utc),
            ),
        ]

        result = await service._deduplicate(items)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_store_items_without_convex(self, service):
        """Test storing returns 0 when no Convex."""
        processed = [
            ProcessedNews(
                source_id="1",
                source="test",
                title="Test",
                url="https://example.com",
                published_at=datetime.now(timezone.utc),
                category=NewsCategory.GENERAL,
                sentiment=Sentiment.from_score(0),
                summary="Test",
                related_tokens=[],
                related_sectors=[],
                related_categories=[],
                importance=Importance(0.5, []),
            ),
        ]

        stored = await service._store_items(processed)
        assert stored == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_without_convex(self, service):
        """Test cleanup returns 0 when no Convex."""
        result = await service._cleanup_expired()
        assert result == 0

    @pytest.mark.asyncio
    async def test_close(self, service):
        """Test service cleanup."""
        await service.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_run_fetch_cycle_returns_stats(self, service):
        """Test fetch cycle returns statistics."""
        # Mock fetch to return empty
        service._fetch_all = AsyncMock(return_value=[])

        stats = await service.run_fetch_cycle()

        assert "started_at" in stats
        assert "items_fetched" in stats
        assert stats["items_fetched"] == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self):
        """Test full pipeline from raw item to processed."""
        # Create raw item
        item = NewsItem(
            source_id="integration-test-1",
            source="test:integration",
            title="Ethereum Upgrade Announced",
            url="https://example.com/eth-upgrade",
            published_at=datetime.now(timezone.utc),
            summary="Major Ethereum upgrade coming next month",
            raw_content="The Ethereum Foundation announced a major upgrade...",
        )

        # Process without LLM
        processor = NewsProcessor(llm_provider=None)
        processed = await processor.process_batch([item])

        assert len(processed) == 1
        p = processed[0]

        # Verify all fields are populated
        assert p.source_id == item.source_id
        assert p.category in NewsCategory
        assert p.sentiment.score >= -1 and p.sentiment.score <= 1
        assert p.summary is not None
        assert p.importance.score >= 0 and p.importance.score <= 1

        # Verify serialization works
        d = p.to_dict()
        assert "category" in d
        assert "sentiment" in d
        assert "relatedTokens" in d

    @pytest.mark.asyncio
    async def test_service_with_mock_convex(self):
        """Test service with mocked Convex client."""
        mock_convex = MagicMock()
        mock_convex.query = AsyncMock(return_value=None)  # No existing items
        mock_convex.mutation = AsyncMock(return_value=[{"created": True}])

        service = NewsFetcherService(
            convex_client=mock_convex,
            llm_provider=None,
        )

        # Mock fetch to return test items
        test_items = [
            NewsItem(
                source_id="mock-1",
                source="test",
                title="Test News",
                url="https://example.com",
                published_at=datetime.now(timezone.utc),
            ),
        ]
        service._fetch_all = AsyncMock(return_value=test_items)

        stats = await service.run_fetch_cycle()

        assert stats["items_fetched"] == 1
        assert stats["items_new"] == 1
        assert stats["items_processed"] == 1
        # Storage was called
        mock_convex.mutation.assert_called()

        await service.close()
