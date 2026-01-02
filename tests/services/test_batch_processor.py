"""
Tests for Batch News Processor

Tests the cost-efficient batch LLM processing of news items.
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from app.services.news_fetcher.batch_processor import (
    BatchNewsProcessor,
    BatchProcessingConfig,
    ProcessingStats,
    process_news_batch,
)
from app.services.news_fetcher.models import (
    NewsCategory,
    NewsItem,
    ProcessedNews,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_news_items():
    """Sample news items for testing."""
    return [
        NewsItem(
            source_id="1",
            source="rss:coindesk",
            title="Ethereum Completes Major Upgrade",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
            summary="Ethereum successfully completes the Dencun upgrade.",
            raw_content="The Ethereum network has successfully completed...",
        ),
        NewsItem(
            source_id="2",
            source="rss:cointelegraph",
            title="SEC Approves New Bitcoin ETF",
            url="https://example.com/2",
            published_at=datetime.utcnow(),
            summary="The SEC has approved another spot Bitcoin ETF.",
        ),
        NewsItem(
            source_id="3",
            source="api:coingecko",
            title="DeFi Protocol Loses $50M in Exploit",
            url="https://example.com/3",
            published_at=datetime.utcnow(),
            summary="A major DeFi protocol suffered a flash loan exploit.",
        ),
    ]


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    return provider


@pytest.fixture
def sample_llm_response():
    """Sample LLM batch response."""
    return [
        {
            "index": 0,
            "category": "upgrade",
            "sentiment": {"score": 0.7, "confidence": 0.9},
            "summary": "Ethereum completes Dencun upgrade successfully.",
            "tokens": [{"symbol": "ETH", "relevance": 0.95}],
            "sectors": ["Infrastructure"],
            "categories": ["l1", "upgrade"],
            "importance": {"score": 0.8, "factors": ["network_upgrade"]},
        },
        {
            "index": 1,
            "category": "regulatory",
            "sentiment": {"score": 0.5, "confidence": 0.8},
            "summary": "SEC approves new spot Bitcoin ETF.",
            "tokens": [{"symbol": "BTC", "relevance": 0.9}],
            "sectors": ["Infrastructure"],
            "categories": ["l1"],
            "importance": {"score": 0.9, "factors": ["regulatory_approval"]},
        },
        {
            "index": 2,
            "category": "hack",
            "sentiment": {"score": -0.8, "confidence": 0.95},
            "summary": "DeFi protocol loses $50M in flash loan exploit.",
            "tokens": [{"symbol": "ETH", "relevance": 0.6}],
            "sectors": ["DeFi"],
            "categories": ["defi"],
            "importance": {"score": 0.95, "factors": ["security_incident"]},
        },
    ]


# =============================================================================
# BatchProcessingConfig Tests
# =============================================================================


class TestBatchProcessingConfig:
    """Tests for BatchProcessingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchProcessingConfig()
        assert config.batch_size == 10
        assert config.max_concurrent_batches == 2
        assert config.max_daily_tokens == 100_000
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchProcessingConfig(
            batch_size=5,
            max_daily_tokens=50_000,
        )
        assert config.batch_size == 5
        assert config.max_daily_tokens == 50_000


# =============================================================================
# ProcessingStats Tests
# =============================================================================


class TestProcessingStats:
    """Tests for ProcessingStats."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = ProcessingStats()
        assert stats.total_items == 0
        assert stats.processed_items == 0
        assert stats.llm_calls == 0
        assert stats.errors == []

    def test_accumulation(self):
        """Test accumulating statistics."""
        stats = ProcessingStats()
        stats.total_items = 10
        stats.processed_items = 8
        stats.failed_items = 2
        stats.llm_calls = 3

        assert stats.total_items == 10
        assert stats.processed_items == 8


# =============================================================================
# BatchNewsProcessor Tests
# =============================================================================


class TestBatchNewsProcessor:
    """Tests for BatchNewsProcessor."""

    @pytest.mark.asyncio
    async def test_process_empty_list(self):
        """Test processing empty list."""
        processor = BatchNewsProcessor()
        processed, stats = await processor.process_items([])

        assert processed == []
        assert stats.total_items == 0
        assert stats.processed_items == 0

    @pytest.mark.asyncio
    async def test_process_with_rules_fallback(self, sample_news_items):
        """Test processing with rule-based fallback (no LLM)."""
        processor = BatchNewsProcessor(llm_provider=None)
        processed, stats = await processor.process_items(sample_news_items)

        assert len(processed) == 3
        assert stats.fallback_used == 3
        assert stats.llm_calls == 0

        # Check rule-based classification
        # Upgrade item should be classified as upgrade
        assert processed[0].category == NewsCategory.UPGRADE

        # SEC/regulatory should be detected
        assert processed[1].category == NewsCategory.REGULATORY

        # Hack/exploit should be detected
        assert processed[2].category == NewsCategory.HACK

    @pytest.mark.asyncio
    async def test_process_with_llm(
        self,
        sample_news_items,
        mock_llm_provider,
        sample_llm_response,
    ):
        """Test processing with LLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = json.dumps(sample_llm_response)
        mock_response.tokens_used = 500
        mock_llm_provider.generate_response = AsyncMock(return_value=mock_response)

        processor = BatchNewsProcessor(llm_provider=mock_llm_provider)
        processed, stats = await processor.process_items(sample_news_items)

        assert len(processed) == 3
        assert stats.llm_calls == 1
        assert stats.tokens_used == 500

        # Check LLM classifications were applied
        assert processed[0].category == NewsCategory.UPGRADE
        assert processed[1].category == NewsCategory.REGULATORY
        assert processed[2].category == NewsCategory.HACK

        # Check sentiment
        assert processed[0].sentiment.score == 0.7
        assert processed[2].sentiment.score == -0.8

        # Check tokens extracted
        assert processed[0].related_tokens[0].symbol == "ETH"
        assert processed[1].related_tokens[0].symbol == "BTC"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, sample_news_items, mock_llm_provider):
        """Test fallback to rules when LLM fails."""
        mock_llm_provider.generate_response = AsyncMock(
            side_effect=Exception("LLM error")
        )

        processor = BatchNewsProcessor(
            llm_provider=mock_llm_provider,
            config=BatchProcessingConfig(fallback_to_rules=True),
        )
        processed, stats = await processor.process_items(sample_news_items)

        assert len(processed) == 3
        assert stats.fallback_used == 3
        assert len(stats.errors) >= 1

    @pytest.mark.asyncio
    async def test_parse_json_response(self, sample_llm_response):
        """Test JSON parsing from LLM response."""
        processor = BatchNewsProcessor()

        # Test plain JSON
        result = processor._parse_json_array(json.dumps(sample_llm_response))
        assert len(result) == 3

        # Test JSON in markdown code block
        markdown = f"```json\n{json.dumps(sample_llm_response)}\n```"
        result = processor._parse_json_array(markdown)
        assert len(result) == 3

        # Test JSON with extra text
        with_text = f"Here is the analysis:\n{json.dumps(sample_llm_response)}"
        result = processor._parse_json_array(with_text)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, mock_llm_provider, sample_llm_response):
        """Test that batch size is respected."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(sample_llm_response[:2])
        mock_response.tokens_used = 300
        mock_llm_provider.generate_response = AsyncMock(return_value=mock_response)

        # Create 5 items with batch size of 2
        items = [
            NewsItem(
                source_id=str(i),
                source="test",
                title=f"Test {i}",
                url=f"https://example.com/{i}",
                published_at=datetime.utcnow(),
            )
            for i in range(5)
        ]

        processor = BatchNewsProcessor(
            llm_provider=mock_llm_provider,
            config=BatchProcessingConfig(batch_size=2),
        )
        processed, stats = await processor.process_items(items)

        # Should have made 3 batches (2+2+1)
        assert stats.llm_calls == 3

    @pytest.mark.asyncio
    async def test_daily_token_limit(self, sample_news_items, mock_llm_provider):
        """Test daily token limit enforcement."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([])
        mock_response.tokens_used = 100_000
        mock_llm_provider.generate_response = AsyncMock(return_value=mock_response)

        processor = BatchNewsProcessor(
            llm_provider=mock_llm_provider,
            config=BatchProcessingConfig(max_daily_tokens=100),
        )

        # First batch should use LLM
        _, stats1 = await processor.process_items(sample_news_items[:1])

        # Second batch should fall back (exceeded budget)
        _, stats2 = await processor.process_items(sample_news_items[1:])

        # Second batch should use fallback
        assert stats2.fallback_used >= 1


# =============================================================================
# Rule-Based Classification Tests
# =============================================================================


class TestRuleBasedClassification:
    """Tests for rule-based classification fallback."""

    @pytest.fixture
    def processor(self):
        return BatchNewsProcessor(llm_provider=None)

    def test_hack_detection(self, processor):
        """Test hack/exploit detection."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="Protocol Suffers $100M Hack",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
            summary="Hackers exploited a vulnerability to steal funds.",
        )
        processed = processor._process_item_with_rules(item)
        assert processed.category == NewsCategory.HACK
        assert processed.sentiment.score < 0

    def test_regulatory_detection(self, processor):
        """Test regulatory news detection."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="SEC Files Lawsuit Against Exchange",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
        )
        processed = processor._process_item_with_rules(item)
        assert processed.category == NewsCategory.REGULATORY

    def test_upgrade_detection(self, processor):
        """Test upgrade news detection."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="Network Completes Hard Fork Upgrade",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
        )
        processed = processor._process_item_with_rules(item)
        assert processed.category == NewsCategory.UPGRADE

    def test_partnership_detection(self, processor):
        """Test partnership detection."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="Protocol Announces Integration with Major Exchange",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
        )
        processed = processor._process_item_with_rules(item)
        assert processed.category == NewsCategory.PARTNERSHIP

    def test_token_extraction(self, processor):
        """Test token symbol extraction."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="ETH and BTC Rally as Market Recovers",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
            summary="ETH leads the rally with 10% gains.",
        )
        processed = processor._process_item_with_rules(item)

        symbols = [t.symbol for t in processed.related_tokens]
        assert "ETH" in symbols
        assert "BTC" in symbols

    def test_sector_detection(self, processor):
        """Test sector detection."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="New DeFi DEX Launches with Innovative Yield Farming",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
        )
        processed = processor._process_item_with_rules(item)
        assert "DeFi" in processed.related_sectors


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for process_news_batch function."""

    @pytest.mark.asyncio
    async def test_process_news_batch(self, sample_news_items):
        """Test the convenience function."""
        processed, stats = await process_news_batch(
            items=sample_news_items,
            llm_provider=None,  # Use rules
        )

        assert len(processed) == 3
        assert stats.total_items == 3
        assert stats.processed_items == 3

    @pytest.mark.asyncio
    async def test_process_news_batch_with_config(self, sample_news_items):
        """Test with custom configuration."""
        config = BatchProcessingConfig(batch_size=1)
        processed, stats = await process_news_batch(
            items=sample_news_items,
            config=config,
        )

        assert len(processed) == 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_malformed_llm_response(self, sample_news_items, mock_llm_provider):
        """Test handling of malformed LLM response."""
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON"
        mock_response.tokens_used = 100
        mock_llm_provider.generate_response = AsyncMock(return_value=mock_response)

        processor = BatchNewsProcessor(
            llm_provider=mock_llm_provider,
            config=BatchProcessingConfig(fallback_to_rules=True),
        )
        processed, stats = await processor.process_items(sample_news_items)

        # Should fall back to rules
        assert len(processed) == 3
        assert stats.fallback_used == 3

    @pytest.mark.asyncio
    async def test_partial_llm_response(
        self,
        sample_news_items,
        mock_llm_provider,
        sample_llm_response,
    ):
        """Test handling when LLM returns fewer results than items."""
        # Only return 2 results for 3 items
        mock_response = MagicMock()
        mock_response.content = json.dumps(sample_llm_response[:2])
        mock_response.tokens_used = 300
        mock_llm_provider.generate_response = AsyncMock(return_value=mock_response)

        processor = BatchNewsProcessor(llm_provider=mock_llm_provider)
        processed, stats = await processor.process_items(sample_news_items)

        # Should still process all items (missing one falls back)
        assert len(processed) == 3
        assert stats.fallback_used >= 1

    @pytest.mark.asyncio
    async def test_empty_content_item(self):
        """Test processing item with minimal content."""
        item = NewsItem(
            source_id="1",
            source="test",
            title="News",
            url="https://example.com/1",
            published_at=datetime.utcnow(),
        )

        processor = BatchNewsProcessor(llm_provider=None)
        processed, stats = await processor.process_items([item])

        assert len(processed) == 1
        assert processed[0].category == NewsCategory.GENERAL
