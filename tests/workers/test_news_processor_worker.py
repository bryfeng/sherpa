"""
Tests for News Processor Worker

Tests the background worker that processes unprocessed news items.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.news_processor_worker import (
    NewsProcessorWorker,
    WorkerConfig,
    WorkerResult,
    run_news_processor_worker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_convex_client():
    """Mock Convex client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    return provider


@pytest.fixture
def sample_unprocessed_items():
    """Sample unprocessed items from Convex."""
    now = datetime.utcnow()
    return [
        {
            "_id": "item1",
            "sourceId": "1",
            "source": "rss:coindesk",
            "title": "Ethereum Completes Upgrade",
            "url": "https://example.com/1",
            "publishedAt": int(now.timestamp() * 1000),
            "summary": "Major upgrade completed.",
        },
        {
            "_id": "item2",
            "sourceId": "2",
            "source": "rss:cointelegraph",
            "title": "Bitcoin ETF Approved",
            "url": "https://example.com/2",
            "publishedAt": int((now - timedelta(hours=1)).timestamp() * 1000),
            "summary": "SEC approves new ETF.",
        },
    ]


# =============================================================================
# WorkerConfig Tests
# =============================================================================


class TestWorkerConfig:
    """Tests for WorkerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkerConfig()
        assert config.batch_size == 10
        assert config.max_items_per_run == 50
        assert config.min_interval_seconds == 60
        assert config.enable_llm is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = WorkerConfig(
            batch_size=5,
            max_items_per_run=20,
            enable_llm=False,
        )
        assert config.batch_size == 5
        assert config.max_items_per_run == 20
        assert config.enable_llm is False


# =============================================================================
# WorkerResult Tests
# =============================================================================


class TestWorkerResult:
    """Tests for WorkerResult."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime(2025, 1, 1, 12, 0, 0)
        end = datetime(2025, 1, 1, 12, 0, 30)

        result = WorkerResult(
            started_at=start,
            ended_at=end,
            items_fetched=10,
            items_processed=8,
            items_updated=8,
            items_failed=2,
            llm_calls=2,
            tokens_used=1000,
            errors=[],
        )

        assert result.duration_seconds == 30.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = WorkerResult(
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            ended_at=datetime(2025, 1, 1, 12, 0, 30),
            items_fetched=10,
            items_processed=8,
            items_updated=8,
            items_failed=2,
            llm_calls=2,
            tokens_used=1000,
            errors=["Error 1"],
        )

        d = result.to_dict()
        assert d["itemsFetched"] == 10
        assert d["itemsProcessed"] == 8
        assert d["llmCalls"] == 2
        assert d["durationSeconds"] == 30.0
        assert len(d["errors"]) == 1


# =============================================================================
# NewsProcessorWorker Tests
# =============================================================================


class TestNewsProcessorWorker:
    """Tests for NewsProcessorWorker."""

    @pytest.mark.asyncio
    async def test_run_no_items(self, mock_convex_client):
        """Test run with no unprocessed items."""
        mock_convex_client.query.return_value = []

        worker = NewsProcessorWorker(
            convex_client=mock_convex_client,
        )
        result = await worker.run()

        assert result.items_fetched == 0
        assert result.items_processed == 0
        assert result.items_updated == 0
        mock_convex_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_items(self, mock_convex_client, sample_unprocessed_items):
        """Test run with unprocessed items."""
        mock_convex_client.query.return_value = sample_unprocessed_items
        mock_convex_client.mutation.return_value = None

        worker = NewsProcessorWorker(
            convex_client=mock_convex_client,
            llm_provider=None,  # Use rule-based
        )
        result = await worker.run()

        assert result.items_fetched == 2
        assert result.items_processed == 2
        assert result.items_updated == 2
        assert result.items_failed == 0

        # Should have called updateProcessing for each item
        assert mock_convex_client.mutation.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_update_error(self, mock_convex_client, sample_unprocessed_items):
        """Test handling of update errors."""
        mock_convex_client.query.return_value = sample_unprocessed_items
        mock_convex_client.mutation.side_effect = [
            None,  # First update succeeds
            Exception("Update failed"),  # Second fails
        ]

        worker = NewsProcessorWorker(
            convex_client=mock_convex_client,
        )
        result = await worker.run()

        assert result.items_fetched == 2
        assert result.items_updated == 1
        assert result.items_failed == 1
        assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_min_interval_enforcement(self, mock_convex_client):
        """Test minimum interval between runs."""
        mock_convex_client.query.return_value = []

        worker = NewsProcessorWorker(
            convex_client=mock_convex_client,
            config=WorkerConfig(min_interval_seconds=60),
        )

        # First run
        result1 = await worker.run()
        assert result1.items_fetched == 0

        # Second run immediately (should be skipped)
        result2 = await worker.run()
        assert result2.items_fetched == 0

        # Only one query should have been made
        assert mock_convex_client.query.call_count == 1

    @pytest.mark.asyncio
    async def test_convert_to_news_items(self, mock_convex_client, sample_unprocessed_items):
        """Test conversion of Convex docs to NewsItem objects."""
        worker = NewsProcessorWorker(convex_client=mock_convex_client)
        items = worker._convert_to_news_items(sample_unprocessed_items)

        assert len(items) == 2
        assert items[0].title == "Ethereum Completes Upgrade"
        assert items[0].source == "rss:coindesk"
        assert items[1].title == "Bitcoin ETF Approved"

    @pytest.mark.asyncio
    async def test_update_processed(self, mock_convex_client):
        """Test updating a processed item in Convex."""
        from app.services.news_fetcher.models import (
            NewsCategory,
            ProcessedNews,
            Sentiment,
            Importance,
        )

        raw_item = {
            "source": "test",
            "sourceId": "1",
        }

        processed = ProcessedNews(
            source_id="1",
            source="test",
            title="Test",
            url="https://example.com",
            published_at=datetime.utcnow(),
            category=NewsCategory.UPGRADE,
            sentiment=Sentiment.from_score(0.5, 0.8),
            summary="Test summary",
            related_tokens=[],
            related_sectors=["DeFi"],
            related_categories=["defi"],
            importance=Importance(score=0.7, factors=["test"]),
        )

        mock_convex_client.mutation.return_value = None

        worker = NewsProcessorWorker(convex_client=mock_convex_client)
        await worker._update_processed(raw_item, processed)

        mock_convex_client.mutation.assert_called_once()
        call_args = mock_convex_client.mutation.call_args
        assert call_args[0][0] == "news:updateProcessing"
        assert call_args[0][1]["category"] == "upgrade"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunNewsProcessorWorker:
    """Tests for run_news_processor_worker function."""

    @pytest.mark.asyncio
    async def test_run_news_processor_worker(self, mock_convex_client):
        """Test the convenience function."""
        mock_convex_client.query.return_value = []

        result = await run_news_processor_worker(
            convex_client=mock_convex_client,
        )

        assert result.items_fetched == 0
        assert isinstance(result, WorkerResult)

    @pytest.mark.asyncio
    async def test_run_with_custom_config(self, mock_convex_client):
        """Test with custom configuration."""
        mock_convex_client.query.return_value = []

        config = WorkerConfig(max_items_per_run=10)
        result = await run_news_processor_worker(
            convex_client=mock_convex_client,
            config=config,
        )

        # Should have queried with limit=10
        mock_convex_client.query.assert_called_with(
            "news:getUnprocessed",
            {"limit": 10},
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_fetch_error(self, mock_convex_client):
        """Test handling of fetch error."""
        mock_convex_client.query.side_effect = Exception("Database error")

        worker = NewsProcessorWorker(convex_client=mock_convex_client)
        result = await worker.run()

        assert result.items_fetched == 0
        assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_malformed_items(self, mock_convex_client):
        """Test handling of malformed items from Convex."""
        mock_convex_client.query.return_value = [
            {"title": "Missing required fields"},  # Missing sourceId, source, etc.
            {
                "sourceId": "1",
                "source": "test",
                "title": "Valid item",
                "url": "https://example.com",
                "publishedAt": int(datetime.utcnow().timestamp() * 1000),
            },
        ]
        mock_convex_client.mutation.return_value = None

        worker = NewsProcessorWorker(convex_client=mock_convex_client)
        result = await worker.run()

        # Should process the valid item
        assert result.items_processed >= 1

    @pytest.mark.asyncio
    async def test_concurrent_batches(self, mock_convex_client):
        """Test concurrent batch processing."""
        # Create many items to trigger multiple batches
        items = [
            {
                "_id": f"item{i}",
                "sourceId": str(i),
                "source": "test",
                "title": f"News {i}",
                "url": f"https://example.com/{i}",
                "publishedAt": int(datetime.utcnow().timestamp() * 1000),
            }
            for i in range(25)
        ]
        mock_convex_client.query.return_value = items
        mock_convex_client.mutation.return_value = None

        worker = NewsProcessorWorker(
            convex_client=mock_convex_client,
            config=WorkerConfig(batch_size=10),
        )
        result = await worker.run()

        assert result.items_fetched == 25
        assert result.items_updated == 25
