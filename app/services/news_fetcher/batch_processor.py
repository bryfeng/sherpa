"""
Batch News LLM Processor

Cost-efficient batch processing of news items using a single LLM call
for multiple items. Designed to minimize API costs while maintaining quality.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Importance,
    NewsCategory,
    NewsItem,
    ProcessedNews,
    RelatedToken,
    Sentiment,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    batch_size: int = 10  # Items per LLM call
    max_concurrent_batches: int = 2  # Concurrent LLM calls
    max_tokens_per_batch: int = 4000  # Max tokens for batch response
    max_daily_tokens: int = 100_000  # Daily token budget
    temperature: float = 0.1  # Low for consistent classification
    retry_on_parse_error: bool = True
    fallback_to_rules: bool = True


@dataclass
class ProcessingStats:
    """Statistics from batch processing."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    fallback_used: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


# Batch classification prompt - processes multiple items at once
BATCH_CLASSIFICATION_PROMPT = """You are a crypto news analyst. Analyze the following news articles and provide structured classification for EACH one.

For each article, extract:
1. **category**: One of: regulatory, technical, partnership, tokenomics, market, hack, upgrade, general
2. **sentiment**: score (-1 to 1) and confidence (0-1)
3. **summary**: 1-2 sentence summary
4. **tokens**: Related token symbols with relevance (0-1)
5. **sectors**: From: DeFi, Infrastructure, Gaming, Social, AI_Data, Meme
6. **categories**: Tags from: defi, dex, lending, yield, stablecoin, l1, l2, oracle, bridge, gaming, nft, social, ai, meme, governance
7. **importance**: score (0-1) and factors

Respond with a JSON array containing one object per article, in the same order as the input.
Each object should have this structure:
{
  "index": number,
  "category": "string",
  "sentiment": {"score": number, "confidence": number},
  "summary": "string",
  "tokens": [{"symbol": "string", "relevance": number}],
  "sectors": ["string"],
  "categories": ["string"],
  "importance": {"score": number, "factors": ["string"]}
}

IMPORTANT: Return ONLY the JSON array, no markdown code blocks or extra text."""


class BatchNewsProcessor:
    """
    Cost-efficient batch processor for news items.

    Processes multiple news items in a single LLM call to minimize API costs.
    Falls back to rule-based classification when LLM is unavailable.
    """

    def __init__(
        self,
        llm_provider: Any = None,
        config: Optional[BatchProcessingConfig] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            llm_provider: LLM provider with generate_response method
            config: Processing configuration
        """
        self._llm = llm_provider
        self._config = config or BatchProcessingConfig()
        self._tokens_used_today = 0
        self._last_reset_date: Optional[datetime] = None

    async def process_items(
        self,
        items: List[NewsItem],
    ) -> Tuple[List[ProcessedNews], ProcessingStats]:
        """
        Process news items in batches.

        Args:
            items: News items to process

        Returns:
            Tuple of (processed items, statistics)
        """
        start_time = datetime.utcnow()
        stats = ProcessingStats(total_items=len(items))

        if not items:
            return [], stats

        # Reset daily token counter if needed
        self._check_daily_reset()

        # Split into batches
        batches = [
            items[i:i + self._config.batch_size]
            for i in range(0, len(items), self._config.batch_size)
        ]

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self._config.max_concurrent_batches)
        all_results: List[ProcessedNews] = []

        async def process_batch_with_semaphore(batch: List[NewsItem]) -> List[ProcessedNews]:
            async with semaphore:
                return await self._process_batch(batch, stats)

        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                stats.errors.append(str(result))
                logger.error(f"Batch processing error: {result}")
            elif result:
                all_results.extend(result)

        stats.processed_items = len(all_results)
        stats.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return all_results, stats

    async def _process_batch(
        self,
        items: List[NewsItem],
        stats: ProcessingStats,
    ) -> List[ProcessedNews]:
        """Process a single batch of items."""
        # Check token budget
        if self._tokens_used_today >= self._config.max_daily_tokens:
            logger.warning("Daily token budget exceeded, using rule-based fallback")
            return self._process_batch_with_rules(items, stats)

        if self._llm is None:
            return self._process_batch_with_rules(items, stats)

        try:
            # Build batch prompt
            batch_content = self._build_batch_content(items)

            # Make LLM call
            classifications, tokens_used, error = await self._classify_batch_with_llm(batch_content, len(items))
            stats.llm_calls += 1
            stats.tokens_used += tokens_used

            if error:
                stats.errors.append(error)

            if classifications is None:
                return self._process_batch_with_rules(items, stats)

            # Parse results
            return self._parse_batch_results(items, classifications, stats)

        except Exception as e:
            logger.error(f"Batch LLM error: {e}")
            stats.errors.append(str(e))

            if self._config.fallback_to_rules:
                return self._process_batch_with_rules(items, stats)
            return []

    def _build_batch_content(self, items: List[NewsItem]) -> str:
        """Build content string for batch classification."""
        parts = []

        for i, item in enumerate(items):
            item_parts = [f"[Article {i}]", f"Title: {item.title}"]

            if item.summary:
                item_parts.append(f"Summary: {item.summary[:300]}")
            elif item.raw_content:
                item_parts.append(f"Content: {item.raw_content[:300]}")

            item_parts.append(f"Source: {item.source}")
            parts.append("\n".join(item_parts))

        return "\n\n---\n\n".join(parts)

    async def _classify_batch_with_llm(
        self,
        content: str,
        item_count: int,
    ) -> Tuple[Optional[List[Dict[str, Any]]], int, Optional[str]]:
        """Classify batch using LLM.

        Returns:
            Tuple of (classifications, tokens_used, error_message)
        """
        try:
            from app.providers.llm.base import LLMMessage

            messages = [
                LLMMessage(role="system", content=BATCH_CLASSIFICATION_PROMPT),
                LLMMessage(
                    role="user",
                    content=f"Analyze these {item_count} articles:\n\n{content}",
                ),
            ]

            response = await self._llm.generate_response(
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens_per_batch,
            )

            tokens_used = response.tokens_used or 0

            # Track daily token usage
            self._tokens_used_today += tokens_used

            if not response.content:
                return None, tokens_used, "Empty LLM response"

            return self._parse_json_array(response.content), tokens_used, None

        except Exception as e:
            error_msg = f"LLM batch classification error: {e}"
            logger.error(error_msg)
            return None, 0, error_msg

    def _parse_json_array(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse JSON array from LLM response."""
        # Try direct JSON parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from response
        array_match = re.search(r"\[[\s\S]*\]", content)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to extract from markdown code block
        code_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", content)
        if code_match:
            try:
                result = json.loads(code_match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON array from response: {content[:300]}")
        return None

    def _parse_batch_results(
        self,
        items: List[NewsItem],
        classifications: List[Dict[str, Any]],
        stats: ProcessingStats,
    ) -> List[ProcessedNews]:
        """Parse LLM results into ProcessedNews objects."""
        results: List[ProcessedNews] = []

        # Index classifications by their index field
        indexed = {c.get("index", i): c for i, c in enumerate(classifications)}

        for i, item in enumerate(items):
            classification = indexed.get(i)

            if classification is None:
                # Fallback for missing classification
                logger.warning(f"Missing classification for item {i}, using rules")
                results.append(self._process_item_with_rules(item))
                stats.fallback_used += 1
                continue

            try:
                processed = ProcessedNews.from_news_item(
                    item=item,
                    category=self._parse_category(classification.get("category", "general")),
                    sentiment=self._parse_sentiment(classification.get("sentiment", {})),
                    summary=classification.get("summary", item.title)[:500],
                    related_tokens=self._parse_tokens(classification.get("tokens", [])),
                    related_sectors=classification.get("sectors", []),
                    related_categories=classification.get("categories", []),
                    importance=self._parse_importance(classification.get("importance", {})),
                )
                results.append(processed)
            except Exception as e:
                logger.warning(f"Error parsing classification for item {i}: {e}")
                results.append(self._process_item_with_rules(item))
                stats.fallback_used += 1

        return results

    def _process_batch_with_rules(
        self,
        items: List[NewsItem],
        stats: ProcessingStats,
    ) -> List[ProcessedNews]:
        """Process batch using rule-based classification."""
        results = []
        for item in items:
            results.append(self._process_item_with_rules(item))
            stats.fallback_used += 1
        return results

    def _process_item_with_rules(self, item: NewsItem) -> ProcessedNews:
        """Process single item with rule-based classification."""
        content = f"{item.title} {item.summary or ''} {item.raw_content or ''}"
        content_lower = content.lower()

        # Detect category
        category = NewsCategory.GENERAL
        if any(w in content_lower for w in ["hack", "exploit", "stolen", "breach"]):
            category = NewsCategory.HACK
        elif any(w in content_lower for w in ["sec", "regulation", "lawsuit", "legal", "ban"]):
            category = NewsCategory.REGULATORY
        elif any(w in content_lower for w in ["upgrade", "fork", "migration"]):
            category = NewsCategory.UPGRADE
        elif any(w in content_lower for w in ["partnership", "integrat", "collaborat"]):
            category = NewsCategory.PARTNERSHIP
        elif any(w in content_lower for w in ["airdrop", "burn", "tokenomics"]):
            category = NewsCategory.TOKENOMICS
        elif any(w in content_lower for w in ["price", "market", "bull", "bear"]):
            category = NewsCategory.MARKET
        elif any(w in content_lower for w in ["update", "release", "feature"]):
            category = NewsCategory.TECHNICAL

        # Detect sentiment
        sentiment_score = 0.0
        positive = ["bullish", "success", "growth", "launch", "milestone"]
        negative = ["hack", "exploit", "crash", "ban", "loss", "scam"]

        for word in positive:
            if word in content_lower:
                sentiment_score += 0.2
        for word in negative:
            if word in content_lower:
                sentiment_score -= 0.3

        sentiment_score = max(-1, min(1, sentiment_score))

        # Extract tokens
        token_pattern = r"\b([A-Z]{2,6})\b"
        non_tokens = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                      "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "NEW",
                      "SEC", "CEO", "NFT", "TVL", "ETF", "API", "USD"}
        potential = set(re.findall(token_pattern, content))
        tokens = [
            RelatedToken(symbol=t, relevance_score=0.5)
            for t in potential if t not in non_tokens
        ][:5]

        # Detect sectors
        sectors = []
        if any(w in content_lower for w in ["defi", "dex", "lend", "yield"]):
            sectors.append("DeFi")
        if any(w in content_lower for w in ["ethereum", "bitcoin", "solana", "layer"]):
            sectors.append("Infrastructure")
        if any(w in content_lower for w in ["game", "nft", "metaverse"]):
            sectors.append("Gaming")

        # Calculate importance
        importance_score = 0.3
        factors = []
        if category == NewsCategory.HACK:
            importance_score = 0.9
            factors.append("security_incident")
        elif category == NewsCategory.REGULATORY:
            importance_score = 0.7
            factors.append("regulatory_news")

        return ProcessedNews.from_news_item(
            item=item,
            category=category,
            sentiment=Sentiment.from_score(sentiment_score, 0.5),
            summary=item.summary or item.title[:200],
            related_tokens=tokens,
            related_sectors=sectors or ["Infrastructure"],
            related_categories=[],
            importance=Importance(score=importance_score, factors=factors),
        )

    def _parse_category(self, category_str: str) -> NewsCategory:
        """Parse category string to enum."""
        try:
            return NewsCategory(category_str.lower())
        except ValueError:
            return NewsCategory.GENERAL

    def _parse_sentiment(self, data: Dict[str, Any]) -> Sentiment:
        """Parse sentiment data."""
        score = data.get("score", 0.0)
        confidence = data.get("confidence", 0.5)
        return Sentiment.from_score(score, confidence)

    def _parse_tokens(self, tokens_data: List[Dict[str, Any]]) -> List[RelatedToken]:
        """Parse related tokens."""
        tokens = []
        for t in tokens_data[:10]:
            try:
                symbol = t.get("symbol", "").upper()
                relevance = t.get("relevance", 0.5)
                if symbol and len(symbol) <= 10:
                    tokens.append(RelatedToken(
                        symbol=symbol,
                        relevance_score=max(0, min(1, relevance)),
                    ))
            except Exception:
                continue
        return tokens

    def _parse_importance(self, data: Dict[str, Any]) -> Importance:
        """Parse importance data."""
        score = data.get("score", 0.3)
        factors = data.get("factors", [])
        return Importance(score=max(0, min(1, score)), factors=factors[:5])

    def _check_daily_reset(self):
        """Reset daily token counter if new day."""
        today = datetime.utcnow().date()
        if self._last_reset_date != today:
            self._tokens_used_today = 0
            self._last_reset_date = today

    @property
    def tokens_remaining_today(self) -> int:
        """Get remaining token budget for today."""
        self._check_daily_reset()
        return max(0, self._config.max_daily_tokens - self._tokens_used_today)


async def process_news_batch(
    items: List[NewsItem],
    llm_provider: Any = None,
    config: Optional[BatchProcessingConfig] = None,
) -> Tuple[List[ProcessedNews], ProcessingStats]:
    """
    Convenience function for batch processing news items.

    Args:
        items: News items to process
        llm_provider: LLM provider instance
        config: Processing configuration

    Returns:
        Tuple of (processed items, statistics)
    """
    processor = BatchNewsProcessor(
        llm_provider=llm_provider,
        config=config,
    )
    return await processor.process_items(items)
