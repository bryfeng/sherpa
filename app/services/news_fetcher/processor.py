"""
News LLM Processor

Processes raw news items using LLM for:
- Category classification
- Sentiment analysis
- Token extraction
- Sector/category tagging
- Importance scoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Importance,
    NewsCategory,
    NewsItem,
    ProcessedNews,
    RelatedToken,
    Sentiment,
    SentimentLabel,
)

logger = logging.getLogger(__name__)

# System prompt for news classification
CLASSIFICATION_PROMPT = """You are a crypto news analyst. Analyze the following news article and extract structured information.

For each article, provide:
1. **Category**: One of: regulatory, technical, partnership, tokenomics, market, hack, upgrade, general
   - regulatory: Laws, regulations, government actions, compliance
   - technical: Protocol updates, technical developments, code changes
   - partnership: Partnerships, integrations, collaborations
   - tokenomics: Token burns, airdrops, supply changes, staking updates
   - market: Price movements, trading volume, market analysis
   - hack: Security incidents, exploits, vulnerabilities
   - upgrade: Network upgrades, hard forks, migrations
   - general: General news that doesn't fit other categories

2. **Sentiment**: A score from -1 (very negative) to 1 (very positive) and confidence (0-1)
   - Very negative: Major hacks, significant losses, regulatory bans
   - Negative: Minor issues, delays, small price drops
   - Neutral: Factual updates, routine announcements
   - Positive: Partnerships, upgrades, adoption news
   - Very positive: Major milestones, significant wins

3. **Related Tokens**: List token symbols mentioned with relevance (0-1)
   - High relevance (0.8-1.0): Token is the main subject
   - Medium relevance (0.5-0.7): Token is mentioned significantly
   - Low relevance (0.1-0.4): Token is mentioned briefly

4. **Sectors**: Relevant sectors from: DeFi, Infrastructure, Gaming, Social, AI_Data, Meme
5. **Categories**: Relevant tags from: defi, dex, lending, yield, stablecoin, l1, l2, oracle, bridge, gaming, nft, social, ai, meme, governance

6. **Importance**: Score (0-1) with factors explaining why
   - High importance (0.8-1.0): Major protocol events, hacks, regulatory changes
   - Medium importance (0.5-0.7): Significant updates, partnerships
   - Low importance (0.1-0.4): Routine updates, minor news

7. **Summary**: A concise 1-2 sentence summary focusing on the key point.

Respond in JSON format only, no markdown code blocks:
{
  "category": "string",
  "sentiment": {"score": number, "confidence": number},
  "summary": "string",
  "related_tokens": [{"symbol": "string", "relevance": number}],
  "sectors": ["string"],
  "categories": ["string"],
  "importance": {"score": number, "factors": ["string"]}
}"""


class NewsProcessor:
    """
    Processes news items using LLM for classification and sentiment.

    Uses batch processing for efficiency.
    """

    def __init__(
        self,
        llm_provider: Any = None,
        batch_size: int = 5,
        max_concurrent: int = 3,
    ):
        """
        Initialize the news processor.

        Args:
            llm_provider: LLM provider instance (with generate_response method)
            batch_size: Number of items to process in each batch
            max_concurrent: Maximum concurrent LLM calls
        """
        self._llm_provider = llm_provider
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(self, items: List[NewsItem]) -> List[ProcessedNews]:
        """
        Process a batch of news items.

        Args:
            items: List of news items to process

        Returns:
            List of processed news items
        """
        if not items:
            return []

        # Process items concurrently with rate limiting
        tasks = [self._process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[ProcessedNews] = []
        for item, result in zip(items, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing news item: {result}")
                # Create a default processed item
                processed.append(self._create_default_processed(item))
            elif result is not None:
                processed.append(result)

        return processed

    async def _process_with_semaphore(self, item: NewsItem) -> Optional[ProcessedNews]:
        """Process a single item with semaphore rate limiting."""
        async with self._semaphore:
            return await self._process_single(item)

    async def _process_single(self, item: NewsItem) -> Optional[ProcessedNews]:
        """Process a single news item."""
        try:
            # Build content for analysis
            content = self._build_analysis_content(item)

            # Get LLM classification
            classification = await self._classify_with_llm(content)

            if classification is None:
                return self._create_default_processed(item)

            # Parse classification results
            category = self._parse_category(classification.get("category", "general"))
            sentiment = self._parse_sentiment(classification.get("sentiment", {}))
            summary = classification.get("summary", item.summary or item.title)
            related_tokens = self._parse_related_tokens(classification.get("related_tokens", []))
            sectors = classification.get("sectors", [])
            categories = classification.get("categories", [])
            importance = self._parse_importance(classification.get("importance", {}))

            return ProcessedNews.from_news_item(
                item=item,
                category=category,
                sentiment=sentiment,
                summary=summary[:500],
                related_tokens=related_tokens,
                related_sectors=sectors,
                related_categories=categories,
                importance=importance,
            )

        except Exception as e:
            logger.error(f"Error processing news item {item.source_id}: {e}")
            return self._create_default_processed(item)

    def _build_analysis_content(self, item: NewsItem) -> str:
        """Build content string for LLM analysis."""
        parts = [f"Title: {item.title}"]

        if item.summary:
            parts.append(f"Summary: {item.summary}")

        if item.raw_content:
            # Truncate content to avoid token limits
            content = item.raw_content[:2000]
            parts.append(f"Content: {content}")

        parts.append(f"Source: {item.source}")

        return "\n\n".join(parts)

    async def _classify_with_llm(self, content: str) -> Optional[Dict[str, Any]]:
        """Classify content using LLM."""
        if self._llm_provider is None:
            # Fallback to rule-based classification if no LLM
            return self._rule_based_classification(content)

        try:
            from app.providers.llm.base import LLMMessage

            messages = [
                LLMMessage(role="system", content=CLASSIFICATION_PROMPT),
                LLMMessage(role="user", content=content),
            ]

            response = await self._llm_provider.generate_response(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=1000,
            )

            if not response.content:
                return None

            # Parse JSON response
            return self._parse_json_response(response.content)

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return self._rule_based_classification(content)

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the response
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from LLM response: {content[:200]}")
        return None

    def _rule_based_classification(self, content: str) -> Dict[str, Any]:
        """Fallback rule-based classification when LLM is unavailable."""
        content_lower = content.lower()

        # Detect category
        category = "general"
        if any(w in content_lower for w in ["hack", "exploit", "stolen", "breach", "vulnerability"]):
            category = "hack"
        elif any(w in content_lower for w in ["sec", "regulation", "lawsuit", "legal", "ban", "compliance"]):
            category = "regulatory"
        elif any(w in content_lower for w in ["upgrade", "fork", "migration", "v2", "v3"]):
            category = "upgrade"
        elif any(w in content_lower for w in ["partnership", "integrat", "collaborat", "launch"]):
            category = "partnership"
        elif any(w in content_lower for w in ["airdrop", "burn", "tokenomics", "supply"]):
            category = "tokenomics"
        elif any(w in content_lower for w in ["price", "market", "bull", "bear", "rally", "dump"]):
            category = "market"
        elif any(w in content_lower for w in ["update", "release", "feature", "protocol"]):
            category = "technical"

        # Detect sentiment
        sentiment_score = 0.0
        positive_words = ["bullish", "success", "growth", "launch", "partnership", "upgrade", "milestone"]
        negative_words = ["hack", "exploit", "crash", "ban", "lawsuit", "loss", "vulnerability", "scam"]

        for word in positive_words:
            if word in content_lower:
                sentiment_score += 0.2

        for word in negative_words:
            if word in content_lower:
                sentiment_score -= 0.3

        sentiment_score = max(-1, min(1, sentiment_score))

        # Extract token symbols (look for uppercase 2-6 letter words)
        token_pattern = r"\b([A-Z]{2,6})\b"
        potential_tokens = re.findall(token_pattern, content)

        # Filter common words that aren't tokens
        non_tokens = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
                      "WAS", "ONE", "OUR", "OUT", "HAS", "NEW", "SEC", "CEO", "NFT", "TVL", "ETF"}
        related_tokens = [
            {"symbol": t, "relevance": 0.5}
            for t in set(potential_tokens)
            if t not in non_tokens
        ][:5]  # Limit to 5 tokens

        # Detect sectors
        sectors = []
        if any(w in content_lower for w in ["defi", "dex", "lend", "yield", "swap"]):
            sectors.append("DeFi")
        if any(w in content_lower for w in ["ethereum", "bitcoin", "solana", "layer", "chain"]):
            sectors.append("Infrastructure")
        if any(w in content_lower for w in ["game", "play", "nft", "metaverse"]):
            sectors.append("Gaming")
        if any(w in content_lower for w in ["social", "dao", "community"]):
            sectors.append("Social")
        if any(w in content_lower for w in [" ai ", "artificial", "machine learning"]):
            sectors.append("AI_Data")

        # Calculate importance
        importance_score = 0.3  # Base importance
        factors = []

        if category == "hack":
            importance_score = 0.9
            factors.append("security_incident")
        elif category == "regulatory":
            importance_score = 0.7
            factors.append("regulatory_news")
        elif "major" in content_lower or "billion" in content_lower:
            importance_score = 0.6
            factors.append("significant_scale")

        return {
            "category": category,
            "sentiment": {"score": sentiment_score, "confidence": 0.5},
            "summary": content.split("\n")[0][:200],
            "related_tokens": related_tokens,
            "sectors": sectors or ["Infrastructure"],
            "categories": [],
            "importance": {"score": importance_score, "factors": factors},
        }

    def _parse_category(self, category_str: str) -> NewsCategory:
        """Parse category string to enum."""
        try:
            return NewsCategory(category_str.lower())
        except ValueError:
            return NewsCategory.GENERAL

    def _parse_sentiment(self, sentiment_data: Dict[str, Any]) -> Sentiment:
        """Parse sentiment data to model."""
        score = sentiment_data.get("score", 0.0)
        confidence = sentiment_data.get("confidence", 0.5)
        return Sentiment.from_score(score, confidence)

    def _parse_related_tokens(self, tokens_data: List[Dict[str, Any]]) -> List[RelatedToken]:
        """Parse related tokens data."""
        tokens: List[RelatedToken] = []

        for t in tokens_data:
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

        return tokens[:10]  # Limit to 10 tokens

    def _parse_importance(self, importance_data: Dict[str, Any]) -> Importance:
        """Parse importance data."""
        score = importance_data.get("score", 0.3)
        factors = importance_data.get("factors", [])

        return Importance(
            score=max(0, min(1, score)),
            factors=factors[:5],  # Limit factors
        )

    def _create_default_processed(self, item: NewsItem) -> ProcessedNews:
        """Create a default processed item when classification fails."""
        # Use rule-based classification as fallback
        content = self._build_analysis_content(item)
        classification = self._rule_based_classification(content)

        return ProcessedNews.from_news_item(
            item=item,
            category=self._parse_category(classification["category"]),
            sentiment=self._parse_sentiment(classification["sentiment"]),
            summary=item.summary or item.title[:200],
            related_tokens=self._parse_related_tokens(classification["related_tokens"]),
            related_sectors=classification["sectors"],
            related_categories=classification["categories"],
            importance=self._parse_importance(classification["importance"]),
        )


async def process_news_items(
    items: List[NewsItem],
    llm_provider: Any = None,
    batch_size: int = 5,
) -> List[ProcessedNews]:
    """
    Process news items for classification and sentiment.

    Args:
        items: News items to process
        llm_provider: Optional LLM provider
        batch_size: Items per batch

    Returns:
        Processed news items
    """
    processor = NewsProcessor(
        llm_provider=llm_provider,
        batch_size=batch_size,
    )

    all_processed: List[ProcessedNews] = []

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        processed = await processor.process_batch(batch)
        all_processed.extend(processed)

    return all_processed
