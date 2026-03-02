"""News tool handlers: get_news, get_personalized_news, get_token_news."""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


@tool_spec(
    name="get_news",
    description=(
        "Fetch recent cryptocurrency news articles and headlines. "
        "Returns a balanced mix of: news articles (CoinDesk, Cointelegraph, The Block), "
        "trending tokens (CoinGecko), and DeFi updates (DefiLlama TVL changes). "
        "Use this when the user asks about: news, top stories, headlines, "
        "what's happening in crypto, recent updates, breaking news, or stories."
    ),
    parameters=[
        ToolParameter(
            name="category",
            type=ToolParameterType.STRING,
            description="Filter by news category",
            required=False,
            enum=["regulatory", "technical", "partnership", "tokenomics", "market", "hack", "upgrade", "general"],
        ),
        ToolParameter(
            name="limit",
            type=ToolParameterType.INTEGER,
            description="Maximum number of news items to return",
            required=False,
            default=15,
        ),
        ToolParameter(
            name="hours_back",
            type=ToolParameterType.INTEGER,
            description="How many hours back to look for news",
            required=False,
            default=24,
        ),
    ],
)
async def handle_get_news(
    category: Optional[str] = None,
    limit: int = 15,
    hours_back: int = 24,
) -> Dict[str, Any]:
    """Handle recent news fetch with source diversity."""
    from ....db import get_convex_client
    from ....services.news_fetcher.service import NewsFetcherService

    # Friendly source name mapping
    SOURCE_LABELS = {
        "rss:coindesk": "CoinDesk",
        "rss:cointelegraph": "Cointelegraph",
        "rss:theblock": "The Block",
        "rss:decrypt": "Decrypt",
        "rss:bitcoinmagazine": "Bitcoin Magazine",
        "coingecko:trending": "CoinGecko Trending",
        "defillama:tvl": "DeFiLlama TVL",
        "defillama:hacks": "DeFiLlama Security",
    }

    def get_source_type(source: str) -> str:
        """Categorize source into type for display grouping."""
        if source.startswith("rss:"):
            return "news_article"
        elif source.startswith("coingecko:"):
            return "trending"
        elif source.startswith("defillama:"):
            return "defi_update"
        return "other"

    try:
        # Initialize service with Convex client
        convex = get_convex_client()
        service = NewsFetcherService(convex_client=convex)

        # Use diversified query for balanced source representation
        news_items = await service.get_recent_news(
            category=category,
            limit=limit,
            since_hours=hours_back,
            diversified=True,
        )

        # Format news items for response with enhanced metadata
        formatted_items = []
        for item in news_items:
            raw_source = item.get("source", "")
            formatted_items.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": SOURCE_LABELS.get(raw_source, raw_source),
                "source_type": get_source_type(raw_source),
                "published_at": item.get("publishedAt"),
                "category": item.get("category", "general"),
                "sentiment": item.get("sentiment", {}),
                "related_tokens": [
                    t.get("symbol") for t in item.get("relatedTokens", [])
                ],
                "importance": item.get("importance", {}).get("score", 0.5),
            })

        # Count sources for transparency
        source_counts: Dict[str, int] = {}
        for item in formatted_items:
            st = item["source_type"]
            source_counts[st] = source_counts.get(st, 0) + 1

        return {
            "success": True,
            "news": formatted_items,
            "count": len(formatted_items),
            "source_breakdown": source_counts,
            "category_filter": category,
            "hours_back": hours_back,
        }

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="get_personalized_news",
    description=(
        "Fetch news personalized to the user's portfolio holdings. "
        "Returns news ranked by relevance to tokens they hold, "
        "with explanations of why each article is relevant. "
        "Use this when the user asks for news relevant to their portfolio, "
        "news about their holdings, or personalized crypto updates."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The wallet address to personalize news for",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type=ToolParameterType.INTEGER,
            description="Maximum number of news items to return",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="min_relevance",
            type=ToolParameterType.NUMBER,
            description="Minimum relevance score (0-1) to include",
            required=False,
            default=0.2,
        ),
    ],
    requires_address=True,
)
async def handle_get_personalized_news(
    wallet_address: str,
    limit: int = 10,
    min_relevance: float = 0.2,
) -> Dict[str, Any]:
    """Handle personalized news fetch."""
    from ....db import get_convex_client
    from ....services.relevance import RelevanceService
    from ....tools.portfolio import get_portfolio

    try:
        # First get portfolio holdings
        portfolio_result = await get_portfolio(wallet_address, "ethereum")

        if not portfolio_result.data:
            return {
                "success": False,
                "error": "Could not fetch portfolio for personalization",
            }

        # Convert portfolio to holdings format
        holdings = []
        for token in portfolio_result.data.tokens:
            holdings.append({
                "symbol": token.symbol,
                "address": token.address,
                "chainId": 1,  # Ethereum
                "valueUsd": float(token.balance_usd) if token.balance_usd else 0,
            })

        # Get personalized news with Convex client
        convex = get_convex_client()
        service = RelevanceService(convex_client=convex)
        news_items = await service.get_personalized_news(
            wallet_address=wallet_address,
            holdings=holdings,
            limit=limit,
            min_relevance=min_relevance,
        )

        # Format results
        formatted_items = []
        for item in news_items:
            relevance = item.get("relevance", {})
            formatted_items.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("publishedAt"),
                "category": item.get("category", "general"),
                "sentiment": item.get("sentiment", {}),
                "related_tokens": [
                    t.get("symbol") for t in item.get("relatedTokens", [])
                ],
                "relevance_score": relevance.get("score", 0),
                "relevance_level": relevance.get("level", "low"),
                "relevance_explanation": relevance.get("explanation", ""),
            })

        return {
            "success": True,
            "news": formatted_items,
            "count": len(formatted_items),
            "portfolio_tokens": len(holdings),
            "min_relevance": min_relevance,
        }

    except Exception as e:
        logger.error(f"Error fetching personalized news: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="get_token_news",
    description=(
        "Fetch news about specific cryptocurrency tokens. "
        "Returns news articles that mention the specified tokens. "
        "Use this when the user asks about news for a specific token "
        "like 'What's the latest news about ETH?' or 'Any Solana news?'"
    ),
    parameters=[
        ToolParameter(
            name="symbols",
            type=ToolParameterType.ARRAY,
            description="List of token symbols to search for (e.g., ['ETH', 'BTC'])",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type=ToolParameterType.INTEGER,
            description="Maximum number of news items to return",
            required=False,
            default=10,
        ),
    ],
)
async def handle_get_token_news(
    symbols: List[str],
    limit: int = 10,
) -> Dict[str, Any]:
    """Handle token-specific news fetch."""
    from ....db import get_convex_client
    from ....services.news_fetcher.service import NewsFetcherService

    try:
        # Normalize symbols to uppercase
        normalized_symbols = [s.upper() for s in symbols]

        convex = get_convex_client()
        service = NewsFetcherService(convex_client=convex)
        news_items = await service.get_token_news(
            symbols=normalized_symbols,
            limit=limit,
        )

        # Format news items
        formatted_items = []
        for item in news_items:
            related_tokens = item.get("relatedTokens", [])
            # Highlight which queried tokens are mentioned
            matched_tokens = [
                t.get("symbol") for t in related_tokens
                if t.get("symbol", "").upper() in normalized_symbols
            ]

            formatted_items.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("publishedAt"),
                "category": item.get("category", "general"),
                "sentiment": item.get("sentiment", {}),
                "matched_tokens": matched_tokens,
                "all_related_tokens": [t.get("symbol") for t in related_tokens],
            })

        return {
            "success": True,
            "news": formatted_items,
            "count": len(formatted_items),
            "searched_symbols": normalized_symbols,
        }

    except Exception as e:
        logger.error(f"Error fetching token news: {e}")
        return {"success": False, "error": str(e)}
