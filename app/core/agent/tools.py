"""
Tool Registry and Executor for LLM-driven tool calling.

This module provides the infrastructure for defining, registering, and executing
tools that the LLM can call based on semantic understanding of user intent.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ...providers.llm.base import ToolDefinition, ToolParameter, ToolParameterType, ToolCall, ToolResult


@dataclass
class RegisteredTool:
    """A tool registered in the registry with its definition and handler."""
    definition: ToolDefinition
    handler: Callable[..., Coroutine[Any, Any, Any]]
    requires_address: bool = False


class ToolRegistry:
    """
    Registry of available tools that the LLM can call.

    Each tool has a definition (name, description, parameters) and a handler function.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._tools: Dict[str, RegisteredTool] = {}
        self.logger = logger or logging.getLogger(__name__)
        self._register_default_tools()

    def register(
        self,
        name: str,
        definition: ToolDefinition,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        requires_address: bool = False,
    ) -> None:
        """Register a tool with its definition and handler."""
        self._tools[name] = RegisteredTool(
            definition=definition,
            handler=handler,
            requires_address=requires_address,
        )

    def get_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions for passing to the LLM."""
        return [tool.definition for tool in self._tools.values()]

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""

        # Portfolio tool
        self.register(
            "get_portfolio",
            ToolDefinition(
                name="get_portfolio",
                description=(
                    "Fetch the current cryptocurrency portfolio for a wallet address. "
                    "Returns token holdings, balances, and total value in USD. "
                    "Use this when the user asks about their portfolio, holdings, balance, "
                    "tokens, wallet contents, asset allocation, or wants portfolio analysis."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to fetch portfolio for (e.g., 0x...)",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description="The blockchain to query (e.g., 'ethereum', 'solana', 'base')",
                        required=False,
                        default="ethereum",
                    ),
                ],
            ),
            self._handle_get_portfolio,
            requires_address=True,
        )

        # Token chart tool
        self.register(
            "get_token_chart",
            ToolDefinition(
                name="get_token_chart",
                description=(
                    "Fetch price chart data for a cryptocurrency token. "
                    "Returns historical price data, candlestick charts, and price statistics. "
                    "Use this when the user asks about price history, charts, price trends, "
                    "price performance, or wants to see how a token has been doing."
                ),
                parameters=[
                    ToolParameter(
                        name="token_identifier",
                        type=ToolParameterType.STRING,
                        description=(
                            "The token to look up. Can be a symbol (BTC, ETH), "
                            "name (Bitcoin, Ethereum), or contract address (0x...)"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="range",
                        type=ToolParameterType.STRING,
                        description="Time range for the chart",
                        required=False,
                        default="7d",
                        enum=["1d", "7d", "30d", "90d", "180d", "365d", "max"],
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description="Blockchain for contract address lookups",
                        required=False,
                    ),
                ],
            ),
            self._handle_get_token_chart,
        )

        # Trending tokens tool
        self.register(
            "get_trending_tokens",
            ToolDefinition(
                name="get_trending_tokens",
                description=(
                    "Fetch currently trending cryptocurrency tokens with price changes "
                    "and market data. Use this when the user asks about trending tokens, "
                    "top gainers, top losers, hot coins, or what's popular in crypto."
                ),
                parameters=[
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Maximum number of trending tokens to return",
                        required=False,
                        default=10,
                    ),
                    ToolParameter(
                        name="focus_token",
                        type=ToolParameterType.STRING,
                        description=(
                            "Optional specific token symbol to highlight in the results "
                            "(e.g., 'PEPE' if user asks about a specific trending token)"
                        ),
                        required=False,
                    ),
                ],
            ),
            self._handle_get_trending_tokens,
        )

        # Wallet history tool
        self.register(
            "get_wallet_history",
            ToolDefinition(
                name="get_wallet_history",
                description=(
                    "Fetch transaction history and activity summary for a wallet. "
                    "Returns recent transactions, transfers, swaps, and activity analysis. "
                    "Use this when the user asks about their transaction history, "
                    "recent activity, transfers, trades, or wallet movements."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to fetch history for",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description="The blockchain to query",
                        required=False,
                        default="ethereum",
                    ),
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Maximum number of transactions to return",
                        required=False,
                        default=50,
                    ),
                    ToolParameter(
                        name="days",
                        type=ToolParameterType.INTEGER,
                        description="Number of days of history to fetch (alternative to limit)",
                        required=False,
                    ),
                ],
            ),
            self._handle_get_wallet_history,
            requires_address=True,
        )

        # TVL data tool
        self.register(
            "get_tvl_data",
            ToolDefinition(
                name="get_tvl_data",
                description=(
                    "Fetch Total Value Locked (TVL) data for a DeFi protocol. "
                    "Returns TVL time series and statistics. Use this when the user "
                    "asks about TVL, total value locked, or protocol liquidity for "
                    "protocols like Uniswap, Aave, Compound, etc."
                ),
                parameters=[
                    ToolParameter(
                        name="protocol",
                        type=ToolParameterType.STRING,
                        description="The DeFi protocol name (e.g., 'uniswap', 'aave', 'compound')",
                        required=True,
                    ),
                    ToolParameter(
                        name="window",
                        type=ToolParameterType.STRING,
                        description="Time window for TVL data",
                        required=False,
                        default="7d",
                        enum=["7d", "30d"],
                    ),
                ],
            ),
            self._handle_get_tvl_data,
        )

        # News tools
        self.register(
            "get_news",
            ToolDefinition(
                name="get_news",
                description=(
                    "Fetch recent cryptocurrency news articles. "
                    "Returns news items with titles, summaries, sentiment, and related tokens. "
                    "Use this when the user asks about crypto news, recent updates, "
                    "what's happening in crypto, or news about specific topics."
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
                        default=10,
                    ),
                    ToolParameter(
                        name="hours_back",
                        type=ToolParameterType.INTEGER,
                        description="How many hours back to look for news",
                        required=False,
                        default=24,
                    ),
                ],
            ),
            self._handle_get_news,
        )

        self.register(
            "get_personalized_news",
            ToolDefinition(
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
            ),
            self._handle_get_personalized_news,
            requires_address=True,
        )

        self.register(
            "get_token_news",
            ToolDefinition(
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
            ),
            self._handle_get_token_news,
        )

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _handle_get_portfolio(
        self,
        wallet_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """Handle portfolio fetch."""
        from ...tools.portfolio import get_portfolio

        try:
            result = await get_portfolio(wallet_address, chain)
            if result.data:
                return {
                    "success": True,
                    "data": result.data.model_dump(mode='json'),
                    "sources": [s.model_dump(mode='json') for s in result.sources],
                    "warnings": result.warnings or [],
                }
            return {
                "success": False,
                "error": "Failed to fetch portfolio data",
                "warnings": result.warnings or [],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_token_chart(
        self,
        token_identifier: str,
        range: str = "7d",
        chain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle token chart fetch."""
        from ...services.token_chart import get_token_chart as fetch_token_chart
        import re

        try:
            # Determine if identifier is a contract address
            contract_address = None
            symbol = None
            coin_id = None

            if re.match(r'^0x[a-fA-F0-9]{40}$', token_identifier):
                contract_address = token_identifier.lower()
            else:
                # Check if it's a known slug or treat as symbol
                known_slugs = {
                    'bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin',
                    'litecoin', 'polkadot', 'tron', 'avalanche', 'chainlink',
                    'morpho', 'uniswap', 'aave', 'compound', 'maker',
                    'curve-dao-token', 'lido-dao', 'rocket-pool', 'frax',
                    'convex-finance', 'yearn-finance', 'sushi', 'balancer',
                    'pendle', '1inch', 'gmx', 'radiant-capital', 'arbitrum',
                    'optimism', 'polygon', 'base', 'mantle', 'celestia',
                }
                if token_identifier.lower() in known_slugs:
                    coin_id = token_identifier.lower()
                else:
                    symbol = token_identifier.upper()

            chart_data = await fetch_token_chart(
                coin_id=coin_id,
                symbol=symbol,
                contract_address=contract_address,
                chain=chain,
                range_key=range,
                vs_currency="usd",
                include_candles=True,
            )

            return {"success": True, **chart_data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_trending_tokens(
        self,
        limit: int = 10,
        focus_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle trending tokens fetch."""
        from ...services.trending import get_trending_tokens

        try:
            fetch_limit = max(limit, 25) if focus_token else limit
            raw_tokens = list(await get_trending_tokens(limit=fetch_limit))
            fetched_at = datetime.now(timezone.utc).isoformat()

            # Find focus token if specified
            focus = None
            if focus_token:
                focus_lower = focus_token.lower()
                for token in raw_tokens:
                    symbol = (token.get('symbol') or '').lower()
                    name = (token.get('name') or '').lower()
                    if focus_lower in (symbol, name):
                        focus = token
                        break

            # Limit results
            tokens = raw_tokens[:limit]

            # Ensure focus token is in the list if found
            if focus:
                focus_identity = (focus.get('symbol') or '').lower()
                in_list = any(
                    (t.get('symbol') or '').lower() == focus_identity
                    for t in tokens
                )
                if not in_list:
                    tokens.append(focus)

            return {
                "success": True,
                "tokens": tokens,
                "fetched_at": fetched_at,
                "focus": focus,
                "total_available": len(raw_tokens),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_wallet_history(
        self,
        wallet_address: str,
        chain: str = "ethereum",
        limit: Optional[int] = 50,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Handle wallet history fetch."""
        from ...services.activity_summary import get_history_snapshot
        from datetime import timedelta

        try:
            if days:
                now = datetime.now(timezone.utc)
                start = now - timedelta(days=days)
                snapshot, events = await get_history_snapshot(
                    address=wallet_address,
                    chain=chain,
                    start=start,
                    end=now,
                )
            else:
                snapshot, events = await get_history_snapshot(
                    address=wallet_address,
                    chain=chain,
                    limit=limit,
                )

            return {
                "success": True,
                "snapshot": snapshot,
                "events": events,
                "limit": limit if not days else None,
                "days": days,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_tvl_data(
        self,
        protocol: str,
        window: str = "7d",
    ) -> Dict[str, Any]:
        """Handle TVL data fetch."""
        from ...tools.defillama import get_tvl_series

        try:
            timestamps, tvl_values = await get_tvl_series(
                protocol=protocol.lower(),
                window=window,
            )

            if not timestamps or not tvl_values:
                return {
                    "success": False,
                    "error": f"No TVL data found for protocol: {protocol}",
                }

            # Calculate stats
            stats = {}
            if tvl_values:
                stats = {
                    "start_value": tvl_values[0],
                    "end_value": tvl_values[-1],
                    "min_value": min(tvl_values),
                    "max_value": max(tvl_values),
                    "change_absolute": tvl_values[-1] - tvl_values[0],
                    "change_percent": (
                        ((tvl_values[-1] - tvl_values[0]) / tvl_values[0] * 100)
                        if tvl_values[0] > 0 else 0
                    ),
                }

            return {
                "success": True,
                "protocol": protocol,
                "window": window,
                "timestamps": timestamps,
                "tvl": tvl_values,
                "stats": stats,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_news(
        self,
        category: Optional[str] = None,
        limit: int = 10,
        hours_back: int = 24,
    ) -> Dict[str, Any]:
        """Handle recent news fetch."""
        from ...services.news_fetcher.service import NewsFetcherService

        try:
            # Initialize service (will use Convex client from context if available)
            service = NewsFetcherService()

            news_items = await service.get_recent_news(
                category=category,
                limit=limit,
                since_hours=hours_back,
            )

            # Format news items for response
            formatted_items = []
            for item in news_items:
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
                    "importance": item.get("importance", {}).get("score", 0.5),
                })

            return {
                "success": True,
                "news": formatted_items,
                "count": len(formatted_items),
                "category_filter": category,
                "hours_back": hours_back,
            }

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_personalized_news(
        self,
        wallet_address: str,
        limit: int = 10,
        min_relevance: float = 0.2,
    ) -> Dict[str, Any]:
        """Handle personalized news fetch."""
        from ...services.relevance import RelevanceService
        from ...tools.portfolio import get_portfolio

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

            # Get personalized news
            service = RelevanceService()
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
            self.logger.error(f"Error fetching personalized news: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_token_news(
        self,
        symbols: List[str],
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Handle token-specific news fetch."""
        from ...services.news_fetcher.service import NewsFetcherService

        try:
            # Normalize symbols to uppercase
            normalized_symbols = [s.upper() for s in symbols]

            service = NewsFetcherService()
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
            self.logger.error(f"Error fetching token news: {e}")
            return {"success": False, "error": str(e)}


class ToolExecutor:
    """
    Executes tool calls requested by the LLM.

    Supports parallel execution of independent tool calls.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        logger: Optional[logging.Logger] = None,
    ):
        self.registry = registry
        self.logger = logger or logging.getLogger(__name__)

    async def execute_single(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return the result."""
        tool = self.registry.get_tool(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=f"Unknown tool: {tool_call.name}",
            )

        try:
            result = await tool.handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                result=result,
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Tool execution error for {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=str(e),
            )

    async def execute_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls in parallel."""
        if not tool_calls:
            return []

        tasks = [self.execute_single(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to ToolResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_call_id=tool_calls[i].id,
                    result=None,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results
