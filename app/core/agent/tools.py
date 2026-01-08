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

        # =====================================================================
        # Policy Tools
        # =====================================================================

        self.register(
            "get_risk_policy",
            ToolDefinition(
                name="get_risk_policy",
                description=(
                    "Get the current risk policy settings for a wallet address. "
                    "Returns risk limits like max position size, max slippage, daily limits, etc. "
                    "Use this when the user asks about their risk settings, trading limits, "
                    "position limits, or wants to know their current risk configuration."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to get risk policy for",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_risk_policy,
            requires_address=True,
        )

        self.register(
            "update_risk_policy",
            ToolDefinition(
                name="update_risk_policy",
                description=(
                    "Update risk policy settings for a wallet address. "
                    "Allows setting limits like max position size, max slippage, daily volume limits, etc. "
                    "Use this when the user wants to change their risk settings, set trading limits, "
                    "adjust position limits, or configure their risk preferences."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to update risk policy for",
                        required=True,
                    ),
                    ToolParameter(
                        name="max_position_percent",
                        type=ToolParameterType.NUMBER,
                        description="Maximum percentage of portfolio in a single asset (e.g., 25.0 for 25%)",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_position_value_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum USD value in a single position",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_daily_volume_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum daily trading volume in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_daily_loss_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum daily realized loss in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_single_tx_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum single transaction value in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="require_approval_above_usd",
                        type=ToolParameterType.NUMBER,
                        description="Require manual approval for transactions above this USD amount",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_slippage_percent",
                        type=ToolParameterType.NUMBER,
                        description="Maximum allowed slippage percentage (e.g., 3.0 for 3%)",
                        required=False,
                    ),
                    ToolParameter(
                        name="enabled",
                        type=ToolParameterType.BOOLEAN,
                        description="Enable or disable the risk policy",
                        required=False,
                    ),
                ],
            ),
            self._handle_update_risk_policy,
            requires_address=True,
        )

        self.register(
            "check_action_allowed",
            ToolDefinition(
                name="check_action_allowed",
                description=(
                    "Check if a proposed trading action is allowed by policy rules. "
                    "Evaluates the action against session, risk, and system policies. "
                    "Returns whether the action is approved, any violations or warnings, "
                    "and the risk score. Use this before executing trades to verify compliance."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address performing the action",
                        required=True,
                    ),
                    ToolParameter(
                        name="action_type",
                        type=ToolParameterType.STRING,
                        description="Type of action (swap, bridge, transfer, approve)",
                        required=True,
                        enum=["swap", "bridge", "transfer", "approve"],
                    ),
                    ToolParameter(
                        name="value_usd",
                        type=ToolParameterType.NUMBER,
                        description="Value of the transaction in USD",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain_id",
                        type=ToolParameterType.INTEGER,
                        description="Chain ID for the transaction (1=Ethereum, 137=Polygon, etc.)",
                        required=False,
                        default=1,
                    ),
                    ToolParameter(
                        name="token_in",
                        type=ToolParameterType.STRING,
                        description="Input token symbol or address (for swaps)",
                        required=False,
                    ),
                    ToolParameter(
                        name="token_out",
                        type=ToolParameterType.STRING,
                        description="Output token symbol or address (for swaps)",
                        required=False,
                    ),
                    ToolParameter(
                        name="slippage_percent",
                        type=ToolParameterType.NUMBER,
                        description="Slippage tolerance percentage (for swaps)",
                        required=False,
                    ),
                    ToolParameter(
                        name="contract_address",
                        type=ToolParameterType.STRING,
                        description="Contract address being interacted with",
                        required=False,
                    ),
                ],
            ),
            self._handle_check_action_allowed,
            requires_address=True,
        )

        self.register(
            "get_system_status",
            ToolDefinition(
                name="get_system_status",
                description=(
                    "Check the current system status and platform operational state. "
                    "Returns whether the system is operational, any emergency stops, "
                    "maintenance windows, blocked contracts/tokens, and allowed chains. "
                    "Use this to check if the platform is available for trading."
                ),
                parameters=[],
            ),
            self._handle_get_system_status,
        )

        # =====================================================================
        # Strategy Engine Tools
        # =====================================================================

        self.register(
            "list_dca_strategies",
            ToolDefinition(
                name="list_dca_strategies",
                description=(
                    "List all DCA (Dollar Cost Averaging) strategies for a wallet. "
                    "Returns strategy names, status, tokens being traded, amounts, and schedules. "
                    "Use this when the user asks about their DCA strategies, automated trading, "
                    "recurring buys, or scheduled investments."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to list strategies for",
                        required=True,
                    ),
                    ToolParameter(
                        name="status",
                        type=ToolParameterType.STRING,
                        description="Filter by status",
                        required=False,
                        enum=["draft", "pending_session", "active", "paused", "completed", "failed", "expired"],
                    ),
                ],
            ),
            self._handle_list_dca_strategies,
            requires_address=True,
        )

        self.register(
            "get_dca_strategy",
            ToolDefinition(
                name="get_dca_strategy",
                description=(
                    "Get detailed information about a specific DCA strategy including "
                    "configuration, execution stats, and recent execution history. "
                    "Use this when the user asks for details about a specific strategy."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the DCA strategy to get",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_dca_strategy,
        )

        self.register(
            "create_dca_strategy",
            ToolDefinition(
                name="create_dca_strategy",
                description=(
                    "Create a new DCA (Dollar Cost Averaging) strategy for automated recurring trades. "
                    "Sets up a schedule to buy a target token using a source token at regular intervals. "
                    "Use this when the user wants to set up a DCA, recurring buy, or automated investment."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to create the strategy for",
                        required=True,
                    ),
                    ToolParameter(
                        name="name",
                        type=ToolParameterType.STRING,
                        description="Name for the strategy (e.g., 'Weekly ETH DCA')",
                        required=True,
                    ),
                    ToolParameter(
                        name="from_token_symbol",
                        type=ToolParameterType.STRING,
                        description="Symbol of token to spend (e.g., 'USDC')",
                        required=True,
                    ),
                    ToolParameter(
                        name="to_token_symbol",
                        type=ToolParameterType.STRING,
                        description="Symbol of token to buy (e.g., 'ETH')",
                        required=True,
                    ),
                    ToolParameter(
                        name="amount_per_execution_usd",
                        type=ToolParameterType.NUMBER,
                        description="Amount in USD to spend per execution",
                        required=True,
                    ),
                    ToolParameter(
                        name="frequency",
                        type=ToolParameterType.STRING,
                        description="How often to execute",
                        required=True,
                        enum=["hourly", "daily", "weekly", "biweekly", "monthly"],
                    ),
                    ToolParameter(
                        name="chain_id",
                        type=ToolParameterType.INTEGER,
                        description="Chain ID (1=Ethereum, 137=Polygon, 8453=Base, etc.)",
                        required=False,
                        default=1,
                    ),
                    ToolParameter(
                        name="max_slippage_percent",
                        type=ToolParameterType.NUMBER,
                        description="Maximum slippage tolerance in percent (e.g., 1.0 for 1%)",
                        required=False,
                        default=1.0,
                    ),
                    ToolParameter(
                        name="max_gas_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum gas to pay per execution in USD",
                        required=False,
                        default=10.0,
                    ),
                    ToolParameter(
                        name="max_total_spend_usd",
                        type=ToolParameterType.NUMBER,
                        description="Total budget - stop when this amount is spent",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_executions",
                        type=ToolParameterType.INTEGER,
                        description="Maximum number of executions before stopping",
                        required=False,
                    ),
                ],
            ),
            self._handle_create_dca_strategy,
            requires_address=True,
        )

        self.register(
            "pause_dca_strategy",
            ToolDefinition(
                name="pause_dca_strategy",
                description=(
                    "Pause an active DCA strategy. The strategy can be resumed later. "
                    "Use this when the user wants to temporarily stop a DCA strategy."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to pause",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type=ToolParameterType.STRING,
                        description="Optional reason for pausing",
                        required=False,
                    ),
                ],
            ),
            self._handle_pause_dca_strategy,
        )

        self.register(
            "resume_dca_strategy",
            ToolDefinition(
                name="resume_dca_strategy",
                description=(
                    "Resume a paused DCA strategy. Schedules the next execution. "
                    "Use this when the user wants to restart a paused DCA strategy."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to resume",
                        required=True,
                    ),
                ],
            ),
            self._handle_resume_dca_strategy,
        )

        self.register(
            "stop_dca_strategy",
            ToolDefinition(
                name="stop_dca_strategy",
                description=(
                    "Stop/complete a DCA strategy permanently. "
                    "Use this when the user wants to end a DCA strategy."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to stop",
                        required=True,
                    ),
                ],
            ),
            self._handle_stop_dca_strategy,
        )

        self.register(
            "get_dca_executions",
            ToolDefinition(
                name="get_dca_executions",
                description=(
                    "Get the execution history for a DCA strategy. "
                    "Returns past trades, amounts, prices, and any errors. "
                    "Use this when the user asks about their DCA history or past executions."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy",
                        required=True,
                    ),
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Maximum number of executions to return",
                        required=False,
                        default=10,
                    ),
                ],
            ),
            self._handle_get_dca_executions,
        )

        self.register(
            "update_dca_strategy",
            ToolDefinition(
                name="update_dca_strategy",
                description=(
                    "Update configuration of a DCA strategy (only when paused or draft). "
                    "Use this when the user wants to change their DCA settings."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to update",
                        required=True,
                    ),
                    ToolParameter(
                        name="amount_per_execution_usd",
                        type=ToolParameterType.NUMBER,
                        description="New amount per execution in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="frequency",
                        type=ToolParameterType.STRING,
                        description="New execution frequency",
                        required=False,
                        enum=["hourly", "daily", "weekly", "biweekly", "monthly"],
                    ),
                    ToolParameter(
                        name="max_slippage_percent",
                        type=ToolParameterType.NUMBER,
                        description="New max slippage in percent",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_gas_usd",
                        type=ToolParameterType.NUMBER,
                        description="New max gas in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_total_spend_usd",
                        type=ToolParameterType.NUMBER,
                        description="New total budget in USD",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_executions",
                        type=ToolParameterType.INTEGER,
                        description="New max number of executions",
                        required=False,
                    ),
                ],
            ),
            self._handle_update_dca_strategy,
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
        from ...db import get_convex_client
        from ...services.news_fetcher.service import NewsFetcherService

        try:
            # Initialize service with Convex client
            convex = get_convex_client()
            service = NewsFetcherService(convex_client=convex)

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
        from ...db import get_convex_client
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
            self.logger.error(f"Error fetching personalized news: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_token_news(
        self,
        symbols: List[str],
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Handle token-specific news fetch."""
        from ...db import get_convex_client
        from ...services.news_fetcher.service import NewsFetcherService

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
            self.logger.error(f"Error fetching token news: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Policy Tool Handlers
    # =========================================================================

    async def _handle_get_risk_policy(
        self,
        wallet_address: str,
    ) -> Dict[str, Any]:
        """Handle getting risk policy for a wallet."""
        from ...db import get_convex_client
        from ..policy.models import RiskPolicyConfig

        try:
            convex = get_convex_client()

            # Try to fetch from Convex
            policy_data = await convex.query(
                "riskPolicies:getByWallet",
                {"walletAddress": wallet_address.lower()},
            )

            if policy_data and policy_data.get("config"):
                config = policy_data["config"]
                return {
                    "success": True,
                    "wallet_address": wallet_address,
                    "policy": {
                        "max_position_percent": config.get("maxPositionPercent", 25.0),
                        "max_position_value_usd": config.get("maxPositionValueUsd", 10000),
                        "max_daily_volume_usd": config.get("maxDailyVolumeUsd", 50000),
                        "max_daily_loss_usd": config.get("maxDailyLossUsd", 1000),
                        "max_single_tx_usd": config.get("maxSingleTxUsd", 5000),
                        "require_approval_above_usd": config.get("requireApprovalAboveUsd", 2000),
                        "max_slippage_percent": config.get("maxSlippagePercent", 3.0),
                        "warn_slippage_percent": config.get("warnSlippagePercent", 1.5),
                        "max_gas_percent": config.get("maxGasPercent", 5.0),
                        "warn_gas_percent": config.get("warnGasPercent", 2.0),
                        "min_liquidity_usd": config.get("minLiquidityUsd", 100000),
                        "enabled": config.get("enabled", True),
                    },
                    "updated_at": policy_data.get("updatedAt"),
                    "is_default": False,
                }
            else:
                # Return default policy
                default_config = RiskPolicyConfig()
                return {
                    "success": True,
                    "wallet_address": wallet_address,
                    "policy": default_config.to_dict(),
                    "is_default": True,
                    "message": "No custom policy found, using defaults",
                }

        except Exception as e:
            self.logger.error(f"Error fetching risk policy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_update_risk_policy(
        self,
        wallet_address: str,
        max_position_percent: Optional[float] = None,
        max_position_value_usd: Optional[float] = None,
        max_daily_volume_usd: Optional[float] = None,
        max_daily_loss_usd: Optional[float] = None,
        max_single_tx_usd: Optional[float] = None,
        require_approval_above_usd: Optional[float] = None,
        max_slippage_percent: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Handle updating risk policy for a wallet."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            # Get existing policy or defaults
            existing = await convex.query(
                "riskPolicies:getByWallet",
                {"walletAddress": wallet_address.lower()},
            )

            # Build config with existing values as base
            if existing and existing.get("config"):
                config = existing["config"].copy()
            else:
                config = {
                    "maxPositionPercent": 25.0,
                    "maxPositionValueUsd": 10000,
                    "maxDailyVolumeUsd": 50000,
                    "maxDailyLossUsd": 1000,
                    "maxSingleTxUsd": 5000,
                    "requireApprovalAboveUsd": 2000,
                    "maxSlippagePercent": 3.0,
                    "warnSlippagePercent": 1.5,
                    "maxGasPercent": 5.0,
                    "warnGasPercent": 2.0,
                    "minLiquidityUsd": 100000,
                    "enabled": True,
                }

            # Apply updates
            updates_made = []
            if max_position_percent is not None:
                config["maxPositionPercent"] = max_position_percent
                updates_made.append(f"max_position_percent={max_position_percent}%")
            if max_position_value_usd is not None:
                config["maxPositionValueUsd"] = max_position_value_usd
                updates_made.append(f"max_position_value_usd=${max_position_value_usd}")
            if max_daily_volume_usd is not None:
                config["maxDailyVolumeUsd"] = max_daily_volume_usd
                updates_made.append(f"max_daily_volume_usd=${max_daily_volume_usd}")
            if max_daily_loss_usd is not None:
                config["maxDailyLossUsd"] = max_daily_loss_usd
                updates_made.append(f"max_daily_loss_usd=${max_daily_loss_usd}")
            if max_single_tx_usd is not None:
                config["maxSingleTxUsd"] = max_single_tx_usd
                updates_made.append(f"max_single_tx_usd=${max_single_tx_usd}")
            if require_approval_above_usd is not None:
                config["requireApprovalAboveUsd"] = require_approval_above_usd
                updates_made.append(f"require_approval_above_usd=${require_approval_above_usd}")
            if max_slippage_percent is not None:
                config["maxSlippagePercent"] = max_slippage_percent
                updates_made.append(f"max_slippage_percent={max_slippage_percent}%")
            if enabled is not None:
                config["enabled"] = enabled
                updates_made.append(f"enabled={enabled}")

            # Save to Convex
            await convex.mutation(
                "riskPolicies:upsert",
                {
                    "walletAddress": wallet_address.lower(),
                    "config": config,
                },
            )

            return {
                "success": True,
                "wallet_address": wallet_address,
                "updates_made": updates_made,
                "new_policy": config,
                "message": f"Updated {len(updates_made)} risk policy settings",
            }

        except Exception as e:
            self.logger.error(f"Error updating risk policy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_check_action_allowed(
        self,
        wallet_address: str,
        action_type: str,
        value_usd: float,
        chain_id: int = 1,
        token_in: Optional[str] = None,
        token_out: Optional[str] = None,
        slippage_percent: Optional[float] = None,
        contract_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle checking if an action is allowed by policies."""
        from decimal import Decimal
        from ...db import get_convex_client
        from ..policy import PolicyEngine, ActionContext, RiskPolicyConfig, SystemPolicyConfig

        try:
            convex = get_convex_client()

            # Fetch risk policy for this wallet
            risk_policy_data = await convex.query(
                "riskPolicies:getByWallet",
                {"walletAddress": wallet_address.lower()},
            )

            risk_config = None
            if risk_policy_data and risk_policy_data.get("config"):
                risk_config = RiskPolicyConfig.from_dict(risk_policy_data["config"])
            else:
                risk_config = RiskPolicyConfig()

            # Fetch system policy
            system_policy_data = await convex.query("systemPolicy:get", {})
            system_config = SystemPolicyConfig()
            if system_policy_data:
                system_config = SystemPolicyConfig(
                    emergency_stop=system_policy_data.get("emergencyStop", False),
                    emergency_stop_reason=system_policy_data.get("emergencyStopReason"),
                    in_maintenance=system_policy_data.get("inMaintenance", False),
                    maintenance_message=system_policy_data.get("maintenanceMessage"),
                    blocked_contracts=system_policy_data.get("blockedContracts", []),
                    blocked_tokens=system_policy_data.get("blockedTokens", []),
                    blocked_chains=system_policy_data.get("blockedChains", []),
                    allowed_chains=system_policy_data.get("allowedChains", []),
                    protocol_whitelist_enabled=system_policy_data.get("protocolWhitelistEnabled", False),
                    allowed_protocols=system_policy_data.get("allowedProtocols", []),
                    max_single_tx_usd=Decimal(str(system_policy_data.get("maxSingleTxUsd", 100000))),
                )

            # Build action context
            context = ActionContext(
                session_id="agent-check",
                wallet_address=wallet_address.lower(),
                action_type=action_type,
                chain_id=chain_id,
                value_usd=Decimal(str(value_usd)),
                contract_address=contract_address,
                token_in=token_in,
                token_out=token_out,
                slippage_percent=slippage_percent,
            )

            # Evaluate policies
            engine = PolicyEngine(
                risk_config=risk_config,
                system_config=system_config,
            )
            result = engine.evaluate(context)

            return {
                "success": True,
                "approved": result.approved,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level.value,
                "requires_approval": result.requires_approval,
                "approval_reason": result.approval_reason,
                "violations": [v.to_dict() for v in result.violations],
                "warnings": [w.to_dict() for w in result.warnings],
                "action": {
                    "type": action_type,
                    "value_usd": value_usd,
                    "chain_id": chain_id,
                    "token_in": token_in,
                    "token_out": token_out,
                },
            }

        except Exception as e:
            self.logger.error(f"Error checking action: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_system_status(self) -> Dict[str, Any]:
        """Handle getting system status."""
        from ...db import get_convex_client
        from ..policy import PolicyEngine, SystemPolicyConfig

        try:
            convex = get_convex_client()

            # Fetch system policy
            system_policy_data = await convex.query("systemPolicy:get", {})

            if system_policy_data:
                is_operational = not (
                    system_policy_data.get("emergencyStop", False) or
                    system_policy_data.get("inMaintenance", False)
                )

                return {
                    "success": True,
                    "operational": is_operational,
                    "emergency_stop": system_policy_data.get("emergencyStop", False),
                    "emergency_stop_reason": system_policy_data.get("emergencyStopReason"),
                    "in_maintenance": system_policy_data.get("inMaintenance", False),
                    "maintenance_message": system_policy_data.get("maintenanceMessage"),
                    "blocked_contracts_count": len(system_policy_data.get("blockedContracts", [])),
                    "blocked_tokens_count": len(system_policy_data.get("blockedTokens", [])),
                    "blocked_chains": system_policy_data.get("blockedChains", []),
                    "allowed_chains": system_policy_data.get("allowedChains", []),
                    "protocol_whitelist_enabled": system_policy_data.get("protocolWhitelistEnabled", False),
                    "max_single_tx_usd": system_policy_data.get("maxSingleTxUsd", 100000),
                    "updated_at": system_policy_data.get("updatedAt"),
                }
            else:
                # No system policy configured, assume operational with defaults
                return {
                    "success": True,
                    "operational": True,
                    "emergency_stop": False,
                    "in_maintenance": False,
                    "message": "No system policy configured, using defaults",
                }

        except Exception as e:
            self.logger.error(f"Error fetching system status: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Strategy Tool Handlers
    # =========================================================================

    async def _handle_list_dca_strategies(
        self,
        wallet_address: str,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle listing DCA strategies for a wallet."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            args = {"walletAddress": wallet_address.lower()}
            if status:
                args["status"] = status

            strategies = await convex.query("dca:listByWallet", args)

            if not strategies:
                return {
                    "success": True,
                    "strategies": [],
                    "count": 0,
                    "message": "No DCA strategies found for this wallet",
                }

            formatted = []
            for s in strategies:
                formatted.append({
                    "id": s.get("_id"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "from_token": s.get("fromToken", {}).get("symbol"),
                    "to_token": s.get("toToken", {}).get("symbol"),
                    "amount_per_execution_usd": s.get("amountPerExecutionUsd"),
                    "frequency": s.get("frequency"),
                    "total_executions": s.get("totalExecutions", 0),
                    "successful_executions": s.get("successfulExecutions", 0),
                    "total_spent_usd": s.get("totalAmountSpentUsd", 0),
                    "total_tokens_acquired": s.get("totalTokensAcquired", "0"),
                    "average_price_usd": s.get("averagePriceUsd"),
                    "next_execution_at": s.get("nextExecutionAt"),
                    "created_at": s.get("createdAt"),
                })

            return {
                "success": True,
                "strategies": formatted,
                "count": len(formatted),
            }

        except Exception as e:
            self.logger.error(f"Error listing DCA strategies: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_dca_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Handle getting a single DCA strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            strategy = await convex.query("dca:get", {"id": strategy_id})

            if not strategy:
                return {"success": False, "error": "Strategy not found"}

            # Get recent executions
            executions = await convex.query(
                "dca:getExecutions",
                {"strategyId": strategy_id, "limit": 5},
            )

            return {
                "success": True,
                "strategy": {
                    "id": strategy.get("_id"),
                    "name": strategy.get("name"),
                    "description": strategy.get("description"),
                    "status": strategy.get("status"),
                    "from_token": strategy.get("fromToken"),
                    "to_token": strategy.get("toToken"),
                    "amount_per_execution_usd": strategy.get("amountPerExecutionUsd"),
                    "frequency": strategy.get("frequency"),
                    "max_slippage_bps": strategy.get("maxSlippageBps"),
                    "max_gas_usd": strategy.get("maxGasUsd"),
                    "max_total_spend_usd": strategy.get("maxTotalSpendUsd"),
                    "max_executions": strategy.get("maxExecutions"),
                    "stats": {
                        "total_executions": strategy.get("totalExecutions", 0),
                        "successful_executions": strategy.get("successfulExecutions", 0),
                        "failed_executions": strategy.get("failedExecutions", 0),
                        "skipped_executions": strategy.get("skippedExecutions", 0),
                        "total_spent_usd": strategy.get("totalAmountSpentUsd", 0),
                        "total_tokens_acquired": strategy.get("totalTokensAcquired", "0"),
                        "average_price_usd": strategy.get("averagePriceUsd"),
                    },
                    "next_execution_at": strategy.get("nextExecutionAt"),
                    "last_execution_at": strategy.get("lastExecutionAt"),
                    "last_error": strategy.get("lastError"),
                    "created_at": strategy.get("createdAt"),
                },
                "recent_executions": executions or [],
            }

        except Exception as e:
            self.logger.error(f"Error getting DCA strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_create_dca_strategy(
        self,
        wallet_address: str,
        name: str,
        from_token_symbol: str,
        to_token_symbol: str,
        amount_per_execution_usd: float,
        frequency: str,
        chain_id: int = 1,
        max_slippage_percent: float = 1.0,
        max_gas_usd: float = 10.0,
        max_total_spend_usd: Optional[float] = None,
        max_executions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Handle creating a new DCA strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            # Get wallet and user info
            wallet = await convex.query(
                "wallets:getByAddress",
                {"address": wallet_address.lower()},
            )

            if not wallet:
                return {"success": False, "error": "Wallet not found. Please connect your wallet first."}

            # Token address lookups would go here in production
            from_token = {
                "symbol": from_token_symbol.upper(),
                "address": "0x",
                "chainId": chain_id,
                "decimals": 18 if from_token_symbol.upper() == "ETH" else 6,
            }
            to_token = {
                "symbol": to_token_symbol.upper(),
                "address": "0x",
                "chainId": chain_id,
                "decimals": 18,
            }

            # Convert slippage percent to basis points
            max_slippage_bps = int(max_slippage_percent * 100)

            args = {
                "userId": wallet.get("userId"),
                "walletId": wallet.get("_id"),
                "walletAddress": wallet_address.lower(),
                "name": name,
                "fromToken": from_token,
                "toToken": to_token,
                "amountPerExecutionUsd": amount_per_execution_usd,
                "frequency": frequency,
                "executionHourUtc": 12,
                "maxSlippageBps": max_slippage_bps,
                "maxGasUsd": max_gas_usd,
            }

            if max_total_spend_usd is not None:
                args["maxTotalSpendUsd"] = max_total_spend_usd
            if max_executions is not None:
                args["maxExecutions"] = max_executions

            strategy_id = await convex.mutation("dca:create", args)

            return {
                "success": True,
                "strategy_id": strategy_id,
                "name": name,
                "status": "draft",
                "message": (
                    f"DCA strategy '{name}' created! "
                    f"Will buy {to_token_symbol} with {from_token_symbol} "
                    f"for ${amount_per_execution_usd} {frequency}. "
                    "Activate the strategy with a session key to start."
                ),
            }

        except Exception as e:
            self.logger.error(f"Error creating DCA strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_pause_dca_strategy(
        self,
        strategy_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle pausing a DCA strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            args = {"strategyId": strategy_id}
            if reason:
                args["reason"] = reason

            await convex.mutation("dca:pause", args)

            return {
                "success": True,
                "strategy_id": strategy_id,
                "status": "paused",
                "message": "DCA strategy paused successfully",
            }

        except Exception as e:
            self.logger.error(f"Error pausing DCA strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_resume_dca_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Handle resuming a paused DCA strategy."""
        from ...db import get_convex_client
        import time

        try:
            convex = get_convex_client()

            # Schedule next execution for 1 hour from now
            next_execution_at = int(time.time() * 1000) + (60 * 60 * 1000)

            await convex.mutation(
                "dca:resume",
                {"strategyId": strategy_id, "nextExecutionAt": next_execution_at},
            )

            return {
                "success": True,
                "strategy_id": strategy_id,
                "status": "active",
                "next_execution_at": next_execution_at,
                "message": "DCA strategy resumed successfully",
            }

        except Exception as e:
            self.logger.error(f"Error resuming DCA strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_stop_dca_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Handle stopping a DCA strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            await convex.mutation("dca:stop", {"strategyId": strategy_id})

            return {
                "success": True,
                "strategy_id": strategy_id,
                "status": "completed",
                "message": "DCA strategy stopped successfully",
            }

        except Exception as e:
            self.logger.error(f"Error stopping DCA strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_dca_executions(
        self,
        strategy_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Handle getting DCA execution history."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            executions = await convex.query(
                "dca:getExecutions",
                {"strategyId": strategy_id, "limit": limit},
            )

            if not executions:
                return {
                    "success": True,
                    "executions": [],
                    "count": 0,
                    "message": "No executions yet for this strategy",
                }

            formatted = []
            for e in executions:
                formatted.append({
                    "execution_number": e.get("executionNumber"),
                    "status": e.get("status"),
                    "scheduled_at": e.get("scheduledAt"),
                    "started_at": e.get("startedAt"),
                    "completed_at": e.get("completedAt"),
                    "tx_hash": e.get("txHash"),
                    "input_amount": e.get("actualInputAmount"),
                    "output_amount": e.get("actualOutputAmount"),
                    "price_usd": e.get("actualPriceUsd"),
                    "gas_usd": e.get("gasUsd"),
                    "skip_reason": e.get("skipReason"),
                    "error_message": e.get("errorMessage"),
                })

            return {
                "success": True,
                "executions": formatted,
                "count": len(formatted),
            }

        except Exception as e:
            self.logger.error(f"Error getting DCA executions: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_update_dca_strategy(
        self,
        strategy_id: str,
        amount_per_execution_usd: Optional[float] = None,
        frequency: Optional[str] = None,
        max_slippage_percent: Optional[float] = None,
        max_gas_usd: Optional[float] = None,
        max_total_spend_usd: Optional[float] = None,
        max_executions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Handle updating a DCA strategy configuration."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            args = {"strategyId": strategy_id}
            updates_made = []

            if amount_per_execution_usd is not None:
                args["amountPerExecutionUsd"] = amount_per_execution_usd
                updates_made.append(f"amount=${amount_per_execution_usd}")
            if frequency is not None:
                args["frequency"] = frequency
                updates_made.append(f"frequency={frequency}")
            if max_slippage_percent is not None:
                args["maxSlippageBps"] = int(max_slippage_percent * 100)
                updates_made.append(f"max_slippage={max_slippage_percent}%")
            if max_gas_usd is not None:
                args["maxGasUsd"] = max_gas_usd
                updates_made.append(f"max_gas=${max_gas_usd}")
            if max_total_spend_usd is not None:
                args["maxTotalSpendUsd"] = max_total_spend_usd
                updates_made.append(f"max_total=${max_total_spend_usd}")
            if max_executions is not None:
                args["maxExecutions"] = max_executions
                updates_made.append(f"max_executions={max_executions}")

            if len(updates_made) == 0:
                return {"success": False, "error": "No updates provided"}

            await convex.mutation("dca:updateConfig", args)

            return {
                "success": True,
                "strategy_id": strategy_id,
                "updates_made": updates_made,
                "message": f"Updated {len(updates_made)} settings",
            }

        except Exception as e:
            self.logger.error(f"Error updating DCA strategy: {e}")
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
