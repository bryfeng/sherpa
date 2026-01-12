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
            "list_strategies",
            ToolDefinition(
                name="list_strategies",
                description=(
                    "List all automated trading strategies for a wallet. "
                    "Returns strategy names, types, status, and configurations. "
                    "Strategy types include: dca (dollar cost averaging), rebalance, "
                    "limit_order, stop_loss, take_profit, and more. "
                    "Use this when the user asks about their strategies, automated trading, "
                    "or scheduled investments."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to list strategies for",
                        required=True,
                    ),
                    ToolParameter(
                        name="strategy_type",
                        type=ToolParameterType.STRING,
                        description="Filter by strategy type",
                        required=False,
                        enum=["dca", "rebalance", "limit_order", "stop_loss", "take_profit"],
                    ),
                    ToolParameter(
                        name="status",
                        type=ToolParameterType.STRING,
                        description="Filter by status",
                        required=False,
                        enum=["draft", "active", "paused", "completed", "failed", "expired"],
                    ),
                ],
            ),
            self._handle_list_strategies,
            requires_address=True,
        )

        self.register(
            "get_strategy",
            ToolDefinition(
                name="get_strategy",
                description=(
                    "Get detailed information about a specific strategy including "
                    "configuration, execution stats, and recent execution history. "
                    "Use this when the user asks for details about a specific strategy."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to get",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_strategy,
        )

        self.register(
            "create_strategy",
            ToolDefinition(
                name="create_strategy",
                description=(
                    "Create a new automated trading strategy. "
                    "Supports multiple strategy types: "
                    "- dca: Dollar cost averaging - buy tokens at regular intervals "
                    "- rebalance: Maintain target portfolio allocations "
                    "- limit_order: Execute when price reaches target "
                    "- stop_loss: Sell when price drops below threshold "
                    "- take_profit: Sell when price rises above target "
                    "Use this when the user wants to set up any automated trading strategy."
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
                        description="Name for the strategy (e.g., 'Weekly ETH Buy', 'BTC Stop Loss')",
                        required=True,
                    ),
                    ToolParameter(
                        name="strategy_type",
                        type=ToolParameterType.STRING,
                        description="Type of strategy to create",
                        required=True,
                        enum=["dca", "rebalance", "limit_order", "stop_loss", "take_profit"],
                    ),
                    ToolParameter(
                        name="config",
                        type=ToolParameterType.OBJECT,
                        description=(
                            "Strategy configuration object. Fields depend on strategy_type: "
                            "DCA: {from_token, to_token, amount_usd, frequency} "
                            "Rebalance: {target_allocations, threshold_percent} "
                            "Limit/Stop/TakeProfit: {token, trigger_price_usd, amount, side}"
                        ),
                        required=True,
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
                        description="Maximum slippage tolerance in percent",
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
                ],
            ),
            self._handle_create_strategy,
            requires_address=True,
        )

        self.register(
            "pause_strategy",
            ToolDefinition(
                name="pause_strategy",
                description=(
                    "Pause an active strategy. The strategy can be resumed later. "
                    "Use this when the user wants to temporarily stop any strategy."
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
            self._handle_pause_strategy,
        )

        self.register(
            "resume_strategy",
            ToolDefinition(
                name="resume_strategy",
                description=(
                    "Resume a paused strategy. Schedules the next execution. "
                    "Use this when the user wants to restart a paused strategy."
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
            self._handle_resume_strategy,
        )

        self.register(
            "stop_strategy",
            ToolDefinition(
                name="stop_strategy",
                description=(
                    "Stop/complete a strategy permanently. "
                    "Use this when the user wants to end any strategy."
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
            self._handle_stop_strategy,
        )

        self.register(
            "get_strategy_executions",
            ToolDefinition(
                name="get_strategy_executions",
                description=(
                    "Get the execution history for a strategy. "
                    "Returns past trades, amounts, prices, and any errors. "
                    "Use this when the user asks about strategy history or past executions."
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
            self._handle_get_strategy_executions,
        )

        # Phase 13: Strategy execution approval tool
        self.register(
            "approve_strategy_execution",
            ToolDefinition(
                name="approve_strategy_execution",
                description=(
                    "Approve or reject a pending strategy execution. "
                    "Use this when the user says 'approve', 'yes', 'execute', 'do it', 'go ahead' "
                    "in response to a strategy execution approval request. "
                    "Also use when user says 'skip', 'no', 'reject', 'cancel' to skip the execution. "
                    "The execution_id should be taken from the approval request message metadata."
                ),
                parameters=[
                    ToolParameter(
                        name="execution_id",
                        type=ToolParameterType.STRING,
                        description="The execution ID from the approval request",
                        required=True,
                    ),
                    ToolParameter(
                        name="approve",
                        type=ToolParameterType.BOOLEAN,
                        description="True to approve and execute, False to skip/reject",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type=ToolParameterType.STRING,
                        description="Optional reason for skipping (if approve=False)",
                        required=False,
                    ),
                ],
            ),
            self._handle_approve_strategy_execution,
            requires_address=True,
        )

        self.register(
            "update_strategy",
            ToolDefinition(
                name="update_strategy",
                description=(
                    "Update configuration of a strategy (only when paused or draft). "
                    "Use this when the user wants to change strategy settings."
                ),
                parameters=[
                    ToolParameter(
                        name="strategy_id",
                        type=ToolParameterType.STRING,
                        description="The ID of the strategy to update",
                        required=True,
                    ),
                    ToolParameter(
                        name="config",
                        type=ToolParameterType.OBJECT,
                        description="Updated configuration fields (strategy-type specific)",
                        required=False,
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
                ],
            ),
            self._handle_update_strategy,
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

    async def _handle_list_strategies(
        self,
        wallet_address: str,
        strategy_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle listing strategies for a wallet."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            # Query the general strategies table
            args = {"walletAddress": wallet_address.lower()}
            if status:
                args["status"] = status

            strategies = await convex.query("strategies:listByWallet", args)

            if not strategies:
                return {
                    "success": True,
                    "strategies": [],
                    "count": 0,
                    "message": "No strategies found for this wallet",
                }

            formatted = []
            for s in strategies:
                strategy_data = {
                    "id": s.get("_id"),
                    "name": s.get("name"),
                    "type": s.get("strategyType", "custom"),
                    "status": s.get("status"),
                    "config": s.get("config", {}),
                    "total_executions": s.get("totalExecutions", 0),
                    "next_execution_at": s.get("nextExecutionAt"),
                    "created_at": s.get("createdAt"),
                }

                # Filter by type if specified
                if strategy_type and strategy_data["type"] != strategy_type:
                    continue

                formatted.append(strategy_data)

            return {
                "success": True,
                "strategies": formatted,
                "count": len(formatted),
            }

        except Exception as e:
            self.logger.error(f"Error listing strategies: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Handle getting a single strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            strategy = await convex.query("strategies:get", {"strategyId": strategy_id})

            if not strategy:
                return {"success": False, "error": "Strategy not found"}

            # Get recent executions
            executions = await convex.query(
                "strategies:getWithExecutions",
                {"strategyId": strategy_id, "limit": 5},
            )

            return {
                "success": True,
                "strategy": {
                    "id": strategy.get("_id"),
                    "name": strategy.get("name"),
                    "description": strategy.get("description"),
                    "type": strategy.get("strategyType", "custom"),
                    "status": strategy.get("status"),
                    "config": strategy.get("config", {}),
                    "stats": {
                        "total_executions": strategy.get("totalExecutions", 0),
                        "successful_executions": strategy.get("successfulExecutions", 0),
                        "failed_executions": strategy.get("failedExecutions", 0),
                    },
                    "next_execution_at": strategy.get("nextExecutionAt"),
                    "last_execution_at": strategy.get("lastExecutionAt"),
                    "last_error": strategy.get("lastError"),
                    "created_at": strategy.get("createdAt"),
                },
                "recent_executions": executions.get("executions", []) if executions else [],
            }

        except Exception as e:
            self.logger.error(f"Error getting strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_create_strategy(
        self,
        wallet_address: str,
        name: str,
        strategy_type: str,
        config: Dict[str, Any],
        chain_id: int = 1,
        max_slippage_percent: float = 1.0,
        max_gas_usd: float = 10.0,
        chain: Optional[str] = None,  # Accept 'chain' param injected by ReAct loop
        **kwargs,  # Accept any other injected params
    ) -> Dict[str, Any]:
        """Handle creating a new strategy."""
        # Map chain name to chain_id if provided
        if chain and chain_id == 1:  # Only override if chain_id is default
            chain_map = {
                "ethereum": 1,
                "polygon": 137,
                "base": 8453,
                "arbitrum": 42161,
                "optimism": 10,
                "solana": -1,  # Special case
            }
            chain_id = chain_map.get(chain.lower(), 1)
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            # Determine chain for lookup
            chain_name = chain or "ethereum"
            if chain_id and chain_id != 1:
                chain_id_to_name = {137: "polygon", 8453: "base", 42161: "arbitrum", 10: "optimism"}
                chain_name = chain_id_to_name.get(chain_id, "ethereum")

            # Get or create user and wallet in one call
            # This handles all the registration logic automatically
            result = await convex.mutation(
                "users:getOrCreateByWallet",
                {"address": wallet_address.lower(), "chain": chain_name},
            )

            wallet = result.get("wallet") if result else None
            user = result.get("user") if result else None
            is_new = result.get("isNew", False) if result else False

            if is_new:
                self.logger.info(f"Auto-registered new user and wallet for {wallet_address}")

            if not wallet or not user:
                return {"success": False, "error": "Could not find or create wallet. Please try again."}

            # Validate config based on strategy type
            if strategy_type == "dca":
                required_fields = ["from_token", "to_token", "amount_usd", "frequency"]
                for field in required_fields:
                    if field not in config:
                        return {"success": False, "error": f"DCA strategy requires '{field}' in config"}
            elif strategy_type == "rebalance":
                if "target_allocations" not in config:
                    return {"success": False, "error": "Rebalance strategy requires 'target_allocations' in config"}
            elif strategy_type in ["limit_order", "stop_loss", "take_profit"]:
                required_fields = ["token", "trigger_price_usd", "amount"]
                for field in required_fields:
                    if field not in config:
                        return {"success": False, "error": f"{strategy_type} strategy requires '{field}' in config"}

            # Convert slippage percent to basis points
            max_slippage_bps = int(max_slippage_percent * 100)

            args = {
                "userId": user.get("_id"),
                "walletAddress": wallet_address.lower(),
                "name": name,
                "strategyType": strategy_type,
                "config": {
                    **config,
                    "chainId": chain_id,
                    "maxSlippageBps": max_slippage_bps,
                    "maxGasUsd": max_gas_usd,
                },
            }

            strategy_id = await convex.mutation("strategies:create", args)

            return {
                "success": True,
                "strategy_id": strategy_id,
                "name": name,
                "type": strategy_type,
                "status": "draft",
                "message": f"Strategy '{name}' ({strategy_type}) created. Activate with a session key to start.",
            }

        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_pause_strategy(
        self,
        strategy_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle pausing a strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            await convex.mutation("strategies:pause", {"strategyId": strategy_id})

            return {
                "success": True,
                "strategy_id": strategy_id,
                "status": "paused",
                "message": "Strategy paused successfully",
            }

        except Exception as e:
            self.logger.error(f"Error pausing strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_resume_strategy(
        self,
        strategy_id: str,
        session_key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle resuming a paused strategy.

        NOTE: Without a session_key_id, the strategy will go to 'pending_session' status.
        A session key is required for actual automated execution.
        """
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            args = {"strategyId": strategy_id}
            if session_key_id:
                args["sessionKeyId"] = session_key_id

            await convex.mutation("strategies:activate", args)

            if session_key_id:
                return {
                    "success": True,
                    "strategy_id": strategy_id,
                    "status": "active",
                    "message": "Strategy activated with session key. Automated execution enabled.",
                }
            else:
                return {
                    "success": True,
                    "strategy_id": strategy_id,
                    "status": "pending_session",
                    "message": (
                        "Strategy is ready but requires a session key for automated execution. "
                        "Please authorize a session key to enable automatic trading."
                    ),
                }

        except Exception as e:
            self.logger.error(f"Error resuming strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_stop_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Handle stopping a strategy."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            await convex.mutation(
                "strategies:updateStatus",
                {"strategyId": strategy_id, "status": "archived"},
            )

            return {
                "success": True,
                "strategy_id": strategy_id,
                "status": "archived",
                "message": "Strategy stopped successfully",
            }

        except Exception as e:
            self.logger.error(f"Error stopping strategy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_strategy_executions(
        self,
        strategy_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Handle getting strategy execution history."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            result = await convex.query(
                "strategies:getWithExecutions",
                {"strategyId": strategy_id, "limit": limit},
            )

            if not result or not result.get("executions"):
                return {
                    "success": True,
                    "executions": [],
                    "count": 0,
                    "message": "No executions yet for this strategy",
                }

            executions = result.get("executions", [])
            formatted = []
            for e in executions:
                formatted.append({
                    "execution_id": e.get("_id"),
                    "status": e.get("status"),
                    "started_at": e.get("startedAt"),
                    "completed_at": e.get("completedAt"),
                    "result": e.get("result"),
                    "error": e.get("error"),
                })

            return {
                "success": True,
                "executions": formatted,
                "count": len(formatted),
            }

        except Exception as e:
            self.logger.error(f"Error getting strategy executions: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_approve_strategy_execution(
        self,
        execution_id: str,
        approve: bool,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle approval or rejection of a pending strategy execution.

        Phase 13: This is called when the user responds to an approval request
        in chat. It calls the appropriate Convex mutation to update the execution
        state.
        """
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            # Get the execution to verify it exists and is awaiting approval
            execution = await convex.query(
                "strategyExecutions:get",
                {"executionId": execution_id},
            )

            if not execution:
                return {
                    "success": False,
                    "error": f"Execution not found: {execution_id}",
                }

            current_state = execution.get("currentState")
            if current_state != "awaiting_approval":
                return {
                    "success": False,
                    "error": f"Execution is not awaiting approval (current state: {current_state})",
                }

            if approve:
                # Approve the execution - transitions to "executing" state
                await convex.mutation(
                    "strategyExecutions:approve",
                    {
                        "executionId": execution_id,
                        "approverAddress": execution.get("walletAddress", ""),
                    },
                )

                strategy = execution.get("strategy", {})
                strategy_name = strategy.get("name", "Strategy")

                return {
                    "success": True,
                    "execution_id": execution_id,
                    "status": "executing",
                    "message": (
                        f"Approved! {strategy_name} execution is now in progress. "
                        "You'll be prompted to sign the transaction in your wallet."
                    ),
                }

            else:
                # Skip/reject the execution - transitions to "cancelled" state
                await convex.mutation(
                    "strategyExecutions:skip",
                    {
                        "executionId": execution_id,
                        "reason": reason or "User skipped this execution",
                    },
                )

                return {
                    "success": True,
                    "execution_id": execution_id,
                    "status": "cancelled",
                    "message": (
                        "Execution skipped. The strategy will attempt again at the next scheduled time."
                    ),
                }

        except Exception as e:
            self.logger.error(f"Error handling strategy execution approval: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_update_strategy(
        self,
        strategy_id: str,
        config: Optional[Dict[str, Any]] = None,
        max_slippage_percent: Optional[float] = None,
        max_gas_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Handle updating a strategy configuration."""
        from ...db import get_convex_client

        try:
            convex = get_convex_client()

            updates = {}
            updates_made = []

            if config is not None:
                updates["config"] = config
                updates_made.append("config updated")
            if max_slippage_percent is not None:
                if "config" not in updates:
                    updates["config"] = {}
                updates["config"]["maxSlippageBps"] = int(max_slippage_percent * 100)
                updates_made.append(f"max_slippage={max_slippage_percent}%")
            if max_gas_usd is not None:
                if "config" not in updates:
                    updates["config"] = {}
                updates["config"]["maxGasUsd"] = max_gas_usd
                updates_made.append(f"max_gas=${max_gas_usd}")

            if len(updates_made) == 0:
                return {"success": False, "error": "No updates provided"}

            await convex.mutation(
                "strategies:update",
                {"strategyId": strategy_id, **updates},
            )

            return {
                "success": True,
                "strategy_id": strategy_id,
                "updates_made": updates_made,
                "message": f"Updated {len(updates_made)} settings",
            }

        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
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
