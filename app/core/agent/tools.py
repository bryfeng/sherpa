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
                    "and market data. Use this when the user asks specifically about "
                    "trending tokens, price movers, top gainers/losers, hot coins, or "
                    "which tokens are pumping. NOT for news articles or stories."
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

        # =====================================================================
        # Solana Swap Tools
        # =====================================================================

        self.register(
            "get_solana_swap_quote",
            ToolDefinition(
                name="get_solana_swap_quote",
                description=(
                    "Get a swap quote for exchanging tokens WITHIN Solana using Jupiter DEX. "
                    "This is for Solana-to-Solana token swaps only (e.g., SOL to USDC, BONK to JUP). "
                    "Returns the expected output amount, price impact, and a transaction ready for signing. "
                    "NOTE: For cross-chain transfers TO or FROM Solana, use get_bridge_quote instead."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The Solana wallet address (public key) performing the swap",
                        required=True,
                    ),
                    ToolParameter(
                        name="input_token",
                        type=ToolParameterType.STRING,
                        description=(
                            "The input token symbol or mint address "
                            "(e.g., 'SOL', 'USDC', or full mint address)"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="output_token",
                        type=ToolParameterType.STRING,
                        description=(
                            "The output token symbol or mint address "
                            "(e.g., 'SOL', 'USDC', or full mint address)"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="amount",
                        type=ToolParameterType.NUMBER,
                        description="Amount of input token to swap (in human-readable units, not lamports)",
                        required=True,
                    ),
                    ToolParameter(
                        name="slippage_bps",
                        type=ToolParameterType.INTEGER,
                        description="Slippage tolerance in basis points (50 = 0.5%, default)",
                        required=False,
                        default=50,
                    ),
                ],
            ),
            self._handle_get_solana_swap_quote,
            requires_address=True,
        )

        # =====================================================================
        # EVM Swap Tools (via Relay)
        # =====================================================================

        self.register(
            "get_swap_quote",
            ToolDefinition(
                name="get_swap_quote",
                description=(
                    "Get a swap quote for exchanging tokens on a SINGLE EVM chain only. "
                    "Use this ONLY when both input and output tokens are on THE SAME blockchain. "
                    "Examples: 'swap ETH for USDC on Ethereum', 'exchange WBTC to ETH on Base'. "
                    "DO NOT use this if user mentions two different chains - use get_bridge_quote instead. "
                    "Supports ETH, USDC, USDT, WBTC, and many other EVM tokens. "
                    "NOTE: For cross-chain (different source/destination chains), use get_bridge_quote."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The EVM wallet address (0x...) performing the swap",
                        required=True,
                    ),
                    ToolParameter(
                        name="input_token",
                        type=ToolParameterType.STRING,
                        description=(
                            "The input token symbol or contract address "
                            "(e.g., 'ETH', 'USDC', 'WBTC', or '0x...')"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="output_token",
                        type=ToolParameterType.STRING,
                        description=(
                            "The output token symbol or contract address "
                            "(e.g., 'ETH', 'USDC', 'WBTC', or '0x...')"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="amount",
                        type=ToolParameterType.NUMBER,
                        description="Amount of input token to swap (in human-readable units)",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description=(
                            "The blockchain to swap on (e.g., 'ethereum', 'base', 'arbitrum', "
                            "'optimism', 'polygon'). Defaults to 'ethereum'."
                        ),
                        required=False,
                        default="ethereum",
                    ),
                    ToolParameter(
                        name="slippage_percent",
                        type=ToolParameterType.NUMBER,
                        description="Slippage tolerance as a percentage (e.g., 0.5 for 0.5%). Defaults to 0.5%.",
                        required=False,
                        default=0.5,
                    ),
                ],
            ),
            self._handle_get_swap_quote,
            requires_address=True,
        )

        # =====================================================================
        # Cross-Chain Bridge Tools (via Relay)
        # =====================================================================

        self.register(
            "get_bridge_quote",
            ToolDefinition(
                name="get_bridge_quote",
                description=(
                    "REQUIRED: Call this tool for ANY cross-chain token operation - including swaps, bridges, or transfers "
                    "where the source and destination are DIFFERENT blockchains. "
                    "Supports ALL EVM chains (Ethereum/mainnet, Base, Arbitrum, Optimism, Polygon, Ink, zkSync, Scroll) and Solana. "
                    "KEY PATTERN: If user mentions TWO DIFFERENT chains, use this tool. 'mainnet' = Ethereum. "
                    "Examples: 'swap USDC.e on Ink to USDC on mainnet', 'bridge ETH to Base', 'transfer from Polygon to Arbitrum', "
                    "'move USDC from Ink chain to Ethereum', 'swap tokens from Base to mainnet'. "
                    "Returns real quotes with exact amounts and fees - NEVER guess or make up numbers. "
                    "NOTE: For SAME-chain swaps only (e.g., ETH to USDC both on Ethereum), use get_swap_quote instead."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address performing the bridge (0x... for EVM, base58 for Solana)",
                        required=True,
                    ),
                    ToolParameter(
                        name="from_chain",
                        type=ToolParameterType.STRING,
                        description=(
                            "The source blockchain name (e.g., 'ethereum', 'base', 'arbitrum', "
                            "'optimism', 'polygon', 'ink', 'zksync', 'solana', etc.)"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="to_chain",
                        type=ToolParameterType.STRING,
                        description=(
                            "The destination blockchain name (e.g., 'ethereum', 'base', 'arbitrum', "
                            "'optimism', 'polygon', 'ink', 'zksync', 'solana', etc.)"
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="token",
                        type=ToolParameterType.STRING,
                        description=(
                            "The token to bridge (e.g., 'ETH', 'USDC', 'USDT'). "
                            "The same token type will be received on the destination chain."
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="amount",
                        type=ToolParameterType.NUMBER,
                        description="Amount of token to bridge (in human-readable units)",
                        required=True,
                    ),
                    ToolParameter(
                        name="destination_address",
                        type=ToolParameterType.STRING,
                        description=(
                            "Optional: Recipient address on the destination chain. "
                            "If not provided, tokens are sent to the same address (if compatible) "
                            "or the user must specify for cross-ecosystem bridges (EVM ↔ Solana)."
                        ),
                        required=False,
                    ),
                ],
            ),
            self._handle_get_bridge_quote,
            requires_address=True,
        )

        # =====================================================================
        # Copy Trading Tools
        # =====================================================================

        self.register(
            "get_top_traders",
            ToolDefinition(
                name="get_top_traders",
                description=(
                    "Discover top-performing traders for a specific token using Birdeye analytics. "
                    "Returns a list of wallet addresses ranked by PnL, win rate, or volume. "
                    "Use this when the user wants to find successful traders to copy, "
                    "asks 'who are the best traders for X token', or wants wallet recommendations."
                ),
                parameters=[
                    ToolParameter(
                        name="token_address",
                        type=ToolParameterType.STRING,
                        description="The token mint address to find top traders for",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description="Blockchain (default: solana)",
                        required=False,
                        default="solana",
                    ),
                    ToolParameter(
                        name="time_frame",
                        type=ToolParameterType.STRING,
                        description="Time frame for rankings (24h, 7d, 30d)",
                        required=False,
                        default="7d",
                        enum=["24h", "7d", "30d"],
                    ),
                    ToolParameter(
                        name="sort_by",
                        type=ToolParameterType.STRING,
                        description="Sort criteria (pnl, volume, trades, win_rate)",
                        required=False,
                        default="pnl",
                        enum=["pnl", "volume", "trades"],
                    ),
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Number of traders to return (max 20)",
                        required=False,
                        default=10,
                    ),
                ],
            ),
            self._handle_get_top_traders,
        )

        self.register(
            "get_trader_profile",
            ToolDefinition(
                name="get_trader_profile",
                description=(
                    "Get detailed analytics for a specific trader wallet including "
                    "portfolio, PnL, trade history, and performance metrics. "
                    "Use this when the user wants to analyze a wallet before copying it, "
                    "or asks about a specific trader's performance."
                ),
                parameters=[
                    ToolParameter(
                        name="wallet_address",
                        type=ToolParameterType.STRING,
                        description="The trader's wallet address to analyze",
                        required=True,
                    ),
                    ToolParameter(
                        name="chain",
                        type=ToolParameterType.STRING,
                        description="Blockchain (default: solana)",
                        required=False,
                        default="solana",
                    ),
                ],
            ),
            self._handle_get_trader_profile,
        )

        self.register(
            "start_copy_trading",
            ToolDefinition(
                name="start_copy_trading",
                description=(
                    "Start copying trades from a leader wallet. When the leader makes a swap, "
                    "you'll receive a notification to approve the copy trade. "
                    "Use this when the user says 'copy this wallet', 'follow this trader', "
                    "or wants to mirror someone's trades."
                ),
                parameters=[
                    ToolParameter(
                        name="leader_address",
                        type=ToolParameterType.STRING,
                        description="The wallet address to copy trades from",
                        required=True,
                    ),
                    ToolParameter(
                        name="leader_chain",
                        type=ToolParameterType.STRING,
                        description="The leader's blockchain (e.g., 'solana', 'ethereum')",
                        required=True,
                    ),
                    ToolParameter(
                        name="follower_address",
                        type=ToolParameterType.STRING,
                        description="Your wallet address that will execute copy trades",
                        required=True,
                    ),
                    ToolParameter(
                        name="follower_chain",
                        type=ToolParameterType.STRING,
                        description="Your wallet's blockchain",
                        required=True,
                    ),
                    ToolParameter(
                        name="sizing_mode",
                        type=ToolParameterType.STRING,
                        description="How to size copy trades: 'fixed' (fixed USD), 'percentage' (% of portfolio), 'proportional' (match leader's %)",
                        required=False,
                        default="fixed",
                        enum=["fixed", "percentage", "proportional"],
                    ),
                    ToolParameter(
                        name="size_value",
                        type=ToolParameterType.NUMBER,
                        description="Size value: USD amount for 'fixed', percentage for 'percentage' (e.g., 5 = 5%)",
                        required=False,
                        default=100,
                    ),
                    ToolParameter(
                        name="max_trade_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum USD per copy trade (safety limit)",
                        required=False,
                        default=1000,
                    ),
                ],
            ),
            self._handle_start_copy_trading,
            requires_address=True,
        )

        self.register(
            "stop_copy_trading",
            ToolDefinition(
                name="stop_copy_trading",
                description=(
                    "Stop copying trades from a leader wallet. "
                    "Use this when the user wants to unfollow a trader or stop copying."
                ),
                parameters=[
                    ToolParameter(
                        name="relationship_id",
                        type=ToolParameterType.STRING,
                        description="The copy trading relationship ID to stop",
                        required=True,
                    ),
                ],
            ),
            self._handle_stop_copy_trading,
            requires_address=True,
        )

        self.register(
            "list_copy_relationships",
            ToolDefinition(
                name="list_copy_relationships",
                description=(
                    "List all copy trading relationships for the user. "
                    "Shows which wallets the user is currently following. "
                    "Use this when the user asks 'who am I copying', 'show my copy trades', "
                    "or wants to see their copy trading status."
                ),
                parameters=[],
            ),
            self._handle_list_copy_relationships,
            requires_address=True,
        )

        self.register(
            "get_pending_copy_trades",
            ToolDefinition(
                name="get_pending_copy_trades",
                description=(
                    "Get copy trades waiting for user approval. "
                    "Returns trades that were detected from followed wallets but need manual approval. "
                    "Use this when the user asks about pending trades or copy trade notifications."
                ),
                parameters=[],
            ),
            self._handle_get_pending_copy_trades,
            requires_address=True,
        )

        self.register(
            "approve_copy_trade",
            ToolDefinition(
                name="approve_copy_trade",
                description=(
                    "Approve and execute a pending copy trade. "
                    "Use this when the user wants to proceed with a copy trade notification."
                ),
                parameters=[
                    ToolParameter(
                        name="execution_id",
                        type=ToolParameterType.STRING,
                        description="The execution ID to approve",
                        required=True,
                    ),
                ],
            ),
            self._handle_approve_copy_trade,
            requires_address=True,
        )

        self.register(
            "reject_copy_trade",
            ToolDefinition(
                name="reject_copy_trade",
                description=(
                    "Reject/skip a pending copy trade. "
                    "Use this when the user doesn't want to execute a specific copy trade."
                ),
                parameters=[
                    ToolParameter(
                        name="execution_id",
                        type=ToolParameterType.STRING,
                        description="The execution ID to reject",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type=ToolParameterType.STRING,
                        description="Optional reason for rejection",
                        required=False,
                    ),
                ],
            ),
            self._handle_reject_copy_trade,
            requires_address=True,
        )

        # =====================================================================
        # Polymarket Tools
        # =====================================================================

        self.register(
            "get_polymarket_markets",
            ToolDefinition(
                name="get_polymarket_markets",
                description=(
                    "Get prediction markets from Polymarket. "
                    "Can filter by category (politics, crypto, sports), search by query, "
                    "get trending markets, or markets closing soon. "
                    "Use this when the user wants to explore or discover prediction markets."
                ),
                parameters=[
                    ToolParameter(
                        name="category",
                        type=ToolParameterType.STRING,
                        description="Category filter: politics, crypto, sports, entertainment, science, economics",
                        required=False,
                    ),
                    ToolParameter(
                        name="query",
                        type=ToolParameterType.STRING,
                        description="Search query to find markets by question text",
                        required=False,
                    ),
                    ToolParameter(
                        name="trending",
                        type=ToolParameterType.BOOLEAN,
                        description="Get trending markets by volume (default: false)",
                        required=False,
                    ),
                    ToolParameter(
                        name="closing_soon_hours",
                        type=ToolParameterType.INTEGER,
                        description="Get markets closing within this many hours",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Max number of markets to return (default: 20)",
                        required=False,
                    ),
                ],
            ),
            self._handle_get_polymarket_markets,
        )

        self.register(
            "get_polymarket_market",
            ToolDefinition(
                name="get_polymarket_market",
                description=(
                    "Get detailed information about a specific Polymarket prediction market. "
                    "Includes current prices, orderbook depth, and market metadata. "
                    "Use this when the user wants to see details about a specific market."
                ),
                parameters=[
                    ToolParameter(
                        name="market_id",
                        type=ToolParameterType.STRING,
                        description="The market condition ID",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_polymarket_market,
        )

        self.register(
            "get_polymarket_portfolio",
            ToolDefinition(
                name="get_polymarket_portfolio",
                description=(
                    "Get user's Polymarket portfolio with positions and P&L. "
                    "Shows all open prediction market positions, cost basis, current value, and profit/loss."
                ),
                parameters=[],
            ),
            self._handle_get_polymarket_portfolio,
            requires_address=True,
        )

        self.register(
            "get_polymarket_quote",
            ToolDefinition(
                name="get_polymarket_quote",
                description=(
                    "Get a quote for buying or selling shares in a Polymarket prediction. "
                    "Returns the number of shares, average price, and potential profit. "
                    "Does NOT execute the trade - use this to show the user what they would get."
                ),
                parameters=[
                    ToolParameter(
                        name="market_id",
                        type=ToolParameterType.STRING,
                        description="The market condition ID",
                        required=True,
                    ),
                    ToolParameter(
                        name="outcome",
                        type=ToolParameterType.STRING,
                        description="The outcome to trade (e.g., 'Yes', 'No')",
                        required=True,
                    ),
                    ToolParameter(
                        name="side",
                        type=ToolParameterType.STRING,
                        description="BUY or SELL",
                        required=True,
                    ),
                    ToolParameter(
                        name="amount_usd",
                        type=ToolParameterType.NUMBER,
                        description="Amount in USDC to spend (for BUY) or shares to sell (for SELL)",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_polymarket_quote,
            requires_address=True,
        )

        self.register(
            "analyze_polymarket",
            ToolDefinition(
                name="analyze_polymarket",
                description=(
                    "Get AI analysis of a Polymarket prediction market. "
                    "Provides summary, key factors, sentiment, and potential recommendations. "
                    "Use this when the user wants insights before making a prediction."
                ),
                parameters=[
                    ToolParameter(
                        name="market_id",
                        type=ToolParameterType.STRING,
                        description="The market condition ID to analyze",
                        required=True,
                    ),
                ],
            ),
            self._handle_analyze_polymarket,
        )

        # =====================================================================
        # Polymarket Copy Trading Tools
        # =====================================================================

        self.register(
            "get_polymarket_top_traders",
            ToolDefinition(
                name="get_polymarket_top_traders",
                description=(
                    "Get top Polymarket traders by performance. "
                    "Shows leaderboard of most profitable prediction market traders. "
                    "Use this when the user wants to find traders to copy."
                ),
                parameters=[
                    ToolParameter(
                        name="sort_by",
                        type=ToolParameterType.STRING,
                        description="Metric to sort by: roi, pnl, win_rate, volume (default: roi)",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type=ToolParameterType.INTEGER,
                        description="Number of traders to return (default: 20)",
                        required=False,
                    ),
                ],
            ),
            self._handle_get_polymarket_top_traders,
        )

        self.register(
            "get_polymarket_trader_profile",
            ToolDefinition(
                name="get_polymarket_trader_profile",
                description=(
                    "Get detailed profile of a Polymarket trader. "
                    "Shows performance metrics, positions, risk score, and trading style. "
                    "Use this before deciding to copy a trader."
                ),
                parameters=[
                    ToolParameter(
                        name="address",
                        type=ToolParameterType.STRING,
                        description="Trader's wallet address",
                        required=True,
                    ),
                ],
            ),
            self._handle_get_polymarket_trader_profile,
        )

        self.register(
            "start_polymarket_copy",
            ToolDefinition(
                name="start_polymarket_copy",
                description=(
                    "Start copying a Polymarket trader. "
                    "Creates a copy relationship with configurable sizing and filters. "
                    "You'll be notified when the trader makes trades for approval."
                ),
                parameters=[
                    ToolParameter(
                        name="leader_address",
                        type=ToolParameterType.STRING,
                        description="Address of the trader to copy",
                        required=True,
                    ),
                    ToolParameter(
                        name="sizing_mode",
                        type=ToolParameterType.STRING,
                        description="How to size positions: percentage, fixed, proportional (default: percentage)",
                        required=False,
                    ),
                    ToolParameter(
                        name="size_value",
                        type=ToolParameterType.NUMBER,
                        description="Size value: percentage (e.g., 10 for 10%) or fixed USD amount",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_exposure_usd",
                        type=ToolParameterType.NUMBER,
                        description="Maximum total exposure in USD (default: 1000)",
                        required=False,
                    ),
                ],
            ),
            self._handle_start_polymarket_copy,
            requires_address=True,
        )

        self.register(
            "stop_polymarket_copy",
            ToolDefinition(
                name="stop_polymarket_copy",
                description=(
                    "Stop copying a Polymarket trader. "
                    "Deactivates the copy relationship but keeps existing positions."
                ),
                parameters=[
                    ToolParameter(
                        name="relationship_id",
                        type=ToolParameterType.STRING,
                        description="The copy relationship ID to stop",
                        required=True,
                    ),
                ],
            ),
            self._handle_stop_polymarket_copy,
            requires_address=True,
        )

        self.register(
            "list_polymarket_copy_relationships",
            ToolDefinition(
                name="list_polymarket_copy_relationships",
                description=(
                    "List all Polymarket traders the user is copying. "
                    "Shows relationship status, stats, and current exposure."
                ),
                parameters=[],
            ),
            self._handle_list_polymarket_copy_relationships,
            requires_address=True,
        )

        self.register(
            "get_pending_polymarket_copies",
            ToolDefinition(
                name="get_pending_polymarket_copies",
                description=(
                    "Get pending Polymarket copy trades awaiting approval. "
                    "Shows trades detected from leaders that need user action."
                ),
                parameters=[],
            ),
            self._handle_get_pending_polymarket_copies,
            requires_address=True,
        )

        self.register(
            "approve_polymarket_copy",
            ToolDefinition(
                name="approve_polymarket_copy",
                description=(
                    "Approve a pending Polymarket copy trade. "
                    "Gets a quote and prepares the transaction for signing."
                ),
                parameters=[
                    ToolParameter(
                        name="execution_id",
                        type=ToolParameterType.STRING,
                        description="The execution ID to approve",
                        required=True,
                    ),
                ],
            ),
            self._handle_approve_polymarket_copy,
            requires_address=True,
        )

        self.register(
            "reject_polymarket_copy",
            ToolDefinition(
                name="reject_polymarket_copy",
                description=(
                    "Reject a pending Polymarket copy trade. "
                    "Skips this trade without executing."
                ),
                parameters=[
                    ToolParameter(
                        name="execution_id",
                        type=ToolParameterType.STRING,
                        description="The execution ID to reject",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type=ToolParameterType.STRING,
                        description="Optional reason for rejection",
                        required=False,
                    ),
                ],
            ),
            self._handle_reject_polymarket_copy,
            requires_address=True,
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
        limit: int = 15,
        hours_back: int = 24,
    ) -> Dict[str, Any]:
        """Handle recent news fetch with source diversity."""
        from ...db import get_convex_client
        from ...services.news_fetcher.service import NewsFetcherService

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
                return {
                    "success": True,
                    "wallet_address": wallet_address,
                    "policy": None,
                    "is_default": False,
                    "policy_missing": True,
                    "message": "No risk policy configured. Draft a policy to enable autonomous execution.",
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

            # Get existing policy
            existing = await convex.query(
                "riskPolicies:getByWallet",
                {"walletAddress": wallet_address.lower()},
            )

            if not existing or not existing.get("config"):
                return {
                    "success": False,
                    "error": "Risk policy not configured. Create a full policy before updating.",
                    "policy_missing": True,
                }

            # Build config with existing values as base
            config = existing["config"].copy()

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

            if risk_policy_data and risk_policy_data.get("config"):
                risk_config = RiskPolicyConfig.from_dict(risk_policy_data["config"])
            else:
                return {
                    "success": True,
                    "approved": False,
                    "policy_missing": True,
                    "requires_approval": False,
                    "violations": [
                        {
                            "policyType": "risk",
                            "policyName": "risk_policy_missing",
                            "severity": "block",
                            "message": "No risk policy configured. Draft a policy to enable autonomous execution.",
                        }
                    ],
                    "warnings": [],
                }

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

    # =========================================================================
    # Solana Swap Handlers
    # =========================================================================

    async def _handle_get_solana_swap_quote(
        self,
        wallet_address: str,
        input_token: str,
        output_token: str,
        amount: float,
        slippage_bps: int = 50,
    ) -> Dict[str, Any]:
        """Handle Solana swap quote request via Jupiter."""
        from decimal import Decimal
        from ...providers.jupiter import (
            get_jupiter_swap_provider,
            JupiterQuoteError,
            JupiterSwapError,
            NATIVE_SOL_MINT,
            USDC_MINT,
            USDT_MINT,
        )
        from ..swap.constants import TOKEN_REGISTRY, SOLANA_CHAIN_ID

        try:
            jupiter = get_jupiter_swap_provider()

            # Resolve token symbols to mint addresses
            solana_tokens = TOKEN_REGISTRY.get(SOLANA_CHAIN_ID, {})

            def resolve_mint(token: str) -> str:
                """Resolve symbol to mint address or return as-is if already an address."""
                upper = token.upper()
                if upper in solana_tokens:
                    return str(solana_tokens[upper]['address'])
                # Check if it looks like a Solana address (32-44 chars, base58)
                if len(token) >= 32 and len(token) <= 44:
                    return token
                # Common aliases
                if upper in ('SOL', 'SOLANA'):
                    return NATIVE_SOL_MINT
                if upper == 'USDC':
                    return USDC_MINT
                if upper == 'USDT':
                    return USDT_MINT
                # Return as-is, let Jupiter handle the error
                return token

            input_mint = resolve_mint(input_token)
            output_mint = resolve_mint(output_token)

            # Get input token decimals
            input_decimals = 9  # default SOL decimals
            for symbol, meta in solana_tokens.items():
                if str(meta.get('address')) == input_mint:
                    input_decimals = int(meta.get('decimals', 9))
                    break

            # Convert amount to smallest units
            amount_decimal = Decimal(str(amount))
            amount_lamports = int(amount_decimal * (Decimal(10) ** input_decimals))

            if amount_lamports <= 0:
                return {
                    "success": False,
                    "error": "Amount must be greater than 0",
                }

            # Get quote from Jupiter
            quote = await jupiter.get_swap_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_lamports,
                slippage_bps=slippage_bps,
            )

            # Build swap transaction
            swap_result = await jupiter.build_swap_transaction(
                quote=quote,
                user_public_key=wallet_address,
            )

            # Get output token decimals
            output_decimals = 9
            for symbol, meta in solana_tokens.items():
                if str(meta.get('address')) == output_mint:
                    output_decimals = int(meta.get('decimals', 9))
                    break

            # Calculate human-readable amounts
            output_amount = Decimal(str(quote.out_amount)) / (Decimal(10) ** output_decimals)
            min_output_amount = Decimal(str(quote.other_amount_threshold)) / (Decimal(10) ** output_decimals)

            # Get token symbols
            input_symbol = input_token.upper()
            output_symbol = output_token.upper()
            if quote.input_token:
                input_symbol = quote.input_token.symbol
            if quote.output_token:
                output_symbol = quote.output_token.symbol

            return {
                "success": True,
                "chain": "solana",
                "provider": "jupiter",
                "input_token": {
                    "symbol": input_symbol,
                    "mint": input_mint,
                    "amount": str(amount),
                    "amount_lamports": amount_lamports,
                },
                "output_token": {
                    "symbol": output_symbol,
                    "mint": output_mint,
                    "amount_estimate": str(output_amount),
                    "min_amount": str(min_output_amount),
                },
                "price_impact_percent": quote.price_impact_pct,
                "slippage_bps": slippage_bps,
                "transaction": {
                    "swap_transaction_base64": swap_result.swap_transaction,
                    "last_valid_block_height": swap_result.last_valid_block_height,
                    "priority_fee_lamports": swap_result.priority_fee_lamports,
                    "compute_unit_limit": swap_result.compute_unit_limit,
                },
                "instructions": [
                    f"Swap {amount} {input_symbol} for ~{output_amount:.6f} {output_symbol} on Solana",
                    f"Minimum output: {min_output_amount:.6f} {output_symbol} (with {slippage_bps/100}% slippage)",
                    "Sign the transaction in your Solana wallet to execute the swap.",
                ],
            }

        except JupiterQuoteError as e:
            self.logger.warning(f"Jupiter quote error: {e}")
            return {
                "success": False,
                "error": f"Could not get swap quote: {e}",
                "hint": "Check that the token symbols are valid and you have sufficient balance.",
            }
        except JupiterSwapError as e:
            self.logger.warning(f"Jupiter swap build error: {e}")
            return {
                "success": False,
                "error": f"Could not build swap transaction: {e}",
            }
        except Exception as e:
            self.logger.error(f"Error in Solana swap quote: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # EVM Swap Handlers (via Relay)
    # =========================================================================

    async def _handle_get_swap_quote(
        self,
        wallet_address: str,
        input_token: str,
        output_token: str,
        amount: float,
        chain: str = "ethereum",
        slippage_percent: float = 0.5,
    ) -> Dict[str, Any]:
        """Handle EVM swap quote request via Relay."""
        from decimal import Decimal
        from ...providers.relay import RelayProvider
        from ..bridge.chain_registry import get_chain_registry
        from ..swap.constants import TOKEN_REGISTRY, TOKEN_ALIAS_MAP

        try:
            # Resolve chain name to chain ID
            registry = await get_chain_registry()
            chain_id = registry.get_chain_id(chain.lower())
            if chain_id is None:
                return {
                    "success": False,
                    "error": f"Unknown chain: {chain}",
                    "hint": f"Supported chains include: ethereum, base, arbitrum, optimism, polygon",
                }

            # For Solana, redirect to the Solana swap tool
            if isinstance(chain_id, str) and chain_id.lower() == "solana":
                return {
                    "success": False,
                    "error": "For Solana swaps, use get_solana_swap_quote instead",
                    "hint": "This tool is for EVM chains only. Use get_solana_swap_quote for Solana-to-Solana swaps.",
                }

            # Resolve token symbols to addresses
            def resolve_token(token: str, chain_id: int) -> Optional[Dict[str, Any]]:
                """Resolve token symbol to metadata."""
                chain_tokens = TOKEN_REGISTRY.get(chain_id, {})
                alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})

                # Check if it's already an address
                if token.startswith("0x") and len(token) == 42:
                    # Look up by address
                    for sym, meta in chain_tokens.items():
                        if str(meta.get("address", "")).lower() == token.lower():
                            return {"symbol": sym, **meta}
                    # Return as-is if not found in registry
                    return {"symbol": token[:8], "address": token, "decimals": 18, "is_native": False}

                # Resolve alias to symbol
                symbol = alias_map.get(token.lower(), token.upper())
                if symbol in chain_tokens:
                    return {"symbol": symbol, **chain_tokens[symbol]}

                # Common native token handling
                if token.lower() in ("eth", "ether", "native"):
                    return {
                        "symbol": "ETH",
                        "address": "0x0000000000000000000000000000000000000000",
                        "decimals": 18,
                        "is_native": True,
                    }

                return None

            input_meta = resolve_token(input_token, chain_id)
            output_meta = resolve_token(output_token, chain_id)

            if not input_meta:
                return {
                    "success": False,
                    "error": f"Unknown input token: {input_token}",
                    "hint": f"Try using the full contract address or a common symbol like ETH, USDC, USDT",
                }

            if not output_meta:
                return {
                    "success": False,
                    "error": f"Unknown output token: {output_token}",
                    "hint": f"Try using the full contract address or a common symbol like ETH, USDC, USDT",
                }

            # Convert amount to base units
            amount_decimal = Decimal(str(amount))
            decimals = int(input_meta.get("decimals", 18))
            amount_base_units = int(amount_decimal * (Decimal(10) ** decimals))

            if amount_base_units <= 0:
                return {
                    "success": False,
                    "error": "Amount must be greater than 0",
                }

            # Build Relay quote request
            relay = RelayProvider()
            relay_payload = {
                "user": wallet_address,
                "originChainId": chain_id,
                "destinationChainId": chain_id,  # Same chain for swap
                "originCurrency": input_meta["address"],
                "destinationCurrency": output_meta["address"],
                "recipient": wallet_address,
                "tradeType": "EXACT_INPUT",
                "amount": str(amount_base_units),
                "referrer": "sherpa.chat",
                "useExternalLiquidity": True,
            }

            quote = await relay.quote(relay_payload)

            # Parse quote response
            details = quote.get("details", {})
            output_decimals = int(output_meta.get("decimals", 18))

            # Get output amount
            currency_out = details.get("currencyOut", {})
            output_amount_raw = currency_out.get("amount", "0")
            output_amount = Decimal(output_amount_raw) / (Decimal(10) ** output_decimals)

            # Get fees
            total_fee_usd = Decimal(str(details.get("totalFeeUsd", "0")))

            # Get steps/transactions
            steps = quote.get("steps", [])

            chain_name = registry.get_chain_name(chain_id)

            return {
                "success": True,
                "provider": "relay",
                "chain": chain_name,
                "chain_id": chain_id,
                "input_token": {
                    "symbol": input_meta["symbol"],
                    "address": input_meta["address"],
                    "amount": str(amount),
                    "amount_base_units": str(amount_base_units),
                },
                "output_token": {
                    "symbol": output_meta["symbol"],
                    "address": output_meta["address"],
                    "amount_estimate": str(output_amount),
                },
                "fees": {
                    "total_usd": str(total_fee_usd),
                    "slippage_percent": slippage_percent,
                },
                "steps_count": len(steps),
                "quote_data": quote,  # Full quote for execution
                "instructions": [
                    f"Swap {amount} {input_meta['symbol']} for ~{output_amount:.6f} {output_meta['symbol']} on {chain_name}",
                    f"Estimated fees: ${total_fee_usd:.2f}",
                    "Review and sign the transaction in your wallet to execute the swap.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error in EVM swap quote: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Cross-Chain Bridge Handlers (via Relay)
    # =========================================================================

    async def _handle_get_bridge_quote(
        self,
        wallet_address: str,
        from_chain: str,
        to_chain: str,
        token: str,
        amount: float,
        destination_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle cross-chain bridge quote request via Relay."""
        from decimal import Decimal
        from ...providers.relay import RelayProvider
        from ..bridge.chain_registry import get_chain_registry
        from ..swap.constants import TOKEN_REGISTRY, NATIVE_SOL_MINT

        try:
            # Resolve chain names to chain IDs
            registry = await get_chain_registry()

            from_chain_id = registry.get_chain_id(from_chain.lower())
            to_chain_id = registry.get_chain_id(to_chain.lower())

            if from_chain_id is None:
                supported = registry.get_supported_chain_names(10)
                return {
                    "success": False,
                    "error": f"Unknown source chain: {from_chain}",
                    "hint": f"Supported chains include: {', '.join(supported)}",
                }

            if to_chain_id is None:
                supported = registry.get_supported_chain_names(10)
                return {
                    "success": False,
                    "error": f"Unknown destination chain: {to_chain}",
                    "hint": f"Supported chains include: {', '.join(supported)}",
                }

            # Same chain = not a bridge
            if from_chain_id == to_chain_id:
                return {
                    "success": False,
                    "error": "Source and destination chains are the same",
                    "hint": "For same-chain swaps, use get_swap_quote (EVM) or get_solana_swap_quote (Solana)",
                }

            # Determine if either chain is Solana
            from_is_solana = isinstance(from_chain_id, str) and from_chain_id.lower() == "solana"
            to_is_solana = isinstance(to_chain_id, str) and to_chain_id.lower() == "solana"

            # For cross-ecosystem bridges (EVM ↔ Solana), destination address is required
            if (from_is_solana or to_is_solana) and not destination_address:
                if from_is_solana and not to_is_solana:
                    return {
                        "success": False,
                        "error": "Destination EVM address required for Solana → EVM bridge",
                        "hint": "Please provide the destination_address parameter with the EVM wallet address (0x...)",
                    }
                elif to_is_solana and not from_is_solana:
                    return {
                        "success": False,
                        "error": "Destination Solana address required for EVM → Solana bridge",
                        "hint": "Please provide the destination_address parameter with the Solana wallet address",
                    }

            recipient = destination_address or wallet_address

            # Equivalent token mapping for bridged variants
            # Maps bridged token symbols to their canonical form
            EQUIVALENT_TOKENS = {
                "usdc.e": "usdc",
                "usdc.b": "usdc",
                "usdce": "usdc",
                "usdt.e": "usdt",
                "usdt.b": "usdt",
                "weth.e": "weth",
                "dai.e": "dai",
            }

            def get_token_address(token: str, chain_id, is_solana: bool, allow_equivalent: bool = False) -> tuple:
                """Get token address and decimals for a chain.

                Args:
                    token: Token symbol or address
                    chain_id: Target chain ID
                    is_solana: Whether the chain is Solana
                    allow_equivalent: If True, try equivalent tokens (e.g., USDC.e → USDC)

                Returns:
                    Tuple of (address, decimals) or (None, None) if not found
                """
                token_lower = token.lower()

                if is_solana:
                    # Solana token addresses
                    solana_tokens = {
                        "sol": (NATIVE_SOL_MINT, 9),
                        "usdc": ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
                        "usdt": ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", 6),
                    }
                    if token_lower in solana_tokens:
                        return solana_tokens[token_lower]
                    # Try equivalent token
                    if allow_equivalent and token_lower in EQUIVALENT_TOKENS:
                        equiv = EQUIVALENT_TOKENS[token_lower]
                        if equiv in solana_tokens:
                            return solana_tokens[equiv]
                    # Check if it's already a valid Solana address (base58, 32-44 chars)
                    if len(token) >= 32 and len(token) <= 44 and token[0].isalnum():
                        return (token, 9)
                    return (None, None)
                else:
                    # EVM token addresses
                    chain_tokens = TOKEN_REGISTRY.get(chain_id, {})
                    for sym, meta in chain_tokens.items():
                        if sym.lower() == token_lower or token_lower in [a.lower() for a in meta.get("aliases", [])]:
                            return (meta["address"], meta["decimals"])

                    # Try equivalent token (e.g., USDC.e → USDC)
                    if allow_equivalent and token_lower in EQUIVALENT_TOKENS:
                        equiv = EQUIVALENT_TOKENS[token_lower]
                        for sym, meta in chain_tokens.items():
                            if sym.lower() == equiv or equiv in [a.lower() for a in meta.get("aliases", [])]:
                                return (meta["address"], meta["decimals"])

                    # Native ETH
                    if token_lower in ("eth", "ether", "native"):
                        return ("0x0000000000000000000000000000000000000000", 18)

                    # Check if it's already a valid EVM address
                    if token.startswith("0x") and len(token) == 42:
                        return (token, 18)

                    return (None, None)

            # Resolve source token (exact match required)
            from_address, from_decimals = get_token_address(token, from_chain_id, from_is_solana, allow_equivalent=False)
            if from_address is None:
                return {
                    "success": False,
                    "error": f"Token '{token}' not found on source chain",
                    "hint": f"Make sure the token exists on the source chain",
                }

            # Resolve destination token (allow equivalent tokens for bridging)
            to_address, _ = get_token_address(token, to_chain_id, to_is_solana, allow_equivalent=True)
            if to_address is None:
                return {
                    "success": False,
                    "error": f"Token '{token}' (or equivalent) not found on destination chain",
                    "hint": f"The destination chain may not support this token",
                }

            # Convert amount to base units
            amount_decimal = Decimal(str(amount))
            amount_base_units = int(amount_decimal * (Decimal(10) ** from_decimals))

            if amount_base_units <= 0:
                return {
                    "success": False,
                    "error": "Amount must be greater than 0",
                }

            # Build Relay quote request
            from ..chain_types import RELAY_SOLANA_CHAIN_ID

            relay = RelayProvider()
            relay_payload = {
                "user": wallet_address,
                "originChainId": from_chain_id if not from_is_solana else RELAY_SOLANA_CHAIN_ID,
                "destinationChainId": to_chain_id if not to_is_solana else RELAY_SOLANA_CHAIN_ID,
                "originCurrency": from_address,
                "destinationCurrency": to_address,
                "recipient": recipient,
                "tradeType": "EXACT_INPUT",
                "amount": str(amount_base_units),
                "referrer": "sherpa.chat",
            }

            quote = await relay.quote(relay_payload)

            # Parse quote response
            details = quote.get("details", {})

            # Get output amount
            currency_out = details.get("currencyOut", {})
            output_amount_raw = currency_out.get("amount", "0")
            to_decimals = currency_out.get("decimals", from_decimals)
            output_amount = Decimal(output_amount_raw) / (Decimal(10) ** to_decimals)

            # Get fees
            total_fee_usd = Decimal(str(details.get("totalFeeUsd", "0")))

            # Get time estimate
            time_estimate = details.get("timeEstimate", 0)  # seconds

            # Get steps/transactions
            steps = quote.get("steps", [])

            from_chain_name = registry.get_chain_name(from_chain_id) if not from_is_solana else "Solana"
            to_chain_name = registry.get_chain_name(to_chain_id) if not to_is_solana else "Solana"

            return {
                "success": True,
                "provider": "relay",
                "bridge_type": "cross_chain",
                "from_chain": {
                    "name": from_chain_name,
                    "chain_id": from_chain_id,
                    "is_solana": from_is_solana,
                },
                "to_chain": {
                    "name": to_chain_name,
                    "chain_id": to_chain_id,
                    "is_solana": to_is_solana,
                },
                "token": {
                    "symbol": token.upper(),
                    "input_address": from_address,
                    "amount": str(amount),
                    "amount_base_units": str(amount_base_units),
                },
                "output": {
                    "amount_estimate": str(output_amount),
                    "recipient": recipient,
                },
                "fees": {
                    "total_usd": str(total_fee_usd),
                },
                "time_estimate_seconds": time_estimate,
                "steps_count": len(steps),
                "quote_data": quote,  # Full quote for execution
                "instructions": [
                    f"Bridge {amount} {token.upper()} from {from_chain_name} to {to_chain_name}",
                    f"Expected output: ~{output_amount:.6f} {token.upper()}",
                    f"Estimated fees: ${total_fee_usd:.2f}",
                    f"Estimated time: {time_estimate // 60} min {time_estimate % 60} sec" if time_estimate else "Time varies",
                    "Review and sign the transaction(s) in your wallet to execute the bridge.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error in bridge quote: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Copy Trading Handlers
    # =========================================================================

    async def _handle_get_top_traders(
        self,
        token_address: str,
        chain: str = "solana",
        time_frame: str = "7d",
        sort_by: str = "pnl",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Get top traders for a token via Birdeye."""
        try:
            from ...providers.birdeye import get_birdeye_provider

            birdeye = get_birdeye_provider()

            if not await birdeye.ready():
                return {
                    "success": False,
                    "error": "Birdeye API not configured. Please set BIRDEYE_API_KEY.",
                }

            result = await birdeye.get_top_traders_by_token(
                token_address=token_address,
                chain=chain,
                time_frame=time_frame,
                sort_by=sort_by,
                limit=min(limit, 20),
            )

            if "error" in result and result.get("error"):
                return {"success": False, "error": result["error"]}

            traders = result.get("traders", [])

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "time_frame": time_frame,
                "sort_by": sort_by,
                "total_found": result.get("total", len(traders)),
                "traders": [
                    {
                        "address": t.get("address"),
                        "pnl_usd": str(t.get("pnl_usd", 0)),
                        "volume_usd": str(t.get("volume_usd", 0)),
                        "trade_count": t.get("trade_count"),
                        "win_rate": t.get("win_rate"),
                    }
                    for t in traders
                ],
                "instructions": [
                    f"Found {len(traders)} top traders for this token over {time_frame}.",
                    "To copy a trader, use start_copy_trading with their address.",
                    "To analyze a trader first, use get_trader_profile.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error fetching top traders: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_trader_profile(
        self,
        wallet_address: str,
        chain: str = "solana",
    ) -> Dict[str, Any]:
        """Get detailed profile for a trader wallet."""
        try:
            from ...providers.birdeye import get_birdeye_provider

            birdeye = get_birdeye_provider()

            if not await birdeye.ready():
                return {
                    "success": False,
                    "error": "Birdeye API not configured. Please set BIRDEYE_API_KEY.",
                }

            # Fetch portfolio, PnL, and trade history in parallel
            import asyncio

            portfolio_task = birdeye.get_wallet_portfolio(wallet_address, chain)
            pnl_task = birdeye.get_wallet_pnl(wallet_address, chain)
            trades_task = birdeye.get_wallet_trade_history(wallet_address, chain, limit=20)

            portfolio, pnl, trades = await asyncio.gather(
                portfolio_task, pnl_task, trades_task
            )

            return {
                "success": True,
                "address": wallet_address,
                "chain": chain,
                "portfolio": {
                    "total_value_usd": str(portfolio.get("total_value_usd", 0)),
                    "token_count": portfolio.get("token_count", 0),
                    "top_holdings": [
                        {
                            "symbol": h.get("symbol"),
                            "value_usd": str(h.get("value_usd", 0)),
                        }
                        for h in portfolio.get("holdings", [])[:5]
                    ],
                },
                "performance": {
                    "total_pnl_usd": str(pnl.get("total_pnl_usd", 0)),
                    "realized_pnl_usd": str(pnl.get("realized_pnl_usd", 0)),
                    "unrealized_pnl_usd": str(pnl.get("unrealized_pnl_usd", 0)),
                    "win_rate": pnl.get("win_rate"),
                    "trade_count": pnl.get("trade_count"),
                },
                "recent_trades": [
                    {
                        "from_token": t.get("from_token"),
                        "to_token": t.get("to_token"),
                        "timestamp": t.get("timestamp").isoformat() if t.get("timestamp") else None,
                    }
                    for t in trades.get("trades", [])[:5]
                ],
                "instructions": [
                    "This trader's profile shows their portfolio, performance, and recent activity.",
                    "To start copying this trader, use start_copy_trading with their address.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error fetching trader profile: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_start_copy_trading(
        self,
        leader_address: str,
        leader_chain: str,
        follower_address: str,
        follower_chain: str,
        sizing_mode: str = "fixed",
        size_value: float = 100,
        max_trade_usd: float = 1000,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start copying a wallet."""
        try:
            from decimal import Decimal
            from ..copy_trading.manager import CopyTradingManager
            from ..copy_trading.models import CopyConfig, SizingMode
            from ...db import get_convex_client
            from ...services.events.service import get_event_monitoring_service

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated to start copy trading.",
                }

            # Get services
            convex = get_convex_client()
            event_service = get_event_monitoring_service()

            # Create manager
            manager = CopyTradingManager(convex_client=convex)

            # Create config
            config = CopyConfig(
                leader_address=leader_address,
                leader_chain=leader_chain,
                sizing_mode=SizingMode(sizing_mode),
                size_value=Decimal(str(size_value)),
                max_trade_usd=Decimal(str(max_trade_usd)),
            )

            # Start relationship
            relationship = await manager.start_copying(
                user_id=_user_id,
                follower_address=follower_address,
                follower_chain=follower_chain,
                config=config,
            )

            # Subscribe to leader wallet events
            from ...services.events.models import ChainType

            chain_type = ChainType(leader_chain.lower())
            await event_service.subscribe_address(
                address=leader_address,
                chain=chain_type,
                user_id=_user_id,
                label=f"Copy trading: {leader_address[:8]}...",
            )

            return {
                "success": True,
                "relationship_id": relationship.id,
                "leader": {
                    "address": leader_address,
                    "chain": leader_chain,
                },
                "follower": {
                    "address": follower_address,
                    "chain": follower_chain,
                },
                "config": {
                    "sizing_mode": sizing_mode,
                    "size_value": str(size_value),
                    "max_trade_usd": str(max_trade_usd),
                },
                "status": "active",
                "instructions": [
                    f"Now following {leader_address[:8]}... on {leader_chain}.",
                    "You'll receive notifications when they make trades.",
                    "Each trade requires your manual approval before execution.",
                    "Use list_copy_relationships to see all wallets you're following.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error starting copy trading: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_stop_copy_trading(
        self,
        relationship_id: str,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop copying a wallet."""
        try:
            from ..copy_trading.manager import CopyTradingManager
            from ...db import get_convex_client

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated.",
                }

            convex = get_convex_client()
            manager = CopyTradingManager(convex_client=convex)

            relationship = await manager.stop_copying(relationship_id)

            return {
                "success": True,
                "relationship_id": relationship_id,
                "status": "stopped",
                "message": f"Stopped copying {relationship.config.leader_address[:8]}...",
            }

        except Exception as e:
            self.logger.error(f"Error stopping copy trading: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_list_copy_relationships(
        self,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List user's copy trading relationships."""
        try:
            from ..copy_trading.manager import CopyTradingManager
            from ...db import get_convex_client

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated.",
                }

            convex = get_convex_client()
            manager = CopyTradingManager(convex_client=convex)

            relationships = await manager.get_relationships_for_user(_user_id)

            return {
                "success": True,
                "total": len(relationships),
                "active": sum(1 for r in relationships if r.is_active and not r.is_paused),
                "relationships": [
                    {
                        "id": r.id,
                        "leader_address": r.config.leader_address,
                        "leader_chain": r.config.leader_chain,
                        "is_active": r.is_active,
                        "is_paused": r.is_paused,
                        "total_trades": r.total_trades,
                        "successful_trades": r.successful_trades,
                        "total_volume_usd": str(r.total_volume_usd),
                    }
                    for r in relationships
                ],
            }

        except Exception as e:
            self.logger.error(f"Error listing copy relationships: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_pending_copy_trades(
        self,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get pending copy trade approvals."""
        try:
            from ..copy_trading.manager import CopyTradingManager
            from ...db import get_convex_client

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated.",
                }

            convex = get_convex_client()
            manager = CopyTradingManager(convex_client=convex)

            pending = await manager.get_pending_approvals(_user_id)

            return {
                "success": True,
                "total_pending": len(pending),
                "pending_trades": [
                    {
                        "execution_id": p.id,
                        "relationship_id": p.relationship_id,
                        "leader_address": p.signal.leader_address,
                        "action": p.signal.action.value if hasattr(p.signal.action, "value") else p.signal.action,
                        "token_in": p.signal.token_in_symbol,
                        "token_out": p.signal.token_out_symbol,
                        "value_usd": str(p.signal.value_usd) if p.signal.value_usd else None,
                        "calculated_size_usd": str(p.calculated_size_usd) if p.calculated_size_usd else None,
                        "timestamp": p.signal.timestamp.isoformat(),
                    }
                    for p in pending
                ],
                "instructions": [
                    f"You have {len(pending)} copy trade(s) pending approval.",
                    "Use approve_copy_trade to execute, or reject_copy_trade to skip.",
                ] if pending else ["No pending copy trades."],
            }

        except Exception as e:
            self.logger.error(f"Error fetching pending trades: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_approve_copy_trade(
        self,
        execution_id: str,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Approve and execute a pending copy trade."""
        try:
            from ..copy_trading.manager import CopyTradingManager
            from ..copy_trading.models import CopyExecutionStatus
            from ...db import get_convex_client

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated.",
                }

            convex = get_convex_client()
            manager = CopyTradingManager(convex_client=convex)

            execution = await manager.approve_execution(execution_id, _user_id)

            status = execution.status.value if hasattr(execution.status, "value") else str(execution.status)

            if execution.status == CopyExecutionStatus.COMPLETED:
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "status": status,
                    "tx_hash": execution.tx_hash,
                    "actual_size_usd": str(execution.actual_size_usd) if execution.actual_size_usd else None,
                    "message": "Copy trade executed successfully!",
                }
            elif execution.status == CopyExecutionStatus.EXPIRED:
                return {
                    "success": False,
                    "execution_id": execution_id,
                    "status": status,
                    "error": execution.error_message or "Trade expired",
                }
            else:
                return {
                    "success": False,
                    "execution_id": execution_id,
                    "status": status,
                    "error": execution.error_message or "Execution failed",
                }

        except Exception as e:
            self.logger.error(f"Error approving copy trade: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_reject_copy_trade(
        self,
        execution_id: str,
        reason: Optional[str] = None,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reject a pending copy trade."""
        try:
            from ..copy_trading.manager import CopyTradingManager
            from ...db import get_convex_client

            if not _user_id:
                return {
                    "success": False,
                    "error": "User must be authenticated.",
                }

            convex = get_convex_client()
            manager = CopyTradingManager(convex_client=convex)

            execution = await manager.reject_execution(execution_id, _user_id, reason)

            return {
                "success": True,
                "execution_id": execution_id,
                "status": "rejected",
                "reason": reason or "User rejected",
                "message": "Copy trade skipped.",
            }

        except Exception as e:
            self.logger.error(f"Error rejecting copy trade: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Polymarket Handlers
    # =========================================================================

    async def _handle_get_polymarket_markets(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
        trending: bool = False,
        closing_soon_hours: Optional[int] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get Polymarket prediction markets."""
        try:
            from ..polymarket.trading import get_polymarket_trading_service

            service = get_polymarket_trading_service()
            markets = await service.get_markets(
                category=category,
                query=query,
                trending=trending,
                closing_soon_hours=closing_soon_hours,
                limit=limit,
            )

            return {
                "success": True,
                "count": len(markets),
                "markets": [
                    {
                        "market_id": m.condition_id,
                        "question": m.question,
                        "outcomes": m.outcomes,
                        "prices": {
                            t.outcome: f"{float(t.price)*100:.1f}%"
                            for t in m.tokens
                        },
                        "volume_usd": float(m.volume),
                        "volume_24h_usd": float(m.volume_24h),
                        "end_date": m.end_date.isoformat() if m.end_date else None,
                        "active": m.active,
                        "tags": m.tags[:3],
                    }
                    for m in markets
                ],
            }

        except Exception as e:
            self.logger.error(f"Error fetching Polymarket markets: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_polymarket_market(
        self,
        market_id: str,
    ) -> Dict[str, Any]:
        """Get detailed Polymarket market info."""
        try:
            from ..polymarket.trading import get_polymarket_trading_service

            service = get_polymarket_trading_service()
            details = await service.get_market_details(market_id)

            if not details:
                return {"success": False, "error": "Market not found"}

            market = details["market"]
            orderbooks = details.get("orderbooks", {})
            spreads = details.get("spreads", {})

            return {
                "success": True,
                "market": {
                    "market_id": market.condition_id,
                    "question": market.question,
                    "description": market.description,
                    "outcomes": market.outcomes,
                    "prices": {
                        t.outcome: {
                            "price": float(t.price),
                            "probability": f"{float(t.price)*100:.1f}%",
                            "token_id": t.token_id,
                        }
                        for t in market.tokens
                    },
                    "volume_usd": float(market.volume),
                    "volume_24h_usd": float(market.volume_24h),
                    "liquidity_usd": float(market.liquidity),
                    "end_date": market.end_date.isoformat() if market.end_date else None,
                    "active": market.active,
                    "resolved": market.resolved,
                    "tags": market.tags,
                },
                "orderbook_depth": {
                    outcome: {
                        "best_bid": float(ob.best_bid) if ob.best_bid else None,
                        "best_ask": float(ob.best_ask) if ob.best_ask else None,
                        "spread": float(ob.spread) if ob.spread else None,
                        "bid_levels": len(ob.bids),
                        "ask_levels": len(ob.asks),
                    }
                    for outcome, ob in orderbooks.items()
                },
            }

        except Exception as e:
            self.logger.error(f"Error fetching Polymarket market: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_polymarket_portfolio(
        self,
        _wallet_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get user's Polymarket portfolio."""
        try:
            from ..polymarket.trading import get_polymarket_trading_service

            if not _wallet_address:
                return {"success": False, "error": "Wallet address required"}

            service = get_polymarket_trading_service()
            portfolio = await service.get_portfolio(_wallet_address)

            return {
                "success": True,
                "portfolio": {
                    "address": portfolio.address,
                    "total_value_usd": float(portfolio.total_value),
                    "total_cost_basis_usd": float(portfolio.total_cost_basis),
                    "total_pnl_usd": float(portfolio.total_pnl),
                    "total_pnl_pct": portfolio.total_pnl_pct,
                    "open_positions": portfolio.open_positions_count,
                    "winning_positions": portfolio.winning_positions,
                    "losing_positions": portfolio.losing_positions,
                },
                "positions": [
                    {
                        "market_question": p.market_question,
                        "outcome": p.outcome_name,
                        "shares": float(p.size),
                        "avg_price": float(p.avg_price),
                        "current_price": float(p.current_price),
                        "value_usd": float(p.current_value),
                        "cost_basis_usd": float(p.cost_basis),
                        "pnl_usd": float(p.unrealized_pnl),
                        "pnl_pct": p.unrealized_pnl_pct,
                        "market_end_date": p.market_end_date.isoformat() if p.market_end_date else None,
                    }
                    for p in portfolio.positions
                ],
            }

        except Exception as e:
            self.logger.error(f"Error fetching Polymarket portfolio: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_polymarket_quote(
        self,
        market_id: str,
        outcome: str,
        side: str,
        amount_usd: float,
        _wallet_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a quote for a Polymarket trade."""
        try:
            from decimal import Decimal
            from ..polymarket.trading import get_polymarket_trading_service
            from ...providers.polymarket.models import OrderSide

            service = get_polymarket_trading_service()

            side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

            if side_enum == OrderSide.BUY:
                quote = await service.get_buy_quote(
                    market_id=market_id,
                    outcome=outcome,
                    amount_usd=Decimal(str(amount_usd)),
                )
            else:
                quote = await service.get_sell_quote(
                    market_id=market_id,
                    outcome=outcome,
                    shares=Decimal(str(amount_usd)),  # For sell, amount is shares
                    address=_wallet_address,
                )

            if not quote:
                return {"success": False, "error": "Could not generate quote"}

            result = {
                "success": True,
                "quote": {
                    "market_id": quote.market_id,
                    "outcome": quote.outcome_name,
                    "side": quote.side.value,
                    "amount_usd": float(quote.amount_usd),
                    "shares": float(quote.shares),
                    "avg_price": float(quote.avg_price),
                    "price_impact_pct": quote.price_impact_pct,
                },
            }

            if side_enum == OrderSide.BUY:
                result["quote"]["max_payout_usd"] = float(quote.max_payout) if quote.max_payout else None
                result["quote"]["potential_profit_usd"] = float(quote.potential_profit) if quote.potential_profit else None
                result["quote"]["potential_profit_pct"] = quote.potential_profit_pct

            result["instructions"] = [
                f"{'Buying' if side_enum == OrderSide.BUY else 'Selling'} {float(quote.shares):.2f} shares of {outcome}",
                f"Average price: ${float(quote.avg_price):.4f} per share",
                f"Price impact: {quote.price_impact_pct:.2f}%",
            ]

            if quote.max_payout:
                result["instructions"].append(
                    f"If {outcome} wins, you'll receive ${float(quote.max_payout):.2f} (potential profit: ${float(quote.potential_profit):.2f})"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error getting Polymarket quote: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_analyze_polymarket(
        self,
        market_id: str,
    ) -> Dict[str, Any]:
        """Analyze a Polymarket market."""
        try:
            from ..polymarket.trading import get_polymarket_trading_service

            service = get_polymarket_trading_service()
            analysis = await service.analyze_market(market_id)

            if not analysis:
                return {"success": False, "error": "Market not found"}

            return {
                "success": True,
                "analysis": {
                    "market_id": analysis.market_id,
                    "question": analysis.question,
                    "current_prices": {
                        "yes": f"{float(analysis.current_yes_price)*100:.1f}%",
                        "no": f"{float(analysis.current_no_price)*100:.1f}%",
                    },
                    "summary": analysis.summary,
                    "key_factors": analysis.key_factors,
                    "sentiment": analysis.sentiment,
                    "confidence": analysis.confidence,
                    "volume_trend": analysis.volume_trend,
                    "recommendation": {
                        "side": analysis.recommended_side,
                        "reason": analysis.recommended_reason,
                    } if analysis.recommended_side else None,
                    "analyzed_at": analysis.analyzed_at.isoformat(),
                },
            }

        except Exception as e:
            self.logger.error(f"Error analyzing Polymarket market: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Polymarket Copy Trading Handlers
    # =========================================================================

    async def _handle_get_polymarket_top_traders(
        self,
        sort_by: str = "roi",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get top Polymarket traders."""
        try:
            from ...services.polymarket_analytics import get_leaderboard

            leaderboard = get_leaderboard()
            entries = await leaderboard.get_leaderboard(
                sort_by=sort_by,
                limit=limit,
                min_trades=10,
            )

            return {
                "success": True,
                "count": len(entries),
                "sort_by": sort_by,
                "traders": [
                    {
                        "rank": e.rank,
                        "address": e.address,
                        "total_pnl_usd": float(e.total_pnl_usd),
                        "roi_pct": e.roi_pct,
                        "win_rate": e.win_rate,
                        "total_volume_usd": float(e.total_volume_usd),
                        "active_positions": e.active_positions,
                        "total_trades": e.total_trades,
                        "follower_count": e.follower_count,
                    }
                    for e in entries
                ],
                "instructions": [
                    "Use get_polymarket_trader_profile to see detailed stats for a specific trader.",
                    "Use start_polymarket_copy to begin copying a trader.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting top Polymarket traders: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_polymarket_trader_profile(
        self,
        address: str,
    ) -> Dict[str, Any]:
        """Get detailed Polymarket trader profile."""
        try:
            from ...services.polymarket_analytics import get_trader_tracker

            tracker = get_trader_tracker()
            profile = await tracker.get_trader_profile(address)

            return {
                "success": True,
                "profile": {
                    "address": profile.address,
                    "is_experienced": profile.is_experienced,
                    "is_profitable": profile.is_profitable,
                    "performance": {
                        "total_pnl_usd": float(profile.metrics.total_pnl_usd),
                        "roi_pct": profile.metrics.roi_pct,
                        "win_rate": profile.metrics.win_rate,
                        "total_trades": profile.metrics.total_trades,
                        "total_volume_usd": float(profile.metrics.total_volume_usd),
                        "brier_score": profile.metrics.brier_score,
                    },
                    "current_state": {
                        "active_positions": profile.active_positions,
                        "total_exposure_usd": float(profile.total_exposure_usd),
                        "avg_position_size_usd": float(profile.avg_position_size_usd),
                    },
                    "trading_style": {
                        "preferred_categories": profile.preferred_categories,
                        "avg_hold_time_days": profile.avg_hold_time_days,
                        "trades_per_week": profile.trades_per_week,
                    },
                    "risk": {
                        "risk_score": profile.risk_score,
                        "diversification_score": profile.diversification_score,
                        "max_single_bet_pct": profile.max_single_bet_pct,
                    },
                    "social": {
                        "follower_count": profile.follower_count,
                        "total_copied_volume_usd": float(profile.total_copied_volume_usd),
                    },
                    "last_trade_at": profile.last_trade_at.isoformat() if profile.last_trade_at else None,
                },
                "instructions": [
                    "Use start_polymarket_copy to begin copying this trader.",
                ] if profile.is_experienced else [
                    "This trader has limited history. Consider waiting for more data.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting Polymarket trader profile: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_start_polymarket_copy(
        self,
        leader_address: str,
        sizing_mode: str = "percentage",
        size_value: float = 10.0,
        max_exposure_usd: float = 1000.0,
        _wallet_address: Optional[str] = None,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start copying a Polymarket trader."""
        try:
            from decimal import Decimal
            from ..polymarket.copy_trading import (
                get_polymarket_copy_manager,
                PolymarketCopyConfig,
                PMSizingMode,
            )

            if not _wallet_address or not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            # Parse sizing mode
            mode_map = {
                "percentage": PMSizingMode.PERCENTAGE,
                "fixed": PMSizingMode.FIXED,
                "proportional": PMSizingMode.PROPORTIONAL,
            }
            mode = mode_map.get(sizing_mode.lower(), PMSizingMode.PERCENTAGE)

            config = PolymarketCopyConfig(
                leaderAddress=leader_address,
                sizingMode=mode,
                sizeValue=Decimal(str(size_value)),
                maxExposureUsd=Decimal(str(max_exposure_usd)),
            )

            manager = get_polymarket_copy_manager()
            relationship = await manager.start_copying(
                user_id=_user_id,
                follower_address=_wallet_address,
                config=config,
            )

            return {
                "success": True,
                "relationship_id": relationship.id,
                "leader_address": leader_address,
                "sizing": f"{size_value}% of leader's position" if mode == PMSizingMode.PERCENTAGE else f"${size_value} per trade",
                "max_exposure": f"${max_exposure_usd}",
                "message": f"Now copying {leader_address}. You'll be notified when they make trades.",
                "instructions": [
                    "You will receive pending approvals when this trader makes trades.",
                    "Use get_pending_polymarket_copies to see trades awaiting approval.",
                    "Use approve_polymarket_copy or reject_polymarket_copy to handle them.",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error starting Polymarket copy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_stop_polymarket_copy(
        self,
        relationship_id: str,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop copying a Polymarket trader."""
        try:
            from ..polymarket.copy_trading import get_polymarket_copy_manager

            if not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            manager = get_polymarket_copy_manager()
            relationship = await manager.stop_copying(relationship_id)

            return {
                "success": True,
                "relationship_id": relationship_id,
                "leader_address": relationship.config.leader_address,
                "message": "Stopped copying this trader. Your existing positions remain unchanged.",
            }

        except Exception as e:
            self.logger.error(f"Error stopping Polymarket copy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_list_polymarket_copy_relationships(
        self,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List Polymarket copy relationships."""
        try:
            from ..polymarket.copy_trading import get_polymarket_copy_manager

            if not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            manager = get_polymarket_copy_manager()
            relationships = await manager.get_relationships_for_user(_user_id)

            return {
                "success": True,
                "count": len(relationships),
                "relationships": [
                    {
                        "relationship_id": r.id,
                        "leader_address": r.config.leader_address,
                        "is_active": r.is_active,
                        "is_paused": r.is_paused,
                        "pause_reason": r.pause_reason,
                        "sizing": f"{r.config.size_value}% ({r.config.sizing_mode.value})",
                        "stats": {
                            "total_copied": r.total_copied_positions,
                            "successful": r.successful_copies,
                            "failed": r.failed_copies,
                            "skipped": r.skipped_copies,
                            "total_volume_usd": float(r.total_volume_usd),
                        },
                        "current_exposure_usd": float(r.current_exposure_usd),
                        "max_exposure_usd": float(r.config.max_exposure_usd),
                        "last_copy_at": r.last_copy_at.isoformat() if r.last_copy_at else None,
                    }
                    for r in relationships
                ],
            }

        except Exception as e:
            self.logger.error(f"Error listing Polymarket copy relationships: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_get_pending_polymarket_copies(
        self,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get pending Polymarket copy executions."""
        try:
            from ..polymarket.copy_trading import get_polymarket_copy_manager

            if not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            manager = get_polymarket_copy_manager()
            pending = await manager.get_pending_approvals(_user_id)

            return {
                "success": True,
                "count": len(pending),
                "pending_copies": [
                    {
                        "execution_id": p.id,
                        "leader_address": p.leader_address,
                        "action": p.leader_action,
                        "market_question": p.market_question,
                        "outcome": p.outcome,
                        "leader_value_usd": float(p.leader_value_usd),
                        "your_calculated_value_usd": float(p.calculated_value_usd) if p.calculated_value_usd else None,
                        "detected_at": p.detected_at.isoformat(),
                        "expires_at": p.expires_at.isoformat() if p.expires_at else None,
                    }
                    for p in pending
                ],
                "instructions": [
                    f"You have {len(pending)} pending copy trade(s).",
                    "Use approve_polymarket_copy to execute, or reject_polymarket_copy to skip.",
                ] if pending else ["No pending copy trades."],
            }

        except Exception as e:
            self.logger.error(f"Error getting pending Polymarket copies: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_approve_polymarket_copy(
        self,
        execution_id: str,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Approve a pending Polymarket copy execution."""
        try:
            from ..polymarket.copy_trading import get_polymarket_copy_manager

            if not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            manager = get_polymarket_copy_manager()
            execution = await manager.approve_execution(execution_id, _user_id)

            result = {
                "success": True,
                "execution_id": execution_id,
                "status": execution.status.value,
                "market_question": execution.market_question,
                "outcome": execution.outcome,
                "action": execution.leader_action,
            }

            if execution.quote:
                result["quote"] = {
                    "shares": float(execution.quote.shares),
                    "avg_price": float(execution.quote.avg_price),
                    "amount_usd": float(execution.quote.amount_usd),
                    "price_impact_pct": execution.quote.price_impact_pct,
                }
                if execution.quote.potential_profit:
                    result["quote"]["potential_profit_usd"] = float(execution.quote.potential_profit)
                    result["quote"]["potential_profit_pct"] = execution.quote.potential_profit_pct

            result["instructions"] = [
                "Quote ready. Please sign the transaction in your wallet to complete.",
                "The trade will be executed on Polymarket.",
            ]

            return result

        except Exception as e:
            self.logger.error(f"Error approving Polymarket copy: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_reject_polymarket_copy(
        self,
        execution_id: str,
        reason: Optional[str] = None,
        _user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reject a pending Polymarket copy execution."""
        try:
            from ..polymarket.copy_trading import get_polymarket_copy_manager

            if not _user_id:
                return {"success": False, "error": "User must be authenticated"}

            manager = get_polymarket_copy_manager()
            execution = await manager.reject_execution(execution_id, _user_id, reason)

            return {
                "success": True,
                "execution_id": execution_id,
                "status": "rejected",
                "reason": reason or "User rejected",
                "message": "Copy trade skipped.",
            }

        except Exception as e:
            self.logger.error(f"Error rejecting Polymarket copy: {e}")
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
