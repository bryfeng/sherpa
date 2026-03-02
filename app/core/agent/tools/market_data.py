"""Market-data tool handlers: token charts, trending tokens, TVL data."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# get_token_chart
# ---------------------------------------------------------------------------

@tool_spec(
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
    requires_address=False,
)
async def handle_get_token_chart(
    token_identifier: str,
    range: str = "7d",
    chain: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle token chart fetch."""
    from ....services.token_chart import get_token_chart as fetch_token_chart
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


# ---------------------------------------------------------------------------
# get_trending_tokens
# ---------------------------------------------------------------------------

@tool_spec(
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
    requires_address=False,
)
async def handle_get_trending_tokens(
    limit: int = 10,
    focus_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle trending tokens fetch."""
    from ....services.trending import get_trending_tokens

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


# ---------------------------------------------------------------------------
# get_tvl_data
# ---------------------------------------------------------------------------

@tool_spec(
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
    requires_address=False,
)
async def handle_get_tvl_data(
    protocol: str,
    window: str = "7d",
) -> Dict[str, Any]:
    """Handle TVL data fetch."""
    from ....tools.defillama import get_tvl_series

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
