"""Portfolio-related tool handlers: portfolio fetch, wallet history, chain settings."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


@tool_spec(
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
    requires_address=True,
)
async def handle_get_portfolio(
    wallet_address: str,
    chain: str = "ethereum",
) -> Dict[str, Any]:
    """Handle portfolio fetch."""
    from ....tools.portfolio import get_portfolio

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


@tool_spec(
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
    requires_address=True,
)
async def handle_get_wallet_history(
    wallet_address: str,
    chain: str = "ethereum",
    limit: Optional[int] = 50,
    days: Optional[int] = None,
) -> Dict[str, Any]:
    """Handle wallet history fetch."""
    from ....services.activity_summary import get_history_snapshot
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


@tool_spec(
    name="update_portfolio_chains",
    description=(
        "Update which blockchain networks are included in the user's portfolio view. "
        "Use this when the user wants to add or remove chains from their portfolio, "
        "enable or disable specific networks, or change their portfolio chain settings. "
        "Examples: 'Add Base to my portfolio', 'Remove Polygon from portfolio view', "
        "'Only show Ethereum holdings', 'Enable Arbitrum in my portfolio'."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The wallet address to update portfolio chains for",
            required=True,
        ),
        ToolParameter(
            name="chains",
            type=ToolParameterType.ARRAY,
            description=(
                "List of chain identifiers to enable in the portfolio view. "
                "Valid chains: ethereum, optimism, bnb, polygon, zksync, morph, base, "
                "arbitrum, avalanche, ink, linea, blast, scroll, zora. "
                "This replaces all current settings - include all chains the user wants enabled."
            ),
            required=True,
        ),
    ],
    requires_address=True,
)
async def handle_update_portfolio_chains(
    wallet_address: str,
    chains: List[str],
) -> Dict[str, Any]:
    """Handle updating portfolio chain settings for a wallet."""
    from ....db import get_convex_client

    # Valid chain identifiers
    valid_chains = {
        "ethereum", "optimism", "bnb", "polygon", "zksync", "morph",
        "base", "arbitrum", "avalanche", "ink", "linea", "blast",
        "scroll", "zora", "solana"
    }

    try:
        # Validate chains
        if not chains or len(chains) == 0:
            return {
                "success": False,
                "error": "At least one chain must be enabled",
            }

        # Normalize chain names (lowercase, strip whitespace)
        normalized_chains = [c.lower().strip() for c in chains]

        # Validate each chain
        invalid_chains = [c for c in normalized_chains if c not in valid_chains]
        if invalid_chains:
            return {
                "success": False,
                "error": f"Invalid chain(s): {', '.join(invalid_chains)}. Valid chains are: {', '.join(sorted(valid_chains))}",
            }

        convex = get_convex_client()

        # Update the user preferences in Convex
        await convex.mutation(
            "userPreferences:setEnabledChains",
            {
                "walletAddress": wallet_address.lower(),
                "chains": normalized_chains,
            },
        )

        return {
            "success": True,
            "wallet_address": wallet_address,
            "enabled_chains": normalized_chains,
            "message": f"Portfolio now showing {len(normalized_chains)} chain(s): {', '.join(normalized_chains)}",
        }

    except Exception as e:
        logger.error(f"Error updating portfolio chains: {e}")
        return {"success": False, "error": str(e)}
