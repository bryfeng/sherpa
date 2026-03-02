"""Copy trading tool handlers: discover traders, manage copy relationships, approve/reject trades."""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


@tool_spec(
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
)
async def handle_get_top_traders(
    token_address: str,
    chain: str = "solana",
    time_frame: str = "7d",
    sort_by: str = "pnl",
    limit: int = 10,
) -> Dict[str, Any]:
    """Get top traders for a token via Birdeye."""
    try:
        from ....providers.birdeye import get_birdeye_provider

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
        logger.error(f"Error fetching top traders: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
)
async def handle_get_trader_profile(
    wallet_address: str,
    chain: str = "solana",
) -> Dict[str, Any]:
    """Get detailed profile for a trader wallet."""
    try:
        from ....providers.birdeye import get_birdeye_provider

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
        logger.error(f"Error fetching trader profile: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_start_copy_trading(
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
        from ...copy_trading.manager import CopyTradingManager
        from ...copy_trading.models import CopyConfig, SizingMode
        from ....db import get_convex_client
        from ....services.events.service import get_event_monitoring_service

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
        from ....services.events.models import ChainType

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
        logger.error(f"Error starting copy trading: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_stop_copy_trading(
    relationship_id: str,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Stop copying a wallet."""
    try:
        from ...copy_trading.manager import CopyTradingManager
        from ....db import get_convex_client

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
        logger.error(f"Error stopping copy trading: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="list_copy_relationships",
    description=(
        "List all copy trading relationships for the user. "
        "Shows which wallets the user is currently following. "
        "Use this when the user asks 'who am I copying', 'show my copy trades', "
        "or wants to see their copy trading status."
    ),
    parameters=[],
    requires_address=True,
)
async def handle_list_copy_relationships(
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List user's copy trading relationships."""
    try:
        from ...copy_trading.manager import CopyTradingManager
        from ....db import get_convex_client

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
        logger.error(f"Error listing copy relationships: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="get_pending_copy_trades",
    description=(
        "Get copy trades waiting for user approval. "
        "Returns trades that were detected from followed wallets but need manual approval. "
        "Use this when the user asks about pending trades or copy trade notifications."
    ),
    parameters=[],
    requires_address=True,
)
async def handle_get_pending_copy_trades(
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get pending copy trade approvals."""
    try:
        from ...copy_trading.manager import CopyTradingManager
        from ....db import get_convex_client

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
        logger.error(f"Error fetching pending trades: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_approve_copy_trade(
    execution_id: str,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve and execute a pending copy trade."""
    try:
        from ...copy_trading.manager import CopyTradingManager
        from ...copy_trading.models import CopyExecutionStatus
        from ....db import get_convex_client

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
        logger.error(f"Error approving copy trade: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_reject_copy_trade(
    execution_id: str,
    reason: Optional[str] = None,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Reject a pending copy trade."""
    try:
        from ...copy_trading.manager import CopyTradingManager
        from ....db import get_convex_client

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
        logger.error(f"Error rejecting copy trade: {e}")
        return {"success": False, "error": str(e)}
