"""Polymarket tool handlers: markets, portfolio, quotes, analysis, and copy trading."""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# =========================================================================
# Polymarket Market Tools
# =========================================================================


@tool_spec(
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
)
async def handle_get_polymarket_markets(
    category: Optional[str] = None,
    query: Optional[str] = None,
    trending: bool = False,
    closing_soon_hours: Optional[int] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get Polymarket prediction markets."""
    try:
        from ...polymarket.trading import get_polymarket_trading_service

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
        logger.error(f"Error fetching Polymarket markets: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
)
async def handle_get_polymarket_market(
    market_id: str,
) -> Dict[str, Any]:
    """Get detailed Polymarket market info."""
    try:
        from ...polymarket.trading import get_polymarket_trading_service

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
        logger.error(f"Error fetching Polymarket market: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="get_polymarket_portfolio",
    description=(
        "Get user's Polymarket portfolio with positions and P&L. "
        "Shows all open prediction market positions, cost basis, current value, and profit/loss."
    ),
    parameters=[],
    requires_address=True,
)
async def handle_get_polymarket_portfolio(
    _wallet_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Get user's Polymarket portfolio."""
    try:
        from ...polymarket.trading import get_polymarket_trading_service

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
        logger.error(f"Error fetching Polymarket portfolio: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_get_polymarket_quote(
    market_id: str,
    outcome: str,
    side: str,
    amount_usd: float,
    _wallet_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a quote for a Polymarket trade."""
    try:
        from decimal import Decimal
        from ...polymarket.trading import get_polymarket_trading_service
        from ....providers.polymarket.models import OrderSide

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
        logger.error(f"Error getting Polymarket quote: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
)
async def handle_analyze_polymarket(
    market_id: str,
) -> Dict[str, Any]:
    """Analyze a Polymarket market."""
    try:
        from ...polymarket.trading import get_polymarket_trading_service

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
        logger.error(f"Error analyzing Polymarket market: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# Polymarket Copy Trading Tools
# =========================================================================


@tool_spec(
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
)
async def handle_get_polymarket_top_traders(
    sort_by: str = "roi",
    limit: int = 20,
) -> Dict[str, Any]:
    """Get top Polymarket traders."""
    try:
        from ....services.polymarket_analytics import get_leaderboard

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
        logger.error(f"Error getting top Polymarket traders: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
)
async def handle_get_polymarket_trader_profile(
    address: str,
) -> Dict[str, Any]:
    """Get detailed Polymarket trader profile."""
    try:
        from ....services.polymarket_analytics import get_trader_tracker

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
        logger.error(f"Error getting Polymarket trader profile: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_start_polymarket_copy(
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
        from ...polymarket.copy_trading import (
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
        logger.error(f"Error starting Polymarket copy: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_stop_polymarket_copy(
    relationship_id: str,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Stop copying a Polymarket trader."""
    try:
        from ...polymarket.copy_trading import get_polymarket_copy_manager

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
        logger.error(f"Error stopping Polymarket copy: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="list_polymarket_copy_relationships",
    description=(
        "List all Polymarket traders the user is copying. "
        "Shows relationship status, stats, and current exposure."
    ),
    parameters=[],
    requires_address=True,
)
async def handle_list_polymarket_copy_relationships(
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List Polymarket copy relationships."""
    try:
        from ...polymarket.copy_trading import get_polymarket_copy_manager

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
        logger.error(f"Error listing Polymarket copy relationships: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="get_pending_polymarket_copies",
    description=(
        "Get pending Polymarket copy trades awaiting approval. "
        "Shows trades detected from leaders that need user action."
    ),
    parameters=[],
    requires_address=True,
)
async def handle_get_pending_polymarket_copies(
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get pending Polymarket copy executions."""
    try:
        from ...polymarket.copy_trading import get_polymarket_copy_manager

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
        logger.error(f"Error getting pending Polymarket copies: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_approve_polymarket_copy(
    execution_id: str,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve a pending Polymarket copy execution."""
    try:
        from ...polymarket.copy_trading import get_polymarket_copy_manager

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
        logger.error(f"Error approving Polymarket copy: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
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
    requires_address=True,
)
async def handle_reject_polymarket_copy(
    execution_id: str,
    reason: Optional[str] = None,
    _user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Reject a pending Polymarket copy execution."""
    try:
        from ...polymarket.copy_trading import get_polymarket_copy_manager

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
        logger.error(f"Error rejecting Polymarket copy: {e}")
        return {"success": False, "error": str(e)}
