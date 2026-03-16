"""Selective tool loading — keyword-based routing to tool groups.

Only sends relevant tool definitions to the LLM, reducing input tokens
by 50-80% on most queries.
"""

from enum import Enum
from typing import Dict, FrozenSet, List, Set, Tuple


class ToolGroup(str, Enum):
    CORE = "core"
    TRADING = "trading"
    STRATEGY = "strategy"
    POLICY = "policy"
    COPY_TRADING = "copy_trading"
    POLYMARKET = "polymarket"


# (group, tool_names, activation_keywords)
# CORE has no keywords — it is always included.
TOOL_GROUPS: List[Tuple[ToolGroup, FrozenSet[str], FrozenSet[str]]] = [
    (
        ToolGroup.CORE,
        frozenset({
            "get_portfolio",
            "get_wallet_history",
            "update_portfolio_chains",
            "get_token_chart",
            "get_trending_tokens",
            "get_tvl_data",
            "get_news",
            "get_personalized_news",
            "get_token_news",
        }),
        frozenset(),  # always active
    ),
    (
        ToolGroup.TRADING,
        frozenset({
            "get_swap_quote",
            "get_bridge_quote",
            "get_solana_swap_quote",
            "execute_transfer",
        }),
        frozenset({
            "swap", "bridge", "transfer", "send", "buy", "sell",
            "trade", "convert", "exchange", "quote",
        }),
    ),
    (
        ToolGroup.STRATEGY,
        frozenset({
            "list_strategies",
            "get_strategy",
            "create_strategy",
            "pause_strategy",
            "resume_strategy",
            "stop_strategy",
            "get_strategy_executions",
            "approve_strategy_execution",
            "update_strategy",
        }),
        frozenset({
            "strategy", "dca", "dollar cost", "recurring", "automate",
            "schedule", "hourly", "daily", "weekly", "rebalance",
        }),
    ),
    (
        ToolGroup.POLICY,
        frozenset({
            "get_risk_policy",
            "update_risk_policy",
            "check_action_allowed",
            "get_system_status",
        }),
        frozenset({
            "risk", "policy", "limit", "slippage", "position size",
            "system status",
        }),
    ),
    (
        ToolGroup.COPY_TRADING,
        frozenset({
            "get_top_traders",
            "get_trader_profile",
            "start_copy_trading",
            "stop_copy_trading",
            "list_copy_relationships",
            "get_pending_copy_trades",
            "approve_copy_trade",
            "reject_copy_trade",
        }),
        frozenset({
            "copy", "follow", "mirror", "top traders", "leader",
        }),
    ),
    (
        ToolGroup.POLYMARKET,
        frozenset({
            "get_polymarket_markets",
            "get_polymarket_market",
            "get_polymarket_portfolio",
            "get_polymarket_quote",
            "analyze_polymarket",
            "get_polymarket_top_traders",
            "get_polymarket_trader_profile",
            "start_polymarket_copy",
            "stop_polymarket_copy",
            "list_polymarket_copy_relationships",
            "get_pending_polymarket_copies",
            "approve_polymarket_copy",
            "reject_polymarket_copy",
        }),
        frozenset({
            "polymarket", "prediction", "bet", "odds", "probability",
        }),
    ),
]

# Pre-compute lookup for fast keyword scanning
_KEYWORD_TO_GROUPS: Dict[str, List[ToolGroup]] = {}
for _group, _tools, _keywords in TOOL_GROUPS:
    for _kw in _keywords:
        _KEYWORD_TO_GROUPS.setdefault(_kw, []).append(_group)

# Sort keywords longest-first so multi-word phrases match before substrings
_SORTED_KEYWORDS: List[str] = sorted(_KEYWORD_TO_GROUPS.keys(), key=len, reverse=True)

# Pre-compute group → tool names
_GROUP_TOOL_MAP: Dict[ToolGroup, FrozenSet[str]] = {
    group: tools for group, tools, _kw in TOOL_GROUPS
}


def resolve_groups(user_message: str) -> Set[ToolGroup]:
    """Scan *user_message* for keywords and return matching groups.

    CORE is always included regardless of message content.
    """
    active: Set[ToolGroup] = {ToolGroup.CORE}
    if not user_message:
        return active

    msg_lower = user_message.lower()
    for keyword in _SORTED_KEYWORDS:
        if keyword in msg_lower:
            for group in _KEYWORD_TO_GROUPS[keyword]:
                active.add(group)

    return active


def get_tool_names_for_groups(groups: Set[ToolGroup]) -> Set[str]:
    """Return the union of tool names for the given groups."""
    names: Set[str] = set()
    for group in groups:
        tools = _GROUP_TOOL_MAP.get(group)
        if tools:
            names |= tools
    return names
