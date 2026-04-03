"""
Agent Harness API — tool discovery and headless execution.

Exposes the ToolRegistry to external AI agent harnesses without
requiring natural-language chat. Three endpoints:

  GET  /agent/tools           — list all tools (optional ?category= filter)
  GET  /agent/tools/{name}    — single tool detail
  POST /agent/execute         — headless tool execution
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..auth import optional_auth, TokenPayload
from ..core.agent.tools import ToolRegistry, ToolExecutor
from ..providers.llm.base import ToolCall
from ..types.agent import (
    ActionStatus,
    ExecuteRequest,
    ExecuteResponse,
    StructuredActionResult,
    ToolInfo,
    ToolListResponse,
    ToolParameterSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent")

# ---------------------------------------------------------------------------
# Module-level singletons (no Agent/LLM dependency)
# ---------------------------------------------------------------------------
_registry = ToolRegistry()
_executor = ToolExecutor(_registry)

# ---------------------------------------------------------------------------
# Static mappings
# ---------------------------------------------------------------------------

HUMAN_APPROVAL_TOOLS: set[str] = {
    "get_swap_quote",
    "get_bridge_quote",
    "get_solana_swap_quote",
    "get_polymarket_quote",
    "approve_strategy_execution",
    "create_strategy",
    "approve_copy_trade",
    "approve_polymarket_copy",
    "execute_transfer",
}

TOOL_NEXT_ACTIONS: dict[str, list[str]] = {
    "get_portfolio": ["get_token_chart", "get_news", "get_wallet_history"],
    "get_token_chart": ["get_swap_quote", "get_news"],
    "get_trending_tokens": ["get_token_chart", "get_token_news"],
    "get_wallet_history": ["get_portfolio"],
    "get_tvl_data": ["get_token_chart"],
    "get_news": ["get_token_chart", "get_trending_tokens"],
    "get_personalized_news": ["get_token_chart", "get_portfolio"],
    "get_token_news": ["get_token_chart"],
    "get_risk_policy": ["update_risk_policy"],
    "update_risk_policy": ["get_risk_policy", "check_action_allowed"],
    "check_action_allowed": ["get_risk_policy"],
    "get_swap_quote": ["approve_strategy_execution", "check_action_allowed"],
    "get_bridge_quote": ["approve_strategy_execution", "check_action_allowed"],
    "get_solana_swap_quote": ["approve_strategy_execution", "check_action_allowed"],
    "list_strategies": ["create_strategy", "get_strategy"],
    "get_strategy": ["pause_strategy", "stop_strategy", "get_strategy_executions"],
    "create_strategy": ["list_strategies"],
    "get_top_traders": ["get_trader_profile", "start_copy_trading"],
    "get_trader_profile": ["start_copy_trading"],
    "start_copy_trading": ["list_copy_relationships"],
    "list_copy_relationships": ["stop_copy_trading"],
    "get_pending_copy_trades": ["approve_copy_trade", "reject_copy_trade"],
    "get_polymarket_markets": ["get_polymarket_market", "analyze_polymarket"],
    "get_polymarket_market": ["get_polymarket_quote", "analyze_polymarket"],
    "get_polymarket_portfolio": ["get_polymarket_markets"],
    "get_polymarket_quote": ["approve_strategy_execution"],
    "analyze_polymarket": ["get_polymarket_quote"],
    "get_polymarket_top_traders": ["get_polymarket_trader_profile"],
    "get_polymarket_trader_profile": ["start_polymarket_copy"],
    "start_polymarket_copy": ["list_polymarket_copy_relationships"],
    "get_pending_polymarket_copies": ["approve_polymarket_copy", "reject_polymarket_copy"],
}

TOOL_CATEGORIES: dict[str, str] = {
    # Portfolio
    "get_portfolio": "portfolio",
    "get_wallet_history": "portfolio",
    "update_portfolio_chains": "portfolio",
    # Market data
    "get_token_chart": "market_data",
    "get_trending_tokens": "market_data",
    "get_tvl_data": "market_data",
    # News
    "get_news": "news",
    "get_personalized_news": "news",
    "get_token_news": "news",
    # Policy
    "get_risk_policy": "policy",
    "update_risk_policy": "policy",
    "check_action_allowed": "policy",
    # Strategy
    "list_strategies": "strategy",
    "get_strategy": "strategy",
    "create_strategy": "strategy",
    "pause_strategy": "strategy",
    "resume_strategy": "strategy",
    "stop_strategy": "strategy",
    "get_strategy_executions": "strategy",
    "approve_strategy_execution": "strategy",
    "update_strategy": "strategy",
    # Trading
    "get_swap_quote": "trading",
    "get_bridge_quote": "trading",
    "get_solana_swap_quote": "trading",
    "execute_transfer": "trading",
    # Copy trading
    "get_top_traders": "copy_trading",
    "get_trader_profile": "copy_trading",
    "start_copy_trading": "copy_trading",
    "stop_copy_trading": "copy_trading",
    "list_copy_relationships": "copy_trading",
    "get_pending_copy_trades": "copy_trading",
    "approve_copy_trade": "copy_trading",
    "reject_copy_trade": "copy_trading",
    # Prediction market
    "get_polymarket_markets": "prediction_market",
    "get_polymarket_market": "prediction_market",
    "get_polymarket_portfolio": "prediction_market",
    "get_polymarket_quote": "prediction_market",
    "analyze_polymarket": "prediction_market",
    "get_polymarket_top_traders": "prediction_market",
    "get_polymarket_trader_profile": "prediction_market",
    "start_polymarket_copy": "prediction_market",
    "stop_polymarket_copy": "prediction_market",
    "list_polymarket_copy_relationships": "prediction_market",
    "get_pending_polymarket_copies": "prediction_market",
    "approve_polymarket_copy": "prediction_market",
    "reject_polymarket_copy": "prediction_market",
    # System
    "get_system_status": "system",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registered_tool_to_info(name: str) -> Optional[ToolInfo]:
    """Convert a RegisteredTool from the registry into a ToolInfo response model."""
    tool = _registry.get_tool(name)
    if tool is None:
        return None

    params = [
        ToolParameterSchema(
            name=p.name,
            type=p.type.value,
            description=p.description,
            required=p.required,
            enum=p.enum,
            default=p.default,
        )
        for p in tool.definition.parameters
    ]

    return ToolInfo(
        name=tool.definition.name,
        description=tool.definition.description,
        parameters=params,
        requires_address=tool.requires_address,
        category=TOOL_CATEGORIES.get(name),
    )


# ---------------------------------------------------------------------------
# GET /agent/tools
# ---------------------------------------------------------------------------

@router.get("/tools", response_model=ToolListResponse)
async def list_tools(category: Optional[str] = None) -> ToolListResponse:
    """List all registered tools, optionally filtered by category."""
    all_names = list(_registry._tools.keys())

    if category:
        all_names = [n for n in all_names if TOOL_CATEGORIES.get(n) == category]

    tools = []
    for name in all_names:
        info = _registered_tool_to_info(name)
        if info is not None:
            tools.append(info)

    return ToolListResponse(tools=tools, count=len(tools))


# ---------------------------------------------------------------------------
# GET /agent/tools/{tool_name}
# ---------------------------------------------------------------------------

@router.get("/tools/{tool_name}", response_model=ToolInfo)
async def get_tool_detail(tool_name: str) -> ToolInfo:
    """Get detailed info for a single tool."""
    info = _registered_tool_to_info(tool_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    return info


# ---------------------------------------------------------------------------
# POST /agent/execute
# ---------------------------------------------------------------------------

@router.post("/execute", response_model=ExecuteResponse)
async def execute_tool(
    request: ExecuteRequest,
    auth: Optional[TokenPayload] = Depends(optional_auth),
) -> ExecuteResponse:
    """Execute a single tool headlessly and return a structured result."""

    # 1. Look up tool
    tool = _registry.get_tool(request.tool)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {request.tool}")

    # 2. Validate requires_address
    if tool.requires_address and not request.address:
        raise HTTPException(
            status_code=422,
            detail=f"Tool '{request.tool}' requires an address but none was provided",
        )

    # 3. Auth check — if authenticated, address must match token subject
    if auth and request.address:
        if request.address.lower() != auth.sub.lower():
            raise HTTPException(
                status_code=403,
                detail="You can only execute tools against your own wallet address",
            )

    # 4. Inject wallet_address / chain into params if tool requires them
    params = dict(request.params)
    if tool.requires_address and request.address:
        params.setdefault("wallet_address", request.address)
    if request.chain:
        params.setdefault("chain", request.chain)

    # 5. Dry run — validate only
    if request.dry_run:
        return ExecuteResponse(
            result=StructuredActionResult(
                action_type=request.tool,
                status=ActionStatus.PENDING,
                data=None,
                requires_human=request.tool in HUMAN_APPROVAL_TOOLS,
                next_actions=TOOL_NEXT_ACTIONS.get(request.tool, []),
            ),
            dry_run=True,
        )

    # 6. Execute
    start = time.monotonic()
    tool_call = ToolCall(id="agent-harness", name=request.tool, arguments=params)
    tool_result = await _executor.execute_single(tool_call)
    latency_ms = round((time.monotonic() - start) * 1000, 2)

    # 7. Wrap into StructuredActionResult
    if tool_result.error:
        result = StructuredActionResult(
            action_type=request.tool,
            status=ActionStatus.FAILED,
            data=None,
            error=tool_result.error,
            requires_human=request.tool in HUMAN_APPROVAL_TOOLS,
            next_actions=[],
            latency_ms=latency_ms,
        )
    else:
        result = StructuredActionResult(
            action_type=request.tool,
            status=ActionStatus.SUCCESS,
            data=tool_result.result,
            requires_human=request.tool in HUMAN_APPROVAL_TOOLS,
            next_actions=TOOL_NEXT_ACTIONS.get(request.tool, []),
            latency_ms=latency_ms,
        )

    return ExecuteResponse(result=result, dry_run=False)
