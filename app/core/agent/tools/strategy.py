"""Strategy management tool handlers.

Converted from the monolithic tools.py – each handler is a standalone
async function decorated with @tool_spec.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# =========================================================================
# list_strategies
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_list_strategies(
    wallet_address: str,
    strategy_type: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle listing strategies for a wallet."""
    from ....db import get_convex_client

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
        logger.error(f"Error listing strategies: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# get_strategy
# =========================================================================

@tool_spec(
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
)
async def handle_get_strategy(
    strategy_id: str,
) -> Dict[str, Any]:
    """Handle getting a single strategy."""
    from ....db import get_convex_client

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
        logger.error(f"Error getting strategy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# create_strategy
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_create_strategy(
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
    from ....db import get_convex_client

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
            logger.info(f"Auto-registered new user and wallet for {wallet_address}")

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

        # Normalize config to canonical camelCase nested format
        from app.core.strategies.config_normalizer import normalize_strategy_config
        normalized_config = normalize_strategy_config(strategy_type, {
            **config,
            "chainId": chain_id,
            "maxSlippageBps": max_slippage_bps,
            "maxGasUsd": max_gas_usd,
        })

        args = {
            "userId": user.get("_id"),
            "walletAddress": wallet_address.lower(),
            "name": name,
            "strategyType": strategy_type,
            "config": normalized_config,
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
        logger.error(f"Error creating strategy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# pause_strategy
# =========================================================================

@tool_spec(
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
)
async def handle_pause_strategy(
    strategy_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle pausing a strategy."""
    from ....db import get_convex_client

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
        logger.error(f"Error pausing strategy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# resume_strategy
# =========================================================================

@tool_spec(
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
)
async def handle_resume_strategy(
    strategy_id: str,
    session_key_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle resuming a paused strategy.

    NOTE: Without a session_key_id, the strategy will go to 'pending_session' status.
    A session key is required for actual automated execution.
    """
    from ....db import get_convex_client

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
        logger.error(f"Error resuming strategy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# stop_strategy
# =========================================================================

@tool_spec(
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
)
async def handle_stop_strategy(
    strategy_id: str,
) -> Dict[str, Any]:
    """Handle stopping a strategy."""
    from ....db import get_convex_client

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
        logger.error(f"Error stopping strategy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# get_strategy_executions
# =========================================================================

@tool_spec(
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
)
async def handle_get_strategy_executions(
    strategy_id: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """Handle getting strategy execution history."""
    from ....db import get_convex_client

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
        logger.error(f"Error getting strategy executions: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# approve_strategy_execution
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_approve_strategy_execution(
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
    from ....db import get_convex_client

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
        logger.error(f"Error handling strategy execution approval: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# update_strategy
# =========================================================================

@tool_spec(
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
)
async def handle_update_strategy(
    strategy_id: str,
    config: Optional[Dict[str, Any]] = None,
    max_slippage_percent: Optional[float] = None,
    max_gas_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Handle updating a strategy configuration."""
    from ....db import get_convex_client

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
        logger.error(f"Error updating strategy: {e}")
        return {"success": False, "error": str(e)}
