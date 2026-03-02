"""Policy tool handlers: risk policy CRUD, action checking, system status."""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# =========================================================================
# get_risk_policy
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_get_risk_policy(
    wallet_address: str,
) -> Dict[str, Any]:
    """Handle getting risk policy for a wallet."""
    from ....db import get_convex_client
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
        logger.error(f"Error fetching risk policy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# update_risk_policy
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_update_risk_policy(
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
    from ....db import get_convex_client

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
        logger.error(f"Error updating risk policy: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# check_action_allowed
# =========================================================================

@tool_spec(
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
    requires_address=True,
)
async def handle_check_action_allowed(
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
    from ....db import get_convex_client
    from ...policy import PolicyEngine, ActionContext, RiskPolicyConfig, SystemPolicyConfig

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
        logger.error(f"Error checking action: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# get_system_status
# =========================================================================

@tool_spec(
    name="get_system_status",
    description=(
        "Check the current system status and platform operational state. "
        "Returns whether the system is operational, any emergency stops, "
        "maintenance windows, blocked contracts/tokens, and allowed chains. "
        "Use this to check if the platform is available for trading."
    ),
    parameters=[],
)
async def handle_get_system_status() -> Dict[str, Any]:
    """Handle getting system status."""
    from ....db import get_convex_client
    from ...policy import PolicyEngine, SystemPolicyConfig

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
        logger.error(f"Error fetching system status: {e}")
        return {"success": False, "error": str(e)}
