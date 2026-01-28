"""
Permission request and grant API endpoints.

Handles the flow for granting Smart Session permissions:
1. Frontend requests a permission grant configuration
2. Backend validates and returns the config with transaction data
3. Frontend builds and sends the grant transaction
4. User signs the transaction
5. Backend records the active session

This API works with both:
- Rhinestone Smart Sessions (EVM, on-chain enforcement)
- Swig Session Authorities (Solana, on-chain enforcement)
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.db import get_convex_client
from app.core.wallet.smart_sessions import (
    SmartSessionConfig,
    SmartSessionsHelper,
    SpendingLimit,
    TimeConstraint,
    ActionType,
    validate_session_config,
    build_permission_grant_data,
    SmartSessionError,
)
from app.services.session_wallet_service import (
    get_session_wallet_service,
    SessionWalletError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/permissions", tags=["permissions"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class PermissionRequestModel(BaseModel):
    """Request to create a Smart Session permission grant."""

    wallet_address: str = Field(..., description="User's EOA or smart account address")
    chain_type: str = Field(
        default="evm",
        description="Chain type: 'evm' for Rhinestone, 'solana' for Swig",
    )
    session_type: str = Field(
        default="swap",
        description="Preset session type: 'swap', 'dca', 'copy_trading', 'custom'",
    )

    # Limits (optional, uses defaults if not specified)
    max_per_tx_usd: Optional[float] = Field(
        default=None,
        description="Maximum USD value per transaction",
    )
    max_daily_usd: Optional[float] = Field(
        default=None,
        description="Maximum daily spending limit",
    )
    max_total_usd: Optional[float] = Field(
        default=None,
        description="Maximum total spending over session lifetime",
    )

    # Time constraints
    valid_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Session validity in days (1-365)",
    )

    # Optional restrictions
    allowed_chains: Optional[List[int]] = Field(
        default=None,
        description="Allowed chain IDs (empty = all chains)",
    )
    allowed_tokens: Optional[List[str]] = Field(
        default=None,
        description="Allowed token addresses (empty = all tokens)",
    )
    allowed_actions: Optional[List[str]] = Field(
        default=None,
        description="Allowed action types (empty = use preset)",
    )

    # Metadata
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for this session",
    )


class PermissionGrantResponse(BaseModel):
    """Response with permission grant configuration."""

    session_key_address: str = Field(
        ...,
        alias="sessionKeyAddress",
        description="The address that will sign transactions",
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Smart Session configuration for frontend",
    )
    grant_data: Dict[str, Any] = Field(
        ...,
        alias="grantData",
        description="Transaction data for the permission grant",
    )
    chain_type: str = Field(..., alias="chainType")
    valid_until: int = Field(
        ...,
        alias="validUntil",
        description="Unix timestamp when session expires",
    )
    instructions: str = Field(
        ...,
        description="Instructions for the user",
    )

    class Config:
        populate_by_name = True


class ActivePermissionResponse(BaseModel):
    """Response for an active permission/session."""

    session_key_address: str = Field(..., alias="sessionKeyAddress")
    chain_type: str = Field(..., alias="chainType")
    session_type: str = Field(..., alias="sessionType")
    allowed_actions: List[str] = Field(..., alias="allowedActions")
    spending_limit: Dict[str, Any] = Field(..., alias="spendingLimit")
    valid_until: int = Field(..., alias="validUntil")
    created_at: int = Field(..., alias="createdAt")
    status: str
    usage: Dict[str, Any]

    class Config:
        populate_by_name = True


class RevokePermissionRequest(BaseModel):
    """Request to revoke a permission."""

    session_key_address: str = Field(..., alias="sessionKeyAddress")
    chain_type: str = Field(default="evm", alias="chainType")
    reason: Optional[str] = Field(default=None)

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/request", response_model=PermissionGrantResponse)
async def request_permission_grant(
    request: PermissionRequestModel,
) -> PermissionGrantResponse:
    """
    Request a Smart Session permission grant configuration.

    This endpoint:
    1. Gets or creates a session signing key (Turnkey wallet)
    2. Builds the Smart Session configuration
    3. Returns the data needed for the frontend to create the grant transaction

    The user must then sign the grant transaction to enable autonomous execution.

    Args:
        request: Permission request parameters

    Returns:
        Permission grant configuration for frontend
    """
    # Validate chain type
    if request.chain_type not in ["evm", "solana"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported chain type: {request.chain_type}",
        )

    # Check if Rhinestone/Swig is enabled
    if request.chain_type == "evm" and not settings.enable_rhinestone:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EVM Smart Sessions (Rhinestone) not enabled",
        )
    if request.chain_type == "solana" and not settings.enable_swig:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Solana Smart Sessions (Swig) not enabled",
        )

    # Get or create session signing key
    if not settings.enable_turnkey:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session key infrastructure (Turnkey) not enabled",
        )

    try:
        session_wallet_service = get_session_wallet_service()
        session_wallet = await session_wallet_service.get_or_create_session_wallet(
            wallet_address=request.wallet_address,
            chain_type=request.chain_type,
            label=request.label,
        )
        session_key_address = session_wallet.turnkey_address
    except SessionWalletError as e:
        logger.error(f"Failed to get session key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to provision session key: {e}",
        )

    # Build configuration based on session type and chain
    if request.chain_type == "evm":
        config = _build_evm_session_config(request, session_key_address)
    else:
        config = _build_solana_session_config(request, session_key_address)

    # Validate the configuration
    validation = validate_session_config(config)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration: {validation['errors']}",
        )

    # Build grant data
    try:
        grant_data = build_permission_grant_data(config)
    except SmartSessionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return PermissionGrantResponse(
        sessionKeyAddress=session_key_address,
        config=config.to_module_config(),
        grantData=grant_data,
        chainType=request.chain_type,
        validUntil=config.time_constraint.valid_until,
        instructions=(
            f"Sign the permission grant transaction to enable autonomous {request.session_type} "
            f"execution. This session will be valid for {request.valid_days} days and has a "
            f"spending limit of ${config.spending_limit.max_per_tx_usd} per transaction."
        ),
    )


@router.post("/confirm")
async def confirm_permission_grant(
    wallet_address: str,
    session_key_address: str,
    chain_type: str = "evm",
    tx_hash: Optional[str] = None,
    session_type: str = "swap",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Confirm that a permission grant transaction was signed and submitted.

    Call this after the user signs the grant transaction to record
    the active session in the database.

    Args:
        wallet_address: User's wallet address
        session_key_address: The session key that was granted permissions
        chain_type: "evm" or "solana"
        tx_hash: Transaction hash of the grant (optional for EVM)
        session_type: Type of session (swap, dca, copy_trading)
        config: The configuration that was used

    Returns:
        Confirmation with session details
    """
    convex = get_convex_client()

    # Store the active session in Convex
    try:
        await convex.mutation(
            "smartSessions:create",
            {
                "walletAddress": wallet_address.lower(),
                "sessionKeyAddress": session_key_address.lower(),
                "chainType": chain_type,
                "sessionType": session_type,
                "grantTxHash": tx_hash,
                "config": config or {},
                "createdAt": int(time.time() * 1000),
                "status": "active",
            },
        )
    except Exception as e:
        logger.error(f"Failed to record permission grant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record permission: {e}",
        )

    return {
        "success": True,
        "sessionKeyAddress": session_key_address,
        "chainType": chain_type,
        "sessionType": session_type,
        "message": (
            f"Permission granted successfully. The session key {session_key_address[:10]}... "
            f"can now execute {session_type} operations on your behalf."
        ),
    }


@router.get("/{wallet_address}", response_model=List[ActivePermissionResponse])
async def list_active_permissions(
    wallet_address: str,
    chain_type: Optional[str] = None,
    include_expired: bool = False,
) -> List[ActivePermissionResponse]:
    """
    List all active permissions for a wallet.

    Args:
        wallet_address: User's wallet address
        chain_type: Filter by chain type (optional)
        include_expired: Include expired sessions

    Returns:
        List of active permissions
    """
    convex = get_convex_client()

    try:
        sessions = await convex.query(
            "smartSessions:listByWallet",
            {
                "walletAddress": wallet_address.lower(),
                "chainType": chain_type,
                "includeExpired": include_expired,
            },
        )
    except Exception as e:
        logger.error(f"Failed to list permissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list permissions: {e}",
        )

    return [
        ActivePermissionResponse(
            sessionKeyAddress=s["sessionKeyAddress"],
            chainType=s["chainType"],
            sessionType=s.get("sessionType", "unknown"),
            allowedActions=s.get("config", {}).get("allowedActions", []),
            spendingLimit=s.get("config", {}).get("spendingLimit", {}),
            validUntil=s.get("config", {}).get("timeConstraint", {}).get("validUntil", 0),
            createdAt=s["createdAt"],
            status=s["status"],
            usage=s.get("usage", {"totalSpentUsd": "0", "transactionCount": 0}),
        )
        for s in (sessions or [])
    ]


@router.get("/{wallet_address}/{session_key_address}")
async def get_permission_details(
    wallet_address: str,
    session_key_address: str,
    chain_type: str = "evm",
) -> Dict[str, Any]:
    """
    Get details for a specific permission.

    Args:
        wallet_address: User's wallet address
        session_key_address: The session key address
        chain_type: "evm" or "solana"

    Returns:
        Permission details including usage stats
    """
    convex = get_convex_client()

    try:
        session = await convex.query(
            "smartSessions:get",
            {
                "walletAddress": wallet_address.lower(),
                "sessionKeyAddress": session_key_address.lower(),
                "chainType": chain_type,
            },
        )
    except Exception as e:
        logger.error(f"Failed to get permission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get permission: {e}",
        )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found",
        )

    return {
        "sessionKeyAddress": session["sessionKeyAddress"],
        "chainType": session["chainType"],
        "sessionType": session.get("sessionType", "unknown"),
        "config": session.get("config", {}),
        "createdAt": session["createdAt"],
        "status": session["status"],
        "usage": session.get("usage", {}),
        "grantTxHash": session.get("grantTxHash"),
    }


@router.post("/revoke")
async def revoke_permission(
    request: RevokePermissionRequest,
    wallet_address: str,
) -> Dict[str, Any]:
    """
    Revoke a permission/session.

    Note: This marks the session as revoked in our database.
    For EVM, the on-chain permission must be revoked separately
    by the user calling the Smart Account's revoke function.

    Args:
        request: Revoke request with session key and reason
        wallet_address: User's wallet address (for authorization)

    Returns:
        Confirmation of revocation
    """
    convex = get_convex_client()

    try:
        await convex.mutation(
            "smartSessions:revoke",
            {
                "walletAddress": wallet_address.lower(),
                "sessionKeyAddress": request.session_key_address.lower(),
                "chainType": request.chain_type,
                "reason": request.reason,
                "revokedAt": int(time.time() * 1000),
            },
        )
    except Exception as e:
        logger.error(f"Failed to revoke permission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke permission: {e}",
        )

    return {
        "success": True,
        "sessionKeyAddress": request.session_key_address,
        "message": (
            "Permission revoked in our system. For EVM, you should also revoke "
            "the on-chain permission via your Smart Account's management interface."
        ),
    }


@router.post("/validate")
async def validate_permission(
    wallet_address: str,
    session_key_address: str,
    chain_type: str = "evm",
    action_type: str = "swap",
    value_usd: float = 0,
) -> Dict[str, Any]:
    """
    Validate if a session can perform an action.

    Use this before executing to check if the session has
    sufficient permissions and limits.

    Args:
        wallet_address: User's wallet address
        session_key_address: The session key to validate
        chain_type: "evm" or "solana"
        action_type: The action to validate
        value_usd: The value of the action in USD

    Returns:
        Validation result
    """
    convex = get_convex_client()

    try:
        session = await convex.query(
            "smartSessions:get",
            {
                "walletAddress": wallet_address.lower(),
                "sessionKeyAddress": session_key_address.lower(),
                "chainType": chain_type,
            },
        )
    except Exception as e:
        logger.error(f"Failed to get permission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate: {e}",
        )

    if not session:
        return {
            "valid": False,
            "errors": ["Session not found"],
        }

    errors = []

    # Check status
    if session["status"] != "active":
        errors.append(f"Session is {session['status']}")

    # Check expiry
    config = session.get("config", {})
    valid_until = config.get("timeConstraint", {}).get("validUntil", 0)
    if valid_until and valid_until < int(time.time()):
        errors.append("Session has expired")

    # Check action type
    allowed_actions = config.get("allowedActions", [])
    if allowed_actions and action_type not in allowed_actions:
        errors.append(f"Action '{action_type}' not permitted")

    # Check spending limit
    spending_limit = config.get("spendingLimit", {})
    max_per_tx = Decimal(spending_limit.get("maxPerTx", "0")) / Decimal("1000000")
    if value_usd > float(max_per_tx):
        errors.append(f"Value ${value_usd} exceeds per-tx limit ${max_per_tx}")

    # Check daily limit if applicable
    usage = session.get("usage", {})
    if spending_limit.get("maxDaily"):
        max_daily = Decimal(spending_limit["maxDaily"]) / Decimal("1000000")
        spent_today = Decimal(usage.get("spentTodayUsd", "0"))
        if spent_today + Decimal(str(value_usd)) > max_daily:
            remaining = max_daily - spent_today
            errors.append(f"Would exceed daily limit. Remaining: ${remaining}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "session": {
            "sessionKeyAddress": session["sessionKeyAddress"],
            "status": session["status"],
            "usage": usage,
        },
    }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _build_evm_session_config(
    request: PermissionRequestModel,
    session_key_address: str,
) -> SmartSessionConfig:
    """Build EVM Smart Session configuration from request."""
    helper = SmartSessionsHelper

    # Use preset or custom configuration
    if request.session_type == "swap":
        config = helper.create_swap_session(
            session_key_address=session_key_address,
            owner_address=request.wallet_address,
            max_per_tx_usd=Decimal(str(request.max_per_tx_usd or 500)),
            max_daily_usd=Decimal(str(request.max_daily_usd)) if request.max_daily_usd else None,
            valid_days=request.valid_days,
            chains=request.allowed_chains,
        )
    elif request.session_type == "dca":
        config = helper.create_dca_session(
            session_key_address=session_key_address,
            owner_address=request.wallet_address,
            max_per_execution_usd=Decimal(str(request.max_per_tx_usd or 100)),
            max_total_usd=Decimal(str(request.max_total_usd or 5000)),
            valid_days=request.valid_days,
            chains=request.allowed_chains,
        )
    elif request.session_type == "copy_trading":
        config = helper.create_copy_trading_session(
            session_key_address=session_key_address,
            owner_address=request.wallet_address,
            max_per_copy_usd=Decimal(str(request.max_per_tx_usd or 250)),
            max_daily_usd=Decimal(str(request.max_daily_usd or 1000)),
            valid_days=request.valid_days,
            chains=request.allowed_chains,
        )
    else:
        # Custom configuration
        config = SmartSessionConfig(
            session_key_address=session_key_address,
            owner_address=request.wallet_address,
            allowed_actions=[ActionType(a) for a in (request.allowed_actions or ["swap"])],
            allowed_chains=request.allowed_chains or [],
            allowed_tokens=request.allowed_tokens or [],
            spending_limit=SpendingLimit(
                max_per_tx_usd=Decimal(str(request.max_per_tx_usd or 500)),
                max_daily_usd=Decimal(str(request.max_daily_usd)) if request.max_daily_usd else None,
                max_total_usd=Decimal(str(request.max_total_usd)) if request.max_total_usd else None,
            ),
            time_constraint=TimeConstraint(
                valid_after=int(time.time()),
                valid_until=int(time.time()) + (request.valid_days * 24 * 60 * 60),
            ),
            label=request.label or f"Custom {request.session_type} Session",
        )

    # Override label if provided
    if request.label:
        config.label = request.label

    return config


def _build_solana_session_config(
    request: PermissionRequestModel,
    session_key_address: str,
) -> SmartSessionConfig:
    """
    Build Solana Swig session configuration from request.

    Note: Swig uses a different on-chain format, but we use the same
    SmartSessionConfig model for consistency in the API.
    """
    # For now, use the same config structure
    # The frontend will translate to Swig-specific format
    return _build_evm_session_config(request, session_key_address)
