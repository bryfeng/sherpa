"""
Swig Wallets API

Endpoints for Solana Swig Smart Wallet management:
- Wallet status
- Session authority management
- Role-based permissions
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import settings
from ..providers.swig import get_swig_provider, SwigError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/swig-wallets", tags=["swig-wallets"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SwigWalletStatusResponse(BaseModel):
    """Swig wallet status."""
    wallet_address: str
    owner_address: str
    is_deployed: bool
    active_sessions: int = 0
    total_sessions: int = 0


class SwigSessionRequest(BaseModel):
    """Request to create a Swig session."""
    role: str = Field(default="agent", description="Role name: agent, dca, copy_trading")
    spending_limit_usd: float = Field(..., description="Maximum spending in USD")
    allowed_programs: List[str] = Field(default_factory=list, description="Allowed program IDs")
    allowed_tokens: List[str] = Field(default_factory=list, description="Allowed token mints")
    allowed_actions: List[str] = Field(default_factory=list, description="Allowed actions")
    validity_days: int = Field(default=7, description="Session validity in days")


class SwigSessionResponse(BaseModel):
    """Swig session details."""
    authority_id: str
    wallet_address: str
    role: str
    spending_limit_usd: Optional[float] = None
    total_spent_usd: Optional[float] = None
    spending_remaining_usd: Optional[float] = None
    allowed_programs: List[str] = []
    allowed_tokens: List[str] = []
    allowed_actions: List[str] = []
    expires_at: Optional[str] = None
    status: Optional[str] = None
    is_active: bool = False


class SwigSessionListResponse(BaseModel):
    """List of Swig sessions."""
    success: bool
    wallet_address: str
    sessions: List[SwigSessionResponse]
    active_count: int = 0
    error: Optional[str] = None


class SwigSessionValidationRequest(BaseModel):
    """Request to validate a session."""
    amount_usd: Optional[float] = None
    action: Optional[str] = None
    program_id: Optional[str] = None
    token_mint: Optional[str] = None


class SwigSessionValidationResponse(BaseModel):
    """Session validation result."""
    valid: bool
    error: Optional[str] = None
    spending_remaining_usd: Optional[float] = None


class SwigSessionUsageRequest(BaseModel):
    """Request to record session usage."""
    spent_usd: float
    tx_signature: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/{wallet_address}/status",
    response_model=SwigWalletStatusResponse,
    summary="Get Swig wallet status",
    description="Check if a Solana wallet has a deployed Swig smart wallet.",
)
async def get_wallet_status(wallet_address: str) -> SwigWalletStatusResponse:
    """Get Swig wallet deployment status."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()
        status = await provider.get_wallet_status(wallet_address)

        return SwigWalletStatusResponse(
            wallet_address=status.wallet_address,
            owner_address=status.owner_address,
            is_deployed=status.is_deployed,
            active_sessions=status.active_sessions,
            total_sessions=status.total_sessions,
        )
    except SwigError as e:
        logger.error(f"Failed to get Swig wallet status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{wallet_address}/sessions",
    response_model=SwigSessionListResponse,
    summary="List Swig sessions",
    description="List all session authorities for a Swig wallet.",
)
async def list_sessions(
    wallet_address: str,
    include_expired: bool = Query(False, description="Include expired sessions"),
    include_revoked: bool = Query(False, description="Include revoked sessions"),
) -> SwigSessionListResponse:
    """List all sessions for a wallet."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()
        authorities = await provider.list_session_authorities(
            wallet_address=wallet_address,
            include_expired=include_expired,
            include_revoked=include_revoked,
        )

        sessions = [
            SwigSessionResponse(
                authority_id=a.authority_id,
                wallet_address=a.wallet_address,
                role=a.role,
                spending_limit_usd=float(a.spending_limit_usd) if a.spending_limit_usd else None,
                total_spent_usd=float(a.total_spent_usd) if a.total_spent_usd else None,
                spending_remaining_usd=float(a.spending_remaining_usd) if a.spending_remaining_usd else None,
                allowed_programs=a.allowed_programs,
                allowed_tokens=a.allowed_tokens,
                allowed_actions=a.allowed_actions,
                expires_at=a.expires_at.isoformat() if a.expires_at else None,
                status=a.status,
                is_active=a.is_active,
            )
            for a in authorities
        ]

        active_count = sum(1 for s in sessions if s.is_active)

        return SwigSessionListResponse(
            success=True,
            wallet_address=wallet_address,
            sessions=sessions,
            active_count=active_count,
        )
    except SwigError as e:
        logger.error(f"Failed to list Swig sessions: {e}")
        return SwigSessionListResponse(
            success=False,
            wallet_address=wallet_address,
            sessions=[],
            error=str(e),
        )


@router.post(
    "/{wallet_address}/sessions",
    response_model=SwigSessionResponse,
    summary="Create Swig session",
    description="Create a new session authority with role-based permissions.",
)
async def create_session(
    wallet_address: str,
    request: SwigSessionRequest,
) -> SwigSessionResponse:
    """Create a new session authority."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()

        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(days=request.validity_days)

        authority = await provider.create_session_authority(
            wallet_address=wallet_address,
            role=request.role,
            spending_limit_usd=Decimal(str(request.spending_limit_usd)),
            allowed_programs=request.allowed_programs,
            allowed_tokens=request.allowed_tokens,
            allowed_actions=request.allowed_actions,
            expires_at=expires_at,
        )

        return SwigSessionResponse(
            authority_id=authority.authority_id,
            wallet_address=authority.wallet_address,
            role=authority.role,
            spending_limit_usd=float(authority.spending_limit_usd) if authority.spending_limit_usd else None,
            total_spent_usd=float(authority.total_spent_usd) if authority.total_spent_usd else 0,
            spending_remaining_usd=float(authority.spending_remaining_usd) if authority.spending_remaining_usd else None,
            allowed_programs=authority.allowed_programs,
            allowed_tokens=authority.allowed_tokens,
            allowed_actions=authority.allowed_actions,
            expires_at=authority.expires_at.isoformat() if authority.expires_at else None,
            status=authority.status,
            is_active=authority.is_active,
        )
    except SwigError as e:
        logger.error(f"Failed to create Swig session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sessions/{authority_id}",
    response_model=SwigSessionResponse,
    summary="Get Swig session",
    description="Get details of a specific session authority.",
)
async def get_session(authority_id: str) -> SwigSessionResponse:
    """Get session details by ID."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()
        authority = await provider.get_session_authority(authority_id)

        return SwigSessionResponse(
            authority_id=authority.authority_id,
            wallet_address=authority.wallet_address,
            role=authority.role,
            spending_limit_usd=float(authority.spending_limit_usd) if authority.spending_limit_usd else None,
            total_spent_usd=float(authority.total_spent_usd) if authority.total_spent_usd else 0,
            spending_remaining_usd=float(authority.spending_remaining_usd) if authority.spending_remaining_usd else None,
            allowed_programs=authority.allowed_programs,
            allowed_tokens=authority.allowed_tokens,
            allowed_actions=authority.allowed_actions,
            expires_at=authority.expires_at.isoformat() if authority.expires_at else None,
            status=authority.status,
            is_active=authority.is_active,
        )
    except SwigError as e:
        logger.error(f"Failed to get Swig session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sessions/{authority_id}/validate",
    response_model=SwigSessionValidationResponse,
    summary="Validate Swig session",
    description="Check if a session can execute an operation.",
)
async def validate_session(
    authority_id: str,
    request: SwigSessionValidationRequest,
) -> SwigSessionValidationResponse:
    """Validate session for an operation."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()

        is_valid, error = await provider.validate_session(
            authority_id=authority_id,
            amount_usd=Decimal(str(request.amount_usd)) if request.amount_usd else None,
            action=request.action,
            program_id=request.program_id,
            token_mint=request.token_mint,
        )

        # Get spending remaining
        spending_remaining = None
        if is_valid:
            authority = await provider.get_session_authority(authority_id)
            if authority.spending_remaining_usd:
                spending_remaining = float(authority.spending_remaining_usd)

        return SwigSessionValidationResponse(
            valid=is_valid,
            error=error,
            spending_remaining_usd=spending_remaining,
        )
    except SwigError as e:
        logger.error(f"Failed to validate Swig session: {e}")
        return SwigSessionValidationResponse(
            valid=False,
            error=str(e),
        )


@router.post(
    "/sessions/{authority_id}/record-usage",
    summary="Record session usage",
    description="Record spending against a session authority.",
)
async def record_usage(
    authority_id: str,
    request: SwigSessionUsageRequest,
) -> Dict[str, Any]:
    """Record usage against a session."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()
        result = await provider.record_session_usage(
            authority_id=authority_id,
            spent_usd=Decimal(str(request.spent_usd)),
            tx_signature=request.tx_signature,
        )

        return {"success": True, "result": result}
    except SwigError as e:
        logger.error(f"Failed to record Swig session usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sessions/{authority_id}/revoke",
    summary="Revoke Swig session",
    description="Revoke a session authority.",
)
async def revoke_session(authority_id: str) -> Dict[str, Any]:
    """Revoke a session authority."""
    if not settings.enable_swig:
        raise HTTPException(status_code=503, detail="Swig not enabled")

    try:
        provider = get_swig_provider()
        result = await provider.revoke_session_authority(authority_id)

        return {"success": True, "result": result}
    except SwigError as e:
        logger.error(f"Failed to revoke Swig session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
