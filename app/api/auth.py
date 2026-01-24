"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from app.auth import (
    AuthService,
    get_auth_service,
    AuthSession,
    AuthError,
    SessionExpiredError,
    require_auth,
)
from app.auth.models import (
    VerifyRequest,
    RefreshRequest,
    TokenPayload,
    Scope,
)


router = APIRouter(prefix="/auth", tags=["auth"])


class NonceRequest(BaseModel):
    """Request for nonce generation."""
    wallet_address: str
    chain: Optional[str] = None


class NonceResponse(BaseModel):
    """Response with nonce for wallet sign-in."""
    nonce: str
    expires_at: str


class AuthResponse(BaseModel):
    """Response after successful authentication."""
    access_token: str
    refresh_token: str
    expires_at: str
    wallet_address: str
    chain_id: int | str
    user_id: Optional[str] = None
    wallet_id: Optional[str] = None
    scopes: list[str]


class SessionInfo(BaseModel):
    """Current session info."""
    wallet_address: str
    chain_id: int | str
    user_id: Optional[str] = None
    wallet_id: Optional[str] = None
    scopes: list[str]


@router.post("/nonce", response_model=NonceResponse)
async def get_nonce(
    request: NonceRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Generate a nonce for wallet sign-in.

    The client should use this nonce when creating the wallet sign-in message.
    Nonce expires after 10 minutes.
    """
    result = await auth_service.generate_nonce(request.wallet_address, request.chain)
    return NonceResponse(
        nonce=result["nonce"],
        expires_at=result["expires_at"],
    )


@router.post("/verify", response_model=AuthResponse)
async def verify_signature(
    request: VerifyRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Verify a wallet signature and create a session.

    The client should:
    1. Create a wallet sign-in message with the nonce from /auth/nonce
    2. Sign the message with the wallet
    3. Send the message, signature, and chain here

    Returns JWT tokens for authenticated requests.
    """
    try:
        # Verify the signature
        wallet = await auth_service.verify_signature(
            message=request.message,
            signature=request.signature,
            chain=request.chain,
        )

        # Create a session
        session = await auth_service.create_session(wallet)

        return AuthResponse(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_at=session.expires_at.isoformat(),
            wallet_address=session.wallet_address,
            chain_id=session.chain_id,
            user_id=session.user_id,
            wallet_id=session.wallet_id,
            scopes=[s.value if isinstance(s, Scope) else s for s in session.scopes],
        )

    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    request: RefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Refresh an existing session.

    Use this when the access token is about to expire.
    """
    try:
        session = await auth_service.refresh_session(request.refresh_token)

        return AuthResponse(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_at=session.expires_at.isoformat(),
            wallet_address=session.wallet_address,
            chain_id=session.chain_id,
            user_id=session.user_id,
            wallet_id=session.wallet_id,
            scopes=[s.value if isinstance(s, Scope) else s for s in session.scopes],
        )

    except SessionExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please sign in again.",
        )
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.get("/me", response_model=SessionInfo)
async def get_current_session(
    auth: TokenPayload = Depends(require_auth),
):
    """
    Get the current authenticated session info.

    Requires a valid access token.
    """
    return SessionInfo(
        wallet_address=auth.sub,
        chain_id=auth.chain_id,
        user_id=auth.user_id,
        wallet_id=auth.wallet_id,
        scopes=auth.scopes,
    )


@router.post("/logout")
async def logout(
    auth: TokenPayload = Depends(require_auth),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Logout and revoke the current session.
    """
    await auth_service.revoke_session(auth.session_id)
    return {"message": "Successfully logged out"}
