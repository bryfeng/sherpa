"""
Authentication models and exceptions.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class AuthError(Exception):
    """Base authentication error."""
    pass


class SessionExpiredError(AuthError):
    """Session has expired."""
    pass


class InvalidSignatureError(AuthError):
    """SIWE signature is invalid."""
    pass


class InvalidNonceError(AuthError):
    """Nonce is invalid or expired."""
    pass


class Scope(str, Enum):
    """Permission scopes for API access."""
    READ_PORTFOLIO = "read:portfolio"
    READ_HISTORY = "read:history"
    EXECUTE_STRATEGY = "execute:strategy"
    MANAGE_STRATEGIES = "manage:strategies"
    ADMIN = "admin"


class VerifiedWallet(BaseModel):
    """Wallet verified via SIWE signature."""
    address: str
    chain_id: int


class AuthSession(BaseModel):
    """Authenticated session."""
    session_id: str
    wallet_address: str
    chain_id: int
    user_id: Optional[str] = None
    wallet_id: Optional[str] = None
    access_token: str
    refresh_token: str
    expires_at: datetime
    scopes: List[Scope] = Field(default_factory=lambda: [
        Scope.READ_PORTFOLIO,
        Scope.READ_HISTORY,
    ])


class NonceResponse(BaseModel):
    """Response for nonce generation."""
    nonce: str
    expires_at: datetime


class VerifyRequest(BaseModel):
    """Request to verify SIWE signature."""
    message: str
    signature: str


class RefreshRequest(BaseModel):
    """Request to refresh session."""
    refresh_token: str


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # wallet address
    session_id: str
    chain_id: int
    user_id: Optional[str] = None
    wallet_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    type: str = "access"  # "access" or "refresh"
