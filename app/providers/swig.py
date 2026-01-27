"""
Swig Smart Wallet Provider (Solana).

Provides role-based session management for Solana smart wallets.

Features:
- Session authority creation with spending limits
- Role-based permissions (agent, dca, copy_trading)
- Transaction signing via session keys
- Gas reimbursement coordination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx

from .base import Provider
from ..config import settings

logger = logging.getLogger(__name__)


class SwigError(Exception):
    """Swig provider error."""
    pass


class SwigSessionError(SwigError):
    """Session-specific error."""
    def __init__(self, message: str, session_id: Optional[str] = None):
        super().__init__(message)
        self.session_id = session_id


@dataclass
class SwigConfig:
    base_url: str
    api_key: str


@dataclass
class SwigSessionAuthority:
    """Represents a Swig session authority/role."""
    authority_id: str
    wallet_address: str
    role: str = "agent"
    spending_limit_usd: Optional[Decimal] = None
    total_spent_usd: Optional[Decimal] = None
    allowed_programs: List[str] = field(default_factory=list)
    allowed_tokens: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    status: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "SwigSessionAuthority":
        expires_at = None
        if data.get("expiresAt"):
            try:
                expires_at = datetime.fromisoformat(data["expiresAt"])
            except ValueError:
                expires_at = None

        spending_limit = None
        if data.get("spendingLimitUsd"):
            spending_limit = Decimal(str(data["spendingLimitUsd"]))

        total_spent = None
        if data.get("totalSpentUsd"):
            total_spent = Decimal(str(data["totalSpentUsd"]))

        return cls(
            authority_id=data.get("authorityId") or data.get("id") or "",
            wallet_address=data.get("walletAddress") or data.get("wallet") or "",
            role=data.get("role", "agent"),
            spending_limit_usd=spending_limit,
            total_spent_usd=total_spent,
            allowed_programs=data.get("allowedPrograms", []),
            allowed_tokens=data.get("allowedTokens", []),
            allowed_actions=data.get("allowedActions", []),
            expires_at=expires_at,
            status=data.get("status"),
            raw=data,
        )

    @property
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        if self.status != "active":
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True

    @property
    def spending_remaining_usd(self) -> Optional[Decimal]:
        """Calculate remaining spending budget."""
        if self.spending_limit_usd is None:
            return None
        spent = self.total_spent_usd or Decimal("0")
        return max(Decimal("0"), self.spending_limit_usd - spent)


@dataclass
class SwigWalletStatus:
    """Status of a Swig smart wallet."""
    wallet_address: str
    owner_address: str
    is_deployed: bool = False
    active_sessions: int = 0
    total_sessions: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "SwigWalletStatus":
        return cls(
            wallet_address=data.get("walletAddress") or data.get("address") or "",
            owner_address=data.get("ownerAddress") or data.get("owner") or "",
            is_deployed=data.get("isDeployed", False),
            active_sessions=data.get("activeSessions", 0),
            total_sessions=data.get("totalSessions", 0),
            raw=data,
        )


class SwigProvider(Provider):
    """
    Provider for Swig Solana smart wallet operations.

    Handles:
    - Wallet status queries
    - Session authority creation and management
    - Role-based permission grants
    - Transaction signing coordination
    - Gas/fee reimbursement

    Usage:
        provider = get_swig_provider()

        # Create a session with spending limit
        session = await provider.create_session_authority(
            wallet_address="...",
            role="agent",
            spending_limit_usd=Decimal("100"),
            allowed_programs=["JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"],
            allowed_actions=["swap"],
            expires_at=datetime.utcnow() + timedelta(days=7),
        )

        # Validate session before execution
        is_valid = await provider.validate_session(session.authority_id, amount_usd=50)
    """

    name = "swig"
    timeout_s = 20

    def __init__(self, config: Optional[SwigConfig] = None) -> None:
        self._config = config or SwigConfig(
            base_url=settings.swig_base_url,
            api_key=settings.swig_api_key,
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def ready(self) -> bool:
        return bool(self._config.base_url and self._config.api_key) and settings.enable_swig

    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {"status": "disabled", "reason": "Swig not configured"}

        try:
            result = await self._request("GET", "/health")
            return {"status": "healthy", "result": result}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    # =========================================================================
    # Wallet Management
    # =========================================================================

    async def get_wallet_status(self, wallet_address: str) -> SwigWalletStatus:
        """Get status of a Swig smart wallet."""
        try:
            result = await self._request("GET", f"/wallets/{wallet_address}")
            if not isinstance(result, dict):
                raise SwigError("Invalid Swig response for wallet status")
            return SwigWalletStatus.from_api(result)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return SwigWalletStatus(
                    wallet_address=wallet_address,
                    owner_address="",
                    is_deployed=False,
                )
            raise

    # =========================================================================
    # Session Authority Management
    # =========================================================================

    async def create_session_authority(
        self,
        wallet_address: str,
        role: str = "agent",
        spending_limit_usd: Optional[Decimal] = None,
        allowed_programs: Optional[List[str]] = None,
        allowed_tokens: Optional[List[str]] = None,
        allowed_actions: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        permissions: Optional[Dict[str, Any]] = None,
    ) -> SwigSessionAuthority:
        """
        Create a new session authority with role-based permissions.

        Args:
            wallet_address: Swig wallet address
            role: Role name (agent, dca, copy_trading)
            spending_limit_usd: Maximum spending in USD
            allowed_programs: List of allowed Solana program IDs
            allowed_tokens: List of allowed token mint addresses
            allowed_actions: List of allowed actions (swap, transfer)
            expires_at: Session expiration datetime
            permissions: Raw permissions dict (legacy support)

        Returns:
            SwigSessionAuthority with session details
        """
        payload: Dict[str, Any] = {
            "walletAddress": wallet_address,
            "role": role,
        }

        # Build permissions from structured args or use raw dict
        if permissions:
            payload["permissions"] = permissions
        else:
            perms: Dict[str, Any] = {}
            if spending_limit_usd is not None:
                perms["spendingLimitUsd"] = float(spending_limit_usd)
            if allowed_programs:
                perms["allowedPrograms"] = allowed_programs
            if allowed_tokens:
                perms["allowedTokens"] = allowed_tokens
            if allowed_actions:
                perms["allowedActions"] = allowed_actions
            if perms:
                payload["permissions"] = perms

        if expires_at:
            payload["expiresAt"] = expires_at.isoformat()

        logger.info(f"Creating Swig session for wallet {wallet_address}, role={role}")
        result = await self._request("POST", "/session-authorities", json=payload)

        if not isinstance(result, dict):
            raise SwigError("Invalid Swig response for session authority creation")
        return SwigSessionAuthority.from_api(result)

    async def get_session_authority(self, authority_id: str) -> SwigSessionAuthority:
        """Get session authority by ID."""
        result = await self._request("GET", f"/session-authorities/{authority_id}")
        if not isinstance(result, dict):
            raise SwigError("Invalid Swig response for session authority")
        return SwigSessionAuthority.from_api(result)

    async def list_session_authorities(
        self,
        wallet_address: str,
        include_expired: bool = False,
        include_revoked: bool = False,
    ) -> List[SwigSessionAuthority]:
        """List all session authorities for a wallet."""
        params = {"walletAddress": wallet_address}
        if include_expired:
            params["includeExpired"] = "true"
        if include_revoked:
            params["includeRevoked"] = "true"

        result = await self._request("GET", "/session-authorities", params=params)
        if not isinstance(result, list):
            result = result.get("sessions", []) if isinstance(result, dict) else []
        return [SwigSessionAuthority.from_api(s) for s in result]

    async def revoke_session_authority(self, authority_id: str) -> Dict[str, Any]:
        """Revoke a session authority."""
        logger.info(f"Revoking Swig session {authority_id}")
        return await self._request("POST", f"/session-authorities/{authority_id}/revoke", json={})

    async def validate_session(
        self,
        authority_id: str,
        amount_usd: Optional[Decimal] = None,
        action: Optional[str] = None,
        program_id: Optional[str] = None,
        token_mint: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a session can execute an operation.

        Args:
            authority_id: Session authority ID
            amount_usd: Amount in USD (for spending limit check)
            action: Action type (swap, transfer)
            program_id: Program to interact with
            token_mint: Token mint address

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            session = await self.get_session_authority(authority_id)
        except Exception as e:
            return False, f"Failed to fetch session: {e}"

        # Check active status
        if not session.is_active:
            return False, f"Session is not active (status: {session.status})"

        # Check spending limit
        if amount_usd is not None and session.spending_remaining_usd is not None:
            if amount_usd > session.spending_remaining_usd:
                return False, f"Exceeds spending limit: ${amount_usd} > ${session.spending_remaining_usd} remaining"

        # Check allowed actions
        if action and session.allowed_actions:
            if action not in session.allowed_actions:
                return False, f"Action '{action}' not allowed"

        # Check allowed programs
        if program_id and session.allowed_programs:
            if program_id not in session.allowed_programs:
                return False, f"Program '{program_id}' not allowed"

        # Check allowed tokens
        if token_mint and session.allowed_tokens:
            if token_mint not in session.allowed_tokens:
                return False, f"Token '{token_mint}' not allowed"

        return True, None

    async def record_session_usage(
        self,
        authority_id: str,
        spent_usd: Decimal,
        tx_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record spending against a session authority."""
        payload: Dict[str, Any] = {
            "spentUsd": float(spent_usd),
        }
        if tx_signature:
            payload["txSignature"] = tx_signature

        return await self._request(
            "POST",
            f"/session-authorities/{authority_id}/record-usage",
            json=payload,
        )

    # =========================================================================
    # Reimbursement
    # =========================================================================

    async def build_reimbursement(self, authority_id: str, tx_signature: str) -> Dict[str, Any]:
        """Build a reimbursement transaction for gas fees."""
        payload = {"txSignature": tx_signature}
        return await self._request(
            "POST",
            f"/session-authorities/{authority_id}/reimburse",
            json=payload,
        )

    # =========================================================================
    # HTTP Client
    # =========================================================================

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not await self.ready():
            raise SwigError("Swig provider is not configured")

        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)

        url = f"{self._config.base_url.rstrip('/')}{path}"
        response = await self._client.request(
            method,
            url,
            json=json,
            params=params,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
        )
        response.raise_for_status()
        return response.json()


_swig_provider: Optional[SwigProvider] = None


def get_swig_provider() -> SwigProvider:
    global _swig_provider
    if _swig_provider is None:
        _swig_provider = SwigProvider()
    return _swig_provider

