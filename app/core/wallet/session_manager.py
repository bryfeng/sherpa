"""
Session key manager for autonomous agent wallet access.

Manages the lifecycle of session keys:
- Creation with configurable permissions and limits
- Validation before each action
- Usage tracking and limit enforcement
- Revocation
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from app.db.convex_client import ConvexClient, get_convex_client

from .models import (
    SessionKey,
    SessionKeyStatus,
    Permission,
    ValueLimit,
    ChainAllowlist,
    ContractAllowlist,
    TokenAllowlist,
    ActionRequest,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class SessionKeyError(Exception):
    """Base exception for session key errors."""
    pass


class SessionNotFoundError(SessionKeyError):
    """Session key not found."""
    pass


class SessionExpiredError(SessionKeyError):
    """Session key has expired."""
    pass


class SessionRevokedError(SessionKeyError):
    """Session key has been revoked."""
    pass


class PermissionDeniedError(SessionKeyError):
    """Action not permitted by session key."""
    pass


class LimitExceededError(SessionKeyError):
    """Value or transaction limit exceeded."""
    pass


class SessionKeyManager:
    """
    Manages session keys for autonomous agent wallet access.

    Session keys provide:
    - Time-limited access (default: 24 hours)
    - Permission scoping (swap, bridge, transfer, etc.)
    - Value limits (per-tx and total)
    - Contract allowlisting
    - Chain restrictions
    - Full audit trail
    """

    def __init__(self, convex: Optional[ConvexClient] = None):
        self.convex = convex or get_convex_client()
        self._cache: Dict[str, SessionKey] = {}

    async def create_session(
        self,
        wallet_address: str,
        permissions: Optional[Set[Permission]] = None,
        expires_in_hours: int = 24,
        max_value_per_tx_usd: Decimal = Decimal("1000"),
        max_total_value_usd: Decimal = Decimal("10000"),
        max_transactions: Optional[int] = None,
        allowed_chains: Optional[Set[int]] = None,
        allowed_contracts: Optional[Set[str]] = None,
        allowed_tokens: Optional[Set[str]] = None,
        agent_id: Optional[str] = None,
    ) -> SessionKey:
        """
        Create a new session key for a wallet.

        Args:
            wallet_address: The wallet address to create a session for
            permissions: Set of allowed actions (default: swap only)
            expires_in_hours: Session validity period (default: 24 hours)
            max_value_per_tx_usd: Maximum value per transaction
            max_total_value_usd: Maximum total value across all transactions
            max_transactions: Maximum number of transactions (None = unlimited)
            allowed_chains: Allowed chain IDs (None = all chains)
            allowed_contracts: Allowed contract addresses (None = all contracts)
            allowed_tokens: Allowed token addresses (None = all tokens)
            agent_id: ID of the agent using this session

        Returns:
            The created SessionKey
        """
        session = SessionKey(
            session_id=SessionKey.generate_session_id(),
            wallet_address=wallet_address.lower(),
            agent_id=agent_id,
            permissions=permissions or {Permission.SWAP},
            value_limits=ValueLimit(
                max_value_per_tx_usd=max_value_per_tx_usd,
                max_total_value_usd=max_total_value_usd,
                max_transactions=max_transactions,
            ),
            chain_allowlist=ChainAllowlist(
                allowed_chain_ids=allowed_chains or set()
            ),
            contract_allowlist=ContractAllowlist(
                allowed_addresses=allowed_contracts or set()
            ),
            token_allowlist=TokenAllowlist(
                allowed_tokens=allowed_tokens or set()
            ),
            expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours),
        )

        # Store in Convex
        await self.convex.mutation(
            "sessionKeys:create",
            session.to_dict(),
        )

        # Cache locally
        self._cache[session.session_id] = session

        logger.info(
            f"Created session key {session.session_id} for {wallet_address}, "
            f"expires at {session.expires_at.isoformat()}"
        )

        return session

    async def get_session(
        self,
        session_id: str,
        validate: bool = True,
    ) -> SessionKey:
        """
        Get a session key by ID.

        Args:
            session_id: The session ID
            validate: Whether to validate the session is still active

        Returns:
            The SessionKey

        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionExpiredError: If session has expired
            SessionRevokedError: If session was revoked
        """
        # Check cache first
        if session_id in self._cache:
            session = self._cache[session_id]
        else:
            # Fetch from Convex
            data = await self.convex.query(
                "sessionKeys:get",
                {"sessionId": session_id},
            )

            if not data:
                raise SessionNotFoundError(f"Session {session_id} not found")

            session = SessionKey.from_dict(data)
            self._cache[session_id] = session

        if validate:
            self._validate_session_status(session)

        return session

    async def validate_action(
        self,
        request: ActionRequest,
    ) -> ValidationResult:
        """
        Validate if an action is permitted by a session key.

        Args:
            request: The action request to validate

        Returns:
            ValidationResult with valid=True if permitted
        """
        errors: List[str] = []
        warnings: List[str] = []

        try:
            session = await self.get_session(request.session_id, validate=True)
        except SessionNotFoundError:
            return ValidationResult(
                valid=False,
                errors=["Session key not found"],
            )
        except SessionExpiredError:
            return ValidationResult(
                valid=False,
                errors=["Session key has expired"],
            )
        except SessionRevokedError:
            return ValidationResult(
                valid=False,
                errors=["Session key has been revoked"],
            )

        # Check permission
        if request.action_type not in session.permissions:
            errors.append(
                f"Action '{request.action_type.value}' not permitted. "
                f"Allowed: {[p.value for p in session.permissions]}"
            )

        # Check chain
        if not session.chain_allowlist.is_allowed(request.chain_id):
            errors.append(
                f"Chain {request.chain_id} not allowed. "
                f"Allowed: {list(session.chain_allowlist.allowed_chain_ids)}"
            )

        # Check contract
        if not session.contract_allowlist.is_allowed(request.contract_address):
            errors.append(
                f"Contract {request.contract_address} not allowed"
            )

        # Check value limits
        if not session.value_limits.can_execute(request.value_usd):
            if request.value_usd > session.value_limits.max_value_per_tx_usd:
                errors.append(
                    f"Transaction value ${request.value_usd} exceeds "
                    f"per-tx limit ${session.value_limits.max_value_per_tx_usd}"
                )
            elif (session.value_limits.total_value_used_usd + request.value_usd >
                  session.value_limits.max_total_value_usd):
                remaining = (session.value_limits.max_total_value_usd -
                           session.value_limits.total_value_used_usd)
                errors.append(
                    f"Transaction would exceed total limit. "
                    f"Remaining: ${remaining}"
                )
            elif (session.value_limits.max_transactions and
                  session.value_limits.transaction_count >= session.value_limits.max_transactions):
                errors.append(
                    f"Transaction count limit ({session.value_limits.max_transactions}) reached"
                )

        # Check tokens if specified
        if request.token_in:
            if not session.token_allowlist.is_allowed(request.chain_id, request.token_in):
                errors.append(f"Input token {request.token_in} not allowed")
        if request.token_out:
            if not session.token_allowlist.is_allowed(request.chain_id, request.token_out):
                errors.append(f"Output token {request.token_out} not allowed")

        # Add warnings for approaching limits
        if session.value_limits.max_transactions:
            remaining_txs = (session.value_limits.max_transactions -
                           session.value_limits.transaction_count)
            if remaining_txs <= 5:
                warnings.append(f"Only {remaining_txs} transactions remaining")

        remaining_value = (session.value_limits.max_total_value_usd -
                         session.value_limits.total_value_used_usd)
        if remaining_value < Decimal("100"):
            warnings.append(f"Only ${remaining_value} value remaining")

        # Check expiry
        time_remaining = session.expires_at - datetime.utcnow()
        if time_remaining < timedelta(hours=1):
            warnings.append(f"Session expires in {time_remaining}")

        return ValidationResult(
            valid=len(errors) == 0,
            session_key=session if len(errors) == 0 else None,
            errors=errors,
            warnings=warnings,
        )

    async def record_usage(
        self,
        session_id: str,
        action_type: Permission,
        value_usd: Decimal,
        tx_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record usage of a session key after successful action.

        Args:
            session_id: The session ID
            action_type: The action that was performed
            value_usd: The USD value of the transaction
            tx_hash: The transaction hash
            metadata: Additional metadata
        """
        session = await self.get_session(session_id, validate=False)

        # Update limits
        session.value_limits.record_transaction(value_usd)
        session.last_used_at = datetime.utcnow()

        # Check if limits exhausted
        if session.value_limits.max_transactions:
            if session.value_limits.transaction_count >= session.value_limits.max_transactions:
                session.status = SessionKeyStatus.EXHAUSTED
        if session.value_limits.total_value_used_usd >= session.value_limits.max_total_value_usd:
            session.status = SessionKeyStatus.EXHAUSTED

        # Update in Convex
        await self.convex.mutation(
            "sessionKeys:recordUsage",
            {
                "sessionId": session_id,
                "valueUsd": str(value_usd),
                "transactionCount": session.value_limits.transaction_count,
                "totalValueUsedUsd": str(session.value_limits.total_value_used_usd),
                "status": session.status.value,
                "lastUsedAt": int(datetime.utcnow().timestamp() * 1000),
                "usageEntry": {
                    "actionType": action_type.value,
                    "valueUsd": str(value_usd),
                    "txHash": tx_hash,
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    "metadata": metadata,
                },
            },
        )

        # Update cache
        self._cache[session_id] = session

        logger.info(
            f"Recorded usage for session {session_id}: "
            f"{action_type.value} ${value_usd}, "
            f"total: ${session.value_limits.total_value_used_usd}"
        )

    async def revoke_session(
        self,
        session_id: str,
        reason: str = "User requested revocation",
    ) -> None:
        """
        Revoke a session key.

        Args:
            session_id: The session ID to revoke
            reason: Reason for revocation
        """
        session = await self.get_session(session_id, validate=False)
        session.status = SessionKeyStatus.REVOKED
        session.revoked_at = datetime.utcnow()
        session.revoke_reason = reason

        # Update in Convex
        await self.convex.mutation(
            "sessionKeys:revoke",
            {
                "sessionId": session_id,
                "revokedAt": int(session.revoked_at.timestamp() * 1000),
                "revokeReason": reason,
            },
        )

        # Update cache
        self._cache[session_id] = session

        logger.info(f"Revoked session {session_id}: {reason}")

    async def list_sessions(
        self,
        wallet_address: str,
        include_expired: bool = False,
        include_revoked: bool = False,
    ) -> List[SessionKey]:
        """
        List all session keys for a wallet.

        Args:
            wallet_address: The wallet address
            include_expired: Include expired sessions
            include_revoked: Include revoked sessions

        Returns:
            List of SessionKey objects
        """
        data = await self.convex.query(
            "sessionKeys:listByWallet",
            {
                "walletAddress": wallet_address.lower(),
                "includeExpired": include_expired,
                "includeRevoked": include_revoked,
            },
        )

        sessions = [SessionKey.from_dict(d) for d in (data or [])]

        # Update cache
        for session in sessions:
            self._cache[session.session_id] = session

        return sessions

    async def cleanup_expired(self) -> int:
        """
        Mark expired sessions and return count.

        Returns:
            Number of sessions marked as expired
        """
        result = await self.convex.mutation(
            "sessionKeys:cleanupExpired",
            {"now": int(datetime.utcnow().timestamp() * 1000)},
        )

        count = result.get("expiredCount", 0) if result else 0

        # Clear cache of expired sessions
        self._cache = {
            k: v for k, v in self._cache.items()
            if v.status == SessionKeyStatus.ACTIVE
        }

        logger.info(f"Cleaned up {count} expired sessions")
        return count

    def _validate_session_status(self, session: SessionKey) -> None:
        """Validate session is still active."""
        if session.status == SessionKeyStatus.REVOKED:
            raise SessionRevokedError(
                f"Session {session.session_id} was revoked: {session.revoke_reason}"
            )

        if session.status == SessionKeyStatus.EXHAUSTED:
            raise LimitExceededError(
                f"Session {session.session_id} has exhausted its limits"
            )

        if datetime.utcnow() > session.expires_at:
            raise SessionExpiredError(
                f"Session {session.session_id} expired at {session.expires_at}"
            )

        if session.status != SessionKeyStatus.ACTIVE:
            raise SessionKeyError(
                f"Session {session.session_id} is not active: {session.status.value}"
            )


# Singleton instance
_session_manager: Optional[SessionKeyManager] = None


def get_session_manager() -> SessionKeyManager:
    """Get the singleton session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionKeyManager()
    return _session_manager
