"""
Wallet Management Module

Provides session key management for autonomous agent execution:
- SessionKeyManager: Create, validate, and revoke session keys
- Permission scoping: Control what actions are allowed
- Value limits: Cap transaction values
- Contract allowlisting: Restrict to trusted contracts

Usage:
    from app.core.wallet import (
        SessionKeyManager,
        get_session_manager,
        Permission,
        SessionKey,
    )

    # Get the singleton manager
    manager = get_session_manager()

    # Create a session key
    session = await manager.create_session(
        wallet_address="0x...",
        permissions={Permission.SWAP, Permission.BRIDGE},
        max_value_per_tx_usd=Decimal("500"),
        expires_in_hours=24,
    )

    # Validate an action
    result = await manager.validate_action(ActionRequest(
        session_id=session.session_id,
        action_type=Permission.SWAP,
        chain_id=1,
        contract_address="0x...",
        value_usd=Decimal("100"),
    ))

    if result.valid:
        # Execute the action
        ...
        # Record usage
        await manager.record_usage(
            session_id=session.session_id,
            action_type=Permission.SWAP,
            value_usd=Decimal("100"),
            tx_hash="0x...",
        )
"""

from .models import (
    Permission,
    SessionKeyStatus,
    ValueLimit,
    ChainAllowlist,
    ContractAllowlist,
    TokenAllowlist,
    SessionKey,
    ActionRequest,
    ValidationResult,
)

from .session_manager import (
    SessionKeyManager,
    SessionKeyError,
    SessionNotFoundError,
    SessionExpiredError,
    SessionRevokedError,
    PermissionDeniedError,
    LimitExceededError,
    get_session_manager,
)
from .swig_session import (
    SwigSessionAuthorityManager,
    SwigSessionConfig,
    SwigSessionError,
)

__all__ = [
    # Models
    "Permission",
    "SessionKeyStatus",
    "ValueLimit",
    "ChainAllowlist",
    "ContractAllowlist",
    "TokenAllowlist",
    "SessionKey",
    "ActionRequest",
    "ValidationResult",
    # Manager
    "SessionKeyManager",
    "SessionKeyError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "SessionRevokedError",
    "PermissionDeniedError",
    "LimitExceededError",
    "get_session_manager",
    "SwigSessionAuthorityManager",
    "SwigSessionConfig",
    "SwigSessionError",
]
