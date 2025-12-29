"""
Session key and wallet management models.

Session keys provide time-limited, permission-scoped access for autonomous
agent execution without requiring full wallet control.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import secrets


class Permission(str, Enum):
    """Actions that can be permitted for a session key."""
    SWAP = "swap"
    BRIDGE = "bridge"
    TRANSFER = "transfer"
    APPROVE = "approve"
    STAKE = "stake"
    UNSTAKE = "unstake"
    CLAIM = "claim"
    WRAP = "wrap"
    UNWRAP = "unwrap"


class SessionKeyStatus(str, Enum):
    """Status of a session key."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    EXHAUSTED = "exhausted"  # Value/count limits reached


@dataclass
class ValueLimit:
    """Value limits for a session key."""
    max_value_per_tx_usd: Decimal = Decimal("1000")
    max_total_value_usd: Decimal = Decimal("10000")
    max_transactions: Optional[int] = None  # None = unlimited

    # Tracking (updated during use)
    total_value_used_usd: Decimal = Decimal("0")
    transaction_count: int = 0

    def can_execute(self, value_usd: Decimal) -> bool:
        """Check if a transaction with this value can be executed."""
        if value_usd > self.max_value_per_tx_usd:
            return False
        if self.total_value_used_usd + value_usd > self.max_total_value_usd:
            return False
        if self.max_transactions and self.transaction_count >= self.max_transactions:
            return False
        return True

    def record_transaction(self, value_usd: Decimal) -> None:
        """Record a transaction against the limits."""
        self.total_value_used_usd += value_usd
        self.transaction_count += 1


@dataclass
class ContractAllowlist:
    """Allowed contracts for a session key."""
    # If empty, all contracts allowed (less secure)
    allowed_addresses: Set[str] = field(default_factory=set)

    # Known safe contracts by category
    allowed_protocols: Set[str] = field(default_factory=set)  # e.g., "uniswap", "relay"

    def is_allowed(self, contract_address: str) -> bool:
        """Check if a contract is allowed."""
        if not self.allowed_addresses and not self.allowed_protocols:
            # No restrictions
            return True
        return contract_address.lower() in {a.lower() for a in self.allowed_addresses}


@dataclass
class ChainAllowlist:
    """Allowed chains for a session key."""
    allowed_chain_ids: Set[int] = field(default_factory=set)

    def is_allowed(self, chain_id: int) -> bool:
        """Check if a chain is allowed."""
        if not self.allowed_chain_ids:
            return True  # No restrictions
        return chain_id in self.allowed_chain_ids


@dataclass
class TokenAllowlist:
    """Allowed tokens for a session key."""
    # Token addresses (chain_id:address format)
    allowed_tokens: Set[str] = field(default_factory=set)

    # Allow all stablecoins
    allow_stablecoins: bool = True

    # Allow native tokens (ETH, MATIC, etc.)
    allow_native: bool = True

    def is_allowed(self, chain_id: int, token_address: str) -> bool:
        """Check if a token is allowed."""
        if not self.allowed_tokens:
            return True  # No restrictions

        key = f"{chain_id}:{token_address.lower()}"
        return key in {t.lower() for t in self.allowed_tokens}


@dataclass
class SessionKey:
    """
    A session key grants time-limited, permission-scoped access to execute
    transactions on behalf of a wallet.

    Session keys are stored in Convex and validated before each action.
    """
    session_id: str
    wallet_address: str
    agent_id: Optional[str] = None  # The agent using this key

    # Permissions
    permissions: Set[Permission] = field(default_factory=lambda: {Permission.SWAP})

    # Limits
    value_limits: ValueLimit = field(default_factory=ValueLimit)
    chain_allowlist: ChainAllowlist = field(default_factory=ChainAllowlist)
    contract_allowlist: ContractAllowlist = field(default_factory=ContractAllowlist)
    token_allowlist: TokenAllowlist = field(default_factory=TokenAllowlist)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))

    # Status
    status: SessionKeyStatus = SessionKeyStatus.ACTIVE
    revoked_at: Optional[datetime] = None
    revoke_reason: Optional[str] = None

    # Audit
    last_used_at: Optional[datetime] = None
    usage_log: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"sess_{secrets.token_urlsafe(32)}"

    @property
    def is_valid(self) -> bool:
        """Check if the session key is currently valid."""
        if self.status != SessionKeyStatus.ACTIVE:
            return False
        # Handle both naive and timezone-aware datetimes
        now = datetime.now(self.expires_at.tzinfo) if self.expires_at.tzinfo else datetime.utcnow()
        if now > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "sessionId": self.session_id,
            "walletAddress": self.wallet_address,
            "agentId": self.agent_id,
            "permissions": [p.value for p in self.permissions],
            "valueLimits": {
                "maxValuePerTxUsd": str(self.value_limits.max_value_per_tx_usd),
                "maxTotalValueUsd": str(self.value_limits.max_total_value_usd),
                "maxTransactions": self.value_limits.max_transactions,
                "totalValueUsedUsd": str(self.value_limits.total_value_used_usd),
                "transactionCount": self.value_limits.transaction_count,
            },
            "chainAllowlist": list(self.chain_allowlist.allowed_chain_ids),
            "contractAllowlist": list(self.contract_allowlist.allowed_addresses),
            "tokenAllowlist": list(self.token_allowlist.allowed_tokens),
            "createdAt": int(self.created_at.timestamp() * 1000),
            "expiresAt": int(self.expires_at.timestamp() * 1000),
            "status": self.status.value,
            "revokedAt": int(self.revoked_at.timestamp() * 1000) if self.revoked_at else None,
            "revokeReason": self.revoke_reason,
            "lastUsedAt": int(self.last_used_at.timestamp() * 1000) if self.last_used_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionKey":
        """Create from dictionary (from storage)."""
        value_limits_data = data.get("valueLimits", {})
        return cls(
            session_id=data["sessionId"],
            wallet_address=data["walletAddress"],
            agent_id=data.get("agentId"),
            permissions={Permission(p) for p in data.get("permissions", ["swap"])},
            value_limits=ValueLimit(
                max_value_per_tx_usd=Decimal(value_limits_data.get("maxValuePerTxUsd", "1000")),
                max_total_value_usd=Decimal(value_limits_data.get("maxTotalValueUsd", "10000")),
                max_transactions=value_limits_data.get("maxTransactions"),
                total_value_used_usd=Decimal(value_limits_data.get("totalValueUsedUsd", "0")),
                transaction_count=value_limits_data.get("transactionCount", 0),
            ),
            chain_allowlist=ChainAllowlist(
                allowed_chain_ids=set(data.get("chainAllowlist", []))
            ),
            contract_allowlist=ContractAllowlist(
                allowed_addresses=set(data.get("contractAllowlist", []))
            ),
            token_allowlist=TokenAllowlist(
                allowed_tokens=set(data.get("tokenAllowlist", []))
            ),
            created_at=datetime.fromtimestamp(data["createdAt"] / 1000),
            expires_at=datetime.fromtimestamp(data["expiresAt"] / 1000),
            status=SessionKeyStatus(data.get("status", "active")),
            revoked_at=datetime.fromtimestamp(data["revokedAt"] / 1000) if data.get("revokedAt") else None,
            revoke_reason=data.get("revokeReason"),
            last_used_at=datetime.fromtimestamp(data["lastUsedAt"] / 1000) if data.get("lastUsedAt") else None,
        )


@dataclass
class ActionRequest:
    """A request to perform an action using a session key."""
    session_id: str
    action_type: Permission
    chain_id: int
    contract_address: str
    value_usd: Decimal
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    description: str = ""


@dataclass
class ValidationResult:
    """Result of validating an action against a session key."""
    valid: bool
    session_key: Optional[SessionKey] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors) if self.errors else ""
