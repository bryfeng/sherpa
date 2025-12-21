"""Core data models for the planning system.

This module defines:
- TokenReference: Resolved token with chain, address, and metadata
- PolicyConstraints: Guardrails that can be stored/validated onchain
- Action: Validated intent ready for execution
- Plan: Collection of actions with status tracking
- DecisionLog: Audit trail for autonomous decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .protocol import ActionType, AmountSpec, TradeIntent


class PlanStatus(str, Enum):
    """Status of a plan in its lifecycle."""

    DRAFT = "draft"  # Being created, not ready
    PENDING_APPROVAL = "pending_approval"  # Waiting for user approval
    APPROVED = "approved"  # Approved, ready to execute
    EXECUTING = "executing"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed during execution
    CANCELLED = "cancelled"  # Cancelled by user


class ApprovalLevel(str, Enum):
    """Level of approval required for a plan."""

    NONE = "none"  # Auto-execute (within policy bounds)
    CONFIRMATION = "confirmation"  # Simple confirmation
    EXPLICIT_APPROVAL = "explicit_approval"  # Requires explicit approval


# Type alias for chain IDs (int for EVM, 'solana' for Solana)
ChainId = Union[int, str]


@dataclass
class TokenReference:
    """Resolved token reference with full metadata.

    This is the output of token resolution - a fully qualified token
    that can be used for quoting and execution.

    Attributes:
        chain_id: Chain the token is on
        address: Contract address (or mint for Solana)
        symbol: Token symbol
        decimals: Token decimals
        name: Full token name
        confidence: Confidence score from resolution (0.0 to 1.0)
        logo_uri: Optional logo URL
    """

    chain_id: ChainId
    address: str
    symbol: str
    decimals: int
    name: str = ""
    confidence: float = 1.0
    logo_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "address": self.address,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "name": self.name,
            "confidence": self.confidence,
            "logo_uri": self.logo_uri,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenReference:
        return cls(
            chain_id=data["chain_id"],
            address=data["address"],
            symbol=data["symbol"],
            decimals=data["decimals"],
            name=data.get("name", ""),
            confidence=data.get("confidence", 1.0),
            logo_uri=data.get("logo_uri"),
        )


@dataclass
class PolicyConstraints:
    """Guardrails that can be stored and validated onchain.

    These constraints are designed to be serializable to JSON/CBOR
    for storage in ERC-7208 data containers, enabling trustless
    policy enforcement.

    Attributes:
        max_slippage_bps: Maximum slippage in basis points
        per_trade_usd_cap: Maximum USD value per trade
        daily_usd_cap: Maximum USD value per day
        allowed_chains: List of allowed chain IDs
        blocked_tokens: List of blocked token addresses
        auto_approve_threshold_usd: Threshold for auto-approval
        require_explicit_approval: Whether to require explicit approval
    """

    max_slippage_bps: int = 100
    per_trade_usd_cap: Decimal = Decimal("300")
    daily_usd_cap: Decimal = Decimal("5000")
    allowed_chains: List[ChainId] = field(default_factory=lambda: [1, 8453, 42161, 10, 137, "solana"])
    blocked_tokens: List[str] = field(default_factory=list)
    auto_approve_threshold_usd: Decimal = Decimal("100")
    require_explicit_approval: bool = False

    def validate_intent(self, intent: TradeIntent, estimated_usd: Decimal) -> List[str]:
        """Validate an intent against policy constraints.

        Args:
            intent: The trade intent to validate
            estimated_usd: Estimated USD value of the trade

        Returns:
            List of policy violation messages (empty if valid)
        """
        violations: List[str] = []

        # Check chain
        if intent.chain_id not in self.allowed_chains:
            violations.append(f"Chain {intent.chain_id} is not in allowed chains")

        # Check blocked tokens
        token_in_addr = intent.token_in.address.lower() if hasattr(intent.token_in, "address") else ""
        token_out_addr = intent.token_out.address.lower() if hasattr(intent.token_out, "address") else ""

        for blocked in self.blocked_tokens:
            blocked_lower = blocked.lower()
            if token_in_addr == blocked_lower:
                violations.append(f"Token {intent.token_in.symbol} is blocked")
            if token_out_addr == blocked_lower:
                violations.append(f"Token {intent.token_out.symbol} is blocked")

        # Check per-trade cap
        if estimated_usd > self.per_trade_usd_cap:
            violations.append(
                f"Trade value ${estimated_usd} exceeds per-trade cap ${self.per_trade_usd_cap}"
            )

        return violations

    def get_approval_level(self, estimated_usd: Decimal) -> ApprovalLevel:
        """Determine the approval level required for a trade.

        Args:
            estimated_usd: Estimated USD value of the trade

        Returns:
            Required approval level
        """
        if self.require_explicit_approval:
            return ApprovalLevel.EXPLICIT_APPROVAL

        if estimated_usd <= self.auto_approve_threshold_usd:
            return ApprovalLevel.NONE

        if estimated_usd <= self.per_trade_usd_cap:
            return ApprovalLevel.CONFIRMATION

        return ApprovalLevel.EXPLICIT_APPROVAL

    def to_json(self) -> Dict[str, Any]:
        """Serialize for storage (onchain or Redis)."""
        return {
            "max_slippage_bps": self.max_slippage_bps,
            "per_trade_usd_cap": str(self.per_trade_usd_cap),
            "daily_usd_cap": str(self.daily_usd_cap),
            "allowed_chains": self.allowed_chains,
            "blocked_tokens": self.blocked_tokens,
            "auto_approve_threshold_usd": str(self.auto_approve_threshold_usd),
            "require_explicit_approval": self.require_explicit_approval,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> PolicyConstraints:
        return cls(
            max_slippage_bps=data.get("max_slippage_bps", 100),
            per_trade_usd_cap=Decimal(data.get("per_trade_usd_cap", "300")),
            daily_usd_cap=Decimal(data.get("daily_usd_cap", "5000")),
            allowed_chains=data.get("allowed_chains", [1, 8453, 42161, 10, 137, "solana"]),
            blocked_tokens=data.get("blocked_tokens", []),
            auto_approve_threshold_usd=Decimal(data.get("auto_approve_threshold_usd", "100")),
            require_explicit_approval=data.get("require_explicit_approval", False),
        )


@dataclass
class Action:
    """Validated intent ready for execution.

    An Action is created from a TradeIntent after:
    1. Policy validation passes
    2. Token resolution is complete
    3. Quote is fetched (estimated_usd is known)

    Attributes:
        action_id: Unique identifier for this action
        action_type: Type of action (swap, bridge, etc.)
        chain_id: Chain to execute on
        token_in: Resolved input token
        token_out: Resolved output token
        amount: Amount specification
        estimated_usd: Estimated USD value
        quote_payload: Provider-specific quote data
        status: Current action status
        tx_hash: Transaction hash (if executed)
        error: Error message (if failed)
    """

    action_id: str
    action_type: ActionType
    chain_id: ChainId
    token_in: TokenReference
    token_out: TokenReference
    amount: AmountSpec
    estimated_usd: Decimal
    quote_payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    tx_hash: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "chain_id": self.chain_id,
            "token_in": self.token_in.to_dict(),
            "token_out": self.token_out.to_dict(),
            "amount": self.amount.to_dict(),
            "estimated_usd": str(self.estimated_usd),
            "quote_payload": self.quote_payload,
            "status": self.status,
            "tx_hash": self.tx_hash,
            "error": self.error,
        }


@dataclass
class DecisionLog:
    """Audit trail for autonomous decisions.

    Records the full context of a decision for transparency and debugging.
    Designed to be stored onchain (ERC-7208) for verifiable agent behavior.

    Attributes:
        decision_id: Unique identifier
        agent_id: Agent that made the decision
        strategy_type: Strategy that produced the decision
        timestamp: When the decision was made
        inputs: Data sources used (e.g., price feeds)
        features: Computed features (e.g., momentum scores)
        policy: Policy constraints that were applied
        intents: Intents produced by the strategy
        actions: Actions that were validated and executed
        transactions: Transaction details
        result: Final outcome
    """

    decision_id: str
    agent_id: str
    strategy_type: str
    timestamp: datetime
    inputs: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    intents: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "strategy_type": self.strategy_type,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "features": self.features,
            "policy": self.policy,
            "intents": self.intents,
            "actions": self.actions,
            "transactions": self.transactions,
            "result": self.result,
        }


@dataclass
class Plan:
    """A collection of actions to be executed.

    Plans can be:
    - Interactive: Created from user chat, requires approval
    - Autonomous: Created by strategy, may auto-execute

    Attributes:
        plan_id: Unique identifier
        agent_id: Agent that owns this plan (if autonomous)
        conversation_id: Conversation that created this plan (if interactive)
        intents: Original intents from strategy
        actions: Validated actions ready for execution
        policy: Policy constraints applied
        status: Current plan status
        approval_level: Level of approval required
        decision_log: Audit trail for this plan
        created_at: When the plan was created
        updated_at: When the plan was last updated
        warnings: Any warnings generated during planning
    """

    plan_id: str
    intents: List[TradeIntent]
    actions: List[Action]
    policy: PolicyConstraints
    status: PlanStatus = PlanStatus.DRAFT
    approval_level: ApprovalLevel = ApprovalLevel.CONFIRMATION
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    decision_log: Optional[DecisionLog] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    warnings: List[str] = field(default_factory=list)

    def total_estimated_usd(self) -> Decimal:
        """Calculate total estimated USD value of all actions."""
        return sum((a.estimated_usd for a in self.actions), Decimal("0"))

    def is_executable(self) -> bool:
        """Check if the plan is ready to execute."""
        return self.status == PlanStatus.APPROVED and len(self.actions) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "intents": [i.to_dict() for i in self.intents],
            "actions": [a.to_dict() for a in self.actions],
            "policy": self.policy.to_json(),
            "status": self.status.value,
            "approval_level": self.approval_level.value,
            "total_estimated_usd": str(self.total_estimated_usd()),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "warnings": self.warnings,
        }
