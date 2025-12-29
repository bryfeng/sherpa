"""
Strategy State Machine Models

Defines states, transitions, and events for strategy execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class StrategyState(str, Enum):
    """States a strategy can be in during execution."""

    IDLE = "idle"                         # Not running, ready to start
    ANALYZING = "analyzing"               # Gathering market data, evaluating conditions
    PLANNING = "planning"                 # Determining actions to take
    AWAITING_APPROVAL = "awaiting_approval"  # Waiting for human approval
    EXECUTING = "executing"               # Executing transactions
    MONITORING = "monitoring"             # Waiting for transaction confirmations
    COMPLETED = "completed"               # Successfully finished
    FAILED = "failed"                     # Failed with error
    PAUSED = "paused"                     # Manually paused by user
    CANCELLED = "cancelled"               # Cancelled before completion


class StateTransitionTrigger(str, Enum):
    """What triggered a state transition."""

    AUTOMATIC = "automatic"               # System-triggered transition
    USER_ACTION = "user_action"           # User initiated (approve, pause, cancel)
    TIMEOUT = "timeout"                   # Timed out waiting
    ERROR = "error"                       # Error occurred
    CONDITION_MET = "condition_met"       # Strategy condition was met
    TRANSACTION_CONFIRMED = "tx_confirmed"  # Transaction was confirmed
    TRANSACTION_FAILED = "tx_failed"      # Transaction failed


@dataclass
class StateTransition:
    """Record of a state transition."""

    id: str = field(default_factory=lambda: str(uuid4()))
    from_state: StrategyState = StrategyState.IDLE
    to_state: StrategyState = StrategyState.IDLE
    trigger: StateTransitionTrigger = StateTransitionTrigger.AUTOMATIC
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context about the transition
    reason: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Error info if transition was due to failure
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fromState": self.from_state.value,
            "toState": self.to_state.value,
            "trigger": self.trigger.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "context": self.context,
            "errorMessage": self.error_message,
            "errorCode": self.error_code,
        }


@dataclass
class ExecutionStep:
    """A single step in strategy execution."""

    id: str = field(default_factory=lambda: str(uuid4()))
    step_number: int = 0
    description: str = ""
    action_type: str = ""  # "swap", "bridge", "transfer", "approve", etc.

    # Status
    status: str = "pending"  # pending, executing, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Input/output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # Transaction info
    tx_hash: Optional[str] = None
    chain_id: Optional[int] = None
    gas_used: Optional[int] = None
    gas_price_gwei: Optional[float] = None

    # Error info
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stepNumber": self.step_number,
            "description": self.description,
            "actionType": self.action_type,
            "status": self.status,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "inputData": self.input_data,
            "outputData": self.output_data,
            "txHash": self.tx_hash,
            "chainId": self.chain_id,
            "gasUsed": self.gas_used,
            "gasPriceGwei": self.gas_price_gwei,
            "errorMessage": self.error_message,
            "retryCount": self.retry_count,
        }


@dataclass
class ExecutionContext:
    """Context maintained during strategy execution."""

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    strategy_id: str = ""
    wallet_address: str = ""

    # Current state
    current_state: StrategyState = StrategyState.IDLE
    state_entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Execution plan
    steps: List[ExecutionStep] = field(default_factory=list)
    current_step_index: int = 0

    # State history
    state_history: List[StateTransition] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Approval info
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    recoverable: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if execution is in a terminal state."""
        return self.current_state in (
            StrategyState.COMPLETED,
            StrategyState.FAILED,
            StrategyState.CANCELLED,
        )

    @property
    def is_active(self) -> bool:
        """Check if execution is actively running."""
        return self.current_state in (
            StrategyState.ANALYZING,
            StrategyState.PLANNING,
            StrategyState.EXECUTING,
            StrategyState.MONITORING,
        )

    @property
    def current_step(self) -> Optional[ExecutionStep]:
        """Get the current execution step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def completed_steps(self) -> List[ExecutionStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == "completed"]

    @property
    def failed_steps(self) -> List[ExecutionStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == "failed"]

    @property
    def progress_percent(self) -> float:
        """Calculate execution progress as a percentage."""
        if not self.steps:
            return 0.0
        completed = len([s for s in self.steps if s.status in ("completed", "skipped")])
        return (completed / len(self.steps)) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total execution duration."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executionId": self.execution_id,
            "strategyId": self.strategy_id,
            "walletAddress": self.wallet_address,
            "currentState": self.current_state.value,
            "stateEnteredAt": self.state_entered_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "currentStepIndex": self.current_step_index,
            "stateHistory": [t.to_dict() for t in self.state_history],
            "createdAt": self.created_at.isoformat(),
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "requiresApproval": self.requires_approval,
            "approvalReason": self.approval_reason,
            "approvedBy": self.approved_by,
            "approvedAt": self.approved_at.isoformat() if self.approved_at else None,
            "errorMessage": self.error_message,
            "errorCode": self.error_code,
            "recoverable": self.recoverable,
            "isTerminal": self.is_terminal,
            "isActive": self.is_active,
            "progressPercent": self.progress_percent,
            "durationSeconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
        message: Optional[str] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.message = message or f"Cannot transition from {from_state.value} to {to_state.value}"
        super().__init__(self.message)


class ExecutionError(Exception):
    """Raised when strategy execution fails."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(message)
