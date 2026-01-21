"""
Copy Trading Models

Data structures for copy trading configuration, relationships, and execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import uuid


class TradeAction(str, Enum):
    """Types of trades that can be copied."""

    SWAP = "swap"
    BRIDGE = "bridge"
    LP_ADD = "lp_add"
    LP_REMOVE = "lp_remove"


class SizingMode(str, Enum):
    """Position sizing modes for copy trading."""

    PERCENTAGE = "percentage"  # Size as % of follower's portfolio
    FIXED = "fixed"  # Fixed USD amount per trade
    PROPORTIONAL = "proportional"  # Proportional to leader's trade size


class CopyExecutionStatus(str, Enum):
    """Status of a copy trade execution."""

    PENDING = "pending"  # Waiting to execute
    PENDING_APPROVAL = "pending_approval"  # Waiting for user approval
    QUEUED = "queued"  # In execution queue
    EXECUTING = "executing"  # Currently executing
    COMPLETED = "completed"  # Successfully executed
    FAILED = "failed"  # Execution failed
    SKIPPED = "skipped"  # Skipped due to filters/limits
    CANCELLED = "cancelled"  # Manually cancelled
    EXPIRED = "expired"  # Approval window expired


class SkipReason(str, Enum):
    """Reasons for skipping a copy trade."""

    VALUE_TOO_LOW = "value_too_low"
    VALUE_TOO_HIGH = "value_too_high"
    TOKEN_BLACKLISTED = "token_blacklisted"
    TOKEN_NOT_WHITELISTED = "token_not_whitelisted"
    ACTION_NOT_ALLOWED = "action_not_allowed"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    VOLUME_LIMIT_REACHED = "volume_limit_reached"
    TRADE_TOO_OLD = "trade_too_old"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    SLIPPAGE_TOO_HIGH = "slippage_too_high"
    SESSION_EXPIRED = "session_expired"
    PAUSED = "paused"


class CopyConfig(BaseModel):
    """Configuration for following a wallet."""

    # Target wallet (leader)
    leader_address: str
    leader_chain: str
    leader_label: Optional[str] = None

    # Sizing strategy
    sizing_mode: SizingMode = SizingMode.PERCENTAGE
    size_value: Decimal = Decimal("5")  # % of portfolio, fixed USD, or multiplier

    # Filters
    min_trade_usd: Decimal = Decimal("10")
    max_trade_usd: Optional[Decimal] = None
    token_whitelist: Optional[List[str]] = None  # Only copy these tokens
    token_blacklist: Optional[List[str]] = None  # Never copy these tokens
    allowed_actions: List[str] = Field(default_factory=lambda: ["swap"])

    # Timing
    delay_seconds: int = 0  # 0 = immediate, >0 = delayed execution
    max_delay_seconds: int = 300  # Skip trade if older than this

    # Risk controls
    max_slippage_bps: int = 100  # 1%
    max_daily_trades: int = 20
    max_daily_volume_usd: Decimal = Decimal("10000")

    # Session key for autonomous execution
    session_key_id: Optional[str] = None

    def is_token_allowed(self, token_address: str) -> bool:
        """Check if a token is allowed based on whitelist/blacklist."""
        token_lower = token_address.lower()

        # Check blacklist first
        if self.token_blacklist:
            blacklist = {t.lower() for t in self.token_blacklist}
            if token_lower in blacklist:
                return False

        # Check whitelist if specified
        if self.token_whitelist:
            whitelist = {t.lower() for t in self.token_whitelist}
            return token_lower in whitelist

        return True

    def is_action_allowed(self, action: "TradeAction | str") -> bool:
        """Check if an action type is allowed."""
        action_str = action.value if isinstance(action, TradeAction) else action
        return action_str.lower() in [a.lower() for a in self.allowed_actions]


class CopyRelationship(BaseModel):
    """Represents a follower -> leader copy relationship."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    follower_address: str
    follower_chain: str

    # Configuration
    config: CopyConfig

    # Status
    is_active: bool = True
    is_paused: bool = False
    pause_reason: Optional[str] = None

    # Daily tracking (reset daily)
    daily_trade_count: int = 0
    daily_volume_usd: Decimal = Decimal("0")
    daily_reset_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Lifetime stats
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    skipped_trades: int = 0
    total_volume_usd: Decimal = Decimal("0")
    total_pnl_usd: Optional[Decimal] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_copy_at: Optional[datetime] = None

    def can_execute_trade(self, trade_value_usd: Decimal) -> tuple[bool, Optional[SkipReason]]:
        """Check if we can execute another trade today."""
        if not self.is_active:
            return False, SkipReason.PAUSED

        if self.is_paused:
            return False, SkipReason.PAUSED

        # Reset daily counters if needed
        self._check_daily_reset()

        # Check daily trade limit
        if self.daily_trade_count >= self.config.max_daily_trades:
            return False, SkipReason.DAILY_LIMIT_REACHED

        # Check daily volume limit
        if self.daily_volume_usd + trade_value_usd > self.config.max_daily_volume_usd:
            return False, SkipReason.VOLUME_LIMIT_REACHED

        return True, None

    def _check_daily_reset(self):
        """Reset daily counters if it's a new day."""
        now = datetime.now(timezone.utc)
        if now.date() > self.daily_reset_at.date():
            self.daily_trade_count = 0
            self.daily_volume_usd = Decimal("0")
            self.daily_reset_at = now

    def record_trade(
        self,
        success: bool,
        volume_usd: Decimal,
        skipped: bool = False,
    ):
        """Record a trade attempt."""
        self.total_trades += 1
        self.daily_trade_count += 1

        if skipped:
            self.skipped_trades += 1
        elif success:
            self.successful_trades += 1
            self.total_volume_usd += volume_usd
            self.daily_volume_usd += volume_usd
        else:
            self.failed_trades += 1

        self.last_copy_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class TradeSignal(BaseModel):
    """A trade signal from a leader that may trigger a copy."""

    # Source
    leader_address: str
    leader_chain: str

    # Transaction info
    tx_hash: str
    block_number: Optional[int] = None
    timestamp: datetime

    # Trade details
    action: TradeAction = TradeAction.SWAP
    token_in_address: str
    token_in_symbol: Optional[str] = None
    token_in_amount: Optional[Decimal] = None
    token_out_address: str
    token_out_symbol: Optional[str] = None
    token_out_amount: Optional[Decimal] = None

    # Value
    value_usd: Optional[Decimal] = None

    # Metadata
    dex: Optional[str] = None  # "uniswap_v3", "jupiter", etc.
    dex_name: Optional[str] = None  # Alias for dex
    raw_data: Optional[Dict[str, Any]] = None


class CopyExecution(BaseModel):
    """Record of a copy trade execution attempt."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relationship_id: str
    signal: TradeSignal

    # Execution details
    status: CopyExecutionStatus = CopyExecutionStatus.PENDING
    skip_reason: Optional[SkipReason] = None

    # Sizing
    calculated_size_usd: Optional[Decimal] = None
    actual_size_usd: Optional[Decimal] = None

    # Transaction
    tx_hash: Optional[str] = None
    gas_used: Optional[int] = None
    gas_price_gwei: Optional[Decimal] = None
    gas_cost_usd: Optional[Decimal] = None

    # Result
    token_out_amount: Optional[Decimal] = None
    slippage_bps: Optional[int] = None
    error_message: Optional[str] = None

    # Timing
    signal_received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_started_at: Optional[datetime] = None
    execution_completed_at: Optional[datetime] = None

    @property
    def execution_delay_seconds(self) -> Optional[float]:
        """Calculate delay from signal to execution start."""
        if self.execution_started_at:
            return (self.execution_started_at - self.signal_received_at).total_seconds()
        return None

    @property
    def execution_duration_seconds(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.execution_started_at and self.execution_completed_at:
            return (self.execution_completed_at - self.execution_started_at).total_seconds()
        return None


class LeaderProfile(BaseModel):
    """Performance profile for a copyable wallet (leader)."""

    address: str
    chain: str

    # Labels
    label: Optional[str] = None
    notes: Optional[str] = None

    # Performance metrics
    total_trades: int = 0
    win_rate: Optional[float] = None  # % of profitable trades
    avg_trade_pnl_pct: Optional[float] = None
    total_pnl_usd: Optional[Decimal] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None

    # Activity metrics
    avg_trades_per_day: Optional[float] = None
    avg_hold_time_hours: Optional[float] = None
    most_traded_tokens: List[str] = Field(default_factory=list)
    preferred_sectors: List[str] = Field(default_factory=list)

    # Risk metrics
    avg_position_size_pct: Optional[float] = None
    max_position_size_pct: Optional[float] = None
    uses_leverage: bool = False

    # Social proof
    follower_count: int = 0
    total_copied_volume_usd: Decimal = Decimal("0")

    # Status
    is_active: bool = True
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Data quality
    data_quality_score: float = 0.0  # 0-1, how much history we have
    last_analyzed_at: Optional[datetime] = None

    @property
    def risk_score(self) -> float:
        """Calculate a risk score (0-1, higher = riskier)."""
        score = 0.5  # Base score

        # Higher max drawdown = higher risk
        if self.max_drawdown_pct:
            if self.max_drawdown_pct > 50:
                score += 0.3
            elif self.max_drawdown_pct > 30:
                score += 0.2
            elif self.max_drawdown_pct > 20:
                score += 0.1

        # Leverage = higher risk
        if self.uses_leverage:
            score += 0.2

        # Lower win rate = higher risk
        if self.win_rate:
            if self.win_rate < 40:
                score += 0.2
            elif self.win_rate < 50:
                score += 0.1

        return min(score, 1.0)


class CopyTradingStats(BaseModel):
    """Aggregated stats for a user's copy trading activity."""

    user_id: str

    # Relationship counts
    active_relationships: int = 0
    total_relationships: int = 0

    # Trade counts
    total_copy_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    skipped_trades: int = 0

    # Volume
    total_volume_usd: Decimal = Decimal("0")
    today_volume_usd: Decimal = Decimal("0")

    # P&L
    total_pnl_usd: Optional[Decimal] = None
    realized_pnl_usd: Optional[Decimal] = None
    unrealized_pnl_usd: Optional[Decimal] = None

    # By leader
    by_leader: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Timing
    last_copy_at: Optional[datetime] = None
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
