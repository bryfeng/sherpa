"""Agent configuration models for the planning system.

This module defines:
- AgentType: Types of autonomous agents
- AgentStatus: Status of an agent
- WalletConfig: Wallet configuration per chain
- ScheduleConfig: Scheduling configuration
- AgentConfig: Complete agent configuration (ERC-7208 ready)

Design Principle:
    All configuration is JSON-serializable for onchain storage.
    The AgentConfig can be stored in an ERC-7208 data container
    for trustless configuration management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .models import ChainId, PolicyConstraints


class AgentType(str, Enum):
    """Types of autonomous agents."""

    DCA = "dca"  # Dollar Cost Averaging
    MOMENTUM = "momentum"  # Momentum trading
    COPY = "copy"  # Copy trading
    POLYMARKET = "polymarket"  # Prediction markets
    YIELD = "yield"  # Yield rotation
    CUSTOM = "custom"  # Custom strategy


class AgentStatus(str, Enum):
    """Status of an autonomous agent."""

    ACTIVE = "active"  # Running normally
    PAUSED = "paused"  # Temporarily paused
    STOPPED = "stopped"  # Permanently stopped
    ERROR = "error"  # In error state


class WalletConfig(BaseModel):
    """Configuration for a wallet on a specific chain.

    Attributes:
        chain_id: Chain identifier
        address: Wallet address
        is_primary: Whether this is the primary wallet for the chain
    """

    chain_id: ChainId = Field(..., description="Chain ID (int for EVM, 'solana' for Solana)")
    address: str = Field(..., description="Wallet address")
    is_primary: bool = Field(default=True, description="Whether this is the primary wallet")

    class Config:
        frozen = True


class ScheduleConfig(BaseModel):
    """Configuration for agent scheduling.

    Supports cron expressions and interval-based scheduling.

    Attributes:
        type: Schedule type ('cron' or 'interval')
        cron: Cron expression (for type='cron')
        interval_seconds: Interval in seconds (for type='interval')
        timezone: Timezone for scheduling (default: UTC)
        enabled: Whether scheduling is enabled
    """

    type: str = Field(default="cron", description="Schedule type: 'cron' or 'interval'")
    cron: Optional[str] = Field(default="0 9 * * 1", description="Cron expression")
    interval_seconds: Optional[int] = Field(default=None, description="Interval in seconds")
    timezone: str = Field(default="UTC", description="Timezone for scheduling")
    enabled: bool = Field(default=True, description="Whether scheduling is enabled")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "cron": self.cron,
            "interval_seconds": self.interval_seconds,
            "timezone": self.timezone,
            "enabled": self.enabled,
        }


class PolicyConfig(BaseModel):
    """Policy configuration wrapper for Pydantic compatibility.

    Wraps PolicyConstraints for use in AgentConfig.
    """

    max_slippage_bps: int = Field(default=100, ge=0, le=1000)
    per_trade_usd_cap: Decimal = Field(default=Decimal("300"))
    daily_usd_cap: Decimal = Field(default=Decimal("5000"))
    allowed_chains: List[Union[int, str]] = Field(
        default_factory=lambda: [1, 8453, 42161, 10, 137, "solana"]
    )
    blocked_tokens: List[str] = Field(default_factory=list)
    auto_approve_threshold_usd: Decimal = Field(default=Decimal("100"))
    require_explicit_approval: bool = Field(default=False)

    class Config:
        json_encoders = {Decimal: str}

    def to_policy_constraints(self) -> PolicyConstraints:
        """Convert to PolicyConstraints dataclass."""
        return PolicyConstraints(
            max_slippage_bps=self.max_slippage_bps,
            per_trade_usd_cap=self.per_trade_usd_cap,
            daily_usd_cap=self.daily_usd_cap,
            allowed_chains=self.allowed_chains,
            blocked_tokens=self.blocked_tokens,
            auto_approve_threshold_usd=self.auto_approve_threshold_usd,
            require_explicit_approval=self.require_explicit_approval,
        )


class DCAStrategyParams(BaseModel):
    """Strategy-specific parameters for DCA strategy."""

    source_token: str = Field(default="USDC", description="Token to spend")
    target_tokens: List[str] = Field(..., description="Tokens to accumulate")
    amount_per_execution: Decimal = Field(..., description="USD amount per execution")
    allocation: Dict[str, float] = Field(..., description="Allocation weights")
    default_chain: Union[int, str] = Field(default=1, description="Default chain for swaps")

    class Config:
        json_encoders = {Decimal: str}

    def validate_allocation(self) -> List[str]:
        """Validate allocation weights sum to 1.0."""
        errors = []
        total = sum(self.allocation.values())
        if abs(total - 1.0) > 0.001:
            errors.append(f"Allocation weights must sum to 1.0, got {total}")

        for token in self.target_tokens:
            if token not in self.allocation:
                errors.append(f"Missing allocation for target token: {token}")

        return errors


class AgentConfig(BaseModel):
    """Complete agent configuration - designed for onchain storage.

    This model is JSON-serializable and can be stored in an ERC-7208
    data container for trustless agent configuration.

    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name
        type: Agent type (DCA, Momentum, etc.)
        wallets: List of wallet configurations
        policy: Policy constraints
        strategy_params: Strategy-specific parameters
        schedule: Scheduling configuration
        status: Current agent status
        owner: Owner address (for access control)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(default="", description="Human-readable name")
    type: AgentType = Field(..., description="Agent type")
    wallets: List[WalletConfig] = Field(..., description="Wallet configurations")
    policy: PolicyConfig = Field(default_factory=PolicyConfig, description="Policy constraints")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig, description="Schedule config")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Agent status")
    owner: Optional[str] = Field(default=None, description="Owner address")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }
        use_enum_values = True

    def get_primary_wallet(self, chain_id: Optional[ChainId] = None) -> Optional[WalletConfig]:
        """Get the primary wallet, optionally for a specific chain."""
        for wallet in self.wallets:
            if chain_id is not None and wallet.chain_id != chain_id:
                continue
            if wallet.is_primary:
                return wallet
        return self.wallets[0] if self.wallets else None

    def get_policy_constraints(self) -> PolicyConstraints:
        """Get PolicyConstraints dataclass from config."""
        return self.policy.to_policy_constraints()

    def to_cbor(self) -> bytes:
        """Serialize for onchain storage (ERC-7208)."""
        import cbor2

        return cbor2.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_cbor(cls, data: bytes) -> AgentConfig:
        """Deserialize from onchain storage."""
        import cbor2

        return cls.model_validate(cbor2.loads(data))


@dataclass
class PlanningContext:
    """Context for plan creation.

    Aggregates all the information needed to create a plan,
    whether from interactive chat or autonomous strategy.

    Attributes:
        conversation_id: Conversation ID (for interactive mode)
        agent_config: Agent configuration (for autonomous mode)
        wallet_address: Primary wallet address
        chain_id: Default chain ID
        portfolio_tokens: Current portfolio tokens
        timestamp: Context creation time
    """

    wallet_address: str
    chain_id: ChainId = 1
    conversation_id: Optional[str] = None
    agent_config: Optional[AgentConfig] = None
    portfolio_tokens: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_default_policy(self) -> PolicyConstraints:
        """Get policy constraints from agent config or defaults."""
        if self.agent_config:
            return self.agent_config.get_policy_constraints()
        return PolicyConstraints()
