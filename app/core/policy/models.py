"""
Policy Engine Models

Defines the core types for the unified policy enforcement system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class PolicyType(str, Enum):
    """Types of policies that can be evaluated."""
    SESSION = "session"      # Session key constraints
    RISK = "risk"            # User risk preferences
    SYSTEM = "system"        # Platform-wide rules
    FEE = "fee"              # Fee policy (gas abstraction / paymaster)


class ViolationSeverity(str, Enum):
    """Severity of a policy violation."""
    BLOCK = "block"          # Action cannot proceed
    WARN = "warn"            # Action can proceed with warning
    INFO = "info"            # Informational only


class RiskLevel(str, Enum):
    """Risk level assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyViolation:
    """A single policy violation."""
    policy_type: PolicyType
    policy_name: str
    severity: ViolationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    # For user-friendly display
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policyType": self.policy_type.value,
            "policyName": self.policy_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }


@dataclass
class PolicyResult:
    """Result of evaluating all policies for an action."""
    approved: bool
    violations: List[PolicyViolation] = field(default_factory=list)
    warnings: List[PolicyViolation] = field(default_factory=list)

    # Risk assessment
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (high risk)
    risk_level: RiskLevel = RiskLevel.LOW

    # Requires human approval even if technically allowed
    requires_approval: bool = False
    approval_reason: Optional[str] = None

    # Metadata
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    evaluation_time_ms: float = 0.0

    @property
    def blocking_violations(self) -> List[PolicyViolation]:
        """Get only blocking violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.BLOCK]

    @property
    def error_message(self) -> str:
        """Get a combined error message from blocking violations."""
        if not self.blocking_violations:
            return ""
        return "; ".join(v.message for v in self.blocking_violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "riskScore": self.risk_score,
            "riskLevel": self.risk_level.value,
            "requiresApproval": self.requires_approval,
            "approvalReason": self.approval_reason,
            "evaluatedAt": self.evaluated_at.isoformat(),
            "evaluationTimeMs": self.evaluation_time_ms,
        }


@dataclass
class ActionContext:
    """
    Context for an action being evaluated by the policy engine.

    This is the input to the policy engine - contains all information
    needed to evaluate whether an action should be allowed.
    """
    # Session info
    session_id: str
    wallet_address: str

    # Action details
    action_type: str  # "swap", "bridge", "transfer", etc.
    chain_id: int

    # Value info
    value_usd: Decimal

    # Contract/token info
    contract_address: Optional[str] = None
    token_in: Optional[str] = None
    token_out: Optional[str] = None

    # For swaps
    slippage_percent: Optional[float] = None

    # For gas estimation
    estimated_gas_usd: Optional[Decimal] = None

    # Portfolio context (for concentration checks)
    portfolio_value_usd: Optional[Decimal] = None
    current_position_percent: Optional[float] = None  # Current % in token_out

    # User context
    user_id: Optional[str] = None
    daily_volume_usd: Optional[Decimal] = None  # Today's trading volume
    daily_loss_usd: Optional[Decimal] = None    # Today's realized losses

    # Additional metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "walletAddress": self.wallet_address,
            "actionType": self.action_type,
            "chainId": self.chain_id,
            "valueUsd": str(self.value_usd),
            "contractAddress": self.contract_address,
            "tokenIn": self.token_in,
            "tokenOut": self.token_out,
            "slippagePercent": self.slippage_percent,
            "estimatedGasUsd": str(self.estimated_gas_usd) if self.estimated_gas_usd else None,
            "portfolioValueUsd": str(self.portfolio_value_usd) if self.portfolio_value_usd else None,
            "description": self.description,
        }


# =============================================================================
# Risk Policy Configuration
# =============================================================================

@dataclass
class RiskPolicyConfig:
    """
    User-configurable risk policy settings.

    These can be set per-user or per-session to control risk exposure.
    """
    # Position limits
    max_position_percent: float = 25.0          # Max % of portfolio in single asset
    max_position_value_usd: Decimal = Decimal("10000")  # Max USD in single position

    # Daily limits
    max_daily_volume_usd: Decimal = Decimal("50000")    # Max daily trading volume
    max_daily_loss_usd: Decimal = Decimal("1000")       # Max daily realized loss

    # Transaction limits
    max_single_tx_usd: Decimal = Decimal("5000")        # Max single transaction
    require_approval_above_usd: Decimal = Decimal("2000")  # Require approval above this

    # Slippage tolerance
    max_slippage_percent: float = 3.0           # Max allowed slippage
    warn_slippage_percent: float = 1.5          # Warn above this slippage

    # Gas limits
    max_gas_percent: float = 5.0                # Max gas as % of tx value
    warn_gas_percent: float = 2.0               # Warn above this gas %

    # Liquidity requirements
    min_liquidity_usd: Decimal = Decimal("100000")  # Min pool liquidity

    # Enable/disable specific checks
    enabled: bool = True
    check_position_limits: bool = True
    check_daily_limits: bool = True
    check_slippage: bool = True
    check_gas: bool = True
    check_liquidity: bool = False  # Requires external data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maxPositionPercent": self.max_position_percent,
            "maxPositionValueUsd": str(self.max_position_value_usd),
            "maxDailyVolumeUsd": str(self.max_daily_volume_usd),
            "maxDailyLossUsd": str(self.max_daily_loss_usd),
            "maxSingleTxUsd": str(self.max_single_tx_usd),
            "requireApprovalAboveUsd": str(self.require_approval_above_usd),
            "maxSlippagePercent": self.max_slippage_percent,
            "warnSlippagePercent": self.warn_slippage_percent,
            "maxGasPercent": self.max_gas_percent,
            "warnGasPercent": self.warn_gas_percent,
            "minLiquidityUsd": str(self.min_liquidity_usd),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskPolicyConfig":
        return cls(
            max_position_percent=data.get("maxPositionPercent", 25.0),
            max_position_value_usd=Decimal(data.get("maxPositionValueUsd", "10000")),
            max_daily_volume_usd=Decimal(data.get("maxDailyVolumeUsd", "50000")),
            max_daily_loss_usd=Decimal(data.get("maxDailyLossUsd", "1000")),
            max_single_tx_usd=Decimal(data.get("maxSingleTxUsd", "5000")),
            require_approval_above_usd=Decimal(data.get("requireApprovalAboveUsd", "2000")),
            max_slippage_percent=data.get("maxSlippagePercent", 3.0),
            warn_slippage_percent=data.get("warnSlippagePercent", 1.5),
            max_gas_percent=data.get("maxGasPercent", 5.0),
            warn_gas_percent=data.get("warnGasPercent", 2.0),
            min_liquidity_usd=Decimal(data.get("minLiquidityUsd", "100000")),
            enabled=data.get("enabled", True),
            check_position_limits=data.get("checkPositionLimits", True),
            check_daily_limits=data.get("checkDailyLimits", True),
            check_slippage=data.get("checkSlippage", True),
            check_gas=data.get("checkGas", True),
            check_liquidity=data.get("checkLiquidity", False),
        )


# =============================================================================
# Fee Policy Configuration
# =============================================================================

@dataclass
class FeePolicyConfig:
    """
    Fee policy settings for gas abstraction and paymaster routing.

    Stored in Convex feeConfigs table and evaluated before autonomous execution.
    """
    chain_id: Any
    stablecoin_symbol: str = "USDC"
    stablecoin_address: Optional[str] = None
    stablecoin_decimals: int = 6
    allow_native_fallback: bool = False
    native_symbol: str = "ETH"
    native_decimals: int = 18
    fee_asset_order: List[str] = field(default_factory=lambda: ["stablecoin"])
    reimbursement_mode: str = "none"  # per_tx | batch | none
    is_enabled: bool = False
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    missing: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeePolicyConfig":
        chain_id = data.get("chainId")
        stablecoin_address = data.get("stablecoinAddress")
        if isinstance(chain_id, int) and stablecoin_address:
            stablecoin_address = stablecoin_address.lower()
        return cls(
            chain_id=chain_id,
            stablecoin_symbol=data.get("stablecoinSymbol", "USDC"),
            stablecoin_address=stablecoin_address,
            stablecoin_decimals=data.get("stablecoinDecimals", 6),
            allow_native_fallback=data.get("allowNativeFallback", False),
            native_symbol=data.get("nativeSymbol", "ETH"),
            native_decimals=data.get("nativeDecimals", 18),
            fee_asset_order=data.get("feeAssetOrder", ["stablecoin"]),
            reimbursement_mode=data.get("reimbursementMode", "none"),
            is_enabled=data.get("isEnabled", False),
            updated_at=datetime.fromtimestamp(data.get("updatedAt") / 1000) if data.get("updatedAt") else None,
            updated_by=data.get("updatedBy"),
            missing=False,
        )

    @classmethod
    def missing_for_chain(cls, chain_id: Any) -> "FeePolicyConfig":
        return cls(chain_id=chain_id, is_enabled=False, missing=True)


# =============================================================================
# System Policy Configuration
# =============================================================================

@dataclass
class SystemPolicyConfig:
    """
    Platform-wide system policy settings.

    These are controlled by the platform, not users.
    """
    # Emergency controls
    emergency_stop: bool = False
    emergency_stop_reason: Optional[str] = None

    # Maintenance windows
    in_maintenance: bool = False
    maintenance_message: Optional[str] = None

    # Blocked addresses (known scams, exploits)
    blocked_contracts: List[str] = field(default_factory=list)
    blocked_tokens: List[str] = field(default_factory=list)

    # Allowed protocols (whitelist mode)
    protocol_whitelist_enabled: bool = False
    allowed_protocols: List[str] = field(default_factory=list)

    # Global limits
    max_single_tx_usd: Decimal = Decimal("100000")  # Platform max per tx

    # Chain restrictions
    blocked_chains: List[int] = field(default_factory=list)
    allowed_chains: List[int] = field(default_factory=list)  # If empty, all allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emergencyStop": self.emergency_stop,
            "emergencyStopReason": self.emergency_stop_reason,
            "inMaintenance": self.in_maintenance,
            "maintenanceMessage": self.maintenance_message,
            "blockedContracts": self.blocked_contracts,
            "blockedTokens": self.blocked_tokens,
            "protocolWhitelistEnabled": self.protocol_whitelist_enabled,
            "allowedProtocols": self.allowed_protocols,
            "maxSingleTxUsd": str(self.max_single_tx_usd),
            "blockedChains": self.blocked_chains,
            "allowedChains": self.allowed_chains,
        }
