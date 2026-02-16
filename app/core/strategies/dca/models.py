"""
DCA Strategy Models

Data models for DCA strategy configuration and execution.
Supports both EVM chains (int chain IDs) and Solana (str chain ID).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Type alias for chain IDs (int for EVM, "solana" for Solana)
ChainId = Union[int, str]


def is_solana_chain(chain_id: ChainId) -> bool:
    """Check if a chain ID represents Solana."""
    if isinstance(chain_id, str):
        return chain_id.lower() == "solana"
    return False


class DCAFrequency(str, Enum):
    """DCA execution frequency."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class DCAStatus(str, Enum):
    """DCA strategy status."""
    DRAFT = "draft"
    PENDING_SESSION = "pending_session"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ExecutionStatus(str, Enum):
    """Individual execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SkipReason(str, Enum):
    """Reason for skipping an execution."""
    GAS_TOO_HIGH = "gas_too_high"
    PRICE_ABOVE_LIMIT = "price_above_limit"
    PRICE_BELOW_LIMIT = "price_below_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    SESSION_EXPIRED = "session_expired"
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    MANUALLY_SKIPPED = "manually_skipped"


@dataclass
class TokenInfo:
    """Token information for DCA."""
    symbol: str
    address: str
    chain_id: ChainId  # int for EVM, "solana" for Solana
    decimals: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "address": self.address,
            "chainId": self.chain_id,
            "decimals": self.decimals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenInfo:
        return cls(
            symbol=data["symbol"],
            address=data["address"],
            chain_id=data["chainId"],
            decimals=data["decimals"],
        )

    @property
    def is_solana(self) -> bool:
        """Check if this token is on Solana."""
        return is_solana_chain(self.chain_id)


@dataclass
class DCAConfig:
    """DCA strategy configuration (user-editable parameters)."""
    # What to buy
    from_token: TokenInfo
    to_token: TokenInfo
    amount_per_execution_usd: Decimal

    # Schedule
    frequency: DCAFrequency
    execution_hour_utc: int = 9  # Default 9am UTC
    execution_day_of_week: Optional[int] = None  # 0=Sunday, for weekly
    execution_day_of_month: Optional[int] = None  # 1-31, for monthly
    cron_expression: Optional[str] = None  # For custom

    # Constraints
    max_slippage_bps: int = 100  # 1% default
    max_gas_usd: Decimal = Decimal("10")
    skip_if_gas_above_usd: Optional[Decimal] = None
    pause_if_price_above_usd: Optional[Decimal] = None
    pause_if_price_below_usd: Optional[Decimal] = None
    min_amount_out: Optional[Decimal] = None

    # Budget limits
    max_total_spend_usd: Optional[Decimal] = None
    max_executions: Optional[int] = None
    end_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fromToken": self.from_token.to_dict(),
            "toToken": self.to_token.to_dict(),
            "amountPerExecutionUsd": float(self.amount_per_execution_usd),
            "frequency": self.frequency.value,
            "executionHourUtc": self.execution_hour_utc,
            "executionDayOfWeek": self.execution_day_of_week,
            "executionDayOfMonth": self.execution_day_of_month,
            "cronExpression": self.cron_expression,
            "maxSlippageBps": self.max_slippage_bps,
            "maxGasUsd": float(self.max_gas_usd),
            "skipIfGasAboveUsd": float(self.skip_if_gas_above_usd) if self.skip_if_gas_above_usd else None,
            "pauseIfPriceAboveUsd": float(self.pause_if_price_above_usd) if self.pause_if_price_above_usd else None,
            "pauseIfPriceBelowUsd": float(self.pause_if_price_below_usd) if self.pause_if_price_below_usd else None,
            "minAmountOut": str(self.min_amount_out) if self.min_amount_out else None,
            "maxTotalSpendUsd": float(self.max_total_spend_usd) if self.max_total_spend_usd else None,
            "maxExecutions": self.max_executions,
            "endDate": int(self.end_date.timestamp() * 1000) if self.end_date else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DCAConfig:
        return cls(
            from_token=TokenInfo.from_dict(data["fromToken"]),
            to_token=TokenInfo.from_dict(data["toToken"]),
            amount_per_execution_usd=Decimal(str(data["amountPerExecutionUsd"])),
            frequency=DCAFrequency(data["frequency"]),
            execution_hour_utc=data.get("executionHourUtc", 9),
            execution_day_of_week=data.get("executionDayOfWeek"),
            execution_day_of_month=data.get("executionDayOfMonth"),
            cron_expression=data.get("cronExpression"),
            max_slippage_bps=data.get("maxSlippageBps", 100),
            max_gas_usd=Decimal(str(data.get("maxGasUsd", 10))),
            skip_if_gas_above_usd=Decimal(str(data["skipIfGasAboveUsd"])) if data.get("skipIfGasAboveUsd") else None,
            pause_if_price_above_usd=Decimal(str(data["pauseIfPriceAboveUsd"])) if data.get("pauseIfPriceAboveUsd") else None,
            pause_if_price_below_usd=Decimal(str(data["pauseIfPriceBelowUsd"])) if data.get("pauseIfPriceBelowUsd") else None,
            min_amount_out=Decimal(data["minAmountOut"]) if data.get("minAmountOut") else None,
            max_total_spend_usd=Decimal(str(data["maxTotalSpendUsd"])) if data.get("maxTotalSpendUsd") else None,
            max_executions=data.get("maxExecutions"),
            end_date=datetime.fromtimestamp(data["endDate"] / 1000) if data.get("endDate") else None,
        )


@dataclass
class DCAStats:
    """DCA strategy lifetime statistics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    skipped_executions: int = 0
    total_amount_spent_usd: Decimal = Decimal("0")
    total_tokens_acquired: Decimal = Decimal("0")
    average_price_usd: Optional[Decimal] = None
    last_execution_at: Optional[datetime] = None
    last_error: Optional[str] = None


def _resolve_token(value: Any, chain_id: int = 1) -> TokenInfo:
    """Resolve a token from either a dict or a symbol string.

    The AI agent chat flow stores tokens as plain symbol strings
    (e.g. "USDC"), while the DCA-specific flow stores full objects.
    """
    if isinstance(value, dict):
        # Guard A: dict without address — try resolving by symbol
        if "address" not in value and "symbol" in value:
            symbol = value["symbol"].upper()
            chain_id_from_dict = int(value.get("chainId", chain_id))
            _KNOWN_TOKENS_GUARD: Dict[int, Dict[str, tuple]] = {
                1: {
                    "ETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
                    "WETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
                    "USDC": ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
                    "USDT": ("0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
                    "DAI": ("0x6B175474E89094C44Da98b954EedeAC495271d0F", 18),
                    "WBTC": ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8),
                },
                8453: {
                    "ETH": ("0x4200000000000000000000000000000000000006", 18),
                    "WETH": ("0x4200000000000000000000000000000000000006", 18),
                    "USDC": ("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 6),
                },
            }
            chain_tokens = _KNOWN_TOKENS_GUARD.get(chain_id_from_dict, {})
            if symbol in chain_tokens:
                addr, decimals = chain_tokens[symbol]
                return TokenInfo(symbol=symbol, address=addr, chain_id=chain_id_from_dict, decimals=decimals)
        return TokenInfo.from_dict(value)

    symbol = str(value).upper()
    _KNOWN_TOKENS: Dict[int, Dict[str, tuple]] = {
        1: {  # Ethereum mainnet
            "ETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
            "WETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
            "USDC": ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
            "USDT": ("0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
            "DAI": ("0x6B175474E89094C44Da98b954EedeAC495271d0F", 18),
            "WBTC": ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8),
        },
        8453: {  # Base
            "ETH": ("0x4200000000000000000000000000000000000006", 18),
            "WETH": ("0x4200000000000000000000000000000000000006", 18),
            "USDC": ("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 6),
        },
    }
    chain_tokens = _KNOWN_TOKENS.get(chain_id, {})
    if symbol in chain_tokens:
        addr, decimals = chain_tokens[symbol]
        return TokenInfo(symbol=symbol, address=addr, chain_id=chain_id, decimals=decimals)

    # Fallback: use zero address (executor will need to resolve)
    return TokenInfo(symbol=symbol, address="0x" + "0" * 40, chain_id=chain_id, decimals=18)


@dataclass
class DCAStrategy:
    """Complete DCA strategy with config and state."""
    # Identity
    id: str
    user_id: str
    wallet_id: str
    wallet_address: str
    name: str
    description: Optional[str] = None

    # Configuration
    config: DCAConfig = field(default_factory=lambda: None)  # type: ignore

    # State
    status: DCAStatus = DCAStatus.DRAFT
    pause_reason: Optional[str] = None
    session_key_id: Optional[str] = None  # Legacy off-chain session key
    smart_session_id: Optional[str] = None  # Rhinestone on-chain Smart Session

    # Scheduling
    next_execution_at: Optional[datetime] = None
    last_execution_at: Optional[datetime] = None

    # Stats
    stats: DCAStats = field(default_factory=DCAStats)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None

    @classmethod
    def from_strategies_table(cls, data: Dict[str, Any]) -> "DCAStrategy":
        """Create from generic strategies table document.

        The strategies table stores DCA config in a `config` blob
        rather than as top-level fields like dcaStrategies does.
        Handles both camelCase and snake_case keys, and both
        full token objects and plain symbol strings.
        """
        cfg = data.get("config", {})

        # Determine chain ID (may be stored as float from Convex)
        chain_id = int(cfg.get("chainId", cfg.get("chain_id", 1)))

        # Resolve tokens — handle both dict objects and plain strings
        from_token_raw = cfg.get("fromToken") or cfg.get("from_token")
        to_token_raw = cfg.get("toToken") or cfg.get("to_token")

        config = DCAConfig(
            from_token=_resolve_token(from_token_raw, chain_id),
            to_token=_resolve_token(to_token_raw, chain_id),
            amount_per_execution_usd=Decimal(str(cfg.get("amountPerExecutionUsd", cfg.get("amount_usd", cfg.get("amount", 0))))),
            frequency=DCAFrequency(cfg.get("frequency", "daily")),
            execution_hour_utc=cfg.get("executionHourUtc", 9),
            execution_day_of_week=cfg.get("executionDayOfWeek"),
            execution_day_of_month=cfg.get("executionDayOfMonth"),
            cron_expression=cfg.get("cronExpression") or data.get("cronExpression"),
            max_slippage_bps=int(cfg.get("maxSlippageBps", 100)),
            max_gas_usd=Decimal(str(cfg.get("maxGasUsd", 10))),
            skip_if_gas_above_usd=Decimal(str(cfg["skipIfGasAboveUsd"])) if cfg.get("skipIfGasAboveUsd") else None,
            pause_if_price_above_usd=Decimal(str(cfg["pauseIfPriceAboveUsd"])) if cfg.get("pauseIfPriceAboveUsd") else None,
            pause_if_price_below_usd=Decimal(str(cfg["pauseIfPriceBelowUsd"])) if cfg.get("pauseIfPriceBelowUsd") else None,
            min_amount_out=Decimal(cfg["minAmountOut"]) if cfg.get("minAmountOut") else None,
            max_total_spend_usd=Decimal(str(cfg["maxTotalSpendUsd"])) if cfg.get("maxTotalSpendUsd") else None,
            max_executions=cfg.get("maxExecutions"),
            end_date=datetime.fromtimestamp(cfg["endDate"] / 1000) if cfg.get("endDate") else None,
        )

        stats = DCAStats(
            total_executions=data.get("totalExecutions", 0),
            successful_executions=data.get("successfulExecutions", 0),
            failed_executions=data.get("failedExecutions", 0),
            skipped_executions=0,
            total_amount_spent_usd=Decimal("0"),
            total_tokens_acquired=Decimal("0"),
        )

        return cls(
            id=data["_id"],
            user_id=data["userId"],
            wallet_id=data.get("walletId", ""),
            wallet_address=data["walletAddress"],
            name=data["name"],
            description=data.get("description"),
            config=config,
            status=DCAStatus(data["status"]),
            session_key_id=data.get("sessionKeyId"),
            smart_session_id=data.get("smartSessionId"),
            next_execution_at=datetime.fromtimestamp(data["nextExecutionAt"] / 1000) if data.get("nextExecutionAt") else None,
            last_execution_at=datetime.fromtimestamp(data["lastExecutedAt"] / 1000) if data.get("lastExecutedAt") else None,
            stats=stats,
            created_at=datetime.fromtimestamp(data["createdAt"] / 1000),
            updated_at=datetime.fromtimestamp(data["updatedAt"] / 1000),
        )

    @classmethod
    def from_convex(cls, data: Dict[str, Any]) -> DCAStrategy:
        """Create from Convex document."""
        config = DCAConfig(
            from_token=TokenInfo.from_dict(data["fromToken"]),
            to_token=TokenInfo.from_dict(data["toToken"]),
            amount_per_execution_usd=Decimal(str(data["amountPerExecutionUsd"])),
            frequency=DCAFrequency(data["frequency"]),
            execution_hour_utc=data.get("executionHourUtc", 9),
            execution_day_of_week=data.get("executionDayOfWeek"),
            execution_day_of_month=data.get("executionDayOfMonth"),
            cron_expression=data.get("cronExpression"),
            max_slippage_bps=data.get("maxSlippageBps", 100),
            max_gas_usd=Decimal(str(data.get("maxGasUsd", 10))),
            skip_if_gas_above_usd=Decimal(str(data["skipIfGasAboveUsd"])) if data.get("skipIfGasAboveUsd") else None,
            pause_if_price_above_usd=Decimal(str(data["pauseIfPriceAboveUsd"])) if data.get("pauseIfPriceAboveUsd") else None,
            pause_if_price_below_usd=Decimal(str(data["pauseIfPriceBelowUsd"])) if data.get("pauseIfPriceBelowUsd") else None,
            min_amount_out=Decimal(data["minAmountOut"]) if data.get("minAmountOut") else None,
            max_total_spend_usd=Decimal(str(data["maxTotalSpendUsd"])) if data.get("maxTotalSpendUsd") else None,
            max_executions=data.get("maxExecutions"),
            end_date=datetime.fromtimestamp(data["endDate"] / 1000) if data.get("endDate") else None,
        )

        stats = DCAStats(
            total_executions=data.get("totalExecutions", 0),
            successful_executions=data.get("successfulExecutions", 0),
            failed_executions=data.get("failedExecutions", 0),
            skipped_executions=data.get("skippedExecutions", 0),
            total_amount_spent_usd=Decimal(str(data.get("totalAmountSpentUsd", 0))),
            total_tokens_acquired=Decimal(data.get("totalTokensAcquired", "0")),
            average_price_usd=Decimal(str(data["averagePriceUsd"])) if data.get("averagePriceUsd") else None,
            last_execution_at=datetime.fromtimestamp(data["lastExecutionAt"] / 1000) if data.get("lastExecutionAt") else None,
            last_error=data.get("lastError"),
        )

        return cls(
            id=data["_id"],
            user_id=data["userId"],
            wallet_id=data["walletId"],
            wallet_address=data["walletAddress"],
            name=data["name"],
            description=data.get("description"),
            config=config,
            status=DCAStatus(data["status"]),
            pause_reason=data.get("pauseReason"),
            session_key_id=data.get("sessionKeyId"),
            smart_session_id=data.get("smartSessionId"),
            next_execution_at=datetime.fromtimestamp(data["nextExecutionAt"] / 1000) if data.get("nextExecutionAt") else None,
            last_execution_at=datetime.fromtimestamp(data["lastExecutionAt"] / 1000) if data.get("lastExecutionAt") else None,
            stats=stats,
            created_at=datetime.fromtimestamp(data["createdAt"] / 1000),
            updated_at=datetime.fromtimestamp(data["updatedAt"] / 1000),
            activated_at=datetime.fromtimestamp(data["activatedAt"] / 1000) if data.get("activatedAt") else None,
        )


@dataclass
class MarketConditions:
    """Market conditions at execution time.

    Supports both EVM (gas_gwei) and Solana (priority_fee_lamports).
    """
    token_price_usd: Decimal
    estimated_gas_usd: Decimal
    # EVM-specific
    gas_gwei: Optional[Decimal] = None
    # Solana-specific
    priority_fee_lamports: Optional[int] = None
    is_solana: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "tokenPriceUsd": float(self.token_price_usd),
            "estimatedGasUsd": float(self.estimated_gas_usd),
            "isSolana": self.is_solana,
        }
        if self.gas_gwei is not None:
            result["gasGwei"] = float(self.gas_gwei)
        if self.priority_fee_lamports is not None:
            result["priorityFeeLamports"] = self.priority_fee_lamports
        return result


@dataclass
class ExecutionQuote:
    """Swap quote for execution."""
    input_amount: Decimal
    expected_output_amount: Decimal
    minimum_output_amount: Decimal
    price_impact_bps: int
    route: Optional[str] = None
    raw_quote: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputAmount": str(self.input_amount),
            "expectedOutputAmount": str(self.expected_output_amount),
            "minimumOutputAmount": str(self.minimum_output_amount),
            "priceImpactBps": self.price_impact_bps,
            "route": self.route,
            "rawQuote": self.raw_quote,
        }


@dataclass
class DCAExecution:
    """Individual DCA execution record."""
    id: str
    strategy_id: str
    execution_number: int
    chain_id: ChainId  # int for EVM, "solana" for Solana

    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    skip_reason: Optional[SkipReason] = None

    # Market conditions
    market_conditions: Optional[MarketConditions] = None
    quote: Optional[ExecutionQuote] = None

    # Transaction results
    tx_hash: Optional[str] = None
    actual_input_amount: Optional[Decimal] = None
    actual_output_amount: Optional[Decimal] = None
    actual_price_usd: Optional[Decimal] = None
    gas_used: Optional[int] = None
    gas_price_gwei: Optional[Decimal] = None
    gas_usd: Optional[Decimal] = None

    # Error info
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Timing
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def from_convex(cls, data: Dict[str, Any]) -> DCAExecution:
        """Create from Convex document."""
        market_conditions = None
        if data.get("marketConditions"):
            mc = data["marketConditions"]
            market_conditions = MarketConditions(
                token_price_usd=Decimal(str(mc["tokenPriceUsd"])),
                gas_gwei=Decimal(str(mc["gasGwei"])),
                estimated_gas_usd=Decimal(str(mc["estimatedGasUsd"])),
            )

        quote = None
        if data.get("quote"):
            q = data["quote"]
            quote = ExecutionQuote(
                input_amount=Decimal(q["inputAmount"]),
                expected_output_amount=Decimal(q["expectedOutputAmount"]),
                minimum_output_amount=Decimal(q["minimumOutputAmount"]),
                price_impact_bps=q["priceImpactBps"],
                route=q.get("route"),
            )

        return cls(
            id=data["_id"],
            strategy_id=data["strategyId"],
            execution_number=data["executionNumber"],
            chain_id=data["chainId"],
            status=ExecutionStatus(data["status"]),
            skip_reason=SkipReason(data["skipReason"]) if data.get("skipReason") else None,
            market_conditions=market_conditions,
            quote=quote,
            tx_hash=data.get("txHash"),
            actual_input_amount=Decimal(data["actualInputAmount"]) if data.get("actualInputAmount") else None,
            actual_output_amount=Decimal(data["actualOutputAmount"]) if data.get("actualOutputAmount") else None,
            actual_price_usd=Decimal(str(data["actualPriceUsd"])) if data.get("actualPriceUsd") else None,
            gas_used=data.get("gasUsed"),
            gas_price_gwei=Decimal(str(data["gasPriceGwei"])) if data.get("gasPriceGwei") else None,
            gas_usd=Decimal(str(data["gasUsd"])) if data.get("gasUsd") else None,
            error_message=data.get("errorMessage"),
            error_code=data.get("errorCode"),
            scheduled_at=datetime.fromtimestamp(data["scheduledAt"] / 1000),
            started_at=datetime.fromtimestamp(data["startedAt"] / 1000) if data.get("startedAt") else None,
            completed_at=datetime.fromtimestamp(data["completedAt"] / 1000) if data.get("completedAt") else None,
        )


@dataclass
class SessionKeyRequirements:
    """Session key requirements for a DCA strategy."""
    permissions: List[str]  # ["swap"]
    value_per_tx_usd: Decimal
    total_value_usd: Decimal
    token_allowlist: List[str]
    chain_allowlist: List[ChainId]  # List of chain IDs (int for EVM, "solana" for Solana)
    duration_days: int = 30

    @classmethod
    def for_dca_strategy(cls, config: DCAConfig, executions_estimate: int = 52) -> SessionKeyRequirements:
        """Generate session key requirements for a DCA config.

        Args:
            config: DCA configuration
            executions_estimate: Estimated number of executions (default 52 for weekly over 1 year)
        """
        # Add 10% buffer to amount for slippage
        value_per_tx = config.amount_per_execution_usd * Decimal("1.1")

        # Total value based on max spend or estimated executions
        if config.max_total_spend_usd:
            total_value = config.max_total_spend_usd * Decimal("1.1")
        else:
            total_value = value_per_tx * executions_estimate

        return cls(
            permissions=["swap"],
            value_per_tx_usd=value_per_tx,
            total_value_usd=total_value,
            token_allowlist=[config.from_token.symbol, config.to_token.symbol],
            chain_allowlist=[config.from_token.chain_id],
            duration_days=30,  # Renew monthly
        )
