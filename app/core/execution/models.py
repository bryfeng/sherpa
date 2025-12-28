"""
Transaction execution models and types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class TransactionType(str, Enum):
    """Types of transactions."""
    SWAP = "swap"
    BRIDGE = "bridge"
    TRANSFER = "transfer"
    APPROVE = "approve"
    STAKE = "stake"
    UNSTAKE = "unstake"


class TransactionStatus(str, Enum):
    """Transaction lifecycle status."""
    PENDING = "pending"          # Created, not yet submitted
    SUBMITTED = "submitted"      # Broadcast to network
    CONFIRMING = "confirming"    # Waiting for confirmations
    CONFIRMED = "confirmed"      # Successfully confirmed
    FAILED = "failed"            # Submission failed
    REVERTED = "reverted"        # On-chain revert
    TIMEOUT = "timeout"          # Confirmation timeout
    CANCELLED = "cancelled"      # User cancelled


@dataclass
class GasEstimate:
    """Gas estimation for a transaction."""
    gas_limit: int
    gas_price_wei: int
    max_fee_per_gas: Optional[int] = None      # EIP-1559
    max_priority_fee_per_gas: Optional[int] = None  # EIP-1559
    estimated_cost_wei: int = 0
    estimated_cost_usd: float = 0.0

    def __post_init__(self):
        if self.estimated_cost_wei == 0:
            self.estimated_cost_wei = self.gas_limit * self.gas_price_wei


@dataclass
class PreparedTransaction:
    """A transaction ready to be signed and broadcast."""
    tx_id: str                                  # Internal tracking ID
    tx_type: TransactionType
    chain_id: int
    from_address: str
    to_address: str
    data: str                                   # Encoded calldata (hex)
    value: int = 0                              # Wei to send
    gas_estimate: Optional[GasEstimate] = None
    nonce: Optional[int] = None

    # Metadata
    description: str = ""
    quote_id: Optional[str] = None              # Relay requestId
    related_tx_ids: List[str] = field(default_factory=list)  # e.g., approval tx
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signing."""
        tx = {
            "from": self.from_address,
            "to": self.to_address,
            "data": self.data,
            "value": hex(self.value),
            "chainId": hex(self.chain_id),
        }
        if self.nonce is not None:
            tx["nonce"] = hex(self.nonce)
        if self.gas_estimate:
            tx["gas"] = hex(self.gas_estimate.gas_limit)
            if self.gas_estimate.max_fee_per_gas:
                tx["maxFeePerGas"] = hex(self.gas_estimate.max_fee_per_gas)
                tx["maxPriorityFeePerGas"] = hex(self.gas_estimate.max_priority_fee_per_gas or 0)
            else:
                tx["gasPrice"] = hex(self.gas_estimate.gas_price_wei)
        return tx


@dataclass
class TransactionResult:
    """Result of a transaction execution."""
    tx_id: str
    tx_hash: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PENDING
    chain_id: int = 1

    # Confirmation details
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    gas_used: Optional[int] = None
    effective_gas_price: Optional[int] = None

    # Timing
    submitted_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None

    # Error info
    error: Optional[str] = None
    revert_reason: Optional[str] = None

    # Output data (decoded events, etc.)
    output_data: Optional[Dict[str, Any]] = None

    @property
    def is_success(self) -> bool:
        return self.status == TransactionStatus.CONFIRMED

    @property
    def is_final(self) -> bool:
        return self.status in {
            TransactionStatus.CONFIRMED,
            TransactionStatus.FAILED,
            TransactionStatus.REVERTED,
            TransactionStatus.TIMEOUT,
            TransactionStatus.CANCELLED,
        }


@dataclass
class SwapQuote:
    """Parsed swap quote from Relay."""
    request_id: str
    chain_id: int
    wallet_address: str

    # Tokens
    token_in_address: str
    token_in_symbol: str
    token_in_decimals: int
    amount_in: int                              # In smallest units

    token_out_address: str
    token_out_symbol: str
    token_out_decimals: int
    amount_out_estimate: int                    # In smallest units

    # Pricing
    price_in_usd: float
    price_out_usd: float
    value_in_usd: float
    value_out_usd: float

    # Fees
    gas_fee_usd: float
    relay_fee_usd: float
    total_fee_usd: float
    slippage_bps: int

    # Execution data
    tx: Optional[Dict[str, Any]] = None         # Main transaction
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    signatures: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    expires_at: Optional[datetime] = None
    time_estimate_seconds: Optional[int] = None

    # Raw data
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class BridgeQuote:
    """Parsed bridge quote from Relay."""
    request_id: str
    origin_chain_id: int
    destination_chain_id: int
    wallet_address: str
    recipient_address: str

    # Tokens
    token_in_address: str
    token_in_symbol: str
    token_in_decimals: int
    amount_in: int

    token_out_address: str
    token_out_symbol: str
    token_out_decimals: int
    amount_out_estimate: int

    # Pricing
    price_in_usd: float
    price_out_usd: float
    value_in_usd: float
    value_out_usd: float

    # Fees
    gas_fee_usd: float
    relay_fee_usd: float
    cross_chain_fee_usd: float
    total_fee_usd: float
    slippage_bps: int

    # Execution data
    tx: Optional[Dict[str, Any]] = None
    deposit_address: Optional[str] = None
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    signatures: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    expires_at: Optional[datetime] = None
    time_estimate_seconds: Optional[int] = None

    # Raw data
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionContext:
    """Context for executing a transaction or set of transactions."""
    wallet_address: str
    chain_id: int

    # Execution settings
    max_gas_price_gwei: Optional[float] = None
    max_priority_fee_gwei: Optional[float] = None
    gas_multiplier: float = 1.1                 # Safety margin

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    confirmation_timeout_seconds: int = 300
    required_confirmations: int = 1

    # Nonce management
    use_pending_nonce: bool = True
    nonce_override: Optional[int] = None

    # Tracking
    execution_id: Optional[str] = None          # Links to strategy execution
    user_id: Optional[str] = None
    wallet_id: Optional[str] = None
