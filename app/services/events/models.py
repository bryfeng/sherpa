"""
Event Monitoring Models

Data structures for event monitoring, webhooks, and activity tracking.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import uuid


class EventType(str, Enum):
    """Types of blockchain events we monitor."""

    # Wallet events
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    NATIVE_TRANSFER_IN = "native_transfer_in"
    NATIVE_TRANSFER_OUT = "native_transfer_out"

    # DEX events
    SWAP = "swap"
    LP_ADD = "lp_add"
    LP_REMOVE = "lp_remove"

    # DeFi events
    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"
    BORROW = "borrow"
    REPAY = "repay"
    LIQUIDATION = "liquidation"
    STAKE = "stake"
    UNSTAKE = "unstake"
    CLAIM_REWARDS = "claim_rewards"

    # Bridge events
    BRIDGE_INITIATE = "bridge_initiate"
    BRIDGE_COMPLETE = "bridge_complete"

    # NFT events
    NFT_MINT = "nft_mint"
    NFT_TRANSFER = "nft_transfer"
    NFT_SALE = "nft_sale"

    # Approval events
    APPROVAL = "approval"
    APPROVAL_FOR_ALL = "approval_for_all"

    # Contract interactions
    CONTRACT_CALL = "contract_call"

    # Polymarket events (Polygon)
    PM_ORDER_PLACED = "pm_order_placed"
    PM_ORDER_FILLED = "pm_order_filled"
    PM_POSITION_OPENED = "pm_position_opened"
    PM_POSITION_CLOSED = "pm_position_closed"

    # Unknown/other
    UNKNOWN = "unknown"


class ChainType(str, Enum):
    """Supported blockchain types."""

    # EVM chains
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"
    AVALANCHE = "avalanche"
    BSC = "bsc"

    # Non-EVM
    SOLANA = "solana"


# Chain ID mapping
CHAIN_ID_MAP: Dict[int, ChainType] = {
    1: ChainType.ETHEREUM,
    137: ChainType.POLYGON,
    42161: ChainType.ARBITRUM,
    10: ChainType.OPTIMISM,
    8453: ChainType.BASE,
    43114: ChainType.AVALANCHE,
    56: ChainType.BSC,
}

CHAIN_NAME_MAP: Dict[str, ChainType] = {
    "eth-mainnet": ChainType.ETHEREUM,
    "polygon-mainnet": ChainType.POLYGON,
    "arb-mainnet": ChainType.ARBITRUM,
    "opt-mainnet": ChainType.OPTIMISM,
    "base-mainnet": ChainType.BASE,
    "avax-mainnet": ChainType.AVALANCHE,
    "bnb-mainnet": ChainType.BSC,
    "solana-mainnet": ChainType.SOLANA,
}


class TokenTransfer(BaseModel):
    """Token transfer within a transaction."""

    token_address: str
    token_symbol: Optional[str] = None
    token_name: Optional[str] = None
    token_decimals: Optional[int] = None
    from_address: str
    to_address: str
    amount_raw: str  # Raw amount as string to preserve precision
    amount: Optional[Decimal] = None  # Parsed amount with decimals
    value_usd: Optional[Decimal] = None

    @property
    def is_native(self) -> bool:
        """Check if this is a native token transfer (ETH, SOL, etc.)."""
        return self.token_address.lower() in (
            "0x0000000000000000000000000000000000000000",
            "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "native",
            "so11111111111111111111111111111111111111112",  # Wrapped SOL
        )


class SwapDetails(BaseModel):
    """Details of a swap transaction."""

    dex_name: Optional[str] = None
    dex_router: Optional[str] = None
    token_in: TokenTransfer
    token_out: TokenTransfer
    slippage_bps: Optional[int] = None


class TransactionAction(BaseModel):
    """Parsed action from a transaction."""

    type: EventType
    protocol: Optional[str] = None  # e.g., "uniswap", "aave", "lido"
    data: Dict[str, Any] = Field(default_factory=dict)

    # For swaps
    swap: Optional[SwapDetails] = None

    # For transfers
    transfer: Optional[TokenTransfer] = None


class ParsedTransaction(BaseModel):
    """Fully parsed blockchain transaction."""

    # Identity
    tx_hash: str
    chain: ChainType
    block_number: int
    block_timestamp: datetime

    # Addresses
    from_address: str
    to_address: Optional[str] = None
    contract_address: Optional[str] = None

    # Value
    value_wei: str = "0"
    value_native: Optional[Decimal] = None
    gas_used: Optional[int] = None
    gas_price_wei: Optional[str] = None
    gas_cost_usd: Optional[Decimal] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Parsed data
    method_name: Optional[str] = None
    method_signature: Optional[str] = None
    actions: List[TransactionAction] = Field(default_factory=list)
    token_transfers: List[TokenTransfer] = Field(default_factory=list)

    # Raw data
    raw_input: Optional[str] = None
    raw_logs: Optional[List[Dict[str, Any]]] = None

    @property
    def primary_action(self) -> Optional[TransactionAction]:
        """Get the primary action of this transaction."""
        if self.actions:
            return self.actions[0]
        return None

    @property
    def event_type(self) -> EventType:
        """Get the primary event type."""
        if self.actions:
            return self.actions[0].type
        return EventType.UNKNOWN


class WalletActivity(BaseModel):
    """Activity record for a watched wallet."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wallet_address: str
    chain: ChainType
    event_type: EventType

    # Transaction info
    tx_hash: str
    block_number: int
    timestamp: datetime

    # Direction relative to watched wallet
    direction: str  # "in", "out", "internal"

    # Value
    value_usd: Optional[Decimal] = None

    # Parsed details
    parsed_tx: Optional[ParsedTransaction] = None

    # Counterparty
    counterparty_address: Optional[str] = None
    counterparty_label: Optional[str] = None  # If known (e.g., "Uniswap V3 Router")

    # For copy trading
    is_copyable: bool = False
    copy_relevance_score: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class SubscriptionStatus(str, Enum):
    """Status of a webhook subscription."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    EXPIRED = "expired"


class Subscription(BaseModel):
    """Subscription to watch an address or contract."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None

    # What to watch
    address: str
    chain: ChainType
    event_types: List[EventType] = Field(default_factory=list)

    # Webhook config
    webhook_id: Optional[str] = None  # External webhook ID (Alchemy/Helius)
    status: SubscriptionStatus = SubscriptionStatus.PENDING

    # Metadata
    label: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status == SubscriptionStatus.ACTIVE


# Callback types
EventCallback = Callable[[WalletActivity], Awaitable[None]]


class WebhookPayload(BaseModel):
    """Normalized webhook payload from any provider."""

    provider: str  # "alchemy" or "helius"
    chain: ChainType
    webhook_id: Optional[str] = None
    webhook_type: Optional[str] = None

    # Event data
    events: List[Dict[str, Any]] = Field(default_factory=list)

    # Raw payload for debugging
    raw: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    received_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookConfig(BaseModel):
    """Configuration for creating a webhook."""

    chain: ChainType
    addresses: List[str]
    webhook_url: str
    event_types: Optional[List[str]] = None  # Provider-specific event types

    # Alchemy-specific
    alchemy_network: Optional[str] = None

    # Helius-specific
    helius_webhook_type: Optional[str] = None


class WebhookRegistration(BaseModel):
    """Result of registering a webhook with a provider."""

    provider: str
    webhook_id: str
    chain: ChainType
    addresses: List[str]
    webhook_url: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
