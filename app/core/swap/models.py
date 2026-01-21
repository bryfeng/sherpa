"""Typed models used by the swap subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..bridge.chain_registry import ChainId


@dataclass
class SwapState:
    """Per-conversation swap state used for follow-ups."""

    context: Dict[str, Any] = field(default_factory=dict)
    quote_params: Dict[str, Any] = field(default_factory=dict)
    last_prompt: Optional[str] = None
    status: Optional[str] = None
    panel: Optional[Dict[str, Any]] = None
    summary_reply: Optional[str] = None
    summary_tool: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    quote_id: Optional[str] = None
    # Solana-specific: base64 transaction for signing
    solana_tx_base64: Optional[str] = None
    # Solana-specific: block height for transaction validity
    last_valid_block_height: Optional[int] = None


@dataclass
class SwapResult:
    """Structured result returned to the main agent pipeline."""

    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    panel: Optional[Dict[str, Any]] = None
    summary_reply: Optional[str] = None
    summary_tool: Optional[str] = None
    tx: Optional[Dict[str, Any]] = None
    # Solana-specific: base64 encoded transaction
    solana_tx: Optional[str] = None
    # Solana-specific: block height for transaction validity
    last_valid_block_height: Optional[int] = None
    # Chain information
    chain_id: Optional[ChainId] = None
    is_solana: bool = False


@dataclass
class SolanaSwapQuote:
    """Quote data specific to Solana Jupiter swaps."""

    input_mint: str
    output_mint: str
    input_amount: int  # lamports/smallest units
    output_amount: int  # lamports/smallest units
    min_output_amount: int  # with slippage
    slippage_bps: int
    price_impact_pct: float
    route_info: List[Dict[str, Any]] = field(default_factory=list)
    # Raw quote response for building transaction
    quote_response: Optional[Dict[str, Any]] = None
    # Token metadata
    input_token: Optional[Dict[str, Any]] = None
    output_token: Optional[Dict[str, Any]] = None

