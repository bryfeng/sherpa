"""Typed models used by the bridge subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional


@dataclass
class BridgeState:
    """Per-conversation bridge state used for follow-ups."""

    context: Dict[str, Any] = field(default_factory=dict)
    quote_params: Dict[str, Any] = field(default_factory=dict)
    price: Optional[Dict[str, Any]] = None
    last_prompt: Optional[str] = None
    status: Optional[str] = None
    panel: Optional[Dict[str, Any]] = None
    summary_reply: Optional[str] = None
    summary_tool: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    quote_id: Optional[str] = None
    route_request_hash: Optional[str] = None


@dataclass
class BridgeResult:
    """Structured result returned to the main agent pipeline."""

    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    panel: Optional[Dict[str, Any]] = None
    summary_reply: Optional[str] = None
    summary_tool: Optional[str] = None
    tx: Optional[Dict[str, Any]] = None


@dataclass
class QuoteContext:
    conversation_id: str
    request_timestamp: datetime
    amount_eth: Optional[Decimal]
    amount_usd: Optional[Decimal]

