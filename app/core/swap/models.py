"""Typed models used by the swap subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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

