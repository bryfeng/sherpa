"""
Pydantic models for the Agent Harness API.

Provides structured types for tool discovery, headless execution,
and machine-readable action results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# StructuredActionResult
# ---------------------------------------------------------------------------

class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class StructuredActionResult(BaseModel):
    """Wraps any tool result into a machine-readable envelope."""

    action_type: str = Field(description="Tool name that produced this result")
    status: ActionStatus
    data: Optional[Any] = Field(default=None, description="Raw tool result dict")
    requires_human: bool = Field(
        default=False,
        description="Whether this action needs wallet signature or human approval",
    )
    next_actions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up tool names",
    )
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Tool Discovery
# ---------------------------------------------------------------------------

class ToolParameterSchema(BaseModel):
    """JSON-Schema-compatible parameter descriptor."""

    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


class ToolInfo(BaseModel):
    """Discoverable metadata for a single tool."""

    name: str
    description: str
    parameters: List[ToolParameterSchema] = Field(default_factory=list)
    requires_address: bool = False
    category: Optional[str] = None


class ToolListResponse(BaseModel):
    tools: List[ToolInfo]
    count: int


# ---------------------------------------------------------------------------
# Execute Request / Response
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    tool: str = Field(description="Tool name to execute")
    params: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = Field(
        default=False,
        description="If true, validate only — do not execute",
    )
    address: Optional[str] = Field(
        default=None,
        description="Wallet address (required for address-dependent tools)",
    )
    chain: Optional[str] = Field(
        default=None,
        description="Blockchain to target (e.g. 'ethereum', 'base', 'solana')",
    )


class ExecuteResponse(BaseModel):
    result: StructuredActionResult
    dry_run: bool = False
