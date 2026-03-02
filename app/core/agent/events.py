"""Typed event stream for the agent pipeline.

Events are emitted during the ReAct loop to provide real-time visibility
into tool execution and LLM interactions.
"""

import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Base class for all agent pipeline events."""
    type: str
    timestamp: float = Field(default_factory=time.time)


class ToolStarted(AgentEvent):
    """Emitted when a tool begins execution."""
    type: str = "tool_started"
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0


class ToolCompleted(AgentEvent):
    """Emitted when a tool finishes execution."""
    type: str = "tool_completed"
    tool_name: str
    success: bool
    latency_ms: float = 0.0
    iteration: int = 0


class LLMStarted(AgentEvent):
    """Emitted when an LLM call begins."""
    type: str = "llm_started"
    iteration: int = 0
    tool_count: int = 0


class LLMCompleted(AgentEvent):
    """Emitted when an LLM call finishes."""
    type: str = "llm_completed"
    iteration: int = 0
    tool_calls_requested: int = 0
    tokens_used: Optional[int] = None


class LoopFinished(AgentEvent):
    """Emitted when the ReAct loop completes."""
    type: str = "loop_finished"
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0


class SteeringMessage(AgentEvent):
    """Injected by callers to steer the agent mid-loop."""
    type: str = "steering"
    content: str = ""
