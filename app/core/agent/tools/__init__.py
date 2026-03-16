"""Tool package — backward-compatible re-exports."""

from .base import RegisteredTool, ToolSpec, tool_spec
from .groups import resolve_groups, get_tool_names_for_groups
from .registry import ToolContext, ToolRegistry, ToolExecutor

__all__ = [
    "RegisteredTool",
    "ToolContext",
    "ToolSpec",
    "tool_spec",
    "ToolRegistry",
    "ToolExecutor",
    "resolve_groups",
    "get_tool_names_for_groups",
]
