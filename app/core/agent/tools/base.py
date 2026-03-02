"""Base tool infrastructure: decorator, ToolSpec, RegisteredTool."""

from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional

from ....providers.llm.base import ToolDefinition, ToolParameter, ToolParameterType


@dataclass
class RegisteredTool:
    """A tool registered in the registry with its definition and handler."""
    definition: ToolDefinition
    handler: Callable[..., Coroutine[Any, Any, Any]]
    requires_address: bool = False


@dataclass
class ToolSpec:
    """Metadata attached to a handler by the @tool_spec decorator."""
    definition: ToolDefinition
    handler: Callable[..., Coroutine[Any, Any, Any]]
    requires_address: bool = False


def tool_spec(
    name: str,
    description: str,
    parameters: List[ToolParameter],
    requires_address: bool = False,
):
    """Decorator that wraps an async handler with its ToolDefinition metadata."""
    def decorator(fn: Callable) -> Callable:
        fn._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            definition=ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
            ),
            handler=fn,
            requires_address=requires_address,
        )
        return fn
    return decorator
