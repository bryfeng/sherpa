"""ToolRegistry and ToolExecutor — auto-discovers @tool_spec-decorated handlers."""

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from ....providers.llm.base import ToolCall, ToolResult
from .base import RegisteredTool, ToolSpec


@dataclass
class ToolContext:
    """Ambient context auto-injected into tool calls by ToolExecutor."""
    wallet_address: Optional[str] = None
    chain: Optional[str] = None
    user_id: Optional[str] = None

# Modules containing @tool_spec decorated handlers
_TOOL_MODULES = [
    "app.core.agent.tools.portfolio",
    "app.core.agent.tools.market_data",
    "app.core.agent.tools.news",
    "app.core.agent.tools.policy",
    "app.core.agent.tools.strategy",
    "app.core.agent.tools.trading",
    "app.core.agent.tools.copy_trading",
    "app.core.agent.tools.polymarket",
]


def _collect_tool_specs() -> List[ToolSpec]:
    """Import each tool module and collect all @tool_spec decorated functions."""
    specs: List[ToolSpec] = []
    for module_path in _TOOL_MODULES:
        module = importlib.import_module(module_path)
        for _name, obj in inspect.getmembers(module, inspect.isfunction):
            spec = getattr(obj, "_tool_spec", None)
            if isinstance(spec, ToolSpec):
                specs.append(spec)
    return specs


class ToolRegistry:
    """Registry of available tools that the LLM can call.

    Each tool has a definition (name, description, parameters) and a handler function.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._tools: Dict[str, RegisteredTool] = {}
        self.logger = logger or logging.getLogger(__name__)
        self._register_default_tools()

    def register(
        self,
        name: str,
        definition: Any,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        requires_address: bool = False,
    ) -> None:
        """Register a tool with its definition and handler."""
        self._tools[name] = RegisteredTool(
            definition=definition,
            handler=handler,
            requires_address=requires_address,
        )

    def get_definitions(self, tool_names: Optional[Set[str]] = None) -> list:
        """Get tool definitions for passing to the LLM.

        If *tool_names* is provided, only definitions whose name is in the set
        are returned.  Pass ``None`` (default) to return all tools.
        """
        if tool_names is None:
            return [tool.definition for tool in self._tools.values()]
        return [
            tool.definition
            for name, tool in self._tools.items()
            if name in tool_names
        ]

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def _register_default_tools(self) -> None:
        """Auto-discover and register all @tool_spec decorated handlers."""
        specs = _collect_tool_specs()
        for spec in specs:
            self._tools[spec.definition.name] = RegisteredTool(
                definition=spec.definition,
                handler=spec.handler,
                requires_address=spec.requires_address,
            )
        self.logger.debug("Registered %d tools from %d modules", len(specs), len(_TOOL_MODULES))


class ToolExecutor:
    """Executes tool calls requested by the LLM.

    Supports parallel execution of independent tool calls.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        logger: Optional[logging.Logger] = None,
    ):
        self.registry = registry
        self.logger = logger or logging.getLogger(__name__)

    async def execute_single(
        self,
        tool_call: ToolCall,
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute a single tool call and return the result.

        If *context* is provided and the tool ``requires_address``, the
        executor auto-injects ``wallet_address`` and ``chain`` into the
        call arguments when they are accepted by the tool definition and
        not already supplied by the caller.
        """
        tool = self.registry.get_tool(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=f"Unknown tool: {tool_call.name}",
            )

        # Auto-inject context fields for tools that require a wallet
        if context and tool.requires_address:
            param_names = {p.name for p in tool.definition.parameters}
            if context.wallet_address and 'wallet_address' in param_names:
                tool_call.arguments.setdefault('wallet_address', context.wallet_address)
            if context.chain:
                if 'chain' in param_names:
                    tool_call.arguments.setdefault('chain', context.chain)
                elif 'chain_id' in param_names:
                    chain_map = {
                        "ethereum": 1, "polygon": 137, "base": 8453,
                        "arbitrum": 42161, "optimism": 10,
                    }
                    tool_call.arguments.setdefault(
                        'chain_id', chain_map.get(context.chain.lower(), 1)
                    )

        try:
            result = await tool.handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                result=result,
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Tool execution error for {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                result=None,
                error=str(e),
            )

    async def execute_parallel(
        self,
        tool_calls: List[ToolCall],
        context: Optional[ToolContext] = None,
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel."""
        if not tool_calls:
            return []

        tasks = [self.execute_single(tc, context=context) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_call_id=tool_calls[i].id,
                    result=None,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results
