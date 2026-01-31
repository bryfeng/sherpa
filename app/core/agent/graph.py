"""LangGraph-powered orchestration for the Sherpa agent pipeline.

This module uses LangGraph to orchestrate the agent processing flow.
The pipeline now uses LLM-driven tool selection via the ReAct loop
instead of keyword-based tool detection.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:  # pragma: no cover
    from .base import Agent, AgentResponse
    from ...providers.llm.base import LLMMessage, LLMResponse
    from ...types.requests import ChatRequest
else:  # pragma: no cover - runtime fallbacks for type hints
    Agent = Any  # type: ignore
    AgentResponse = Any  # type: ignore
    LLMMessage = Any  # type: ignore
    LLMResponse = Any  # type: ignore
    ChatRequest = Any  # type: ignore


class AgentProcessState(TypedDict, total=False):
    """Mutable state object passed between LangGraph nodes."""

    request: "ChatRequest"
    conversation_id: str
    persona_name: Optional[str]
    start_time: datetime
    style_response: Optional["AgentResponse"]
    persona: Optional[str]
    context_messages: List["LLMMessage"]
    tool_data: Dict[str, Any]
    llm_response: Optional["LLMResponse"]
    final_response: Optional["AgentResponse"]
    use_legacy_tools: bool  # Flag to use legacy keyword-based tools


def build_agent_process_graph(agent: "Agent"):
    """Compile the LangGraph pipeline that powers ``Agent.process_message``.

    New Flow (LLM-driven tool selection):
        handle_style → determine_persona → prepare_context → run_react_loop
        → handle_bridge → handle_swap → format_response → update_conversation

    The ReAct loop handles all tool calling through the LLM's native tool calling
    capabilities, replacing keyword-based detection.
    """

    graph: StateGraph[AgentProcessState] = StateGraph(AgentProcessState)

    async def handle_style(state: AgentProcessState) -> AgentProcessState:
        style_response = await agent._handle_style_processing(  # pylint: disable=protected-access
            state['request'].messages,
            state['conversation_id'],
        )
        if style_response:
            return {'final_response': style_response, 'style_response': style_response}
        return {}

    async def determine_persona(state: AgentProcessState) -> AgentProcessState:
        persona = await agent._determine_persona(  # pylint: disable=protected-access
            state['request'].messages,
            state['conversation_id'],
            state.get('persona_name'),
        )
        return {'persona': persona}

    async def prepare_context(state: AgentProcessState) -> AgentProcessState:
        persona = state['persona'] or 'friendly'
        context_messages = await agent._prepare_context(  # pylint: disable=protected-access
            state['request'],
            state['conversation_id'],
            persona,
        )
        return {'context_messages': context_messages}

    async def run_react_loop(state: AgentProcessState) -> AgentProcessState:
        """Run the LLM-driven ReAct loop for tool selection and execution."""
        llm_response, tool_data = await agent._run_react_loop(  # pylint: disable=protected-access
            state.get('context_messages', []),
            state['request'],
            state['conversation_id'],
        )
        return {'llm_response': llm_response, 'tool_data': tool_data}

    async def handle_bridge(state: AgentProcessState) -> AgentProcessState:
        """DISABLED: Bridge quotes now handled by LLM via get_bridge_quote tool.

        The LLM's semantic understanding is better than keyword matching.
        Keeping this node as pass-through for graph compatibility.
        """
        return state

    async def handle_swap(state: AgentProcessState) -> AgentProcessState:
        """DISABLED: Swap quotes now handled by LLM via get_swap_quote tool.

        The LLM's semantic understanding is better than keyword matching.
        Keeping this node as pass-through for graph compatibility.
        """
        return state

    async def format_response(state: AgentProcessState) -> AgentProcessState:
        persona = state['persona'] or 'friendly'
        final_response = await agent._format_response(  # pylint: disable=protected-access
            state['llm_response'],
            state.get('tool_data', {}),
            persona,
            state['conversation_id'],
            state['start_time'],
        )
        return {'final_response': final_response}

    async def update_conversation(state: AgentProcessState) -> AgentProcessState:
        await agent._update_conversation_state(  # pylint: disable=protected-access
            state['conversation_id'],
            state['request'],
            state['final_response'],
        )
        return {}

    # Register graph nodes - simplified flow
    graph.add_node('handle_style', handle_style)
    graph.add_node('determine_persona', determine_persona)
    graph.add_node('prepare_context', prepare_context)
    graph.add_node('run_react_loop', run_react_loop)
    graph.add_node('handle_bridge', handle_bridge)
    graph.add_node('handle_swap', handle_swap)
    graph.add_node('format_response', format_response)
    graph.add_node('update_conversation', update_conversation)

    graph.set_entry_point('handle_style')

    def _style_condition(state: AgentProcessState) -> str:
        return 'style_complete' if state.get('final_response') else 'continue'

    graph.add_conditional_edges(
        'handle_style',
        _style_condition,
        {
            'style_complete': END,
            'continue': 'determine_persona',
        },
    )

    # Simplified flow: persona → context → ReAct loop → bridge/swap handling → format
    graph.add_edge('determine_persona', 'prepare_context')
    graph.add_edge('prepare_context', 'run_react_loop')
    graph.add_edge('run_react_loop', 'handle_bridge')
    graph.add_edge('handle_bridge', 'handle_swap')
    graph.add_edge('handle_swap', 'format_response')
    graph.add_edge('format_response', 'update_conversation')
    graph.add_edge('update_conversation', END)

    return graph.compile()
