"""LangGraph-powered orchestration for the Sherpa agent pipeline."""

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


def build_agent_process_graph(agent: "Agent"):
    """Compile the LangGraph pipeline that powers ``Agent.process_message``."""

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

    async def ensure_portfolio(state: AgentProcessState) -> AgentProcessState:
        await agent._ensure_portfolio_context(  # pylint: disable=protected-access
            state['conversation_id'],
            state['request'],
        )
        return {}

    async def prepare_context(state: AgentProcessState) -> AgentProcessState:
        persona = state['persona'] or 'friendly'
        context_messages = await agent._prepare_context(  # pylint: disable=protected-access
            state['request'],
            state['conversation_id'],
            persona,
        )
        return {'context_messages': context_messages}

    async def execute_tools(state: AgentProcessState) -> AgentProcessState:
        tool_data = await agent._execute_tools(  # pylint: disable=protected-access
            state['request'],
            state['conversation_id'],
        )
        return {'tool_data': tool_data}

    async def handle_bridge(state: AgentProcessState) -> AgentProcessState:
        tool_data = dict(state.get('tool_data', {}))
        wallet_address = tool_data.get('_address')
        if wallet_address is None:
            wallet_address = agent._extract_wallet_address(state['request'])  # pylint: disable=protected-access
            if wallet_address:
                tool_data['_address'] = wallet_address
        bridge_quote = await agent.bridge_manager.maybe_handle(
            state['request'],
            state['conversation_id'],
            wallet_address=wallet_address,
            default_chain=getattr(state['request'], 'chain', None),
        )
        if bridge_quote:
            tool_data['bridge_quote'] = bridge_quote
        return {'tool_data': tool_data}

    async def augment_tvl(state: AgentProcessState) -> AgentProcessState:
        tool_data = dict(state.get('tool_data', {}))
        try:
            if agent._needs_tvl_data(state['request']):  # pylint: disable=protected-access
                params = agent._extract_tvl_params(state['request'])  # pylint: disable=protected-access
                ts, tvl = await get_tvl_series(protocol=params['protocol'], window=params['window'])
                stats = agent._compute_tvl_stats(ts, tvl)  # pylint: disable=protected-access
                tool_data['defillama_tvl'] = {
                    'protocol': params['protocol'],
                    'window': params['window'],
                    'timestamps': ts,
                    'tvl': tvl,
                    'stats': stats,
                }
                if agent.context_manager:
                    try:
                        await agent.context_manager.set_active_focus(  # type: ignore[attr-defined]
                            state['conversation_id'],
                            {
                                'entity': params['protocol'],
                                'protocol': params['protocol'],
                                'metric': 'tvl',
                                'window': params['window'],
                                'chain': state['request'].chain,
                                'stats': stats,
                                'source': 'defillama',
                            },
                        )
                    except Exception as exc:  # pragma: no cover - logging only
                        agent.logger.warning('Failed setting episodic focus: %s', exc)
        except Exception as exc:  # pragma: no cover - logging only
            agent.logger.error('DefiLlama TVL fetch error: %s', exc)
        return {'tool_data': tool_data}

    async def generate_llm(state: AgentProcessState) -> AgentProcessState:
        persona = state['persona'] or 'friendly'
        llm_response = await agent._generate_llm_response(  # pylint: disable=protected-access
            state.get('context_messages', []),
            state.get('tool_data', {}),
            persona,
            state['conversation_id'],
        )
        return {'llm_response': llm_response}

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

    # Register graph nodes
    graph.add_node('handle_style', handle_style)
    graph.add_node('determine_persona', determine_persona)
    graph.add_node('ensure_portfolio', ensure_portfolio)
    graph.add_node('prepare_context', prepare_context)
    graph.add_node('execute_tools', execute_tools)
    graph.add_node('augment_tvl', augment_tvl)
    graph.add_node('handle_bridge', handle_bridge)
    graph.add_node('generate_llm', generate_llm)
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

    graph.add_edge('determine_persona', 'ensure_portfolio')
    graph.add_edge('ensure_portfolio', 'prepare_context')
    graph.add_edge('prepare_context', 'execute_tools')
    graph.add_edge('execute_tools', 'handle_bridge')
    graph.add_edge('handle_bridge', 'augment_tvl')
    graph.add_edge('augment_tvl', 'generate_llm')
    graph.add_edge('generate_llm', 'format_response')
    graph.add_edge('format_response', 'update_conversation')
    graph.add_edge('update_conversation', END)

    return graph.compile()


# Local imports placed at bottom to avoid circular dependencies
from ...tools.defillama import get_tvl_series  # noqa: E402
