"""
Core Agent System

This module contains the main Agent class that orchestrates LLM interactions,
persona management, and tool integration for intelligent portfolio analysis.
"""

import copy
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from statistics import mean, pstdev
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import httpx
from pydantic import BaseModel, Field

from ...config import settings
from ...providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse,
    ToolDefinition, ToolCall, ToolResult,
)
from ...tools.portfolio import get_portfolio
from ...services.trending import get_trending_tokens
from ...services.token_chart import get_token_chart as fetch_token_chart
from ...services.activity_summary import get_history_snapshot
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from ..bridge import BridgeManager
from ..swap import SwapManager
from .styles import StyleManager, ResponseStyle
from .tools import ToolRegistry, ToolExecutor
from .graph import build_agent_process_graph


HISTORY_SYNC_DAY_CAP = 90
DEFAULT_HISTORY_LIMIT = settings.history_summary_default_limit


class AgentResponse(BaseModel):
    """Structured response from the Agent system"""
    
    reply: str = Field(description="Natural language response from the agent")
    panels: Dict[str, Any] = Field(default_factory=dict, description="Structured data panels")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Data source attributions")
    
    # Agent-specific metadata
    agent_metadata: Dict[str, Any] = Field(default_factory=dict, description="Agent processing metadata")
    persona_used: Optional[str] = Field(default=None, description="Persona that generated this response")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    tokens_used: Optional[int] = Field(default=None, description="Total tokens consumed")
    processing_time_ms: Optional[float] = Field(default=None, description="Response processing time")


@dataclass
class HistoryWindowContext:
    start: datetime
    end: datetime
    requested_start: datetime
    requested_end: datetime
    requested_days: int
    applied_days: int

    @property
    def clamped(self) -> bool:
        return self.applied_days < self.requested_days


class ToolRequest(BaseModel):
    """Structured request for tool execution"""
    
    tool_name: str
    parameters: Dict[str, Any]
    priority: int = Field(default=1, description="Tool execution priority")


class Agent:
    """
    Core Agent class that orchestrates LLM interactions with persona management,
    context handling, and tool integration for intelligent portfolio analysis.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        persona_manager: Optional['PersonaManager'] = None,
        context_manager: Optional['ContextManager'] = None,
        style_manager: Optional['StyleManager'] = None,
        logger: Optional[logging.Logger] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.context_manager = context_manager
        self.style_manager = style_manager or StyleManager()
        self.logger = logger or logging.getLogger(__name__)
        self.provider_id = provider_id or llm_provider.__class__.__name__.lower()
        self.model_id = model_id or getattr(llm_provider, "model", None)

        # Agent state
        self._active_conversations: Dict[str, Dict] = {}
        # Style state per conversation
        self._conversation_styles: Dict[str, ResponseStyle] = {}

        # Tool calling infrastructure
        self.tool_registry = ToolRegistry(logger=self.logger)
        self.tool_executor = ToolExecutor(self.tool_registry, logger=self.logger)
        self._max_tool_iterations = 5  # Maximum ReAct loop iterations

        # LangGraph pipeline orchestrating the processing flow
        self._process_graph = build_agent_process_graph(self)
        # Bridge + swap orchestration managers
        self.bridge_manager = BridgeManager(logger=self.logger)
        self.swap_manager = SwapManager(logger=self.logger)
        
    async def process_message(
        self,
        request: ChatRequest,
        conversation_id: Optional[str] = None,
        persona_name: Optional[str] = None
    ) -> AgentResponse:
        """
        Process a chat message through the full agent pipeline:
        1. Generate conversation ID if needed
        2. Determine persona to use
        3. Prepare context with conversation history and tool data
        4. Generate LLM response
        5. Format response for API compatibility
        """
        start_time = datetime.now()
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        try:
            graph_state = {
                'request': request,
                'conversation_id': conversation_id,
                'persona_name': persona_name,
                'start_time': start_time,
            }
            result_state = await self._process_graph.ainvoke(graph_state)
            agent_response = result_state.get('final_response')
            if agent_response is None:
                raise RuntimeError('Agent pipeline failed to produce a response')

            return agent_response
            
        except Exception as e:
            self.logger.error(f"Agent processing error: {str(e)}", exc_info=True)
            return await self._create_error_response(
                str(e), 
                conversation_id, 
                start_time
            )

    async def _determine_persona(
        self,
        messages: List[ChatMessage],
        conversation_id: str,
        requested_persona: Optional[str] = None
    ) -> str:
        """Determine which persona to use for this conversation"""
        
        # If persona explicitly requested, use it
        if requested_persona and self.persona_manager:
            return requested_persona
            
        # Check for persona switching commands in the latest message
        if messages and self.persona_manager:
            latest_message = messages[-1].content.lower()
            if latest_message.startswith('/persona '):
                requested = latest_message.replace('/persona ', '').strip()
                if self.persona_manager.has_persona(requested):
                    return requested
                    
        # Use context-aware persona detection if available
        if self.persona_manager and self.context_manager:
            detected = await self.persona_manager.detect_persona_from_context(
                messages[-1].content if messages else ""
            )
            if detected:
                return detected
                
        # Default to friendly crypto guide
        return "friendly"

    async def _prepare_context(
        self,
        request: ChatRequest,
        conversation_id: str,
        persona_name: str
    ) -> List[LLMMessage]:
        """Prepare the full context for LLM including system prompt and conversation history"""

        context_messages = []

        # Add system prompt from persona
        if self.persona_manager:
            persona = self.persona_manager.get_persona(persona_name)
            system_prompt = persona.get_system_prompt()
            context_messages.append(LLMMessage(role="system", content=system_prompt))
        else:
            # Default system prompt if no persona manager
            default_prompt = self._get_default_system_prompt()
            context_messages.append(LLMMessage(role="system", content=default_prompt))

        # Add wallet address context if available
        # This ensures the LLM knows a wallet is connected and won't ask for it
        wallet_address = self._extract_wallet_address(request)
        if wallet_address:
            context_messages.append(LLMMessage(
                role="system",
                content=f"User's connected wallet address: {wallet_address} (chain: {request.chain or 'ethereum'}). Use this address for any portfolio or wallet-related queries - do not ask the user for their wallet address."
            ))

        # Add conversation history if context manager available
        if self.context_manager:
            history = await self.context_manager.get_context(conversation_id)
            # Context manager should return formatted context string
            if history:
                context_messages.append(LLMMessage(role="system", content=f"Conversation context: {history}"))

        # Convert chat messages to LLM format
        for msg in request.messages:
            context_messages.append(LLMMessage(
                role=msg.role,
                content=msg.content
            ))

        return context_messages

    async def _execute_tools(self, request: ChatRequest, conversation_id: str) -> Dict[str, Any]:
        """Execute relevant tools based on the chat request"""

        tool_data: Dict[str, Any] = {}
        wallet_address = self._extract_wallet_address(request)
        tool_data['_address'] = wallet_address

        # Check if this looks like a portfolio request
        needs_portfolio = self._needs_portfolio_data(request)
        if needs_portfolio:
            try:
                if wallet_address:
                    # Fetch portfolio data
                    portfolio_result = await get_portfolio(wallet_address, request.chain)

                    if portfolio_result.data:
                        tool_data['portfolio'] = {
                            'data': portfolio_result.data.model_dump(mode='json'),
                            'sources': [s.model_dump(mode='json') for s in portfolio_result.sources],
                            'warnings': portfolio_result.warnings or []
                        }
                    else:
                        tool_data['portfolio_error'] = {
                            'error': 'Failed to fetch portfolio data',
                            'warnings': portfolio_result.warnings or []
                        }
                        tool_data['needs_portfolio'] = True
                        
            except Exception as e:
                self.logger.error(f"Tool execution error: {str(e)}")
                tool_data['portfolio_error'] = {
                    'error': str(e),
                    'warnings': []
                }
                tool_data['needs_portfolio'] = True

        # Check if trending token intelligence is relevant
        trending_query = self._extract_trending_query(request)
        needs_trending = self._needs_trending_data(request)
        if needs_trending or trending_query:
            try:
                fetch_limit = 25 if trending_query else 12
                raw_tokens = list(await get_trending_tokens(limit=fetch_limit))
                fetched_at = datetime.now(timezone.utc).isoformat()
                focus_token = self._match_trending_token(raw_tokens, trending_query) if trending_query else None

                panel_limit = 10
                tokens_for_panel = raw_tokens[:panel_limit]
                focus_identity = self._token_identity(focus_token) if focus_token else ''
                if focus_token and focus_identity:
                    in_panel = any(
                        self._token_identity(token) == focus_identity
                        for token in tokens_for_panel
                    )
                    if not in_panel:
                        tokens_for_panel.append(focus_token)

                trending_sources = self._collect_trending_sources(tokens_for_panel)
                tool_data['trending_tokens'] = {
                    'tokens': tokens_for_panel,
                    'fetched_at': fetched_at,
                    'query': trending_query,
                    'focus': focus_token,
                    'total_available': len(raw_tokens),
                    'sources': trending_sources,
                }

                if focus_token and self.context_manager:
                    try:
                        await self.context_manager.set_active_focus(  # type: ignore[attr-defined]
                            conversation_id,
                            {
                                'entity': focus_token.get('name') or focus_token.get('symbol'),
                                'symbol': focus_token.get('symbol'),
                                'metric': 'trending_token',
                                'window': '24h',
                                'chain': focus_token.get('platform') or 'multichain',
                                'stats': {
                                    'price_usd': focus_token.get('price_usd'),
                                    'change_24h': focus_token.get('change_24h'),
                                    'volume_24h': focus_token.get('volume_24h'),
                                    'market_cap': focus_token.get('market_cap'),
                                },
                                'source': 'coingecko',
                            },
                        )
                    except Exception as exc:  # pragma: no cover - logging only
                        self.logger.warning('Failed setting trending focus: %s', exc)

            except Exception as exc:
                self.logger.error('Trending token fetch error: %s', exc)
                tool_data['trending_error'] = str(exc)

        requested_limit = self._extract_history_limit(request)
        mentions_window = self._mentions_time_window(request)
        effective_limit = requested_limit if requested_limit is not None else (DEFAULT_HISTORY_LIMIT if not mentions_window else None)
        use_limit_mode = effective_limit is not None

        if self._needs_history_summary(request):
            if not request.address:
                tool_data['history_summary_error'] = 'Connect your wallet to ask about recent history.'
            elif not wallet_address or wallet_address.lower() != request.address.lower():
                tool_data['history_summary_error'] = 'Ask about the wallet that is currently connected before requesting history.'
            else:
                window_context: Optional[HistoryWindowContext] = None
                try:
                    if use_limit_mode:
                        snapshot, events = await get_history_snapshot(
                            address=wallet_address,
                            chain=request.chain,
                            limit=effective_limit,
                        )
                    else:
                        window_context = self._extract_history_window(request)
                        snapshot, events = await get_history_snapshot(
                            address=wallet_address,
                            chain=request.chain,
                            start=window_context.start,
                            end=window_context.end,
                        )
                        self._apply_history_window_metadata(snapshot, window_context)
                    tool_data['history_summary'] = {
                        'snapshot': snapshot,
                        'events': events,
                        'limit': effective_limit if use_limit_mode else None,
                    }
                except Exception as exc:  # pragma: no cover - logging only
                    self.logger.warning('History summary fetch error: %s', exc)
                    tool_data['history_summary_error'] = str(exc)

        chart_request = self._extract_token_chart_request(request)
        if chart_request:
            try:
                chart_payload = await fetch_token_chart(
                    coin_id=chart_request.get('coin_id'),
                    symbol=chart_request.get('symbol'),
                    contract_address=chart_request.get('contract_address'),
                    chain=chart_request.get('chain') or request.chain,
                    range_key=chart_request.get('range') or '7d',
                    vs_currency=chart_request.get('vs_currency') or 'usd',
                    include_candles=True,
                )
                tool_data['token_chart'] = chart_payload
            except Exception as exc:
                self.logger.error('Token chart fetch error: %s', exc)
                tool_data['token_chart_error'] = str(exc)

        return tool_data

    async def _run_react_loop(
        self,
        messages: List[LLMMessage],
        request: ChatRequest,
        conversation_id: str,
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        """
        Run the ReAct (Reasoning + Acting) loop for LLM-driven tool selection.

        The LLM decides which tools to call based on semantic understanding of the
        user's intent. This replaces the keyword-based tool detection approach.

        Returns:
            Tuple of (final_llm_response, tool_data_dict)
        """
        tool_data: Dict[str, Any] = {}

        # Extract wallet address for tools that need it
        wallet_address = self._extract_wallet_address(request)
        tool_data['_address'] = wallet_address

        # Check if provider supports tool calling
        if not self.llm_provider.supports_tools:
            self.logger.debug("LLM provider does not support tools, running without tool calling")
            response = await self.llm_provider.generate_response(
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
            )
            return response, tool_data

        # Get tool definitions
        tools = self.tool_registry.get_definitions()

        # Start the ReAct loop
        current_messages = list(messages)
        final_response: Optional[LLMResponse] = None

        for iteration in range(self._max_tool_iterations):
            self.logger.debug(f"ReAct loop iteration {iteration + 1}/{self._max_tool_iterations}")

            # Get LLM response with tool definitions
            response = await self.llm_provider.generate_response(
                messages=current_messages,
                tools=tools,
                max_tokens=4000,
                temperature=0.7,
            )

            # If no tool calls, we're done
            if not response.tool_calls:
                self.logger.debug("LLM finished without tool calls")
                final_response = response
                break

            self.logger.debug(f"LLM requested {len(response.tool_calls)} tool call(s)")

            # Inject wallet address into tools that require it
            for tc in response.tool_calls:
                tool = self.tool_registry.get_tool(tc.name)
                if tool and tool.requires_address and wallet_address:
                    # Get parameter names from tool definition
                    tool_param_names = {p.name for p in tool.definition.parameters}

                    if 'wallet_address' not in tc.arguments and 'wallet_address' in tool_param_names:
                        tc.arguments['wallet_address'] = wallet_address

                    # Only inject chain if tool accepts it
                    if 'chain' not in tc.arguments and request.chain:
                        if 'chain' in tool_param_names:
                            tc.arguments['chain'] = request.chain
                        elif 'chain_id' in tool_param_names:
                            # Map chain name to chain_id
                            chain_map = {
                                "ethereum": 1, "polygon": 137, "base": 8453,
                                "arbitrum": 42161, "optimism": 10,
                            }
                            tc.arguments['chain_id'] = chain_map.get(request.chain.lower(), 1)

            # Execute all tool calls in parallel
            results = await self.tool_executor.execute_parallel(response.tool_calls)

            # Store results in tool_data and build messages for next iteration
            for tc, result in zip(response.tool_calls, results):
                # Map tool results to expected format in tool_data
                result_data = result.result if result.result else {'error': result.error}
                tool_data[tc.name] = {
                    'call': tc.arguments,
                    'result': result_data,
                }

                # Convert results to format expected by _format_response
                self._map_tool_result_to_legacy_format(tc.name, result_data, tool_data)

            # Append assistant message with tool calls
            current_messages.append(LLMMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            # Append tool results
            for result in results:
                current_messages.append(LLMMessage(
                    role="tool_result",
                    tool_result=result,
                ))

            final_response = response

        if final_response is None:
            # This shouldn't happen, but just in case
            final_response = await self.llm_provider.generate_response(
                messages=current_messages,
                max_tokens=4000,
                temperature=0.7,
            )

        return final_response, tool_data

    def _map_tool_result_to_legacy_format(
        self,
        tool_name: str,
        result_data: Dict[str, Any],
        tool_data: Dict[str, Any],
    ) -> None:
        """
        Map tool results to the legacy format expected by _format_response.

        This maintains backward compatibility during the transition.
        """
        if not isinstance(result_data, dict):
            return

        if tool_name == "get_portfolio":
            if result_data.get('success'):
                tool_data['portfolio'] = {
                    'data': result_data.get('data', {}),
                    'sources': result_data.get('sources', []),
                    'warnings': result_data.get('warnings', []),
                }
            else:
                tool_data['portfolio_error'] = {
                    'error': result_data.get('error', 'Unknown error'),
                    'warnings': result_data.get('warnings', []),
                }
                tool_data['needs_portfolio'] = True

        elif tool_name == "get_token_chart":
            if result_data.get('success'):
                # Remove 'success' key and pass the rest
                chart_data = {k: v for k, v in result_data.items() if k != 'success'}
                tool_data['token_chart'] = chart_data
            else:
                tool_data['token_chart_error'] = result_data.get('error', 'Unknown error')

        elif tool_name == "get_trending_tokens":
            if result_data.get('success'):
                tool_data['trending_tokens'] = {
                    'tokens': result_data.get('tokens', []),
                    'fetched_at': result_data.get('fetched_at'),
                    'query': None,
                    'focus': result_data.get('focus'),
                    'total_available': result_data.get('total_available', 0),
                    'sources': [{'name': 'CoinGecko', 'url': 'https://www.coingecko.com'}],
                }
            else:
                tool_data['trending_error'] = result_data.get('error', 'Unknown error')

        elif tool_name == "get_wallet_history":
            if result_data.get('success'):
                tool_data['history_summary'] = {
                    'snapshot': result_data.get('snapshot', {}),
                    'events': result_data.get('events', []),
                    'limit': result_data.get('limit'),
                }
            else:
                tool_data['history_summary_error'] = result_data.get('error', 'Unknown error')

        elif tool_name == "get_tvl_data":
            if result_data.get('success'):
                tool_data['defillama_tvl'] = {
                    'protocol': result_data.get('protocol'),
                    'window': result_data.get('window'),
                    'timestamps': result_data.get('timestamps', []),
                    'tvl': result_data.get('tvl', []),
                    'stats': result_data.get('stats', {}),
                }
            else:
                tool_data['tvl_error'] = result_data.get('error', 'Unknown error')

        # Strategy tools
        elif tool_name == "list_strategies":
            if result_data.get('success'):
                tool_data['strategies'] = {
                    'success': True,
                    'strategies': result_data.get('strategies', []),
                    'count': result_data.get('count', 0),
                    'wallet_address': result_data.get('wallet_address'),
                }
            else:
                tool_data['strategies_error'] = result_data.get('error', 'Unknown error')

        elif tool_name == "get_strategy":
            if result_data.get('success'):
                tool_data['strategy_detail'] = {
                    'success': True,
                    'strategy': result_data.get('strategy', {}),
                }
            else:
                tool_data['strategy_detail_error'] = result_data.get('error', 'Unknown error')

        elif tool_name in ("create_strategy", "update_strategy", "pause_strategy", "resume_strategy", "stop_strategy"):
            # These mutations don't need panels, but we can track success
            if result_data.get('success'):
                tool_data['strategy_mutation'] = {
                    'success': True,
                    'action': tool_name,
                    'strategy_id': result_data.get('strategy_id'),
                    'message': result_data.get('message'),
                }
            else:
                tool_data['strategy_mutation_error'] = result_data.get('error', 'Unknown error')

        elif tool_name == "get_strategy_executions":
            if result_data.get('success'):
                tool_data['strategy_executions'] = {
                    'success': True,
                    'executions': result_data.get('executions', []),
                    'strategy': result_data.get('strategy', {}),
                }
            else:
                tool_data['strategy_executions_error'] = result_data.get('error', 'Unknown error')

    def _needs_tvl_data(self, request: ChatRequest) -> bool:
        """Determine if the request is asking for TVL/chart data (e.g., Uniswap TVL)."""
        if not request.messages:
            return False
        msg = request.messages[-1].content.lower()
        # Basic heuristic: user mentions TVL and Uniswap
        return ('tvl' in msg or 'total value locked' in msg) and ('uniswap' in msg)

    def _extract_tvl_params(self, request: ChatRequest) -> Dict[str, str]:
        """Extract protocol and range from the message; defaults to uniswap + 7d."""
        protocol = 'uniswap'
        window = '7d'
        msg = request.messages[-1].content.lower() if request.messages else ''
        if '30d' in msg or '30 d' in msg or 'month' in msg or '30 days' in msg:
            window = '30d'
        elif '7d' in msg or '7 d' in msg or 'week' in msg or '7 days' in msg:
            window = '7d'
        # Simple protocol extraction (extend later if needed)
        if 'uniswap' in msg:
            protocol = 'uniswap'
        return {'protocol': protocol, 'window': window}

    def _build_llm_messages(
        self,
        base_messages: List[LLMMessage],
        tool_data: Dict[str, Any],
        conversation_id: str,
    ) -> List[LLMMessage]:
        """Compose the final message list passed to the LLM."""

        messages: List[LLMMessage] = [
            LLMMessage(role=msg.role, content=msg.content)
            for msg in base_messages
        ]

        current_style = self._conversation_styles.get(
            conversation_id,
            self.style_manager.get_current_style(),
        )
        style_modifier = self.style_manager.get_style_modifier_prompt(current_style)
        if style_modifier:
            messages.append(
                LLMMessage(
                    role="system",
                    content=f"Response style guidance: {style_modifier}",
                )
            )

        if tool_data:
            tool_context = self._format_tool_data_for_llm(tool_data)
            messages.append(
                LLMMessage(
                    role="system",
                    content=f"Available data: {tool_context}",
                )
            )
            if 'defillama_tvl' in tool_data:
                messages.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "With the provided TVL time series, write a concise 7-day analysis with 2–4 bullets: "
                            "start vs end, absolute and percent change, min/max with dates, and overall trend. "
                            "Avoid disclaimers about missing charts; assume the user sees the chart panel."
                        ),
                    )
                )

        return messages

    async def _generate_llm_response(
        self,
        messages: List[LLMMessage],
        tool_data: Dict[str, Any],
        persona_name: str,  # noqa: ARG002 - kept for parity with streaming signature
        conversation_id: str,
    ) -> LLMResponse:
        """Generate response from LLM with tool data context and style modifiers."""

        llm_messages = self._build_llm_messages(messages, tool_data, conversation_id)

        response = await self.llm_provider.generate_response(
            messages=llm_messages,
            max_tokens=4000,
            temperature=0.7,
        )

        return response

    async def _stream_llm_response(
        self,
        messages: List[LLMMessage],
        tool_data: Dict[str, Any],
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from the LLM provider."""

        llm_messages = self._build_llm_messages(messages, tool_data, conversation_id)
        try:
            async for chunk in self.llm_provider.generate_streaming_response(  # type: ignore[attr-defined]
                messages=llm_messages,
                max_tokens=4000,
                temperature=0.7,
            ):
                yield chunk
        except AttributeError:
            # Provider does not support streaming; fall back to single response
            response = await self.llm_provider.generate_response(
                messages=llm_messages,
                max_tokens=4000,
                temperature=0.7,
            )
            yield response.content

    async def _format_response(
        self,
        llm_response: LLMResponse,
        tool_data: Dict[str, Any],
        persona_name: str,
        conversation_id: str,
        start_time: datetime
    ) -> AgentResponse:
        """Format the final agent response for API compatibility"""
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract panels and sources from tool data
        panels = {}
        sources = []
        reply_text = llm_response.content
        
        if 'portfolio' in tool_data:
            portfolio_info = tool_data['portfolio']
            portfolio_sources = portfolio_info.get('sources', [])
            panels['portfolio_overview'] = {
                'id': 'portfolio_overview',
                'kind': 'portfolio',
                'title': 'Your Portfolio Snapshot',
                'payload': portfolio_info['data'],
                'sources': portfolio_sources,
                'metadata': {
                    'warnings': portfolio_info.get('warnings', []),
                },
            }
            sources.extend(portfolio_sources)

        if 'history_summary' in tool_data:
            summary_info = tool_data['history_summary']
            summary_payload = summary_info['snapshot']
            metadata: Dict[str, Any] = {}
            existing_metadata = summary_info.get('metadata')
            if isinstance(existing_metadata, dict):
                metadata.update(existing_metadata)
            metadata.setdefault('density', 'full')
            panels['history_summary'] = {
                'id': 'history-summary',
                'kind': 'history-summary',
                'title': 'Wallet Activity Summary',
                'payload': summary_payload,
                'sources': [{'name': summary_payload.get('chain', 'multichain')}],
            }
            if summary_info.get('limit'):
                metadata['sampleLimit'] = summary_info['limit']
            if metadata:
                panels['history_summary']['metadata'] = metadata

        if 'trending_tokens' in tool_data:
            trending_info = tool_data['trending_tokens']
            tokens = trending_info.get('tokens') or []
            if tokens:
                panel_id = 'trending_tokens'
                panel_sources = trending_info.get('sources') or self._collect_trending_sources(tokens)
                metadata = {
                    'layout': 'banner',
                    'totalAvailable': trending_info.get('total_available'),
                    'query': trending_info.get('query'),
                    'focusSymbol': (trending_info.get('focus') or {}).get('symbol'),
                }
                extra_metadata = trending_info.get('metadata') or {}
                for key, value in extra_metadata.items():
                    if value is not None:
                        metadata[key] = value

                panels[panel_id] = {
                    'id': panel_id,
                    'kind': 'trending',
                    'title': trending_info.get('title') or 'Trending Tokens',
                    'payload': {
                        'tokens': tokens,
                        'fetchedAt': trending_info.get('fetched_at'),
                        'focusToken': trending_info.get('focus'),
                        'totalAvailable': trending_info.get('total_available'),
                    },
                    'sources': panel_sources,
                    'metadata': metadata,
                }
                sources.extend(panel_sources)

        # Add DefiLlama TVL panel if available
        if 'defillama_tvl' in tool_data:
            tvl_info = tool_data['defillama_tvl']
            ts = tvl_info.get('timestamps', [])
            tvl = tvl_info.get('tvl', [])
            protocol = tvl_info.get('protocol', 'uniswap')
            window = tvl_info.get('window', '7d')
            # Panel with normalized shape for frontend transformer
            tvl_sources = tvl_info.get('sources') or [
                {
                    'name': 'DefiLlama',
                    'url': f'https://defillama.com/protocol/{protocol}'
                }
            ]
            panels['uniswap_tvl_chart'] = {
                'kind': 'chart',
                'title': f'{protocol.title()} TVL ({window})',
                'payload': {
                    'timestamps': ts,
                    'tvl': tvl,
                    'unit': 'USD',
                    'protocol': protocol,
                    'window': window,
                    'stats': tvl_info.get('stats', {}),
                },
                'sources': tvl_sources,
                'metadata': {
                    'stats': tvl_info.get('stats', {}),
                },
            }
            # Attribution
            sources.extend(tvl_sources)
            # Deterministic short analysis to avoid LLM disclaimers
            stats = tvl_info.get('stats') or {}
            summary = self._compose_tvl_reply(protocol, window, stats)
            if summary:
                reply_text = summary

        if 'token_chart' in tool_data:
            chart_info = tool_data['token_chart']
            metadata = chart_info.get('metadata') or {}
            symbol = metadata.get('symbol') or metadata.get('name') or chart_info.get('coin_id') or 'Token'
            slug = ''.join(ch for ch in str(symbol).lower() if ch.isalnum()) or 'token'
            panel_id = f'{slug}_price_chart'
            panel_title = f"{symbol} price chart ({str(chart_info.get('range', '7d')).upper()})"
            panel_sources = chart_info.get('sources') or [
                {
                    'name': 'CoinGecko',
                    'url': 'https://www.coingecko.com',
                }
            ]
            panels[panel_id] = {
                'id': panel_id,
                'kind': 'chart',
                'title': panel_title,
                'payload': chart_info,
                'sources': panel_sources,
                'metadata': {
                    'symbol': metadata.get('symbol'),
                    'range': chart_info.get('range'),
                },
            }
            sources.extend(panel_sources)
            summary = self._compose_token_chart_reply(symbol, chart_info.get('range', '7d'), chart_info.get('stats'))
            if summary:
                reply_text = summary

        bridge_info = tool_data.get('bridge_quote')
        if bridge_info:
            panel = bridge_info.get('panel')
            if panel and isinstance(panel, dict):
                panel_id = panel.get('id', 'relay_bridge_quote')
                panels[panel_id] = panel
                panel_sources = panel.get('sources') or []
                if panel_sources:
                    sources.extend(panel_sources)
            summary_reply = bridge_info.get('summary_reply')
            if summary_reply:
                reply_text = summary_reply
            else:
                message = bridge_info.get('message')
                if message:
                    reply_text = message

        swap_info = tool_data.get('swap_quote')
        if swap_info:
            panel = swap_info.get('panel')
            if panel and isinstance(panel, dict):
                panel_id = panel.get('id', 'relay_swap_quote')
                panels[panel_id] = panel
                panel_sources = panel.get('sources') or []
                if panel_sources:
                    sources.extend(panel_sources)
            summary_reply = swap_info.get('summary_reply')
            if summary_reply:
                reply_text = summary_reply
            else:
                message = swap_info.get('message')
                if message:
                    reply_text = message

        # Add strategies panel if available
        strategies_info = tool_data.get('strategies')
        if strategies_info and strategies_info.get('success'):
            strategies_list = strategies_info.get('strategies', [])
            panels['my_strategies'] = {
                'id': 'my-strategies',
                'kind': 'my-strategies',
                'title': 'My Strategies',
                'payload': {
                    'strategies': strategies_list,
                    'count': len(strategies_list),
                    'walletAddress': strategies_info.get('wallet_address'),
                },
                'sources': [],
                'metadata': {
                    'density': 'full',
                },
            }

        # Add single strategy detail panel if available
        strategy_detail = tool_data.get('strategy_detail')
        if strategy_detail and strategy_detail.get('success'):
            strategy = strategy_detail.get('strategy', {})
            panels['strategy_detail'] = {
                'id': 'strategy-detail',
                'kind': 'strategy-detail',
                'title': strategy.get('name', 'Strategy Details'),
                'payload': strategy,
                'sources': [],
                'metadata': {
                    'density': 'full',
                },
            }

        # If the user asked for portfolio insights but we couldn't load data, provide a safe, helpful prompt instead of guessing
        if tool_data.get('needs_portfolio') and 'portfolio' not in tool_data:
            addr = tool_data.get('_address')
            if addr:
                reply_text = (
                    f"I can analyze your holdings at {addr}, but I couldn't load the latest portfolio data just now. "
                    f"Would you like me to try again, or specify a different address or chain?"
                )
            else:
                reply_text = (
                    "I don’t see a wallet address yet. Share an address (e.g., 0x…) "
                    "or connect your wallet and I’ll analyze your portfolio without guessing."
                )
        
        return AgentResponse(
            reply=reply_text,
            panels=panels,
            sources=sources,
            agent_metadata={
                'llm_model': llm_response.model,
                'finish_reason': llm_response.finish_reason,
                'tool_data_keys': list(tool_data.keys())
            },
            persona_used=persona_name,
            conversation_id=conversation_id,
            tokens_used=llm_response.tokens_used,
            processing_time_ms=processing_time
        )

    async def _update_conversation_state(
        self,
        conversation_id: str,
        request: ChatRequest,
        response: AgentResponse
    ) -> None:
        """Update conversation state and history"""
        
        # Store conversation state
        self._active_conversations[conversation_id] = {
            'last_activity': datetime.now(),
            'persona': response.persona_used,
            'message_count': len(request.messages)
        }
        
        # Update context manager if available
        if self.context_manager:
            # Add user message
            if request.messages:
                await self.context_manager.add_message(
                    conversation_id,
                    request.messages[-1].model_dump()
                )
                
            # Add assistant response
            await self.context_manager.add_message(
                conversation_id,
                {
                    'role': 'assistant',
                    'content': response.reply
                }
            )

    async def _create_error_response(
        self,
        error_message: str,
        conversation_id: Optional[str],
        start_time: datetime
    ) -> AgentResponse:
        """Create error response when agent processing fails"""
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentResponse(
            reply=f"I apologize, but I encountered an error while processing your request: {error_message}. Please try again.",
            panels={},
            sources=[],
            agent_metadata={'error': error_message},
            conversation_id=conversation_id,
            processing_time_ms=processing_time
        )

    def _needs_portfolio_data(self, request: ChatRequest) -> bool:
        """Determine if the request needs portfolio data"""
        
        if not request.messages:
            return False
            
        # Check for portfolio-related keywords
        last_message = request.messages[-1].content.lower()
        portfolio_keywords = [
            "portfolio", "balance", "tokens", "holdings", "wallet",
            "what's in", "whats in", "analyze", "analysis", "show me",
            "performance", "performing", "returns", "pnl", "gain", "loss",
            "rotate", "rotation", "rebalance", "re-balance", "allocation",
            "diversify", "exposure", "concentration", "positions", "assets",
            "coins"
        ]
        
        return any(keyword in last_message for keyword in portfolio_keywords)

    def _needs_trending_data(self, request: ChatRequest) -> bool:
        """Heuristic detection for trending token queries."""

        if not request.messages:
            return False

        last_message = request.messages[-1].content.lower()
        if 'trending' in last_message:
            return True

        triggers = [
            'top gainer',
            'top gainers',
            'top loser',
            'top losers',
            'top coins',
            'top tokens',
            'hot coin',
            'hot token',
            'momentum coin',
            'momentum token',
        ]

        return any(trigger in last_message for trigger in triggers)

    def _needs_history_summary(self, request: ChatRequest) -> bool:
        if not request.messages:
            return False
        message = request.messages[-1].content.lower()
        triggers = [
            "history",
            "transaction",
            "activity",
            "recent moves",
            "recent transfers",
            "summarize moves",
            "compare this month",
            "export",
        ]
        return any(trigger in message for trigger in triggers)

    def _apply_history_window_metadata(self, snapshot: Dict[str, Any], window: HistoryWindowContext) -> None:
        metadata = snapshot.setdefault('metadata', {})
        metadata.setdefault(
            'requestedWindow',
            {
                'start': window.requested_start.isoformat(),
                'end': window.requested_end.isoformat(),
            },
        )
        metadata.setdefault('requestedWindowDays', window.requested_days)
        metadata['syncWindowDays'] = window.applied_days
        if window.clamped:
            metadata['windowClamped'] = True
            metadata['clampedWindowDays'] = window.applied_days

    def _extract_history_window(self, request: ChatRequest) -> HistoryWindowContext:
        now = datetime.now(timezone.utc)
        message = request.messages[-1].content.lower() if request.messages else ""
        days = 30
        if "last week" in message or "7d" in message:
            days = 7
        elif "90" in message or "quarter" in message:
            days = 90
        elif "year" in message or "12 months" in message:
            days = 180
        else:
            match = re.search(r"last\s+(\d{1,3})\s*(day|week|month)", message)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if unit.startswith("day"):
                    days = value
                elif unit.startswith("week"):
                    days = value * 7
                elif unit.startswith("month"):
                    days = value * 30
        requested_days = max(7, min(days, 365))
        applied_days = max(7, min(requested_days, HISTORY_SYNC_DAY_CAP))
        requested_start = now - timedelta(days=requested_days)
        applied_start = now - timedelta(days=applied_days)
        return HistoryWindowContext(
            start=applied_start,
            end=now,
            requested_start=requested_start,
            requested_end=now,
            requested_days=requested_days,
            applied_days=applied_days,
        )

    def _extract_trending_query(self, request: ChatRequest) -> Optional[str]:
        """Extract a specific token symbol/name from a trending-related query."""

        if not request.messages:
            return None

        message = request.messages[-1].content.strip()
        if not message:
            return None

        lowered = message.lower()
        relevant = (
            'trending' in lowered
            or 'token' in lowered
            or 'coin' in lowered
            or 'ticker' in lowered
            or 'symbol' in lowered
            or 'gainer' in lowered
            or 'loser' in lowered
        )
        if not relevant:
            return None

        patterns = [
            r'(?:token|coin|symbol|ticker)\s+([A-Za-z0-9]{2,15})',
            r'about\s+([A-Za-z0-9]{2,15})\b',
            r'called\s+([A-Za-z0-9]{2,15})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip(".,;:!\"'`")
                if 1 < len(candidate) <= 20:
                    return candidate.upper()

        fallback_matches = re.findall(r'\b[A-Z0-9]{2,10}\b', message)
        for candidate in reversed(fallback_matches):
            if candidate and not candidate.isdigit():
                return candidate.upper()

        return None

    def _extract_token_chart_request(self, request: ChatRequest) -> Optional[Dict[str, Any]]:
        if not request.messages:
            return None

        message = request.messages[-1].content or ''
        lowered = message.lower()

        # Direct chart keywords
        chart_keywords = (
            'chart',
            'candles',
            'candlestick',
            'price action',
            'ohlc',
            'price graph',
            'price plot',
            'price chart',
            'price history',
            'historical price',
            'price over time',
            'price movement',
            'price trend',
            'show.*price',
            'how.*price',
        )

        # Check for direct keyword matches
        has_chart_keyword = any(keyword in lowered for keyword in chart_keywords if '.*' not in keyword)

        # Check for regex patterns (show me X price, how has X price)
        if not has_chart_keyword:
            import re as regex_module
            for pattern in chart_keywords:
                if '.*' in pattern and regex_module.search(pattern, lowered):
                    has_chart_keyword = True
                    break

        # Also trigger on "show me [token]" patterns for known token mentions
        if not has_chart_keyword:
            show_patterns = (
                r"show\s+(?:me\s+)?(?:the\s+)?(\w+)(?:'s)?\s*(?:price|chart|history)?",
                r"(\w+)(?:'s)?\s+(?:price\s+)?history",
                r"what(?:'s|\s+is)\s+(\w+)(?:'s)?\s+price",
                r"how\s+(?:has|is|did)\s+(\w+)\s+(?:been\s+)?(?:doing|performing|price)",
            )
            import re as regex_module
            for pattern in show_patterns:
                match = regex_module.search(pattern, lowered)
                if match:
                    has_chart_keyword = True
                    break

        if not has_chart_keyword:
            return None

        contract_address = self._extract_contract_address(message)
        symbol = self._find_token_symbol(message)

        if not contract_address and not symbol:
            return None

        range_key = self._extract_chart_range(lowered)

        request_payload: Dict[str, Any] = {
            'range': range_key,
            'vs_currency': 'usd',
        }

        if contract_address:
            request_payload['contract_address'] = contract_address
        if symbol and not contract_address:
            # prefer explicit coin id when the symbol looks like a known slug
            known_coin_slugs = {
                'ethereum', 'bitcoin', 'solana', 'cardano', 'dogecoin',
                'litecoin', 'polkadot', 'tron', 'avalanche', 'chainlink',
                'morpho', 'uniswap', 'aave', 'compound', 'maker',
                'curve-dao-token', 'lido-dao', 'rocket-pool', 'frax',
                'convex-finance', 'yearn-finance', 'sushi', 'balancer',
                'pendle', '1inch', 'gmx', 'radiant-capital', 'arbitrum',
                'optimism', 'polygon', 'base', 'mantle', 'celestia',
            }
            if symbol.lower() in known_coin_slugs:
                request_payload['coin_id'] = symbol.lower()
            else:
                request_payload['symbol'] = symbol

        return request_payload

    def _extract_chart_range(self, message_lower: str) -> str:
        if any(term in message_lower for term in ('1d', 'one day', '24h', 'today', 'daily')):
            return '1d'
        if any(term in message_lower for term in ('30d', 'thirty', 'month', '30 days')):
            return '30d'
        if '90d' in message_lower or 'quarter' in message_lower:
            return '90d'
        if '180d' in message_lower or '6 month' in message_lower or 'half year' in message_lower:
            return '180d'
        if '365' in message_lower or 'year' in message_lower or '12 month' in message_lower:
            return '365d'
        if 'max' in message_lower or 'all time' in message_lower:
            return 'max'
        return '7d'

    def _extract_contract_address(self, message: str) -> Optional[str]:
        match = re.search(r'0x[a-fA-F0-9]{40}', message)
        if match:
            return match.group(0).lower()
        return None

    def _extract_history_limit(self, request: ChatRequest) -> Optional[int]:
        if not request.messages:
            return None
        message = request.messages[-1].content.lower()
        pattern = re.search(r"(last|latest|past|previous)\s+(\d{1,4})\s+(tx|transaction|transactions|transfer|transfers)", message)
        if pattern:
            value = int(pattern.group(2))
            return max(10, min(value, 2500))
        return None

    def _mentions_time_window(self, request: ChatRequest) -> bool:
        if not request.messages:
            return False
        message = request.messages[-1].content.lower()
        window_keywords = ["day", "days", "week", "weeks", "month", "months", "year", "years", "since", "past"]
        return any(keyword in message for keyword in window_keywords)

    def _find_token_symbol(self, message: str) -> Optional[str]:
        symbol_matches = re.findall(r'\$([A-Za-z0-9]{2,10})', message)
        if symbol_matches:
            return symbol_matches[0].upper()

        stopwords = {
            'SHOW', 'PLEASE', 'PRICE', 'CHART', 'CANDLE', 'CANDLES', 'TOKEN', 'TOKENS', 'COIN', 'COINS',
            'GRAPH', 'LOOK', 'SEE', 'CAN', 'YOU', 'THE', 'FOR', 'WITH', 'ABOUT', 'NEED', 'HELP', 'THIS',
            'THAT', 'REAL', 'TIME', 'LATEST', 'GIVE', 'GET', 'WHAT', 'AN', 'A', 'ME', 'US', 'PLOT',
            'DISPLAY', 'OF', 'INTO', 'CHECK', 'PRICEACTION', 'PLEASESHOW', 'PLEASES', 'ON', 'TO', 'IS',
            'HISTORY', 'HISTORICAL', 'MOVEMENT', 'TREND', 'ACTION', 'OVER', 'PAST', 'LAST', 'DAYS',
            'WEEK', 'MONTH', 'YEAR', 'HOW', 'HAS', 'BEEN', 'DOING', 'PERFORMING', 'UP', 'DOWN',
            'SPIN', 'CREATE', 'GENERATE', 'MAKE', 'BUILD', 'DRAW', 'RENDER', 'WIDGET',
        }

        words = re.findall(r'\b[a-zA-Z0-9]{2,15}\b', message)
        candidates: List[str] = []
        for word in words:
            upper_word = word.upper()
            if upper_word in stopwords:
                continue
            if upper_word.isdigit():
                continue
            if len(upper_word) > 12:
                continue
            candidates.append(upper_word)

        for candidate in reversed(candidates):
            return candidate
        return None

    def _match_trending_token(
        self,
        tokens: Sequence[Dict[str, Any]],
        query: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Return the first trending token matching the query by symbol, name, or id."""

        if not query:
            return None

        q = query.lower()

        def _normalize(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text.lower() if text else None

        for token in tokens:
            symbol = _normalize(token.get('symbol'))
            name = _normalize(token.get('name'))
            coin_id = _normalize(token.get('id'))
            contract = _normalize(token.get('contract_address'))
            if q in {symbol, name, coin_id, contract}:
                return token

        for token in tokens:
            name = _normalize(token.get('name'))
            if name and q in name:
                return token

        return None

    def _token_identity(self, token: Optional[Dict[str, Any]]) -> str:
        """Return a lowercase identity string for a token to help deduplicate entries."""

        if not token:
            return ''

        for key in ('id', 'contract_address', 'symbol'):
            value = token.get(key)
            if value:
                text = str(value).strip().lower()
                if text:
                    return text
        return ''

    def _collect_trending_sources(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate unique data sources from trending token entries."""

        seen = set()
        sources: List[Dict[str, Any]] = []

        for token in tokens:
            source = token.get('_source')
            if not isinstance(source, dict):
                continue
            name = (source.get('name') or '').strip()
            url = (source.get('url') or '').strip()
            key = (name.lower(), url.lower())
            if key in seen:
                continue
            seen.add(key)
            entry: Dict[str, Any] = {}
            if name:
                entry['name'] = name
            if url:
                entry['url'] = url
            if entry:
                sources.append(entry)

        if not sources:
            sources.append({'name': 'CoinGecko', 'url': 'https://www.coingecko.com'})

        return sources

    def _extract_wallet_address(self, request: ChatRequest) -> Optional[str]:
        """Extract wallet address from request"""
        
        # First check if address is provided directly
        if request.address:
            return request.address
            
        # Then check message content for addresses
        if request.messages:
            last_message = request.messages[-1].content
            # Simple regex for Ethereum addresses
            pattern = r'0x[a-fA-F0-9]{40}'
            matches = re.findall(pattern, last_message)
            if matches:
                return matches[0]
                
        return None

    def _format_tool_data_for_llm(self, tool_data: Dict[str, Any]) -> str:
        """Format tool data into a readable context for the LLM"""

        formatted_parts = []

        # Always include the connected wallet address if available
        # This ensures the LLM knows a wallet is connected even if portfolio fetch fails
        wallet_address = tool_data.get('_address')
        if wallet_address:
            formatted_parts.append(f"Connected wallet: {wallet_address}")

        def _fmt_price(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return 'n/a'
            abs_val = abs(val)
            if abs_val >= 1000:
                return f"${val:,.0f}"
            if abs_val >= 1:
                return f"${val:,.2f}"
            return f"${val:,.6f}"

        def _fmt_pct(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return 'n/a'
            return f"{val:+.2f}%"

        def _fmt_large(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return 'n/a'
            if val >= 1_000_000_000:
                return f"${val/1_000_000_000:.2f}B"
            if val >= 1_000_000:
                return f"${val/1_000_000:.2f}M"
            if val >= 1_000:
                return f"${val/1_000:.2f}K"
            return f"${val:,.0f}"
        
        if 'portfolio' in tool_data:
            portfolio = tool_data['portfolio']['data']
            formatted_parts.append(
                f"Portfolio for {portfolio['address']}: "
                f"Total value: ${portfolio['total_value_usd']} USD, "
                f"{portfolio['token_count']} tokens"
            )
            tokens = portfolio.get('tokens') or []
            if tokens:
                snippets = []
                for token in tokens[:8]:
                    symbol = token.get('symbol') or token.get('name') or 'token'
                    balance = token.get('balance_formatted') or '0'
                    value = _fmt_price(token.get('value_usd'))
                    snippets.append(f"{symbol}: {balance} ({value})")
                remainder = len(tokens) - len(snippets)
                holdings_summary = ", ".join(snippets)
                if remainder > 0:
                    holdings_summary += f", +{remainder} more"
                formatted_parts.append(f"Holdings breakdown: {holdings_summary}")

        if 'portfolio_error' in tool_data:
            error_info = tool_data['portfolio_error']
            formatted_parts.append(f"Portfolio data unavailable: {error_info['error']}")

        if 'trending_tokens' in tool_data:
            trending_info = tool_data['trending_tokens']
            tokens = trending_info.get('tokens') or []
            if tokens:
                snippets = []
                for token in tokens[:5]:
                    symbol = token.get('symbol') or token.get('name') or 'token'
                    price_str = _fmt_price(token.get('price_usd'))
                    change_str = _fmt_pct(token.get('change_24h'))
                    snippets.append(f"{symbol}: {price_str} ({change_str})")
                formatted_parts.append(
                    "Trending tokens (24h): " + ", ".join(snippets)
                )
            focus = trending_info.get('focus')
            if focus:
                focus_name = focus.get('name') or focus.get('symbol') or 'token'
                focus_price = _fmt_price(focus.get('price_usd'))
                focus_change = _fmt_pct(focus.get('change_24h'))
                focus_volume = _fmt_large(focus.get('volume_24h'))
                formatted_parts.append(
                    f"Focus token {focus_name}: {focus_price}, 24h change {focus_change}, volume {focus_volume}"
                )

        if 'history_summary' in tool_data:
            history_info = tool_data['history_summary']
            summary = history_info['snapshot']
            window = summary.get('timeWindow', {})
            totals = summary.get('totals', {})
            metadata = summary.get('metadata') or {}
            limit_sample = history_info.get('limit')
            if limit_sample:
                formatted_parts.append(
                    f"Latest {limit_sample} transfers: "
                    f"inflow {_fmt_large(totals.get('inflowUsd'))}, "
                    f"outflow {_fmt_large(totals.get('outflowUsd'))}, "
                    f"fees {_fmt_large(totals.get('feeUsd'))}."
                )
            else:
                formatted_parts.append(
                    "Wallet history "
                    f"{window.get('start')} -> {window.get('end')}: "
                    f"inflow {_fmt_large(totals.get('inflowUsd'))}, "
                    f"outflow {_fmt_large(totals.get('outflowUsd'))}, "
                    f"fees {_fmt_large(totals.get('feeUsd'))}."
                )
                if metadata.get('windowClamped'):
                    requested_window = _format_requested_window(metadata.get('requestedWindow'))
                    clamp_days = metadata.get('clampedWindowDays') or HISTORY_SYNC_DAY_CAP
                    formatted_parts.append(
                        f"Requested window {requested_window} exceeds the live cap, so I'm showing the most recent {clamp_days} days. "
                        "Ask for an export if you need the full range."
                    )
            for event in (summary.get('notableEvents') or [])[:3]:
                formatted_parts.append(f"History flag ({event.get('severity')}): {event.get('summary')}")

        if 'history_summary_error' in tool_data:
            formatted_parts.append(f"History summary unavailable: {tool_data['history_summary_error']}")

        if 'token_chart' in tool_data:
            chart = tool_data['token_chart']
            stats = chart.get('stats') or {}
            meta = chart.get('metadata') or {}
            symbol = meta.get('symbol') or meta.get('name') or chart.get('coin_id') or 'token'
            latest_str = _fmt_price(stats.get('latest'))
            change_pct = _fmt_pct(stats.get('change_pct'))
            range_key = str(chart.get('range', '7d')).upper()
            high_str = _fmt_price(stats.get('high'))
            low_str = _fmt_price(stats.get('low'))
            formatted_parts.append(
                f"{symbol} {range_key} price: {latest_str} ({change_pct}), range {low_str} to {high_str}"
            )

        if 'token_chart_error' in tool_data:
            formatted_parts.append(f"Token chart unavailable: {tool_data['token_chart_error']}")

        if 'trending_error' in tool_data:
            formatted_parts.append(f"Trending token data unavailable: {tool_data['trending_error']}")
        
        if 'defillama_tvl' in tool_data:
            tvl_obj = tool_data['defillama_tvl']
            protocol = tvl_obj.get('protocol', 'uniswap')
            window = tvl_obj.get('window', '7d')
            ts = tvl_obj.get('timestamps', [])
            series = tvl_obj.get('tvl', [])
            stats = tvl_obj.get('stats', {})
            if series:
                start_v = stats.get('start_value', series[0])
                end_v = stats.get('end_value', series[-1])
                chg = stats.get('abs_change', end_v - start_v)
                pct = stats.get('pct_change', ((end_v - start_v) / start_v * 100) if start_v else 0)
                min_v = stats.get('min_value')
                max_v = stats.get('max_value')
                min_d = stats.get('min_date')
                max_d = stats.get('max_date')
                trend = stats.get('trend')
                formatted_parts.append(
                    f"{protocol.title()} TVL ({window}) – start ${round(float(start_v),2)}, end ${round(float(end_v),2)}, "
                    f"change ${round(float(chg),2)} ({round(float(pct),2)}%), min ${round(float(min_v),2)} on {min_d}, "
                    f"max ${round(float(max_v),2)} on {max_d}, trend {trend}"
                )
            else:
                formatted_parts.append(f"{protocol.title()} TVL ({window}) data available")

        if 'bridge_quote' in tool_data:
            bridge_info = tool_data['bridge_quote']
            summary = bridge_info.get('summary_tool')
            if summary:
                formatted_parts.append(summary)
            else:
                message = bridge_info.get('message')
                if message:
                    formatted_parts.append(f"Bridge status: {message}")

        return " | ".join(formatted_parts) if formatted_parts else "No additional data available"

    def _compute_tvl_stats(self, timestamps: List[int], tvl: List[float]) -> Dict[str, Any]:
        try:
            if not tvl:
                return {}
            n = len(tvl)
            start_value = float(tvl[0])
            end_value = float(tvl[-1])
            abs_change = end_value - start_value
            pct_change = (abs_change / start_value * 100.0) if start_value else 0.0
            min_idx = int(min(range(n), key=lambda i: tvl[i]))
            max_idx = int(max(range(n), key=lambda i: tvl[i]))
            min_value = float(tvl[min_idx])
            max_value = float(tvl[max_idx])
            avg = float(mean(tvl)) if n > 0 else 0.0
            vol = float(pstdev(tvl)) if n > 1 else 0.0
            # Trend detection with a small threshold for noise
            threshold = 0.005  # 0.5%
            trend = 'flat'
            if start_value > 0:
                rel = (end_value - start_value) / start_value
                if rel > threshold:
                    trend = 'up'
                elif rel < -threshold:
                    trend = 'down'
            # Dates
            def to_date(i: int) -> str:
                try:
                    ts = timestamps[i] if i < len(timestamps) else None
                    return datetime.fromtimestamp(ts/1000).date().isoformat() if ts else f"t[{i}]"
                except Exception:
                    return f"t[{i}]"
            return {
                'start_value': start_value,
                'end_value': end_value,
                'abs_change': abs_change,
                'pct_change': pct_change,
                'min_value': min_value,
                'max_value': max_value,
                'min_index': min_idx,
                'max_index': max_idx,
                'min_date': to_date(min_idx),
                'max_date': to_date(max_idx),
                'avg': avg,
                'volatility': vol,
                'trend': trend,
            }
        except Exception:
            return {}

    def _compose_tvl_reply(self, protocol: str, window: str, stats: Dict[str, Any]) -> str:
        try:
            if not stats:
                return ""
            start_v = stats.get('start_value')
            end_v = stats.get('end_value')
            abs_change = stats.get('abs_change')
            pct_change = stats.get('pct_change')
            min_v = stats.get('min_value')
            max_v = stats.get('max_value')
            min_d = stats.get('min_date')
            max_d = stats.get('max_date')
            trend = stats.get('trend', 'flat')
            if start_v is None or end_v is None:
                return ""
            arrow = '↑' if (pct_change or 0) > 0 else ('↓' if (pct_change or 0) < 0 else '→')
            lines = [
                f"{protocol.title()} TVL ({window}) {arrow}",
                f"- Start → End: ${start_v:,.0f} → ${end_v:,.0f} ({abs_change:+,.0f}, {pct_change:+.2f}%)",
            ]
            if min_v is not None and max_v is not None:
                lines.append(f"- Range: min ${min_v:,.0f} on {min_d}, max ${max_v:,.0f} on {max_d}")
            lines.append(f"- Trend: {trend}")
            lines.append("- See the TVL chart panel on the right for details.")
            return "\n".join(lines)
        except Exception:
            return ""

    def _compose_token_chart_reply(self, symbol: str, window: str, stats: Optional[Dict[str, Any]]) -> str:
        try:
            if not stats:
                return ""
            latest = stats.get('latest')
            change_abs = stats.get('change_abs')
            change_pct = stats.get('change_pct')
            high = stats.get('high')
            high_time = stats.get('high_time')
            low = stats.get('low')
            low_time = stats.get('low_time')
            if latest is None or change_pct is None:
                return ""
            arrow = '↑' if change_pct > 0 else ('↓' if change_pct < 0 else '→')
            def _fmt_price(value: Any) -> str:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    return 'n/a'
                abs_val = abs(val)
                if abs_val >= 1000:
                    return f"${val:,.0f}"
                if abs_val >= 1:
                    return f"${val:,.2f}"
                if abs_val >= 0.0001:
                    return f"${val:,.6f}"
                # For very small values like memecoins (PEPE, SHIB, etc.)
                return f"${val:.10f}".rstrip('0').rstrip('.')
            def _fmt_change(value: Any) -> str:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    return 'n/a'
                abs_val = abs(val)
                if abs_val >= 0.01:
                    return f"{val:+.2f}"
                return f"{val:+.8f}".rstrip('0').rstrip('.')
            def _fmt_time(value: Any) -> str:
                try:
                    return datetime.fromtimestamp(int(value)/1000).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    return 'n/a'
            lines = [
                f"{symbol} price ({window.upper()}) {arrow}",
                f"- Latest: {_fmt_price(latest)} ({_fmt_change(change_abs)}, {change_pct:+.2f}%)",
            ]
            if high is not None and low is not None:
                lines.append(
                    f"- Range: high {_fmt_price(high)} on {_fmt_time(high_time)}, low {_fmt_price(low)} on {_fmt_time(low_time)}"
                )
            lines.append("- See the price chart panel on the right for candles and volume.")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt when no persona manager is available"""
        
        return """You are a knowledgeable and friendly crypto portfolio assistant.
        
        Scope:
        - Focus strictly on crypto portfolios, tokens, protocols, and wallet analytics.
        - Treat ambiguous requests (e.g., “rotate”, “performance”) as crypto portfolio topics by default.
        
        Guardrails:
        - Never invent or assume specific holdings. If portfolio data is unavailable, ask for a wallet address or permission to analyze it.
        - If data appears inconsistent, briefly acknowledge uncertainty and ask a targeted clarification.
        
        Style:
        - Be concise, actionable, and concrete.
        - Prefer bullet points for insights and recommendations.
        
        When portfolio data is available, prioritize:
        - Allocation breakdown, concentration, diversification
        - Recent performance, notable movers, risk flags
        - Clear next steps (e.g., rebalance ideas, questions to confirm preferences)"""

    async def _ensure_portfolio_context(self, conversation_id: str, request: ChatRequest) -> None:
        """Proactively fetch and cache portfolio context if address is provided and context is missing."""
        try:
            if not (self.context_manager and request.address):
                return
            # Check current context for portfolio
            ctx = self.context_manager._conversations.get(conversation_id) if hasattr(self.context_manager, '_conversations') else None
            if ctx and getattr(ctx, 'portfolio_context', None):
                return
            wallet_address = self._extract_wallet_address(request)
            if not wallet_address:
                return
            portfolio_result = await get_portfolio(wallet_address, request.chain)
            if portfolio_result.data:
                await self.context_manager.integrate_portfolio_data(conversation_id, portfolio_result.data.model_dump(mode='json'))
        except Exception as e:
            self.logger.debug(f"Proactive portfolio fetch skipped: {e}")

    async def _handle_style_processing(
        self,
        messages: List[ChatMessage],
        conversation_id: str
    ) -> Optional[AgentResponse]:
        """
        Handle style commands and automatic style detection.
        Returns AgentResponse if a style command was processed, None otherwise.
        """
        if not messages:
            return None
            
        latest_message = messages[-1].content
        
        # Check for explicit style commands first
        style_command = self.style_manager.parse_style_command(latest_message)
        if style_command:
            # Update conversation style
            self._conversation_styles[conversation_id] = style_command
            self.style_manager.set_style(style_command)
            
            # Return confirmation response
            style_info = self.style_manager.get_style_info(style_command)
            return AgentResponse(
                reply=f"✨ Style switched to **{style_info.get('name', style_command.value.title())}**!\n\n"
                      f"{style_info.get('description', 'Style updated successfully.')}\n\n"
                      f"All my responses will now use this style. You can switch styles anytime with `/style [name]` "
                      f"or type `/style help` to see all available options.",
                panels={},
                sources=[],
                agent_metadata={'style_switched': style_command.value},
                conversation_id=conversation_id,
                processing_time_ms=0.0
            )
        
        # Check for style help command
        if latest_message.lower().strip() in ['/style help', '/style', '/styles']:
            help_text = self.style_manager.format_style_help()
            return AgentResponse(
                reply=help_text,
                panels={},
                sources=[],
                agent_metadata={'style_help': True},
                conversation_id=conversation_id,
                processing_time_ms=0.0
            )
        
        # Automatic style detection (only if no conversation style set yet)
        if conversation_id not in self._conversation_styles:
            detected_style = self.style_manager.detect_style_from_message(latest_message)
            if detected_style:
                self._conversation_styles[conversation_id] = detected_style
                self.style_manager.set_style(detected_style)
                
                # Log the automatic detection
                self.logger.info(f"Auto-detected style '{detected_style.value}' for conversation {conversation_id}")
        
        return None
    
    def to_chat_response(self, agent_response: AgentResponse) -> ChatResponse:
        """Convert AgentResponse to ChatResponse for API compatibility"""
        
        return ChatResponse(
            reply=agent_response.reply,
            panels=agent_response.panels,
            sources=agent_response.sources,
            conversation_id=agent_response.conversation_id,
            llm_provider=self.provider_id,
            llm_model=self.model_id or getattr(self.llm_provider, "model", None),
        )


def _format_requested_window(window: Optional[Dict[str, Any]]) -> str:
    if not window or not isinstance(window, dict):
        return "the requested range"
    start_label = _fmt_iso_day(window.get('start'))
    end_label = _fmt_iso_day(window.get('end'))
    if start_label and end_label:
        return f"{start_label} -> {end_label}"
    return "the requested range"


def _fmt_iso_day(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        return value
