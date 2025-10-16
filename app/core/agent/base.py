"""
Core Agent System

This module contains the main Agent class that orchestrates LLM interactions,
persona management, and tool integration for intelligent portfolio analysis.
"""

import copy
import logging
import re
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union
import uuid

import httpx
from pydantic import BaseModel, Field

from ...providers.llm.base import LLMProvider, LLMMessage, LLMResponse
from ...tools.portfolio import get_portfolio
from ...services.trending import get_trending_tokens
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from ..bridge import BridgeManager
from ..swap import SwapManager
from .styles import StyleManager, ResponseStyle
from .graph import build_agent_process_graph


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
        logger: Optional[logging.Logger] = None
    ):
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.context_manager = context_manager
        self.style_manager = style_manager or StyleManager()
        self.logger = logger or logging.getLogger(__name__)
        
        # Agent state
        self._active_conversations: Dict[str, Dict] = {}
        # Style state per conversation
        self._conversation_styles: Dict[str, ResponseStyle] = {}
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
                            'data': portfolio_result.data.model_dump(),
                            'sources': [s.model_dump() for s in portfolio_result.sources],
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

        return tool_data

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
                await self.context_manager.integrate_portfolio_data(conversation_id, portfolio_result.data.model_dump())
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
        )
