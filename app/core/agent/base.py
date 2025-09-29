"""
Core Agent System

This module contains the main Agent class that orchestrates LLM interactions,
persona management, and tool integration for intelligent portfolio analysis.
"""

import copy
import logging
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import httpx
from pydantic import BaseModel, Field

from ...providers.bungee import BungeeProvider
from ...providers.coingecko import CoingeckoProvider
from ...providers.llm.base import LLMProvider, LLMMessage, LLMResponse
from ...tools.defillama import get_tvl_series
from ...tools.portfolio import get_portfolio
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from .styles import StyleManager, ResponseStyle


BRIDGE_KEYWORDS = (
    'bridge',
    'bridging',
    'move to',
    'send to',
    'transfer to',
    'port to',
)

BRIDGE_FOLLOWUP_KEYWORDS = (
    'quote',
    'retry',
    'try again',
    'again',
    'get the quote',
    'get quote',
    'do it',
    'go ahead',
    'please',
)

CHAIN_METADATA: Dict[int, Dict[str, Any]] = {
    1: {
        'name': 'Ethereum',
        'aliases': ['ethereum', 'eth', 'mainnet'],
        'native_symbol': 'ETH',
    },
    8453: {
        'name': 'Base',
        'aliases': ['base', 'base mainnet'],
        'native_symbol': 'ETH',
    },
    42161: {
        'name': 'Arbitrum',
        'aliases': ['arbitrum', 'arb'],
        'native_symbol': 'ETH',
    },
    10: {
        'name': 'Optimism',
        'aliases': ['optimism', 'op'],
        'native_symbol': 'ETH',
    },
    137: {
        'name': 'Polygon',
        'aliases': ['polygon', 'matic', 'matic pos'],
        'native_symbol': 'MATIC',
    },
}

CHAIN_ALIAS_TO_ID: Dict[str, int] = {
    alias: chain_id
    for chain_id, details in CHAIN_METADATA.items()
    for alias in details.get('aliases', [])
}

DEFAULT_CHAIN_NAME_TO_ID: Dict[str, int] = {
    details['name'].lower(): chain_id for chain_id, details in CHAIN_METADATA.items()
}

NATIVE_PLACEHOLDER = '0x0000000000000000000000000000000000000000'
USD_UNITS = {'usd', 'dollar', 'dollars', 'usdc', 'buck', 'bucks'}
ETH_UNITS = {'eth', 'weth'}
BRIDGE_SOURCE = {'name': 'Socket (Bungee)', 'url': 'https://socket.tech'}


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
        # Lightweight cache for frequently used market data
        self._eth_price_cache: Optional[Tuple[Dict[str, Any], datetime]] = None
        # Track in-progress bridge requests per conversation for follow-ups
        self._pending_bridge: Dict[str, Dict[str, Any]] = {}
        
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
            # Step 1: Handle style commands and detection
            style_response = await self._handle_style_processing(
                request.messages, 
                conversation_id
            )
            
            # If style command was processed, return style help response
            if style_response:
                return style_response
            
            # Step 2: Determine persona
            current_persona = await self._determine_persona(
                request.messages, 
                conversation_id, 
                persona_name
            )
            
            # Ensure portfolio context is present when wallet is available (proactive fetch once)
            await self._ensure_portfolio_context(conversation_id, request)

            # Step 3: Prepare context (now includes style modifiers)
            context_messages = await self._prepare_context(
                request, 
                conversation_id, 
                current_persona
            )
            
            # Step 4: Execute tools if needed
            tool_data = await self._execute_tools(request, conversation_id)
            # Add TVL data if applicable
            try:
                if self._needs_tvl_data(request):
                    params = self._extract_tvl_params(request)
                    ts, tvl = await get_tvl_series(protocol=params['protocol'], window=params['window'])
                    stats = self._compute_tvl_stats(ts, tvl)
                    tool_data['defillama_tvl'] = {
                        'protocol': params['protocol'],
                        'window': params['window'],
                        'timestamps': ts,
                        'tvl': tvl,
                        'stats': stats,
                    }
                    # Update episodic focus in conversation memory
                    if self.context_manager:
                        try:
                            await self.context_manager.set_active_focus(
                                conversation_id,
                                {
                                    'entity': params['protocol'],
                                    'protocol': params['protocol'],
                                    'metric': 'tvl',
                                    'window': params['window'],
                                    'chain': request.chain,
                                    'stats': stats,
                                    'source': 'defillama',
                                },
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed setting episodic focus: {str(e)}")
            except Exception as e:
                self.logger.error(f"DefiLlama TVL fetch error: {str(e)}")
            
            # Step 5: Generate LLM response
            llm_response = await self._generate_llm_response(
                context_messages,
                tool_data,
                current_persona,
                conversation_id
            )
            
            # Step 5: Format final response
            agent_response = await self._format_response(
                llm_response,
                tool_data,
                current_persona,
                conversation_id,
                start_time
            )
            
            # Step 6: Update conversation state
            await self._update_conversation_state(
                conversation_id,
                request,
                agent_response
            )
            
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

        tool_data = {}

        # Check if this looks like a portfolio request
        needs_portfolio = self._needs_portfolio_data(request)
        if needs_portfolio:
            try:
                # Extract wallet address
                wallet_address = self._extract_wallet_address(request)
                tool_data['_address'] = wallet_address
                
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

        # Bridge quote / execution prep when user explicitly asks to bridge
        bridge_quote = await self._maybe_fetch_bridge_quote(request, conversation_id)
        if bridge_quote:
            tool_data['bridge_quote'] = bridge_quote

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

    async def _generate_llm_response(
        self,
        messages: List[LLMMessage],
        tool_data: Dict[str, Any],
        persona_name: str,
        conversation_id: str
    ) -> LLMResponse:
        """Generate response from LLM with tool data context and style modifiers"""
        
        # Add style modifier to system context
        current_style = self._conversation_styles.get(conversation_id, self.style_manager.get_current_style())
        style_modifier = self.style_manager.get_style_modifier_prompt(current_style)
        
        if style_modifier:
            messages.append(LLMMessage(
                role="system",
                content=f"Response style guidance: {style_modifier}"
            ))
        
        # Add tool data to context if available
        if tool_data:
            tool_context = self._format_tool_data_for_llm(tool_data)
            messages.append(LLMMessage(
                role="system",
                content=f"Available data: {tool_context}"
            ))
            if 'defillama_tvl' in tool_data:
                messages.append(LLMMessage(
                    role="system",
                    content=(
                        "With the provided TVL time series, write a concise 7-day analysis with 2–4 bullets: "
                        "start vs end, absolute and percent change, min/max with dates, and overall trend. "
                        "Avoid disclaimers about missing charts; assume the user sees the chart panel."
                    )
                ))
            
        # Generate response from LLM
        response = await self.llm_provider.generate_response(
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        return response

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
                panel_id = panel.get('id', 'bungee_bridge_quote')
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

    async def _maybe_fetch_bridge_quote(self, request: ChatRequest, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Detect bridge intent, call the public Bungee API, and persist follow-up context."""

        if not request.messages:
            return None

        latest_message = request.messages[-1].content.strip()
        if not latest_message:
            return None

        message_lower = latest_message.lower()
        pending_state = self._pending_bridge.get(conversation_id)

        explicit_bridge = self._is_bridge_query(message_lower)
        followup = False
        if not explicit_bridge and pending_state:
            followup = self._is_bridge_followup(message_lower)

        if not explicit_bridge and not followup:
            return None

        context = pending_state.get('context', {}) if pending_state else {}
        current_context: Dict[str, Any] = dict(context)
        user_address = self._extract_wallet_address(request) or context.get('user_address')

        def finalize_result(
            status: str,
            *,
            message: Optional[str] = None,
            panel: Optional[Dict[str, Any]] = None,
            summary_reply: Optional[str] = None,
            summary_tool: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            context_updates: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            result: Dict[str, Any] = {'status': status}
            if message:
                result['message'] = message
            if panel:
                result['panel'] = panel
            if summary_reply:
                result['summary_reply'] = summary_reply
            if summary_tool:
                result['summary_tool'] = summary_tool
            if extra:
                result.update(extra)

            merged_context = {k: v for k, v in current_context.items() if v is not None}
            if context_updates:
                merged_context.update({k: v for k, v in context_updates.items() if v is not None})

            pending_entry = {
                'context': merged_context,
                'last_prompt': latest_message,
                'status': status,
                'last_result': result,
            }
            if panel:
                pending_entry['panel'] = panel
            if summary_reply:
                pending_entry['summary_reply'] = summary_reply
            if summary_tool:
                pending_entry['summary_tool'] = summary_tool
            self._pending_bridge[conversation_id] = pending_entry
            return result

        if not user_address:
            return finalize_result(
                'needs_address',
                message='I can prep a Bungee bridge, but I need the wallet address that will sign it.',
            )

        from_chain_id = current_context.get('from_chain_id')
        to_chain_id = current_context.get('to_chain_id')
        if explicit_bridge:
            chain_params = self._infer_bridge_params(message_lower, getattr(request, 'chain', None))
            from_chain_id = chain_params.get('from_chain_id') or from_chain_id
            to_chain_id = chain_params.get('to_chain_id') or to_chain_id
        if from_chain_id is None:
            from_chain_id = 1
        if to_chain_id is None:
            return finalize_result(
                'needs_chain',
                message='Which chain should I bridge to? Try “bridge … to Base/Arbitrum/Optimism/Polygon”.',
                context_updates={'user_address': user_address, 'from_chain_id': from_chain_id},
            )
        current_context.update({'user_address': user_address, 'from_chain_id': from_chain_id, 'to_chain_id': to_chain_id})

        amount_eth: Optional[Decimal] = None
        amount_usd: Optional[Decimal] = None
        price_info: Optional[Dict[str, Any]] = pending_state.get('price') if pending_state else None

        if explicit_bridge:
            amount_result = await self._resolve_bridge_amount(message_lower)
            if amount_result.get('status') != 'ok':
                status = amount_result.get('status', 'needs_amount')
                if status == 'price_unavailable':
                    msg = 'I could not fetch the current ETH price to convert the USD amount. Try again soon or specify the amount in ETH.'
                elif status == 'needs_amount':
                    msg = 'Tell me how much to bridge — e.g., “bridge 0.05 ETH” or “bridge $25 worth of ETH to Base”.'
                else:
                    msg = 'Unable to determine the bridge amount from that request.'
                return finalize_result(status, message=msg)
            amount_eth = amount_result['amount_eth']
            amount_usd = amount_result.get('amount_usd')
            price_info = amount_result.get('price')
        else:
            if current_context.get('amount_eth') is not None:
                try:
                    amount_eth = Decimal(str(current_context['amount_eth']))
                except (InvalidOperation, TypeError, ValueError):
                    amount_eth = None
            if current_context.get('amount_usd') is not None:
                try:
                    amount_usd = Decimal(str(current_context['amount_usd']))
                except (InvalidOperation, TypeError, ValueError):
                    amount_usd = None

        if amount_eth is None:
            return finalize_result(
                'needs_amount',
                message='Tell me how much to bridge — e.g., “bridge 0.05 ETH” or “bridge $25 worth of ETH to Base”.',
            )

        from_decimals = 18
        try:
            amount_wei_dec = (amount_eth * (Decimal(10) ** from_decimals)).quantize(Decimal('1'), rounding=ROUND_DOWN)
        except (InvalidOperation, ValueError):
            return finalize_result(
                'needs_amount',
                message='The bridge amount looks invalid. Try a slightly larger value.',
            )

        if amount_wei_dec <= 0:
            return finalize_result(
                'needs_amount',
                message='The bridge amount is too small to execute.',
            )

        amount_wei = str(int(amount_wei_dec))
        amount_eth_str = self._decimal_to_str(amount_eth)
        amount_usd_str = self._decimal_to_str(amount_usd) if amount_usd is not None else None
        current_context.update({'amount_eth': amount_eth_str, 'amount_usd': amount_usd_str, 'amount_wei': amount_wei})

        from_symbol = self._chain_native_symbol(from_chain_id)
        to_symbol = self._chain_native_symbol(to_chain_id)

        quote_params = {
            'originChainId': str(from_chain_id),
            'destinationChainId': str(to_chain_id),
            'userAddress': user_address,
            'receiverAddress': user_address,
            'inputToken': NATIVE_PLACEHOLDER,
            'outputToken': NATIVE_PLACEHOLDER,
            'inputAmount': amount_wei,
            'enableManual': 'true',
        }

        panel_payload: Dict[str, Any] = {
            'from_chain_id': from_chain_id,
            'from_chain': self._chain_name(from_chain_id),
            'to_chain_id': to_chain_id,
            'to_chain': self._chain_name(to_chain_id),
            'amounts': {
                'requested_eth': amount_eth_str,
                'requested_usd': amount_usd_str,
                'input_amount_wei': amount_wei,
            },
            'price_reference': {
                'usd': str(price_info['price']) if price_info and price_info.get('price') else None,
                'source': price_info.get('source') if price_info else None,
            },
            'status': 'pending',
        }

        pending_entry = {
            'context': {k: v for k, v in current_context.items() if v is not None},
            'quote_params': copy.deepcopy(quote_params),
            'price': price_info,
            'last_prompt': latest_message,
        }
        self._pending_bridge[conversation_id] = pending_entry

        provider = BungeeProvider()

        try:
            quote_response = await provider.quote(quote_params)
        except httpx.HTTPStatusError as exc:  # type: ignore
            status_code = exc.response.status_code
            message = (
                'Bungee rejected the request (401). Add a valid BUNGEE_API_KEY to enable automated bridges.'
                if status_code == 401
                else f'Bungee quote failed with status {status_code}.'
            )
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            summary = (
                f"Bridge {amount_eth_str} {from_symbol} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}."
                f"\n⚠️ {message}"
            )
            panel = {
                'id': 'bungee_bridge_quote',
                'kind': 'card',
                'title': f"Bungee Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error', 'http_status': status_code},
            }
            return finalize_result(
                'error',
                message=message,
                panel=panel,
                summary_reply=summary,
                summary_tool=summary,
                extra={'detail': str(exc)},
            )
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Bungee quote failed: {exc}")
            message = 'I could not reach Bungee to fetch a bridge route. Please try again shortly.'
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            summary = (
                f"Bridge {amount_eth_str} {from_symbol} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}."
                f"\n⚠️ {message}"
            )
            panel = {
                'id': 'bungee_bridge_quote',
                'kind': 'card',
                'title': f"Bungee Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result(
                'error',
                message=message,
                panel=panel,
                summary_reply=summary,
                summary_tool=summary,
                extra={'detail': str(exc)},
            )

        quote_result = quote_response.get('result') if isinstance(quote_response, dict) else {}
        if not quote_result:
            message = 'Bungee did not return any route data for that request.'
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            panel = {
                'id': 'bungee_bridge_quote',
                'kind': 'card',
                'title': f"Bungee Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result('error', message=message, panel=panel)

        manual_routes = quote_result.get('manualRoutes') or []
        default_route = quote_result.get('autoRoute') or {}

        primary_route: Dict[str, Any] = {}
        route_source = 'manual'
        if manual_routes:
            primary_route = manual_routes[0]
        elif default_route:
            primary_route = default_route
            route_source = 'default'
        else:
            message = 'Bungee did not return a usable bridge route. Try adjusting the amount or tokens.'
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            panel = {
                'id': 'bungee_bridge_quote',
                'kind': 'card',
                'title': f"Bungee Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result('error', message=message, panel=panel)

        quote_id = primary_route.get('quoteId') or quote_result.get('quoteId')
        route_request_hash = primary_route.get('requestHash') or primary_route.get('routeId')

        panel_payload.update({
            'quote_id': quote_id,
            'route_request_hash': route_request_hash,
            'route_source': route_source,
            'input': quote_result.get('input'),
            'output': primary_route.get('output'),
            'approval_data': primary_route.get('approvalData'),
            'sign_typed_data': primary_route.get('signTypedData'),
            'route_details': primary_route,
        })

        build_payload = None
        tx_result: Optional[Dict[str, Any]] = None
        build_error: Optional[str] = None

        if quote_id:
            build_params = {
                'quoteId': quote_id,
            }
            try:
                build_response = await provider.build_tx(build_params)
                build_payload = build_response
                tx_result = build_response.get('result') if isinstance(build_response, dict) else None
                panel_payload['tx_ready'] = bool(tx_result)
            except httpx.HTTPStatusError as exc:  # type: ignore
                build_error = f'Bungee build-tx failed with status {exc.response.status_code}: {exc.response.text}'
                panel_payload.setdefault('issues', []).append(build_error)
                panel_payload['tx_ready'] = False
            except Exception as exc:  # pragma: no cover
                build_error = f'Bungee build-tx error: {exc}'
                panel_payload.setdefault('issues', []).append(build_error)
                panel_payload['tx_ready'] = False
        else:
            build_error = 'Quote did not provide a quoteId for manual build.'
            panel_payload.setdefault('issues', []).append(build_error)
            panel_payload['tx_ready'] = False

        input_token = quote_result.get('input', {}).get('token', {})
        output_token = primary_route.get('output', {}).get('token', {})
        input_amount_raw = quote_result.get('input', {}).get('amount') or amount_wei
        output_amount_raw = primary_route.get('output', {}).get('amount')
        input_decimals = input_token.get('decimals', from_decimals)
        output_decimals = output_token.get('decimals', 18)

        try:
            routed_from_amount = Decimal(str(input_amount_raw)) / (Decimal(10) ** int(input_decimals))
        except (InvalidOperation, TypeError, ValueError):
            routed_from_amount = amount_eth

        try:
            routed_to_amount = Decimal(str(output_amount_raw)) / (Decimal(10) ** int(output_decimals)) if output_amount_raw else None
        except (InvalidOperation, TypeError, ValueError):
            routed_to_amount = None

        gas_usd = primary_route.get('gasFee')
        output_usd = primary_route.get('output', {}).get('valueInUsd')
        eta_seconds = primary_route.get('estimatedTime')
        eta_readable = self._format_eta_minutes(eta_seconds)
        bridge_name = (primary_route.get('routeDetails') or {}).get('name')
        approval_data = primary_route.get('approvalData') or {}
        needs_approval = bool(approval_data)

        panel_payload['status'] = 'ok' if tx_result else 'quote_only'
        panel_payload['usd_estimates'] = {
            'output': output_usd,
            'gas': gas_usd,
        }
        if tx_result:
            panel_payload['tx'] = tx_result
        if build_payload:
            panel_payload['build'] = build_payload

        summary_lines = [
            f"✅ {self._decimal_to_str(routed_from_amount)} {input_token.get('symbol', from_symbol)} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}"
        ]
        if routed_to_amount is not None:
            summary_lines.append(
                f"Estimated arrival: {self._decimal_to_str(routed_to_amount)} {output_token.get('symbol', to_symbol)}"
            )
        if output_usd is not None:
            try:
                summary_lines[-1] += f" (~${float(output_usd):.2f})"
            except (ValueError, TypeError):
                pass
        if gas_usd is not None:
            try:
                summary_lines.append(f"Bridge cost approx ${float(gas_usd):.2f} in fees")
            except (ValueError, TypeError):
                pass
        if eta_readable:
            summary_lines.append(f"ETA ≈ {eta_readable}")
        if bridge_name:
            summary_lines.append(f"Route: {bridge_name}")
        if needs_approval:
            approval_token_address = approval_data.get('tokenAddress')
            approval_amount = approval_data.get('amount') or approval_data.get('minimumApprovalAmount')
            approval_line = 'Approve the input token before bridging.'
            if approval_amount:
                try:
                    approval_amt_float = Decimal(str(approval_amount)) / (Decimal(10) ** int(input_decimals))
                    approval_line = f"Approve {self._decimal_to_str(approval_amt_float)} {input_token.get('symbol', from_symbol)}"
                except (InvalidOperation, TypeError, ValueError):
                    approval_line = f"Approve {approval_amount} units"
            if approval_token_address:
                approval_line += f" at {approval_token_address}"
            summary_lines.append(approval_line)
        if not tx_result:
            if build_error:
                summary_lines.append(f"⚠️ Manual build unavailable: {build_error}")
            else:
                summary_lines.append('Review the quote details in the bridge panel to proceed manually.')

        summary_reply = "\n".join(summary_lines)
        summary_tool = (
            f"Bridge plan: {self._decimal_to_str(routed_from_amount)} {input_token.get('symbol', from_symbol)} → "
            f"{output_token.get('symbol', to_symbol)} on {self._chain_name(to_chain_id)}"
        )

        panel_metadata = {
            'status': panel_payload['status'],
            'route_source': route_source,
            'quote_id': quote_id,
            'route_request_hash': route_request_hash,
            'needs_approval': needs_approval,
        }

        panel = {
            'id': 'bungee_bridge_quote',
            'kind': 'card',
            'title': f"Bungee Bridge: {input_token.get('symbol', from_symbol)} → {output_token.get('symbol', to_symbol)}",
            'payload': panel_payload,
            'sources': [BRIDGE_SOURCE],
            'metadata': panel_metadata,
        }

        pending_entry.update({
            'status': panel_payload['status'],
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
            'last_result': {
                'status': panel_payload['status'],
                'panel': panel,
                'summary_reply': summary_reply,
                'summary_tool': summary_tool,
            },
            'quote_id': quote_id,
            'route_request_hash': route_request_hash,
        })
        self._pending_bridge[conversation_id] = pending_entry

        result_payload: Dict[str, Any] = {
            'status': panel_payload['status'],
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
        }
        if build_error:
            result_payload['message'] = build_error
        if tx_result:
            result_payload['tx'] = tx_result

        return result_payload
    def _is_bridge_query(self, message: str) -> bool:
        if re.search(r'\bbridge\b', message):
            return True
        return any(keyword in message for keyword in BRIDGE_KEYWORDS)

    def _is_bridge_followup(self, message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False
        if any(keyword in message for keyword in BRIDGE_FOLLOWUP_KEYWORDS):
            return True
        if stripped in {'yes', 'y', 'yep', 'sure', 'ok', 'okay', 'please', 'please do'}:
            return True
        return False

    def _infer_bridge_params(self, message: str, default_chain: Optional[str]) -> Dict[str, Optional[int]]:
        msg = message.lower()
        to_chain_id = self._detect_chain(msg, ['to', 'onto', 'into', 'towards'])
        if to_chain_id is None:
            to_chain_id = self._detect_chain_arrow(msg)

        default_chain_id = None
        if default_chain:
            lower = default_chain.lower()
            default_chain_id = CHAIN_ALIAS_TO_ID.get(lower) or DEFAULT_CHAIN_NAME_TO_ID.get(lower)
        if default_chain_id is None:
            default_chain_id = 1

        from_chain_id = self._detect_chain(msg, ['from', 'off', 'out of'])
        if from_chain_id is None:
            candidate = self._detect_chain(msg, ['on'])
            if candidate and candidate != to_chain_id:
                from_chain_id = candidate
        if from_chain_id is None:
            from_chain_id = default_chain_id

        if to_chain_id is None:
            for alias, chain_id in CHAIN_ALIAS_TO_ID.items():
                if alias in {'eth', 'ethereum', 'mainnet'}:
                    continue
                if alias in msg and chain_id != from_chain_id:
                    to_chain_id = chain_id
                    break

        return {'from_chain_id': from_chain_id, 'to_chain_id': to_chain_id}

    def _detect_chain(self, message: str, keywords: List[str]) -> Optional[int]:
        for alias, chain_id in CHAIN_ALIAS_TO_ID.items():
            escaped = re.escape(alias)
            for keyword in keywords:
                pattern = rf'\b{keyword}\s+{escaped}\b'
                if re.search(pattern, message):
                    return chain_id
        return None

    def _detect_chain_arrow(self, message: str) -> Optional[int]:
        for alias, chain_id in CHAIN_ALIAS_TO_ID.items():
            needle = f'->{alias}'
            if needle in message or f'-> {alias}' in message:
                return chain_id
        return None

    async def _resolve_bridge_amount(self, message: str) -> Dict[str, Any]:
        parsed = self._parse_bridge_amount(message)
        unit = parsed.get('unit')
        amount_eth = parsed.get('amount_eth')
        amount_usd = parsed.get('amount_usd')

        if unit is None or (amount_eth is None and amount_usd is None):
            return {'status': 'needs_amount'}

        if unit == 'usd':
            price_info = await self._get_eth_price_usd()
            if not price_info or not price_info.get('price'):
                return {'status': 'price_unavailable'}
            price = price_info['price']
            try:
                amount_eth = (amount_usd / price).quantize(Decimal('1e-18'), rounding=ROUND_DOWN)
            except (InvalidOperation, TypeError):
                return {'status': 'needs_amount'}
            if amount_eth <= 0:
                return {'status': 'needs_amount'}
            return {
                'status': 'ok',
                'amount_eth': amount_eth,
                'amount_usd': amount_usd,
                'price': price_info,
            }

        # amount provided in ETH (or WETH)
        if amount_eth is None or amount_eth <= 0:
            return {'status': 'needs_amount'}

        price_info = await self._get_eth_price_usd()
        usd_value = None
        if price_info and price_info.get('price'):
            try:
                usd_value = (amount_eth * price_info['price']).quantize(Decimal('1e-2'), rounding=ROUND_DOWN)
            except (InvalidOperation, TypeError):
                usd_value = None

        return {
            'status': 'ok',
            'amount_eth': amount_eth,
            'amount_usd': usd_value,
            'price': price_info,
        }

    def _parse_bridge_amount(self, message: str) -> Dict[str, Optional[Decimal]]:
        pattern = r'(\$)?\s*(\d+(?:\.\d+)?)\s*(usd|usdc|dollar|dollars|buck|bucks|eth|weth)?'
        for match in re.finditer(pattern, message):
            symbol, number, unit = match.groups()
            try:
                value = Decimal(number)
            except (InvalidOperation, TypeError):
                continue
            if value <= 0:
                continue
            unit_lower = unit.lower() if unit else None
            if symbol or (unit_lower and unit_lower in USD_UNITS):
                return {'amount_usd': value, 'amount_eth': None, 'unit': 'usd'}
            if unit_lower and unit_lower in ETH_UNITS:
                return {'amount_eth': value, 'amount_usd': None, 'unit': 'eth'}

        return {'amount_eth': None, 'amount_usd': None, 'unit': None}

    async def _get_eth_price_usd(self) -> Optional[Dict[str, Any]]:
        try:
            if self._eth_price_cache:
                cached_value, cached_time = self._eth_price_cache
                if (datetime.now() - cached_time).total_seconds() < 60:
                    return cached_value
        except Exception:
            pass

        provider = CoingeckoProvider()
        try:
            if not await provider.ready():
                return None
            data = await provider.get_eth_price()
        except Exception as exc:  # pragma: no cover
            self.logger.debug(f"ETH price fetch failed: {exc}")
            return None

        price = data.get('price_usd')
        if price is None:
            return None

        price_info = {
            'price': Decimal(str(price)),
            'source': data.get('_source'),
        }
        self._eth_price_cache = (price_info, datetime.now())
        return price_info

    def _decimal_to_str(self, value: Decimal, places: int = 6) -> str:
        precision = max(0, min(places, 18))
        quant = Decimal('1') if precision == 0 else Decimal(1).scaleb(-precision)
        try:
            quantized = value.quantize(quant, rounding=ROUND_DOWN)
        except (InvalidOperation, TypeError):
            quantized = value
        formatted = format(quantized.normalize(), 'f')
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted or '0'

    def _chain_name(self, chain_id: int) -> str:
        meta = CHAIN_METADATA.get(chain_id)
        return meta.get('name') if meta else f'Chain {chain_id}'

    def _chain_native_symbol(self, chain_id: int) -> str:
        meta = CHAIN_METADATA.get(chain_id)
        return meta.get('native_symbol', 'ETH') if meta else 'ETH'

    def _format_eta_minutes(self, seconds: Optional[Any]) -> Optional[str]:
        try:
            sec = float(seconds)
        except (TypeError, ValueError):
            return None
        if sec <= 0:
            return None
        if sec < 90:
            return f"{sec:.0f} sec"
        minutes = sec / 60.0
        return f"{minutes:.1f} min"

    def _format_tool_data_for_llm(self, tool_data: Dict[str, Any]) -> str:
        """Format tool data into a readable context for the LLM"""
        
        formatted_parts = []
        
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
