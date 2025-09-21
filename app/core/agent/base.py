"""
Core Agent System

This module contains the main Agent class that orchestrates LLM interactions,
persona management, and tool integration for intelligent portfolio analysis.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import uuid
from statistics import mean, pstdev

from ...providers.llm.base import LLMProvider, LLMMessage, LLMResponse
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from ...tools.portfolio import get_portfolio
from ...tools.defillama import get_tvl_series
from .styles import StyleManager, ResponseStyle


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
            tool_data = await self._execute_tools(request)
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

    async def _execute_tools(self, request: ChatRequest) -> Dict[str, Any]:
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
            panels['portfolio'] = portfolio_info['data']
            sources.extend(portfolio_info['sources'])

        # Add DefiLlama TVL panel if available
        if 'defillama_tvl' in tool_data:
            tvl_info = tool_data['defillama_tvl']
            ts = tvl_info.get('timestamps', [])
            tvl = tvl_info.get('tvl', [])
            protocol = tvl_info.get('protocol', 'uniswap')
            window = tvl_info.get('window', '7d')
            # Panel with normalized shape for frontend transformer
            panels['uniswap_tvl_chart'] = {
                'kind': 'chart',
                'title': f'{protocol.title()} TVL ({window})',
                'payload': {
                    'timestamps': ts,
                    'tvl': tvl,
                    'unit': 'USD',
                    'source': {
                        'name': 'DefiLlama',
                        'url': f'https://defillama.com/protocol/{protocol}'
                    },
                },
            }
            # Attribution
            sources.append({'name': 'DefiLlama', 'url': 'https://defillama.com'})
            # Deterministic short analysis to avoid LLM disclaimers
            stats = tvl_info.get('stats') or {}
            summary = self._compose_tvl_reply(protocol, window, stats)
            if summary:
                reply_text = summary

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
            import re
            pattern = r'0x[a-fA-F0-9]{40}'
            matches = re.findall(pattern, last_message)
            if matches:
                return matches[0]
                
        return None

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
