"""
Core Agent System

This module contains the main Agent class that orchestrates LLM interactions,
persona management, and tool integration for intelligent portfolio analysis.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field

from ...config import settings
from ...providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse,
)
from ...tools.portfolio import get_portfolio
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from .events import SteeringMessage, ToolStarted, ToolCompleted, LLMStarted, LLMCompleted, LoopFinished
from .panels import PANEL_BUILDERS
from .styles import StyleManager, ResponseStyle
from .tools import ToolContext, ToolRegistry, ToolExecutor


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
            # 1. Style intercept (early return)
            style_response = await self._handle_style_processing(
                request.messages, conversation_id
            )
            if style_response:
                return style_response

            # 2. Persona
            persona = await self._determine_persona(
                request.messages, conversation_id, persona_name
            )

            # 3. Context
            context_messages = await self._prepare_context(
                request, conversation_id, persona
            )

            # 4. ReAct loop
            llm_response, tool_data = await self._run_react_loop(
                context_messages, request, conversation_id
            )

            # 5. Format
            final_response = await self._format_response(
                llm_response, tool_data, persona, conversation_id, start_time
            )

            # 6. Persist
            await self._update_conversation_state(
                conversation_id, request, final_response
            )

            return final_response
            
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
                content=(
                    f"CRITICAL - USER'S WALLET ADDRESS: {wallet_address}\n"
                    f"Chain context: {request.chain or 'ethereum'}\n"
                    f"This wallet is ALREADY CONNECTED and authenticated to this session.\n"
                    f"RULES:\n"
                    f"1. ALWAYS use {wallet_address} for portfolio, swap, bridge, and wallet operations\n"
                    f"2. NEVER ask the user for their wallet address - you have it above\n"
                    f"3. If asked 'what is my wallet address', reply with: {wallet_address}\n"
                    f"4. NEVER make up or hallucinate a different wallet address"
                )
            ))

        # Add conversation history if context manager available
        if self.context_manager:
            history = await self.context_manager.get_context(conversation_id)
            # Context manager should return formatted context string
            if history:
                context_messages.append(LLMMessage(role="system", content=f"Conversation context: {history}"))

        # Add explicit tool-use instruction
        # This ensures the LLM knows it MUST call tools for real-time data
        context_messages.append(LLMMessage(
            role="system",
            content=(
                "TOOL USAGE REQUIREMENT:\n"
                "You have access to tools for fetching real-time blockchain data. "
                "When the user asks about swaps, bridges, portfolio, prices, or any blockchain data:\n"
                "1. You MUST call the appropriate tool to get real data\n"
                "2. NEVER make up prices, quotes, fees, or amounts - always use tool results\n"
                "3. For cross-chain transfers (bridging): use get_bridge_quote\n"
                "4. For same-chain swaps on EVM: use get_swap_quote\n"
                "5. For Solana swaps: use get_solana_swap_quote\n"
                "6. For portfolio/balance queries: use get_portfolio\n"
                "If a tool call fails, report the error - do not make up data."
            )
        ))

        # Convert chat messages to LLM format
        for msg in request.messages:
            context_messages.append(LLMMessage(
                role=msg.role,
                content=msg.content
            ))

        return context_messages

    async def _run_react_loop(
        self,
        messages: List[LLMMessage],
        request: ChatRequest,
        conversation_id: str,
        event_queue: Optional[asyncio.Queue] = None,
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        """
        Run the ReAct (Reasoning + Acting) loop for LLM-driven tool selection.

        The LLM decides which tools to call based on semantic understanding of the
        user's intent. This replaces the keyword-based tool detection approach.

        Returns:
            Tuple of (final_llm_response, tool_data_dict)
        """
        tool_data: Dict[str, Any] = {}
        loop_start = time.monotonic()
        total_tool_calls = 0

        def _emit(event):
            if event_queue is not None:
                event_queue.put_nowait(event)

        # Build tool context for automatic wallet injection
        wallet_address = self._extract_wallet_address(request)
        tool_data['_address'] = wallet_address
        tool_ctx = ToolContext(
            wallet_address=wallet_address,
            chain=request.chain,
        )

        # Check if provider supports tool calling
        if not self.llm_provider.supports_tools:
            self.logger.debug("LLM provider does not support tools, running without tool calling")
            response = await self.llm_provider.generate_response(
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
            )
            quote_intent = self._detect_quote_intent(request)
            if quote_intent:
                tool_data["_quote_intent"] = quote_intent
            return response, tool_data

        # Get tool definitions
        tools = self.tool_registry.get_definitions()
        tool_names = [t.name for t in tools]
        self.logger.info(f"ReAct loop starting with {len(tools)} tools: {tool_names}")

        # Start the ReAct loop
        current_messages = list(messages)
        final_response: Optional[LLMResponse] = None
        actual_iterations = 0

        for iteration in range(self._max_tool_iterations):
            actual_iterations = iteration + 1
            self.logger.info(f"ReAct loop iteration {iteration + 1}/{self._max_tool_iterations}")

            # Check for steering messages injected by the caller
            if event_queue is not None:
                while not event_queue.empty():
                    msg = event_queue.get_nowait()
                    if isinstance(msg, SteeringMessage) and msg.content:
                        current_messages.append(LLMMessage(role="user", content=msg.content))
                        self.logger.info("Steering message injected: %s", msg.content[:100])

            # Get LLM response with tool definitions
            _emit(LLMStarted(iteration=iteration, tool_count=len(tools)))
            llm_t0 = time.monotonic()
            response = await self.llm_provider.generate_response(
                messages=current_messages,
                tools=tools,
                max_tokens=4000,
                temperature=0.7,
            )
            _emit(LLMCompleted(
                iteration=iteration,
                tool_calls_requested=len(response.tool_calls) if response.tool_calls else 0,
                tokens_used=getattr(response, 'tokens_used', None),
            ))

            # If no tool calls, we're done
            if not response.tool_calls:
                self.logger.info(
                    "LLM finished without tool calls. Content preview: %s",
                    (response.content or "")[:200]
                )
                final_response = response
                break

            tool_call_names = [tc.name for tc in response.tool_calls]
            self.logger.info(f"LLM requested {len(response.tool_calls)} tool call(s): {tool_call_names}")

            # Emit ToolStarted for each tool call
            for tc in response.tool_calls:
                _emit(ToolStarted(tool_name=tc.name, arguments=tc.arguments, iteration=iteration))

            # Execute all tool calls in parallel (ToolContext auto-injects wallet/chain)
            exec_t0 = time.monotonic()
            results = await self.tool_executor.execute_parallel(response.tool_calls, context=tool_ctx)
            exec_elapsed_ms = (time.monotonic() - exec_t0) * 1000

            # Emit ToolCompleted for each tool call
            for tc, result in zip(response.tool_calls, results):
                _emit(ToolCompleted(
                    tool_name=tc.name,
                    success=result.error is None,
                    latency_ms=exec_elapsed_ms,
                    iteration=iteration,
                ))
            total_tool_calls += len(response.tool_calls)

            # Store results in tool_data and build messages for next iteration
            for tc, result in zip(response.tool_calls, results):
                # Map tool results to expected format in tool_data
                result_data = result.result if result.result else {'error': result.error}
                tool_data[tc.name] = {
                    'call': tc.arguments,
                    'result': result_data,
                }

                # Persist portfolio data to conversation context for cross-turn memory
                if tc.name == "get_portfolio" and isinstance(result_data, dict) and result_data.get('success'):
                    portfolio_data = result_data.get('data')
                    if portfolio_data and self.context_manager:
                        try:
                            await self.context_manager.integrate_portfolio_data(conversation_id, portfolio_data)
                        except Exception as e:
                            self.logger.debug(f"Failed to persist portfolio context: {e}")

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

        _emit(LoopFinished(
            total_iterations=actual_iterations,
            total_tool_calls=total_tool_calls,
            total_latency_ms=(time.monotonic() - loop_start) * 1000,
        ))

        quote_intent = self._detect_quote_intent(request)
        if quote_intent:
            tool_data["_quote_intent"] = quote_intent

        return final_response, tool_data

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

        # Apply guardrails before rendering content/panels.
        self._apply_quote_guard(llm_response, tool_data)

        # Build panels from tool results using panel builders
        panels: Dict[str, Any] = {}
        sources: List[Dict[str, Any]] = []
        reply_text = llm_response.content or ""
        has_portfolio_error = False

        for tool_name, entry in tool_data.items():
            if tool_name.startswith('_'):
                continue
            if not isinstance(entry, dict):
                continue
            result_data = entry.get('result')
            if not isinstance(result_data, dict):
                continue

            # Track portfolio errors for the fallback message
            if tool_name == 'get_portfolio' and not result_data.get('success'):
                has_portfolio_error = True

            builder = PANEL_BUILDERS.get(tool_name)
            if builder is None:
                continue

            panel_result = builder(result_data)
            if panel_result is None:
                continue

            panels.update(panel_result.panels)
            sources.extend(panel_result.sources)
            if panel_result.reply_text:
                reply_text = panel_result.reply_text

        # If portfolio fetch failed, provide a safe helpful prompt
        if has_portfolio_error and 'portfolio_overview' not in panels:
            addr = tool_data.get('_address')
            if addr:
                reply_text = (
                    f"I can analyze your holdings at {addr}, but I couldn't load the latest portfolio data just now. "
                    f"Would you like me to try again, or specify a different address or chain?"
                )
            else:
                reply_text = (
                    "I don't see a wallet address yet. Share an address (e.g., 0x…) "
                    "or connect your wallet and I'll analyze your portfolio without guessing."
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

    def _detect_quote_intent(self, request: ChatRequest) -> Optional[str]:
        """Heuristic detection for swap/bridge quote intent."""
        if not request.messages:
            return None
        message = request.messages[-1].content.lower().strip()
        if not message:
            return None

        # Strategy / automation context — the user is creating a recurring
        # strategy, not requesting an immediate swap/bridge quote.
        strategy_markers = (
            "strategy",
            "dca",
            "dollar cost",
            "dollar-cost",
            "recurring",
            "automate",
            "automation",
            "schedule",
            "every hour",
            "every day",
            "every week",
            "daily",
            "weekly",
            "hourly",
            "create a",
            "set up a",
            "setup a",
        )
        if any(marker in message for marker in strategy_markers):
            return None

        info_markers = (
            "what is",
            "what are",
            "what's",
            "whats",
            "how does",
            "how do",
            "how to",
            "explain",
            "guide",
            "tutorial",
            "tell me about",
        )
        action_markers = (
            "swap",
            "bridge",
            "transfer",
            "send",
            "convert",
            "exchange",
            "trade",
        )
        quote_markers = (
            "quote",
            "rate",
            "how much",
            "estimate",
            "estimated",
            "fees",
            "fee",
        )
        bridge_markers = (
            "bridge",
            "cross-chain",
            "cross chain",
            "mainnet",
            "l1",
            "l2",
        )
        swap_markers = (
            "swap",
            "convert",
            "exchange",
            "trade",
        )
        transfer_markers = (
            "transfer",
            "send",
        )
        # If the message contains a 0x address, "send/transfer" likely means a
        # direct token transfer, not a bridge/swap quote.
        direct_transfer_markers = ("0x",)

        if any(marker in message for marker in info_markers) and any(marker in message for marker in action_markers):
            return None

        if not (any(marker in message for marker in quote_markers) or any(marker in message for marker in action_markers)):
            return None

        # Direct transfer to an address — not a quote intent
        has_transfer = any(marker in message for marker in transfer_markers)
        has_address = any(marker in message for marker in direct_transfer_markers)
        if has_transfer and has_address and not any(marker in message for marker in bridge_markers) and not any(marker in message for marker in swap_markers):
            return None

        if any(marker in message for marker in bridge_markers) or has_transfer:
            return "bridge"
        if any(marker in message for marker in swap_markers):
            return "swap"

        return "swap"

    def _build_missing_quote_reply(self, quote_kind: str) -> str:
        """Standardized message when a quote tool was not called."""
        action = "bridge" if quote_kind == "bridge" else "swap"
        return (
            f"I don't have a live {action} quote yet because the quote tool wasn't run. "
            f"Tell me the amount and the source/destination chains (or the chain for a same-chain swap), "
            f"and I'll fetch a real-time {action} quote."
        )

    def _resolve_quote_tool_state(self, quote_kind: str, tool_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if a relevant quote tool ran successfully."""
        legacy_key = "bridge_quote" if quote_kind == "bridge" else "swap_quote"
        if tool_data.get(legacy_key):
            return True, None

        tool_names = []
        if quote_kind == "bridge":
            tool_names = ["get_bridge_quote"]
        else:
            tool_names = ["get_swap_quote", "get_solana_swap_quote"]

        for tool_name in tool_names:
            entry = tool_data.get(tool_name)
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if isinstance(result, dict):
                if result.get("success") is True:
                    return True, None
                if result.get("success") is False:
                    return False, result.get("error") or "Quote tool failed"

        return False, None

    def _apply_quote_guard(self, llm_response: LLMResponse, tool_data: Dict[str, Any]) -> bool:
        """Override the response if a quote was requested but no tool ran."""
        if tool_data.get("_quote_guard_applied"):
            return False

        quote_kind = tool_data.get("_quote_intent")
        if not quote_kind:
            return False

        # If execute_transfer ran, this is a transfer — not a missing quote.
        if tool_data.get("execute_transfer"):
            return False

        # If a strategy/DCA tool ran, the user is setting up automation.
        if tool_data.get("create_strategy") or tool_data.get("create_dca_strategy"):
            return False

        has_quote, error = self._resolve_quote_tool_state(quote_kind, tool_data)
        if has_quote:
            return False

        if error:
            reply = (
                f"I couldn't fetch a live {quote_kind} quote yet: {error}. "
                f"Want me to try again or adjust the details?"
            )
            reason = "quote_tool_error"
        else:
            reply = self._build_missing_quote_reply(quote_kind)
            reason = "missing_quote_tool"

        llm_response.content = reply
        tool_data["_quote_guard_applied"] = True
        tool_data["_quote_guard_reason"] = reason
        return True

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

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt when no persona manager is available"""

        return """You are a knowledgeable and friendly crypto portfolio assistant.

        Scope:
        - Focus strictly on crypto portfolios, tokens, protocols, and wallet analytics.
        - Treat ambiguous requests (e.g., "rotate", "performance") as crypto portfolio topics by default.

        CRITICAL - TOOL USAGE:
        You have access to tools for fetching real-time data. You MUST use these tools instead of making up data:
        - get_portfolio: Fetch wallet holdings and balances
        - get_bridge_quote: Get quotes for cross-chain transfers (bridging tokens between different blockchains)
        - get_swap_quote: Get quotes for same-chain token swaps (EVM chains)
        - get_solana_swap_quote: Get quotes for Solana token swaps
        - get_token_chart: Fetch price history and charts
        - get_trending_tokens: Get currently trending tokens

        NEVER make up or hallucinate:
        - Token prices, amounts, or fees
        - Bridge quotes or swap quotes
        - Portfolio holdings or balances
        - Transaction data

        If the user asks about swapping, bridging, or transferring tokens:
        1. ALWAYS call the appropriate tool (get_bridge_quote for cross-chain, get_swap_quote for same-chain)
        2. Use the wallet address from context
        3. Only respond with real data from the tool result

        Guardrails:
        - Never invent or assume specific holdings.
        - If data appears inconsistent, briefly acknowledge uncertainty and ask a targeted clarification.
        - IMPORTANT: If a wallet address is provided in the context, ALWAYS use it. Never ask the user for their wallet address if one is already connected.

        Style:
        - Be concise, actionable, and concrete.
        - Prefer bullet points for insights and recommendations.

        When portfolio data is available, prioritize:
        - Allocation breakdown, concentration, diversification
        - Recent performance, notable movers, risk flags
        - Clear next steps (e.g., rebalance ideas, questions to confirm preferences)"""

    def _extract_wallet_address(self, request: ChatRequest) -> Optional[str]:
        """Extract wallet address from request."""
        if request.address:
            return request.address
        if request.messages:
            last_message = request.messages[-1].content
            pattern = r'0x[a-fA-F0-9]{40}'
            matches = re.findall(pattern, last_message)
            if matches:
                return matches[0]
        return None

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

