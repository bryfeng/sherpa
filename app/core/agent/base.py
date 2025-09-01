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

from ...providers.llm.base import LLMProvider, LLMMessage, LLMResponse
from ...types.requests import ChatRequest, ChatMessage
from ...types.responses import ChatResponse
from ...tools.portfolio import get_portfolio


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
        logger: Optional[logging.Logger] = None
    ):
        self.llm_provider = llm_provider
        self.persona_manager = persona_manager
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Agent state
        self._active_conversations: Dict[str, Dict] = {}
        
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
            # Step 1: Determine persona
            current_persona = await self._determine_persona(
                request.messages, 
                conversation_id, 
                persona_name
            )
            
            # Step 2: Prepare context
            context_messages = await self._prepare_context(
                request, 
                conversation_id, 
                current_persona
            )
            
            # Step 3: Execute tools if needed
            tool_data = await self._execute_tools(request)
            
            # Step 4: Generate LLM response
            llm_response = await self._generate_llm_response(
                context_messages,
                tool_data,
                current_persona
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
        if self._needs_portfolio_data(request):
            try:
                # Extract wallet address
                wallet_address = self._extract_wallet_address(request)
                
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
                        
            except Exception as e:
                self.logger.error(f"Tool execution error: {str(e)}")
                tool_data['portfolio_error'] = {
                    'error': str(e),
                    'warnings': []
                }
                
        return tool_data

    async def _generate_llm_response(
        self,
        messages: List[LLMMessage],
        tool_data: Dict[str, Any],
        persona_name: str
    ) -> LLMResponse:
        """Generate response from LLM with tool data context"""
        
        # Add tool data to context if available
        if tool_data:
            tool_context = self._format_tool_data_for_llm(tool_data)
            messages.append(LLMMessage(
                role="system",
                content=f"Available data: {tool_context}"
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
        
        if 'portfolio' in tool_data:
            portfolio_info = tool_data['portfolio']
            panels['portfolio'] = portfolio_info['data']
            sources.extend(portfolio_info['sources'])
            
        return AgentResponse(
            reply=llm_response.content,
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
            "what's in", "whats in", "analyze", "show me"
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
            
        return " | ".join(formatted_parts) if formatted_parts else "No additional data available"

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt when no persona manager is available"""
        
        return """You are a knowledgeable and friendly crypto portfolio assistant. 
        
        Your role is to help users understand and analyze their cryptocurrency portfolios. 
        You should:
        - Provide clear, helpful analysis of wallet data
        - Explain crypto concepts in accessible terms
        - Offer insights about portfolio composition and trends
        - Be encouraging and supportive
        - Maintain a conversational, approachable tone
        
        When portfolio data is available, focus on providing meaningful insights about the holdings, their values, and portfolio composition."""

    def to_chat_response(self, agent_response: AgentResponse) -> ChatResponse:
        """Convert AgentResponse to ChatResponse for API compatibility"""
        
        return ChatResponse(
            reply=agent_response.reply,
            panels=agent_response.panels,
            sources=agent_response.sources
        )
