"""
Enhanced Chat System with LLM Agent Integration

This module replaces hardcoded intent classification and template responses 
with intelligent LLM-powered conversations using the Agent system.
"""

import logging
from typing import Optional
from ..types import ChatRequest, ChatResponse
from ..providers.llm import get_llm_provider
from ..core.agent import Agent, PersonaManager, ContextManager

# Global agent instance for chat processing
_agent: Optional[Agent] = None
_logger = logging.getLogger(__name__)


def _get_agent() -> Agent:
    """Get or create the global Agent instance"""
    global _agent
    
    if _agent is None:
        try:
            # Initialize LLM provider
            llm_provider = get_llm_provider()
            
            # Initialize persona and context managers
            persona_manager = PersonaManager()
            context_manager = ContextManager()
            
            # Create agent instance
            _agent = Agent(
                llm_provider=llm_provider,
                persona_manager=persona_manager,
                context_manager=context_manager,
                logger=_logger
            )
            
            _logger.info("Agent system initialized successfully")
            
        except Exception as e:
            _logger.error(f"Failed to initialize Agent system: {str(e)}")
            raise RuntimeError(f"Agent initialization failed: {str(e)}")
    
    return _agent


async def run_chat(request: ChatRequest) -> ChatResponse:
    """
    Process chat request using the LLM-powered Agent system.
    
    This replaces the old hardcoded intent classification with intelligent 
    AI responses while maintaining full API compatibility.
    """
    
    try:
        # Get the agent instance
        agent = _get_agent()

        # Determine effective conversation_id (wallet-scoped reuse)
        effective_conversation_id = None
        try:
            if agent.context_manager:
                effective_conversation_id = agent.context_manager.get_or_create_conversation(
                    address=getattr(request, 'address', None),
                    conversation_id=getattr(request, 'conversation_id', None),
                )
        except Exception as e:
            _logger.warning(f"Conversation routing fallback: {str(e)}")
        
        # Process the message through the agent
        agent_response = await agent.process_message(
            request=request,
            conversation_id=effective_conversation_id or getattr(request, 'conversation_id', None),
            persona_name=getattr(request, 'persona', None)
        )
        
        # Convert agent response to API-compatible format
        chat_response = agent.to_chat_response(agent_response)
        
        # Log successful processing
        _logger.info(
            f"Chat processed successfully - Persona: {agent_response.persona_used}, "
            f"Tokens: {agent_response.tokens_used}, "
            f"Time: {agent_response.processing_time_ms:.1f}ms"
        )
        
        return chat_response
        
    except Exception as e:
        _logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        
        # Return fallback response on error
        return ChatResponse(
            reply="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            panels={},
            sources=[]
        )


# Legacy function compatibility (deprecated but maintained for backward compatibility)
def extract_wallet_address(text: str) -> Optional[str]:
    """
    DEPRECATED: Extract wallet address from text using regex.
    This function is now handled by the Agent system but maintained for compatibility.
    """
    import re
    pattern = r'0x[a-fA-F0-9]{40}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None


def classify_intent(message: str) -> str:
    """
    DEPRECATED: Simple intent classification based on keywords.
    This function is now handled by the LLM Agent but maintained for compatibility.
    """
    message_lower = message.lower()
    portfolio_keywords = ["portfolio", "balance", "tokens", "holdings", "wallet", "what's in", "whats in"]
    
    if any(keyword in message_lower for keyword in portfolio_keywords):
        return "portfolio"
    
    return "unknown"
