"""
Enhanced Chat System with LLM Agent Integration

This module replaces hardcoded intent classification and template responses 
with intelligent LLM-powered conversations using the Agent system.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi.encoders import jsonable_encoder

from ..config import settings
from ..types import ChatRequest, ChatResponse
from ..providers.llm import get_llm_provider
from ..providers.llm.base import LLMResponse
from ..core.agent import Agent, PersonaManager, ContextManager
from ..db.convex_client import get_convex_client

# Cache of agents keyed by provider/model so multiple LLMs can coexist
_agent_cache: Dict[str, Agent] = {}
_agent_provider_cache: Dict[str, str] = {}
_logger = logging.getLogger(__name__)


def _normalize_provider(provider_name: Optional[str]) -> str:
    return (provider_name or settings.llm_provider or "anthropic").lower()


def _normalize_model(model: Optional[str]) -> str:
    return (model or settings.llm_model or "").strip()


def _agent_cache_key(provider_name: Optional[str], model: Optional[str]) -> str:
    provider = _normalize_provider(provider_name)
    model_value = _normalize_model(model)
    return f"{provider}:{model_value}"


def _create_agent(provider_name: Optional[str] = None, model: Optional[str] = None) -> Agent:
    llm_provider = get_llm_provider(provider_name=provider_name, model=model)
    resolved_provider = _normalize_provider(provider_name)
    resolved_model = llm_provider.model

    # Get Convex client for conversation persistence
    try:
        convex_client = get_convex_client()
    except Exception as e:
        _logger.warning(f"Convex client not available, conversations will not persist: {e}")
        convex_client = None

    persona_manager = PersonaManager()
    context_manager = ContextManager(
        llm_provider=llm_provider,
        convex_client=convex_client,
        max_tokens=settings.context_window_size,
    )

    agent = Agent(
        llm_provider=llm_provider,
        persona_manager=persona_manager,
        context_manager=context_manager,
        logger=_logger,
        provider_id=resolved_provider,
        model_id=resolved_model,
    )

    _logger.info(
        "Agent system initialized for provider=%s model=%s",
        resolved_provider,
        resolved_model,
    )
    key = _agent_cache_key(provider_name, model)
    _agent_provider_cache[key] = _normalize_provider(provider_name)
    return agent


def _get_agent(provider_name: Optional[str] = None, model: Optional[str] = None) -> Agent:
    key = _agent_cache_key(provider_name, model)
    agent = _agent_cache.get(key)
    if agent is None:
        agent = _create_agent(provider_name, model)
        _agent_cache[key] = agent
    else:
        desired_provider = _normalize_provider(provider_name)
        cached_provider = _agent_provider_cache.get(key)
        if cached_provider != desired_provider:
            agent = _create_agent(provider_name, model)
            _agent_cache[key] = agent
    return agent


def _sse_event(payload: Dict[str, Any]) -> str:
    encoded = jsonable_encoder(payload)
    return f"data: {json.dumps(encoded, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


async def run_chat(request: ChatRequest) -> ChatResponse:
    """Process chat request using the LLM-powered Agent system."""

    try:
        agent = _get_agent(
            getattr(request, "llm_provider", None),
            getattr(request, "llm_model", None),
        )

        effective_conversation_id = None
        try:
            if agent.context_manager:
                effective_conversation_id = agent.context_manager.get_or_create_conversation(
                    address=getattr(request, 'address', None),
                    conversation_id=getattr(request, 'conversation_id', None),
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning(f"Conversation routing fallback: {exc}")

        agent_response = await agent.process_message(
            request=request,
            conversation_id=effective_conversation_id or getattr(request, 'conversation_id', None),
            persona_name=getattr(request, 'persona', None)
        )

        chat_response = agent.to_chat_response(agent_response)

        _logger.info(
            "Chat processed successfully (provider=%s model=%s) - Persona: %s, Tokens: %s, Time: %.1fms",
            getattr(request, "llm_provider", None) or settings.llm_provider,
            getattr(request, "llm_model", None) or settings.llm_model,
            agent_response.persona_used,
            agent_response.tokens_used,
            agent_response.processing_time_ms or 0.0,
        )

        return chat_response

    except Exception as exc:  # pragma: no cover - defensive fallback
        _logger.error(f"Chat processing error: {exc}", exc_info=True)
        provider_hint = _normalize_provider(getattr(request, "llm_provider", None))
        model_hint = _normalize_model(getattr(request, "llm_model", None)) or settings.resolve_default_model(provider_hint)
        return ChatResponse(
            reply="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            panels={},
            sources=[],
            llm_provider=provider_hint,
            llm_model=model_hint,
        )


def stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream chat responses using the configured LLM provider."""

    async def event_generator() -> AsyncGenerator[str, None]:
        agent = _get_agent(
            getattr(request, "llm_provider", None),
            getattr(request, "llm_model", None),
        )

        start_time = datetime.now()
        conversation_id: Optional[str] = getattr(request, "conversation_id", None)

        if agent.context_manager:
            try:
                conversation_id = agent.context_manager.get_or_create_conversation(
                    address=getattr(request, 'address', None),
                    conversation_id=conversation_id,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                _logger.warning("Conversation routing fallback: %s", exc)

        if conversation_id is None:
            conversation_id = f"guest-{uuid.uuid4().hex[:8]}"

        try:
            # Check for style commands first
            style_response = await agent._handle_style_processing(  # pylint: disable=protected-access
                request.messages,
                conversation_id,
            )
            if style_response:
                chat_response = agent.to_chat_response(style_response)
                yield _sse_event({
                    'type': 'final',
                    'response': chat_response.model_dump(),
                })
                yield _sse_done()
                return

            # Determine persona
            persona_name = await agent._determine_persona(  # pylint: disable=protected-access
                request.messages,
                conversation_id,
                getattr(request, 'persona', None),
            )

            # Ensure portfolio context is loaded
            await agent._ensure_portfolio_context(conversation_id, request)  # pylint: disable=protected-access

            # Prepare context messages
            context_messages = await agent._prepare_context(  # pylint: disable=protected-access
                request,
                conversation_id,
                persona_name,
            )

            # ============================================================
            # NEW: Use ReAct loop for LLM-driven tool calling
            # This allows the LLM to decide which tools to call based on
            # semantic understanding, rather than keyword matching
            # ============================================================
            llm_response, tool_data = await agent._run_react_loop(  # pylint: disable=protected-access
                context_messages,
                request,
                conversation_id,
            )

            # Get wallet address from tool_data or extract it
            wallet_address = tool_data.get('_address')
            if wallet_address is None:
                wallet_address = agent._extract_wallet_address(request)  # pylint: disable=protected-access
                if wallet_address:
                    tool_data['_address'] = wallet_address

            # Get portfolio tokens and chain for bridge/swap handling
            portfolio_tokens = None
            portfolio_chain = None
            portfolio_entry = tool_data.get('get_portfolio') if isinstance(tool_data, dict) else None
            if isinstance(portfolio_entry, dict):
                result = portfolio_entry.get('result')
                if isinstance(result, dict):
                    data = result.get('data')
                    if isinstance(data, dict):
                        portfolio_tokens = data.get('tokens')
                        portfolio_chain = data.get('chain')
            if portfolio_tokens is None and agent.context_manager:
                try:
                    conversation = agent.context_manager._conversations.get(conversation_id)  # type: ignore[attr-defined]
                    if conversation and getattr(conversation, 'portfolio_context', None):
                        portfolio_tokens = conversation.portfolio_context.get('tokens')  # type: ignore[call-arg]
                        if portfolio_chain is None:
                            portfolio_chain = conversation.portfolio_context.get('chain')  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - best-effort only
                    portfolio_tokens = None

            # Use portfolio chain if request chain is default, otherwise use request chain
            request_chain = getattr(request, 'chain', None)
            effective_chain = portfolio_chain if (portfolio_chain and request_chain in (None, 'ethereum')) else request_chain

            # Handle bridge quotes (special case, keyword-based for now)
            bridge_quote = await agent.bridge_manager.maybe_handle(  # type: ignore[attr-defined]
                request,
                conversation_id,
                wallet_address=wallet_address,
                default_chain=effective_chain,
            )
            if bridge_quote:
                tool_data['bridge_quote'] = bridge_quote

            # Handle swap quotes (special case, keyword-based for now)
            swap_quote = await agent.swap_manager.maybe_handle(  # type: ignore[attr-defined]
                request,
                conversation_id,
                wallet_address=wallet_address,
                default_chain=effective_chain,
                portfolio_tokens=portfolio_tokens,
            )
            if swap_quote:
                tool_data['swap_quote'] = swap_quote

            # ============================================================
            # Stream the response text in chunks for better UX
            # The ReAct loop already completed, so we stream the result
            # ============================================================
            response_text = llm_response.content or ""

            # Handle case where LLM didn't generate text (e.g., all tool calls failed)
            if not response_text:
                # Check if there were tool errors
                tool_errors = []
                for tool_name, tool_result in tool_data.items():
                    if tool_name.startswith('_'):
                        continue
                    if isinstance(tool_result, dict):
                        result = tool_result.get('result', {})
                        if isinstance(result, dict) and not result.get('success', True):
                            tool_errors.append(f"{tool_name}: {result.get('error', 'Unknown error')}")

                if tool_errors:
                    response_text = (
                        "I encountered some issues while processing your request:\n\n"
                        + "\n".join(f"- {err}" for err in tool_errors)
                        + "\n\nPlease try again or let me know if you need help with something else."
                    )
                else:
                    response_text = "I processed your request but didn't generate a response. Please try again."

                # Update llm_response content for format_response
                llm_response.content = response_text

            chunk_size = 20  # Characters per chunk for smooth streaming

            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield _sse_event({
                    'type': 'delta',
                    'delta': chunk,
                })
                # Small delay for smoother streaming effect (optional)
                # await asyncio.sleep(0.01)

            # Format the final response
            final_response = await agent._format_response(  # pylint: disable=protected-access
                llm_response,
                tool_data,
                persona_name,
                conversation_id,
                start_time,
            )

            # Update conversation state
            await agent._update_conversation_state(  # pylint: disable=protected-access
                conversation_id,
                request,
                final_response,
            )

            # Send final response with panels and metadata
            chat_response = agent.to_chat_response(final_response)
            yield _sse_event({
                'type': 'final',
                'response': chat_response.model_dump(),
            })

        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.error("Streaming chat error: %s", exc, exc_info=True)
            yield _sse_event({
                'type': 'error',
                'message': str(exc),
            })
        finally:
            yield _sse_done()

    return event_generator()


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
