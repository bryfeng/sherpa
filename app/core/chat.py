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
from ..tools.defillama import get_tvl_series
from ..core.agent import Agent, PersonaManager, ContextManager

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

    persona_manager = PersonaManager()
    context_manager = ContextManager(
        llm_provider=llm_provider,
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

            persona_name = await agent._determine_persona(  # pylint: disable=protected-access
                request.messages,
                conversation_id,
                getattr(request, 'persona', None),
            )

            await agent._ensure_portfolio_context(conversation_id, request)  # pylint: disable=protected-access

            context_messages = await agent._prepare_context(  # pylint: disable=protected-access
                request,
                conversation_id,
                persona_name,
            )

            tool_data = await agent._execute_tools(  # pylint: disable=protected-access
                request,
                conversation_id,
            )
            if not isinstance(tool_data, dict):
                tool_data = {}

            wallet_address = tool_data.get('_address')
            if wallet_address is None:
                wallet_address = agent._extract_wallet_address(request)  # pylint: disable=protected-access
                if wallet_address:
                    tool_data['_address'] = wallet_address

            bridge_quote = await agent.bridge_manager.maybe_handle(  # type: ignore[attr-defined]
                request,
                conversation_id,
                wallet_address=wallet_address,
                default_chain=getattr(request, 'chain', None),
            )
            if bridge_quote:
                tool_data['bridge_quote'] = bridge_quote

            portfolio_tokens = None
            portfolio_entry = tool_data.get('portfolio') if isinstance(tool_data, dict) else None
            if isinstance(portfolio_entry, dict):
                data = portfolio_entry.get('data')
                if isinstance(data, dict):
                    portfolio_tokens = data.get('tokens')
            if portfolio_tokens is None and agent.context_manager:
                try:
                    conversation = agent.context_manager._conversations.get(conversation_id)  # type: ignore[attr-defined]
                    if conversation and getattr(conversation, 'portfolio_context', None):
                        portfolio_tokens = conversation.portfolio_context.get('tokens')  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - best-effort only
                    portfolio_tokens = None

            swap_quote = await agent.swap_manager.maybe_handle(  # type: ignore[attr-defined]
                request,
                conversation_id,
                wallet_address=wallet_address,
                default_chain=getattr(request, 'chain', None),
                portfolio_tokens=portfolio_tokens,
            )
            if swap_quote:
                tool_data['swap_quote'] = swap_quote

            if agent._needs_tvl_data(request):  # pylint: disable=protected-access
                try:
                    params = agent._extract_tvl_params(request)  # pylint: disable=protected-access
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
                                conversation_id,
                                {
                                    'entity': params['protocol'],
                                    'protocol': params['protocol'],
                                    'metric': 'tvl',
                                    'window': params['window'],
                                    'chain': getattr(request, 'chain', None),
                                    'stats': stats,
                                    'source': 'defillama',
                                },
                            )
                        except Exception as focus_exc:  # pragma: no cover - logging only
                            agent.logger.warning('Failed setting episodic focus: %s', focus_exc)
                except Exception as tvl_exc:  # pragma: no cover - defensive logging
                    agent.logger.error('DefiLlama TVL fetch error: %s', tvl_exc)

            accumulated_text = ""

            async for delta in agent._stream_llm_response(  # pylint: disable=protected-access
                context_messages,
                tool_data,
                conversation_id,
            ):
                if not delta:
                    continue
                accumulated_text += delta
                yield _sse_event({
                    'type': 'delta',
                    'delta': delta,
                })

            llm_response = LLMResponse(content=accumulated_text)
            final_response = await agent._format_response(  # pylint: disable=protected-access
                llm_response,
                tool_data,
                persona_name,
                conversation_id,
                start_time,
            )

            await agent._update_conversation_state(  # pylint: disable=protected-access
                conversation_id,
                request,
                final_response,
            )

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
