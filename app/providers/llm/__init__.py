from typing import Any, Dict, Type, Optional

from .base import LLMProvider, LLMMessage, LLMResponse, LLMProviderError
from .anthropic import AnthropicProvider
from .zai import ZAIProvider

PROVIDER_ALIAS_MAP: Dict[str, str] = {
    "claude": "anthropic",
    "z": "zai",
}

PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "anthropic": "Anthropic Claude",
    "zai": "Zeta AI",
}


def canonical_provider_name(name: str) -> str:
    """Normalize provider aliases to their canonical identifier."""

    return PROVIDER_ALIAS_MAP.get(name.lower(), name.lower())


# Registry of available LLM providers
PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,  # Alias for Anthropic
    "zai": ZAIProvider,
    "z": ZAIProvider,
}


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    @staticmethod
    def create_provider(
        provider_name: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMProvider:
        """Create an LLM provider instance."""

        provider_key = canonical_provider_name(provider_name)
        if provider_key not in PROVIDER_REGISTRY:
            available_providers = ", ".join(PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )

        provider_class = PROVIDER_REGISTRY[provider_key]

        if not model:
            raise ValueError(f"No model provided for provider '{provider_key}'.")

        try:
            return provider_class(api_key=api_key, model=model, **kwargs)
        except ImportError as exc:  # pragma: no cover - dependency issues
            raise ImportError(
                f"Required dependencies for {provider_key} provider not installed. "
                f"Error: {exc}"
            )


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Return metadata about supported LLM providers, grouped by canonical id."""

    providers_info: Dict[str, Dict[str, Any]] = {}

    from ...config import settings  # Local import to avoid circular dependency

    seen = set()
    for registry_name in PROVIDER_REGISTRY:
        provider_name = canonical_provider_name(registry_name)
        if provider_name in seen:
            continue
        seen.add(provider_name)

        display_name = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title())
        models = settings.provider_models_catalog.get(provider_name, [])

        info: Dict[str, Any] = {
            "status": "available",
            "default_model": settings.resolve_default_model(provider_name),
            "description": f"{display_name} API",
            "display_name": display_name,
            "models": models,
        }

        try:
            if provider_name == "anthropic":
                import anthropic  # type: ignore  # noqa: F401
                info["description"] = "Anthropic Claude API"
            elif provider_name == "zai":
                info["description"] = "Z AI Chat Completions API"
        except ImportError:
            if provider_name == "anthropic":
                info.update(
                    {
                        "status": "unavailable",
                        "reason": "Missing dependencies",
                        "description": "Anthropic Claude API (install: pip install anthropic)",
                    }
                )

        providers_info[provider_name] = info

    return providers_info


def get_llm_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Instantiate an LLM provider according to configuration overrides."""

    from ...config import settings

    provider_input = (provider_name or "").strip().lower() or None
    provider_explicit = provider_input is not None

    model_input = (model or "").strip() or None

    requested_provider = provider_input or settings.llm_provider
    resolved_provider = canonical_provider_name(requested_provider)

    if not provider_explicit and model_input:
        detected_provider = settings.resolve_provider_for_model(model_input)
        if detected_provider:
            resolved_provider = canonical_provider_name(detected_provider)

    if resolved_provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif resolved_provider == "zai":
        api_key = settings.z_api_key
    else:
        api_key = None

    if not api_key:
        raise ValueError(f"No API key configured for provider: {resolved_provider}")

    resolved_model = model_input or settings.llm_model
    if isinstance(resolved_model, str):
        resolved_model = resolved_model.strip() or None

    allowed_models = settings.provider_models_catalog.get(resolved_provider, [])
    allowed_ids = {entry.get("id") for entry in allowed_models if entry.get("id")}

    if resolved_model is None:
        resolved_model = settings.resolve_default_model(resolved_provider)
    elif allowed_ids and resolved_model not in allowed_ids:
        if not provider_explicit and model_input:
            detected_provider = settings.resolve_provider_for_model(model_input)
            if detected_provider:
                new_provider = canonical_provider_name(detected_provider)
                if new_provider != resolved_provider:
                    resolved_provider = new_provider
                    if resolved_provider == "anthropic":
                        api_key = settings.anthropic_api_key
                    elif resolved_provider == "zai":
                        api_key = settings.z_api_key
                    else:
                        api_key = None
                    if not api_key:
                        raise ValueError(f"No API key configured for provider: {resolved_provider}")
                    allowed_models = settings.provider_models_catalog.get(resolved_provider, [])
                    allowed_ids = {entry.get("id") for entry in allowed_models if entry.get("id")}
                    if allowed_ids and model_input in allowed_ids:
                        resolved_model = model_input
                    else:
                        resolved_model = settings.resolve_default_model(resolved_provider)
                else:
                    resolved_model = settings.resolve_default_model(resolved_provider)
            else:
                resolved_model = settings.resolve_default_model(resolved_provider)
        else:
            resolved_model = settings.resolve_default_model(resolved_provider)

    return LLMProviderFactory.create_provider(
        provider_name=resolved_provider,
        api_key=api_key,
        model=resolved_model,
        **kwargs,
    )


__all__ = [
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMProviderError",
    "AnthropicProvider",
    "ZAIProvider",
    "LLMProviderFactory",
    "get_available_providers",
    "get_llm_provider",
    "PROVIDER_REGISTRY",
    "canonical_provider_name",
]
