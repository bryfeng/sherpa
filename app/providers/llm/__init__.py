from typing import Dict, Type, Optional

from .base import LLMProvider, LLMMessage, LLMResponse, LLMProviderError
from .anthropic import AnthropicProvider
from .zai import ZAIProvider

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

        provider_name = provider_name.lower()
        if provider_name not in PROVIDER_REGISTRY:
            available_providers = ", ".join(PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )

        provider_class = PROVIDER_REGISTRY[provider_name]

        if model is None:
            if provider_name in ["anthropic", "claude"]:
                model = "claude-sonnet-4-20250514"
            elif provider_name in ["zai", "z"]:
                model = "glm-4.6"

        try:
            return provider_class(api_key=api_key, model=model, **kwargs)
        except ImportError as exc:  # pragma: no cover - dependency issues
            raise ImportError(
                f"Required dependencies for {provider_name} provider not installed. "
                f"Error: {exc}"
            )


def get_available_providers() -> Dict[str, Dict[str, str]]:
    """Return metadata about supported LLM providers."""

    providers_info: Dict[str, Dict[str, str]] = {}

    for provider_name in PROVIDER_REGISTRY:
        try:
            if provider_name in ["anthropic", "claude"]:
                import anthropic  # type: ignore  # noqa: F401
                providers_info[provider_name] = {
                    "status": "available",
                    "default_model": "claude-sonnet-4-20250514",
                    "description": "Anthropic Claude API",
                }
            elif provider_name in ["zai", "z"]:
                providers_info[provider_name] = {
                    "status": "available",
                    "default_model": "glm-4.6",
                    "description": "Z AI Chat Completions API",
                }
        except ImportError:
            if provider_name in ["anthropic", "claude"]:
                providers_info[provider_name] = {
                    "status": "unavailable",
                    "reason": "Missing dependencies",
                    "description": "Anthropic Claude API (install: pip install anthropic)",
                }
            else:
                providers_info[provider_name] = {
                    "status": "available",
                    "default_model": "glm-4.6",
                    "description": "Z AI Chat Completions API",
                }

    return providers_info


def get_llm_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Instantiate an LLM provider according to configuration overrides."""

    from ...config import settings

    resolved_provider = (provider_name or settings.llm_provider).lower()

    if resolved_provider in ["anthropic", "claude"]:
        api_key = settings.anthropic_api_key
    elif resolved_provider in ["zai", "z"]:
        api_key = settings.z_api_key
    else:
        api_key = None

    if not api_key:
        raise ValueError(f"No API key configured for provider: {resolved_provider}")

    resolved_model = model or settings.llm_model
    if isinstance(resolved_model, str):
        resolved_model = resolved_model.strip() or None

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
]
