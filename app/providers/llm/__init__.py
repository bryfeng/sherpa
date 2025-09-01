from typing import Dict, Type, Optional
from .base import LLMProvider, LLMMessage, LLMResponse, LLMProviderError
from .anthropic import AnthropicProvider

# Registry of available LLM providers
PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,  # Alias for anthropic
}


class LLMProviderFactory:
    """Factory for creating LLM provider instances"""
    
    @staticmethod
    def create_provider(
        provider_name: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider ("anthropic", "openai", etc.)
            api_key: API key for the provider
            model: Model name (optional, uses provider default if not specified)
            **kwargs: Additional provider-specific configuration
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If required dependencies are not installed
        """
        provider_name = provider_name.lower()
        
        if provider_name not in PROVIDER_REGISTRY:
            available_providers = ", ".join(PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = PROVIDER_REGISTRY[provider_name]
        
        # Set default models if not specified
        if model is None:
            if provider_name in ["anthropic", "claude"]:
                model = "claude-sonnet-4-20250514"
        
        try:
            return provider_class(api_key=api_key, model=model, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"Required dependencies for {provider_name} provider not installed. "
                f"Error: {e}"
            )


def get_available_providers() -> Dict[str, Dict[str, str]]:
    """Get information about available providers
    
    Returns:
        Dictionary with provider info including default models and status
    """
    providers_info = {}
    
    for provider_name, provider_class in PROVIDER_REGISTRY.items():
        try:
            # Try to import dependencies to check availability
            if provider_name in ["anthropic", "claude"]:
                import anthropic
                providers_info[provider_name] = {
                    "status": "available",
                    "default_model": "claude-3-sonnet-20240229",
                    "description": "Anthropic Claude API"
                }
        except ImportError:
            providers_info[provider_name] = {
                "status": "unavailable",
                "reason": "Missing dependencies",
                "description": "Anthropic Claude API (install: pip install anthropic)"
            }
    
    return providers_info


def get_llm_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """Get an LLM provider instance using configuration
    
    Args:
        provider_name: Optional provider name override
        **kwargs: Additional provider configuration
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If configuration is invalid
        ImportError: If required dependencies are missing
    """
    from ...config import settings
    
    # Use provided provider name or default from config
    provider_name = provider_name or settings.llm_provider
    
    # Get API key from config
    api_key = None
    if provider_name in ["anthropic", "claude"]:
        api_key = settings.anthropic_api_key
    
    if not api_key:
        raise ValueError(f"No API key configured for provider: {provider_name}")
    
    # Create provider using factory
    return LLMProviderFactory.create_provider(
        provider_name=provider_name,
        api_key=api_key,
        **kwargs
    )


# Export main classes and functions
__all__ = [
    "LLMProvider",
    "LLMMessage", 
    "LLMResponse",
    "LLMProviderError",
    "AnthropicProvider",
    "LLMProviderFactory",
    "get_available_providers",
    "get_llm_provider",
    "PROVIDER_REGISTRY"
]
