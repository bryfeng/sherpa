from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Server Settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # External API Keys
    alchemy_api_key: str = Field(default="", description="Alchemy API key")
    coingecko_api_key: str = Field(default="", description="Coingecko API key")

    # Cache Settings
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache size")

    # Rate Limiting
    max_concurrent_requests: int = Field(default=10, description="Max concurrent API requests")
    request_timeout_seconds: int = Field(default=30, description="Request timeout")

    # Provider Toggles
    enable_alchemy: bool = Field(default=True, description="Enable Alchemy provider")
    enable_coingecko: bool = Field(default=True, description="Enable Coingecko provider")

    # LLM Provider Settings
    llm_provider: str = Field(default="anthropic", description="Default LLM provider")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    
    # LLM Configuration
    llm_model: str = Field(default="claude-3-sonnet-20240229", description="Default LLM model")
    max_tokens: int = Field(default=4000, description="Maximum tokens for LLM response")
    temperature: float = Field(default=0.7, description="LLM temperature setting")
    context_window_size: int = Field(default=8000, description="Context window size for conversations")
    
    # LLM Features
    enable_streaming: bool = Field(default=False, description="Enable streaming responses")
    enable_llm_health_check: bool = Field(default=True, description="Enable LLM provider health checks")

    @property
    def has_alchemy_key(self) -> bool:
        return bool(self.alchemy_api_key)

    @property
    def has_coingecko_key(self) -> bool:
        return bool(self.coingecko_api_key)
    
    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)
    
    @property
    def has_llm_key(self) -> bool:
        """Check if we have an API key for the configured LLM provider"""
        if self.llm_provider.lower() in ["anthropic", "claude"]:
            return self.has_anthropic_key
        elif self.llm_provider.lower() in ["openai", "gpt"]:
            return self.has_openai_key
        return False


# Global settings instance
settings = Settings()
