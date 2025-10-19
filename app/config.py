from decimal import Decimal
from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
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
    enable_gmx: bool = Field(default=True, description="Enable GMX v2 perps provider")
    enable_perennial: bool = Field(default=True, description="Enable Perennial perps provider")
    enable_cex_proxy: bool = Field(default=False, description="Enable centralized exchange proxy provider")

    # Pro entitlement (token gating)
    pro_token_address: str = Field(
        default="",
        description="Contract address that unlocks Pro access when held",
    )
    pro_token_chain: str = Field(
        default="ethereum",
        description="Chain identifier for the entitlement token",
    )
    pro_token_standard: str = Field(
        default="erc20",
        description="Token standard for entitlement (erc20, erc721, erc1155)",
    )
    pro_token_id: str | None = Field(
        default=None,
        description="Specific token ID required for ERC-1155 gating (decimal or hex)",
    )
    pro_token_decimals: int = Field(
        default=18,
        description="Decimals for ERC-20 entitlement balance checks",
    )
    pro_token_min_balance: Decimal = Field(
        default=Decimal("0"),
        description="Minimum balance required to unlock Pro (human units)",
    )

    relay_base_url: str = Field(
        default="",
        description="Override the default Relay API base URL",
    )

    # Feature Flags
    feature_flag_fake_perps: bool = Field(default=True, description="Use deterministic perps data mocks")

    # LLM Provider Settings
    llm_provider: str = Field(default="anthropic", description="Default LLM provider")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    z_api_key: str = Field(default="", description="Z AI API key")
    
    # LLM Configuration
    llm_model: str = Field(default="claude-3-5-sonnet-20241022", description="Default LLM model")
    max_tokens: int = Field(default=4000, description="Maximum tokens for LLM response")
    temperature: float = Field(default=0.7, description="LLM temperature setting")
    context_window_size: int = Field(default=8000, description="Context window size for conversations")
    
    # LLM Features
    enable_streaming: bool = Field(default=False, description="Enable streaming responses")
    enable_llm_health_check: bool = Field(default=True, description="Enable LLM provider health checks")

    # Risk & Policy Defaults
    default_max_leverage: float = Field(default=3.0, description="Default maximum leverage for perps simulations")
    default_max_daily_loss_usd: float = Field(default=300.0, description="Default maximum daily loss budget in USD")
    default_max_position_notional_usd: float = Field(default=5000.0, description="Default maximum per-position notional in USD")
    default_per_trade_risk_cap_usd: float = Field(default=150.0, description="Default maximum per-trade risk in USD")
    default_kelly_cap: float = Field(default=0.5, description="Maximum Kelly fraction allowed for sizing suggestions")
    default_var_conf: float = Field(default=0.95, description="Default confidence level for VaR/ES calculations")
    default_provider_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "anthropic": "claude-3-5-sonnet-20241022",
            "claude": "claude-3-5-sonnet-20241022",
            "zai": "glm-4.6",
            "z": "glm-4.6",
        },
        description="Provider-specific default model overrides",
    )

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
    def has_zai_key(self) -> bool:
        return bool(self.z_api_key)

    @property
    def has_llm_key(self) -> bool:
        """Check if we have an API key for the configured LLM provider"""
        if self.llm_provider.lower() in ["anthropic", "claude"]:
            return self.has_anthropic_key
        elif self.llm_provider.lower() in ["openai", "gpt"]:
            return self.has_openai_key
        elif self.llm_provider.lower() in ["zai", "z"]:
            return self.has_zai_key
        return False

    def resolve_default_model(self, provider: str) -> str:
        provider_lower = provider.lower()
        return self.default_provider_models.get(provider_lower, self.llm_model)


# Global settings instance
settings = Settings()
