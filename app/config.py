import os

from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        """Ensure we pick up legacy environment variable aliases."""

        super().model_post_init(__context)

        if not self.z_api_key:
            fallback = os.getenv("ZAI_API_KEY") or os.getenv("ZAI_KEY")
            if fallback:
                object.__setattr__(self, "z_api_key", fallback)

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
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting middleware")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent API requests")
    request_timeout_seconds: int = Field(default=30, description="Request timeout")

    # Provider Toggles
    enable_alchemy: bool = Field(default=True, description="Enable Alchemy provider")
    enable_coingecko: bool = Field(default=True, description="Enable Coingecko provider")
    enable_gmx: bool = Field(default=True, description="Enable GMX v2 perps provider")
    enable_perennial: bool = Field(default=True, description="Enable Perennial perps provider")
    enable_cex_proxy: bool = Field(default=False, description="Enable centralized exchange proxy provider")
    enable_erc4337: bool = Field(default=False, description="Enable ERC-4337 smart wallet execution")
    enable_alchemy_wallet_api: bool = Field(
        default=False,
        description="Enable Alchemy Wallet APIs for smart wallet execution",
    )

    # Solana / Non-EVM Providers
    solana_helius_api_key: str = Field(
        default="",
        description="Helius API key used for Solana portfolio aggregation",
    )
    solana_balances_base_url: str = Field(
        default="https://api.helius.xyz",
        description="Base URL for Solana balances API",
    )

    # Birdeye Provider (DeFi analytics, top traders)
    birdeye_api_key: str = Field(
        default="",
        description="Birdeye API key for DeFi analytics and trader discovery",
    )
    enable_birdeye: bool = Field(
        default=True,
        description="Enable Birdeye provider for trader analytics",
    )

    # Webhook Configuration (Event Monitoring)
    alchemy_webhook_signing_key: str = Field(
        default="",
        description="Alchemy webhook signing key for signature verification",
    )
    helius_api_key: str = Field(
        default="",
        description="Helius API key for Solana webhooks (can be same as solana_helius_api_key)",
    )
    webhook_base_url: str = Field(
        default="",
        description="Base URL for webhook endpoints (e.g., https://api.yourapp.com)",
    )

    # Jupiter Provider (Solana token list and prices)
    enable_jupiter: bool = Field(
        default=True,
        description="Enable Jupiter provider for Solana token lookups",
    )
    jupiter_cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for Jupiter token list cache (default: 1 hour)",
    )

    # ERC-4337 Providers
    erc4337_bundler_url: str = Field(
        default="",
        description="ERC-4337 bundler RPC URL",
    )
    erc4337_bundler_urls: Dict[int, str] = Field(
        default_factory=dict,
        description="Optional JSON map of chain_id -> bundler RPC URL",
    )
    erc4337_paymaster_url: str = Field(
        default="",
        description="ERC-4337 paymaster RPC URL",
    )
    erc4337_paymaster_rpc_method: str = Field(
        default="pm_sponsorUserOperation",
        description="RPC method used to request paymaster sponsorship",
    )
    erc4337_entrypoint_address: str = Field(
        default="",
        description="ERC-4337 EntryPoint contract address",
    )
    erc4337_account_execute_signature: str = Field(
        default="execute(address,uint256,bytes)",
        description="Smart account execute function signature for callData encoding",
    )
    erc4337_account_execute_selector: str = Field(
        default="",
        description="Override execute function selector (0x....) if signature differs",
    )

    # Alchemy Wallet APIs
    alchemy_wallet_api_url: str = Field(
        default="",
        description="Override Alchemy Wallet API base URL (defaults to https://api.g.alchemy.com/v2/{alchemy_api_key})",
    )
    alchemy_wallet_api_timeout_seconds: int = Field(
        default=20,
        description="Alchemy Wallet API HTTP timeout (seconds)",
    )

    # Swig (Solana Smart Wallet)
    enable_swig: bool = Field(default=False, description="Enable Swig smart wallet integration")
    swig_base_url: str = Field(
        default="",
        description="Swig API base URL",
    )
    swig_api_key: str = Field(
        default="",
        description="Swig API key",
    )

    # Rhinestone (Smart Wallet + Intent Infrastructure)
    enable_rhinestone: bool = Field(
        default=False,
        description="Enable Rhinestone Smart Wallet and Intent infrastructure",
    )
    rhinestone_api_key: str = Field(
        default="",
        description="Rhinestone API key (optional for development)",
    )
    rhinestone_base_url: str = Field(
        default="https://api.rhinestone.dev",
        description="Rhinestone API base URL",
    )

    redis_url: str = Field(
        default="",
        description="Redis connection string used for caching wallet history summaries",
    )
    aws_s3_export_bucket: str = Field(
        default="",
        description="Bucket name used for historical export artifacts",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS/minio region for export bucket",
    )

    # Convex Database
    convex_url: str = Field(
        default="",
        description="Convex deployment URL (e.g., https://your-deployment.convex.cloud)",
    )
    convex_deploy_key: str = Field(
        default="",
        description="Convex deploy key for server-side mutations",
    )
    convex_internal_api_key: str = Field(
        default="",
        description="Internal API key for Convex HTTP actions",
    )

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
    z_api_key: str = Field(
        default="",
        description="Z AI API key",
        validation_alias=AliasChoices("z_api_key", "zai_api_key", "Z_API_KEY", "ZAI_API_KEY"),
    )
    
    # LLM Configuration
    llm_model: str = Field(default="claude-sonnet-4-5-20250929", description="Default LLM model")
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
    provider_models_catalog: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=lambda: {
            "anthropic": [
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "label": "Claude Sonnet 4.5",
                    "description": "Balanced depth and latency for daily use.",
                    "default": True,
                },
            ],
            "zai": [
                {
                    "id": "glm-4.7",
                    "label": "Zeta GLM 4.7",
                    "description": "Relay-native experimentation model.",
                    "default": True,
                }
            ],
        },
        description="Provider models metadata surfaced to clients",
    )
    history_summary_default_limit: int = Field(
        default=250,
        ge=10,
        le=2500,
        description="Default number of transactions to include when no explicit window is provided",
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
    def has_convex(self) -> bool:
        """Check if Convex is configured."""
        return bool(self.convex_url)

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
        options = self.provider_models_catalog.get(provider_lower, [])
        for option in options:
            default_flag = option.get("default")
            if isinstance(default_flag, str):
                is_default = default_flag.lower() in {"true", "1", "yes"}
            else:
                is_default = bool(default_flag)
            if is_default:
                return option.get("id", self.llm_model)
        if options:
            return options[0].get("id", self.llm_model)
        return self.llm_model

    def resolve_provider_for_model(self, model_id: str) -> Optional[str]:
        target = (model_id or "").strip().lower()
        if not target:
            return None
        for provider, options in self.provider_models_catalog.items():
            for option in options:
                option_id = option.get("id")
                if option_id and option_id.lower() == target:
                    return provider
        return None

    # Agent Runtime
    agent_runtime_enabled: bool = Field(
        default=True,
        description="Start the background agent runtime alongside FastAPI",
    )
    agent_runtime_default_interval_seconds: int = Field(
        default=60,
        description="Default tick interval for background strategies",
    )
    agent_runtime_max_concurrency: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent strategy tasks",
    )
    agent_runtime_tick_timeout_seconds: int = Field(
        default=20,
        ge=1,
        description="Max seconds to allow a strategy tick to run before timing out",
    )


# Global settings instance
settings = Settings()
