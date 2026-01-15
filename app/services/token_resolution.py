"""
Centralized Token Resolution Service with multi-source lookup and confidence scoring.

This service consolidates all token identification logic into a single entry point,
providing:
- Multi-chain token registry with aliases (EVM + Solana)
- Portfolio-aware resolution (user's holdings get priority)
- API fallbacks (CoinGecko, Alchemy, Jupiter)
- Confidence scoring for disambiguation
- Spam detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from ..core.bridge.constants import NATIVE_PLACEHOLDER
from ..core.bridge.chain_registry import get_registry_sync
from ..core.chain_types import ChainId, SOLANA_CHAIN_ID, is_solana_chain, is_evm_chain_id
from .address import is_valid_solana_address


class ResolutionSource(str, Enum):
    """Source of a token match, ordered by reliability."""

    EXACT_ADDRESS = "exact_address"
    REGISTRY = "registry"
    PORTFOLIO = "portfolio"
    COINGECKO = "coingecko"
    ALCHEMY = "alchemy"
    JUPITER = "jupiter"
    FUZZY = "fuzzy"


@dataclass
class TokenMatch:
    """A resolved token with confidence metadata."""

    chain_id: ChainId  # int for EVM chains, "solana" for Solana
    address: str  # EVM contract address or Solana mint address
    symbol: str
    name: str
    decimals: int
    confidence: float  # 0.0 - 1.0
    source: ResolutionSource
    is_native: bool = False
    coingecko_id: Optional[str] = None
    logo_url: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)  # Jupiter tags for Solana tokens

    @property
    def canonical_id(self) -> str:
        """Unique identifier: chain_id:address."""
        # For EVM, lowercase the address. For Solana, keep original case (Base58).
        addr = self.address.lower() if is_evm_chain_id(self.chain_id) else self.address
        return f"{self.chain_id}:{addr}"

    @property
    def is_solana(self) -> bool:
        """Check if this token is on Solana."""
        return is_solana_chain(self.chain_id)

    @property
    def is_evm(self) -> bool:
        """Check if this token is on an EVM chain."""
        return is_evm_chain_id(self.chain_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "address": self.address,
            "symbol": self.symbol,
            "name": self.name,
            "decimals": self.decimals,
            "confidence": self.confidence,
            "source": self.source.value,
            "is_native": self.is_native,
            "coingecko_id": self.coingecko_id,
            "logo_url": self.logo_url,
            "canonical_id": self.canonical_id,
            "is_solana": self.is_solana,
            "tags": self.tags,
        }


@dataclass
class AmbiguityResult:
    """Result when token resolution is ambiguous."""

    query: str
    matches: List[TokenMatch]
    reason: str

    @property
    def options_text(self) -> str:
        """Human-readable options for disambiguation."""
        registry = get_registry_sync()
        lines = []
        for i, m in enumerate(self.matches[:5], 1):
            if m.is_solana:
                chain_name = "Solana"
            else:
                chain_name = registry.get_chain_name(m.chain_id)
            lines.append(f"{i}. {m.symbol} ({m.name}) on {chain_name}")
        return "\n".join(lines)


# Confidence thresholds
CONFIDENCE_EXACT_ADDRESS = 1.0
CONFIDENCE_REGISTRY_EXACT = 0.95
CONFIDENCE_PORTFOLIO = 0.90
CONFIDENCE_REGISTRY_ALIAS = 0.85
CONFIDENCE_JUPITER_EXACT = 0.80  # Jupiter exact symbol match
CONFIDENCE_COINGECKO_EXACT = 0.80
CONFIDENCE_JUPITER_SEARCH = 0.60  # Jupiter search result
CONFIDENCE_COINGECKO_SEARCH = 0.60
CONFIDENCE_ALCHEMY = 0.55
CONFIDENCE_FUZZY = 0.40

# Ambiguity threshold - below this, we should ask the user
AMBIGUITY_THRESHOLD = 0.70

# Solana native token placeholder (wrapped SOL mint)
SOLANA_NATIVE_MINT = "So11111111111111111111111111111111111111112"


# =============================================================================
# CONSOLIDATED TOKEN REGISTRY
# =============================================================================
# Multi-chain token registry with addresses, decimals, and aliases.
# Addresses are lowercased for consistent comparison.

TOKEN_REGISTRY: Dict[int, Dict[str, Dict[str, Any]]] = {
    # -------------------------------------------------------------------------
    # Ethereum Mainnet (Chain ID: 1)
    # -------------------------------------------------------------------------
    1: {
        "ETH": {
            "symbol": "ETH",
            "name": "Ethereum",
            "address": NATIVE_PLACEHOLDER,
            "decimals": 18,
            "is_native": True,
            "coingecko_id": "ethereum",
            "aliases": {"eth", "ether", "ethereum", "native"},
        },
        "WETH": {
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "weth",
            "aliases": {"weth", "wrapped eth", "wrapped ether"},
        },
        "USDC": {
            "symbol": "USDC",
            "name": "USD Coin",
            "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin",
            "aliases": {"usdc", "usd coin", "circle usd"},
        },
        "USDT": {
            "symbol": "USDT",
            "name": "Tether USD",
            "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "tether",
            "aliases": {"usdt", "tether"},
        },
        "DAI": {
            "symbol": "DAI",
            "name": "Dai Stablecoin",
            "address": "0x6b175474e89094c44da98b954eedeac495271d0f",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "dai",
            "aliases": {"dai"},
        },
        "WBTC": {
            "symbol": "WBTC",
            "name": "Wrapped BTC",
            "address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
            "decimals": 8,
            "is_native": False,
            "coingecko_id": "wrapped-bitcoin",
            "aliases": {"wbtc", "wrapped btc", "wrapped bitcoin"},
        },
        "LINK": {
            "symbol": "LINK",
            "name": "ChainLink Token",
            "address": "0x514910771af9ca656af840dff83e8264ecf986ca",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "chainlink",
            "aliases": {"link", "chainlink"},
        },
        "UNI": {
            "symbol": "UNI",
            "name": "Uniswap",
            "address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "uniswap",
            "aliases": {"uni", "uniswap"},
        },
        "AAVE": {
            "symbol": "AAVE",
            "name": "Aave Token",
            "address": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "aave",
            "aliases": {"aave"},
        },
        "MKR": {
            "symbol": "MKR",
            "name": "Maker",
            "address": "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "maker",
            "aliases": {"mkr", "maker"},
        },
        "COMP": {
            "symbol": "COMP",
            "name": "Compound",
            "address": "0xc00e94cb662c3520282e6f5717214004a7f26888",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "compound-governance-token",
            "aliases": {"comp", "compound"},
        },
        "SHIB": {
            "symbol": "SHIB",
            "name": "Shiba Inu",
            "address": "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "shiba-inu",
            "aliases": {"shib", "shiba", "shiba inu"},
        },
        "MATIC": {
            "symbol": "MATIC",
            "name": "Polygon",
            "address": "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "matic-network",
            "aliases": {"matic", "polygon"},
        },
        "SUSHI": {
            "symbol": "SUSHI",
            "name": "SushiToken",
            "address": "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "sushi",
            "aliases": {"sushi", "sushiswap"},
        },
        "YFI": {
            "symbol": "YFI",
            "name": "yearn.finance",
            "address": "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "yearn-finance",
            "aliases": {"yfi", "yearn"},
        },
    },

    # -------------------------------------------------------------------------
    # Base (Chain ID: 8453)
    # -------------------------------------------------------------------------
    8453: {
        "ETH": {
            "symbol": "ETH",
            "name": "Ethereum",
            "address": NATIVE_PLACEHOLDER,
            "decimals": 18,
            "is_native": True,
            "coingecko_id": "ethereum",
            "aliases": {"eth", "ether", "native"},
        },
        "WETH": {
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "address": "0x4200000000000000000000000000000000000006",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "weth",
            "aliases": {"weth", "wrapped eth"},
        },
        "USDC": {
            "symbol": "USDC",
            "name": "USD Coin",
            "address": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin",
            "aliases": {"usdc", "usd coin"},
        },
        "USDbC": {
            "symbol": "USDbC",
            "name": "USD Base Coin (Bridged)",
            "address": "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "bridged-usd-coin-base",
            "aliases": {"usdbc", "bridged usdc"},
        },
        "DAI": {
            "symbol": "DAI",
            "name": "Dai Stablecoin",
            "address": "0x50c5725949a6f0c72e6c4a641f24049a917db0cb",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "dai",
            "aliases": {"dai"},
        },
        "cbETH": {
            "symbol": "cbETH",
            "name": "Coinbase Wrapped Staked ETH",
            "address": "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "coinbase-wrapped-staked-eth",
            "aliases": {"cbeth", "coinbase eth"},
        },
    },

    # -------------------------------------------------------------------------
    # Arbitrum One (Chain ID: 42161)
    # -------------------------------------------------------------------------
    42161: {
        "ETH": {
            "symbol": "ETH",
            "name": "Ethereum",
            "address": NATIVE_PLACEHOLDER,
            "decimals": 18,
            "is_native": True,
            "coingecko_id": "ethereum",
            "aliases": {"eth", "ether", "native"},
        },
        "WETH": {
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "address": "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "weth",
            "aliases": {"weth", "wrapped eth"},
        },
        "USDC": {
            "symbol": "USDC",
            "name": "USD Coin",
            "address": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin",
            "aliases": {"usdc", "usd coin"},
        },
        "USDC.e": {
            "symbol": "USDC.e",
            "name": "Bridged USDC",
            "address": "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin-ethereum-bridged",
            "aliases": {"usdc.e", "bridged usdc", "usdce"},
        },
        "USDT": {
            "symbol": "USDT",
            "name": "Tether USD",
            "address": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "tether",
            "aliases": {"usdt", "tether"},
        },
        "DAI": {
            "symbol": "DAI",
            "name": "Dai Stablecoin",
            "address": "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "dai",
            "aliases": {"dai"},
        },
        "ARB": {
            "symbol": "ARB",
            "name": "Arbitrum",
            "address": "0x912ce59144191c1204e64559fe8253a0e49e6548",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "arbitrum",
            "aliases": {"arb", "arbitrum"},
        },
        "WBTC": {
            "symbol": "WBTC",
            "name": "Wrapped BTC",
            "address": "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f",
            "decimals": 8,
            "is_native": False,
            "coingecko_id": "wrapped-bitcoin",
            "aliases": {"wbtc", "wrapped btc"},
        },
        "GMX": {
            "symbol": "GMX",
            "name": "GMX",
            "address": "0xfc5a1a6eb076a2c7ad06ed22c90d7e710e35ad0a",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "gmx",
            "aliases": {"gmx"},
        },
    },

    # -------------------------------------------------------------------------
    # Optimism (Chain ID: 10)
    # -------------------------------------------------------------------------
    10: {
        "ETH": {
            "symbol": "ETH",
            "name": "Ethereum",
            "address": NATIVE_PLACEHOLDER,
            "decimals": 18,
            "is_native": True,
            "coingecko_id": "ethereum",
            "aliases": {"eth", "ether", "native"},
        },
        "WETH": {
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "address": "0x4200000000000000000000000000000000000006",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "weth",
            "aliases": {"weth", "wrapped eth"},
        },
        "USDC": {
            "symbol": "USDC",
            "name": "USD Coin",
            "address": "0x0b2c639c533813f4aa9d7837caf62653d097ff85",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin",
            "aliases": {"usdc", "usd coin"},
        },
        "USDC.e": {
            "symbol": "USDC.e",
            "name": "Bridged USDC",
            "address": "0x7f5c764cbc14f9669b88837ca1490cca17c31607",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin-ethereum-bridged",
            "aliases": {"usdc.e", "bridged usdc", "usdce"},
        },
        "USDT": {
            "symbol": "USDT",
            "name": "Tether USD",
            "address": "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "tether",
            "aliases": {"usdt", "tether"},
        },
        "DAI": {
            "symbol": "DAI",
            "name": "Dai Stablecoin",
            "address": "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "dai",
            "aliases": {"dai"},
        },
        "OP": {
            "symbol": "OP",
            "name": "Optimism",
            "address": "0x4200000000000000000000000000000000000042",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "optimism",
            "aliases": {"op", "optimism"},
        },
        "WBTC": {
            "symbol": "WBTC",
            "name": "Wrapped BTC",
            "address": "0x68f180fcce6836688e9084f035309e29bf0a2095",
            "decimals": 8,
            "is_native": False,
            "coingecko_id": "wrapped-bitcoin",
            "aliases": {"wbtc", "wrapped btc"},
        },
    },

    # -------------------------------------------------------------------------
    # Polygon (Chain ID: 137)
    # -------------------------------------------------------------------------
    137: {
        "MATIC": {
            "symbol": "MATIC",
            "name": "Polygon",
            "address": NATIVE_PLACEHOLDER,
            "decimals": 18,
            "is_native": True,
            "coingecko_id": "matic-network",
            "aliases": {"matic", "polygon", "native"},
        },
        "WMATIC": {
            "symbol": "WMATIC",
            "name": "Wrapped Matic",
            "address": "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "wmatic",
            "aliases": {"wmatic", "wrapped matic"},
        },
        "WETH": {
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "address": "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "weth",
            "aliases": {"weth", "wrapped eth", "eth"},
        },
        "USDC": {
            "symbol": "USDC",
            "name": "USD Coin",
            "address": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin",
            "aliases": {"usdc", "usd coin"},
        },
        "USDC.e": {
            "symbol": "USDC.e",
            "name": "Bridged USDC",
            "address": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "usd-coin-ethereum-bridged",
            "aliases": {"usdc.e", "bridged usdc", "usdce"},
        },
        "USDT": {
            "symbol": "USDT",
            "name": "Tether USD",
            "address": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
            "decimals": 6,
            "is_native": False,
            "coingecko_id": "tether",
            "aliases": {"usdt", "tether"},
        },
        "DAI": {
            "symbol": "DAI",
            "name": "Dai Stablecoin",
            "address": "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "dai",
            "aliases": {"dai"},
        },
        "WBTC": {
            "symbol": "WBTC",
            "name": "Wrapped BTC",
            "address": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6",
            "decimals": 8,
            "is_native": False,
            "coingecko_id": "wrapped-bitcoin",
            "aliases": {"wbtc", "wrapped btc"},
        },
        "LINK": {
            "symbol": "LINK",
            "name": "ChainLink Token",
            "address": "0x53e0bca35ec356bd5dddfebbd1fc0fd03fabad39",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "chainlink",
            "aliases": {"link", "chainlink"},
        },
        "AAVE": {
            "symbol": "AAVE",
            "name": "Aave Token",
            "address": "0xd6df932a45c0f255f85145f286ea0b292b21c90b",
            "decimals": 18,
            "is_native": False,
            "coingecko_id": "aave",
            "aliases": {"aave"},
        },
    },
}


# =============================================================================
# SOLANA TOKEN REGISTRY
# =============================================================================
# Minimal static registry for top Solana tokens.
# For other tokens, the service falls back to Jupiter API.

SOLANA_TOKEN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "SOL": {
        "symbol": "SOL",
        "name": "Solana",
        "address": SOLANA_NATIVE_MINT,
        "decimals": 9,
        "is_native": True,
        "coingecko_id": "solana",
        "aliases": {"sol", "solana", "native"},
    },
    "USDC": {
        "symbol": "USDC",
        "name": "USD Coin",
        "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "usd-coin",
        "aliases": {"usdc"},
    },
    "USDT": {
        "symbol": "USDT",
        "name": "Tether USD",
        "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "tether",
        "aliases": {"usdt", "tether"},
    },
    "BONK": {
        "symbol": "BONK",
        "name": "Bonk",
        "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "decimals": 5,
        "is_native": False,
        "coingecko_id": "bonk",
        "aliases": {"bonk"},
    },
    "JUP": {
        "symbol": "JUP",
        "name": "Jupiter",
        "address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "jupiter-exchange-solana",
        "aliases": {"jup", "jupiter"},
    },
    "RAY": {
        "symbol": "RAY",
        "name": "Raydium",
        "address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "raydium",
        "aliases": {"ray", "raydium"},
    },
    "WIF": {
        "symbol": "WIF",
        "name": "dogwifhat",
        "address": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "dogwifcoin",
        "aliases": {"wif", "dogwifhat"},
    },
    "PYTH": {
        "symbol": "PYTH",
        "name": "Pyth Network",
        "address": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "pyth-network",
        "aliases": {"pyth"},
    },
    "JTO": {
        "symbol": "JTO",
        "name": "Jito",
        "address": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
        "decimals": 9,
        "is_native": False,
        "coingecko_id": "jito-governance-token",
        "aliases": {"jto", "jito"},
    },
    "ORCA": {
        "symbol": "ORCA",
        "name": "Orca",
        "address": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
        "decimals": 6,
        "is_native": False,
        "coingecko_id": "orca",
        "aliases": {"orca"},
    },
}


# Build lookup indexes at import time for fast resolution
# EVM indexes
_ADDRESS_INDEX: Dict[str, Dict[int, Dict[str, Any]]] = {}  # address -> chain_id -> token_data
_SYMBOL_INDEX: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> [token_data with chain_id]
_ALIAS_INDEX: Dict[str, List[Dict[str, Any]]] = {}  # alias -> [token_data with chain_id]

# Solana indexes
_SOLANA_ADDRESS_INDEX: Dict[str, Dict[str, Any]] = {}  # mint_address -> token_data
_SOLANA_SYMBOL_INDEX: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> [token_data]
_SOLANA_ALIAS_INDEX: Dict[str, List[Dict[str, Any]]] = {}  # alias -> [token_data]

# Build EVM indexes
for chain_id, tokens in TOKEN_REGISTRY.items():
    for symbol, data in tokens.items():
        token_with_chain = {**data, "chain_id": chain_id}

        # Index by address
        addr = data["address"].lower()
        if addr not in _ADDRESS_INDEX:
            _ADDRESS_INDEX[addr] = {}
        _ADDRESS_INDEX[addr][chain_id] = token_with_chain

        # Index by symbol
        sym_lower = symbol.lower()
        if sym_lower not in _SYMBOL_INDEX:
            _SYMBOL_INDEX[sym_lower] = []
        _SYMBOL_INDEX[sym_lower].append(token_with_chain)

        # Index by aliases
        for alias in data.get("aliases", set()):
            alias_lower = alias.lower()
            if alias_lower not in _ALIAS_INDEX:
                _ALIAS_INDEX[alias_lower] = []
            _ALIAS_INDEX[alias_lower].append(token_with_chain)

# Build Solana indexes
for symbol, data in SOLANA_TOKEN_REGISTRY.items():
    token_with_chain = {**data, "chain_id": SOLANA_CHAIN_ID}

    # Index by mint address (case-sensitive for Solana)
    _SOLANA_ADDRESS_INDEX[data["address"]] = token_with_chain

    # Index by symbol
    sym_lower = symbol.lower()
    if sym_lower not in _SOLANA_SYMBOL_INDEX:
        _SOLANA_SYMBOL_INDEX[sym_lower] = []
    _SOLANA_SYMBOL_INDEX[sym_lower].append(token_with_chain)

    # Index by aliases
    for alias in data.get("aliases", set()):
        alias_lower = alias.lower()
        if alias_lower not in _SOLANA_ALIAS_INDEX:
            _SOLANA_ALIAS_INDEX[alias_lower] = []
        _SOLANA_ALIAS_INDEX[alias_lower].append(token_with_chain)


# Spam detection patterns
SPAM_PATTERNS = [
    "visit", "claim", "reward", "airdrop", "free",
    "bonus", "gift", "$", "1000", "10000", "million",
    ".com", ".io", ".xyz", "http",
]


def _is_likely_spam(symbol: str, name: str) -> bool:
    """Detect likely spam/scam tokens."""
    symbol_lower = symbol.lower()
    name_lower = name.lower()

    for pattern in SPAM_PATTERNS:
        if pattern in symbol_lower or pattern in name_lower:
            return True

    if len(name) > 50:
        return True

    # Symbols with numbers (except common ones like USDC2, MP3)
    if any(char.isdigit() for char in symbol) and symbol.upper() not in {"3CRV", "FRAX3CRV"}:
        return True

    return False


def _is_valid_evm_address(query: str) -> bool:
    """Check if query looks like an EVM address."""
    return bool(re.match(r"^0x[a-fA-F0-9]{40}$", query))


def _normalize_query(query: str) -> str:
    """Normalize user input for matching."""
    return query.strip().lower()


def _detect_chain_from_query(query: str) -> Optional[ChainId]:
    """
    Attempt to detect which chain a query is targeting based on address format.

    Returns:
        - SOLANA_CHAIN_ID if query looks like a Solana address
        - None if query looks like an EVM address (could be any EVM chain)
        - None if query is a symbol (ambiguous)
    """
    query_clean = query.strip()

    # Solana mint address: Base58, 32-44 chars, no 0x prefix
    if is_valid_solana_address(query_clean):
        return SOLANA_CHAIN_ID

    # EVM address: 0x prefix, 40 hex chars - but could be any EVM chain
    if _is_valid_evm_address(query_clean):
        return None

    # Symbol "SOL" is only on Solana
    if query_clean.upper() == "SOL":
        return SOLANA_CHAIN_ID

    return None


class TokenResolutionService:
    """
    Centralized service for resolving token queries to canonical token data.

    Supports both EVM chains (via CoinGecko) and Solana (via Jupiter).

    Resolution priority:
    1. Exact contract/mint address match (highest confidence)
    2. Registry symbol match (chain-specific if chain_id provided)
    3. Registry alias match
    4. Portfolio context match (if provided)
    5. API search: CoinGecko for EVM, Jupiter for Solana (async)
    6. Fuzzy matching (lowest confidence)
    """

    def __init__(
        self,
        coingecko_provider: Optional[Any] = None,
        jupiter_provider: Optional[Any] = None,
    ):
        """
        Initialize the service.

        Args:
            coingecko_provider: Optional CoingeckoProvider instance for EVM API lookups.
            jupiter_provider: Optional JupiterProvider instance for Solana API lookups.
        """
        self._coingecko = coingecko_provider
        self._jupiter = jupiter_provider

    async def resolve(
        self,
        query: str,
        *,
        chain_id: Optional[ChainId] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        include_spam: bool = False,
        include_solana: bool = True,
    ) -> List[TokenMatch]:
        """
        Resolve a token query to a list of matches ranked by confidence.

        Args:
            query: Token symbol, name, alias, or contract/mint address.
            chain_id: Optional chain filter. If None, returns matches across all chains.
                      Use SOLANA_CHAIN_ID ("solana") for Solana-only search.
            portfolio: Optional user portfolio for context-aware resolution.
            top_k: Maximum number of matches to return.
            include_spam: Whether to include likely spam tokens.
            include_solana: Whether to include Solana tokens in results.

        Returns:
            List of TokenMatch objects sorted by confidence (descending).
        """
        query_normalized = _normalize_query(query)
        if not query_normalized:
            return []

        matches: List[TokenMatch] = []
        seen_canonical: Set[str] = set()

        def _add_match(match: TokenMatch) -> None:
            if match.canonical_id not in seen_canonical:
                if include_spam or not _is_likely_spam(match.symbol, match.name):
                    seen_canonical.add(match.canonical_id)
                    matches.append(match)

        # Detect if query indicates a specific chain
        detected_chain = _detect_chain_from_query(query)
        effective_chain = chain_id if chain_id is not None else detected_chain

        # Determine which chain types to search
        search_evm = effective_chain is None or is_evm_chain_id(effective_chain)
        search_solana = include_solana and (effective_chain is None or is_solana_chain(effective_chain))

        # 1. Check if query is a contract/mint address
        if is_valid_solana_address(query.strip()) and search_solana:
            # Solana mint address
            solana_addr_matches = self._resolve_solana_by_address(query.strip())
            for m in solana_addr_matches:
                _add_match(m)
        elif _is_valid_evm_address(query_normalized) and search_evm:
            # EVM contract address
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            addr_matches = self._resolve_by_address(query_normalized, evm_chain)
            for m in addr_matches:
                _add_match(m)

        # 2. Check registry by exact symbol (EVM)
        if search_evm:
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            symbol_matches = self._resolve_by_symbol(query_normalized, evm_chain)
            for m in symbol_matches:
                _add_match(m)

        # 3. Check Solana registry by symbol
        if search_solana:
            solana_symbol_matches = self._resolve_solana_by_symbol(query_normalized)
            for m in solana_symbol_matches:
                _add_match(m)

        # 4. Check registry by alias (EVM)
        if search_evm:
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            alias_matches = self._resolve_by_alias(query_normalized, evm_chain)
            for m in alias_matches:
                _add_match(m)

        # 5. Check Solana registry by alias
        if search_solana:
            solana_alias_matches = self._resolve_solana_by_alias(query_normalized)
            for m in solana_alias_matches:
                _add_match(m)

        # 6. Check portfolio context
        if portfolio:
            portfolio_matches = self._resolve_by_portfolio(query_normalized, portfolio, effective_chain)
            for m in portfolio_matches:
                _add_match(m)

        # 7. Try API search if we don't have high-confidence matches
        needs_api_search = not matches or matches[0].confidence < CONFIDENCE_REGISTRY_ALIAS

        if needs_api_search:
            # CoinGecko for EVM
            if search_evm and self._coingecko:
                evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
                coingecko_matches = await self._resolve_by_coingecko(query, evm_chain)
                for m in coingecko_matches:
                    _add_match(m)

            # Jupiter for Solana
            if search_solana and self._jupiter:
                jupiter_matches = await self._resolve_by_jupiter(query)
                for m in jupiter_matches:
                    _add_match(m)

        # Sort by confidence descending, then by symbol alphabetically
        matches.sort(key=lambda m: (-m.confidence, m.symbol))

        return matches[:top_k]

    def resolve_sync(
        self,
        query: str,
        *,
        chain_id: Optional[ChainId] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        include_solana: bool = True,
    ) -> List[TokenMatch]:
        """
        Synchronous resolution using only static registry (no API calls).

        Use this for fast resolution when async is not available.
        """
        query_normalized = _normalize_query(query)
        if not query_normalized:
            return []

        matches: List[TokenMatch] = []
        seen_canonical: Set[str] = set()

        def _add_match(match: TokenMatch) -> None:
            if match.canonical_id not in seen_canonical:
                if not _is_likely_spam(match.symbol, match.name):
                    seen_canonical.add(match.canonical_id)
                    matches.append(match)

        # Detect chain from query
        detected_chain = _detect_chain_from_query(query)
        effective_chain = chain_id if chain_id is not None else detected_chain

        search_evm = effective_chain is None or is_evm_chain_id(effective_chain)
        search_solana = include_solana and (effective_chain is None or is_solana_chain(effective_chain))

        # Address resolution
        if is_valid_solana_address(query.strip()) and search_solana:
            for m in self._resolve_solana_by_address(query.strip()):
                _add_match(m)
        elif _is_valid_evm_address(query_normalized) and search_evm:
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            for m in self._resolve_by_address(query_normalized, evm_chain):
                _add_match(m)

        # Symbol resolution
        if search_evm:
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            for m in self._resolve_by_symbol(query_normalized, evm_chain):
                _add_match(m)

        if search_solana:
            for m in self._resolve_solana_by_symbol(query_normalized):
                _add_match(m)

        # Alias resolution
        if search_evm:
            evm_chain = effective_chain if is_evm_chain_id(effective_chain) else None
            for m in self._resolve_by_alias(query_normalized, evm_chain):
                _add_match(m)

        if search_solana:
            for m in self._resolve_solana_by_alias(query_normalized):
                _add_match(m)

        # Portfolio context
        if portfolio:
            for m in self._resolve_by_portfolio(query_normalized, portfolio, effective_chain):
                _add_match(m)

        matches.sort(key=lambda m: (-m.confidence, m.symbol))
        return matches[:top_k]

    def _resolve_by_address(
        self,
        address: str,
        chain_id: Optional[int],
    ) -> List[TokenMatch]:
        """Resolve by exact contract address."""
        address_lower = address.lower()
        matches: List[TokenMatch] = []

        if address_lower in _ADDRESS_INDEX:
            chain_tokens = _ADDRESS_INDEX[address_lower]

            if chain_id is not None and chain_id in chain_tokens:
                # Exact chain match
                data = chain_tokens[chain_id]
                matches.append(self._make_match(data, CONFIDENCE_EXACT_ADDRESS, ResolutionSource.EXACT_ADDRESS))
            else:
                # Return all chains with this address
                for cid, data in chain_tokens.items():
                    if chain_id is None or cid == chain_id:
                        matches.append(self._make_match(data, CONFIDENCE_EXACT_ADDRESS, ResolutionSource.EXACT_ADDRESS))

        return matches

    def _resolve_by_symbol(
        self,
        symbol: str,
        chain_id: Optional[int],
    ) -> List[TokenMatch]:
        """Resolve by exact symbol match."""
        matches: List[TokenMatch] = []
        symbol_lower = symbol.lower()

        if symbol_lower in _SYMBOL_INDEX:
            for data in _SYMBOL_INDEX[symbol_lower]:
                if chain_id is None or data["chain_id"] == chain_id:
                    matches.append(self._make_match(data, CONFIDENCE_REGISTRY_EXACT, ResolutionSource.REGISTRY))

        return matches

    def _resolve_by_alias(
        self,
        alias: str,
        chain_id: Optional[int],
    ) -> List[TokenMatch]:
        """Resolve by alias match."""
        matches: List[TokenMatch] = []
        alias_lower = alias.lower()

        if alias_lower in _ALIAS_INDEX:
            for data in _ALIAS_INDEX[alias_lower]:
                if chain_id is None or data["chain_id"] == chain_id:
                    matches.append(self._make_match(data, CONFIDENCE_REGISTRY_ALIAS, ResolutionSource.REGISTRY))

        return matches

    def _resolve_by_portfolio(
        self,
        query: str,
        portfolio: Dict[str, Any],
        chain_id: Optional[int],
    ) -> List[TokenMatch]:
        """Resolve using user's portfolio holdings."""
        matches: List[TokenMatch] = []
        query_lower = query.lower()

        # Portfolio expected format: {"tokens": [{"symbol": "ETH", "address": "0x...", ...}]}
        tokens = portfolio.get("tokens", [])
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue

                token_symbol = str(token.get("symbol", "")).lower()
                token_name = str(token.get("name", "")).lower()
                token_address = str(token.get("address", "")).lower()
                token_chain = token.get("chain_id")

                if chain_id is not None and token_chain != chain_id:
                    continue

                # Match by symbol, name, or address
                if query_lower in (token_symbol, token_name) or query_lower == token_address:
                    matches.append(TokenMatch(
                        chain_id=token_chain or 1,
                        address=token.get("address", NATIVE_PLACEHOLDER),
                        symbol=token.get("symbol", "UNKNOWN"),
                        name=token.get("name", "Unknown Token"),
                        decimals=token.get("decimals", 18),
                        confidence=CONFIDENCE_PORTFOLIO,
                        source=ResolutionSource.PORTFOLIO,
                        is_native=token.get("is_native", False),
                    ))

        return matches

    async def _resolve_by_coingecko(
        self,
        query: str,
        chain_id: Optional[int],
    ) -> List[TokenMatch]:
        """Resolve using CoinGecko API search (EVM only)."""
        if not self._coingecko:
            return []

        matches: List[TokenMatch] = []

        try:
            # First check if it's a contract address
            if _is_valid_evm_address(query):
                # Try to get token info by contract
                info = await self._coingecko.get_token_info(query)
                if info and info.get("symbol"):
                    matches.append(TokenMatch(
                        chain_id=chain_id or 1,
                        address=query.lower(),
                        symbol=info.get("symbol", "").upper(),
                        name=info.get("name", ""),
                        decimals=info.get("decimals", 18),
                        confidence=CONFIDENCE_COINGECKO_EXACT,
                        source=ResolutionSource.COINGECKO,
                        coingecko_id=info.get("id"),
                        logo_url=info.get("image"),
                    ))
            else:
                # Search by symbol/name
                results = await self._coingecko.search_coins(query, limit=5)
                for coin in results:
                    coin_symbol = str(coin.get("symbol", "")).upper()
                    coin_name = coin.get("name", "")
                    coin_id = coin.get("id", "")

                    # Higher confidence for exact symbol match
                    if coin_symbol.lower() == query.lower():
                        confidence = CONFIDENCE_COINGECKO_EXACT
                    else:
                        confidence = CONFIDENCE_COINGECKO_SEARCH

                    matches.append(TokenMatch(
                        chain_id=chain_id or 1,
                        address=NATIVE_PLACEHOLDER,  # Address unknown from search
                        symbol=coin_symbol,
                        name=coin_name,
                        decimals=18,  # Default, would need another API call
                        confidence=confidence,
                        source=ResolutionSource.COINGECKO,
                        coingecko_id=coin_id,
                        logo_url=coin.get("thumb"),
                    ))
        except Exception:
            # Silently fail API lookups
            pass

        return matches

    # -------------------------------------------------------------------------
    # Solana Resolution Methods
    # -------------------------------------------------------------------------

    def _resolve_solana_by_address(self, mint_address: str) -> List[TokenMatch]:
        """Resolve Solana token by mint address from static registry."""
        matches: List[TokenMatch] = []

        if mint_address in _SOLANA_ADDRESS_INDEX:
            data = _SOLANA_ADDRESS_INDEX[mint_address]
            matches.append(self._make_match(data, CONFIDENCE_EXACT_ADDRESS, ResolutionSource.EXACT_ADDRESS))

        return matches

    def _resolve_solana_by_symbol(self, symbol: str) -> List[TokenMatch]:
        """Resolve Solana token by symbol from static registry."""
        matches: List[TokenMatch] = []
        symbol_lower = symbol.lower()

        if symbol_lower in _SOLANA_SYMBOL_INDEX:
            for data in _SOLANA_SYMBOL_INDEX[symbol_lower]:
                matches.append(self._make_match(data, CONFIDENCE_REGISTRY_EXACT, ResolutionSource.REGISTRY))

        return matches

    def _resolve_solana_by_alias(self, alias: str) -> List[TokenMatch]:
        """Resolve Solana token by alias from static registry."""
        matches: List[TokenMatch] = []
        alias_lower = alias.lower()

        if alias_lower in _SOLANA_ALIAS_INDEX:
            for data in _SOLANA_ALIAS_INDEX[alias_lower]:
                matches.append(self._make_match(data, CONFIDENCE_REGISTRY_ALIAS, ResolutionSource.REGISTRY))

        return matches

    async def _resolve_by_jupiter(self, query: str) -> List[TokenMatch]:
        """Resolve Solana token using Jupiter API."""
        if not self._jupiter:
            return []

        matches: List[TokenMatch] = []

        try:
            # Check if it's a mint address
            if is_valid_solana_address(query.strip()):
                token = await self._jupiter.get_token_by_mint(query.strip())
                if token:
                    matches.append(TokenMatch(
                        chain_id=SOLANA_CHAIN_ID,
                        address=token.address,
                        symbol=token.symbol,
                        name=token.name,
                        decimals=token.decimals,
                        confidence=CONFIDENCE_JUPITER_EXACT,
                        source=ResolutionSource.JUPITER,
                        coingecko_id=token.coingecko_id,
                        logo_url=token.logo_uri,
                        tags=token.tags or [],
                    ))
            else:
                # Search by symbol
                results = await self._jupiter.search_by_symbol(query, limit=5, exact_match=False)
                for token in results:
                    # Higher confidence for exact symbol match
                    if token.symbol.lower() == query.lower():
                        confidence = CONFIDENCE_JUPITER_EXACT
                    else:
                        confidence = CONFIDENCE_JUPITER_SEARCH

                    matches.append(TokenMatch(
                        chain_id=SOLANA_CHAIN_ID,
                        address=token.address,
                        symbol=token.symbol,
                        name=token.name,
                        decimals=token.decimals,
                        confidence=confidence,
                        source=ResolutionSource.JUPITER,
                        coingecko_id=token.coingecko_id,
                        logo_url=token.logo_uri,
                        tags=token.tags or [],
                    ))
        except Exception:
            # Silently fail API lookups
            pass

        return matches

    def _make_match(self, data: Dict[str, Any], confidence: float, source: ResolutionSource) -> TokenMatch:
        """Create a TokenMatch from registry data."""
        return TokenMatch(
            chain_id=data["chain_id"],
            address=data["address"],
            symbol=data["symbol"],
            name=data["name"],
            decimals=data["decimals"],
            confidence=confidence,
            source=source,
            is_native=data.get("is_native", False),
            coingecko_id=data.get("coingecko_id"),
            aliases=set(data.get("aliases", [])),
        )

    def is_ambiguous(self, matches: List[TokenMatch], threshold: float = AMBIGUITY_THRESHOLD) -> bool:
        """
        Check if resolution is ambiguous.

        Returns True if:
        - No matches found
        - Top match confidence is below threshold
        - Multiple high-confidence matches exist (same confidence within 0.1)
        """
        if not matches:
            return True

        if matches[0].confidence < threshold:
            return True

        # Check for multiple equally-confident matches
        if len(matches) > 1:
            top_confidence = matches[0].confidence
            similar_count = sum(1 for m in matches if abs(m.confidence - top_confidence) < 0.1)
            if similar_count > 1:
                return True

        return False

    def get_ambiguity_result(self, query: str, matches: List[TokenMatch]) -> AmbiguityResult:
        """Create an AmbiguityResult for user clarification."""
        if not matches:
            reason = f"No tokens found matching '{query}'"
        elif matches[0].confidence < AMBIGUITY_THRESHOLD:
            reason = f"Low confidence match for '{query}'"
        else:
            reason = f"Multiple tokens match '{query}'"

        return AmbiguityResult(query=query, matches=matches, reason=reason)

    def get_best_match(self, matches: List[TokenMatch]) -> Optional[TokenMatch]:
        """Get the highest confidence match, or None if ambiguous."""
        if not matches or self.is_ambiguous(matches):
            return None
        return matches[0]


# Module-level singleton for convenience
_default_service: Optional[TokenResolutionService] = None


def get_token_resolution_service(
    coingecko_provider: Optional[Any] = None,
    jupiter_provider: Optional[Any] = None,
) -> TokenResolutionService:
    """
    Get or create the default TokenResolutionService instance.

    Args:
        coingecko_provider: Optional CoingeckoProvider for EVM token lookups.
        jupiter_provider: Optional JupiterProvider for Solana token lookups.

    Returns:
        TokenResolutionService instance.
    """
    global _default_service
    if _default_service is None or coingecko_provider is not None or jupiter_provider is not None:
        _default_service = TokenResolutionService(
            coingecko_provider=coingecko_provider,
            jupiter_provider=jupiter_provider,
        )
    return _default_service


__all__ = [
    "TokenMatch",
    "TokenResolutionService",
    "ResolutionSource",
    "AmbiguityResult",
    "get_token_resolution_service",
    "TOKEN_REGISTRY",
    "SOLANA_TOKEN_REGISTRY",
    "SOLANA_CHAIN_ID",
    "AMBIGUITY_THRESHOLD",
]
