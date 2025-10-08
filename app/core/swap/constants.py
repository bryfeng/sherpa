"""Constants and token metadata for swap orchestration."""

from __future__ import annotations

from typing import Dict, Tuple

from ..bridge.constants import (
    CHAIN_ALIAS_TO_ID,
    CHAIN_METADATA,
    DEFAULT_CHAIN_NAME_TO_ID,
    NATIVE_PLACEHOLDER,
)

SWAP_KEYWORDS: Tuple[str, ...] = (
    'swap',
    'trade',
    'convert',
    'exchange',
)

SWAP_FOLLOWUP_KEYWORDS: Tuple[str, ...] = (
    'refresh',
    'refresh quote',
    'refresh swap quote',
    'quote',
    'retry',
    'try again',
    're-quote',
    'again',
)

SWAP_SOURCE = {'name': 'Relay', 'url': 'https://relay.link'}

# Minimal token registry keyed by chain ID → token symbol → metadata.
# Addresses intentionally lowercased to simplify comparisons.
TOKEN_REGISTRY: Dict[int, Dict[str, Dict[str, object]]] = {
    1: {  # Ethereum mainnet
        'ETH': {
            'symbol': 'ETH',
            'address': NATIVE_PLACEHOLDER,
            'decimals': 18,
            'is_native': True,
            'aliases': {'eth', 'native'},
        },
        'WETH': {
            'symbol': 'WETH',
            'address': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
            'decimals': 18,
            'is_native': False,
            'aliases': {'weth', 'wrapped eth'},
        },
        'USDC': {
            'symbol': 'USDC',
            'address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
            'decimals': 6,
            'is_native': False,
            'aliases': {'usdc', 'usd coin'},
        },
        'USDT': {
            'symbol': 'USDT',
            'address': '0xdac17f958d2ee523a2206206994597c13d831ec7',
            'decimals': 6,
            'is_native': False,
            'aliases': {'usdt', 'tether'},
        },
        'DAI': {
            'symbol': 'DAI',
            'address': '0x6b175474e89094c44da98b954eedeac495271d0f',
            'decimals': 18,
            'is_native': False,
            'aliases': {'dai'},
        },
        'WBTC': {
            'symbol': 'WBTC',
            'address': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
            'decimals': 8,
            'is_native': False,
            'aliases': {'wbtc', 'wrapped btc', 'btc'},
        },
    },
}

# Flattened alias map built at import time
TOKEN_ALIAS_MAP: Dict[int, Dict[str, str]] = {}
for chain_id, entries in TOKEN_REGISTRY.items():
    alias_map: Dict[str, str] = {}
    for symbol, metadata in entries.items():
        alias_map[symbol.lower()] = symbol
        for alias in metadata.get('aliases', set()):  # type: ignore[assignment]
            alias_map[str(alias).lower()] = symbol
    TOKEN_ALIAS_MAP[chain_id] = alias_map

GLOBAL_TOKEN_ALIASES: Dict[str, str] = {}
for alias_map in TOKEN_ALIAS_MAP.values():
    for alias, symbol in alias_map.items():
        GLOBAL_TOKEN_ALIASES.setdefault(alias, symbol)

__all__ = [
    'SWAP_KEYWORDS',
    'SWAP_FOLLOWUP_KEYWORDS',
    'SWAP_SOURCE',
    'TOKEN_REGISTRY',
    'TOKEN_ALIAS_MAP',
    'GLOBAL_TOKEN_ALIASES',
    'CHAIN_ALIAS_TO_ID',
    'CHAIN_METADATA',
    'DEFAULT_CHAIN_NAME_TO_ID',
    'NATIVE_PLACEHOLDER',
]
