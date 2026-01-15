"""Constants and token metadata for swap orchestration."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from ..bridge.constants import NATIVE_PLACEHOLDER
from ..bridge.chain_registry import (
    get_registry_sync,
    get_chain_alias_to_id,
    get_chain_metadata,
    ChainId,
)


# Backwards-compatible lazy accessors for chain data
# These are functions that return the current registry state
def _get_chain_alias_to_id() -> Dict[str, ChainId]:
    """Get alias -> chain_id mapping from registry."""
    return get_chain_alias_to_id()


def _get_chain_metadata() -> Dict[ChainId, Dict[str, Any]]:
    """Get chain_id -> metadata mapping from registry."""
    return get_chain_metadata()


def _get_default_chain_name_to_id() -> Dict[str, ChainId]:
    """Get chain name -> chain_id mapping from registry."""
    registry = get_registry_sync()
    result: Dict[str, ChainId] = {}
    for chain_id, chain_data in registry._chains.items():
        name = (chain_data.get("displayName") or chain_data.get("name") or "").lower()
        if name:
            result[name] = chain_id
    return result


# For backwards compatibility, provide these as module-level names
# Note: These are now dynamically generated from the registry
# Code should migrate to using get_registry_sync() directly
class _LazyChainDict:
    """Lazy dict-like accessor for chain data from registry."""

    def __init__(self, getter):
        self._getter = getter

    def __getitem__(self, key):
        return self._getter()[key]

    def __contains__(self, key):
        return key in self._getter()

    def get(self, key, default=None):
        return self._getter().get(key, default)

    def items(self):
        return self._getter().items()

    def keys(self):
        return self._getter().keys()

    def values(self):
        return self._getter().values()

    def __iter__(self):
        return iter(self._getter())


CHAIN_ALIAS_TO_ID = _LazyChainDict(_get_chain_alias_to_id)
CHAIN_METADATA = _LazyChainDict(_get_chain_metadata)
DEFAULT_CHAIN_NAME_TO_ID = _LazyChainDict(_get_default_chain_name_to_id)

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
