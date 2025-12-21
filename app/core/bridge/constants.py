"""Constants and metadata for bridge orchestration."""

from typing import Any, Dict, Set, Union, Literal

# ChainId type for type hints (mirrors core/chain_types.py)
ChainId = Union[int, Literal["solana"]]

BRIDGE_KEYWORDS = (
    'bridge',
    'bridging',
    'move to',
    'send to',
    'transfer to',
    'port to',
)

BRIDGE_FOLLOWUP_KEYWORDS = (
    'quote',
    'retry',
    'try again',
    'again',
    'get the quote',
    'get quote',
    'refresh quote',
    'refresh bridge quote',
    'refresh',
    'do it',
    'go ahead',
    'please',
)

CHAIN_METADATA: Dict[int, Dict[str, Any]] = {
    1: {
        'name': 'Ethereum',
        'aliases': ['ethereum', 'eth', 'mainnet', 'main net', 'main-net', 'ethereum mainnet', 'layer1', 'layer 1', 'layer-1', 'l1'],
        'native_symbol': 'ETH',
        'native_token': '0x0000000000000000000000000000000000000000',
        'native_decimals': 18,
    },
    8453: {
        'name': 'Base',
        'aliases': ['base', 'base mainnet'],
        'native_symbol': 'ETH',
        'native_token': '0x0000000000000000000000000000000000000000',
        'native_decimals': 18,
    },
    42161: {
        'name': 'Arbitrum',
        'aliases': ['arbitrum', 'arb'],
        'native_symbol': 'ETH',
        'native_token': '0x0000000000000000000000000000000000000000',
        'native_decimals': 18,
    },
    10: {
        'name': 'Optimism',
        'aliases': ['optimism', 'op'],
        'native_symbol': 'ETH',
        'native_token': '0x0000000000000000000000000000000000000000',
        'native_decimals': 18,
    },
    137: {
        'name': 'Polygon',
        'aliases': ['polygon', 'matic', 'matic pos'],
        'native_symbol': 'MATIC',
        'native_token': '0x0000000000000000000000000000000000000000',
        'native_decimals': 18,
        'chain_type': 'evm',
    },
    # Non-EVM chains use string identifiers
    "solana": {
        'name': 'Solana',
        'aliases': ['solana', 'sol'],
        'native_symbol': 'SOL',
        'native_token': 'So11111111111111111111111111111111111111112',
        'native_decimals': 9,
        'chain_type': 'solana',
    },
}

CHAIN_ALIAS_TO_ID: Dict[str, ChainId] = {
    alias: chain_id
    for chain_id, details in CHAIN_METADATA.items()
    for alias in details.get('aliases', [])
}

DEFAULT_CHAIN_NAME_TO_ID: Dict[str, ChainId] = {
    details['name'].lower(): chain_id for chain_id, details in CHAIN_METADATA.items()
}

NATIVE_PLACEHOLDER = '0x0000000000000000000000000000000000000000'
USD_UNITS: Set[str] = {'usd', 'dollar', 'dollars', 'usdc', 'buck', 'bucks'}
ETH_UNITS: Set[str] = {'eth', 'weth'}
BRIDGE_SOURCE = {'name': 'Relay', 'url': 'https://relay.link'}
