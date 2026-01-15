"""Constants for bridge orchestration.

Note: Chain metadata is now fetched dynamically from Relay API via ChainRegistry.
See chain_registry.py for the dynamic chain support implementation.
"""

from typing import Set

# Keywords for detecting bridge intent in user messages
BRIDGE_KEYWORDS = (
    'bridge',
    'bridging',
    'move to',
    'send to',
    'transfer to',
    'port to',
)

# Keywords for detecting follow-up messages in bridge conversations
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

# Native token placeholder address (used for ETH on most chains)
NATIVE_PLACEHOLDER = '0x0000000000000000000000000000000000000000'

# Units for amount parsing
USD_UNITS: Set[str] = {'usd', 'dollar', 'dollars', 'usdc', 'buck', 'bucks'}
ETH_UNITS: Set[str] = {'eth', 'weth'}

# Bridge provider attribution
BRIDGE_SOURCE = {'name': 'Relay', 'url': 'https://relay.link'}
