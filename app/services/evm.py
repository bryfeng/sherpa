"""Utilities for working with EVM-compatible chains."""

from __future__ import annotations

from typing import Dict, Optional, Set

from ..core.bridge.chain_registry import get_registry_sync


def get_evm_chain_ids() -> Set[int]:
    """Get the set of supported EVM chain IDs from the registry.

    Returns all integer chain IDs from the dynamic chain registry.
    """
    registry = get_registry_sync()
    return {
        chain_id
        for chain_id in registry._chains.keys()
        if isinstance(chain_id, int)
    }


# For backwards compatibility, provide as a class that acts like a set
class _DynamicEVMChainSet:
    """Dynamic set of EVM chain IDs that reads from the registry."""

    def __contains__(self, item):
        return item in get_evm_chain_ids()

    def __iter__(self):
        return iter(get_evm_chain_ids())

    def __len__(self):
        return len(get_evm_chain_ids())


EVM_CHAIN_IDS = _DynamicEVMChainSet()

# Mapping from Coingecko platform slug → EVM chain ID.
# Only include platforms that Relay currently supports; Solana and other
# non-EVM chains will be introduced later.
COINGECKO_PLATFORM_TO_CHAIN_ID: Dict[str, int] = {
    'ethereum': 1,
    'arbitrum-one': 42161,
    'optimistic-ethereum': 10,
    'polygon-pos': 137,
    'base': 8453,
}


def is_evm_chain(chain_id: int) -> bool:
    """Return ``True`` if the chain ID is part of the supported EVM set."""

    return chain_id in EVM_CHAIN_IDS


def chain_id_from_coingecko_platform(platform: str) -> Optional[int]:
    """Map a Coingecko platform slug to a supported EVM chain ID.

    Returns ``None`` when the platform is either unknown or not EVM-compatible.
    """

    return COINGECKO_PLATFORM_TO_CHAIN_ID.get(platform.lower())


__all__ = [
    'EVM_CHAIN_IDS',
    'COINGECKO_PLATFORM_TO_CHAIN_ID',
    'is_evm_chain',
    'chain_id_from_coingecko_platform',
]

