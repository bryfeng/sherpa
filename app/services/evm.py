"""Utilities for working with EVM-compatible chains."""

from __future__ import annotations

from typing import Dict, Optional, Set

from ..core.bridge.constants import CHAIN_METADATA

# Explicit allow-list of EVM chain IDs currently supported by Sherpa services.
EVM_CHAIN_IDS: Set[int] = set(CHAIN_METADATA.keys())

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

