"""
Chain identification types and utilities.

This module provides a unified ChainId type that supports both:
- EVM chains (identified by integer chain IDs: 1, 137, 8453, etc.)
- Non-EVM chains (identified by string literals: "solana", etc.)

This design allows for type-safe chain handling while minimizing disruption
to existing EVM-focused code.
"""

from __future__ import annotations

from typing import Literal, Union

# ChainId represents any supported blockchain
# - EVM chains use their standard integer chain IDs (e.g., 1 for Ethereum)
# - Non-EVM chains use string identifiers (e.g., "solana")
#
# Future extensibility: add more Literal types as needed
# ChainId = Union[int, Literal["solana", "cosmos-hub", "bitcoin"]]
ChainId = Union[int, Literal["solana"]]

# Sentinel values for non-EVM chains
SOLANA_CHAIN_ID: Literal["solana"] = "solana"

# Default chain when none specified
DEFAULT_CHAIN_ID: int = 1  # Ethereum mainnet


def is_solana_chain(chain_id: ChainId) -> bool:
    """Check if the chain ID represents Solana."""
    return chain_id == SOLANA_CHAIN_ID


def is_evm_chain_id(chain_id: ChainId) -> bool:
    """Check if the chain ID represents an EVM-compatible chain."""
    return isinstance(chain_id, int)


def normalize_to_chain_id(chain: str | int | None) -> ChainId:
    """
    Convert user input to a canonical ChainId.

    Args:
        chain: Chain identifier as string (e.g., "ethereum", "sol", "base")
               or integer (e.g., 1, 137) or None.

    Returns:
        Canonical ChainId (int for EVM, "solana" for Solana).

    Raises:
        ValueError: If the chain identifier is not recognized.

    Examples:
        >>> normalize_to_chain_id("ethereum")
        1
        >>> normalize_to_chain_id("sol")
        "solana"
        >>> normalize_to_chain_id(137)
        137
        >>> normalize_to_chain_id(None)
        1
    """
    if chain is None:
        return DEFAULT_CHAIN_ID

    if isinstance(chain, int):
        return chain

    chain_lower = chain.lower().strip()

    # Solana aliases
    if chain_lower in ("sol", "solana"):
        return SOLANA_CHAIN_ID

    # EVM chain aliases -> chain ID mapping
    # Use dynamic ChainRegistry instead of hardcoded constants
    from ..core.bridge.chain_registry import get_registry_sync

    registry = get_registry_sync()
    resolved = registry.get_chain_id(chain_lower)
    if resolved is not None:
        return resolved

    # Try parsing as integer
    try:
        return int(chain)
    except ValueError:
        pass

    raise ValueError(f"Unknown chain identifier: {chain!r}")


def chain_id_to_name(chain_id: ChainId) -> str:
    """
    Get the human-readable name for a chain ID.

    Args:
        chain_id: The chain identifier.

    Returns:
        Human-readable chain name.
    """
    if chain_id == SOLANA_CHAIN_ID:
        return "Solana"

    # Use dynamic ChainRegistry instead of hardcoded constants
    from ..core.bridge.chain_registry import get_registry_sync

    return get_registry_sync().get_chain_name(chain_id)


__all__ = [
    "ChainId",
    "SOLANA_CHAIN_ID",
    "DEFAULT_CHAIN_ID",
    "is_solana_chain",
    "is_evm_chain_id",
    "normalize_to_chain_id",
    "chain_id_to_name",
]
