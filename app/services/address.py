"""Helpers for normalizing chain identifiers and validating wallet addresses."""

from __future__ import annotations

import re
from functools import lru_cache

_BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
_HEX_ALPHABET = set("0123456789abcdefABCDEF")

_EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_SUI_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

_CHAIN_ALIASES = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "mainnet": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "sui": "sui",
    "matic": "polygon",
    "polygon": "polygon",
    "base-mainnet": "base",
    "base": "base",
}

# Subset of chains we currently support for portfolio aggregation.
_SUPPORTED_EVM_CHAINS = {
    "ethereum",
    "mainnet",
    "polygon",
    "base",
}


def normalize_chain(chain: str | None) -> str:
    """Collapse user-provided chain identifiers into canonical slugs."""

    if not chain:
        return "ethereum"
    canonical = _CHAIN_ALIASES.get(chain.lower().strip())
    return canonical or chain.lower().strip()


def is_supported_chain(chain: str) -> bool:
    """Return True if the chain is one we attempt to serve today."""

    if chain in _SUPPORTED_EVM_CHAINS:
        return True
    return chain in {"solana"}


def is_evm_chain(chain: str) -> bool:
    return chain in _SUPPORTED_EVM_CHAINS


@lru_cache(maxsize=128)
def is_valid_solana_address(address: str) -> bool:
    if not address:
        return False
    length = len(address)
    if length < 32 or length > 44:
        return False
    return all(ch in _BASE58_ALPHABET for ch in address)


def is_valid_sui_address(address: str) -> bool:
    if not address.startswith("0x"):
        return False
    hex_part = address[2:]
    if not hex_part or len(hex_part) > 64:
        return False
    if len(hex_part) % 2 != 0:
        return False
    return all(ch in _HEX_ALPHABET for ch in hex_part)


def is_valid_address_for_chain(address: str, chain: str) -> bool:
    if not address:
        return False
    if is_evm_chain(chain):
        return bool(_EVM_ADDRESS_RE.fullmatch(address))
    if chain == "solana":
        return is_valid_solana_address(address)
    if chain == "sui":
        return is_valid_sui_address(address)
    # Fallback to strict EVM checksum pattern for unknown chains.
    return bool(_EVM_ADDRESS_RE.fullmatch(address))


__all__ = [
    "normalize_chain",
    "is_supported_chain",
    "is_evm_chain",
    "is_valid_address_for_chain",
    "is_valid_solana_address",
    "is_valid_sui_address",
]
