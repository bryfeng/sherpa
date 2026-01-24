"""
UserOperation calldata builders.
"""

from __future__ import annotations

from typing import Optional

from eth_utils import keccak

from app.config import settings


def _strip_0x(value: str) -> str:
    return value[2:] if value.startswith("0x") else value


def _encode_uint(value: int) -> str:
    if value < 0:
        raise ValueError("Value must be non-negative")
    return hex(value)[2:].rjust(64, "0")


def _encode_address(address: str) -> str:
    addr = _strip_0x(address).lower()
    if len(addr) != 40:
        raise ValueError(f"Invalid address length: {address}")
    return addr.rjust(64, "0")


def _encode_bytes(data: str) -> str:
    hex_data = _strip_0x(data)
    if len(hex_data) % 2 != 0:
        raise ValueError("Byte data must have an even-length hex string")
    data_len = len(hex_data) // 2
    padded_len = ((data_len + 31) // 32) * 32
    padding = "0" * ((padded_len - data_len) * 2)
    return _encode_uint(data_len) + hex_data + padding


def _selector_from_signature(signature: str) -> str:
    selector = keccak(text=signature)[:4].hex()
    return f"0x{selector}"


def get_execute_selector(
    signature: Optional[str] = None,
    selector_override: Optional[str] = None,
) -> str:
    selector_override = selector_override or settings.erc4337_account_execute_selector
    if selector_override:
        if not selector_override.startswith("0x") or len(selector_override) != 10:
            raise ValueError("Execute selector override must be 4 bytes (0x........)")
        return selector_override

    signature = signature or settings.erc4337_account_execute_signature
    return _selector_from_signature(signature)


def build_execute_call_data(
    to_address: str,
    value_wei: int,
    data: str,
    *,
    signature: Optional[str] = None,
    selector_override: Optional[str] = None,
) -> str:
    """
    Build calldata for execute(address,uint256,bytes).
    """
    selector = get_execute_selector(signature, selector_override)
    head = (
        _encode_address(to_address)
        + _encode_uint(value_wei)
        + _encode_uint(96)  # offset to bytes data
    )
    tail = _encode_bytes(data)
    return selector + head + tail


def build_entrypoint_get_nonce_call(sender: str, key: int = 0) -> str:
    """
    Build calldata for EntryPoint.getNonce(address,uint192).
    """
    selector = _selector_from_signature("getNonce(address,uint192)")
    head = _encode_address(sender) + _encode_uint(key)
    return selector + head
