"""
Strategy Config Normalizer

Maps any config format variant (snake_case flat, camelCase nested, mixed)
to a canonical camelCase nested format expected by the executor and frontend.

Idempotent: normalizing an already-normalized config is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def normalize_strategy_config(strategy_type: str, raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a strategy config to canonical camelCase nested format.

    Handles these input variants:
      - snake_case flat:  {"from_token": "USDC", "to_token": "ETH", "amount_usd": 5}
      - camelCase nested: {"fromToken": {"symbol": "USDC"}, "toToken": {"symbol": "ETH"}, "amountPerExecution": 5}
      - Mixed formats

    Returns a new dict (does not mutate the input).
    """
    if not raw_config:
        return raw_config

    config = dict(raw_config)

    if strategy_type == "dca":
        config = _normalize_dca_config(config)

    return config


def _normalize_dca_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DCA-specific config fields."""
    result = {}

    # --- fromToken ---
    from_token = _build_token_obj(
        config,
        camel_key="fromToken",
        snake_symbol="from_token",
        snake_address="from_token_address",
    )
    if from_token:
        result["fromToken"] = from_token

    # --- toToken ---
    to_token = _build_token_obj(
        config,
        camel_key="toToken",
        snake_symbol="to_token",
        snake_address="to_token_address",
    )
    if to_token:
        result["toToken"] = to_token

    # --- amountPerExecution ---
    amount = (
        config.get("amountPerExecution")
        or config.get("amount_per_execution")
        or config.get("amount_usd")
        or config.get("amountUsd")
        or config.get("amount")
    )
    if amount is not None:
        result["amountPerExecution"] = amount

    # --- chainId ---
    chain_id = config.get("chainId") or config.get("chain_id")
    if chain_id is not None:
        result["chainId"] = int(chain_id)

    # --- maxSlippageBps ---
    slippage = config.get("maxSlippageBps") or config.get("max_slippage_bps")
    if slippage is not None:
        result["maxSlippageBps"] = int(slippage)

    # --- maxGasUsd ---
    gas = config.get("maxGasUsd") or config.get("max_gas_usd")
    if gas is not None:
        result["maxGasUsd"] = gas

    # --- frequency (pass through) ---
    freq = config.get("frequency")
    if freq is not None:
        result["frequency"] = freq

    # Preserve any other camelCase keys not explicitly handled
    _passthrough_keys = {
        "fromToken", "toToken", "amountPerExecution", "chainId",
        "maxSlippageBps", "maxGasUsd", "frequency",
        # snake_case keys we already consumed
        "from_token", "from_token_address", "to_token", "to_token_address",
        "amount_usd", "amount_per_execution", "amountUsd", "amount",
        "chain_id", "max_slippage_bps", "max_gas_usd",
    }
    for key, value in config.items():
        if key not in _passthrough_keys:
            result[key] = value

    return result


def _build_token_obj(
    config: Dict[str, Any],
    camel_key: str,
    snake_symbol: str,
    snake_address: str,
) -> Dict[str, Any] | None:
    """
    Build a canonical token object {symbol, address, chainId} from config.

    Handles:
      - Already-nested: config["fromToken"] = {"symbol": "USDC", "address": "0x..."}
      - Flat snake_case: config["from_token"] = "USDC", config["from_token_address"] = "0x..."
      - Flat string that looks like an address
    """
    existing = config.get(camel_key)

    if isinstance(existing, dict):
        # Already in nested format — return as-is
        return dict(existing)

    # Build from flat fields
    token_obj: Dict[str, Any] = {}

    # Symbol: from camelCase string value or snake_case field
    symbol = existing if isinstance(existing, str) else config.get(snake_symbol)
    if isinstance(symbol, str):
        if symbol.startswith("0x") and len(symbol) == 42:
            token_obj["address"] = symbol
        else:
            token_obj["symbol"] = symbol

    # Address from explicit snake_case field
    address = config.get(snake_address)
    if address:
        token_obj["address"] = str(address)

    # Inherit chainId if present at top level
    chain_id = config.get("chainId") or config.get("chain_id")
    if chain_id is not None:
        token_obj["chainId"] = int(chain_id)

    return token_obj if token_obj else None
