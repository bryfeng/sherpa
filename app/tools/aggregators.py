from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, Optional

import httpx

from ..core.bridge.chain_registry import get_registry_sync
from ..core.chain_types import ChainId, is_solana_chain
from ..core.swap.constants import SWAP_SOURCE, TOKEN_ALIAS_MAP, TOKEN_REGISTRY
from ..providers.relay import RelayProvider

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


async def quote_swap_simple(
    token_in: str,
    token_out: str,
    amount_in: float,
    slippage_bps: int = 50,
    *,
    chain: Optional[str] = None,
    wallet_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a live Relay quote for the requested swap."""

    chain_id = _resolve_chain_id(chain)
    token_in_meta = _resolve_token(chain_id, token_in)
    token_out_meta = _resolve_token(chain_id, token_out)

    amount_decimal = _to_decimal(amount_in)
    if amount_decimal is None or amount_decimal <= 0:
        raise ValueError("Amount must be greater than zero")

    amount_base_units = _to_base_units(amount_decimal, int(token_in_meta.get("decimals", 18)))
    if amount_base_units <= 0:
        raise ValueError("Amount too small after accounting for token decimals")

    wallet = wallet_address or token_in_meta.get("default_wallet")
    if not wallet:
        raise ValueError("wallet_address is required to fetch a Relay quote")

    relay_payload: Dict[str, Any] = {
        "user": wallet,
        "originChainId": chain_id,
        "destinationChainId": chain_id,
        "originCurrency": token_in_meta.get("address", ZERO_ADDRESS),
        "destinationCurrency": token_out_meta.get("address", ZERO_ADDRESS),
        "recipient": wallet,
        "tradeType": "EXACT_INPUT",
        "amount": str(amount_base_units),
        "referrer": "sherpa.chat",
        "useExternalLiquidity": True,
        "useDepositAddress": False,
        "topupGas": False,
        "slippageBps": max(0, int(slippage_bps)),
    }

    provider = RelayProvider()
    try:
        quote = await provider.quote(relay_payload)
    except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
        detail = exc.response.text or exc.response.reason_phrase or "Relay request failed"
        raise ValueError(f"Relay quote failed: {detail}")
    except httpx.RequestError as exc:
        raise ValueError(f"Relay network error: {exc}")

    steps = quote.get("steps") or []
    if not steps:
        raise ValueError("Relay did not return any executable steps for this swap")

    details = quote.get("details") or {}
    fees = quote.get("fees") or {}

    currency_in = details.get("currencyIn") or {}
    currency_out = details.get("currencyOut") or {}

    output_tokens = _amount_from_raw(currency_out.get("amount"), token_out_meta.get("decimals", 18))
    if output_tokens is None:
        output_tokens = _to_decimal(currency_out.get("amountFormatted"))
    amount_out_est = float(output_tokens) if output_tokens is not None else 0.0

    amount_in_tokens = float(amount_decimal)

    price_in_dec, price_out_dec, warnings = _extract_prices(currency_in, currency_out, amount_decimal, output_tokens)
    price_in_usd = float(price_in_dec) if price_in_dec is not None else 0.0
    price_out_usd = float(price_out_dec) if price_out_dec is not None else 0.0

    fee_usd = _total_fee_usd(fees)
    if fee_usd is not None and price_in_dec and price_in_dec > 0:
        try:
            fee_est = float(fee_usd / price_in_dec)
        except (InvalidOperation, ZeroDivisionError):
            fee_est = 0.0
    else:
        fee_est = 0.0

    request_id = quote.get("requestId") or quote.get("id")
    tx_items = steps[0].get("items") if isinstance(steps[0], dict) else None
    primary_tx = (tx_items or [{}])[0].get("data") if tx_items else None

    route_payload = {
        "kind": "relay",
        "request_id": request_id,
        "steps": steps,
        "transactions": quote.get("transactions") or [],
        "fees": fees,
        "details": details,
        "expires_at": details.get("expiresAt") or details.get("expiry"),
        "slippage_bps": slippage_bps,
        "relay_payload": relay_payload,
        "tx_ready": bool(primary_tx),
    }
    if primary_tx:
        route_payload["primary_tx"] = primary_tx
        route_payload["tx"] = primary_tx

    response = {
        "success": True,
        "from": token_in_meta["symbol"],
        "to": token_out_meta["symbol"],
        "amount_in": amount_in_tokens,
        "price_in_usd": price_in_usd,
        "price_out_usd": price_out_usd,
        "amount_out_est": amount_out_est,
        "fee_est": fee_est,
        "slippage_bps": slippage_bps,
        "route": route_payload,
        "sources": [dict(SWAP_SOURCE)],
        "warnings": warnings,
        "wallet": {"address": wallet},
        "chain_id": chain_id,
        "quote_type": "swap",
    }

    return response


def _resolve_chain_id(chain: Optional[str]) -> int:
    """
    Resolve chain string to an integer chain ID.

    Note: This function only supports EVM chains. Solana swaps are not
    supported by the Relay aggregator and will raise an error.
    """
    if chain is None:
        return 1
    chain_str = str(chain).strip()
    if not chain_str:
        return 1
    lower = chain_str.lower()

    # Check for Solana - not supported for swaps via Relay
    if lower in ("sol", "solana"):
        raise ValueError(
            "Solana swaps are not yet supported via Relay. "
            "Use Jupiter (https://jup.ag) for Solana swaps."
        )

    # Use dynamic chain registry
    registry = get_registry_sync()
    chain_id = registry.get_chain_id(lower)
    if chain_id is not None:
        # Guard against non-EVM chains
        if is_solana_chain(chain_id):
            raise ValueError(
                "Solana swaps are not yet supported via Relay. "
                "Use Jupiter (https://jup.ag) for Solana swaps."
            )
        # Ensure we return an int for EVM chains
        if isinstance(chain_id, int):
            return chain_id

    # Try parsing as integer chain ID
    try:
        return int(chain_str, 10)
    except ValueError:
        raise ValueError(f"Unsupported chain identifier: {chain}")


def _resolve_token(chain_id: int, token: str) -> Dict[str, Any]:
    registry = TOKEN_REGISTRY.get(chain_id)
    if not registry:
        raise ValueError(f"Unsupported chain_id for swaps: {chain_id}")

    token_clean = str(token or "").strip()
    if not token_clean:
        raise ValueError("Token symbol or address is required")

    token_lower = token_clean.lower()
    if token_lower.startswith("0x") and len(token_clean) == 42:
        for meta in registry.values():
            if str(meta.get("address", "")).lower() == token_lower:
                return meta
        raise ValueError(f"Token address not supported on chain {chain_id}: {token}")

    alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})
    canonical = alias_map.get(token_lower) or token_clean.upper()
    if canonical in registry:
        return registry[canonical]
    raise ValueError(f"Token not supported on chain {chain_id}: {token}")


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _to_base_units(amount: Decimal, decimals: int) -> int:
    if decimals < 0:
        decimals = 0
    quantized = (amount * (Decimal(10) ** decimals)).quantize(Decimal("1"), rounding=ROUND_DOWN)
    return int(quantized)


def _amount_from_raw(raw: Any, decimals: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    try:
        decs = int(decimals)
    except (TypeError, ValueError):
        decs = 18
    try:
        return Decimal(str(raw)) / (Decimal(10) ** decs)
    except (InvalidOperation, TypeError, ValueError):
        return None


def _extract_prices(
    currency_in: Dict[str, Any],
    currency_out: Dict[str, Any],
    amount_in_tokens: Decimal,
    amount_out_tokens: Optional[Decimal],
) -> tuple[Optional[Decimal], Optional[Decimal], list]:
    warnings: list[str] = []

    price_in = _preferred_price(currency_in.get("priceUsd"), currency_in.get("amountUsd"), amount_in_tokens)
    if price_in is None:
        warnings.append("price_in_usd unavailable from Relay response")

    out_amount = amount_out_tokens or _to_decimal(currency_out.get("amountFormatted"))
    price_out = None
    if out_amount is not None and out_amount > 0:
        price_out = _preferred_price(currency_out.get("priceUsd"), currency_out.get("amountUsd"), out_amount)
    if price_out is None:
        if out_amount is None:
            warnings.append("amount_out_est unavailable from Relay response")
        warnings.append("price_out_usd unavailable from Relay response")

    return price_in, price_out, warnings


def _preferred_price(price_field: Any, amount_usd: Any, amount_tokens: Decimal) -> Optional[Decimal]:
    if price_field is not None:
        price_dec = _to_decimal(price_field)
        if price_dec is not None and price_dec > 0:
            return price_dec
    usd_dec = _to_decimal(amount_usd)
    if usd_dec is not None and amount_tokens > 0:
        try:
            return usd_dec / amount_tokens
        except (InvalidOperation, ZeroDivisionError):
            return None
    return None


def _total_fee_usd(fees: Dict[str, Any]) -> Optional[Decimal]:
    total = Decimal("0")
    seen = False
    for value in fees.values():
        if not isinstance(value, dict):
            continue
        amount_usd = value.get("amountUsd")
        if amount_usd is None:
            continue
        dec = _to_decimal(amount_usd)
        if dec is None:
            continue
        total += dec
        seen = True
    return total if seen else None
