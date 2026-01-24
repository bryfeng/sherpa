"""
Helpers to convert Relay quotes into execution SwapQuote objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import SwapQuote


def _first_tx_from_relay(quote: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = quote.get("steps") or []
    for step in steps:
        items = step.get("items") if isinstance(step, dict) else None
        if not items:
            continue
        for item in items:
            data = item.get("data") if isinstance(item, dict) else None
            if isinstance(data, dict):
                return data
    return None


def relay_quote_to_swap_quote(
    quote: Dict[str, Any],
    *,
    wallet_address: str,
    chain_id: int,
    token_in_address: str,
    token_in_symbol: str,
    token_in_decimals: int,
    token_out_address: str,
    token_out_symbol: str,
    token_out_decimals: int,
    slippage_bps: int,
) -> SwapQuote:
    details = quote.get("details") or {}
    currency_in = details.get("currencyIn") or {}
    currency_out = details.get("currencyOut") or {}

    amount_in = int(currency_in.get("amount", 0) or 0)
    amount_out = int(currency_out.get("amount", 0) or 0)

    price_in_usd = float(currency_in.get("priceUsd", 0) or 0)
    price_out_usd = float(currency_out.get("priceUsd", 0) or 0)
    value_in_usd = float(details.get("amountInUsd", 0) or 0)
    value_out_usd = float(details.get("amountOutUsd", 0) or 0)

    gas_fee_usd = float(details.get("gasFeeUsd", 0) or 0)
    relay_fee_usd = float(details.get("relayFeeUsd", 0) or details.get("relayerFeeUsd", 0) or 0)
    total_fee_usd = float(details.get("totalFeeUsd", 0) or 0)

    request_id = quote.get("requestId") or quote.get("id") or ""
    tx = _first_tx_from_relay(quote)

    return SwapQuote(
        request_id=request_id,
        chain_id=chain_id,
        wallet_address=wallet_address,
        token_in_address=currency_in.get("address") or token_in_address,
        token_in_symbol=currency_in.get("symbol") or token_in_symbol,
        token_in_decimals=int(currency_in.get("decimals", token_in_decimals)),
        amount_in=amount_in,
        token_out_address=currency_out.get("address") or token_out_address,
        token_out_symbol=currency_out.get("symbol") or token_out_symbol,
        token_out_decimals=int(currency_out.get("decimals", token_out_decimals)),
        amount_out_estimate=amount_out,
        price_in_usd=price_in_usd,
        price_out_usd=price_out_usd,
        value_in_usd=value_in_usd,
        value_out_usd=value_out_usd,
        gas_fee_usd=gas_fee_usd,
        relay_fee_usd=relay_fee_usd,
        total_fee_usd=total_fee_usd,
        slippage_bps=int(details.get("slippageBps", slippage_bps)),
        tx=tx,
        approvals=quote.get("approvals") or [],
        signatures=quote.get("signatures") or [],
        expires_at=None,
        time_estimate_seconds=None,
        raw_response=quote,
    )

