"""Fetch swap/bridge quotes from Relay (EVM) and Jupiter (Solana)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..providers.relay import RelayProvider
from ..providers.jupiter import (
    JupiterSwapProvider,
    JupiterQuote,
    JupiterSwapResult,
    JupiterQuoteError,
    JupiterSwapError,
    get_jupiter_swap_provider,
)


@dataclass
class RelayQuoteResult:
    """Structured result from a Relay quote request."""
    ok: bool
    quote_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_detail: Optional[str] = None
    is_network_error: bool = False


@dataclass
class JupiterQuoteResult:
    """Structured result from a Jupiter quote + transaction build."""
    ok: bool
    quote: Optional[JupiterQuote] = None
    swap_result: Optional[JupiterSwapResult] = None
    error_message: Optional[str] = None


async def fetch_relay_quote(
    relay_payload: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> RelayQuoteResult:
    """Fetch a swap/bridge quote from Relay."""
    log = logger or logging.getLogger(__name__)
    provider = RelayProvider()

    try:
        quote_response = await provider.quote(relay_payload)
    except httpx.HTTPStatusError as exc:
        detail_text = ''
        try:
            detail_text = (exc.response.text or '').strip()
        except Exception:
            detail_text = ''
        log.warning('Relay swap quote error: %s', detail_text or exc, exc_info=False)
        return RelayQuoteResult(
            ok=False,
            error_message='Relay could not produce a swap route for that request.',
            error_detail=detail_text,
        )
    except httpx.RequestError as exc:
        log.warning('Relay swap quote network error: %s', exc)
        return RelayQuoteResult(
            ok=False,
            error_message='I could not reach Relay for that swap quote. Please try again in a moment.',
            is_network_error=True,
        )

    if not isinstance(quote_response, dict) or not quote_response.get('steps'):
        return RelayQuoteResult(
            ok=False,
            quote_response=quote_response,
            error_message='Relay did not return any swap data.',
            error_detail='Relay did not return any swap steps.',
        )

    return RelayQuoteResult(ok=True, quote_response=quote_response)


async def fetch_jupiter_quote(
    *,
    input_mint: str,
    output_mint: str,
    amount_base_units: int,
    wallet: str,
    slippage_bps: int = 50,
    jupiter_provider: Optional[JupiterSwapProvider] = None,
    logger: Optional[logging.Logger] = None,
) -> JupiterQuoteResult:
    """Fetch a Jupiter swap quote and build the unsigned transaction."""
    log = logger or logging.getLogger(__name__)
    jupiter = jupiter_provider or get_jupiter_swap_provider()

    try:
        quote: JupiterQuote = await jupiter.get_swap_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount_base_units,
            slippage_bps=slippage_bps,
        )
    except JupiterQuoteError as exc:
        log.warning('Jupiter quote error: %s', exc)
        return JupiterQuoteResult(ok=False, error_message=f'Jupiter could not produce a swap route: {exc}')

    try:
        swap_result: JupiterSwapResult = await jupiter.build_swap_transaction(
            quote=quote,
            user_public_key=wallet,
        )
    except JupiterSwapError as exc:
        log.warning('Jupiter swap build error: %s', exc)
        return JupiterQuoteResult(ok=False, quote=quote, error_message=f'Jupiter could not build swap transaction: {exc}')

    return JupiterQuoteResult(ok=True, quote=quote, swap_result=swap_result)


def extract_relay_steps(
    steps_raw: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalise raw Relay steps into (steps, transactions, approvals, signatures)."""
    normalized_steps: List[Dict[str, Any]] = []
    transactions: List[Dict[str, Any]] = []
    approvals: List[Dict[str, Any]] = []
    signatures: List[Dict[str, Any]] = []

    for step in steps_raw:
        items = []
        for item in step.get('items', []):
            item_entry: Dict[str, Any] = {'status': item.get('status'), 'data': item.get('data')}
            if 'check' in item:
                item_entry['check'] = item['check']
            if 'type' in item:
                item_entry['type'] = item['type']
            items.append(item_entry)

            data_obj = item.get('data') or {}
            if isinstance(data_obj, dict):
                if 'to' in data_obj and ('data' in data_obj or 'value' in data_obj):
                    transactions.append({
                        'step_id': step.get('id'),
                        'action': step.get('action'),
                        'description': step.get('description'),
                        'status': item.get('status'),
                        'data': data_obj,
                        'check': item.get('check'),
                    })
                elif {'spender', 'amount'} <= set(data_obj.keys()) or ('spender' in data_obj and 'value' in data_obj):
                    approvals.append({
                        'step_id': step.get('id'),
                        'action': step.get('action'),
                        'status': item.get('status'),
                        'data': data_obj,
                    })
                elif any(key in data_obj for key in ('typedData', 'domain', 'types', 'message')):
                    signatures.append({
                        'step_id': step.get('id'),
                        'action': step.get('action'),
                        'status': item.get('status'),
                        'data': data_obj,
                    })

        step_entry: Dict[str, Any] = {
            'id': step.get('id'),
            'kind': step.get('kind'),
            'action': step.get('action'),
            'description': step.get('description'),
            'items': items,
        }
        if step.get('requestId'):
            step_entry['requestId'] = step['requestId']
        normalized_steps.append(step_entry)

    return normalized_steps, transactions, approvals, signatures
