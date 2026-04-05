"""Build frontend panel payloads and summary text for swap/bridge quotes."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from ..core.swap.constants import SWAP_SOURCE

JUPITER_SWAP_SOURCE = {'name': 'Jupiter', 'url': 'https://jup.ag'}


# ---------------------------------------------------------------------------
# Decimal / display helpers
# ---------------------------------------------------------------------------

def decimal_to_str(value: Optional[Decimal]) -> str:
    if value is None:
        return '—'
    q = value.normalize()
    return format(q, 'f').rstrip('0').rstrip('.') if '.' in format(q, 'f') else format(q, 'f')


def format_eta_seconds(seconds: Optional[Any]) -> Optional[str]:
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return None
    if sec <= 0:
        return None
    if sec < 90:
        return f"{sec:.0f} sec"
    minutes = sec / 60.0
    return f"{minutes:.1f} min"


def amount_to_decimal(raw: Any, decimals_hint: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    try:
        decimals = int(decimals_hint)
    except (TypeError, ValueError):
        decimals = 18
    try:
        return Decimal(str(raw)) / (Decimal(10) ** decimals)
    except (InvalidOperation, TypeError, ValueError):
        return None


def total_fee_usd(fee_dict: Dict[str, Any]) -> Optional[Decimal]:
    total = Decimal('0')
    seen = False
    for fee_data in fee_dict.values():
        if not isinstance(fee_data, dict):
            continue
        amount_usd = fee_data.get('amountUsd')
        if amount_usd is None:
            continue
        try:
            total += Decimal(str(amount_usd))
            seen = True
        except (InvalidOperation, TypeError, ValueError):
            continue
    return total if seen else None


# ---------------------------------------------------------------------------
# Relay panel
# ---------------------------------------------------------------------------

def build_relay_panel(
    *,
    token_in_meta: Dict[str, Any],
    token_out_meta: Dict[str, Any],
    amount_decimal: Decimal,
    origin_chain_id: Any,
    destination_chain_id: Any,
    is_cross_chain: bool,
    wallet: str,
    relay_payload: Dict[str, Any],
    amount_base_units_str: str,
    usd_amount: Optional[Decimal],
    percent_fraction: Optional[Decimal],
    chain_name_fn,
    quote_response: Dict[str, Any],
    normalized_steps: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]],
    approvals: List[Dict[str, Any]],
    signatures: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """Build panel payload + card dict + summary_reply + summary_tool for a Relay quote.

    Returns (panel, panel_card, summary_reply, summary_tool).
    """

    panel_payload: Dict[str, Any] = {
        'status': 'pending',
        'chain_id': origin_chain_id,
        'destination_chain_id': destination_chain_id if is_cross_chain else None,
        'chain': chain_name_fn(origin_chain_id),
        'destination_chain': chain_name_fn(destination_chain_id) if is_cross_chain else None,
        'is_cross_chain': is_cross_chain,
        'wallet': {'address': wallet},
        'provider': 'relay',
        'relay_request': relay_payload,
        'quote_type': 'bridge' if is_cross_chain else 'swap',
        'tokens': {
            'input': token_in_meta,
            'output': token_out_meta,
        },
        'amounts': {
            'input': str(amount_decimal),
            'input_base_units': amount_base_units_str,
        },
    }
    if percent_fraction is not None:
        panel_payload['amounts']['input_share_percent'] = decimal_to_str(percent_fraction * Decimal('100'))
    if usd_amount is not None:
        panel_payload['amounts']['input_usd'] = decimal_to_str(usd_amount)

    fees = quote_response.get('fees') or {}
    details = quote_response.get('details') or {}

    request_id = None
    if normalized_steps:
        request_id = normalized_steps[0].get('requestId')
    if not request_id:
        request_id = quote_response.get('requestId') or quote_response.get('id')

    primary_tx_entry = transactions[0] if transactions else None
    primary_tx = primary_tx_entry.get('data') if primary_tx_entry else None

    input_currency = details.get('currencyIn') or {}
    output_currency = details.get('currencyOut') or {}

    input_symbol = input_currency.get('currency', {}).get('symbol', token_in_meta['symbol'])
    output_symbol = output_currency.get('currency', {}).get('symbol', token_out_meta['symbol'])

    input_decimals = input_currency.get('currency', {}).get('decimals', token_in_meta['decimals'])
    output_decimals = output_currency.get('currency', {}).get('decimals', token_out_meta['decimals'])

    routed_from_amount = amount_to_decimal(input_currency.get('amount'), input_decimals)
    if routed_from_amount is None:
        routed_from_amount = amount_decimal
    routed_to_amount = amount_to_decimal(output_currency.get('amount'), output_decimals)

    input_amount_display = input_currency.get('amountFormatted')
    if not input_amount_display and routed_from_amount is not None:
        input_amount_display = decimal_to_str(routed_from_amount)

    output_amount_display: Optional[str] = None
    if routed_to_amount is not None:
        output_amount_display = output_currency.get('amountFormatted') or decimal_to_str(routed_to_amount)

    output_usd = output_currency.get('amountUsd')

    time_estimate = details.get('timeEstimate')
    eta_seconds = None
    if time_estimate is not None:
        try:
            eta_seconds = float(time_estimate) * 60.0
        except (TypeError, ValueError):
            eta_seconds = None
    eta_readable = format_eta_seconds(eta_seconds)

    fee_usd = total_fee_usd(fees)

    panel_payload.update({
        'status': 'ok' if transactions else 'quote_only',
        'request_id': request_id,
        'steps': normalized_steps,
        'transactions': transactions,
        'fees': fees,
        'details': details,
        'eta_seconds': eta_seconds,
        'tx_ready': bool(primary_tx),
        'quote_expiry': details.get('expiresAt') or details.get('expiry'),
    })
    if primary_tx:
        primary_tx['chainId'] = origin_chain_id
        panel_payload['tx'] = primary_tx
    if approvals:
        panel_payload['approvals'] = approvals
    if signatures:
        panel_payload['signatures'] = signatures
    if primary_tx_entry and primary_tx_entry.get('check'):
        panel_payload['status_check'] = primary_tx_entry['check']

    panel_payload['breakdown'] = {
        'input': {
            'symbol': input_symbol,
            'amount': input_amount_display,
            'token_address': input_currency.get('currency', {}).get('address', token_in_meta['address']),
        },
        'output': {
            'symbol': output_symbol,
            'amount_estimate': output_amount_display,
            'token_address': output_currency.get('currency', {}).get('address', token_out_meta['address']),
            'value_usd': output_usd,
        },
        'fees': {
            'total_usd': float(fee_usd) if fee_usd is not None else None,
            'gas_usd': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else None,
            'slippage_percent': details.get('slippageTolerance', {}).get('destination', {}).get('percent'),
        },
    }
    panel_payload['usd_estimates'] = {
        'output': output_usd,
        'gas': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else (float(fee_usd) if fee_usd is not None else None),
    }
    if usd_amount is not None:
        try:
            panel_payload['usd_estimates']['input_requested'] = float(usd_amount)
        except (TypeError, ValueError):
            panel_payload['usd_estimates']['input_requested'] = None
    panel_payload['instructions'] = [
        'Review the Relay swap quote including output estimate and fees.',
        f'Confirm the {token_in_meta["symbol"]} → {token_out_meta["symbol"]} transaction in your connected wallet.',
        'Complete any approval prompts before executing the swap.',
        'Need fresh pricing? Ask "refresh swap quote".',
    ]
    panel_payload['actions'] = {
        'refresh_quote': 'Say "refresh swap quote" to fetch an updated price.',
        'open_wallet': 'Use your connected wallet to review and submit the prepared swap.',
    }

    # Summary
    if is_cross_chain:
        swap_summary = f"✅ Bridge {decimal_to_str(routed_from_amount or amount_decimal)} {input_symbol} ({chain_name_fn(origin_chain_id)}) → {output_symbol} ({chain_name_fn(destination_chain_id)})"
    else:
        swap_summary = f"✅ Swap {decimal_to_str(routed_from_amount or amount_decimal)} {input_symbol} → {output_symbol} on {chain_name_fn(origin_chain_id)}"
    summary_lines = [swap_summary]
    if usd_amount is not None:
        try:
            summary_lines.insert(0, f"🎯 Target ≈ ${float(usd_amount):.2f} of {input_symbol}")
        except (TypeError, ValueError):
            summary_lines.insert(0, f"🎯 Target amount ≈ ${decimal_to_str(usd_amount)} of {input_symbol}")
    if routed_to_amount is not None:
        arrival_line = f"Estimated output: {decimal_to_str(routed_to_amount)} {output_symbol}"
        if output_usd is not None:
            try:
                arrival_line += f" (~${float(output_usd):.2f})"
            except (ValueError, TypeError):
                pass
        summary_lines.append(arrival_line)
    if fee_usd is not None:
        try:
            summary_lines.append(f"Estimated fees ≈ ${float(fee_usd):.2f}")
        except (ValueError, TypeError):
            pass
    if eta_readable:
        summary_lines.append(f"ETA ≈ {eta_readable}")
    summary_lines.append('Confirm the swap in your connected wallet when prompted.')

    summary_reply = "\n".join(summary_lines)
    if is_cross_chain:
        summary_tool = f"Relay bridge: {decimal_to_str(amount_decimal)} {input_symbol} ({chain_name_fn(origin_chain_id)}) → {output_symbol} ({chain_name_fn(destination_chain_id)})"
    else:
        summary_tool = f"Relay swap: {decimal_to_str(amount_decimal)} {input_symbol} → {output_symbol} on {chain_name_fn(origin_chain_id)}"

    panel = {
        'id': 'relay_swap_quote',
        'kind': 'card',
        'title': f"Relay Swap: {input_symbol} → {output_symbol}",
        'payload': panel_payload,
        'sources': [SWAP_SOURCE],
        'metadata': {'status': panel_payload['status']},
    }

    return panel_payload, panel, summary_reply, summary_tool


def build_relay_error_panel(
    *,
    token_in_meta: Dict[str, Any],
    token_out_meta: Dict[str, Any],
    panel_payload: Dict[str, Any],
    error_detail: Optional[str],
    error_message: str,
) -> Dict[str, Any]:
    """Build a panel dict for a Relay error."""
    panel_payload['status'] = 'error'
    if error_detail:
        panel_payload.setdefault('issues', []).append(error_detail)
    return {
        'id': 'relay_swap_quote',
        'kind': 'card',
        'title': f"Relay Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
        'payload': panel_payload,
        'sources': [SWAP_SOURCE],
        'metadata': {'status': 'error'},
    }


# ---------------------------------------------------------------------------
# Jupiter (Solana) panel
# ---------------------------------------------------------------------------

def build_jupiter_panel(
    *,
    token_in_meta: Dict[str, Any],
    token_out_meta: Dict[str, Any],
    amount_decimal: Decimal,
    wallet: str,
    input_mint: str,
    output_mint: str,
    output_amount_decimal: Decimal,
    min_output_decimal: Decimal,
    quote: Any,
    swap_result: Any,
    usd_amount: Optional[Decimal],
    percent_fraction: Optional[Decimal],
    amount_base_units: int,
    solana_chain_id: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """Build panel payload + card + summary_reply + summary_tool for a Jupiter quote.

    Returns (panel_payload, panel, summary_reply, summary_tool).
    """

    panel_payload: Dict[str, Any] = {
        'status': 'pending',
        'chain_id': solana_chain_id,
        'chain': 'Solana',
        'wallet': {'address': wallet},
        'provider': 'jupiter',
        'quote_type': 'swap',
        'tokens': {
            'input': token_in_meta,
            'output': token_out_meta,
        },
        'amounts': {
            'input': str(amount_decimal),
            'input_base_units': str(amount_base_units),
        },
    }
    if percent_fraction is not None:
        panel_payload['amounts']['input_share_percent'] = decimal_to_str(percent_fraction * Decimal('100'))
    if usd_amount is not None:
        panel_payload['amounts']['input_usd'] = decimal_to_str(usd_amount)

    panel_payload.update({
        'status': 'ok',
        'tx_ready': True,
        'quote_expiry': None,
        'solana_tx': swap_result.swap_transaction,
        'last_valid_block_height': swap_result.last_valid_block_height,
        'priority_fee_lamports': swap_result.priority_fee_lamports,
        'compute_unit_limit': swap_result.compute_unit_limit,
    })

    panel_payload['breakdown'] = {
        'input': {
            'symbol': token_in_meta['symbol'],
            'amount': decimal_to_str(amount_decimal),
            'token_address': input_mint,
        },
        'output': {
            'symbol': token_out_meta['symbol'],
            'amount_estimate': decimal_to_str(output_amount_decimal),
            'min_amount': decimal_to_str(min_output_decimal),
            'token_address': output_mint,
        },
        'fees': {
            'priority_fee_lamports': swap_result.priority_fee_lamports,
            'slippage_bps': quote.slippage_bps,
            'price_impact_pct': quote.price_impact_pct,
        },
    }

    panel_payload['instructions'] = [
        'Review the Jupiter swap quote including output estimate and price impact.',
        f'Confirm the {token_in_meta["symbol"]} → {token_out_meta["symbol"]} transaction in your Solana wallet.',
        'Sign the transaction to execute the swap.',
        'Need fresh pricing? Ask "refresh swap quote".',
    ]

    panel_payload['actions'] = {
        'refresh_quote': 'Say "refresh swap quote" to fetch an updated price.',
        'open_wallet': 'Use your connected Solana wallet to sign and submit the swap.',
    }

    summary_lines = [
        f"✅ Swap {decimal_to_str(amount_decimal)} {token_in_meta['symbol']} → {token_out_meta['symbol']} on Solana"
    ]
    if usd_amount is not None:
        summary_lines.insert(0, f"🎯 Target ≈ ${decimal_to_str(usd_amount)} of {token_in_meta['symbol']}")
    summary_lines.append(f"Estimated output: {decimal_to_str(output_amount_decimal)} {token_out_meta['symbol']}")
    summary_lines.append(f"Minimum output: {decimal_to_str(min_output_decimal)} {token_out_meta['symbol']}")
    if quote.price_impact_pct > 0.1:
        summary_lines.append(f"⚠️ Price impact: {quote.price_impact_pct:.2f}%")
    summary_lines.append('Confirm the swap in your connected Solana wallet when prompted.')

    summary_reply = "\n".join(summary_lines)
    summary_tool = f"Jupiter swap plan: {decimal_to_str(amount_decimal)} {token_in_meta['symbol']} → {token_out_meta['symbol']} on Solana"

    panel = {
        'id': 'jupiter_swap_quote',
        'kind': 'card',
        'title': f"Jupiter Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
        'payload': panel_payload,
        'sources': [JUPITER_SWAP_SOURCE],
        'metadata': {'status': 'ok', 'chain': 'solana'},
    }

    return panel_payload, panel, summary_reply, summary_tool


def build_jupiter_error_panel(
    *,
    token_in_meta: Dict[str, Any],
    token_out_meta: Dict[str, Any],
    panel_payload: Dict[str, Any],
    error_message: str,
) -> Dict[str, Any]:
    """Build a panel dict for a Jupiter error."""
    panel_payload['status'] = 'error'
    panel_payload.setdefault('issues', []).append(error_message)
    return {
        'id': 'jupiter_swap_quote',
        'kind': 'card',
        'title': f"Jupiter Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
        'payload': panel_payload,
        'sources': [JUPITER_SWAP_SOURCE],
        'metadata': {'status': 'error'},
    }
