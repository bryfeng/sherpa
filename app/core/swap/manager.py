"""SwapManager orchestrates Relay-powered token swaps."""

from __future__ import annotations

import copy
import logging
import re
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...providers.relay import RelayProvider
from ...types.requests import ChatRequest
from ..bridge.constants import (
    CHAIN_ALIAS_TO_ID,
    CHAIN_METADATA,
    DEFAULT_CHAIN_NAME_TO_ID,
    NATIVE_PLACEHOLDER,
)
from .constants import (
    GLOBAL_TOKEN_ALIASES,
    SWAP_FOLLOWUP_KEYWORDS,
    SWAP_KEYWORDS,
    SWAP_SOURCE,
    TOKEN_ALIAS_MAP,
    TOKEN_REGISTRY,
)
from .models import SwapResult, SwapState


class SwapManager:
    """Encapsulates swap intent parsing, quoting, and response shaping."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._pending: Dict[str, SwapState] = {}

    async def maybe_handle(
        self,
        request: ChatRequest,
        conversation_id: str,
        *,
        wallet_address: Optional[str],
        default_chain: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not request.messages:
            return None

        latest = request.messages[-1].content.strip()
        if not latest:
            return None

        normalized_message = self._normalize_text(latest)
        state = self._pending.get(conversation_id)

        explicit_swap = self._is_swap_query(normalized_message)
        followup = False
        if not explicit_swap and state:
            followup = self._is_swap_followup(normalized_message)

        if not explicit_swap and not followup:
            return None

        context = dict(state.context) if state else {}

        wallet = wallet_address or context.get('wallet_address')

        def finalize_result(
            status: str,
            *,
            message: Optional[str] = None,
            panel: Optional[Dict[str, Any]] = None,
            summary_reply: Optional[str] = None,
            summary_tool: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            context_updates: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            result: Dict[str, Any] = {'status': status}
            if message:
                result['message'] = message
            if panel:
                result['panel'] = panel
            if summary_reply:
                result['summary_reply'] = summary_reply
            if summary_tool:
                result['summary_tool'] = summary_tool
            if extra:
                result.update(extra)

            merged_context = {k: v for k, v in context.items() if v is not None}
            if context_updates:
                merged_context.update({k: v for k, v in context_updates.items() if v is not None})

            new_state = SwapState(
                context=merged_context,
                quote_params=copy.deepcopy(state.quote_params) if state else {},
                last_prompt=latest,
                status=status,
                panel=panel,
                summary_reply=summary_reply,
                summary_tool=summary_tool,
                last_result=result,
                quote_id=state.quote_id if state else None,
            )
            self._pending[conversation_id] = new_state
            return result

        if not wallet:
            return finalize_result(
                'needs_address',
                message='I can prepare a Relay swap, but I need the wallet address that will sign it.',
            )

        detected_chain = self._detect_chain(normalized_message)
        chain_id = (
            detected_chain
            or context.get('chain_id')
            or self._chain_from_default(default_chain)
            or 1
        )

        if chain_id not in TOKEN_REGISTRY:
            return finalize_result(
                'unsupported_chain',
                message='I can only quote swaps on Ethereum mainnet right now. Try “swap 0.5 ETH to USDC on Ethereum”.',
                context_updates={'wallet_address': wallet},
            )

        token_in_symbol = context.get('token_in_symbol')
        token_out_symbol = context.get('token_out_symbol')
        amount_decimal = self._to_decimal(context.get('input_amount')) if context.get('input_amount') else None

        parsed_amount, parsed_token_in, parsed_token_out = self._parse_swap_request(normalized_message)

        if parsed_token_in:
            token_in_symbol = parsed_token_in
        if parsed_token_out:
            token_out_symbol = parsed_token_out
        if parsed_amount is not None:
            amount_decimal = parsed_amount

        if ('$' in normalized_message) or ('usd' in normalized_message and parsed_amount is None):
            return finalize_result(
                'needs_amount',
                message='USD-denominated swap amounts are not supported yet. Please specify the input token amount, e.g., “swap 0.5 ETH to USDC”.',
                context_updates={'wallet_address': wallet, 'chain_id': chain_id},
            )

        token_in_meta = self._resolve_token(chain_id, token_in_symbol) if token_in_symbol else None
        token_out_meta = self._resolve_token(chain_id, token_out_symbol) if token_out_symbol else None

        if token_in_meta is None or token_out_meta is None:
            supported = self._supported_tokens_string(chain_id)
            return finalize_result(
                'needs_token',
                message=f'I need the tokens for this swap. Supported examples on {self._chain_name(chain_id)}: {supported}. Try “swap 0.25 ETH to USDC”.',
                context_updates={'wallet_address': wallet, 'chain_id': chain_id},
            )

        if token_in_meta['symbol'] == token_out_meta['symbol']:
            return finalize_result(
                'needs_token',
                message='The input and output tokens are the same. Please choose two different assets for the swap.',
                context_updates={'wallet_address': wallet, 'chain_id': chain_id},
            )

        if amount_decimal is None:
            return finalize_result(
                'needs_amount',
                message=f'How much {token_in_meta["symbol"]} should I swap? For example: “swap 0.5 {token_in_meta["symbol"]} to {token_out_meta["symbol"]}”.',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                },
            )

        amount_decimal = self._sanitize_amount(amount_decimal)
        if amount_decimal is None or amount_decimal <= Decimal('0'):
            return finalize_result(
                'needs_amount',
                message='The swap amount looks invalid or too small. Try a larger value.',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                },
            )

        amount_base_units = self._to_base_units(amount_decimal, int(token_in_meta['decimals']))
        if amount_base_units is None or amount_base_units <= 0:
            return finalize_result(
                'needs_amount',
                message='The swap amount is too small after accounting for token decimals. Try a larger amount.',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                },
            )

        amount_base_units_str = str(amount_base_units)

        relay_payload: Dict[str, Any] = {
            'user': wallet,
            'originChainId': chain_id,
            'destinationChainId': chain_id,
            'originCurrency': token_in_meta['address'],
            'destinationCurrency': token_out_meta['address'],
            'recipient': wallet,
            'tradeType': 'EXACT_INPUT',
            'amount': amount_base_units_str,
            'referrer': 'sherpa.chat',
            'useExternalLiquidity': True,
            'useDepositAddress': False,
            'topupGas': False,
        }

        panel_payload: Dict[str, Any] = {
            'status': 'pending',
            'chain_id': chain_id,
            'chain': self._chain_name(chain_id),
            'wallet': {'address': wallet},
            'provider': 'relay',
            'relay_request': relay_payload,
            'quote_type': 'swap',
            'tokens': {
                'input': token_in_meta,
                'output': token_out_meta,
            },
            'amounts': {
                'input': str(amount_decimal),
                'input_base_units': amount_base_units_str,
            },
        }

        context_updates = {
            'wallet_address': wallet,
            'chain_id': chain_id,
            'token_in_symbol': token_in_meta['symbol'],
            'token_out_symbol': token_out_meta['symbol'],
            'input_amount': str(amount_decimal),
            'token_in_address': token_in_meta['address'],
            'token_out_address': token_out_meta['address'],
        }

        pending_entry = SwapState(
            context={k: v for k, v in context_updates.items() if v is not None},
            quote_params=copy.deepcopy(relay_payload),
            last_prompt=latest,
        )
        self._pending[conversation_id] = pending_entry

        provider = RelayProvider()

        try:
            quote_response = await provider.quote(relay_payload)
        except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
            message = 'Relay could not produce a swap route for that request.'
            detail_text = ''
            try:
                detail_text = (exc.response.text or '').strip()
            except Exception:  # pragma: no cover - logging only
                detail_text = ''
            self._logger.warning('Relay swap quote error: %s', detail_text or exc, exc_info=False)
            panel_payload['status'] = 'error'
            panel_payload.setdefault('issues', []).append(detail_text or message)
            panel = {
                'id': 'relay_swap_quote',
                'kind': 'card',
                'title': f"Relay Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
                'payload': panel_payload,
                'sources': [SWAP_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result('error', message=message, panel=panel, context_updates=context_updates)
        except httpx.RequestError as exc:  # network issues
            self._logger.warning('Relay swap quote network error: %s', exc)
            return finalize_result(
                'error',
                message='I could not reach Relay for that swap quote. Please try again in a moment.',
                panel=None,
                context_updates=context_updates,
            )

        if not isinstance(quote_response, dict) or not quote_response.get('steps'):
            panel_payload['status'] = 'error'
            panel_payload.setdefault('issues', []).append('Relay did not return any swap steps.')
            panel = {
                'id': 'relay_swap_quote',
                'kind': 'card',
                'title': f"Relay Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
                'payload': panel_payload,
                'sources': [SWAP_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result('error', message='Relay did not return any swap data.', panel=panel, context_updates=context_updates)

        steps_raw = quote_response.get('steps') or []
        fees = quote_response.get('fees') or {}
        details = quote_response.get('details') or {}

        normalized_steps, transactions, approvals, signatures = self._extract_steps(steps_raw)

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

        routed_from_amount = self._amount_to_decimal(input_currency.get('amount'), input_decimals)
        if routed_from_amount is None:
            routed_from_amount = amount_decimal
        routed_to_amount = self._amount_to_decimal(output_currency.get('amount'), output_decimals)

        input_amount_display = input_currency.get('amountFormatted')
        if not input_amount_display and routed_from_amount is not None:
            input_amount_display = self._decimal_to_str(routed_from_amount)

        if routed_to_amount is not None:
            output_amount_display = output_currency.get('amountFormatted') or self._decimal_to_str(routed_to_amount)
        else:
            output_amount_display = None

        output_usd = output_currency.get('amountUsd')

        time_estimate = details.get('timeEstimate')
        eta_seconds = None
        if time_estimate is not None:
            try:
                eta_seconds = float(time_estimate) * 60.0
            except (TypeError, ValueError):
                eta_seconds = None
        eta_readable = self._format_eta_seconds(eta_seconds)

        total_fee_usd = self._total_fee_usd(fees)

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
                'total_usd': float(total_fee_usd) if total_fee_usd is not None else None,
                'gas_usd': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else None,
                'slippage_percent': details.get('slippageTolerance', {}).get('destination', {}).get('percent'),
            },
        }
        panel_payload['usd_estimates'] = {
            'output': output_usd,
            'gas': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else (float(total_fee_usd) if total_fee_usd is not None else None),
        }
        panel_payload['instructions'] = [
            'Review the Relay swap quote including output estimate and fees.',
            f'Confirm the {token_in_meta["symbol"]} → {token_out_meta["symbol"]} transaction in your connected wallet.',
            'Complete any approval prompts before executing the swap.',
            'Need fresh pricing? Ask “refresh swap quote”.',
        ]
        panel_payload['actions'] = {
            'refresh_quote': 'Say “refresh swap quote” to fetch an updated price.',
            'open_wallet': 'Use your connected wallet to review and submit the prepared swap.',
        }

        summary_lines = [
            f"✅ Swap {self._decimal_to_str(routed_from_amount or amount_decimal)} {input_symbol} → {output_symbol} on {self._chain_name(chain_id)}"
        ]
        if routed_to_amount is not None:
            arrival_line = f"Estimated output: {self._decimal_to_str(routed_to_amount)} {output_symbol}"
            if output_usd is not None:
                try:
                    arrival_line += f" (~${float(output_usd):.2f})"
                except (ValueError, TypeError):
                    pass
            summary_lines.append(arrival_line)
        if total_fee_usd is not None:
            try:
                summary_lines.append(f"Estimated fees ≈ ${float(total_fee_usd):.2f}")
            except (ValueError, TypeError):
                pass
        if eta_readable:
            summary_lines.append(f"ETA ≈ {eta_readable}")
        summary_lines.append('Confirm the swap in your connected wallet when prompted.')

        summary_reply = "\n".join(summary_lines)
        summary_tool = (
            f"Relay swap plan: {self._decimal_to_str(amount_decimal)} {input_symbol} → {output_symbol} on {self._chain_name(chain_id)}"
        )

        panel = {
            'id': 'relay_swap_quote',
            'kind': 'card',
            'title': f"Relay Swap: {input_symbol} → {output_symbol}",
            'payload': panel_payload,
            'sources': [SWAP_SOURCE],
            'metadata': {'status': panel_payload['status']},
        }

        pending_entry.panel = panel
        pending_entry.summary_reply = summary_reply
        pending_entry.summary_tool = summary_tool
        pending_entry.status = panel_payload['status']
        pending_entry.last_result = {
            'status': panel_payload['status'],
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
        }
        pending_entry.context = {k: v for k, v in context_updates.items() if v is not None}
        pending_entry.quote_params = copy.deepcopy(relay_payload)
        pending_entry.quote_id = request_id
        self._pending[conversation_id] = pending_entry

        result = SwapResult(
            status=panel_payload['status'],
            payload=panel_payload,
            panel=panel,
            summary_reply=summary_reply,
            summary_tool=summary_tool,
            message=None,
            tx=primary_tx,
        )
        output_dict: Dict[str, Any] = {
            'status': result.status,
            'panel': result.panel,
            'summary_reply': result.summary_reply,
            'summary_tool': result.summary_tool,
        }
        if result.tx:
            output_dict['tx'] = result.tx
        if transactions:
            output_dict['transactions'] = transactions
        if approvals:
            output_dict['approvals'] = approvals
        if signatures:
            output_dict['signatures'] = signatures
        return output_dict

    def _is_swap_query(self, message: str) -> bool:
        return any(keyword in message for keyword in SWAP_KEYWORDS)

    def _is_swap_followup(self, message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False
        if any(keyword in message for keyword in SWAP_FOLLOWUP_KEYWORDS):
            return True
        if stripped in {'yes', 'y', 'ok', 'okay', 'please', 'sure'}:
            return True
        return False

    def _normalize_text(self, message: str) -> str:
        normalized = message.lower()
        replacements = {
            '->': ' to ',
            '➡️': ' to ',
            ' into ': ' to ',
            ' in to ': ' to ',
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _detect_chain(self, message: str) -> Optional[int]:
        for alias, chain_id in CHAIN_ALIAS_TO_ID.items():
            if alias in message:
                return chain_id
        for chain_id, meta in CHAIN_METADATA.items():
            name = meta.get('name', '').lower()
            if name and name in message:
                return chain_id
        return None

    def _chain_from_default(self, chain_name: Optional[str]) -> Optional[int]:
        if not chain_name:
            return None
        lowered = chain_name.lower()
        return DEFAULT_CHAIN_NAME_TO_ID.get(lowered)

    def _chain_name(self, chain_id: int) -> str:
        meta = CHAIN_METADATA.get(chain_id)
        if not meta:
            return f'Chain {chain_id}'
        return str(meta.get('name', f'Chain {chain_id}'))

    def _resolve_token(self, chain_id: int, token_hint: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token_hint:
            return None
        alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})
        symbol = alias_map.get(token_hint.lower())
        if not symbol:
            global_symbol = GLOBAL_TOKEN_ALIASES.get(token_hint.lower())
            if global_symbol and global_symbol in TOKEN_REGISTRY.get(chain_id, {}):
                symbol = global_symbol
        if not symbol:
            return None
        token_meta = TOKEN_REGISTRY[chain_id][symbol]
        return {
            'symbol': token_meta['symbol'],
            'address': token_meta['address'],
            'decimals': token_meta['decimals'],
            'is_native': token_meta.get('is_native', False),
        }

    def _supported_tokens_string(self, chain_id: int) -> str:
        tokens = TOKEN_REGISTRY.get(chain_id, {})
        return ', '.join(tokens.keys()) if tokens else 'ETH'

    def _parse_swap_request(self, message: str) -> Tuple[Optional[Decimal], Optional[str], Optional[str]]:
        amount: Optional[Decimal] = None
        token_in: Optional[str] = None
        token_out: Optional[str] = None

        amount_match = re.search(r'(?:swap|trade|convert|exchange)\s+(?P<amount>\d+(?:\.\d+)?)', message)
        if amount_match:
            amount = self._to_decimal(amount_match.group('amount'))

        pattern = re.search(r'(?:swap|trade|convert|exchange)\s+(?:\d+(?:\.\d+)?\s*)?(?P<from>[a-zA-Z0-9]{2,10})\s+to\s+(?P<to>[a-zA-Z0-9]{2,10})', message)
        if pattern:
            token_in = pattern.group('from').upper()
            token_out = pattern.group('to').upper()
        else:
            tokens_ordered: List[str] = []
            for candidate in re.findall(r'[a-zA-Z]{2,10}', message):
                lower = candidate.lower()
                if lower in GLOBAL_TOKEN_ALIASES:
                    tokens_ordered.append(GLOBAL_TOKEN_ALIASES[lower])
            if len(tokens_ordered) >= 2:
                token_in, token_out = tokens_ordered[0], tokens_ordered[1]

        if amount is None:
            amount_match_generic = re.search(r'\b(\d+(?:\.\d+)?)\b', message)
            if amount_match_generic:
                amount = self._to_decimal(amount_match_generic.group(1))

        return amount, token_in, token_out

    def _sanitize_amount(self, value: Decimal) -> Optional[Decimal]:
        try:
            return Decimal(value).quantize(Decimal('1.000000000000000000'), rounding=ROUND_DOWN)
        except (InvalidOperation, TypeError, ValueError):
            return None

    def _to_decimal(self, raw: Any) -> Optional[Decimal]:
        if raw is None:
            return None
        try:
            return Decimal(str(raw))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def _to_base_units(self, amount: Decimal, decimals: int) -> Optional[int]:
        try:
            scaled = (amount * (Decimal(10) ** decimals)).quantize(Decimal('1'), rounding=ROUND_DOWN)
        except (InvalidOperation, ValueError):
            return None
        if scaled <= 0:
            return None
        try:
            return int(scaled)
        except (OverflowError, ValueError):
            return None

    def _decimal_to_str(self, value: Optional[Decimal]) -> str:
        if value is None:
            return '—'
        q = value.normalize()
        return format(q, 'f').rstrip('0').rstrip('.') if '.' in format(q, 'f') else format(q, 'f')

    def _format_eta_seconds(self, seconds: Optional[Any]) -> Optional[str]:
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

    def _amount_to_decimal(self, raw: Any, decimals_hint: Any) -> Optional[Decimal]:
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

    def _total_fee_usd(self, fee_dict: Dict[str, Any]) -> Optional[Decimal]:
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

    def _extract_steps(
        self,
        steps_raw: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized_steps: List[Dict[str, Any]] = []
        transactions: List[Dict[str, Any]] = []
        approvals: List[Dict[str, Any]] = []
        signatures: List[Dict[str, Any]] = []

        for step in steps_raw:
            items = []
            for item in step.get('items', []):
                item_entry = {'status': item.get('status'), 'data': item.get('data')}
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

            step_entry = {
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

