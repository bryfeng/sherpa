"""BridgeManager orchestrates bridge quotes and manual build preparation."""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from ...providers.relay import RelayProvider
from ...providers.coingecko import CoingeckoProvider
from ...types.requests import ChatRequest
from .chain_registry import ChainRegistry, ChainId
from .constants import (
    BRIDGE_FOLLOWUP_KEYWORDS,
    BRIDGE_KEYWORDS,
    BRIDGE_SOURCE,
    ETH_UNITS,
    NATIVE_PLACEHOLDER,
    USD_UNITS,
)
from .models import BridgeResult, BridgeState


class BridgeManager:
    """Encapsulates bridge intent parsing, quoting, and manual build preparation.

    Uses ChainRegistry for dynamic chain support instead of hardcoded metadata.
    """

    def __init__(
        self,
        *,
        chain_registry: Optional[ChainRegistry] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._registry = chain_registry or ChainRegistry()
        self._pending: Dict[str, BridgeState] = {}
        self._eth_price_cache: Optional[Tuple[Dict[str, Any], datetime]] = None
        self._coingecko = CoingeckoProvider()

    async def maybe_handle(
        self,
        request: ChatRequest,
        conversation_id: str,
        *,
        wallet_address: Optional[str],
        default_chain: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Primary entry point; mirrors legacy ``_maybe_fetch_bridge_quote`` behaviour."""

        if not request.messages:
            return None

        latest = request.messages[-1].content.strip()
        if not latest:
            return None

        # Ensure chain registry is loaded before processing
        registry_ready = await self._registry.ensure_loaded()
        if not registry_ready:
            self._logger.warning("Chain registry not available, bridge detection may be limited")

        normalized_message = self._normalize_bridge_text(latest)
        message_lower = normalized_message
        pending_state = self._pending.get(conversation_id)

        explicit_bridge = self._is_bridge_query(message_lower)
        followup = False
        chain_hint: Optional[int] = None
        if not explicit_bridge and pending_state:
            followup = self._is_bridge_followup(message_lower)
            if not followup and pending_state.status == 'needs_chain':
                chain_hint = self._detect_chain_anywhere(message_lower)
                if chain_hint is not None:
                    followup = True

        if not explicit_bridge and not followup:
            return None

        state = pending_state or BridgeState()
        context = dict(state.context)

        user_address = wallet_address or context.get('user_address')

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

            new_state = BridgeState(
                context=merged_context,
                last_prompt=latest,
                status=status,
                last_result=result,
                panel=panel,
                summary_reply=summary_reply,
                summary_tool=summary_tool,
                quote_params=state.quote_params,
                price=state.price,
                quote_id=state.quote_id,
                route_request_hash=state.route_request_hash,
            )
            self._pending[conversation_id] = new_state
            return result

        if not user_address:
            return finalize_result(
                'needs_address',
                message='I can prep a Relay bridge, but I need the wallet address that will sign it.',
            )

        from_chain_id = context.get('from_chain_id')
        to_chain_id = context.get('to_chain_id')
        if explicit_bridge:
            chain_params = self._infer_bridge_params(message_lower, default_chain)
            from_chain_id = chain_params.get('from_chain_id') or from_chain_id
            to_chain_id = chain_params.get('to_chain_id') or to_chain_id
        elif chain_hint is not None and to_chain_id is None:
            to_chain_id = chain_hint
        if from_chain_id is None:
            from_chain_id = 1
        if to_chain_id is None:
            # List supported chains dynamically
            supported = self._registry.get_supported_chain_names(limit=8)
            chain_list = ", ".join(supported) if supported else "Base, Arbitrum, Optimism, Polygon"
            return finalize_result(
                'needs_chain',
                message=f'Which chain should I bridge to? I support: {chain_list}. Try "bridge … to [chain name]".',
                context_updates={'user_address': user_address, 'from_chain_id': from_chain_id},
            )
        context.update({'user_address': user_address, 'from_chain_id': from_chain_id, 'to_chain_id': to_chain_id})

        amount_eth: Optional[Decimal] = None
        amount_usd: Optional[Decimal] = None
        price_info = state.price

        if explicit_bridge:
            amount_result = await self._resolve_bridge_amount(message_lower)
            if amount_result.get('status') != 'ok':
                status = amount_result.get('status', 'needs_amount')
                if status == 'price_unavailable':
                    msg = 'I could not fetch the current ETH price to convert the USD amount. Try again soon or specify the amount in ETH.'
                elif status == 'needs_amount':
                    msg = 'Tell me how much to bridge — e.g., “bridge 0.05 ETH” or “bridge $25 worth of ETH to Base”.'
                else:
                    msg = 'Unable to determine the bridge amount from that request.'
                return finalize_result(status, message=msg)
            amount_eth = amount_result['amount_eth']
            amount_usd = amount_result.get('amount_usd')
            price_info = amount_result.get('price')
        else:
            if context.get('amount_eth') is not None:
                try:
                    amount_eth = Decimal(str(context['amount_eth']))
                except (InvalidOperation, TypeError, ValueError):
                    amount_eth = None
            if context.get('amount_usd') is not None:
                try:
                    amount_usd = Decimal(str(context['amount_usd']))
                except (InvalidOperation, TypeError, ValueError):
                    amount_usd = None

        if amount_eth is None:
            return finalize_result(
                'needs_amount',
                message='Tell me how much to bridge — e.g., “bridge 0.05 ETH” or “bridge $25 worth of ETH to Base”.',
            )

        from_decimals_hint = self._native_decimals(from_chain_id)
        try:
            amount_wei_dec = (amount_eth * (Decimal(10) ** from_decimals_hint)).quantize(Decimal('1'), rounding=ROUND_DOWN)
        except (InvalidOperation, ValueError):
            return finalize_result(
                'needs_amount',
                message='The bridge amount looks invalid. Try a slightly larger value.',
            )
        if amount_wei_dec <= 0:
            return finalize_result(
                'needs_amount',
                message='The bridge amount is too small to execute.',
            )

        amount_wei = str(int(amount_wei_dec))
        amount_eth_str = self._decimal_to_str(amount_eth)
        amount_usd_str = self._decimal_to_str(amount_usd) if amount_usd is not None else None
        context.update({'amount_eth': amount_eth_str, 'amount_usd': amount_usd_str, 'amount_wei': amount_wei})

        from_symbol = self._chain_native_symbol(from_chain_id)
        to_symbol = self._chain_native_symbol(to_chain_id)

        input_token_address = self._native_token_address(from_chain_id)
        output_token_address = self._native_token_address(to_chain_id)


        relay_payload = {
            'user': user_address,
            'originChainId': from_chain_id,
            'destinationChainId': to_chain_id,
            'originCurrency': input_token_address,
            'destinationCurrency': output_token_address,
            'recipient': user_address,
            'tradeType': 'EXACT_INPUT',
            'amount': amount_wei,
            'referrer': 'sherpa.chat',
            'useExternalLiquidity': False,
            'useDepositAddress': False,
            'topupGas': False,
        }

        panel_payload: Dict[str, Any] = {
            'from_chain_id': from_chain_id,
            'from_chain': self._chain_name(from_chain_id),
            'to_chain_id': to_chain_id,
            'to_chain': self._chain_name(to_chain_id),
            'amounts': {
                'requested_eth': amount_eth_str,
                'requested_usd': amount_usd_str,
                'input_amount_wei': amount_wei,
            },
            'price_reference': {
                'usd': str(price_info['price']) if price_info and price_info.get('price') else None,
                'source': price_info.get('source') if price_info else None,
            },
            'status': 'pending',
            'wallet': {'address': user_address},
            'provider': 'relay',
            'relay_request': relay_payload,
            'quote_type': 'bridge',
        }

        pending_entry = BridgeState(
            context={k: v for k, v in context.items() if v is not None},
            quote_params=copy.deepcopy(relay_payload),
            price=price_info,
            last_prompt=latest,
        )
        self._pending[conversation_id] = pending_entry

        provider = RelayProvider()

        try:
            quote_response = await provider.quote(relay_payload)
        except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
            status_code = exc.response.status_code
            detail_text = ''
            try:
                detail_text = (exc.response.text or '').strip()
            except Exception:
                detail_text = ''
            preview = detail_text[:200] if detail_text else 'No response details from Relay.'
            message = f"Relay quote failed with status {status_code}: {preview}"
            self._logger.warning('Relay quote error (%s): payload=%s detail=%s', status_code, relay_payload, detail_text)
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            if detail_text:
                panel_payload['error_detail'] = detail_text
            summary = (
                f"Bridge {amount_eth_str} {from_symbol} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}."
                f"\n⚠️ {message}"
            )
            panel = {
                'id': 'relay_bridge_quote',
                'kind': 'card',
                'title': f"Relay Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error', 'http_status': status_code},
            }
            return finalize_result(
                'error',
                message=message,
                panel=panel,
                summary_reply=summary,
                summary_tool=summary,
                extra={'detail': str(exc)},
            )
        except Exception as exc:  # pragma: no cover
            self._logger.error("Relay quote failed: %s", exc)
            message = 'I could not reach Relay to fetch a bridge route. Please try again shortly.'
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            summary = (
                f"Bridge {amount_eth_str} {from_symbol} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}."
                f"\n⚠️ {message}"
            )
            panel = {
                'id': 'relay_bridge_quote',
                'kind': 'card',
                'title': f"Relay Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result(
                'error',
                message=message,
                panel=panel,
                summary_reply=summary,
                summary_tool=summary,
                extra={'detail': str(exc)},
            )

        if not isinstance(quote_response, dict) or not quote_response.get('steps'):
            message = 'Relay did not return any route data for that request.'
            panel_payload.setdefault('issues', []).append(message)
            panel_payload['status'] = 'error'
            panel = {
                'id': 'relay_bridge_quote',
                'kind': 'card',
                'title': f"Relay Bridge: {from_symbol} → {to_symbol}",
                'payload': panel_payload,
                'sources': [BRIDGE_SOURCE],
                'metadata': {'status': 'error'},
            }
            return finalize_result('error', message=message, panel=panel)

        steps_raw = quote_response.get('steps') or []
        fees = quote_response.get('fees') or {}
        details = quote_response.get('details') or {}

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
            for extra_key in ('requestId', 'depositAddress'):
                if step.get(extra_key):
                    step_entry[extra_key] = step[extra_key]
            normalized_steps.append(step_entry)

        request_id = None
        if normalized_steps:
            request_id = normalized_steps[0].get('requestId')
        if not request_id:
            request_id = quote_response.get('requestId') or quote_response.get('id')

        primary_tx_entry = transactions[0] if transactions else None
        primary_tx = primary_tx_entry.get('data') if primary_tx_entry else None

        input_currency = details.get('currencyIn') or {}
        output_currency = details.get('currencyOut') or {}

        input_symbol = input_currency.get('currency', {}).get('symbol', from_symbol)
        output_symbol = output_currency.get('currency', {}).get('symbol', to_symbol)

        def _amount_to_decimal(raw: Any, decimals_hint: int) -> Optional[Decimal]:
            if raw is None:
                return None
            try:
                return Decimal(str(raw)) / (Decimal(10) ** int(decimals_hint))
            except (InvalidOperation, TypeError, ValueError):
                return None

        input_decimals = input_currency.get('currency', {}).get('decimals', self._native_decimals(from_chain_id))
        output_decimals = output_currency.get('currency', {}).get('decimals', self._native_decimals(to_chain_id))

        routed_from_amount = _amount_to_decimal(input_currency.get('amount'), input_decimals) or amount_eth
        routed_to_amount = _amount_to_decimal(output_currency.get('amount'), output_decimals)

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
        eta_readable = self._format_eta_minutes(eta_seconds)

        def _total_fee_usd(fee_dict: Dict[str, Any]) -> Optional[Decimal]:
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

        total_fee_usd = _total_fee_usd(fees)

        route_info = details.get('route') or {}
        route_name = route_info.get('destination', {}).get('router') or route_info.get('origin', {}).get('router') or 'Relay'
        if not route_name or route_name.lower() in {'relay', '0x'}:
            route_name = 'Relay'

        panel_payload.update({
            'status': 'ok' if transactions else 'quote_only',
            'request_id': request_id,
            'steps': normalized_steps,
            'transactions': transactions,
            'fees': fees,
            'details': details,
            'eta_seconds': eta_seconds,
            'tx_ready': bool(primary_tx),
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
                'token_address': input_currency.get('currency', {}).get('address', input_token_address),
            },
            'output': {
                'symbol': output_symbol,
                'amount_estimate': output_amount_display,
                'token_address': output_currency.get('currency', {}).get('address', output_token_address),
                'value_usd': output_usd,
            },
            'fees': {
                'total_usd': float(total_fee_usd) if total_fee_usd is not None else None,
                'gas_usd': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else None,
                'slippage_percent': details.get('slippageTolerance', {}).get('destination', {}).get('percent'),
            },
        }
        panel_payload['breakdown']['eta_seconds'] = eta_seconds
        panel_payload['usd_estimates'] = {
            'output': output_usd,
            'gas': fees.get('gas', {}).get('amountUsd') if isinstance(fees.get('gas'), dict) else (float(total_fee_usd) if total_fee_usd is not None else None),
        }
        panel_payload['instructions'] = [
            'Review the Relay quote details including expected output, fees, and ETA.',
            f'Confirm the transaction from {self._chain_name(from_chain_id)} in your connected wallet.',
            'If Relay prompts for additional approvals or signatures, complete them before executing the bridge.',
            'Quotes change quickly; ask “refresh bridge quote” if you need updated pricing or the transaction fails.',
        ]
        panel_payload['actions'] = {
            'refresh_quote': 'Say “refresh bridge quote” to fetch an updated price.',
            'open_wallet': 'Use your connected wallet to review and submit the prepared transaction.',
        }

        summary_lines = [
            f"✅ {self._decimal_to_str(routed_from_amount)} {input_symbol} from {self._chain_name(from_chain_id)} → {self._chain_name(to_chain_id)}"
        ]
        if routed_to_amount is not None:
            arrival_line = f"Estimated arrival: {self._decimal_to_str(routed_to_amount)} {output_symbol}"
            if output_usd is not None:
                try:
                    arrival_line += f" (~${float(output_usd):.2f})"
                except (ValueError, TypeError):
                    pass
            summary_lines.append(arrival_line)
        if total_fee_usd is not None:
            try:
                summary_lines.append(f"Relay fees ≈ ${float(total_fee_usd):.2f}")
            except (ValueError, TypeError):
                pass
        if eta_readable:
            summary_lines.append(f"ETA ≈ {eta_readable}")
        summary_lines.append(f"Route: {route_name}")
        if signatures:
            summary_lines.append('You will be prompted to sign an additional message before the bridge transaction.')
        summary_lines.append('Confirm the transaction in your connected wallet when the prompt appears.')
        summary_lines.append('Need updated pricing? Ask me to refresh the bridge quote before signing.')

        summary_reply = "\n".join(summary_lines)
        summary_tool = (
            f"Relay bridge plan: {self._decimal_to_str(routed_from_amount)} {input_symbol} → "
            f"{output_symbol} on {self._chain_name(to_chain_id)}"
        )

        panel_metadata = {
            'status': panel_payload['status'],
            'request_id': request_id,
            'has_transactions': bool(transactions),
            'needs_signature': bool(signatures),
        }

        panel = {
            'id': 'relay_bridge_quote',
            'kind': 'card',
            'title': f"Relay Bridge: {input_symbol} → {output_symbol}",
            'payload': panel_payload,
            'sources': [BRIDGE_SOURCE],
            'metadata': panel_metadata,
        }

        pending_entry.status = panel_payload['status']
        pending_entry.panel = panel
        pending_entry.summary_reply = summary_reply
        pending_entry.summary_tool = summary_tool
        pending_entry.last_result = {
            'status': panel_payload['status'],
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
        }
        pending_entry.context = context
        pending_entry.quote_id = request_id
        pending_entry.route_request_hash = request_id
        self._pending[conversation_id] = pending_entry

        result = BridgeResult(
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

    def _is_bridge_query(self, message: str) -> bool:
        if re.search(r'\bbridge\b', message):
            return True
        return any(keyword in message for keyword in BRIDGE_KEYWORDS)

    def _is_bridge_followup(self, message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False
        if any(keyword in message for keyword in BRIDGE_FOLLOWUP_KEYWORDS):
            return True
        if stripped in {'yes', 'y', 'yep', 'sure', 'ok', 'okay', 'please', 'please do'}:
            return True
        return False

    def _normalize_bridge_text(self, message: str) -> str:
        normalized = message.lower()
        replacements = {
            'main net': 'mainnet',
            'main-net': 'mainnet',
            'ethereum main net': 'ethereum mainnet',
            'layer 1': 'mainnet',
            'layer-1': 'mainnet',
            'layer1': 'mainnet',
        }
        for src, dest in replacements.items():
            normalized = normalized.replace(src, dest)
        return normalized

    def _detect_chain_anywhere(self, message: str) -> Optional[ChainId]:
        """Detect any chain reference in the message using dynamic registry."""
        return self._registry.detect_chain_in_text(message)

    def _infer_bridge_params(self, message: str, default_chain: Optional[str]) -> Dict[str, Optional[ChainId]]:
        msg = message.lower()
        to_chain_id = self._detect_chain(msg, ['to', 'onto', 'into', 'towards'])
        if to_chain_id is None:
            to_chain_id = self._detect_chain_arrow(msg)

        # Resolve default chain using registry
        default_chain_id: Optional[ChainId] = None
        if default_chain:
            default_chain_id = self._registry.get_chain_id(default_chain)
        if default_chain_id is None:
            default_chain_id = 1  # Ethereum as fallback

        from_chain_id = self._detect_chain(msg, ['from', 'off', 'out of'])
        if from_chain_id is None:
            candidate = self._detect_chain(msg, ['on'])
            if candidate and candidate != to_chain_id:
                from_chain_id = candidate
        if from_chain_id is None:
            from_chain_id = default_chain_id

        # If destination still not found, look for any chain mention that isn't the source
        if to_chain_id is None:
            alias_map = self._registry.get_all_aliases()
            for alias, chain_id in alias_map.items():
                # Skip common Ethereum aliases when looking for destination
                if alias in {'eth', 'ethereum', 'mainnet', 'l1', 'layer1', 'layer 1'}:
                    continue
                if alias in msg and chain_id != from_chain_id:
                    to_chain_id = chain_id
                    break

        return {'from_chain_id': from_chain_id, 'to_chain_id': to_chain_id}
 
    def _detect_chain(self, message: str, keywords: List[str]) -> Optional[ChainId]:
        """Detect chain following specific keywords (e.g., 'to base', 'from ethereum')."""
        return self._registry.detect_chain_with_preposition(message, keywords)

    def _detect_chain_arrow(self, message: str) -> Optional[ChainId]:
        """Detect chain after arrow notation (e.g., '->base', '-> ink')."""
        alias_map = self._registry.get_all_aliases()
        for alias, chain_id in alias_map.items():
            needle = f'->{alias}'
            if needle in message or f'-> {alias}' in message:
                return chain_id
        return None

    async def _resolve_bridge_amount(self, message: str) -> Dict[str, Any]:
        parsed = self._parse_bridge_amount(message)
        unit = parsed.get('unit')
        amount_eth = parsed.get('amount_eth')
        amount_usd = parsed.get('amount_usd')

        if unit is None or (amount_eth is None and amount_usd is None):
            return {'status': 'needs_amount'}

        if unit == 'usd':
            price_info = await self._get_eth_price_usd()
            if not price_info or not price_info.get('price'):
                return {'status': 'price_unavailable'}
            price = price_info['price']
            try:
                amount_eth = (amount_usd / price).quantize(Decimal('1e-18'), rounding=ROUND_DOWN)
            except (InvalidOperation, TypeError):
                return {'status': 'needs_amount'}
            if amount_eth <= 0:
                return {'status': 'needs_amount'}
            return {
                'status': 'ok',
                'amount_eth': amount_eth,
                'amount_usd': amount_usd,
                'price': price_info,
            }

        if amount_eth is None or amount_eth <= 0:
            return {'status': 'needs_amount'}

        price_info = await self._get_eth_price_usd()
        usd_value = None
        if price_info and price_info.get('price'):
            try:
                usd_value = (amount_eth * price_info['price']).quantize(Decimal('1e-2'), rounding=ROUND_DOWN)
            except (InvalidOperation, TypeError):
                usd_value = None

        return {
            'status': 'ok',
            'amount_eth': amount_eth,
            'amount_usd': usd_value,
            'price': price_info,
        }

    def _parse_bridge_amount(self, message: str) -> Dict[str, Optional[Decimal]]:
        pattern = r'(\$)?\s*(\d+(?:\.\d+)?)\s*(usd|usdc|dollar|dollars|buck|bucks|eth|weth)?'
        for match in re.finditer(pattern, message):
            symbol, number, unit = match.groups()
            try:
                value = Decimal(number)
            except (InvalidOperation, TypeError):
                continue
            if value <= 0:
                continue
            unit_lower = unit.lower() if unit else None
            if symbol or (unit_lower and unit_lower in USD_UNITS):
                return {'amount_usd': value, 'amount_eth': None, 'unit': 'usd'}
            if unit_lower and unit_lower in ETH_UNITS:
                return {'amount_eth': value, 'amount_usd': None, 'unit': 'eth'}

        return {'amount_eth': None, 'amount_usd': None, 'unit': None}

    async def _get_eth_price_usd(self) -> Optional[Dict[str, Any]]:
        try:
            if self._eth_price_cache:
                cached_value, cached_time = self._eth_price_cache
                if (datetime.now() - cached_time).total_seconds() < 60:
                    return cached_value
        except Exception:
            pass

        try:
            if not await self._coingecko.ready():
                return None
            data = await self._coingecko.get_eth_price()
        except Exception as exc:  # pragma: no cover
            self._logger.debug("ETH price fetch failed: %s", exc)
            return None

        price = data.get('price_usd')
        if price is None:
            return None

        price_info = {
            'price': Decimal(str(price)),
            'source': data.get('_source'),
        }
        self._eth_price_cache = (price_info, datetime.now())
        return price_info

    def _decimal_to_str(self, value: Optional[Decimal], places: int = 6) -> str:
        if value is None:
            return '0'
        precision = max(0, min(places, 18))
        quant = Decimal('1') if precision == 0 else Decimal(1).scaleb(-precision)
        try:
            quantized = value.quantize(quant, rounding=ROUND_DOWN)
        except (InvalidOperation, TypeError):
            quantized = value
        formatted = format(quantized.normalize(), 'f')
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted or '0'

    def _chain_name(self, chain_id: ChainId) -> str:
        """Get human-readable chain name from registry."""
        return self._registry.get_chain_name(chain_id)

    def _chain_native_symbol(self, chain_id: ChainId) -> str:
        """Get native token symbol from registry."""
        return self._registry.get_native_symbol(chain_id)

    def _format_eta_minutes(self, seconds: Optional[Any]) -> Optional[str]:
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

    def _native_token_address(self, chain_id: ChainId) -> str:
        """Get native token address from registry."""
        return self._registry.get_native_token_address(chain_id)

    def _native_decimals(self, chain_id: ChainId) -> int:
        """Get native token decimals from registry."""
        return self._registry.get_native_decimals(chain_id)
