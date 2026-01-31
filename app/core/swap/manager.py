"""SwapManager orchestrates Relay-powered token swaps and Jupiter swaps for Solana."""

from __future__ import annotations

import copy
import logging
import re
from decimal import Decimal, InvalidOperation, ROUND_DOWN, DivisionByZero
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...providers.relay import RelayProvider
from ...providers.coingecko import CoingeckoProvider
from ...providers.jupiter import (
    JupiterSwapProvider,
    JupiterQuote,
    JupiterSwapResult,
    JupiterQuoteError,
    JupiterSwapError,
    get_jupiter_swap_provider,
    NATIVE_SOL_MINT,
)
from ...services.tokens import get_token_service, TokenService
from ...types.requests import ChatRequest
from ..bridge.constants import NATIVE_PLACEHOLDER
from ..bridge.chain_registry import get_registry_sync, get_chain_registry, ChainId
from .constants import (
    GLOBAL_TOKEN_ALIASES,
    SWAP_FOLLOWUP_KEYWORDS,
    SWAP_KEYWORDS,
    SWAP_SOURCE,
    TOKEN_ALIAS_MAP,
    TOKEN_REGISTRY,
    SOLANA_CHAIN_ID,
    is_solana_chain,
)
from .models import SwapResult, SwapState, SolanaSwapQuote


JUPITER_SWAP_SOURCE = {'name': 'Jupiter', 'url': 'https://jup.ag'}


class SwapManager:
    """Encapsulates swap intent parsing, quoting, and response shaping.

    Supports both EVM swaps (via Relay) and Solana swaps (via Jupiter).
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        price_provider: Optional[CoingeckoProvider] = None,
        jupiter_provider: Optional[JupiterSwapProvider] = None,
        token_service: Optional[TokenService] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._pending: Dict[str, SwapState] = {}
        self._price_provider = price_provider or CoingeckoProvider()
        self._jupiter_provider = jupiter_provider
        self._token_service = token_service or get_token_service()

    async def maybe_handle(
        self,
        request: ChatRequest,
        conversation_id: str,
        *,
        wallet_address: Optional[str],
        default_chain: Optional[str],
        portfolio_tokens: Optional[List[Dict[str, Any]]] = None,
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

        # Merge portfolio token metadata from current context and latest tool data
        portfolio_tokens_index = self._build_portfolio_token_index(context.get('portfolio_tokens'))
        fresh_portfolio_tokens = self._build_portfolio_token_index(portfolio_tokens)
        if fresh_portfolio_tokens:
            portfolio_tokens_index.update(fresh_portfolio_tokens)
        if portfolio_tokens_index:
            context['portfolio_tokens'] = portfolio_tokens_index

        portfolio_alias_map = self._build_portfolio_alias_map(portfolio_tokens_index) if portfolio_tokens_index else {}

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

        # Ensure chain registry is loaded before chain detection
        # This is async-safe: if already loaded, returns immediately
        registry = get_registry_sync()
        if not registry.is_loaded:
            try:
                await get_chain_registry()  # Async load if not initialized
            except Exception as e:
                self._logger.warning(f"Failed to load chain registry: {e}")

        # Detect chains - check for cross-chain swap first
        origin_chain, destination_chain = self._detect_cross_chain(normalized_message)

        # Determine default/context chain for fallback
        context_chain = context.get('chain_id')
        default_chain_id = self._chain_from_default(default_chain)
        fallback_chain = context_chain or default_chain_id or 1

        # Fill in missing chains from context/default
        if origin_chain is None and destination_chain is not None:
            # Destination specified but not origin - use context/default as origin
            origin_chain = fallback_chain
        elif destination_chain is None and origin_chain is not None:
            # Origin specified but not destination - use context/default as destination
            destination_chain = fallback_chain

        is_cross_chain = origin_chain is not None and destination_chain is not None and origin_chain != destination_chain

        if is_cross_chain:
            origin_chain_id = origin_chain
            destination_chain_id = destination_chain
        else:
            # Single-chain swap - use detected chain or fallback
            detected_chain = self._detect_chain(normalized_message)
            chain_id = detected_chain or fallback_chain
            origin_chain_id = chain_id
            destination_chain_id = chain_id

        # Validate origin chain is supported (check TokenService first, fallback to TOKEN_REGISTRY)
        origin_supported = await self._token_service.is_supported(origin_chain_id) or origin_chain_id in TOKEN_REGISTRY
        if not origin_supported:
            return finalize_result(
                'unsupported_chain',
                message=f'Chain {self._chain_name(origin_chain_id)} is not yet supported for swaps. Try Ethereum, Base, Arbitrum, or Solana.',
                context_updates={'wallet_address': wallet},
            )

        # Validate destination chain for cross-chain swaps
        dest_supported = await self._token_service.is_supported(destination_chain_id) or destination_chain_id in TOKEN_REGISTRY
        if is_cross_chain and not dest_supported:
            return finalize_result(
                'unsupported_chain',
                message=f'Chain {self._chain_name(destination_chain_id)} is not yet supported for swaps. Try Ethereum, Base, Arbitrum, or Solana.',
                context_updates={'wallet_address': wallet},
            )

        token_in_symbol = context.get('token_in_symbol')
        token_out_symbol = context.get('token_out_symbol')
        amount_decimal = self._to_decimal(context.get('input_amount')) if context.get('input_amount') else None

        percent_fraction: Optional[Decimal] = None

        parsed_amount, parsed_token_in, parsed_token_out, amount_currency = self._parse_swap_request(
            normalized_message,
            portfolio_alias_map,
        )
        usd_amount: Optional[Decimal] = None

        if parsed_token_in:
            token_in_symbol = parsed_token_in
        if parsed_token_out:
            token_out_symbol = parsed_token_out
        if parsed_amount is not None:
            if amount_currency == 'USD':
                usd_amount = parsed_amount
            elif amount_currency == 'PERCENT':
                percent_fraction = parsed_amount
            else:
                amount_decimal = parsed_amount

        # Resolve tokens against their respective chains
        token_in_meta = await self._resolve_token(
            origin_chain_id,
            token_in_symbol,
            portfolio_tokens_index,
            portfolio_alias_map,
        ) if token_in_symbol else None
        token_out_meta = await self._resolve_token(
            destination_chain_id,
            token_out_symbol,
            portfolio_tokens_index if not is_cross_chain else None,  # Don't use portfolio for destination chain in cross-chain
            portfolio_alias_map if not is_cross_chain else None,
        ) if token_out_symbol else None

        if token_in_meta is None or token_out_meta is None:
            if is_cross_chain:
                origin_supported = await self._supported_tokens_string(origin_chain_id, portfolio_tokens_index)
                dest_supported = await self._supported_tokens_string(destination_chain_id, None)
                return finalize_result(
                    'needs_token',
                    message=f'I need valid tokens for this cross-chain swap. On {self._chain_name(origin_chain_id)}: {origin_supported}. On {self._chain_name(destination_chain_id)}: {dest_supported}.',
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )
            else:
                supported = await self._supported_tokens_string(origin_chain_id, portfolio_tokens_index)
                return finalize_result(
                    'needs_token',
                    message=f'I need the tokens for this swap. Supported examples on {self._chain_name(origin_chain_id)}: {supported}. Try "swap 0.25 ETH to USDC".',
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

        # For same-chain swaps, tokens must be different (compare by address, not symbol)
        # For cross-chain swaps, same token symbol is allowed (bridging)
        if not is_cross_chain:
            # Compare by address to handle cases like USDC.e vs USDC on same chain
            in_addr = str(token_in_meta.get('address', '')).lower()
            out_addr = str(token_out_meta.get('address', '')).lower()
            if in_addr and out_addr and in_addr == out_addr:
                return finalize_result(
                    'needs_token',
                    message='The input and output tokens are the same. Please choose two different assets for the swap.',
                    context_updates={'wallet_address': wallet, 'chain_id': origin_chain_id},
                )

        if amount_decimal is None and percent_fraction is not None:
            portfolio_entry = None
            if portfolio_tokens_index:
                portfolio_entry = portfolio_tokens_index.get(token_in_meta['symbol'])
            balance_decimal = None
            if portfolio_entry:
                balance_decimal = portfolio_entry.get('balance_decimal')
                if balance_decimal is None:
                    balance_decimal = self._to_decimal(portfolio_entry.get('balance_formatted'))
            if balance_decimal is None or balance_decimal <= Decimal('0'):
                pretty_percent = self._decimal_to_str(percent_fraction * Decimal('100'))
                return finalize_result(
                    'needs_amount',
                    message=(
                        f"I'm not sure how much {token_in_meta['symbol']} you own to calculate {pretty_percent}% for this swap. "
                        f"Please share the exact {token_in_meta['symbol']} amount or refresh your portfolio data."
                    ),
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'],
                        'token_out_symbol': token_out_meta['symbol'],
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

            computed_amount = balance_decimal * percent_fraction
            amount_decimal = self._sanitize_amount(computed_amount)
            if amount_decimal is None or amount_decimal <= Decimal('0'):
                pretty_percent = self._decimal_to_str(percent_fraction * Decimal('100'))
                return finalize_result(
                    'needs_amount',
                    message=(
                        f"{pretty_percent}% of your {token_in_meta['symbol']} balance is too small to swap. "
                        f"Please specify a larger amount."
                    ),
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'],
                        'token_out_symbol': token_out_meta['symbol'],
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

        if amount_decimal is None and usd_amount is not None:
            if usd_amount <= Decimal('0'):
                return finalize_result(
                    'needs_amount',
                    message='The USD amount looks invalid. Please share a positive dollar amount or specify the token amount directly.',
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'],
                        'token_out_symbol': token_out_meta['symbol'],
                    },
                )
            converted_amount = await self._convert_usd_to_input_amount(usd_amount, token_in_meta)
            if converted_amount is None:
                pretty_usd = self._decimal_to_str(usd_amount)
                return finalize_result(
                    'needs_amount',
                    message=(
                        f"I couldn't convert ${pretty_usd} to {token_in_meta['symbol']} right now. "
                        f"Please specify the {token_in_meta['symbol']} amount or try again shortly."
                    ),
                    context_updates={
                        'wallet_address': wallet,
                        'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'],
                        'token_out_symbol': token_out_meta['symbol'],
                    },
                )
            amount_decimal = converted_amount

        if amount_decimal is None:
            return finalize_result(
                'needs_amount',
                message=f'How much {token_in_meta["symbol"]} should I swap? For example: "swap 0.5 {token_in_meta["symbol"]} to {token_out_meta["symbol"]}".',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        amount_decimal = self._sanitize_amount(amount_decimal)
        if amount_decimal is None or amount_decimal <= Decimal('0'):
            return finalize_result(
                'needs_amount',
                message='The swap amount looks invalid or too small. Try a larger value.',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        amount_base_units = self._to_base_units(amount_decimal, int(token_in_meta['decimals']))
        if amount_base_units is None or amount_base_units <= 0:
            return finalize_result(
                'needs_amount',
                message='The swap amount is too small after accounting for token decimals. Try a larger amount.',
                context_updates={
                    'wallet_address': wallet,
                    'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'],
                    'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        # Dispatch to Jupiter for Solana swaps (only for same-chain Solana)
        if not is_cross_chain and is_solana_chain(origin_chain_id):
            return await self._handle_solana_swap(
                wallet=wallet,
                token_in_meta=token_in_meta,
                token_out_meta=token_out_meta,
                amount_decimal=amount_decimal,
                conversation_id=conversation_id,
                context=context,
                latest=latest,
                usd_amount=usd_amount,
                percent_fraction=percent_fraction,
            )

        # EVM swap/bridge via Relay
        amount_base_units_str = str(amount_base_units)

        relay_payload: Dict[str, Any] = {
            'user': wallet,
            'originChainId': origin_chain_id,
            'destinationChainId': destination_chain_id,
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
            'chain_id': origin_chain_id,
            'destination_chain_id': destination_chain_id if is_cross_chain else None,
            'chain': self._chain_name(origin_chain_id),
            'destination_chain': self._chain_name(destination_chain_id) if is_cross_chain else None,
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
            panel_payload['amounts']['input_share_percent'] = self._decimal_to_str(percent_fraction * Decimal('100'))
        if usd_amount is not None:
            panel_payload['amounts']['input_usd'] = self._decimal_to_str(usd_amount)

        context_updates = {
            'wallet_address': wallet,
            'chain_id': origin_chain_id,
            'destination_chain_id': destination_chain_id if is_cross_chain else None,
            'token_in_symbol': token_in_meta['symbol'],
            'token_out_symbol': token_out_meta['symbol'],
            'input_amount': str(amount_decimal),
            'token_in_address': token_in_meta['address'],
            'token_out_address': token_out_meta['address'],
            'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
        }
        if usd_amount is not None:
            context_updates['input_amount_usd'] = str(usd_amount)
        if percent_fraction is not None:
            context_updates['input_amount_percent'] = self._decimal_to_str(percent_fraction * Decimal('100'))

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
        if usd_amount is not None:
            try:
                panel_payload['usd_estimates']['input_requested'] = float(usd_amount)
            except (TypeError, ValueError):
                panel_payload['usd_estimates']['input_requested'] = None
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

        # Build summary message - include both chains for cross-chain swaps
        if is_cross_chain:
            swap_summary = f"✅ Bridge {self._decimal_to_str(routed_from_amount or amount_decimal)} {input_symbol} ({self._chain_name(origin_chain_id)}) → {output_symbol} ({self._chain_name(destination_chain_id)})"
        else:
            swap_summary = f"✅ Swap {self._decimal_to_str(routed_from_amount or amount_decimal)} {input_symbol} → {output_symbol} on {self._chain_name(origin_chain_id)}"
        summary_lines = [swap_summary]
        if usd_amount is not None:
            try:
                summary_lines.insert(0, f"🎯 Target ≈ ${float(usd_amount):.2f} of {input_symbol}")
            except (TypeError, ValueError):
                summary_lines.insert(0, f"🎯 Target amount ≈ ${self._decimal_to_str(usd_amount)} of {input_symbol}")
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
        if is_cross_chain:
            summary_tool = f"Relay bridge: {self._decimal_to_str(amount_decimal)} {input_symbol} ({self._chain_name(origin_chain_id)}) → {output_symbol} ({self._chain_name(destination_chain_id)})"
        else:
            summary_tool = f"Relay swap: {self._decimal_to_str(amount_decimal)} {input_symbol} → {output_symbol} on {self._chain_name(origin_chain_id)}"

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

    def _detect_chain(self, message: str) -> Optional[ChainId]:
        """Detect chain from message using dynamic registry."""
        registry = get_registry_sync()
        return registry.detect_chain_in_text(message)

    def _detect_cross_chain(self, message: str) -> Tuple[Optional[ChainId], Optional[ChainId]]:
        """Detect origin and destination chains for cross-chain swaps.

        Parses patterns like:
        - "from ink to mainnet"
        - "on ink to ethereum"
        - "USDC.e from ink to USDC on mainnet"

        Returns (origin_chain_id, destination_chain_id). Either may be None.
        """
        registry = get_registry_sync()

        # Detect origin chain (from X, on X before "to")
        origin_chain = registry.detect_chain_with_preposition(message, ["from"])
        if origin_chain is None:
            # Check for "on <chain>" pattern before "to" keyword
            on_match = re.search(r'\bon\s+(\w+)\s+to\b', message, re.IGNORECASE)
            if on_match:
                origin_chain = registry.get_chain_id(on_match.group(1))

        # Detect destination chain (to X, on X after token)
        destination_chain = registry.detect_chain_with_preposition(message, ["to"])
        if destination_chain is None:
            # Check for "to <token> on <chain>" pattern
            on_match = re.search(r'\bto\s+\w+\s+on\s+(\w+)', message, re.IGNORECASE)
            if on_match:
                destination_chain = registry.get_chain_id(on_match.group(1))

        # If only one chain detected, check if "to" preposition points to a chain
        # after a token symbol (e.g., "swap USDC.e to USDC on mainnet")
        if destination_chain is None and origin_chain is not None:
            # The "to" might be for the token, not chain. Check for trailing chain.
            trailing_chain = re.search(r'\b(?:on|to)\s+(\w+)\s*$', message, re.IGNORECASE)
            if trailing_chain:
                potential_chain = registry.get_chain_id(trailing_chain.group(1))
                if potential_chain is not None and potential_chain != origin_chain:
                    destination_chain = potential_chain

        return (origin_chain, destination_chain)

    def _chain_from_default(self, chain_name: Optional[str]) -> Optional[ChainId]:
        """Resolve default chain name to chain ID using registry."""
        if not chain_name:
            return None
        registry = get_registry_sync()
        return registry.get_chain_id(chain_name)

    def _chain_name(self, chain_id: ChainId) -> str:
        """Get chain name from registry."""
        registry = get_registry_sync()
        return registry.get_chain_name(chain_id)

    async def _supported_tokens_string(
        self,
        chain_id: ChainId,
        extra_tokens: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        # Get tokens from TokenService (Convex-backed)
        service_symbols = await self._token_service.get_supported_symbols(chain_id)
        tokens = {s.upper() for s in service_symbols}

        # Fallback to TOKEN_REGISTRY for backward compatibility
        for symbol in TOKEN_REGISTRY.get(chain_id, {}).keys():
            tokens.add(symbol.upper())

        # Add extra tokens from portfolio
        if extra_tokens:
            for symbol in extra_tokens.keys():
                tokens.add(symbol.upper())

        default_token = 'SOL' if is_solana_chain(chain_id) else 'ETH'
        return ', '.join(sorted(tokens)) if tokens else default_token

    def _parse_swap_request(
        self,
        message: str,
        extra_aliases: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Decimal], Optional[str], Optional[str], Optional[str]]:
        amount: Optional[Decimal] = None
        token_in: Optional[str] = None
        token_out: Optional[str] = None
        amount_currency: Optional[str] = None

        qualifier = r'(?:about|around|roughly|approximately)\s+'
        usd_prefix_pattern = re.search(
            rf'(?:swap|trade|convert|exchange)\s+(?:{qualifier})?\$(?P<usd>\d+(?:\.\d+)?)',
            message,
        )
        if usd_prefix_pattern:
            amount = self._to_decimal(usd_prefix_pattern.group('usd'))
            amount_currency = 'USD'
        else:
            usd_suffix_pattern = re.search(
                rf'(?:swap|trade|convert|exchange)\s+(?:{qualifier})?(?P<usd>\d+(?:\.\d+)?)\s*(usd|dollars?)\b',
                message,
            )
            if usd_suffix_pattern:
                amount = self._to_decimal(usd_suffix_pattern.group('usd'))
                amount_currency = 'USD'

        if amount is None:
            percent_pattern = re.search(
                r'(?:swap|trade|convert|exchange)[^\d%]*?(?P<percent>\d+(?:\.\d+)?)\s*(?:percent|pct|%)',
                message,
            )
            if percent_pattern:
                percent_value = self._to_decimal(percent_pattern.group('percent'))
                if percent_value is not None:
                    try:
                        amount = (percent_value / Decimal('100')).quantize(Decimal('1.000000000000000000'), rounding=ROUND_DOWN)
                        amount_currency = 'PERCENT'
                    except (InvalidOperation, DivisionByZero):
                        amount = None
                        amount_currency = None

        if amount is None:
            amount_match = re.search(r'(?:swap|trade|convert|exchange)\s+(?P<amount>\d+(?:\.\d+)?)', message)
            if amount_match:
                amount = self._to_decimal(amount_match.group('amount'))
                if amount is not None:
                    amount_currency = 'TOKEN'

        alias_map = dict(GLOBAL_TOKEN_ALIASES)
        if extra_aliases:
            alias_map.update({k.lower(): v for k, v in extra_aliases.items()})

        pattern = re.search(
            r'(?:swap|trade|convert|exchange)\s+(?:\$?\d+(?:\.\d+)?\s*(?:usd|dollars?)?\s*)?(?:of\s+)?(?P<from>[a-zA-Z0-9$]{2,15})\s+to\s+(?P<to>[a-zA-Z0-9$]{2,15})',
            message,
        )
        if pattern:
            raw_in = pattern.group('from').lstrip('$')
            raw_out = pattern.group('to').lstrip('$')
            token_in = alias_map.get(raw_in.lower()) or raw_in.upper()
            token_out = alias_map.get(raw_out.lower()) or raw_out.upper()
        else:
            tokens_ordered: List[str] = []
            for candidate in re.findall(r'[a-zA-Z0-9$]{2,20}', message):
                cleaned = candidate.lstrip('$').lower()
                if cleaned in alias_map:
                    tokens_ordered.append(alias_map[cleaned])
            if len(tokens_ordered) >= 2:
                token_in, token_out = tokens_ordered[0], tokens_ordered[1]

        if amount is None:
            amount_match_generic = re.search(r'\b(\d+(?:\.\d+)?)\b', message)
            if amount_match_generic:
                amount = self._to_decimal(amount_match_generic.group(1))
                if amount is not None and amount_currency is None:
                    amount_currency = 'TOKEN'

        return amount, token_in, token_out, amount_currency

    async def _resolve_token(
        self,
        chain_id: ChainId,
        token_hint: Optional[str],
        extra_tokens: Optional[Dict[str, Dict[str, Any]]] = None,
        extra_aliases: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not token_hint:
            return None

        hint_lower = token_hint.lower()

        # 1. Try TokenService first (Convex-backed, includes aliases)
        # Convert extra_tokens dict to list format for portfolio_tokens param
        portfolio_list = None
        if extra_tokens:
            portfolio_list = [
                {**meta, 'chain_id': chain_id}
                for meta in extra_tokens.values()
            ]

        token_config = await self._token_service.resolve_token(
            chain_id,
            token_hint,
            portfolio_tokens=portfolio_list,
        )

        if token_config:
            return {
                'symbol': token_config.symbol,
                'address': token_config.address,
                'decimals': token_config.decimals,
                'is_native': token_config.is_native,
                'name': token_config.name,
            }

        # 2. Fallback: check extra_aliases from portfolio
        symbol = None
        if extra_aliases:
            symbol = extra_aliases.get(hint_lower)

        # 3. Fallback: check global aliases
        if not symbol:
            global_symbol = GLOBAL_TOKEN_ALIASES.get(hint_lower)
            if global_symbol:
                symbol = global_symbol

        # 4. Fallback: use hint as symbol
        if not symbol:
            symbol = token_hint.upper()

        # 5. Check extra_tokens (portfolio)
        if extra_tokens and symbol in extra_tokens:
            token_meta = extra_tokens[symbol]
            return {
                'symbol': token_meta['symbol'],
                'address': token_meta['address'],
                'decimals': token_meta['decimals'],
                'is_native': token_meta.get('is_native', False),
                'name': token_meta.get('name'),
            }

        # 6. Final fallback: TOKEN_REGISTRY (deprecated, for backward compat)
        registry = TOKEN_REGISTRY.get(chain_id, {})
        if symbol not in registry:
            # Try alias map as last resort
            alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})
            resolved_symbol = alias_map.get(hint_lower)
            if resolved_symbol and resolved_symbol in registry:
                symbol = resolved_symbol
            else:
                return None

        token_meta = registry.get(symbol)
        if not token_meta:
            return None

        return {
            'symbol': token_meta['symbol'],
            'address': token_meta['address'],
            'decimals': token_meta['decimals'],
            'is_native': token_meta.get('is_native', False),
        }

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

    async def _convert_usd_to_input_amount(
        self,
        usd_amount: Decimal,
        token_meta: Dict[str, Any],
    ) -> Optional[Decimal]:
        try:
            usd_value = Decimal(usd_amount)
        except (InvalidOperation, TypeError, ValueError):
            return None
        if usd_value <= 0:
            return None

        provider = self._price_provider
        try:
            ready = await provider.ready()
        except Exception as exc:  # pragma: no cover - logging only
            self._logger.warning('Price provider readiness check failed: %s', exc)
            return None
        if not ready:
            self._logger.info('Price provider unavailable; cannot convert USD-denominated swap amount.')
            return None

        price_decimal: Optional[Decimal] = None

        try:
            if token_meta.get('is_native') or str(token_meta.get('address')).lower() == NATIVE_PLACEHOLDER.lower():
                price_info = await provider.get_eth_price()
                if isinstance(price_info, dict):
                    price_val = price_info.get('price_usd')
                    if price_val is not None:
                        price_decimal = Decimal(str(price_val))
            else:
                raw_address_value = token_meta.get('address')
                if not raw_address_value:
                    return None
                address_lower = str(raw_address_value).lower()
                prices = await provider.get_token_prices([address_lower])
                price_entry = prices.get(address_lower) or prices.get(str(raw_address_value))
                if price_entry and price_entry.get('price_usd') is not None:
                    price_decimal = Decimal(str(price_entry['price_usd']))
        except Exception as exc:  # pragma: no cover - logging only
            self._logger.warning('Failed fetching price for USD conversion: %s', exc)
            return None

        if price_decimal is None or price_decimal <= 0:
            return None

        try:
            token_amount = usd_value / price_decimal
        except (InvalidOperation, DivisionByZero, OverflowError):
            return None

        return self._sanitize_amount(token_amount)

    def _build_portfolio_token_index(
        self,
        tokens_source: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        if not tokens_source:
            return {}

        if isinstance(tokens_source, dict):
            sanitized: Dict[str, Dict[str, Any]] = {}
            for key, value in tokens_source.items():
                entry = self._sanitize_portfolio_token(key, value)
                if entry:
                    sanitized[entry['symbol']] = entry
            return sanitized

        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(tokens_source, list):
            for raw in tokens_source:
                entry = self._sanitize_portfolio_token(None, raw)
                if entry:
                    result[entry['symbol']] = entry
        return result

    def _sanitize_portfolio_token(
        self,
        key: Optional[str],
        raw: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        symbol = str(raw.get('symbol') or key or '').strip().upper()
        if not symbol:
            return None

        decimals_raw = raw.get('decimals')
        try:
            decimals = int(decimals_raw)
        except (TypeError, ValueError):
            return None

        address_raw = raw.get('address')
        address = str(address_raw).lower() if isinstance(address_raw, str) else None
        is_native = not address or address == NATIVE_PLACEHOLDER.lower()
        normalized_address = NATIVE_PLACEHOLDER if is_native else address

        name = str(raw.get('name') or symbol)

        entry: Dict[str, Any] = {
            'symbol': symbol,
            'name': name,
            'address': normalized_address,
            'decimals': decimals,
            'is_native': is_native,
        }

        balance_decimal: Optional[Decimal] = None
        for candidate_key in ('balance_decimal', 'balance', 'balance_formatted', 'quantity'):
            candidate_value = raw.get(candidate_key)
            if candidate_value is None:
                continue
            balance_decimal = self._to_decimal(candidate_value)
            if balance_decimal is not None:
                break
        if balance_decimal is not None:
            entry['balance_decimal'] = balance_decimal
            entry['balance_formatted'] = str(balance_decimal)

        balance_wei_raw = raw.get('balance_wei')
        if balance_wei_raw is not None:
            try:
                entry['balance_wei'] = int(str(balance_wei_raw), 10)
            except (TypeError, ValueError):
                pass

        value_usd_raw = raw.get('value_usd')
        if value_usd_raw is not None:
            try:
                entry['value_usd'] = Decimal(str(value_usd_raw))
            except (InvalidOperation, TypeError, ValueError):
                pass

        return entry

    def _build_portfolio_alias_map(
        self,
        portfolio_tokens: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for symbol, meta in portfolio_tokens.items():
            lower_symbol = symbol.lower()
            alias_map[lower_symbol] = symbol
            alias_map[symbol.lstrip('$').lower()] = symbol

            name = meta.get('name')
            if isinstance(name, str) and name:
                alias_map[name.lower()] = symbol
                alias_map[name.replace(' ', '').lower()] = symbol

            address = meta.get('address')
            if isinstance(address, str) and address and address != NATIVE_PLACEHOLDER:
                alias_map[address.lower()] = symbol

        return alias_map

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

    async def _handle_solana_swap(
        self,
        wallet: str,
        token_in_meta: Dict[str, Any],
        token_out_meta: Dict[str, Any],
        amount_decimal: Decimal,
        conversation_id: str,
        context: Dict[str, Any],
        latest: str,
        usd_amount: Optional[Decimal] = None,
        percent_fraction: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Handle a Solana swap via Jupiter.

        Returns a result dict with the Jupiter quote and unsigned transaction.
        """
        input_mint = str(token_in_meta['address'])
        output_mint = str(token_out_meta['address'])
        input_decimals = int(token_in_meta['decimals'])
        output_decimals = int(token_out_meta['decimals'])

        # Convert to lamports/smallest units
        amount_base_units = self._to_base_units(amount_decimal, input_decimals)
        if amount_base_units is None or amount_base_units <= 0:
            return {
                'status': 'needs_amount',
                'message': 'The swap amount is too small for Solana. Try a larger amount.',
            }

        # Get or create Jupiter provider
        jupiter = self._jupiter_provider or get_jupiter_swap_provider()

        panel_payload: Dict[str, Any] = {
            'status': 'pending',
            'chain_id': SOLANA_CHAIN_ID,
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
            panel_payload['amounts']['input_share_percent'] = self._decimal_to_str(percent_fraction * Decimal('100'))
        if usd_amount is not None:
            panel_payload['amounts']['input_usd'] = self._decimal_to_str(usd_amount)

        context_updates = {
            'wallet_address': wallet,
            'chain_id': SOLANA_CHAIN_ID,
            'token_in_symbol': token_in_meta['symbol'],
            'token_out_symbol': token_out_meta['symbol'],
            'input_amount': str(amount_decimal),
            'token_in_address': input_mint,
            'token_out_address': output_mint,
        }
        if usd_amount is not None:
            context_updates['input_amount_usd'] = str(usd_amount)
        if percent_fraction is not None:
            context_updates['input_amount_percent'] = self._decimal_to_str(percent_fraction * Decimal('100'))

        # Get quote from Jupiter
        try:
            quote: JupiterQuote = await jupiter.get_swap_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_base_units,
                slippage_bps=50,  # 0.5% default slippage
            )
        except JupiterQuoteError as exc:
            self._logger.warning('Jupiter quote error: %s', exc)
            panel_payload['status'] = 'error'
            panel_payload.setdefault('issues', []).append(str(exc))
            panel = {
                'id': 'jupiter_swap_quote',
                'kind': 'card',
                'title': f"Jupiter Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
                'payload': panel_payload,
                'sources': [JUPITER_SWAP_SOURCE],
                'metadata': {'status': 'error'},
            }
            return {'status': 'error', 'message': f'Jupiter could not produce a swap route: {exc}', 'panel': panel}

        # Build the swap transaction
        try:
            swap_result: JupiterSwapResult = await jupiter.build_swap_transaction(
                quote=quote,
                user_public_key=wallet,
            )
        except JupiterSwapError as exc:
            self._logger.warning('Jupiter swap build error: %s', exc)
            panel_payload['status'] = 'error'
            panel_payload.setdefault('issues', []).append(str(exc))
            panel = {
                'id': 'jupiter_swap_quote',
                'kind': 'card',
                'title': f"Jupiter Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
                'payload': panel_payload,
                'sources': [JUPITER_SWAP_SOURCE],
                'metadata': {'status': 'error'},
            }
            return {'status': 'error', 'message': f'Jupiter could not build swap transaction: {exc}', 'panel': panel}

        # Calculate output amounts
        output_amount_decimal = Decimal(str(quote.out_amount)) / (Decimal(10) ** output_decimals)
        min_output_decimal = Decimal(str(quote.other_amount_threshold)) / (Decimal(10) ** output_decimals)

        # Update panel with quote details
        panel_payload.update({
            'status': 'ok',
            'tx_ready': True,
            'quote_expiry': None,  # Jupiter quotes are short-lived (~30s)
            'solana_tx': swap_result.swap_transaction,
            'last_valid_block_height': swap_result.last_valid_block_height,
            'priority_fee_lamports': swap_result.priority_fee_lamports,
            'compute_unit_limit': swap_result.compute_unit_limit,
        })

        panel_payload['breakdown'] = {
            'input': {
                'symbol': token_in_meta['symbol'],
                'amount': self._decimal_to_str(amount_decimal),
                'token_address': input_mint,
            },
            'output': {
                'symbol': token_out_meta['symbol'],
                'amount_estimate': self._decimal_to_str(output_amount_decimal),
                'min_amount': self._decimal_to_str(min_output_decimal),
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
            f"✅ Swap {self._decimal_to_str(amount_decimal)} {token_in_meta['symbol']} → {token_out_meta['symbol']} on Solana"
        ]
        if usd_amount is not None:
            summary_lines.insert(0, f"🎯 Target ≈ ${self._decimal_to_str(usd_amount)} of {token_in_meta['symbol']}")
        summary_lines.append(f"Estimated output: {self._decimal_to_str(output_amount_decimal)} {token_out_meta['symbol']}")
        summary_lines.append(f"Minimum output: {self._decimal_to_str(min_output_decimal)} {token_out_meta['symbol']}")
        if quote.price_impact_pct > 0.1:
            summary_lines.append(f"⚠️ Price impact: {quote.price_impact_pct:.2f}%")
        summary_lines.append('Confirm the swap in your connected Solana wallet when prompted.')

        summary_reply = "\n".join(summary_lines)
        summary_tool = f"Jupiter swap plan: {self._decimal_to_str(amount_decimal)} {token_in_meta['symbol']} → {token_out_meta['symbol']} on Solana"

        panel = {
            'id': 'jupiter_swap_quote',
            'kind': 'card',
            'title': f"Jupiter Swap: {token_in_meta['symbol']} → {token_out_meta['symbol']}",
            'payload': panel_payload,
            'sources': [JUPITER_SWAP_SOURCE],
            'metadata': {'status': 'ok', 'chain': 'solana'},
        }

        # Store state for follow-ups
        new_state = SwapState(
            context={k: v for k, v in context_updates.items() if v is not None},
            quote_params={'input_mint': input_mint, 'output_mint': output_mint, 'amount': amount_base_units},
            last_prompt=latest,
            status='ok',
            panel=panel,
            summary_reply=summary_reply,
            summary_tool=summary_tool,
            last_result={'status': 'ok', 'panel': panel},
            solana_tx_base64=swap_result.swap_transaction,
            last_valid_block_height=swap_result.last_valid_block_height,
        )
        self._pending[conversation_id] = new_state

        return {
            'status': 'ok',
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
            'solana_tx': swap_result.swap_transaction,
            'last_valid_block_height': swap_result.last_valid_block_height,
            'is_solana': True,
        }
