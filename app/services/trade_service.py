"""TradeService — orchestrates: parse > resolve > quote > validate > panel."""

from __future__ import annotations

import copy
import logging
from decimal import Decimal, InvalidOperation, ROUND_DOWN, DivisionByZero
from typing import Any, Dict, List, Optional

from ..core.bridge.constants import NATIVE_PLACEHOLDER
from ..core.bridge.chain_registry import get_registry_sync, get_chain_registry, ChainId
from ..core.swap.constants import (
    GLOBAL_TOKEN_ALIASES,
    SOLANA_CHAIN_ID,
    TOKEN_ALIAS_MAP,
    TOKEN_REGISTRY,
    is_solana_chain,
)
from ..core.swap.models import SwapResult, SwapState
from ..providers.coingecko import CoingeckoProvider
from ..providers.jupiter import (
    JupiterSwapProvider,
    get_jupiter_swap_provider,
)
from ..services.tokens import get_token_service, TokenService
from ..types.requests import ChatRequest

from . import intent_parser
from . import quote_aggregator
from . import panel_builder


logger = logging.getLogger(__name__)


# Equivalent token mapping for bridged variants
EQUIVALENT_TOKENS = {
    "usdc.e": "usdc",
    "usdc.b": "usdc",
    "usdce": "usdc",
    "usdt.e": "usdt",
    "usdt.b": "usdt",
    "weth.e": "weth",
    "dai.e": "dai",
}


class TradeService:
    """Orchestrates swap/bridge flows: parse intent, resolve tokens, fetch quotes, build panels."""

    def __init__(
        self,
        *,
        log: Optional[logging.Logger] = None,
        price_provider: Optional[CoingeckoProvider] = None,
        jupiter_provider: Optional[JupiterSwapProvider] = None,
        token_service: Optional[TokenService] = None,
    ) -> None:
        self._logger = log or logging.getLogger(__name__)
        self._pending: Dict[str, SwapState] = {}
        self._price_provider = price_provider or CoingeckoProvider()
        self._jupiter_provider = jupiter_provider
        self._token_service = token_service or get_token_service()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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

        normalized_message = intent_parser.normalize_text(latest)
        state = self._pending.get(conversation_id)

        explicit_swap = intent_parser.is_swap_query(normalized_message)
        followup = False
        if not explicit_swap and state:
            followup = intent_parser.is_swap_followup(normalized_message)

        if not explicit_swap and not followup:
            return None

        context = dict(state.context) if state else {}

        portfolio_tokens_index = _build_portfolio_token_index(context.get('portfolio_tokens'))
        fresh_portfolio_tokens = _build_portfolio_token_index(portfolio_tokens)
        if fresh_portfolio_tokens:
            portfolio_tokens_index.update(fresh_portfolio_tokens)
        if portfolio_tokens_index:
            context['portfolio_tokens'] = portfolio_tokens_index

        portfolio_alias_map = _build_portfolio_alias_map(portfolio_tokens_index) if portfolio_tokens_index else {}

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

        # Ensure chain registry is loaded
        registry = get_registry_sync()
        if not registry.is_loaded:
            try:
                await get_chain_registry()
            except Exception as e:
                self._logger.warning(f"Failed to load chain registry: {e}")

        # Detect chains
        origin_chain, destination_chain = intent_parser.detect_cross_chain(normalized_message)

        context_chain = context.get('chain_id')
        default_chain_id = intent_parser.chain_from_default(default_chain)
        fallback_chain = context_chain or default_chain_id or 1

        if origin_chain is None and destination_chain is not None:
            origin_chain = fallback_chain
        elif destination_chain is None and origin_chain is not None:
            destination_chain = fallback_chain

        is_cross_chain = origin_chain is not None and destination_chain is not None and origin_chain != destination_chain

        if is_cross_chain:
            origin_chain_id = origin_chain
            destination_chain_id = destination_chain
        else:
            detected_chain = intent_parser.detect_chain(normalized_message)
            chain_id = detected_chain or fallback_chain
            origin_chain_id = chain_id
            destination_chain_id = chain_id

        # Validate chains
        origin_supported = await self._token_service.is_supported(origin_chain_id) or origin_chain_id in TOKEN_REGISTRY
        if not origin_supported:
            return finalize_result(
                'unsupported_chain',
                message=f'Chain {intent_parser.chain_name(origin_chain_id)} is not yet supported for swaps. Try Ethereum, Base, Arbitrum, or Solana.',
                context_updates={'wallet_address': wallet},
            )

        dest_supported = await self._token_service.is_supported(destination_chain_id) or destination_chain_id in TOKEN_REGISTRY
        if is_cross_chain and not dest_supported:
            return finalize_result(
                'unsupported_chain',
                message=f'Chain {intent_parser.chain_name(destination_chain_id)} is not yet supported for swaps. Try Ethereum, Base, Arbitrum, or Solana.',
                context_updates={'wallet_address': wallet},
            )

        # Parse intent
        token_in_symbol = context.get('token_in_symbol')
        token_out_symbol = context.get('token_out_symbol')
        amount_decimal = _to_decimal(context.get('input_amount')) if context.get('input_amount') else None

        percent_fraction: Optional[Decimal] = None

        parsed_amount, parsed_token_in, parsed_token_out, amount_currency = intent_parser.parse_swap_request(
            normalized_message, portfolio_alias_map,
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

        # Resolve tokens
        token_in_meta = await self._resolve_token(
            origin_chain_id, token_in_symbol, portfolio_tokens_index, portfolio_alias_map,
        ) if token_in_symbol else None
        token_out_meta = await self._resolve_token(
            destination_chain_id, token_out_symbol,
            portfolio_tokens_index if not is_cross_chain else None,
            portfolio_alias_map if not is_cross_chain else None,
            allow_equivalent=is_cross_chain,
        ) if token_out_symbol else None

        if token_in_meta is None or token_out_meta is None:
            if is_cross_chain:
                origin_sup = await self._supported_tokens_string(origin_chain_id, portfolio_tokens_index)
                dest_sup = await self._supported_tokens_string(destination_chain_id, None)
                return finalize_result(
                    'needs_token',
                    message=f'I need valid tokens for this cross-chain swap. On {intent_parser.chain_name(origin_chain_id)}: {origin_sup}. On {intent_parser.chain_name(destination_chain_id)}: {dest_sup}.',
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )
            else:
                supported = await self._supported_tokens_string(origin_chain_id, portfolio_tokens_index)
                return finalize_result(
                    'needs_token',
                    message=f'I need the tokens for this swap. Supported examples on {intent_parser.chain_name(origin_chain_id)}: {supported}. Try "swap 0.25 ETH to USDC".',
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

        # Same-token guard
        if not is_cross_chain:
            in_addr = str(token_in_meta.get('address', '')).lower()
            out_addr = str(token_out_meta.get('address', '')).lower()
            if in_addr and out_addr and in_addr == out_addr:
                return finalize_result(
                    'needs_token',
                    message='The input and output tokens are the same. Please choose two different assets for the swap.',
                    context_updates={'wallet_address': wallet, 'chain_id': origin_chain_id},
                )

        # Resolve percent-based amounts
        if amount_decimal is None and percent_fraction is not None:
            portfolio_entry = portfolio_tokens_index.get(token_in_meta['symbol']) if portfolio_tokens_index else None
            balance_decimal = None
            if portfolio_entry:
                balance_decimal = portfolio_entry.get('balance_decimal')
                if balance_decimal is None:
                    balance_decimal = _to_decimal(portfolio_entry.get('balance_formatted'))
            if balance_decimal is None or balance_decimal <= Decimal('0'):
                pretty_percent = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))
                return finalize_result(
                    'needs_amount',
                    message=(
                        f"I'm not sure how much {token_in_meta['symbol']} you own to calculate {pretty_percent}% for this swap. "
                        f"Please share the exact {token_in_meta['symbol']} amount or refresh your portfolio data."
                    ),
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

            computed_amount = balance_decimal * percent_fraction
            amount_decimal = _sanitize_amount(computed_amount)
            if amount_decimal is None or amount_decimal <= Decimal('0'):
                pretty_percent = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))
                return finalize_result(
                    'needs_amount',
                    message=f"{pretty_percent}% of your {token_in_meta['symbol']} balance is too small to swap. Please specify a larger amount.",
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                        'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                    },
                )

        # Resolve USD amounts
        if amount_decimal is None and usd_amount is not None:
            if usd_amount <= Decimal('0'):
                return finalize_result(
                    'needs_amount',
                    message='The USD amount looks invalid. Please share a positive dollar amount or specify the token amount directly.',
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                    },
                )
            converted_amount = await self._convert_usd_to_input_amount(usd_amount, token_in_meta)
            if converted_amount is None:
                pretty_usd = panel_builder.decimal_to_str(usd_amount)
                return finalize_result(
                    'needs_amount',
                    message=(
                        f"I couldn't convert ${pretty_usd} to {token_in_meta['symbol']} right now. "
                        f"Please specify the {token_in_meta['symbol']} amount or try again shortly."
                    ),
                    context_updates={
                        'wallet_address': wallet, 'chain_id': origin_chain_id,
                        'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                    },
                )
            amount_decimal = converted_amount

        if amount_decimal is None:
            return finalize_result(
                'needs_amount',
                message=f'How much {token_in_meta["symbol"]} should I swap? For example: "swap 0.5 {token_in_meta["symbol"]} to {token_out_meta["symbol"]}".',
                context_updates={
                    'wallet_address': wallet, 'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        amount_decimal = _sanitize_amount(amount_decimal)
        if amount_decimal is None or amount_decimal <= Decimal('0'):
            return finalize_result(
                'needs_amount',
                message='The swap amount looks invalid or too small. Try a larger value.',
                context_updates={
                    'wallet_address': wallet, 'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        amount_base_units = _to_base_units(amount_decimal, int(token_in_meta['decimals']))
        if amount_base_units is None or amount_base_units <= 0:
            return finalize_result(
                'needs_amount',
                message='The swap amount is too small after accounting for token decimals. Try a larger amount.',
                context_updates={
                    'wallet_address': wallet, 'chain_id': origin_chain_id,
                    'token_in_symbol': token_in_meta['symbol'], 'token_out_symbol': token_out_meta['symbol'],
                    'portfolio_tokens': portfolio_tokens_index if portfolio_tokens_index else None,
                },
            )

        # ---- Dispatch to Jupiter for same-chain Solana swaps ----
        if not is_cross_chain and is_solana_chain(origin_chain_id):
            return await self._handle_solana_swap(
                wallet=wallet,
                token_in_meta=token_in_meta,
                token_out_meta=token_out_meta,
                amount_decimal=amount_decimal,
                amount_base_units=amount_base_units,
                conversation_id=conversation_id,
                context=context,
                latest=latest,
                usd_amount=usd_amount,
                percent_fraction=percent_fraction,
            )

        # ---- EVM swap/bridge via Relay ----
        return await self._handle_relay_swap(
            wallet=wallet,
            token_in_meta=token_in_meta,
            token_out_meta=token_out_meta,
            amount_decimal=amount_decimal,
            amount_base_units=amount_base_units,
            origin_chain_id=origin_chain_id,
            destination_chain_id=destination_chain_id,
            is_cross_chain=is_cross_chain,
            conversation_id=conversation_id,
            context=context,
            latest=latest,
            usd_amount=usd_amount,
            percent_fraction=percent_fraction,
            portfolio_tokens_index=portfolio_tokens_index,
            state=state,
            finalize_result=finalize_result,
        )

    # ------------------------------------------------------------------
    # Relay (EVM) swap handler
    # ------------------------------------------------------------------

    async def _handle_relay_swap(
        self,
        *,
        wallet: str,
        token_in_meta: Dict[str, Any],
        token_out_meta: Dict[str, Any],
        amount_decimal: Decimal,
        amount_base_units: int,
        origin_chain_id: ChainId,
        destination_chain_id: ChainId,
        is_cross_chain: bool,
        conversation_id: str,
        context: Dict[str, Any],
        latest: str,
        usd_amount: Optional[Decimal],
        percent_fraction: Optional[Decimal],
        portfolio_tokens_index: Optional[Dict[str, Dict[str, Any]]],
        state: Optional[SwapState],
        finalize_result,
    ) -> Dict[str, Any]:
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

        base_panel_payload: Dict[str, Any] = {
            'status': 'pending',
            'chain_id': origin_chain_id,
            'destination_chain_id': destination_chain_id if is_cross_chain else None,
            'chain': intent_parser.chain_name(origin_chain_id),
            'destination_chain': intent_parser.chain_name(destination_chain_id) if is_cross_chain else None,
            'is_cross_chain': is_cross_chain,
            'wallet': {'address': wallet},
            'provider': 'relay',
            'relay_request': relay_payload,
            'quote_type': 'bridge' if is_cross_chain else 'swap',
            'tokens': {'input': token_in_meta, 'output': token_out_meta},
            'amounts': {'input': str(amount_decimal), 'input_base_units': amount_base_units_str},
        }
        if percent_fraction is not None:
            base_panel_payload['amounts']['input_share_percent'] = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))
        if usd_amount is not None:
            base_panel_payload['amounts']['input_usd'] = panel_builder.decimal_to_str(usd_amount)

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
            context_updates['input_amount_percent'] = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))

        pending_entry = SwapState(
            context={k: v for k, v in context_updates.items() if v is not None},
            quote_params=copy.deepcopy(relay_payload),
            last_prompt=latest,
        )
        self._pending[conversation_id] = pending_entry

        # Fetch quote
        qr = await quote_aggregator.fetch_relay_quote(relay_payload, logger=self._logger)
        if not qr.ok:
            if qr.is_network_error:
                return finalize_result('error', message=qr.error_message, panel=None, context_updates=context_updates)
            panel = panel_builder.build_relay_error_panel(
                token_in_meta=token_in_meta, token_out_meta=token_out_meta,
                panel_payload=base_panel_payload, error_detail=qr.error_detail, error_message=qr.error_message or '',
            )
            return finalize_result('error', message=qr.error_message, panel=panel, context_updates=context_updates)

        quote_response = qr.quote_response
        steps_raw = quote_response.get('steps') or []
        normalized_steps, transactions, approvals, signatures = quote_aggregator.extract_relay_steps(steps_raw)

        panel_payload, panel, summary_reply, summary_tool = panel_builder.build_relay_panel(
            token_in_meta=token_in_meta,
            token_out_meta=token_out_meta,
            amount_decimal=amount_decimal,
            origin_chain_id=origin_chain_id,
            destination_chain_id=destination_chain_id,
            is_cross_chain=is_cross_chain,
            wallet=wallet,
            relay_payload=relay_payload,
            amount_base_units_str=amount_base_units_str,
            usd_amount=usd_amount,
            percent_fraction=percent_fraction,
            chain_name_fn=intent_parser.chain_name,
            quote_response=quote_response,
            normalized_steps=normalized_steps,
            transactions=transactions,
            approvals=approvals,
            signatures=signatures,
        )

        primary_tx_entry = transactions[0] if transactions else None
        primary_tx = primary_tx_entry.get('data') if primary_tx_entry else None

        request_id = None
        if normalized_steps:
            request_id = normalized_steps[0].get('requestId')
        if not request_id:
            request_id = quote_response.get('requestId') or quote_response.get('id')

        pending_entry.panel = panel
        pending_entry.summary_reply = summary_reply
        pending_entry.summary_tool = summary_tool
        pending_entry.status = panel_payload['status']
        pending_entry.last_result = {
            'status': panel_payload['status'], 'panel': panel,
            'summary_reply': summary_reply, 'summary_tool': summary_tool,
        }
        pending_entry.context = {k: v for k, v in context_updates.items() if v is not None}
        pending_entry.quote_params = copy.deepcopy(relay_payload)
        pending_entry.quote_id = request_id
        self._pending[conversation_id] = pending_entry

        output_dict: Dict[str, Any] = {
            'status': panel_payload['status'],
            'panel': panel,
            'summary_reply': summary_reply,
            'summary_tool': summary_tool,
        }
        if primary_tx:
            output_dict['tx'] = primary_tx
        if transactions:
            output_dict['transactions'] = transactions
        if approvals:
            output_dict['approvals'] = approvals
        if signatures:
            output_dict['signatures'] = signatures
        return output_dict

    # ------------------------------------------------------------------
    # Jupiter (Solana) swap handler
    # ------------------------------------------------------------------

    async def _handle_solana_swap(
        self,
        *,
        wallet: str,
        token_in_meta: Dict[str, Any],
        token_out_meta: Dict[str, Any],
        amount_decimal: Decimal,
        amount_base_units: int,
        conversation_id: str,
        context: Dict[str, Any],
        latest: str,
        usd_amount: Optional[Decimal] = None,
        percent_fraction: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        input_mint = str(token_in_meta['address'])
        output_mint = str(token_out_meta['address'])
        output_decimals = int(token_out_meta['decimals'])

        jupiter = self._jupiter_provider or get_jupiter_swap_provider()

        base_panel_payload: Dict[str, Any] = {
            'status': 'pending',
            'chain_id': SOLANA_CHAIN_ID,
            'chain': 'Solana',
            'wallet': {'address': wallet},
            'provider': 'jupiter',
            'quote_type': 'swap',
            'tokens': {'input': token_in_meta, 'output': token_out_meta},
            'amounts': {'input': str(amount_decimal), 'input_base_units': str(amount_base_units)},
        }
        if percent_fraction is not None:
            base_panel_payload['amounts']['input_share_percent'] = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))
        if usd_amount is not None:
            base_panel_payload['amounts']['input_usd'] = panel_builder.decimal_to_str(usd_amount)

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
            context_updates['input_amount_percent'] = panel_builder.decimal_to_str(percent_fraction * Decimal('100'))

        # Fetch quote
        jqr = await quote_aggregator.fetch_jupiter_quote(
            input_mint=input_mint, output_mint=output_mint,
            amount_base_units=amount_base_units, wallet=wallet,
            slippage_bps=50, jupiter_provider=jupiter, logger=self._logger,
        )

        if not jqr.ok:
            panel = panel_builder.build_jupiter_error_panel(
                token_in_meta=token_in_meta, token_out_meta=token_out_meta,
                panel_payload=base_panel_payload, error_message=jqr.error_message or '',
            )
            return {'status': 'error', 'message': jqr.error_message, 'panel': panel}

        quote = jqr.quote
        swap_result = jqr.swap_result

        output_amount_decimal = Decimal(str(quote.out_amount)) / (Decimal(10) ** output_decimals)
        min_output_decimal = Decimal(str(quote.other_amount_threshold)) / (Decimal(10) ** output_decimals)

        panel_payload, panel, summary_reply, summary_tool = panel_builder.build_jupiter_panel(
            token_in_meta=token_in_meta,
            token_out_meta=token_out_meta,
            amount_decimal=amount_decimal,
            wallet=wallet,
            input_mint=input_mint,
            output_mint=output_mint,
            output_amount_decimal=output_amount_decimal,
            min_output_decimal=min_output_decimal,
            quote=quote,
            swap_result=swap_result,
            usd_amount=usd_amount,
            percent_fraction=percent_fraction,
            amount_base_units=amount_base_units,
            solana_chain_id=SOLANA_CHAIN_ID,
        )

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

    # ------------------------------------------------------------------
    # Token resolution helpers
    # ------------------------------------------------------------------

    async def _resolve_token(
        self,
        chain_id: ChainId,
        token_hint: Optional[str],
        extra_tokens: Optional[Dict[str, Dict[str, Any]]] = None,
        extra_aliases: Optional[Dict[str, str]] = None,
        allow_equivalent: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not token_hint:
            return None

        hint_lower = token_hint.lower()
        hints_to_try = [hint_lower]
        if allow_equivalent and hint_lower in EQUIVALENT_TOKENS:
            hints_to_try.append(EQUIVALENT_TOKENS[hint_lower])

        portfolio_list = None
        if extra_tokens:
            portfolio_list = [{**meta, 'chain_id': chain_id} for meta in extra_tokens.values()]

        for current_hint in hints_to_try:
            token_config = await self._token_service.resolve_token(
                chain_id, current_hint, portfolio_tokens=portfolio_list,
            )
            if token_config:
                return {
                    'symbol': token_config.symbol,
                    'address': token_config.address,
                    'decimals': token_config.decimals,
                    'is_native': token_config.is_native,
                    'name': token_config.name,
                }

            symbol = None
            if extra_aliases:
                symbol = extra_aliases.get(current_hint)
            if not symbol:
                global_symbol = GLOBAL_TOKEN_ALIASES.get(current_hint)
                if global_symbol:
                    symbol = global_symbol
            if not symbol:
                symbol = current_hint.upper()

            if extra_tokens and symbol in extra_tokens:
                token_meta = extra_tokens[symbol]
                return {
                    'symbol': token_meta['symbol'],
                    'address': token_meta['address'],
                    'decimals': token_meta['decimals'],
                    'is_native': token_meta.get('is_native', False),
                    'name': token_meta.get('name'),
                }

            registry = TOKEN_REGISTRY.get(chain_id, {})
            if symbol in registry:
                token_meta = registry.get(symbol)
                if token_meta:
                    return {
                        'symbol': token_meta['symbol'],
                        'address': token_meta['address'],
                        'decimals': token_meta['decimals'],
                        'is_native': token_meta.get('is_native', False),
                    }

            alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})
            resolved_symbol = alias_map.get(current_hint)
            if resolved_symbol and resolved_symbol in registry:
                token_meta = registry.get(resolved_symbol)
                if token_meta:
                    return {
                        'symbol': token_meta['symbol'],
                        'address': token_meta['address'],
                        'decimals': token_meta['decimals'],
                        'is_native': token_meta.get('is_native', False),
                    }

        return None

    async def _supported_tokens_string(
        self,
        chain_id: ChainId,
        extra_tokens: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        service_symbols = await self._token_service.get_supported_symbols(chain_id)
        tokens = {s.upper() for s in service_symbols}
        for symbol in TOKEN_REGISTRY.get(chain_id, {}).keys():
            tokens.add(symbol.upper())
        if extra_tokens:
            for symbol in extra_tokens.keys():
                tokens.add(symbol.upper())
        default_token = 'SOL' if is_solana_chain(chain_id) else 'ETH'
        return ', '.join(sorted(tokens)) if tokens else default_token

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
        except Exception as exc:
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
        except Exception as exc:
            self._logger.warning('Failed fetching price for USD conversion: %s', exc)
            return None

        if price_decimal is None or price_decimal <= 0:
            return None

        try:
            token_amount = usd_value / price_decimal
        except (InvalidOperation, DivisionByZero, OverflowError):
            return None

        return _sanitize_amount(token_amount)


# ------------------------------------------------------------------
# Module-level helpers (stateless)
# ------------------------------------------------------------------

def _to_decimal(raw: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _sanitize_amount(value: Decimal) -> Optional[Decimal]:
    try:
        return Decimal(value).quantize(Decimal('1.000000000000000000'), rounding=ROUND_DOWN)
    except (InvalidOperation, TypeError, ValueError):
        return None


def _to_base_units(amount: Decimal, decimals: int) -> Optional[int]:
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


def _build_portfolio_token_index(
    tokens_source: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    if not tokens_source:
        return {}

    if isinstance(tokens_source, dict):
        sanitized: Dict[str, Dict[str, Any]] = {}
        for key, value in tokens_source.items():
            entry = _sanitize_portfolio_token(key, value)
            if entry:
                sanitized[entry['symbol']] = entry
        return sanitized

    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(tokens_source, list):
        for raw in tokens_source:
            entry = _sanitize_portfolio_token(None, raw)
            if entry:
                result[entry['symbol']] = entry
    return result


def _sanitize_portfolio_token(
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
        balance_decimal = _to_decimal(candidate_value)
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
