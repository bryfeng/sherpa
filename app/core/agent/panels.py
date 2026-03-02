"""Panel builder functions — one per tool that produces a visible panel.

Each builder receives the raw tool-result dict and returns a PanelResult
(panels dict, sources list, optional reply override) or None on error.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PanelResult:
    panels: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reply_text: Optional[str] = None


# ---------- helpers (moved from Agent) ----------


def collect_trending_sources(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate unique data sources from trending token entries."""
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for token in tokens:
        source = token.get('_source')
        if not isinstance(source, dict):
            continue
        name = (source.get('name') or '').strip()
        url = (source.get('url') or '').strip()
        key = (name.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {}
        if name:
            entry['name'] = name
        if url:
            entry['url'] = url
        if entry:
            sources.append(entry)
    if not sources:
        sources.append({'name': 'CoinGecko', 'url': 'https://www.coingecko.com'})
    return sources


def compose_tvl_reply(protocol: str, window: str, stats: Dict[str, Any]) -> str:
    """Build a deterministic short TVL analysis string."""
    try:
        if not stats:
            return ""
        start_v = stats.get('start_value')
        end_v = stats.get('end_value')
        abs_change = stats.get('abs_change')
        pct_change = stats.get('pct_change')
        min_v = stats.get('min_value')
        max_v = stats.get('max_value')
        min_d = stats.get('min_date')
        max_d = stats.get('max_date')
        trend = stats.get('trend', 'flat')
        if start_v is None or end_v is None:
            return ""
        arrow = '\u2191' if (pct_change or 0) > 0 else ('\u2193' if (pct_change or 0) < 0 else '\u2192')
        lines = [
            f"{protocol.title()} TVL ({window}) {arrow}",
            f"- Start \u2192 End: ${start_v:,.0f} \u2192 ${end_v:,.0f} ({abs_change:+,.0f}, {pct_change:+.2f}%)",
        ]
        if min_v is not None and max_v is not None:
            lines.append(f"- Range: min ${min_v:,.0f} on {min_d}, max ${max_v:,.0f} on {max_d}")
        lines.append(f"- Trend: {trend}")
        lines.append("- See the TVL chart panel on the right for details.")
        return "\n".join(lines)
    except Exception:
        return ""


def compose_token_chart_reply(symbol: str, window: str, stats: Optional[Dict[str, Any]]) -> str:
    """Build a deterministic short price-chart analysis string."""
    try:
        if not stats:
            return ""
        latest = stats.get('latest')
        change_abs = stats.get('change_abs')
        change_pct = stats.get('change_pct')
        high = stats.get('high')
        high_time = stats.get('high_time')
        low = stats.get('low')
        low_time = stats.get('low_time')
        if latest is None or change_pct is None:
            return ""
        arrow = '\u2191' if change_pct > 0 else ('\u2193' if change_pct < 0 else '\u2192')

        def _fmt_price(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return 'n/a'
            abs_val = abs(val)
            if abs_val >= 1000:
                return f"${val:,.0f}"
            if abs_val >= 1:
                return f"${val:,.2f}"
            if abs_val >= 0.0001:
                return f"${val:,.6f}"
            return f"${val:.10f}".rstrip('0').rstrip('.')

        def _fmt_change(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return 'n/a'
            abs_val = abs(val)
            if abs_val >= 0.01:
                return f"{val:+.2f}"
            return f"{val:+.8f}".rstrip('0').rstrip('.')

        def _fmt_time(value: Any) -> str:
            try:
                return datetime.fromtimestamp(int(value) / 1000).strftime('%Y-%m-%d %H:%M')
            except Exception:
                return 'n/a'

        lines = [
            f"{symbol} price ({window.upper()}) {arrow}",
            f"- Latest: {_fmt_price(latest)} ({_fmt_change(change_abs)}, {change_pct:+.2f}%)",
        ]
        if high is not None and low is not None:
            lines.append(
                f"- Range: high {_fmt_price(high)} on {_fmt_time(high_time)}, low {_fmt_price(low)} on {_fmt_time(low_time)}"
            )
        lines.append("- See the price chart panel on the right for candles and volume.")
        return "\n".join(lines)
    except Exception:
        return ""


# ---------- builders ----------


def build_portfolio_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build portfolio overview panel from get_portfolio result."""
    if not result_data.get('success'):
        return None
    data = result_data.get('data', {})
    sources = result_data.get('sources', [])
    return PanelResult(
        panels={
            'portfolio_overview': {
                'id': 'portfolio_overview',
                'kind': 'portfolio',
                'title': 'Your Portfolio Snapshot',
                'payload': data,
                'sources': sources,
                'metadata': {
                    'warnings': result_data.get('warnings', []),
                },
            },
        },
        sources=sources,
    )


def build_token_chart_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build price chart panel from get_token_chart result."""
    if not result_data.get('success'):
        return None
    # chart_info is result_data without 'success'
    chart_info = {k: v for k, v in result_data.items() if k != 'success'}
    metadata = chart_info.get('metadata') or {}
    symbol = metadata.get('symbol') or metadata.get('name') or chart_info.get('coin_id') or 'Token'
    slug = ''.join(ch for ch in str(symbol).lower() if ch.isalnum()) or 'token'
    panel_id = f'{slug}_price_chart'
    panel_title = f"{symbol} price chart ({str(chart_info.get('range', '7d')).upper()})"
    panel_sources = chart_info.get('sources') or [
        {'name': 'CoinGecko', 'url': 'https://www.coingecko.com'},
    ]
    summary = compose_token_chart_reply(symbol, chart_info.get('range', '7d'), chart_info.get('stats'))
    return PanelResult(
        panels={
            panel_id: {
                'id': panel_id,
                'kind': 'chart',
                'title': panel_title,
                'payload': chart_info,
                'sources': panel_sources,
                'metadata': {
                    'symbol': metadata.get('symbol'),
                    'range': chart_info.get('range'),
                },
            },
        },
        sources=panel_sources,
        reply_text=summary or None,
    )


def build_trending_tokens_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build trending tokens panel from get_trending_tokens result."""
    if not result_data.get('success'):
        return None
    tokens = result_data.get('tokens') or []
    if not tokens:
        return None

    panel_sources = [{'name': 'CoinGecko', 'url': 'https://www.coingecko.com'}]
    # If result_data carries its own sources, prefer those; otherwise aggregate
    # from token entries.
    raw_sources = result_data.get('sources')
    if raw_sources:
        panel_sources = raw_sources
    else:
        panel_sources = collect_trending_sources(tokens)

    metadata: Dict[str, Any] = {
        'layout': 'banner',
        'totalAvailable': result_data.get('total_available'),
        'query': None,
        'focusSymbol': (result_data.get('focus') or {}).get('symbol'),
    }
    extra_metadata = result_data.get('metadata') or {}
    for key, value in extra_metadata.items():
        if value is not None:
            metadata[key] = value

    return PanelResult(
        panels={
            'trending_tokens': {
                'id': 'trending_tokens',
                'kind': 'trending',
                'title': result_data.get('title') or 'Trending Tokens',
                'payload': {
                    'tokens': tokens,
                    'fetchedAt': result_data.get('fetched_at'),
                    'focusToken': result_data.get('focus'),
                    'totalAvailable': result_data.get('total_available'),
                },
                'sources': panel_sources,
                'metadata': metadata,
            },
        },
        sources=panel_sources,
    )


def build_wallet_history_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build wallet activity summary panel from get_wallet_history result."""
    if not result_data.get('success'):
        return None
    summary_payload = result_data.get('snapshot', {})
    limit = result_data.get('limit')

    metadata: Dict[str, Any] = {}
    existing_metadata = result_data.get('metadata')
    if isinstance(existing_metadata, dict):
        metadata.update(existing_metadata)
    metadata.setdefault('density', 'full')

    if limit:
        metadata['sampleLimit'] = limit

    panel: Dict[str, Any] = {
        'id': 'history-summary',
        'kind': 'history-summary',
        'title': 'Wallet Activity Summary',
        'payload': summary_payload,
        'sources': [{'name': summary_payload.get('chain', 'multichain')}],
    }
    if metadata:
        panel['metadata'] = metadata

    return PanelResult(
        panels={'history_summary': panel},
        sources=[],
    )


def build_tvl_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build DefiLlama TVL chart panel from get_tvl_data result."""
    if not result_data.get('success'):
        return None
    ts = result_data.get('timestamps', [])
    tvl = result_data.get('tvl', [])
    protocol = result_data.get('protocol', 'uniswap')
    window = result_data.get('window', '7d')
    stats = result_data.get('stats') or {}

    tvl_sources = result_data.get('sources') or [
        {
            'name': 'DefiLlama',
            'url': f'https://defillama.com/protocol/{protocol}',
        },
    ]

    summary = compose_tvl_reply(protocol, window, stats)

    return PanelResult(
        panels={
            'uniswap_tvl_chart': {
                'kind': 'chart',
                'title': f'{protocol.title()} TVL ({window})',
                'payload': {
                    'timestamps': ts,
                    'tvl': tvl,
                    'unit': 'USD',
                    'protocol': protocol,
                    'window': window,
                    'stats': stats,
                },
                'sources': tvl_sources,
                'metadata': {
                    'stats': stats,
                },
            },
        },
        sources=tvl_sources,
        reply_text=summary or None,
    )


def build_strategies_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build strategies list panel from list_strategies result."""
    if not result_data.get('success'):
        return None
    strategies_list = result_data.get('strategies', [])
    return PanelResult(
        panels={
            'my_strategies': {
                'id': 'my-strategies',
                'kind': 'my-strategies',
                'title': 'My Strategies',
                'payload': {
                    'strategies': strategies_list,
                    'count': len(strategies_list),
                    'walletAddress': result_data.get('wallet_address'),
                },
                'sources': [],
                'metadata': {
                    'density': 'full',
                },
            },
        },
        sources=[],
    )


def build_strategy_detail_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build single strategy detail panel from get_strategy result."""
    if not result_data.get('success'):
        return None
    strategy = result_data.get('strategy', {})
    return PanelResult(
        panels={
            'strategy_detail': {
                'id': 'strategy-detail',
                'kind': 'strategy-detail',
                'title': strategy.get('name', 'Strategy Details'),
                'payload': strategy,
                'sources': [],
                'metadata': {
                    'density': 'full',
                },
            },
        },
        sources=[],
    )


def build_bridge_quote_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build bridge quote card panel from get_bridge_quote result."""
    if not result_data.get('success'):
        return None

    from_chain = result_data.get('from_chain', {})
    to_chain = result_data.get('to_chain', {})
    token_info = result_data.get('token', {})
    output_info = result_data.get('output', {})
    fees_info = result_data.get('fees', {})

    # Build panel payload with all data frontend needs for execution
    panel_payload: Dict[str, Any] = {
        'quote_type': 'bridge',
        'provider': 'relay',
        'status': 'ok' if result_data.get('tx_ready') else 'quote_only',
        'tx_ready': result_data.get('tx_ready', False),
        'from_chain_id': from_chain.get('chain_id'),
        'from_chain': from_chain.get('name'),
        'to_chain_id': to_chain.get('chain_id'),
        'to_chain': to_chain.get('name'),
        'wallet': result_data.get('wallet', {}),
        'amounts': result_data.get('amounts', {}),
        'breakdown': result_data.get('breakdown', {}),
        'fees': fees_info,
        'time_estimate_seconds': result_data.get('time_estimate_seconds'),
        'instructions': result_data.get('instructions', []),
    }

    # Add transaction data for execution
    if result_data.get('tx'):
        panel_payload['tx'] = result_data['tx']
    if result_data.get('approval_data'):
        panel_payload['approval_data'] = result_data['approval_data']
    if result_data.get('approvals'):
        panel_payload['approvals'] = result_data['approvals']
    if result_data.get('transactions'):
        panel_payload['transactions'] = result_data['transactions']

    # Build panel
    from_symbol = token_info.get('symbol', 'TOKEN')
    to_symbol = token_info.get('symbol', 'TOKEN')
    panel = {
        'id': 'relay_bridge_quote',
        'kind': 'card',
        'title': f"Bridge: {from_symbol} \u2192 {to_symbol}",
        'payload': panel_payload,
        'sources': [{'name': 'Relay', 'url': 'https://relay.link'}],
        'metadata': {
            'status': panel_payload['status'],
            'has_transactions': result_data.get('tx_ready', False),
        },
    }

    # Build summary reply
    amount = token_info.get('amount', '?')
    output_amount = output_info.get('amount_estimate', '?')
    fee_usd = fees_info.get('total_usd', '?')
    time_sec = result_data.get('time_estimate_seconds', 0)

    summary_lines = [
        f"Bridge {amount} {from_symbol} from {from_chain.get('name')} to {to_chain.get('name')}",
        f"Expected output: ~{output_amount} {to_symbol}",
        f"Estimated fees: ${fee_usd}",
    ]
    if time_sec:
        summary_lines.append(f"Estimated time: {time_sec // 60}m {time_sec % 60}s")
    summary_lines.append("Review and sign the transaction in your wallet to execute.")

    panel_id = panel['id']
    panel_sources = panel.get('sources', [])

    return PanelResult(
        panels={panel_id: panel},
        sources=panel_sources,
        reply_text='\n'.join(summary_lines),
    )


def build_swap_quote_panel(result_data: Dict[str, Any]) -> Optional[PanelResult]:
    """Build swap quote card panel from get_swap_quote result."""
    if not result_data.get('success'):
        return None

    input_token = result_data.get('input_token', {})
    output_token = result_data.get('output_token', {})
    fees_info = result_data.get('fees', {})
    chain_id = result_data.get('chain_id')
    chain_name = result_data.get('chain', 'Unknown')

    # Extract tx data from Relay quote steps
    tx_data = None
    quote_data = result_data.get('quote_data', {})
    steps = quote_data.get('steps', [])
    if steps:
        items = steps[0].get('items', [])
        if items:
            item = items[0]
            tx_data = item.get('data', {})

    # Extract fee/eta from Relay details
    details = quote_data.get('details', {})
    total_fee_usd = fees_info.get('total_usd')
    gas_fee_raw = details.get('feeBreakdown', [{}])
    gas_usd = None
    for fee_entry in (gas_fee_raw if isinstance(gas_fee_raw, list) else []):
        if fee_entry.get('kind') == 'gas':
            gas_usd = fee_entry.get('amountUsd')
            break
    if gas_usd is None:
        gas_usd = total_fee_usd  # fallback to total
    eta_seconds = details.get('timeEstimate')

    in_sym = input_token.get('symbol', 'TOKEN')
    out_sym = output_token.get('symbol', 'TOKEN')
    in_amount = input_token.get('amount', '0')
    out_amount = output_token.get('amount_estimate', '0')

    panel_payload: Dict[str, Any] = {
        'quote_type': 'swap',
        'provider': 'relay',
        'status': 'ok' if tx_data else 'quote_only',
        'tx_ready': bool(tx_data),
        'from_chain_id': chain_id,
        'from_chain': chain_name,
        'to_chain_id': chain_id,
        'to_chain': chain_name,
        'wallet': {'address': details.get('sender', '')},
        # tokens -- primary source for widget symbol display
        'tokens': {
            'input': {
                'symbol': in_sym,
                'address': input_token.get('address', ''),
                'amount': in_amount,
            },
            'output': {
                'symbol': out_sym,
                'address': output_token.get('address', ''),
                'amount_estimate': out_amount,
            },
        },
        'amounts': {
            'input': in_amount,
            'input_base_units': input_token.get('amount_base_units', '0'),
        },
        'breakdown': {
            'input': {
                'token_address': input_token.get('address', ''),
                'symbol': in_sym,
                'amount': in_amount,
            },
            'output': {
                'token_address': output_token.get('address', ''),
                'symbol': out_sym,
                'amount_estimate': out_amount,
            },
            'fees': {
                'total_usd': total_fee_usd,
                'gas_usd': gas_usd,
                'slippage_percent': fees_info.get('slippage_percent'),
            },
            'eta_seconds': eta_seconds,
        },
        'usd_estimates': {
            'gas': gas_usd,
        },
        'instructions': result_data.get('instructions', []),
    }

    # Add transaction data for execution
    if tx_data:
        panel_payload['tx'] = {
            'to': tx_data.get('to', ''),
            'data': tx_data.get('data', ''),
            'value': tx_data.get('value', '0'),
            'chainId': chain_id,
        }

    panel = {
        'id': 'relay_swap_quote',
        'kind': 'card',
        'title': f"Swap: {in_sym} \u2192 {out_sym}",
        'payload': panel_payload,
        'sources': [{'name': 'Relay', 'url': 'https://relay.link'}],
        'metadata': {
            'status': panel_payload['status'],
            'has_transactions': bool(tx_data),
        },
    }

    fee_usd = total_fee_usd or '0'

    summary_lines = [
        f"Swap {in_amount} {in_sym} for ~{out_amount} {out_sym} on {chain_name}",
        f"Estimated fees: ${fee_usd}",
        "Review and sign the transaction in your wallet to execute.",
    ]

    panel_id = panel['id']
    panel_sources = panel.get('sources', [])

    return PanelResult(
        panels={panel_id: panel},
        sources=panel_sources,
        reply_text='\n'.join(summary_lines),
    )


# ---------- registry ----------

PANEL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Optional[PanelResult]]] = {
    'get_portfolio': build_portfolio_panel,
    'get_token_chart': build_token_chart_panel,
    'get_trending_tokens': build_trending_tokens_panel,
    'get_wallet_history': build_wallet_history_panel,
    'get_tvl_data': build_tvl_panel,
    'list_strategies': build_strategies_panel,
    'get_strategy': build_strategy_detail_panel,
    'get_bridge_quote': build_bridge_quote_panel,
    'get_swap_quote': build_swap_quote_panel,
}
