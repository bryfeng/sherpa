"""Smoke test for Relay's public quote endpoint.

Usage:
    python -m sherpa.tests.test_relay_public

The script fetches an ETH → Base quote and prints the returned transaction
payload so we can verify the public integration works without an API key.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import httpx


# Allow running via ``python -m sherpa.tests.test_relay_public`` (repo root)
if 'sherpa' not in sys.modules:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sherpa.app.providers.relay import RelayProvider


QUOTE_PAYLOAD: Dict[str, Any] = {
    'user': '0x50ac5CFcc81BB0872e85255D7079F8a529345D16',
    'originChainId': 1,
    'destinationChainId': 8453,
    'originCurrency': '0x0000000000000000000000000000000000000000',
    'destinationCurrency': '0x0000000000000000000000000000000000000000',
    'recipient': '0x50ac5CFcc81BB0872e85255D7079F8a529345D16',
    'tradeType': 'EXACT_INPUT',
    'amount': '1000000000000000',  # 0.001 ETH
    'referrer': 'sherpa.chat',
    'useExternalLiquidity': False,
    'useDepositAddress': False,
    'topupGas': False,
}


async def main() -> None:
    provider = RelayProvider()

    print('👉 Requesting quote via /quote...')
    try:
        quote = await provider.quote(QUOTE_PAYLOAD)
    except httpx.HTTPStatusError as exc:  # type: ignore[assignment]
        print('❌ Relay quote HTTP error:', exc.response.status_code, exc.response.text)
        return

    steps = quote.get('steps') or []
    fees = quote.get('fees') or {}
    details = quote.get('details') or {}
    print(f"   steps returned={len(steps)}")
    print(f"   fee buckets={list(fees.keys())}")
    currency_out = (details.get('currencyOut') or {}).get('amountFormatted')
    print(f"   estimated arrival={currency_out}")

    if not steps:
        print('❌ No steps returned from Relay. Raw response:', quote)
        return

    deposit_step = steps[0]
    tx_item = (deposit_step.get('items') or [{}])[0]
    tx_data = tx_item.get('data') or {}
    print('   transaction to=', tx_data.get('to'))
    print('   value=', tx_data.get('value'))
    payload_preview = (tx_data.get('data') or '')
    print('   data prefix=', payload_preview[:18])

    if 'check' in tx_item:
        print('   status endpoint=', tx_item['check'])


if __name__ == '__main__':
    asyncio.run(main())
