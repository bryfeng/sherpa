"""Helius-backed Solana balance provider."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import IndexerProvider


class SolanaProvider(IndexerProvider):
    """Fetch Solana balances via the Helius balances API."""

    name = "helius"
    timeout_s = 20

    def __init__(self) -> None:
        self.api_key = settings.solana_helius_api_key
        base_url = settings.solana_balances_base_url or "https://api.helius.xyz"
        self.base_url = base_url.rstrip("/")
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def ready(self) -> bool:
        return bool(self.api_key)

    async def health_check(self) -> Dict[str, Any]:
        if not await self.ready():
            return {"status": "unavailable", "reason": "API key not configured"}
        # Avoid hitting the API on every health check – report configured state.
        return {"status": "configured"}

    async def _fetch_balances(self, address: str) -> Dict[str, Any]:
        cached = self._cache.get(address)
        if cached is not None:
            return cached

        if not await self.ready():
            raise RuntimeError("Solana provider not configured")

        url = f"{self.base_url}/v0/addresses/{address}/balances"
        params = {"api-key": self.api_key}
        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("Unexpected response from Helius balances API")

        self._cache[address] = payload
        return payload

    async def get_native_balance(self, address: str, chain: str = "solana") -> Dict[str, Any]:
        data = await self._fetch_balances(address)
        native = data.get("nativeBalance") or {}

        lamports_raw: Any = native.get("lamports") or native.get("balance") or 0
        try:
            lamports = int(Decimal(str(lamports_raw)))
        except (ValueError, TypeError):
            lamports = 0

        sol_amount_raw: Any = native.get("sol")
        sol_amount = Decimal(str(sol_amount_raw)) if sol_amount_raw is not None else Decimal(lamports) / Decimal(1_000_000_000)

        price = native.get("price") or (native.get("priceInfo") or {}).get("pricePerToken")
        price_decimal = None
        if price is not None:
            try:
                price_decimal = Decimal(str(price))
            except (ValueError, TypeError):
                price_decimal = None

        value_decimal = sol_amount * price_decimal if price_decimal is not None else None

        return {
            "symbol": "SOL",
            "name": "Solana",
            "address": None,
            "decimals": 9,
            "balance_wei": str(lamports),
            "balance_formatted": f"{sol_amount:.6f}",
            "price_usd": price_decimal,
            "value_usd": value_decimal,
            "_source": {"name": "helius", "url": "https://www.helius.dev"},
        }

    async def get_token_balances(self, address: str, chain: str = "solana") -> Dict[str, Any]:
        data = await self._fetch_balances(address)
        items: List[Dict[str, Any]] = data.get("tokens") or data.get("items") or []

        balances: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            mint = item.get("mint") or item.get("address")
            if not mint:
                continue

            # The Helius API sometimes returns the native SOL entry in the token list – skip duplicates.
            if mint.lower() in {
                "so11111111111111111111111111111111111111112",
                "sol",
            }:
                continue

            decimals_raw = item.get("decimals") or 0
            try:
                decimals = int(decimals_raw)
            except (ValueError, TypeError):
                decimals = 0

            amount_human_raw = item.get("amount") or item.get("uiAmount")
            amount_lamports_raw = item.get("lamports") or item.get("amountRaw") or item.get("balance")

            amount_human = None
            if amount_human_raw is not None:
                try:
                    amount_human = Decimal(str(amount_human_raw))
                except (ValueError, TypeError):
                    amount_human = None

            lamports = None
            if amount_lamports_raw is not None:
                try:
                    lamports = int(Decimal(str(amount_lamports_raw)))
                except (ValueError, TypeError):
                    lamports = None

            if lamports is None and amount_human is not None:
                scale = Decimal(10) ** Decimal(decimals)
                lamports = int((amount_human * scale).to_integral_value(rounding=ROUND_DOWN))

            if amount_human is None and lamports is not None:
                scale = Decimal(10) ** Decimal(decimals)
                amount_human = Decimal(lamports) / scale

            price_info = item.get("priceInfo") or {}
            price_per_token = price_info.get("pricePerToken") or price_info.get("price") or item.get("price")
            price_decimal = None
            if price_per_token is not None:
                try:
                    price_decimal = Decimal(str(price_per_token))
                except (ValueError, TypeError):
                    price_decimal = None

            value_decimal = None
            if price_decimal is not None and amount_human is not None:
                value_decimal = amount_human * price_decimal
            elif price_decimal is not None and lamports is not None:
                scale = Decimal(10) ** Decimal(decimals)
                value_decimal = (Decimal(lamports) / scale) * price_decimal

            balances.append(
                {
                    "address": mint,
                    "symbol": item.get("symbol") or item.get("ticker") or "UNKNOWN",
                    "name": item.get("name") or item.get("tokenName") or "Unknown Token",
                    "decimals": decimals,
                    "balance_wei": str(lamports or 0),
                    "balance_formatted": f"{(amount_human or Decimal(0)):.6f}",
                    "price_usd": price_decimal,
                    "value_usd": value_decimal,
                    "_source": {"name": "helius", "url": "https://www.helius.dev"},
                }
            )

        return {"tokens": balances}

    def clear_cache(self) -> None:
        self._cache.clear()
