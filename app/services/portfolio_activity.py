"""Utilities for retrieving normalized wallet activity across chains."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Optional

import httpx

from ..config import settings
from .address import is_evm_chain, normalize_chain

logger = logging.getLogger(__name__)

ActivityDirection = Literal["inflow", "outflow"]


@dataclass(slots=True)
class ActivityEvent:
    tx_hash: str
    timestamp: datetime
    direction: ActivityDirection
    native_amount: Decimal
    symbol: str
    token_address: Optional[str]
    counterparty: Optional[str]
    protocol: Optional[str]
    fee_native: Optional[Decimal]
    chain: str
    raw: dict


class ActivityProviderError(RuntimeError):
    """Raised when a provider fails to return activity."""


async def fetch_activity(
    address: str,
    chain: str,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: Optional[int] = 2500,
) -> list[ActivityEvent]:
    """Fetch normalized wallet activity for the requested chain."""

    effective_limit = limit if limit is not None else 10

    normalized_chain = normalize_chain(chain)

    if normalized_chain == "solana":
        return await _fetch_solana_activity(address, start, end, effective_limit)
    if is_evm_chain(normalized_chain):
        return await _fetch_evm_activity(address, normalized_chain, start, end, effective_limit)

    logger.warning("Unsupported chain for history summary: %s", normalized_chain)
    return []


async def _fetch_evm_activity(
    address: str,
    chain: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
) -> list[ActivityEvent]:
    if not settings.alchemy_api_key:
        logger.warning("Alchemy key not configured; returning empty EVM history")
        return []

    base_url = _resolve_alchemy_url(chain)
    if not base_url:
        logger.warning("Alchemy endpoint not known for chain %s", chain)
        return []

    # Fetch both outgoing (fromAddress) and incoming (toAddress) transfers
    base_params = {
        "category": ["external", "erc20", "erc721", "erc1155"],
        "withMetadata": True,
        "order": "desc",
        "maxCount": hex(max(1, min(limit, 0x3e8))),
    }

    outgoing_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{**base_params, "fromAddress": address}],
    }

    incoming_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "alchemy_getAssetTransfers",
        "params": [{**base_params, "toAddress": address}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        outgoing_resp, incoming_resp = await asyncio.gather(
            client.post(base_url, json=outgoing_payload),
            client.post(base_url, json=incoming_payload),
        )
        outgoing_resp.raise_for_status()
        incoming_resp.raise_for_status()
        outgoing_data = outgoing_resp.json()
        incoming_data = incoming_resp.json()

    # Merge transfers and deduplicate by tx hash
    outgoing_transfers = outgoing_data.get("result", {}).get("transfers", [])
    incoming_transfers = incoming_data.get("result", {}).get("transfers", [])

    seen_hashes: set[str] = set()
    transfers: list[dict] = []
    for transfer in outgoing_transfers + incoming_transfers:
        tx_hash = transfer.get("hash") or transfer.get("uniqueId") or ""
        if tx_hash and tx_hash in seen_hashes:
            continue
        seen_hashes.add(tx_hash)
        transfers.append(transfer)

    results: list[ActivityEvent] = []
    lower_address = address.lower()
    for transfer in transfers:
        ts_str = transfer.get("metadata", {}).get("blockTimestamp")
        if not ts_str:
            continue
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if start and ts < start:
            continue
        if end and ts > end:
            continue

        direction: ActivityDirection = "inflow"
        sender = (transfer.get("from") or "").lower()
        recipient = (transfer.get("to") or "").lower()
        if sender == lower_address and recipient != lower_address:
            direction = "outflow"
        elif recipient == lower_address and sender != lower_address:
            direction = "inflow"
        elif sender == lower_address and recipient == lower_address:
            direction = "inflow"

        value = transfer.get("value")
        if value in (None, ""):
            raw_val = transfer.get("rawContract", {}).get("value")
            decimals = transfer.get("rawContract", {}).get("decimals") or 18
            native_amount = _decode_raw_value(raw_val, decimals)
        else:
            native_amount = _safe_decimal(value)

        fee_native = None
        metadata = transfer.get("metadata", {})
        gas_price = metadata.get("gasPrice")
        gas_used = metadata.get("gasUsed")
        if gas_price and gas_used:
            try:
                fee_native = Decimal(int(gas_price, 16) * int(gas_used, 16)) / Decimal(10**18)
            except Exception:  # noqa: BLE001
                fee_native = None

        event = ActivityEvent(
            tx_hash=transfer.get("hash") or transfer.get("uniqueId") or "",
            timestamp=ts,
            direction=direction,
            native_amount=native_amount,
            symbol=transfer.get("asset") or "ETH",
            token_address=transfer.get("rawContract", {}).get("address"),
            counterparty=transfer.get("from") if direction == "inflow" else transfer.get("to"),
            protocol=transfer.get("category"),
            fee_native=fee_native,
            chain=chain,
            raw=transfer,
        )
        results.append(event)

    return results


def _resolve_alchemy_url(chain: str) -> Optional[str]:
    mapping = {
        "ethereum": f"https://eth-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        "mainnet": f"https://eth-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        "polygon": f"https://polygon-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        "base": f"https://base-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
    }
    return mapping.get(chain)


async def _fetch_solana_activity(
    address: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
) -> list[ActivityEvent]:
    api_key = settings.solana_helius_api_key
    base_url = settings.solana_balances_base_url.rstrip("/")
    if not api_key:
        logger.warning("Helius API key not configured; returning empty Solana history")
        return []

    url = f"{base_url}/v0/addresses/{address}/transactions"
    params = {
        "api-key": api_key,
        "limit": min(limit, 500),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    events: list[ActivityEvent] = []
    for item in payload:
        ts_raw = item.get("timestamp") or item.get("blockTime")
        ts = _coerce_timestamp(ts_raw)
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        account_data = item.get("nativeTransfers") or []
        direction: ActivityDirection = "outflow"
        amount = Decimal("0")
        counterparty = None
        for transfer in account_data:
            if transfer.get("fromUserAccount") == address:
                direction = "outflow"
                amount += _safe_decimal(transfer.get("amount", 0))
                counterparty = transfer.get("toUserAccount")
            elif transfer.get("toUserAccount") == address:
                direction = "inflow"
                amount += _safe_decimal(transfer.get("amount", 0))
                counterparty = transfer.get("fromUserAccount")
        event = ActivityEvent(
            tx_hash=item.get("signature", ""),
            timestamp=ts,
            direction=direction,
            native_amount=amount,
            symbol="SOL",
            token_address=None,
            counterparty=counterparty,
            protocol=item.get("type"),
            fee_native=_safe_decimal(item.get("fee", 0)),
            chain="solana",
            raw=item,
        )
        events.append(event)

    return events


def _decode_raw_value(raw_value: Optional[str], decimals: int) -> Decimal:
    if not raw_value:
        return Decimal("0")
    try:
        as_int = int(raw_value, 16) if isinstance(raw_value, str) else int(raw_value)
        return Decimal(as_int) / Decimal(10**decimals)
    except Exception:  # noqa: BLE001
        return Decimal("0")


def _safe_decimal(value: Optional[object]) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal("0")


def _coerce_timestamp(value: Optional[object]) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


async def fetch_activity_with_fallbacks(
    address: str,
    chain: str,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int = 2500,
) -> list[ActivityEvent]:
    """Wrapper that shields callers from provider exceptions."""

    try:
        return await fetch_activity(address, chain, start=start, end=end, limit=limit)
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network
        logger.warning("History provider returned HTTP error", exc_info=exc)
    except Exception as exc:  # noqa: BLE001 - want to log unexpected errors
        logger.exception("Failed to fetch wallet history", exc_info=exc)
    return []


__all__ = ["ActivityEvent", "fetch_activity", "fetch_activity_with_fallbacks"]
