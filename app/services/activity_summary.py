"""Summarize normalized wallet activity into deterministic payloads."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import httpx

from ..config import settings
from ..telemetry.history import HistoryMetrics, record_history_summary
from ..providers.coingecko import CoingeckoProvider
from .portfolio_activity import ActivityEvent, fetch_activity_with_fallbacks
from .cache import history_cache, history_cache_key

BucketSize = Literal["day", "week"]


async def get_history_snapshot(
    *,
    address: str,
    chain: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[dict, list[dict]]:
    if limit is not None:
        use_cache = False
    if limit is None and (start is None or end is None):
        raise ValueError("start/end required when limit is not provided")

    cache_key = history_cache_key(
        address.lower(),
        chain,
        (start.isoformat() if start else f"limit:{limit}"),
        (end.isoformat() if end else f"limit:{limit}"),
    )
    if use_cache:
        cached = await history_cache.get(cache_key)
        if cached:
            events = cached.pop("_events", [])
            cached["cached"] = True
            record_history_summary(
                address,
                HistoryMetrics(latency_ms=0.0, cache_hit=True, events_count=len(events)),
            )
            return cached, events

    start_time = datetime.now()
    events = await fetch_activity_with_fallbacks(address, chain, start=start, end=end, limit=limit)
    events_payload = await _enrich_events(events)

    if limit is not None:
        if events_payload:
            timestamps = [datetime.fromisoformat(ev["timestamp"]) for ev in events_payload]
            effective_start = min(timestamps).replace(tzinfo=timezone.utc)
            effective_end = max(timestamps).replace(tzinfo=timezone.utc)
        else:
            effective_end = datetime.now(timezone.utc)
            effective_start = effective_end
    else:
        effective_start = start.replace(tzinfo=timezone.utc)  # type: ignore[arg-type]
        effective_end = end.replace(tzinfo=timezone.utc)  # type: ignore[arg-type]

    bucket_size = _resolve_bucket_size(effective_start, effective_end)
    buckets = _bucketize(events_payload, effective_start, effective_end, bucket_size)
    totals = _compute_totals(events_payload)
    notable = _detect_notable_events(events_payload, totals)

    snapshot = {
        "walletAddress": address,
        "chain": chain,
        "timeWindow": {
            "start": effective_start.isoformat(),
            "end": effective_end.isoformat(),
        },
        "bucketSize": bucket_size,
        "totals": totals,
        "notableEvents": notable,
        "buckets": buckets,
        "exportRefs": [],
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    if limit is not None:
        metadata = snapshot.setdefault("metadata", {})
        metadata["sampleLimit"] = limit
    snapshot["_events"] = events_payload

    await history_cache.set(cache_key, snapshot)
    snapshot_cached = dict(snapshot)
    events_for_return = snapshot_cached.pop("_events")
    snapshot_cached["cached"] = False
    record_history_summary(
        address,
        HistoryMetrics(
            latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
            cache_hit=False,
            events_count=len(events_payload),
        ),
    )
    return snapshot_cached, events_for_return


async def _enrich_events(events: Iterable[ActivityEvent]) -> list[dict]:
    events_list = list(events)
    if not events_list:
        return []

    provider = CoingeckoProvider()
    price_map: Dict[str, Decimal] = {}
    token_addresses = {
        e.token_address.lower()
        for e in events_list
        if e.token_address and e.chain in {"ethereum", "mainnet"}
    }
    eth_price = None
    sol_price = None
    if await provider.ready():  # pragma: no branch - small
        if token_addresses:
            prices = await provider.get_token_prices(list(token_addresses))
            for token_addr, payload in prices.items():
                price_val = payload.get("price_usd")
                if price_val is not None:
                    price_map[token_addr] = Decimal(str(price_val))
        eth_payload = await provider.get_eth_price()
        if eth_payload:
            eth_price = Decimal(str(eth_payload.get("price_usd", "0")))

        # Basic SOL support via direct endpoint
        sol_price = await _fetch_simple_price("solana")
    else:
        sol_price = await _fetch_simple_price("solana")

    enriched = []
    for event in events_list:
        price = None
        if event.token_address:
            price = price_map.get(event.token_address.lower())
        elif event.symbol.upper() == "ETH" and eth_price:
            price = eth_price
        elif event.symbol.upper() == "SOL" and sol_price:
            price = sol_price

        usd_value = None
        if price is not None:
            usd_value = float((event.native_amount * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        enriched.append(
            {
                "tx_hash": event.tx_hash,
                "timestamp": event.timestamp.replace(tzinfo=timezone.utc).isoformat(),
                "direction": event.direction,
                "symbol": event.symbol,
                "native_amount": float(event.native_amount),
                "usd_value": usd_value,
                "counterparty": event.counterparty,
                "protocol": event.protocol,
                "fee_native": float(event.fee_native) if event.fee_native else 0,
                "chain": event.chain,
            }
        )
    return enriched


async def _fetch_simple_price(coin_id: str) -> Decimal | None:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        async with httpx.AsyncClient(timeout=10) as client:  # type: ignore[name-defined]
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            price = data.get(coin_id, {}).get("usd")
            if price is not None:
                return Decimal(str(price))
    except Exception:  # noqa: BLE001
        return None
    return None


def _resolve_bucket_size(start: datetime, end: datetime) -> BucketSize:
    days = max((end - start).days, 1)
    return "week" if days > 45 else "day"


def _bucketize(events: list[dict], start: datetime, end: datetime, bucket_size: BucketSize) -> list[dict]:
    span = timedelta(days=7 if bucket_size == "week" else 1)
    bucket_map: Dict[str, dict] = {}
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    ordered_keys: list[str] = []
    while cursor < end:
        bucket_end = cursor + span
        start_iso = cursor.isoformat()
        ordered_keys.append(start_iso)
        bucket_map[start_iso] = {
            "start": start_iso,
            "end": bucket_end.isoformat(),
            "inflow": 0.0,
            "outflow": 0.0,
            "inflowUsd": 0.0,
            "outflowUsd": 0.0,
            "feeUsd": 0.0,
            "transactionsSample": [],
            "_protocols": {},
            "_counterparties": {},
        }
        cursor = bucket_end

    for event in events:
        ts = datetime.fromisoformat(event["timestamp"])
        bucket_start = _bucket_floor(ts, span).isoformat()
        bucket = bucket_map.get(bucket_start)
        if not bucket:
            continue
        amount = float(event.get("native_amount") or 0)
        usd_val = float(event.get("usd_value") or 0)
        if event["direction"] == "inflow":
            bucket["inflow"] += amount
            bucket["inflowUsd"] += usd_val
        else:
            bucket["outflow"] += amount
            bucket["outflowUsd"] += usd_val
        bucket["feeUsd"] += float(event.get("fee_native") or 0)

        protocol_name = (event.get("protocol") or "activity").lower()
        protocol_stats = bucket["_protocols"].setdefault(
            protocol_name,
            {"usd": 0.0, "txCount": 0},
        )
        protocol_stats["usd"] += usd_val
        protocol_stats["txCount"] += 1

        counterparty = event.get("counterparty") or "unknown"
        counterparty_stats = bucket["_counterparties"].setdefault(
            counterparty,
            {"usd": 0.0, "txCount": 0},
        )
        counterparty_stats["usd"] += usd_val
        counterparty_stats["txCount"] += 1

        if len(bucket["transactionsSample"]) < 5:
            bucket["transactionsSample"].append(
                {
                    "timestamp": event["timestamp"],
                    "symbol": event["symbol"],
                    "direction": event["direction"],
                    "usd_value": usd_val,
                    "tx_hash": event["tx_hash"],
                }
            )

    buckets: list[dict] = []
    for key in ordered_keys:
        bucket = bucket_map[key]
        protocols = bucket.pop("_protocols")
        counterparties = bucket.pop("_counterparties")
        bucket["topProtocols"] = _top_entries(protocols)
        bucket["topCounterparties"] = _top_entries(counterparties)
        buckets.append(bucket)
    return buckets


def _bucket_floor(ts: datetime, span: timedelta) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = ts - epoch
    bucket_index = int(delta.total_seconds() // span.total_seconds())
    return epoch + timedelta(seconds=bucket_index * span.total_seconds())


def _top_entries(stats: Dict[str, Dict[str, float]], limit: int = 3) -> list[dict]:
    ordered = sorted(
        stats.items(),
        key=lambda item: (item[1]["usd"], item[1]["txCount"]),
        reverse=True,
    )
    top: list[dict] = []
    for name, payload in ordered[:limit]:
        top.append(
            {
                "name": name,
                "usd": _round_two(payload["usd"]),
                "txCount": payload["txCount"],
            }
        )
    return top


def _round_two(value: float) -> float:
    return float(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _format_usd(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def _compute_totals(events: list[dict]) -> dict:
    inflow_usd = 0.0
    outflow_usd = 0.0
    fee_usd = 0.0
    inflow_native = 0.0
    outflow_native = 0.0
    for event in events:
        amount = float(event.get("native_amount") or 0)
        usd = float(event.get("usd_value") or 0)
        if event["direction"] == "inflow":
            inflow_native += amount
            inflow_usd += usd
        else:
            outflow_native += amount
            outflow_usd += usd
        fee_usd += float(event.get("fee_native") or 0)
    return {
        "inflow": inflow_native,
        "outflow": outflow_native,
        "inflowUsd": _round_two(inflow_usd),
        "outflowUsd": _round_two(outflow_usd),
        "feeUsd": _round_two(fee_usd),
    }


def _detect_notable_events(events: list[dict], totals: dict) -> list[dict]:
    if not events:
        return []

    notable: list[dict] = []
    outflow_vals = [float(ev.get("usd_value") or 0) for ev in events if ev["direction"] == "outflow"]

    if outflow_vals:
        baseline_pool = sorted(outflow_vals)[: max(1, len(outflow_vals) // 2)]
        avg_outflow = mean(baseline_pool)
        spikes = [val for val in outflow_vals if avg_outflow > 0 and val > avg_outflow * 2]
        if spikes:
            severity = "critical" if max(spikes) > avg_outflow * 4 else "warning"
            notable.append(
                {
                    "type": "large_outflow",
                    "severity": severity,
                    "summary": f"{len(spikes)} unusually large outflows detected (>{_format_usd(avg_outflow * 2)} each).",
                }
            )

    timeline = sorted(events, key=lambda ev: ev["timestamp"])
    if len(timeline) >= 2:
        gaps = []
        for idx in range(1, len(timeline)):
            current_ts = datetime.fromisoformat(timeline[idx]["timestamp"])
            prev_ts = datetime.fromisoformat(timeline[idx - 1]["timestamp"])
            gaps.append((current_ts - prev_ts, prev_ts))
        if gaps:
            max_gap, gap_start = max(gaps, key=lambda item: item[0])
            if max_gap >= timedelta(days=21):
                severity = "warning" if max_gap >= timedelta(days=45) else "info"
                notable.append(
                    {
                        "type": "dormant_period",
                        "severity": severity,
                        "summary": f"Dormant for {max_gap.days} days before activity resumed on {gap_start.date():%b %d}.",
                    }
                )

    outflow_total = sum(outflow_vals)
    if outflow_total > 0:
        counterparty_totals: Counter[str] = Counter()
        for ev in events:
            if ev["direction"] != "outflow":
                continue
            counterparty = (ev.get("counterparty") or "unknown").lower()
            counterparty_totals[counterparty] += float(ev.get("usd_value") or 0)
        if counterparty_totals:
            top_counterparty, top_value = counterparty_totals.most_common(1)[0]
            share = top_value / outflow_total if outflow_total else 0
            if share >= 0.6:
                severity = "critical" if share >= 0.8 else "warning"
                notable.append(
                    {
                        "type": "concentrated_outflow",
                        "severity": severity,
                        "summary": f"{top_counterparty} received {share:.0%} of outflows in this window.",
                    }
                )

    if totals.get("outflowUsd", 0) and totals.get("inflowUsd", 0) == 0:
        notable.append(
            {
                "type": "net_outflow_only",
                "severity": "info",
                "summary": "Only outflows recorded in the selected window; no offsetting inflows detected.",
            }
        )

    return notable


detect_notable_events = _detect_notable_events
compute_totals = _compute_totals
