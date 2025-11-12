"""Generate comparative wallet history reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

from .activity_summary import get_history_snapshot
from ..telemetry.history import HistoryMetrics, record_history_summary

DEFAULT_METRICS = [
    "inflowUsd",
    "outflowUsd",
    "feeUsd",
    "protocolCount",
    "counterpartyCount",
]
THRESHOLD_PCT = 0.15


def _aggregate_counts(events: Iterable[dict], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for event in events:
        value = (event.get(key) or "unknown").lower()
        counter[value] += float(event.get("usd_value") or 0)
    return counter


def _metric_value(metric: str, snapshot: dict, events: list[dict]) -> float:
    totals = snapshot.get("totals", {})
    if metric == "protocolCount":
        return float(len({(ev.get("protocol") or "unknown").lower() for ev in events}))
    if metric == "counterpartyCount":
        return float(len({(ev.get("counterparty") or "unknown").lower() for ev in events}))
    return float(totals.get(metric, 0))


def _format_window(window: dict) -> dict:
    return {
        "start": window.get("start"),
        "end": window.get("end"),
    }


def _delta(baseline: float, comparison: float) -> tuple[float | None, str]:
    diff = comparison - baseline
    if baseline == 0:
        return (None, "up" if diff > 0 else "down" if diff < 0 else "flat")
    pct = diff / baseline
    direction = "up" if pct > 0.01 else "down" if pct < -0.01 else "flat"
    return (pct, direction)


def _supporting_table(name: str, baseline_counts: Counter[str], comparison_counts: Counter[str], limit: int = 5) -> dict:
    def top_items(counter: Counter[str]) -> List[dict]:
        return [
            {"name": entry[0], "usd": entry[1]}
            for entry in counter.most_common(limit)
        ]

    return {
        "name": name,
        "baseline": top_items(baseline_counts),
        "comparison": top_items(comparison_counts),
    }


def _time_window_dict(start: datetime, end: datetime) -> dict:
    return {"start": start.isoformat(), "end": end.isoformat()}


async def generate_comparison_report(
    *,
    address: str,
    chain: str,
    baseline_start: datetime,
    baseline_end: datetime,
    comparison_start: datetime,
    comparison_end: datetime,
    metrics: Sequence[str] | None = None,
) -> Dict[str, Any]:
    metrics = list(metrics or DEFAULT_METRICS)
    baseline_snapshot, baseline_events = await get_history_snapshot(
        address=address,
        chain=chain,
        start=baseline_start,
        end=baseline_end,
    )
    comparison_snapshot, comparison_events = await get_history_snapshot(
        address=address,
        chain=chain,
        start=comparison_start,
        end=comparison_end,
    )

    metric_deltas: List[dict] = []
    threshold_flags: List[dict] = []

    for metric in metrics:
        baseline_value = _metric_value(metric, baseline_snapshot, baseline_events)
        comparison_value = _metric_value(metric, comparison_snapshot, comparison_events)
        delta_pct, direction = _delta(baseline_value, comparison_value)
        entry = {
            "metric": metric,
            "baselineValueUsd": baseline_value,
            "comparisonValueUsd": comparison_value,
            "deltaPct": delta_pct,
            "direction": direction,
        }
        metric_deltas.append(entry)
        if delta_pct is not None and abs(delta_pct) >= THRESHOLD_PCT:
            threshold_flags.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "magnitudePct": delta_pct,
                }
            )

    protocol_table = _supporting_table(
        "protocols",
        _aggregate_counts(baseline_events, "protocol"),
        _aggregate_counts(comparison_events, "protocol"),
    )
    counterparty_table = _supporting_table(
        "counterparties",
        _aggregate_counts(baseline_events, "counterparty"),
        _aggregate_counts(comparison_events, "counterparty"),
    )

    report = {
        "baselineWindow": _time_window_dict(baseline_start, baseline_end),
        "comparisonWindow": _time_window_dict(comparison_start, comparison_end),
        "metricDeltas": metric_deltas,
        "thresholdFlags": threshold_flags,
        "supportingTables": [protocol_table, counterparty_table],
    }
    record_history_summary(
        address,
        HistoryMetrics(
            latency_ms=(comparison_end - comparison_start).total_seconds() * 1000,
            cache_hit=False,
            events_count=len(comparison_events),
            comparison=True,
        ),
    )
    return report
