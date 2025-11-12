
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class HistoryMetrics:
    latency_ms: float
    cache_hit: bool
    events_count: int
    comparison: bool = False

metrics_store: Dict[str, HistoryMetrics] = {}


def record_history_summary(address: str, metrics: HistoryMetrics) -> None:
    key = f"{address.lower()}:{'comparison' if metrics.comparison else 'summary'}"
    metrics_store[key] = metrics
    logger.info(
        "history-summary", extra={
            "address": address,
            "latency_ms": metrics.latency_ms,
            "cache_hit": metrics.cache_hit,
            "events": metrics.events_count,
            "comparison": metrics.comparison,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
