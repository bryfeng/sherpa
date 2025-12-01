from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..services.activity_summary import get_history_snapshot
from ..services.history_comparison import generate_comparison_report
from ..services.address import is_valid_address_for_chain, normalize_chain
from ..workers import exports as export_worker

router = APIRouter()


class TimeWindowModel(BaseModel):
    start: datetime = Field(..., description="ISO8601 start timestamp")
    end: datetime = Field(..., description="ISO8601 end timestamp")


class ExportRequest(BaseModel):
    chain: str = Field(default="ethereum")
    format: Literal["csv", "json"] = Field(default="csv")
    timeWindow: TimeWindowModel


class ComparisonRequest(BaseModel):
    chain: str = Field(default="ethereum")
    baseline: TimeWindowModel
    comparison: TimeWindowModel
    metrics: Optional[List[str]] = Field(default=None, description="Metrics to compare")


ComparisonRequest.model_rebuild()

@router.get("/wallets/{address}/history-summary")
async def get_wallet_history_summary(
    address: str,
    chain: str = Query(default="ethereum"),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    windowDays: Optional[int] = Query(default=None, ge=1, le=365),
    limit: Optional[int] = Query(default=None, ge=1, le=2500),
):
    normalized_chain = normalize_chain(chain)
    if not is_valid_address_for_chain(address, normalized_chain):
        raise HTTPException(status_code=400, detail="Invalid address for chain")

    snapshot: dict
    metadata_overrides: dict[str, Any] = {}
    requested_window: Optional[tuple[datetime, datetime]] = None
    effective_limit = limit
    if effective_limit is None and start is None and end is None and windowDays is None:
        effective_limit = settings.history_summary_default_limit

    if effective_limit is not None:
        snapshot, _ = await get_history_snapshot(address=address, chain=normalized_chain, limit=effective_limit)
    else:
        if not start or not end:
            end = datetime.utcnow()
            window_span = windowDays or 30
            start = end - timedelta(days=window_span)
        requested_window = (start, end)

        max_window = timedelta(days=90)
        applied_start = start
        applied_end = end
        requested_span = _span_days(start, end)
        metadata_overrides["requestedWindowDays"] = requested_span
        if applied_end - applied_start > max_window:
            metadata_overrides["windowClamped"] = True
            metadata_overrides["clampedWindowDays"] = max_window.days
            applied_start = applied_end - max_window
        metadata_overrides["syncWindowDays"] = _span_days(applied_start, applied_end)

        snapshot, _ = await get_history_snapshot(address=address, chain=normalized_chain, start=applied_start, end=applied_end)
    if requested_window:
        metadata_overrides.setdefault(
            "requestedWindow",
            {
                "start": _isoformat_utc(requested_window[0]),
                "end": _isoformat_utc(requested_window[1]),
            },
        )
    if metadata_overrides:
        snapshot.setdefault("metadata", {}).update(metadata_overrides)
    exports = await export_worker.list_exports_for_address(address)
    snapshot["exportRefs"] = [export_worker.serialize_metadata(meta) for meta in exports]
    return snapshot


@router.post("/wallets/{address}/history-summary/exports")
async def create_history_export(address: str, payload: ExportRequest, background_tasks: BackgroundTasks):
    normalized_chain = normalize_chain(payload.chain)
    if not is_valid_address_for_chain(address, normalized_chain):
        raise HTTPException(status_code=400, detail="Invalid address for chain")

    snapshot, events = await get_history_snapshot(
        address=address,
        chain=normalized_chain,
        start=payload.timeWindow.start,
        end=payload.timeWindow.end,
    )
    if not events:
        raise HTTPException(status_code=404, detail="No activity found for requested window")

    metadata = await export_worker.request_export(
        address=address,
        chain=normalized_chain,
        events=events,
        snapshot=snapshot,
        export_format=payload.format,
    )
    return export_worker.serialize_metadata(metadata)


@router.get("/wallets/{address}/history-summary/exports/{export_id}")
async def download_history_export(address: str, export_id: str, format: Optional[str] = None):
    metadata = await export_worker.get_export_metadata(export_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Export not found or expired")
    if metadata.address.lower() != address.lower():
        raise HTTPException(status_code=403, detail="Export does not belong to this address")
    fmt = format or metadata.format
    path = export_worker.get_export_path(export_id, fmt)  # type: ignore[arg-type]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing")
    media_type = "text/csv" if fmt == "csv" else "application/json"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/wallets/{address}/history-summary/comparisons")
async def compare_history_summary(address: str, payload: ComparisonRequest):
    normalized_chain = normalize_chain(payload.chain)
    if payload.baseline.start >= payload.baseline.end or payload.comparison.start >= payload.comparison.end:
        raise HTTPException(status_code=400, detail="Invalid time windows")
    if not is_valid_address_for_chain(address, normalized_chain):
        raise HTTPException(status_code=400, detail="Invalid address for chain")
    max_duration = timedelta(days=365)
    if (payload.baseline.end - payload.baseline.start) > max_duration or (payload.comparison.end - payload.comparison.start) > max_duration:
        raise HTTPException(status_code=400, detail="Comparison windows cannot exceed 365 days")

    report = await generate_comparison_report(
        address=address,
        chain=normalized_chain,
        baseline_start=payload.baseline.start,
        baseline_end=payload.baseline.end,
        comparison_start=payload.comparison.start,
        comparison_end=payload.comparison.end,
        metrics=payload.metrics,
    )
    return report


def _isoformat_utc(value: datetime) -> str:
    target = value
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    else:
        target = target.astimezone(timezone.utc)
    return target.isoformat()


def _span_days(start: datetime, end: datetime) -> int:
    span = end - start
    days = span.days
    if span.seconds or span.microseconds:
        days += 1
    return max(1, days)
