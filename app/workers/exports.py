"""Async helpers for generating wallet history exports."""

from __future__ import annotations

import asyncio
import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

from ..config import BASE_DIR

ExportFormat = Literal["csv", "json"]
_EXPORT_ROOT = BASE_DIR / "history_exports"
_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ExportMetadata:
    export_id: str
    address: str
    chain: str
    format: ExportFormat
    created_at: datetime
    expires_at: datetime
    status: Literal["pending", "ready", "failed"]
    download_url: Optional[str] = None
    error: Optional[str] = None


_EXPORTS: dict[str, ExportMetadata] = {}
_LOCK = asyncio.Lock()
_TTL = timedelta(hours=24)


async def request_export(
    *,
    address: str,
    chain: str,
    events: Iterable[dict] | None = None,
    snapshot: dict | None = None,
    comparison: dict | None = None,
    export_format: ExportFormat = "csv",
) -> ExportMetadata:
    export_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    metadata = ExportMetadata(
        export_id=export_id,
        address=address,
        chain=chain,
        format=export_format,
        created_at=created,
        expires_at=created + _TTL,
        status="pending",
    )
    async with _LOCK:
        _EXPORTS[export_id] = metadata

    asyncio.create_task(_generate_export_file(metadata, list(events or []), snapshot, comparison))
    return metadata


async def _generate_export_file(metadata: ExportMetadata, events: list[dict], snapshot: dict | None, comparison: dict | None) -> None:
    try:
        if metadata.format == "json":
            await _write_json(metadata.export_id, events, snapshot, comparison)
        else:
            if comparison:
                await _write_comparison_csv(metadata.export_id, comparison)
            else:
                await _write_csv(metadata.export_id, events)
        download_url = f"/wallets/{metadata.address}/history-summary/exports/{metadata.export_id}?format={metadata.format}"
        await _update_metadata(metadata.export_id, status="ready", download_url=download_url)
    except Exception as exc:  # noqa: BLE001
        await _update_metadata(metadata.export_id, status="failed", error=str(exc))


async def _write_json(export_id: str, events: list[dict], snapshot: dict | None, comparison: dict | None) -> None:
    payload = {}
    if snapshot is not None:
        payload["snapshot"] = snapshot
        payload["events"] = events
    if comparison is not None:
        payload["comparison"] = comparison
    path = _EXPORT_ROOT / f"{export_id}.json"
    path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


async def _write_csv(export_id: str, events: list[dict]) -> None:
    path = _EXPORT_ROOT / f"{export_id}.csv"
    fieldnames = [
        "timestamp",
        "direction",
        "symbol",
        "native_amount",
        "usd_value",
        "counterparty",
        "protocol",
        "tx_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "timestamp": event.get("timestamp"),
                    "direction": event.get("direction"),
                    "symbol": event.get("symbol"),
                    "native_amount": event.get("native_amount"),
                    "usd_value": event.get("usd_value"),
                    "counterparty": event.get("counterparty"),
                    "protocol": event.get("protocol"),
                    "tx_hash": event.get("tx_hash"),
                }
            )


async def _write_comparison_csv(export_id: str, comparison: dict) -> None:
    path = _EXPORT_ROOT / f"{export_id}.csv"
    fieldnames = ["metric", "baseline", "comparison", "deltaPct", "direction"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for entry in comparison.get("metricDeltas", []):
            writer.writerow(
                {
                    "metric": entry.get("metric"),
                    "baseline": entry.get("baselineValueUsd"),
                    "comparison": entry.get("comparisonValueUsd"),
                    "deltaPct": entry.get("deltaPct"),
                    "direction": entry.get("direction"),
                }
            )


def get_export_path(export_id: str, fmt: ExportFormat) -> Path:
    return _EXPORT_ROOT / f"{export_id}.{fmt}"


async def get_export_metadata(export_id: str) -> Optional[ExportMetadata]:
    async with _LOCK:
        metadata = _EXPORTS.get(export_id)
        if metadata and metadata.expires_at < datetime.now(timezone.utc):
            _EXPORTS.pop(export_id, None)
            path = get_export_path(export_id, metadata.format)
            if path.exists():
                path.unlink()
            return None
        return metadata


async def _update_metadata(
    export_id: str,
    *,
    status: Literal["pending", "ready", "failed"],
    download_url: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    async with _LOCK:
        metadata = _EXPORTS.get(export_id)
        if not metadata:
            return
        metadata.status = status
        metadata.download_url = download_url
        metadata.error = error


async def list_exports_for_address(address: str) -> list[ExportMetadata]:
    async with _LOCK:
        now = datetime.now(timezone.utc)
        items = []
        for export in list(_EXPORTS.values()):
            if export.expires_at < now:
                _EXPORTS.pop(export.export_id, None)
                continue
            if export.address.lower() == address.lower():
                items.append(export)
        return items


def serialize_metadata(metadata: ExportMetadata) -> dict:
    return {
        "exportId": metadata.export_id,
        "format": metadata.format,
        "status": metadata.status,
        "downloadUrl": metadata.download_url,
        "createdAt": metadata.created_at.isoformat(),
        "expiresAt": metadata.expires_at.isoformat(),
        "error": metadata.error,
    }


__all__ = [
    "request_export",
    "get_export_metadata",
    "get_export_path",
    "ExportMetadata",
    "list_exports_for_address",
    "serialize_metadata",
]
