import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from typing import Any, Dict, Optional

from ..config import settings
from ..db.convex_client import get_convex_client, ConvexError
from ..providers.alchemy import AlchemyProvider
from ..providers.coingecko import CoingeckoProvider

router = APIRouter()

_PROCESS_START = time.time()


@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint — providers, database, execution stats, uptime."""

    # ----- Providers (existing) -----
    alchemy = AlchemyProvider()
    coingecko = CoingeckoProvider()

    provider_status: Dict[str, Any] = {}
    provider_status["alchemy"] = await alchemy.health_check()
    provider_status["coingecko"] = await coingecko.health_check()

    try:
        from ..core.strategies.dca.executor import RHINESTONE_AVAILABLE
        provider_status["rhinestone"] = {
            "status": "healthy" if (settings.enable_rhinestone and RHINESTONE_AVAILABLE) else "unavailable",
            "enabled": settings.enable_rhinestone,
            "sdk_available": RHINESTONE_AVAILABLE,
            "has_api_key": bool(settings.rhinestone_api_key),
            "has_session_key": bool(settings.rhinestone_session_private_key),
        }
    except Exception:
        provider_status["rhinestone"] = {"status": "unavailable", "enabled": False}

    # ----- Database health -----
    db_status: Dict[str, Any] = {"status": "unknown"}
    try:
        client = get_convex_client()
        t0 = time.perf_counter()
        await client.query("admin/executionHealth:getExecutionHealth")
        db_latency = (time.perf_counter() - t0) * 1000
        db_status = {"status": "healthy", "latency_ms": round(db_latency, 1)}
    except ConvexError as e:
        db_status = {"status": "error", "error": str(e)}
    except Exception as e:
        db_status = {"status": "error", "error": str(e)}

    # ----- Execution stats -----
    executions: Dict[str, Any] = {}
    try:
        client = get_convex_client()
        health_data = await client.query("admin/executionHealth:getExecutionHealth")
        if health_data:
            last_exec_at: Optional[str] = None
            if health_data.get("lastExecutionAt"):
                last_exec_at = datetime.fromtimestamp(
                    health_data["lastExecutionAt"] / 1000, tz=timezone.utc
                ).isoformat()
            executions = {
                "active_strategies": health_data.get("activeStrategies", 0),
                "last_execution_at": last_exec_at,
                "recent_failures_24h": health_data.get("recentFailures24h", 0),
                "stuck_executions": health_data.get("stuckExecutions", 0),
            }
    except Exception:
        executions = {"error": "Failed to fetch execution stats"}

    # ----- Overall status -----
    all_healthy = all(
        status["status"] in ["healthy", "unavailable", "configured"]
        for status in provider_status.values()
    )
    available_providers = sum(
        1 for status in provider_status.values()
        if status["status"] in ("healthy", "configured")
    )
    db_ok = db_status.get("status") == "healthy"
    has_stuck = executions.get("stuck_executions", 0) > 0

    if not db_ok:
        overall = "down"
    elif not all_healthy or available_providers == 0 or has_stuck:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "uptime_seconds": round(time.time() - _PROCESS_START),
        "version": os.environ.get("GIT_SHA", "dev"),
        "providers": provider_status,
        "database": db_status,
        "executions": executions,
    }
