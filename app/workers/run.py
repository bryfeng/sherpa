"""
Worker entry-point for arq.

Start with:
    arq app.workers.run.WorkerSettings
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import RedisSettings

from app.config import settings
from app.workers.strategy_worker import execute_dca_strategy

logger = logging.getLogger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    """Initialise shared resources once when the worker boots."""
    from app.api.dca import get_dca_service

    ctx["dca_service"] = get_dca_service()
    logger.info("strategy worker started")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("strategy worker shutting down")


def _redis_settings() -> RedisSettings:
    url = settings.redis_url
    if not url:
        return RedisSettings()  # default localhost:6379
    return RedisSettings.from_dsn(url)


class WorkerSettings:
    """arq worker configuration — importable as ``app.workers.run.WorkerSettings``."""

    functions = [execute_dca_strategy]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    max_jobs = 10
    job_timeout = 300  # 5 min per strategy execution
