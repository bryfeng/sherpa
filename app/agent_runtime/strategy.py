from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..config import settings
from ..tools.portfolio import get_portfolio
from ..services.trending import get_trending_tokens


class StrategyConfig(BaseModel):
    """Base configuration for strategies."""

    interval_seconds: Optional[float] = Field(
        default=None,
        description="How often to run on_tick; falls back to runtime defaults when unset.",
    )


@dataclass(slots=True)
class RuntimeEvent:
    """Event dispatched to strategies."""

    type: str
    payload: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    origin: str | None = None


class ExecutionContext:
    """Lightweight context passed into strategies."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        runtime: Any = None,
    ) -> None:
        self.logger = logger
        self.runtime = runtime
        self.settings = settings

    async def emit_event(self, event_type: str, payload: Optional[dict[str, Any]] = None) -> None:
        """Forward an event back to the runtime for fan-out delivery."""

        if self.runtime:
            await self.runtime.emit_event(event_type, payload)

    async def fetch_portfolio(self, address: str, chain: str = "ethereum") -> Any:
        """Shared helper to reuse the existing portfolio tooling."""

        return await get_portfolio(address, chain)

    async def fetch_trending_tokens(self, limit: int = 10) -> Any:
        return await get_trending_tokens(limit=limit)


class Strategy:
    """Base class for runtime strategies."""

    id: str = "strategy"
    description: str = "runtime strategy"
    default_interval_seconds: float = 60.0
    ConfigModel = StrategyConfig

    def __init__(self, config: StrategyConfig | None = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or self.ConfigModel()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def interval_seconds(self) -> float:
        if self.config.interval_seconds and self.config.interval_seconds > 0:
            return float(self.config.interval_seconds)
        return float(self.default_interval_seconds or settings.agent_runtime_default_interval_seconds)

    async def on_start(self, ctx: ExecutionContext) -> None:
        """Called once when the runtime boots."""
        return None

    async def on_tick(self, ctx: ExecutionContext) -> None:
        """Called on every scheduled interval."""
        return None

    async def on_event(self, event: RuntimeEvent, ctx: ExecutionContext) -> None:
        """Called for external or cross-strategy events."""
        return None

    async def on_stop(self, ctx: ExecutionContext) -> None:
        """Called during graceful shutdown."""
        return None

