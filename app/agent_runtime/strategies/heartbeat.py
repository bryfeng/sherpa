from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import Field

from ..strategy import ExecutionContext, Strategy, StrategyConfig


class HeartbeatConfig(StrategyConfig):
    wallet_address: Optional[str] = Field(
        default=None,
        description="Optional wallet to monitor; skips portfolio fetch when unset.",
    )
    chain: str = Field(default="ethereum", description="Chain used when fetching the portfolio")


class HeartbeatStrategy(Strategy):
    """Simple liveness + optional portfolio sampler."""

    id = "heartbeat"
    description = "Emits liveness breadcrumbs and optionally samples a wallet snapshot."
    default_interval_seconds = 60.0
    ConfigModel = HeartbeatConfig

    def __init__(self, config: Optional[HeartbeatConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    async def on_start(self, ctx: ExecutionContext) -> None:
        ctx.logger.info("Heartbeat strategy online; interval=%ss", self.interval_seconds)

    async def on_tick(self, ctx: ExecutionContext) -> None:
        now = datetime.now(timezone.utc).isoformat()
        ctx.logger.info("Heartbeat tick at %s", now)

        cfg: HeartbeatConfig = self.config  # type: ignore[assignment]
        if not cfg.wallet_address:
            return

        try:
            portfolio_result = await ctx.fetch_portfolio(cfg.wallet_address, cfg.chain)
            if portfolio_result and getattr(portfolio_result, "data", None):
                total = getattr(portfolio_result.data, "total_value_usd", None)
                ctx.logger.info(
                    "Wallet %s on %s total value=%s (sources=%s)",
                    cfg.wallet_address,
                    cfg.chain,
                    total,
                    [s.name for s in getattr(portfolio_result, "sources", [])],
                )
            elif portfolio_result and getattr(portfolio_result, "warnings", None):
                ctx.logger.warning("Portfolio fetch warnings: %s", "; ".join(portfolio_result.warnings))
        except Exception as exc:  # noqa: BLE001
            ctx.logger.warning("Heartbeat portfolio probe failed: %s", exc, exc_info=True)

