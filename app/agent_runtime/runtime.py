from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .strategy import ExecutionContext, RuntimeEvent, Strategy
from ..config import settings


@dataclass(slots=True)
class StrategyState:
    status: str = "idle"
    run_count: int = 0
    consecutive_errors: int = 0
    last_started: Optional[datetime] = None
    last_completed: Optional[datetime] = None
    last_error: Optional[str] = None
    next_run: Optional[datetime] = None
    paused: bool = False


class AgentRuntime:
    """Lightweight always-on runtime for scheduling strategies."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        max_concurrency: Optional[int] = None,
        tick_timeout_seconds: Optional[int] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("agent_runtime")
        self._strategies: Dict[str, Strategy] = {}
        self._state: Dict[str, StrategyState] = {}
        self._event_queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._started_at: Optional[datetime] = None
        self._max_concurrency = max_concurrency or settings.agent_runtime_max_concurrency
        self._tick_timeout = tick_timeout_seconds or settings.agent_runtime_tick_timeout_seconds

    # ---------------------------
    # Registration
    # ---------------------------
    def register_strategy(self, strategy: Strategy) -> None:
        strategy_id = strategy.id
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy '{strategy_id}' already registered")
        self._strategies[strategy_id] = strategy
        self._state[strategy_id] = StrategyState(
            next_run=datetime.now(timezone.utc),
        )
        self.logger.info("Registered strategy %s", strategy_id)

    def unregister_strategy(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)
        self._state.pop(strategy_id, None)

    # ---------------------------
    # Lifecycle
    # ---------------------------
    async def ensure_started(self) -> None:
        async with self._lock:
            if self._running:
                return
            await self._start_locked()

    async def start(self) -> None:
        async with self._lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        self.logger.info("Agent runtime starting with %d strategies", len(self._strategies))
        for strategy_id, strategy in self._strategies.items():
            ctx = self._make_context(strategy_id)
            try:
                await strategy.on_start(ctx)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Strategy %s on_start failed: %s", strategy_id, exc, exc_info=True)
            state = self._state.get(strategy_id)
            if state:
                state.next_run = datetime.now(timezone.utc)
        self._loop_task = asyncio.create_task(self._run_loop(), name="agent-runtime-loop")

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return
            self._running = False
            self.logger.info("Agent runtime stopping")

            if self._loop_task:
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
                self._loop_task = None

            for task in list(self._inflight):
                task.cancel()
            if self._inflight:
                await asyncio.gather(*self._inflight, return_exceptions=True)
            self._inflight.clear()

            for strategy_id, strategy in self._strategies.items():
                ctx = self._make_context(strategy_id)
                try:
                    await strategy.on_stop(ctx)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Strategy %s on_stop failed: %s", strategy_id, exc, exc_info=True)

    @property
    def is_running(self) -> bool:
        return self._running

    # ---------------------------
    # Scheduling and execution
    # ---------------------------
    async def _run_loop(self) -> None:
        try:
            while self._running:
                await self._dispatch_events()
                await self._schedule_due_ticks()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Runtime loop crashed: %s", exc, exc_info=True)
            self._running = False

    async def _dispatch_events(self) -> None:
        while not self._event_queue.empty():
            event = await self._event_queue.get()
            for strategy_id, strategy in self._strategies.items():
                state = self._state.get(strategy_id)
                if not state or state.paused:
                    continue
                ctx = self._make_context(strategy_id)
                try:
                    await strategy.on_event(event, ctx)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Strategy %s on_event error: %s", strategy_id, exc, exc_info=True)

    async def _schedule_due_ticks(self) -> None:
        now = datetime.now(timezone.utc)
        for strategy_id, strategy in self._strategies.items():
            state = self._state.get(strategy_id)
            if not state or state.paused:
                continue
            if state.status == "running":
                continue
            if state.next_run and state.next_run > now:
                continue
            if len(self._inflight) >= self._max_concurrency:
                break
            task = asyncio.create_task(self._run_strategy_tick(strategy_id, strategy))
            self._inflight.add(task)
            task.add_done_callback(lambda t, strategy_id=strategy_id: self._inflight.discard(t))

    async def _run_strategy_tick(self, strategy_id: str, strategy: Strategy) -> None:
        state = self._state.get(strategy_id)
        if not state:
            return
        state.status = "running"
        state.last_started = datetime.now(timezone.utc)
        ctx = self._make_context(strategy_id)
        try:
            await asyncio.wait_for(strategy.on_tick(ctx), timeout=self._tick_timeout)
            state.last_error = None
            state.consecutive_errors = 0
        except asyncio.TimeoutError:
            state.last_error = f"tick timed out after {self._tick_timeout}s"
            state.consecutive_errors += 1
            self.logger.warning("Strategy %s timed out", strategy_id)
        except Exception as exc:  # noqa: BLE001
            state.last_error = str(exc)
            state.consecutive_errors += 1
            self.logger.warning("Strategy %s tick failed: %s", strategy_id, exc, exc_info=True)
        finally:
            state.run_count += 1
            state.last_completed = datetime.now(timezone.utc)
            interval = strategy.interval_seconds
            backoff_multiplier = min(max(1, state.consecutive_errors), 5)
            delay = interval * backoff_multiplier
            state.next_run = datetime.now(timezone.utc) + timedelta(seconds=delay)
            state.status = "idle"

    async def emit_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        await self._event_queue.put(RuntimeEvent(type=event_type, payload=payload))

    async def run_strategy_now(self, strategy_id: str) -> bool:
        strategy = self._strategies.get(strategy_id)
        state = self._state.get(strategy_id)
        if not strategy or not state:
            return False
        if state.status == "running":
            return False
        task = asyncio.create_task(self._run_strategy_tick(strategy_id, strategy))
        self._inflight.add(task)
        task.add_done_callback(lambda t, strategy_id=strategy_id: self._inflight.discard(t))
        return True

    def pause_strategy(self, strategy_id: str) -> bool:
        state = self._state.get(strategy_id)
        if not state:
            return False
        state.paused = True
        return True

    def resume_strategy(self, strategy_id: str) -> bool:
        state = self._state.get(strategy_id)
        if not state:
            return False
        state.paused = False
        state.next_run = datetime.now(timezone.utc)
        return True

    # ---------------------------
    # Introspection
    # ---------------------------
    def list_strategies(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for strategy_id, strategy in self._strategies.items():
            state = self._state.get(strategy_id) or StrategyState()
            items.append(
                {
                    "id": strategy_id,
                    "description": strategy.description,
                    "interval_seconds": strategy.interval_seconds,
                    "status": state.status,
                    "paused": state.paused,
                    "last_started": _iso(state.last_started),
                    "last_completed": _iso(state.last_completed),
                    "last_error": state.last_error,
                    "run_count": state.run_count,
                    "next_run": _iso(state.next_run),
                    "consecutive_errors": state.consecutive_errors,
                }
            )
        return items

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "started_at": _iso(self._started_at),
            "strategy_count": len(self._strategies),
            "inflight": len(self._inflight),
            "strategies": self.list_strategies(),
        }

    # ---------------------------
    # Helpers
    # ---------------------------
    def _make_context(self, strategy_id: str) -> ExecutionContext:
        child_logger = self.logger.getChild(strategy_id)
        return ExecutionContext(logger=child_logger, runtime=self)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value else None

