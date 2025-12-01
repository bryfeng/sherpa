from __future__ import annotations

import asyncio
import signal

from . import get_runtime, register_builtin_strategies
from ..config import settings


async def _serve() -> None:
    register_builtin_strategies()
    runtime = get_runtime()
    if not settings.agent_runtime_enabled:
        raise RuntimeError("Agent runtime is disabled via configuration")
    await runtime.ensure_started()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(_serve())
