from __future__ import annotations

import logging

from .runtime import AgentRuntime
from .strategies import HeartbeatConfig, HeartbeatStrategy
from ..config import settings


_runtime = AgentRuntime(logger=logging.getLogger("agent_runtime"))
_registered_defaults = False


def get_runtime() -> AgentRuntime:
    return _runtime


def register_builtin_strategies() -> None:
    global _registered_defaults
    if _registered_defaults:
        return
    heartbeat_cfg = HeartbeatConfig(interval_seconds=settings.agent_runtime_default_interval_seconds)
    _runtime.register_strategy(HeartbeatStrategy(config=heartbeat_cfg))
    _registered_defaults = True


__all__ = ["get_runtime", "register_builtin_strategies", "AgentRuntime"]
