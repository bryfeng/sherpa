"""
Strategy State Machine Module

Manages strategy execution state transitions with validation,
persistence, and event handling.
"""

from .handlers import StateHandlers, register_handlers
from .models import (
    ExecutionContext,
    ExecutionError,
    ExecutionStep,
    InvalidTransitionError,
    StateTransition,
    StateTransitionTrigger,
    StrategyState,
)
from .persistence import ExecutionPersistence, create_persistence_callback
from .state_machine import StrategyStateMachine

__all__ = [
    # State Machine
    "StrategyStateMachine",
    # Handlers
    "StateHandlers",
    "register_handlers",
    # Models
    "StrategyState",
    "StateTransition",
    "StateTransitionTrigger",
    "ExecutionContext",
    "ExecutionStep",
    # Errors
    "InvalidTransitionError",
    "ExecutionError",
    # Persistence
    "ExecutionPersistence",
    "create_persistence_callback",
]
