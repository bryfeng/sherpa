"""
Strategy Modules

Contains implementations for various automated trading strategies.
"""

from .dca import (
    DCAStrategy,
    DCAConfig,
    DCAExecution,
    DCAExecutor,
    DCAService,
    DCAFrequency,
    DCAStatus,
    ExecutionStatus,
    SkipReason,
)
from .generic_executor import (
    GenericStrategyExecutor,
    ExecutionResult,
    ExecutionStatus as GenericExecutionStatus,
    SwapParams,
    get_generic_executor,
)

__all__ = [
    # DCA
    "DCAStrategy",
    "DCAConfig",
    "DCAExecution",
    "DCAExecutor",
    "DCAService",
    "DCAFrequency",
    "DCAStatus",
    "ExecutionStatus",
    "SkipReason",
    # Generic (Phase 13)
    "GenericStrategyExecutor",
    "ExecutionResult",
    "GenericExecutionStatus",
    "SwapParams",
    "get_generic_executor",
]
