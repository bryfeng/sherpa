"""
DCA (Dollar Cost Averaging) Strategy Module

Provides automated DCA strategy execution with session key support.
"""

from .models import (
    DCAStrategy,
    DCAConfig,
    DCAExecution,
    DCAFrequency,
    DCAStatus,
    ExecutionStatus,
    SkipReason,
    TokenInfo,
    MarketConditions,
    ExecutionQuote,
    SessionKeyRequirements,
)
from .executor import DCAExecutor
from .service import DCAService
from .scheduler import DCAScheduler

__all__ = [
    # Models
    "DCAStrategy",
    "DCAConfig",
    "DCAExecution",
    "DCAFrequency",
    "DCAStatus",
    "ExecutionStatus",
    "SkipReason",
    "TokenInfo",
    "MarketConditions",
    "ExecutionQuote",
    "SessionKeyRequirements",
    # Service classes
    "DCAExecutor",
    "DCAService",
    "DCAScheduler",
]
