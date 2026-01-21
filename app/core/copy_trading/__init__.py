"""
Copy Trading Module

Enables users to follow and automatically replicate trades from successful wallets.
"""

from .models import (
    CopyConfig,
    CopyRelationship,
    CopyExecution,
    CopyExecutionStatus,
    SizingMode,
    LeaderProfile,
    TradeSignal,
    TradeAction,
    SkipReason,
)
from .sizing import (
    SizingStrategy,
    PercentageSizing,
    FixedSizing,
    ProportionalSizing,
    get_sizing_strategy,
)
from .manager import CopyTradingManager
from .executor import CopyExecutor
from .analytics import LeaderAnalytics
from .event_bridge import (
    CopyTradingEventBridge,
    get_copy_trading_bridge,
    start_copy_trading_bridge,
    stop_copy_trading_bridge,
)

__all__ = [
    # Models
    "CopyConfig",
    "CopyRelationship",
    "CopyExecution",
    "CopyExecutionStatus",
    "SizingMode",
    "LeaderProfile",
    "TradeSignal",
    "TradeAction",
    "SkipReason",
    # Sizing
    "SizingStrategy",
    "PercentageSizing",
    "FixedSizing",
    "ProportionalSizing",
    "get_sizing_strategy",
    # Manager
    "CopyTradingManager",
    # Executor
    "CopyExecutor",
    # Analytics
    "LeaderAnalytics",
    # Event Bridge
    "CopyTradingEventBridge",
    "get_copy_trading_bridge",
    "start_copy_trading_bridge",
    "stop_copy_trading_bridge",
]
