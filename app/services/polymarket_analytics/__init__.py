"""
Polymarket Analytics Service

Track and analyze Polymarket trader performance for copy trading.
"""

from .models import (
    PolymarketTraderProfile,
    PerformanceMetrics,
    TraderPosition,
    HistoricalTrade,
    LeaderboardEntry,
)
from .tracker import PolymarketTraderTracker, get_trader_tracker
from .leaderboard import PolymarketLeaderboard, get_leaderboard

__all__ = [
    # Models
    "PolymarketTraderProfile",
    "PerformanceMetrics",
    "TraderPosition",
    "HistoricalTrade",
    "LeaderboardEntry",
    # Tracker
    "PolymarketTraderTracker",
    "get_trader_tracker",
    # Leaderboard
    "PolymarketLeaderboard",
    "get_leaderboard",
]
