"""
Polymarket Leaderboard Service

Track and rank top Polymarket traders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .models import PolymarketTraderProfile, LeaderboardEntry
from .tracker import PolymarketTraderTracker, get_trader_tracker

logger = logging.getLogger(__name__)


class PolymarketLeaderboard:
    """
    Leaderboard service for Polymarket traders.

    Tracks top traders and provides rankings.
    In production, this would be backed by a database.
    """

    def __init__(
        self,
        tracker: Optional[PolymarketTraderTracker] = None,
        convex_client: Optional[Any] = None,
    ):
        """Initialize leaderboard."""
        self.tracker = tracker or get_trader_tracker()
        self.convex = convex_client

        # In-memory leaderboard cache
        self._leaderboard: List[LeaderboardEntry] = []
        self._last_updated: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    async def get_leaderboard(
        self,
        sort_by: str = "total_pnl",
        limit: int = 50,
        min_trades: int = 10,
        time_period: Optional[str] = None,
    ) -> List[LeaderboardEntry]:
        """
        Get trader leaderboard.

        Args:
            sort_by: Metric to sort by (total_pnl, roi, win_rate, volume)
            limit: Max entries to return
            min_trades: Minimum trades to qualify
            time_period: Filter by time (7d, 30d, all)

        Returns:
            Ranked list of traders
        """
        # In production, this would query from database
        # For now, return cached leaderboard
        if self._leaderboard:
            return self._sort_leaderboard(self._leaderboard, sort_by, limit)

        return []

    async def add_trader_to_leaderboard(
        self,
        address: str,
        refresh: bool = False,
    ) -> Optional[LeaderboardEntry]:
        """
        Add or update a trader in the leaderboard.

        Args:
            address: Trader's wallet address
            refresh: Force refresh profile

        Returns:
            Leaderboard entry or None if insufficient data
        """
        try:
            profile = await self.tracker.get_trader_profile(address, refresh=refresh)

            if not profile.is_experienced:
                logger.info(f"Trader {address} has insufficient trade history")
                return None

            entry = LeaderboardEntry(
                rank=0,  # Will be calculated when sorting
                address=address,
                totalPnlUsd=profile.metrics.total_pnl_usd,
                roiPct=profile.metrics.roi_pct,
                winRate=profile.metrics.win_rate,
                totalVolumeUsd=profile.metrics.total_volume_usd,
                activePositions=profile.active_positions,
                totalTrades=profile.metrics.total_trades,
                followerCount=profile.follower_count,
            )

            # Update or add to leaderboard
            existing_idx = next(
                (i for i, e in enumerate(self._leaderboard) if e.address == address),
                None
            )

            if existing_idx is not None:
                self._leaderboard[existing_idx] = entry
            else:
                self._leaderboard.append(entry)

            self._last_updated = datetime.utcnow()
            return entry

        except Exception as e:
            logger.error(f"Failed to add trader {address} to leaderboard: {e}")
            return None

    async def get_top_traders(
        self,
        metric: str = "roi",
        limit: int = 20,
        min_trades: int = 20,
    ) -> List[PolymarketTraderProfile]:
        """
        Get top traders by metric.

        Args:
            metric: roi, pnl, win_rate, volume
            limit: Max traders to return
            min_trades: Minimum trades required

        Returns:
            List of trader profiles
        """
        leaderboard = await self.get_leaderboard(
            sort_by=metric,
            limit=limit,
            min_trades=min_trades,
        )

        profiles = []
        for entry in leaderboard:
            try:
                profile = await self.tracker.get_trader_profile(entry.address)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to get profile for {entry.address}: {e}")
                continue

        return profiles

    async def search_traders(
        self,
        query: str,
        limit: int = 20,
    ) -> List[LeaderboardEntry]:
        """
        Search traders by address.

        Args:
            query: Search query (address prefix)
            limit: Max results

        Returns:
            Matching leaderboard entries
        """
        query_lower = query.lower()
        matches = [
            entry for entry in self._leaderboard
            if query_lower in entry.address.lower()
        ]
        return matches[:limit]

    def _sort_leaderboard(
        self,
        entries: List[LeaderboardEntry],
        sort_by: str,
        limit: int,
    ) -> List[LeaderboardEntry]:
        """Sort leaderboard by metric."""
        sort_key = {
            "total_pnl": lambda e: float(e.total_pnl_usd),
            "pnl": lambda e: float(e.total_pnl_usd),
            "roi": lambda e: e.roi_pct,
            "win_rate": lambda e: e.win_rate,
            "volume": lambda e: float(e.total_volume_usd),
            "trades": lambda e: e.total_trades,
            "followers": lambda e: e.follower_count,
        }.get(sort_by, lambda e: float(e.total_pnl_usd))

        sorted_entries = sorted(entries, key=sort_key, reverse=True)

        # Assign ranks
        for i, entry in enumerate(sorted_entries):
            entry.rank = i + 1

        return sorted_entries[:limit]

    async def get_trader_rank(
        self,
        address: str,
        metric: str = "total_pnl",
    ) -> Optional[int]:
        """
        Get a trader's rank in the leaderboard.

        Args:
            address: Trader's wallet address
            metric: Metric to rank by

        Returns:
            Rank (1-indexed) or None if not found
        """
        sorted_lb = self._sort_leaderboard(
            self._leaderboard,
            sort_by=metric,
            limit=len(self._leaderboard),
        )

        for entry in sorted_lb:
            if entry.address.lower() == address.lower():
                return entry.rank

        return None


# Singleton instance
_leaderboard_instance: Optional[PolymarketLeaderboard] = None


def get_leaderboard() -> PolymarketLeaderboard:
    """Get singleton leaderboard instance."""
    global _leaderboard_instance
    if _leaderboard_instance is None:
        _leaderboard_instance = PolymarketLeaderboard()
    return _leaderboard_instance
