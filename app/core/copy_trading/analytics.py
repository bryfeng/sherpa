"""
Leader Analytics

Track and analyze performance of wallet leaders for copy trading.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .models import LeaderProfile, TradeSignal

logger = logging.getLogger(__name__)


class LeaderAnalytics:
    """
    Analyze leader wallet performance.

    Calculates:
    - Win rate
    - Average P&L per trade
    - Total P&L
    - Sharpe ratio
    - Maximum drawdown
    - Trading patterns
    """

    def __init__(
        self,
        convex_client: Optional[Any] = None,
        price_provider: Optional[Any] = None,
    ):
        self.convex_client = convex_client
        self.price_provider = price_provider

        # Cache of leader profiles
        self._profiles: Dict[str, LeaderProfile] = {}

    async def get_profile(
        self,
        address: str,
        chain: str,
        refresh: bool = False,
    ) -> Optional[LeaderProfile]:
        """
        Get or create a leader profile.

        Args:
            address: Wallet address
            chain: Chain identifier
            refresh: Force recalculation

        Returns:
            Leader profile with performance metrics
        """
        key = f"{chain}:{address.lower()}"

        # Check cache
        if not refresh and key in self._profiles:
            profile = self._profiles[key]
            # Return if recently analyzed (within 1 hour)
            if profile.last_analyzed_at:
                age = (datetime.now(timezone.utc) - profile.last_analyzed_at).total_seconds()
                if age < 3600:
                    return profile

        # Load from storage or create new
        profile = await self._load_profile(address, chain)
        if not profile:
            profile = LeaderProfile(address=address.lower(), chain=chain)

        # Recalculate if needed
        if refresh or not profile.last_analyzed_at:
            profile = await self._analyze_leader(profile)

        self._profiles[key] = profile
        return profile

    async def get_leaderboard(
        self,
        chain: Optional[str] = None,
        sort_by: str = "total_pnl_usd",
        limit: int = 50,
        min_trades: int = 10,
    ) -> List[LeaderProfile]:
        """
        Get ranked leaderboard of top performers.

        Args:
            chain: Filter by chain (None for all)
            sort_by: Metric to sort by
            limit: Max results
            min_trades: Minimum trades required

        Returns:
            Sorted list of leader profiles
        """
        if not self.convex_client:
            return list(self._profiles.values())[:limit]

        try:
            data = await self.convex_client.query(
                "watchedWallets:getLeaderboard",
                {
                    "chain": chain,
                    "sortBy": sort_by,
                    "limit": limit,
                    "minTrades": min_trades,
                },
            )

            return [self._dict_to_profile(d) for d in data]

        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []

    async def update_from_trade(
        self,
        signal: TradeSignal,
        pnl_usd: Optional[Decimal] = None,
    ):
        """
        Update leader metrics from a new trade.

        Called when we detect a new trade from a watched wallet.
        """
        key = f"{signal.leader_chain}:{signal.leader_address.lower()}"

        profile = self._profiles.get(key)
        if not profile:
            profile = await self.get_profile(signal.leader_address, signal.leader_chain)

        if not profile:
            return

        # Update trade count
        profile.total_trades += 1
        profile.last_active_at = datetime.now(timezone.utc)

        # Update token preferences
        if signal.token_out_symbol and signal.token_out_symbol not in profile.most_traded_tokens:
            profile.most_traded_tokens.append(signal.token_out_symbol)
            # Keep only top 10
            if len(profile.most_traded_tokens) > 10:
                profile.most_traded_tokens = profile.most_traded_tokens[-10:]

        # Update P&L if provided
        if pnl_usd is not None:
            if profile.total_pnl_usd is None:
                profile.total_pnl_usd = Decimal("0")
            profile.total_pnl_usd += pnl_usd

        # Save
        await self._save_profile(profile)

    async def _analyze_leader(self, profile: LeaderProfile) -> LeaderProfile:
        """
        Perform full analysis of a leader's trading history.
        """
        if not self.convex_client:
            profile.last_analyzed_at = datetime.now(timezone.utc)
            return profile

        try:
            # Get trading history
            trades = await self._get_trade_history(
                profile.address,
                profile.chain,
                days=90,
            )

            if not trades:
                profile.last_analyzed_at = datetime.now(timezone.utc)
                return profile

            # Calculate metrics
            profile.total_trades = len(trades)
            profile = self._calculate_win_rate(profile, trades)
            profile = self._calculate_pnl_metrics(profile, trades)
            profile = self._calculate_risk_metrics(profile, trades)
            profile = self._calculate_activity_metrics(profile, trades)
            profile = self._calculate_data_quality(profile, trades)

            profile.last_analyzed_at = datetime.now(timezone.utc)

            # Save updated profile
            await self._save_profile(profile)

        except Exception as e:
            logger.error(f"Error analyzing leader {profile.address}: {e}", exc_info=True)

        return profile

    async def _get_trade_history(
        self,
        address: str,
        chain: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Get historical trades for a wallet."""
        if not self.convex_client:
            return []

        since = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            return await self.convex_client.query(
                "walletActivity:getByAddress",
                {
                    "address": address.lower(),
                    "chain": chain,
                    "eventType": "swap",
                    "limit": 1000,
                    "sinceTimestamp": int(since.timestamp() * 1000),
                },
            )
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def _calculate_win_rate(
        self,
        profile: LeaderProfile,
        trades: List[Dict[str, Any]],
    ) -> LeaderProfile:
        """Calculate win rate from trades."""
        if not trades:
            return profile

        # Group trades by token to calculate P&L
        # This is a simplified calculation - real implementation would
        # need to track positions and actual realized P&L
        profitable = 0
        total = 0

        for i, trade in enumerate(trades):
            # Skip if no value
            if not trade.get("valueUsd"):
                continue

            total += 1

            # Simple heuristic: check if price moved favorably after trade
            # In production, would track actual position P&L
            pnl_indicator = trade.get("pnlUsd") or trade.get("estimatedPnl")
            if pnl_indicator and float(pnl_indicator) > 0:
                profitable += 1

        if total > 0:
            profile.win_rate = profitable / total

        return profile

    def _calculate_pnl_metrics(
        self,
        profile: LeaderProfile,
        trades: List[Dict[str, Any]],
    ) -> LeaderProfile:
        """Calculate P&L metrics."""
        pnls: List[float] = []

        for trade in trades:
            pnl = trade.get("pnlUsd") or trade.get("estimatedPnl")
            if pnl is not None:
                pnls.append(float(pnl))

        if pnls:
            profile.total_pnl_usd = Decimal(str(sum(pnls)))
            profile.avg_trade_pnl_pct = sum(pnls) / len(pnls) / 100  # Assume average trade size

            # Calculate Sharpe ratio (simplified)
            if len(pnls) > 1:
                mean_return = sum(pnls) / len(pnls)
                variance = sum((p - mean_return) ** 2 for p in pnls) / len(pnls)
                std_dev = math.sqrt(variance) if variance > 0 else 1

                # Annualized Sharpe (assuming daily trades)
                profile.sharpe_ratio = (mean_return / std_dev) * math.sqrt(365) if std_dev > 0 else 0

        return profile

    def _calculate_risk_metrics(
        self,
        profile: LeaderProfile,
        trades: List[Dict[str, Any]],
    ) -> LeaderProfile:
        """Calculate risk metrics including max drawdown."""
        cumulative_pnl: List[float] = []
        running_total = 0.0

        for trade in trades:
            pnl = trade.get("pnlUsd") or trade.get("estimatedPnl") or 0
            running_total += float(pnl)
            cumulative_pnl.append(running_total)

        if cumulative_pnl:
            # Calculate max drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0.0

            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

            profile.max_drawdown_pct = max_drawdown * 100

        # Calculate position size metrics
        position_sizes: List[float] = []
        for trade in trades:
            value = trade.get("valueUsd")
            if value:
                position_sizes.append(float(value))

        if position_sizes:
            avg_size = sum(position_sizes) / len(position_sizes)
            max_size = max(position_sizes)

            # As percentage of a hypothetical $10k portfolio
            profile.avg_position_size_pct = avg_size / 10000 * 100
            profile.max_position_size_pct = max_size / 10000 * 100

        return profile

    def _calculate_activity_metrics(
        self,
        profile: LeaderProfile,
        trades: List[Dict[str, Any]],
    ) -> LeaderProfile:
        """Calculate activity patterns."""
        if not trades:
            return profile

        # Calculate trades per day
        if len(trades) >= 2:
            first_trade = trades[-1].get("timestamp", 0)
            last_trade = trades[0].get("timestamp", 0)

            if first_trade and last_trade:
                days = (last_trade - first_trade) / (1000 * 60 * 60 * 24)
                if days > 0:
                    profile.avg_trades_per_day = len(trades) / days

        # Find most traded tokens
        token_counts: Dict[str, int] = {}
        for trade in trades:
            token = trade.get("tokenOutSymbol") or trade.get("tokenInSymbol")
            if token:
                token_counts[token] = token_counts.get(token, 0) + 1

        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        profile.most_traded_tokens = [t[0] for t in sorted_tokens[:10]]

        # Determine preferred sectors (would need token metadata)
        # For now, just use token symbols as a proxy

        return profile

    def _calculate_data_quality(
        self,
        profile: LeaderProfile,
        trades: List[Dict[str, Any]],
    ) -> LeaderProfile:
        """Calculate data quality score."""
        score = 0.0

        # More trades = better data
        if len(trades) >= 100:
            score += 0.3
        elif len(trades) >= 50:
            score += 0.2
        elif len(trades) >= 20:
            score += 0.1

        # Longer history = better
        if trades:
            first_trade = trades[-1].get("timestamp", 0)
            if first_trade:
                age_days = (datetime.now(timezone.utc).timestamp() * 1000 - first_trade) / (1000 * 60 * 60 * 24)
                if age_days >= 90:
                    score += 0.3
                elif age_days >= 30:
                    score += 0.2
                elif age_days >= 7:
                    score += 0.1

        # P&L data available
        has_pnl = any(t.get("pnlUsd") is not None for t in trades)
        if has_pnl:
            score += 0.2

        # Recent activity
        if profile.last_active_at:
            days_inactive = (datetime.now(timezone.utc) - profile.last_active_at).days
            if days_inactive <= 1:
                score += 0.2
            elif days_inactive <= 7:
                score += 0.1

        profile.data_quality_score = min(score, 1.0)
        return profile

    async def _load_profile(
        self,
        address: str,
        chain: str,
    ) -> Optional[LeaderProfile]:
        """Load profile from storage."""
        if not self.convex_client:
            return None

        try:
            data = await self.convex_client.query(
                "watchedWallets:getByAddress",
                {"address": address.lower(), "chain": chain},
            )
            if data:
                return self._dict_to_profile(data)
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")

        return None

    async def _save_profile(self, profile: LeaderProfile):
        """Save profile to storage."""
        key = f"{profile.chain}:{profile.address.lower()}"
        self._profiles[key] = profile

        if self.convex_client:
            try:
                await self.convex_client.mutation(
                    "watchedWallets:upsert",
                    self._profile_to_dict(profile),
                )
            except Exception as e:
                logger.error(f"Failed to save profile: {e}")

    def _profile_to_dict(self, p: LeaderProfile) -> Dict[str, Any]:
        """Convert profile to dict for storage."""
        return {
            "address": p.address.lower(),
            "chain": p.chain,
            "label": p.label,
            "notes": p.notes,
            "totalTrades": p.total_trades,
            "winRate": p.win_rate,
            "avgTradePnlPercent": p.avg_trade_pnl_pct,
            "totalPnlUsd": float(p.total_pnl_usd) if p.total_pnl_usd else None,
            "sharpeRatio": p.sharpe_ratio,
            "maxDrawdownPercent": p.max_drawdown_pct,
            "avgTradesPerDay": p.avg_trades_per_day,
            "avgHoldTimeHours": p.avg_hold_time_hours,
            "mostTradedTokens": p.most_traded_tokens,
            "preferredSectors": p.preferred_sectors,
            "avgPositionSizePercent": p.avg_position_size_pct,
            "maxPositionSizePercent": p.max_position_size_pct,
            "usesLeverage": p.uses_leverage,
            "followerCount": p.follower_count,
            "totalCopiedVolumeUsd": float(p.total_copied_volume_usd),
            "isActive": p.is_active,
            "firstSeenAt": int(p.first_seen_at.timestamp() * 1000),
            "lastActiveAt": int(p.last_active_at.timestamp() * 1000),
            "dataQualityScore": p.data_quality_score,
            "lastAnalyzedAt": int(p.last_analyzed_at.timestamp() * 1000) if p.last_analyzed_at else None,
        }

    def _dict_to_profile(self, d: Dict[str, Any]) -> LeaderProfile:
        """Convert dict from storage to profile."""
        return LeaderProfile(
            address=d["address"],
            chain=d["chain"],
            label=d.get("label"),
            notes=d.get("notes"),
            total_trades=d.get("totalTrades", 0),
            win_rate=d.get("winRate"),
            avg_trade_pnl_pct=d.get("avgTradePnlPercent"),
            total_pnl_usd=Decimal(str(d["totalPnlUsd"])) if d.get("totalPnlUsd") else None,
            sharpe_ratio=d.get("sharpeRatio"),
            max_drawdown_pct=d.get("maxDrawdownPercent"),
            avg_trades_per_day=d.get("avgTradesPerDay"),
            avg_hold_time_hours=d.get("avgHoldTimeHours"),
            most_traded_tokens=d.get("mostTradedTokens", []),
            preferred_sectors=d.get("preferredSectors", []),
            avg_position_size_pct=d.get("avgPositionSizePercent"),
            max_position_size_pct=d.get("maxPositionSizePercent"),
            uses_leverage=d.get("usesLeverage", False),
            follower_count=d.get("followerCount", 0),
            total_copied_volume_usd=Decimal(str(d.get("totalCopiedVolumeUsd", 0))),
            is_active=d.get("isActive", True),
            first_seen_at=datetime.fromtimestamp(d["firstSeenAt"] / 1000, tz=timezone.utc) if d.get("firstSeenAt") else datetime.now(timezone.utc),
            last_active_at=datetime.fromtimestamp(d["lastActiveAt"] / 1000, tz=timezone.utc) if d.get("lastActiveAt") else datetime.now(timezone.utc),
            data_quality_score=d.get("dataQualityScore", 0),
            last_analyzed_at=datetime.fromtimestamp(d["lastAnalyzedAt"] / 1000, tz=timezone.utc) if d.get("lastAnalyzedAt") else None,
        )
