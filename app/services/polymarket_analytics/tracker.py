"""
Polymarket Trader Tracker

Track and analyze Polymarket trader performance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from collections import defaultdict

from app.providers.polymarket import (
    PolymarketClient,
    get_polymarket_client,
    Position,
    Trade,
    Market,
)
from .models import (
    PolymarketTraderProfile,
    PerformanceMetrics,
    TraderPosition,
    HistoricalTrade,
)

logger = logging.getLogger(__name__)


class PolymarketTraderTracker:
    """
    Track and analyze Polymarket trader performance.

    Provides:
    - Trader profiles with performance metrics
    - Position tracking
    - Trade history analysis
    - Brier score calculation for prediction quality
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
    ):
        """Initialize tracker."""
        self.client = client or get_polymarket_client()
        self._profile_cache: Dict[str, PolymarketTraderProfile] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def get_trader_profile(
        self,
        address: str,
        refresh: bool = False,
    ) -> PolymarketTraderProfile:
        """
        Get comprehensive trader profile.

        Args:
            address: Trader's wallet address
            refresh: Force refresh from API

        Returns:
            Trader profile with performance metrics
        """
        # Check cache
        if not refresh and address in self._profile_cache:
            cached = self._profile_cache[address]
            if cached.last_analyzed_at:
                age = datetime.utcnow() - cached.last_analyzed_at
                if age < self._cache_ttl:
                    return cached

        # Fetch data
        positions = await self.get_trader_positions(address)
        trades = await self.get_trader_history(address)

        # Calculate metrics
        metrics = self._calculate_performance(trades, positions)

        # Analyze positions
        active_positions = len(positions)
        total_exposure = sum(p.current_value for p in positions)
        avg_position = total_exposure / active_positions if active_positions > 0 else Decimal("0")
        max_position = max((p.current_value for p in positions), default=Decimal("0"))

        # Calculate max single bet as percentage
        max_single_bet_pct = 0.0
        if total_exposure > 0:
            max_single_bet_pct = float(max_position / total_exposure) * 100

        # Analyze categories
        categories = self._analyze_categories(trades)

        # Calculate hold time
        avg_hold_time = self._calculate_avg_hold_time(trades)

        # Calculate activity rate
        trades_per_week = 0.0
        if trades:
            first_trade = min(t.timestamp for t in trades)
            days_active = (datetime.utcnow() - first_trade).days or 1
            trades_per_week = len(trades) / (days_active / 7)

        # Calculate diversification (based on number of categories and position spread)
        diversification = self._calculate_diversification(positions, categories)

        # Calculate risk score
        risk_score = self._calculate_risk_score(metrics, max_single_bet_pct, diversification)

        profile = PolymarketTraderProfile(
            address=address,
            metrics=metrics,
            activePositions=active_positions,
            totalExposureUsd=total_exposure,
            avgPositionSizeUsd=avg_position,
            maxPositionSizeUsd=max_position,
            maxSingleBetPct=max_single_bet_pct,
            preferredCategories=list(categories.keys())[:5],
            categoryPerformance=categories,
            avgHoldTimeDays=avg_hold_time,
            tradesPerWeek=trades_per_week,
            diversificationScore=diversification,
            riskScore=risk_score,
            usesLeverage=False,  # Polymarket doesn't support leverage
            firstTradeAt=min((t.timestamp for t in trades), default=None) if trades else None,
            lastTradeAt=max((t.timestamp for t in trades), default=None) if trades else None,
            lastAnalyzedAt=datetime.utcnow(),
            dataQualityScore=min(1.0, len(trades) / 50),  # More trades = higher quality
            tradeCountForAnalysis=len(trades),
        )

        self._profile_cache[address] = profile
        return profile

    async def get_trader_positions(self, address: str) -> List[TraderPosition]:
        """
        Get trader's current positions.

        Args:
            address: Trader's wallet address

        Returns:
            List of current positions
        """
        raw_positions = await self.client.get_positions(address)

        positions = []
        for pos in raw_positions:
            # Get market details for context
            market = await self.client.get_market(pos.market_id)

            market_question = pos.market_id
            outcome_name = f"Outcome {pos.outcome_index}"
            market_end_date = None
            market_resolved = False
            market_category = None

            if market:
                market_question = market.question
                market_end_date = market.end_date
                market_resolved = market.resolved
                market_category = market.category.value if market.category else None
                if market.outcomes and pos.outcome_index < len(market.outcomes):
                    outcome_name = market.outcomes[pos.outcome_index]

            # Calculate P&L percentage
            unrealized_pnl_pct = None
            if pos.cost_basis and pos.cost_basis > 0:
                unrealized_pnl_pct = float(pos.unrealized_pnl / pos.cost_basis) * 100

            positions.append(TraderPosition(
                marketId=pos.market_id,
                marketQuestion=market_question,
                outcome=outcome_name,
                tokenId=pos.token_id,
                shares=pos.size,
                avgEntryPrice=pos.avg_price,
                currentPrice=pos.current_price,
                costBasis=pos.cost_basis,
                currentValue=pos.current_value,
                unrealizedPnl=pos.unrealized_pnl,
                unrealizedPnlPct=unrealized_pnl_pct,
                marketEndDate=market_end_date,
                marketResolved=market_resolved,
                marketCategory=market_category,
            ))

        return positions

    async def get_trader_history(
        self,
        address: str,
        limit: int = 200,
    ) -> List[HistoricalTrade]:
        """
        Get trader's trade history.

        Args:
            address: Trader's wallet address
            limit: Max trades to fetch

        Returns:
            List of historical trades
        """
        raw_trades = await self.client.get_trades(address, limit=limit)

        trades = []
        market_cache: Dict[str, Market] = {}

        for trade in raw_trades:
            # Get market details (cached)
            market = market_cache.get(trade.market_id)
            if not market:
                market = await self.client.get_market(trade.market_id)
                if market:
                    market_cache[trade.market_id] = market

            market_question = trade.market_id
            outcome_name = "Unknown"
            winning_outcome = None

            if market:
                market_question = market.question
                # Find outcome name from token
                for token in market.tokens:
                    if token.token_id == trade.token_id:
                        outcome_name = token.outcome
                        break
                if market.resolved:
                    # Find winning outcome
                    for token in market.tokens:
                        if token.winner:
                            winning_outcome = token.outcome
                            break

            trades.append(HistoricalTrade(
                tradeId=trade.id,
                marketId=trade.market_id,
                marketQuestion=market_question,
                outcome=outcome_name,
                tokenId=trade.token_id,
                side=trade.side.value,
                shares=trade.size,
                price=trade.price,
                valueUsd=trade.value_usd,
                fee=trade.fee,
                marketResolved=market.resolved if market else False,
                winningOutcome=winning_outcome,
                timestamp=trade.timestamp,
                transactionHash=trade.transaction_hash,
            ))

        return trades

    def _calculate_performance(
        self,
        trades: List[HistoricalTrade],
        positions: List[TraderPosition],
    ) -> PerformanceMetrics:
        """Calculate performance metrics from trade history."""
        if not trades:
            return PerformanceMetrics(
                totalVolumeUsd=Decimal("0"),
                totalPnlUsd=Decimal("0"),
                roiPct=0.0,
                totalTrades=0,
                winningTrades=0,
                losingTrades=0,
                winRate=0.0,
            )

        total_volume = sum(t.value_usd for t in trades)
        realized_pnl = Decimal("0")
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)

        # Calculate realized P&L from resolved markets
        winning_trades = 0
        losing_trades = 0
        best_trade = Decimal("0")
        worst_trade = Decimal("0")

        # Group trades by market to calculate P&L
        market_trades: Dict[str, List[HistoricalTrade]] = defaultdict(list)
        for trade in trades:
            market_trades[trade.market_id].append(trade)

        for market_id, market_trade_list in market_trades.items():
            # Calculate net position and P&L for this market
            buys = [t for t in market_trade_list if t.side == "BUY"]
            sells = [t for t in market_trade_list if t.side == "SELL"]

            buy_cost = sum(t.value_usd for t in buys)
            sell_proceeds = sum(t.value_usd for t in sells)

            # Check if market resolved
            resolved_trade = next((t for t in market_trade_list if t.market_resolved), None)
            if resolved_trade:
                # For resolved markets, calculate based on outcome
                # If trader held winning outcome, they get $1 per share
                # This is simplified - full implementation would track exact holdings
                pnl = sell_proceeds - buy_cost
                if resolved_trade.payout_received:
                    pnl += resolved_trade.payout_received

                realized_pnl += pnl

                if pnl > 0:
                    winning_trades += 1
                    if pnl > best_trade:
                        best_trade = pnl
                elif pnl < 0:
                    losing_trades += 1
                    if pnl < worst_trade:
                        worst_trade = pnl

        total_pnl = realized_pnl + unrealized_pnl
        roi_pct = float(total_pnl / total_volume * 100) if total_volume > 0 else 0.0

        total_decided = winning_trades + losing_trades
        win_rate = winning_trades / total_decided * 100 if total_decided > 0 else 0.0

        # Calculate streaks
        current_streak, best_streak, worst_streak = self._calculate_streaks(trades)

        # Calculate Brier score for resolved markets
        brier_score = self._calculate_brier_score(trades)

        # Volume in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        volume_30d = sum(t.value_usd for t in trades if t.timestamp > thirty_days_ago)

        return PerformanceMetrics(
            totalVolumeUsd=total_volume,
            volume30dUsd=volume_30d,
            totalPnlUsd=total_pnl,
            realizedPnlUsd=realized_pnl,
            unrealizedPnlUsd=unrealized_pnl,
            roiPct=roi_pct,
            totalTrades=len(trades),
            winningTrades=winning_trades,
            losingTrades=losing_trades,
            winRate=win_rate,
            brierScore=brier_score,
            currentStreak=current_streak,
            bestStreak=best_streak,
            worstStreak=worst_streak,
            bestTradePnlUsd=best_trade if best_trade > 0 else None,
            worstTradePnlUsd=worst_trade if worst_trade < 0 else None,
        )

    def _calculate_brier_score(self, trades: List[HistoricalTrade]) -> Optional[float]:
        """
        Calculate Brier score for prediction quality.

        Brier score measures how well-calibrated predictions are.
        Score of 0 = perfect, 1 = worst possible.
        """
        resolved_buys = [
            t for t in trades
            if t.side == "BUY" and t.market_resolved and t.winning_outcome
        ]

        if len(resolved_buys) < 5:
            return None

        total_score = 0.0
        for trade in resolved_buys:
            # Predicted probability is the price paid
            predicted = float(trade.price)
            # Actual outcome (1 if trader's outcome won, 0 otherwise)
            actual = 1.0 if trade.outcome == trade.winning_outcome else 0.0
            # Brier score component
            total_score += (predicted - actual) ** 2

        return total_score / len(resolved_buys)

    def _calculate_streaks(
        self,
        trades: List[HistoricalTrade],
    ) -> tuple[int, int, int]:
        """Calculate winning/losing streaks."""
        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        current_streak = 0
        best_streak = 0
        worst_streak = 0
        prev_result = None

        for trade in sorted_trades:
            if not trade.market_resolved:
                continue

            is_win = trade.outcome == trade.winning_outcome

            if is_win:
                if prev_result is True or prev_result is None:
                    current_streak = max(1, current_streak + 1)
                else:
                    current_streak = 1
                best_streak = max(best_streak, current_streak)
            else:
                if prev_result is False or prev_result is None:
                    current_streak = min(-1, current_streak - 1)
                else:
                    current_streak = -1
                worst_streak = min(worst_streak, current_streak)

            prev_result = is_win

        return current_streak, best_streak, worst_streak

    def _analyze_categories(
        self,
        trades: List[HistoricalTrade],
    ) -> Dict[str, dict]:
        """Analyze performance by market category."""
        # This would require market category data
        # For now, return empty - can be enhanced with category tracking
        return {}

    def _calculate_avg_hold_time(self, trades: List[HistoricalTrade]) -> float:
        """Calculate average position hold time in days."""
        # Group by market and calculate time between first buy and last sell
        market_times: Dict[str, tuple[datetime, datetime]] = {}

        for trade in trades:
            market_id = trade.market_id
            if market_id not in market_times:
                market_times[market_id] = (trade.timestamp, trade.timestamp)
            else:
                first, last = market_times[market_id]
                market_times[market_id] = (
                    min(first, trade.timestamp),
                    max(last, trade.timestamp),
                )

        if not market_times:
            return 0.0

        total_days = sum(
            (last - first).days
            for first, last in market_times.values()
        )

        return total_days / len(market_times)

    def _calculate_diversification(
        self,
        positions: List[TraderPosition],
        categories: Dict[str, dict],
    ) -> float:
        """
        Calculate diversification score (0-1).

        Higher = more diversified across markets and categories.
        """
        if not positions:
            return 0.5

        # Factor 1: Number of unique markets
        unique_markets = len(set(p.market_id for p in positions))
        market_score = min(1.0, unique_markets / 10)

        # Factor 2: Position concentration (Herfindahl index)
        total_value = sum(p.current_value for p in positions)
        if total_value > 0:
            shares = [float(p.current_value / total_value) for p in positions]
            hhi = sum(s ** 2 for s in shares)
            concentration_score = 1 - hhi  # Invert so higher = more diversified
        else:
            concentration_score = 0.5

        # Combine factors
        return (market_score + concentration_score) / 2

    def _calculate_risk_score(
        self,
        metrics: PerformanceMetrics,
        max_single_bet_pct: float,
        diversification: float,
    ) -> float:
        """
        Calculate risk score (0-1).

        Higher = riskier trading behavior.
        """
        # Factor 1: Position concentration
        concentration_risk = max_single_bet_pct / 100

        # Factor 2: Win rate variance from 50%
        win_rate_risk = abs(metrics.win_rate - 50) / 50

        # Factor 3: Inverse diversification
        diversity_risk = 1 - diversification

        # Factor 4: Trade frequency (more trades = more risk)
        frequency_risk = min(1.0, metrics.total_trades / 100)

        # Weighted average
        return (
            concentration_risk * 0.3 +
            win_rate_risk * 0.2 +
            diversity_risk * 0.3 +
            frequency_risk * 0.2
        )


# Singleton instance
_tracker_instance: Optional[PolymarketTraderTracker] = None


def get_trader_tracker() -> PolymarketTraderTracker:
    """Get singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PolymarketTraderTracker()
    return _tracker_instance
