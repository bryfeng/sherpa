"""
Polymarket Trading Service

High-level service for trading on Polymarket prediction markets.
Handles portfolio management, trade quotes, and AI-powered analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.providers.polymarket import (
    PolymarketClient,
    get_polymarket_client,
    Market,
    Position,
    OrderSide,
    MarketCategory,
)
from .models import (
    PolymarketPortfolio,
    PortfolioPosition,
    MarketAnalysis,
    TradeQuote,
    TradeResult,
)

logger = logging.getLogger(__name__)


class PolymarketTradingService:
    """
    Service for trading on Polymarket prediction markets.

    Provides:
    - Market discovery and search
    - Portfolio tracking with P&L
    - Trade quotes (manual approval flow)
    - AI-powered market analysis
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
    ):
        """
        Initialize trading service.

        Args:
            client: Polymarket client (uses singleton if not provided)
        """
        self.client = client or get_polymarket_client()

    # =========================================================================
    # Market Discovery
    # =========================================================================

    async def get_markets(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
        trending: bool = False,
        closing_soon_hours: Optional[int] = None,
        limit: int = 50,
    ) -> List[Market]:
        """
        Get markets with various filters.

        Args:
            category: Filter by category (politics, crypto, sports, etc.)
            query: Search query
            trending: Get trending markets by volume
            closing_soon_hours: Get markets closing within hours
            limit: Max results

        Returns:
            List of markets
        """
        if trending:
            return await self.client.get_trending_markets(limit=limit)

        if closing_soon_hours:
            return await self.client.get_closing_soon(hours=closing_soon_hours, limit=limit)

        if query:
            return await self.client.search_markets(query=query, limit=limit)

        if category:
            try:
                cat_enum = MarketCategory(category.lower())
                return await self.client.get_markets_by_category(category=cat_enum, limit=limit)
            except ValueError:
                logger.warning(f"Unknown category: {category}, fetching all markets")

        return await self.client.get_markets(active=True, limit=limit)

    async def get_market(self, market_id: str) -> Optional[Market]:
        """
        Get a single market with current prices.

        Args:
            market_id: Market condition ID

        Returns:
            Market or None
        """
        market = await self.client.get_market(market_id)
        if not market:
            return None

        # Enrich with live prices
        for token in market.tokens:
            if token.token_id:
                mid = await self.client.get_midpoint(token.token_id)
                if mid is not None:
                    token.price = mid

        return market

    async def get_market_details(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed market information including orderbook.

        Args:
            market_id: Market condition ID

        Returns:
            Dict with market, orderbooks, and spread data
        """
        market = await self.get_market(market_id)
        if not market:
            return None

        result = {
            "market": market,
            "orderbooks": {},
            "spreads": {},
        }

        for token in market.tokens:
            if token.token_id:
                ob = await self.client.get_orderbook(token.token_id)
                spread = await self.client.get_spread(token.token_id)
                if ob:
                    result["orderbooks"][token.outcome] = ob
                if spread:
                    result["spreads"][token.outcome] = spread

        return result

    # =========================================================================
    # Portfolio
    # =========================================================================

    async def get_portfolio(self, address: str) -> PolymarketPortfolio:
        """
        Get user's complete portfolio with P&L calculations.

        Args:
            address: User's wallet address

        Returns:
            Portfolio with positions and aggregates
        """
        positions = await self.client.get_positions(address)

        # Enrich positions with market context
        enriched_positions: List[PortfolioPosition] = []
        total_value = Decimal("0")
        total_cost_basis = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        total_realized_pnl = Decimal("0")
        winning = 0
        losing = 0

        for pos in positions:
            # Get market details for context
            market = await self.client.get_market(pos.market_id)

            # Determine outcome name
            outcome_name = "Unknown"
            market_question = pos.market_id
            market_end_date = None
            market_resolved = False

            if market:
                market_question = market.question
                market_end_date = market.end_date
                market_resolved = market.resolved
                if market.outcomes and pos.outcome_index < len(market.outcomes):
                    outcome_name = market.outcomes[pos.outcome_index]

            # Calculate P&L percentage
            unrealized_pnl_pct = None
            if pos.cost_basis and pos.cost_basis > 0:
                unrealized_pnl_pct = float(pos.unrealized_pnl / pos.cost_basis) * 100

            enriched = PortfolioPosition(
                marketId=pos.market_id,
                tokenId=pos.token_id,
                outcomeIndex=pos.outcome_index,
                size=pos.size,
                avgPrice=pos.avg_price,
                currentPrice=pos.current_price,
                costBasis=pos.cost_basis,
                currentValue=pos.current_value,
                unrealizedPnl=pos.unrealized_pnl,
                unrealizedPnlPct=unrealized_pnl_pct,
                realizedPnl=pos.realized_pnl,
                marketQuestion=market_question,
                outcomeName=outcome_name,
                marketEndDate=market_end_date,
                marketResolved=market_resolved,
            )

            enriched_positions.append(enriched)

            # Aggregate
            total_value += pos.current_value
            total_cost_basis += pos.cost_basis
            total_unrealized_pnl += pos.unrealized_pnl
            total_realized_pnl += pos.realized_pnl

            if pos.unrealized_pnl > 0:
                winning += 1
            elif pos.unrealized_pnl < 0:
                losing += 1

        # Calculate total P&L percentage
        total_pnl = total_unrealized_pnl + total_realized_pnl
        total_pnl_pct = None
        if total_cost_basis > 0:
            total_pnl_pct = float(total_pnl / total_cost_basis) * 100

        return PolymarketPortfolio(
            address=address,
            positions=enriched_positions,
            totalValue=total_value,
            totalCostBasis=total_cost_basis,
            totalUnrealizedPnl=total_unrealized_pnl,
            totalRealizedPnl=total_realized_pnl,
            totalPnl=total_pnl,
            totalPnlPct=total_pnl_pct,
            openPositionsCount=len(enriched_positions),
            winningPositions=winning,
            losingPositions=losing,
            updatedAt=datetime.utcnow(),
        )

    # =========================================================================
    # Trading (Manual Approval Flow)
    # =========================================================================

    async def get_buy_quote(
        self,
        market_id: str,
        outcome: str,
        amount_usd: Decimal,
    ) -> Optional[TradeQuote]:
        """
        Get a quote for buying shares of an outcome.

        This returns the quote for the user to review and approve.
        Does NOT execute the trade.

        Args:
            market_id: Market condition ID
            outcome: Outcome name (e.g., "Yes", "No")
            amount_usd: Amount in USDC to spend

        Returns:
            Trade quote or None
        """
        market = await self.get_market(market_id)
        if not market:
            logger.error(f"Market not found: {market_id}")
            return None

        # Find the outcome token
        token = None
        for t in market.tokens:
            if t.outcome.lower() == outcome.lower():
                token = t
                break

        if not token or not token.token_id:
            logger.error(f"Outcome not found: {outcome} in market {market_id}")
            return None

        # Get orderbook for price impact calculation
        orderbook = await self.client.get_orderbook(token.token_id)
        if not orderbook or not orderbook.asks:
            logger.error(f"No orderbook for {token.token_id}")
            return None

        # Calculate fill using orderbook
        remaining_usd = amount_usd
        total_shares = Decimal("0")
        total_cost = Decimal("0")
        fills = []

        for level in orderbook.asks:
            if remaining_usd <= 0:
                break

            # Price is per share (0-1), cost = price * shares
            max_shares_at_level = level.size
            max_cost_at_level = level.price * max_shares_at_level

            if max_cost_at_level <= remaining_usd:
                # Take entire level
                fills.append((level.price, max_shares_at_level))
                total_shares += max_shares_at_level
                total_cost += max_cost_at_level
                remaining_usd -= max_cost_at_level
            else:
                # Partial fill at this level
                shares_at_level = remaining_usd / level.price
                fills.append((level.price, shares_at_level))
                total_shares += shares_at_level
                total_cost += remaining_usd
                remaining_usd = Decimal("0")

        if total_shares == 0:
            logger.error(f"Could not fill any shares for {amount_usd} USDC")
            return None

        avg_price = total_cost / total_shares

        # Calculate price impact
        mid_price = orderbook.mid_price or token.price
        price_impact = float((avg_price - mid_price) / mid_price * 100) if mid_price else 0

        # Calculate potential payout (if outcome wins, each share = $1)
        max_payout = total_shares  # Each share pays out $1
        potential_profit = max_payout - total_cost
        potential_profit_pct = float(potential_profit / total_cost * 100) if total_cost else None

        return TradeQuote(
            marketId=market_id,
            tokenId=token.token_id,
            side=OrderSide.BUY,
            outcomeName=outcome,
            amountUsd=amount_usd,
            shares=total_shares,
            price=token.price,
            avgPrice=avg_price,
            estimatedFee=Decimal("0"),  # Polymarket has no taker fees currently
            priceImpactPct=price_impact,
            maxPayout=max_payout,
            potentialProfit=potential_profit,
            potentialProfitPct=potential_profit_pct,
            requiresApproval=True,
        )

    async def get_sell_quote(
        self,
        market_id: str,
        outcome: str,
        shares: Optional[Decimal] = None,
        address: Optional[str] = None,
    ) -> Optional[TradeQuote]:
        """
        Get a quote for selling shares of an outcome.

        Args:
            market_id: Market condition ID
            outcome: Outcome name
            shares: Number of shares to sell (None = sell all)
            address: User address (required if shares is None)

        Returns:
            Trade quote or None
        """
        market = await self.get_market(market_id)
        if not market:
            return None

        # Find the outcome token
        token = None
        for t in market.tokens:
            if t.outcome.lower() == outcome.lower():
                token = t
                break

        if not token or not token.token_id:
            return None

        # If no shares specified, get user's position
        if shares is None:
            if not address:
                logger.error("Must provide address or shares")
                return None

            positions = await self.client.get_positions(address)
            position = next(
                (p for p in positions if p.token_id == token.token_id),
                None
            )
            if not position:
                logger.error(f"No position found for {outcome}")
                return None
            shares = position.size

        # Get orderbook for bids
        orderbook = await self.client.get_orderbook(token.token_id)
        if not orderbook or not orderbook.bids:
            return None

        # Calculate fill using orderbook bids
        remaining_shares = shares
        total_proceeds = Decimal("0")
        fills = []

        for level in orderbook.bids:
            if remaining_shares <= 0:
                break

            if level.size <= remaining_shares:
                fills.append((level.price, level.size))
                total_proceeds += level.price * level.size
                remaining_shares -= level.size
            else:
                fills.append((level.price, remaining_shares))
                total_proceeds += level.price * remaining_shares
                remaining_shares = Decimal("0")

        shares_sold = shares - remaining_shares
        if shares_sold == 0:
            return None

        avg_price = total_proceeds / shares_sold

        # Price impact
        mid_price = orderbook.mid_price or token.price
        price_impact = float((mid_price - avg_price) / mid_price * 100) if mid_price else 0

        return TradeQuote(
            marketId=market_id,
            tokenId=token.token_id,
            side=OrderSide.SELL,
            outcomeName=outcome,
            amountUsd=total_proceeds,
            shares=shares_sold,
            price=token.price,
            avgPrice=avg_price,
            estimatedFee=Decimal("0"),
            priceImpactPct=price_impact,
            requiresApproval=True,
        )

    # =========================================================================
    # Analysis
    # =========================================================================

    async def analyze_market(
        self,
        market_id: str,
        llm_provider: Optional[Any] = None,
    ) -> Optional[MarketAnalysis]:
        """
        Get AI-powered analysis of a market.

        Args:
            market_id: Market condition ID
            llm_provider: LLM provider for analysis

        Returns:
            Market analysis or None
        """
        market = await self.get_market(market_id)
        if not market:
            return None

        # Get current prices
        yes_price = Decimal("0.5")
        no_price = Decimal("0.5")
        for token in market.tokens:
            if token.outcome.lower() == "yes":
                yes_price = token.price
            elif token.outcome.lower() == "no":
                no_price = token.price

        # Basic analysis without LLM
        analysis = MarketAnalysis(
            marketId=market_id,
            question=market.question,
            currentYesPrice=yes_price,
            currentNoPrice=no_price,
            summary=f"Market asks: {market.question}",
            keyFactors=[
                f"Current YES probability: {float(yes_price)*100:.1f}%",
                f"Volume: ${float(market.volume):,.0f}",
                f"Liquidity: ${float(market.liquidity):,.0f}",
            ],
            sentiment="neutral",
            confidence=0.5,
            volumeTrend="stable",
            analyzedAt=datetime.utcnow(),
        )

        # If LLM provider available, enhance with AI analysis
        if llm_provider:
            try:
                enhanced = await self._enhance_with_llm(market, analysis, llm_provider)
                if enhanced:
                    return enhanced
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        return analysis

    async def _enhance_with_llm(
        self,
        market: Market,
        base_analysis: MarketAnalysis,
        llm_provider: Any,
    ) -> Optional[MarketAnalysis]:
        """Enhance analysis with LLM insights."""
        # This would use the LLM to provide deeper analysis
        # For now, return base analysis
        return base_analysis


# Singleton instance
_service_instance: Optional[PolymarketTradingService] = None


def get_polymarket_trading_service() -> PolymarketTradingService:
    """Get singleton trading service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PolymarketTradingService()
    return _service_instance
