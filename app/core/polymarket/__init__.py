"""
Polymarket Trading Module

Core trading logic for Polymarket prediction markets.
"""

from .models import (
    PolymarketPortfolio,
    PortfolioPosition,
    MarketAnalysis,
    TradeQuote,
    TradeResult,
)
from .trading import PolymarketTradingService, get_polymarket_trading_service
from .copy_trading import (
    PolymarketCopyManager,
    get_polymarket_copy_manager,
    PolymarketCopyConfig,
    PolymarketCopyRelationship,
    PolymarketCopyExecution,
    PMSizingMode,
    PMCopyExecutionStatus,
)

__all__ = [
    # Models
    "PolymarketPortfolio",
    "PortfolioPosition",
    "MarketAnalysis",
    "TradeQuote",
    "TradeResult",
    # Service
    "PolymarketTradingService",
    "get_polymarket_trading_service",
    # Copy Trading
    "PolymarketCopyManager",
    "get_polymarket_copy_manager",
    "PolymarketCopyConfig",
    "PolymarketCopyRelationship",
    "PolymarketCopyExecution",
    "PMSizingMode",
    "PMCopyExecutionStatus",
]
