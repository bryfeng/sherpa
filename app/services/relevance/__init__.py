"""
Relevance Scoring Service

Scores content (news, alerts) for relevance to a user's portfolio.
Enables personalized information delivery based on holdings.
"""

from .models import (
    RelevanceScore,
    RelevanceBreakdown,
    RelevanceFactor,
    PortfolioContext,
    ContentContext,
    TokenHolding,
)
from .scorer import RelevanceScorer
from .service import RelevanceService

__all__ = [
    # Models
    "RelevanceScore",
    "RelevanceBreakdown",
    "RelevanceFactor",
    "PortfolioContext",
    "ContentContext",
    "TokenHolding",
    # Scorer
    "RelevanceScorer",
    # Service
    "RelevanceService",
]
