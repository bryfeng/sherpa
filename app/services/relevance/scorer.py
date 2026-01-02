"""
Relevance Scorer

Multi-factor scoring algorithm to determine how relevant content is
to a user's portfolio holdings.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .models import (
    ContentContext,
    PortfolioContext,
    RelevanceBreakdown,
    RelevanceFactor,
    RelevanceScore,
    TokenHolding,
    DEFAULT_WEIGHTS,
    COMPETITOR_MAP,
    CORRELATION_GROUPS,
)

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Scores content relevance to a user's portfolio.

    Uses multiple factors:
    - Direct holdings (token mentioned that user holds)
    - Same project (related tokens from same protocol)
    - Sector match (DeFi, Gaming, etc.)
    - Subsector match (DEX, Lending, etc.)
    - Competitor dynamics
    - Correlation (related assets)
    - Position weight (larger positions = more relevant)
    - Category overlap
    """

    def __init__(
        self,
        weights: Optional[Dict[RelevanceFactor, float]] = None,
        min_score_threshold: float = 0.1,
    ):
        """
        Initialize the scorer.

        Args:
            weights: Custom weights for each factor (uses defaults if None)
            min_score_threshold: Minimum score to consider relevant
        """
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        self._min_threshold = min_score_threshold

        # Normalize weights to sum to 1
        total_weight = sum(self._weights.values())
        if total_weight != 1.0:
            self._weights = {
                k: v / total_weight for k, v in self._weights.items()
            }

    def score(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceScore:
        """
        Calculate relevance score for content against portfolio.

        Args:
            content: The content to score (news item, alert, etc.)
            portfolio: User's portfolio context

        Returns:
            RelevanceScore with breakdown and explanation
        """
        breakdown: List[RelevanceBreakdown] = []

        # Score each factor
        breakdown.append(self._score_direct_holdings(content, portfolio))
        breakdown.append(self._score_same_project(content, portfolio))
        breakdown.append(self._score_sector_match(content, portfolio))
        breakdown.append(self._score_subsector_match(content, portfolio))
        breakdown.append(self._score_competitor(content, portfolio))
        breakdown.append(self._score_correlation(content, portfolio))
        breakdown.append(self._score_position_weight(content, portfolio))
        breakdown.append(self._score_category_overlap(content, portfolio))

        # Calculate final weighted score
        final_score = sum(b.weighted_score for b in breakdown)

        # Apply importance multiplier (high importance news gets boosted)
        importance_boost = 1.0 + (content.importance - 0.5) * 0.2
        final_score = min(1.0, final_score * importance_boost)

        # Generate explanation
        explanation = self._generate_explanation(breakdown, content, portfolio)

        return RelevanceScore(
            score=round(final_score, 4),
            breakdown=breakdown,
            explanation=explanation,
        )

    def _score_direct_holdings(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on directly held tokens mentioned in content."""
        factor = RelevanceFactor.DIRECT_HOLDING
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for token in content.tokens:
            if portfolio.has_symbol(token):
                matched.append(token)
                # Weight by token's relevance in the content
                token_relevance = content.token_relevance.get(token, 0.5)
                score += token_relevance

        # Normalize score (cap at 1.0)
        score = min(1.0, score)

        details = ""
        if matched:
            details = f"You hold {', '.join(matched)}"
        else:
            details = "No direct holdings mentioned"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_same_project(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on same project/protocol mentions."""
        factor = RelevanceFactor.SAME_PROJECT
        weight = self._weights[factor]

        matched = []
        score = 0.0

        # Check if any content projects match portfolio projects
        content_projects = set(p.lower() for p in content.projects)

        for project in content_projects:
            if project in portfolio.projects:
                matched.append(project)
                score += 0.8

        # Also check if content tokens are related to held tokens
        for content_token in content.tokens:
            for holding in portfolio.holdings:
                if content_token in holding.related_tokens:
                    if content_token not in matched:
                        matched.append(f"{content_token} (related to {holding.symbol})")
                        score += 0.5

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Related projects: {', '.join(matched[:3])}"
        else:
            details = "No project overlap"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_sector_match(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on sector overlap."""
        factor = RelevanceFactor.SECTOR_MATCH
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for sector in content.sectors:
            if portfolio.has_sector(sector):
                matched.append(sector)
                # Weight by portfolio's exposure to the sector
                sector_weight = portfolio.get_sector_weight(sector)
                score += sector_weight

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Sector exposure: {', '.join(matched)}"
        else:
            details = "No sector overlap"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_subsector_match(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on subsector overlap (more specific than sector)."""
        factor = RelevanceFactor.SUBSECTOR_MATCH
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for subsector in content.subsectors:
            if subsector in portfolio.subsectors:
                matched.append(subsector)
                subsector_weight = portfolio.subsectors.get(subsector, 0)
                score += subsector_weight * 1.5  # Boost for specific match

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Subsector match: {', '.join(matched)}"
        else:
            details = "No subsector overlap"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_competitor(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on competitor dynamics."""
        factor = RelevanceFactor.COMPETITOR
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for content_token in content.tokens:
            # Check if any held token has this as a competitor
            for holding in portfolio.holdings:
                competitors = COMPETITOR_MAP.get(holding.symbol, [])
                if content_token in competitors:
                    matched.append(f"{content_token} (competitor to {holding.symbol})")
                    score += 0.6

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Competitors affected: {', '.join(matched[:2])}"
        else:
            details = "No competitor impact"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_correlation(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on correlated asset relationships."""
        factor = RelevanceFactor.CORRELATION
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for content_token in content.tokens:
            # Check if content token is correlated with held tokens
            for held_symbol in portfolio.symbols:
                correlated = CORRELATION_GROUPS.get(held_symbol, [])
                if content_token in correlated:
                    matched.append(f"{content_token} (correlated with {held_symbol})")
                    score += 0.5

            # Check reverse correlation
            correlated_to_content = CORRELATION_GROUPS.get(content_token, [])
            for corr in correlated_to_content:
                if corr in portfolio.symbols:
                    key = f"{content_token} affects {corr}"
                    if key not in matched:
                        matched.append(key)
                        score += 0.5

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Correlated: {matched[0]}" if matched else ""
        else:
            details = "No correlation detected"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_position_weight(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on the size of positions mentioned."""
        factor = RelevanceFactor.POSITION_WEIGHT
        weight = self._weights[factor]

        matched = []
        score = 0.0

        for content_token in content.tokens:
            holding = portfolio.get_holding(content_token)
            if holding:
                # Larger positions = more relevant
                # 10% position = 0.2 score, 50% position = 1.0 score
                position_score = min(1.0, holding.percentage / 50)
                score += position_score
                matched.append(f"{content_token} ({holding.percentage:.1f}%)")

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Position sizes: {', '.join(matched[:2])}"
        else:
            details = "No significant positions"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _score_category_overlap(
        self,
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> RelevanceBreakdown:
        """Score based on category tag overlap."""
        factor = RelevanceFactor.CATEGORY_OVERLAP
        weight = self._weights[factor]

        matched = []
        score = 0.0

        content_categories = set(c.lower() for c in content.categories)
        portfolio_categories = set(portfolio.categories.keys())

        overlap = content_categories & portfolio_categories
        matched = list(overlap)

        if overlap:
            # Weight by portfolio exposure to those categories
            for cat in overlap:
                cat_weight = portfolio.categories.get(cat, 0)
                score += cat_weight

        score = min(1.0, score)

        details = ""
        if matched:
            details = f"Categories: {', '.join(matched[:3])}"
        else:
            details = "No category overlap"

        return RelevanceBreakdown(
            factor=factor,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            matched_items=matched,
        )

    def _generate_explanation(
        self,
        breakdown: List[RelevanceBreakdown],
        content: ContentContext,
        portfolio: PortfolioContext,
    ) -> str:
        """Generate human-readable explanation of relevance."""
        # Get top contributing factors
        top_factors = sorted(
            [b for b in breakdown if b.weighted_score > 0.01],
            key=lambda x: x.weighted_score,
            reverse=True
        )[:3]

        if not top_factors:
            return "This content has minimal relevance to your portfolio."

        parts = []

        for factor in top_factors:
            if factor.matched_items:
                if factor.factor == RelevanceFactor.DIRECT_HOLDING:
                    parts.append(f"mentions {', '.join(factor.matched_items[:2])} which you hold")
                elif factor.factor == RelevanceFactor.SECTOR_MATCH:
                    parts.append(f"affects the {', '.join(factor.matched_items[:2])} sector(s) in your portfolio")
                elif factor.factor == RelevanceFactor.COMPETITOR:
                    parts.append(f"involves competitors to your holdings")
                elif factor.factor == RelevanceFactor.CORRELATION:
                    parts.append(f"may impact correlated assets you hold")
                elif factor.factor == RelevanceFactor.SAME_PROJECT:
                    parts.append(f"relates to projects you're invested in")

        if not parts:
            return "This content has some relevance to your portfolio based on category overlap."

        explanation = "Relevant because it " + ", and ".join(parts[:2]) + "."
        return explanation

    def batch_score(
        self,
        contents: List[ContentContext],
        portfolio: PortfolioContext,
    ) -> List[RelevanceScore]:
        """
        Score multiple content items efficiently.

        Args:
            contents: List of content contexts to score
            portfolio: Portfolio context (reused for all)

        Returns:
            List of relevance scores in same order as contents
        """
        return [self.score(content, portfolio) for content in contents]

    def filter_relevant(
        self,
        contents: List[ContentContext],
        portfolio: PortfolioContext,
        min_score: Optional[float] = None,
    ) -> List[tuple[ContentContext, RelevanceScore]]:
        """
        Filter and score contents, returning only relevant ones.

        Args:
            contents: Contents to filter
            portfolio: Portfolio context
            min_score: Minimum score threshold (uses default if None)

        Returns:
            List of (content, score) tuples, sorted by relevance
        """
        threshold = min_score or self._min_threshold

        results = []
        for content in contents:
            score = self.score(content, portfolio)
            if score.score >= threshold:
                results.append((content, score))

        # Sort by relevance (highest first)
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results
